"""
I/O operations for the uthresh pipeline.
Handles data retrieval from Earthdata, S3, and web services.
"""

import asyncio
import io
from datetime import datetime
from typing import Any, TypeAlias

import aiohttp
import rasterio
import s3fs
import xarray as xr

# --- TYPE HINTS ---
# A mapping from a soil variable name (e.g., 'clay') to its fetched value.
SoilResult: TypeAlias = dict[str, float]


# --- STAGE 1: DATA RETRIEVAL (ASYNC ENGINE) ---


class AsyncDustDataEngine:
    """
    Asynchronously orchestrates the retrieval of meteorological and soil data.

    This engine is designed for non-blocking I/O, making it suitable for
    high-performance data gathering. It requires an `aiohttp.ClientSession`
    and an `s3fs.S3FileSystem` to be passed during initialization,
    promoting testability and explicit dependency management.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        s3_filesystem: s3fs.S3FileSystem,
    ) -> None:
        """
        Initializes the asynchronous data engine.

        Parameters
        ----------
        http_session : aiohttp.ClientSession
            An active `aiohttp` session for making HTTP requests (e.g., to SoilGrids).
        s3_filesystem : s3fs.S3FileSystem
            An `s3fs` filesystem object for accessing S3 data (e.g., UFS).
        """
        self.http = http_session
        self.fs_s3 = s3_filesystem
        self.wcs_url = "https://maps.isric.org/mapserv?map=/srv/node/node_modules/soilgrids/maps/soilgrids.map"

    async def fetch_met_ufs(self, dt: datetime, lat: float, lon: float) -> xr.Dataset:
        """
        Asynchronously fetch UFS Replay 3-hourly meteorological data.

        This method uses `asyncio.to_thread` to run the blocking `xarray.open_dataset`
        call in a separate thread, preventing it from blocking the asyncio event loop.

        Parameters
        ----------
        dt : datetime.datetime
            The timestamp for which to fetch the data.
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        xarray.Dataset
            A dataset containing the meteorological variables for the nearest
            grid point.
        """
        bucket = "noaa-ufs-gefsv13replay-pds"
        path = f"s3://{bucket}/{dt.strftime('%Y%m%d/%H')}/atmos/gefs.t{dt.strftime('%H')}z.pgrb2.0p25.f000"

        def _blocking_open_and_select(file_obj: Any) -> xr.Dataset:
            """Helper function to run blocking I/O in a thread."""
            # Use xr.open_dataset within a with-statement to ensure file is closed
            ds = xr.open_dataset(
                file_obj,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            )
            return ds.sel(latitude=lat, longitude=lon, method="nearest")

        # s3fs.S3FileSystem.open is a coroutine if asynchronous=True
        async with self.fs_s3.open(path) as f:
            # xarray's open_dataset is blocking, so we run it in a thread
            return await asyncio.to_thread(_blocking_open_and_select, f)

    async def _fetch_soil_variable(
        self, variable: str, lat: float, lon: float
    ) -> tuple[str, float]:
        """
        Asynchronously fetches a single soil variable from the SoilGrids WCS.

        This is a helper coroutine for `fetch_soilgrids_concurrently`.

        Parameters
        ----------
        variable : str
            The name of the soil variable to fetch (e.g., 'clay', 'soc').
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        Tuple[str, float]
            A tuple containing the variable name and its corresponding value.
        """
        bbox = (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)
        bbox_str = ",".join(map(str, bbox))
        layer = f"{variable}_0-5cm_mean"
        params = {
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": layer,
            "CRS": "EPSG:4326",
            "BBOX": bbox_str,
            "WIDTH": 1,
            "HEIGHT": 1,
            "FORMAT": "image/tiff",
        }
        async with self.http.get(self.wcs_url, params=params) as resp:
            resp.raise_for_status()
            tiff_bytes = await resp.read()

            def _blocking_read_raster(data: bytes) -> float:
                """Helper function to run blocking rasterio in a thread."""
                with rasterio.open(io.BytesIO(data)) as src:
                    return src.read(1)[0, 0]

            # rasterio is blocking, run in thread
            value = await asyncio.to_thread(_blocking_read_raster, tiff_bytes)
            return variable, value

    async def fetch_soilgrids_concurrently(self, lat: float, lon: float) -> SoilResult:
        """
        Fetches all soil properties concurrently using `asyncio.gather`.

        This method significantly speeds up data retrieval by running the
        network requests for each soil variable in parallel.

        Parameters
        ----------
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        SoilResult
            A dictionary containing all fetched soil properties.
        """
        variables = ["clay", "sand", "soc", "bdod"]
        tasks = [self._fetch_soil_variable(var, lat, lon) for var in variables]
        results = await asyncio.gather(*tasks)
        return dict(results)


# --- SYNCHRONOUS WRAPPER (for backward compatibility or simpler scripts) ---


class DustDataEngine:
    """
    Synchronous wrapper for the AsyncDustDataEngine.

    Provides a blocking interface that is easier to use in traditional,
    sequential scripts. This class manages its own asyncio event loop
    and `aiohttp` session internally.

    It is recommended to use this class as a context manager (`with ...:`)
    to ensure resources are properly cleaned up.
    """

    _s3_fs: s3fs.S3FileSystem
    _http_session: aiohttp.ClientSession | None = None
    _async_engine: AsyncDustDataEngine | None = None

    def __init__(self, s3_filesystem: s3fs.S3FileSystem | None = None) -> None:
        """
        Initializes the synchronous wrapper.

        Parameters
        ----------
        s3_filesystem : Optional[s3fs.S3FileSystem]
             An `s3fs` filesystem object, configured for async (`asynchronous=True`).
             If not provided, a default one will be created.
        """
        if s3_filesystem:
            self._s3_fs = s3_filesystem
        else:
            self._s3_fs = s3fs.S3FileSystem(anon=True, asynchronous=True)

    def __enter__(self) -> "DustDataEngine":
        self._http_session = aiohttp.ClientSession()
        self._async_engine = AsyncDustDataEngine(self._http_session, self._s3_fs)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._http_session and not self._http_session.closed:
            # Ensure the loop is running to close the session
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self._http_session.close())
                else:
                    asyncio.run(self._http_session.close())  # pragma: no cover
            except RuntimeError:
                asyncio.run(self._http_session.close())  # pragma: no cover

    @property
    def engine(self) -> AsyncDustDataEngine:
        """Ensures the async engine is initialized and raises a helpful error."""
        if (
            self._async_engine is None
            or self._http_session is None
            or self._http_session.closed
        ):
            raise RuntimeError(
                "DustDataEngine must be used as a context manager "
                "(e.g., `with DustDataEngine() as engine:`)"
            )
        return self._async_engine

    def fetch_met_ufs(self, dt: datetime, lat: float, lon: float) -> xr.Dataset:
        """Synchronously fetch UFS Replay 3-hourly meteorological data."""
        return asyncio.run(self.engine.fetch_met_ufs(dt, lat, lon))

    def fetch_soilgrids(self, lat: float, lon: float) -> dict[str, float]:
        """Synchronously fetch all soil properties."""
        return asyncio.run(self.engine.fetch_soilgrids_concurrently(lat, lon))
