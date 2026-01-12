"""
I/O operations for the uthresh pipeline.
Handles data retrieval from Earthdata, S3, and web services.
"""
import io
from datetime import datetime

import earthaccess
import rasterio
import s3fs
import xarray as xr
from owslib.wcs import WebCoverageService

# --- STAGE 1: DATA RETRIEVAL (EARTHDATA & S3) ---


class DustDataEngine:
    """Orchestrates the retrieval of meteorological and soil data."""

    def __init__(self) -> None:
        """Initializes authentication handlers for Earthdata and S3."""
        self.auth = earthaccess.login(strategy="interactive")
        self.fs_s3 = s3fs.S3FileSystem(anon=True)
        self.wcs_url = "https://maps.isric.org/mapserv?map=/srv/node/node_modules/soilgrids/maps/soilgrids.map"

    def fetch_met_ufs(self, dt: datetime, lat: float, lon: float) -> xr.Dataset:
        """
        Fetch UFS Replay 3-hourly meteorological data for a specific time and location.

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
        with self.fs_s3.open(path) as f:
            ds = xr.open_dataset(
                f,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            )
            return ds.sel(latitude=lat, longitude=lon, method="nearest")

    def fetch_soilgrids(self, lat: float, lon: float) -> dict[str, float]:
        """
        Fetch physical/chemical soil properties from ISRIC SoilGrids via WCS.

        Parameters
        ----------
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        Dict[str, float]
            A dictionary containing soil properties ('clay', 'sand', 'soc', 'bdod').
        """
        wcs = WebCoverageService(self.wcs_url, version="2.0.1")
        bbox = (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)
        res: dict[str, float] = {}
        for var in ["clay", "sand", "soc", "bdod"]:
            layer = f"{var}_0-5cm_mean"
            resp = wcs.getCoverage(
                identifier=layer,
                crs="EPSG:4326",
                bbox=bbox,
                width=1,
                height=1,
                format="image/tiff",
            )
            with rasterio.open(io.BytesIO(resp.read())) as src:
                res[var] = src.read(1)[0, 0]
        return res
