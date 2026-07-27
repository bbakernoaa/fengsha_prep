import logging
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Disable HDF5 file locking before any HDF5/netCDF4 library initialises.
# Required on NFS / Lustre / GPFS filesystems that don't support POSIX locks.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import xarray as xr
try:
    from dask.distributed import Client, get_client
except ImportError:
    Client = None
    get_client = None

from .algorithm import calculate_drag_partition
from .io import get_cmg_data, load_prigent_drag_partition

# Set up a logger for the module
logger = logging.getLogger(__name__)

Sensor = Literal["MODIS", "VNP", "VJ1", "NESDIS"]


def run_drag_partition_pipeline(
    start_date: str,
    end_date: str,
    u10_wind: float | xr.DataArray = None,
    sensor: Sensor = "MODIS",
    data_fetcher: Callable[[str, str, str, Sensor, bool], xr.Dataset | None] = get_cmg_data,
    use_lai: bool = True,
    output_dir: str | Path | None = None,
    ndvi_threshold: float = 0.15,
    use_gvf_adjustment: bool = False,
    use_ndvi_adjustment: bool = False,
    vegetation_gamma: float = 1.0,
    bare_threshold: float = 0.80,
    prigent_path: str | Path | None = "PRIGENT_ET_AL_DRAGPARTITION.nc",
    cleanup_downloads: bool = True,
    cache_dir: str | Path | None = None,
    output_format: str = "netcdf",
) -> xr.Dataset | list[Path]:
    """Automated pipeline to fetch data and calculate the effective drag (feff).

    This function implements a hybrid model to estimate the effective drag
    coefficient by partitioning drag between bare soil, green vegetation,
    and non-photosynthetic (brown) vegetation.

    Parameters
    ----------
    start_date : str
        The start date for the analysis in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the analysis in 'YYYY-MM-DD' format.
    u10_wind : float | xr.DataArray, optional
        Wind speed at 10m height (m/s). Defaults to None.
    sensor : Sensor, optional
        The sensor to use for data retrieval. Can be 'MODIS', 'VNP', 'VJ1', or 'NESDIS'.
        Defaults to 'MODIS'.
    data_fetcher : Callable[[str, str, str, Sensor, bool], xr.Dataset | None], optional
        A function that retrieves CMG data. Defaults to `get_cmg_data`.
        This parameter allows for dependency injection, primarily for testing.
    use_lai : bool, optional
        Whether to include the LAI (green vegetation) component.
        Defaults to True.
    output_dir : str | Path, optional
        If provided, the pipeline will process each day in the range
        individually and save the result as a NetCDF file in this directory.
        Returns a list of paths to the saved files.

    Returns
    -------
    xr.Dataset | list[Path]
        A Dataset containing the results for the whole range, or a list
        of file paths if `output_dir` was specified.
    """

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")

        # Preliminary check for Earthdata login if using NASA data
        if sensor in ["MODIS", "VNP", "VJ1"]:
            import earthaccess
            if not earthaccess.login():
                logger.error("NASA Earthdata login required. Please run `earthaccess.login()` before starting the pipeline.")
                raise RuntimeError("Not logged into NASA Earthdata.")

        # Collect tasks for missing days
        tasks = []
        current = dt_start
        ext = "nc" if output_format == "netcdf" else "zarr"
        while current <= dt_end:
            day_str = current.strftime("%Y-%m-%d")
            out_path = output_dir / f"drag_partition_{sensor}_{day_str}.{ext}"
            if not out_path.exists():
                tasks.append((day_str, out_path))
            else:
                logger.info(f"Output for {day_str} already exists at {out_path}. Skipping.")
            current += timedelta(days=1)

        if tasks:
            logger.info(f"Processing {len(tasks)} days sequentially...")
            for day_str, out_path in tasks:
                logger.info(f"--- Processing Day: {day_str} ---")

                # Use a temporary subdirectory to isolate downloads for this day
                day_cache = Path("data") / f"tmp_{day_str}" if cleanup_downloads else None

                ds_day = run_drag_partition_pipeline(
                    day_str, day_str, u10_wind, sensor, data_fetcher, use_lai,
                    output_dir=None,
                    ndvi_threshold=ndvi_threshold,
                    use_gvf_adjustment=use_gvf_adjustment,
                    use_ndvi_adjustment=use_ndvi_adjustment,
                    vegetation_gamma=vegetation_gamma,
                    bare_threshold=bare_threshold,
                    prigent_path=prigent_path,
                    cache_dir=day_cache,
                    output_format=output_format,
                )

                logger.info(f"Saving {day_str} result to {out_path}...")
                # Load into memory first to release HDF5 source file handles before writing.
                ds_day = ds_day.load()

                if output_format == "zarr":
                    logger.info(f"Writing Zarr dataset to {out_path} with proper chunking...")
                    # Re-chunk for optimal performance and chunk sizes
                    chunks = {}
                    if "time" in ds_day.dims:
                        chunks["time"] = 1
                    if "lat" in ds_day.dims:
                        chunks["lat"] = 1800
                    if "lon" in ds_day.dims:
                        chunks["lon"] = 3600
                    
                    if chunks:
                        ds_day = ds_day.chunk(chunks)
                    
                    ds_day.to_zarr(out_path, mode="w")
                else:
                    encoding = {var: {"zlib": True, "complevel": 5} for var in ds_day.data_vars}
                    # Write to a temp file in the same directory then atomically rename
                    # so a failed write never leaves a corrupt file that blocks future runs.
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        suffix=".nc.tmp", dir=out_path.parent
                    )
                    os.close(tmp_fd)
                    try:
                        ds_day.to_netcdf(tmp_path, encoding=encoding, engine="h5netcdf")
                        os.replace(tmp_path, out_path)
                    except Exception:
                        Path(tmp_path).unlink(missing_ok=True)
                        raise

                if cleanup_downloads and day_cache and day_cache.exists():
                    import shutil
                    logger.info(f"Cleaning up temporary downloads for {day_str}...")
                    shutil.rmtree(day_cache)

        if cleanup_downloads:
            import shutil
            for d in ["nasa", "nesdis"]:
                dir_path = Path("data") / d
                if dir_path.exists():
                    logger.info(f"Cleaning up downloads in {dir_path}...")
                    for file in dir_path.glob("*"):
                        if file.is_file() and file.suffix in [".h5", ".hdf", ".nc", ".hdf4"]:
                            file.unlink()

        # Return list of all paths in order
        saved_paths = []
        current = dt_start
        while current <= dt_end:
            day_str = current.strftime("%Y-%m-%d")
            saved_paths.append(output_dir / f"drag_partition_{sensor}_{day_str}.{ext}")
            current += timedelta(days=1)
        return saved_paths

    logger.info(f"Starting {sensor} drag partition pipeline from {start_date} to {end_date} (use_lai={use_lai})...")

    # Check for Dask Client to ensure parallel execution
    if get_client:
        try:
            client = get_client()
            logger.info(f"Using existing Dask client: {client.dashboard_link}")
        except ValueError:
            logger.warning("No Dask client found. Calculation may be slow. Consider initializing a Client().")

    logger.info(f"Fetching {sensor} BRDF Parameters data (C1)...")
    ds_brdf = data_fetcher("brdf", start_date, end_date, sensor, False, cache_dir=cache_dir)

    ds_albedo = None
    ds_nbar = None
    if sensor not in ["VNP", "VJ1"]:
        logger.info(f"Fetching {sensor} Albedo data (C3)...")
        ds_albedo = data_fetcher("albedo", start_date, end_date, sensor, True, cache_dir=cache_dir)

        logger.info(f"Fetching {sensor} NBAR data (C4)...")
        ds_nbar = data_fetcher("nbar", start_date, end_date, sensor, True, cache_dir=cache_dir)
    else:
        logger.info(f"Skipping Albedo (C3) and NBAR (C4) fetch for {sensor} (using C1 fallbacks).")

    ds_lai = None
    if use_lai:
        logger.info(f"Fetching {sensor} LAI data...")
        ds_lai = data_fetcher("lai", start_date, end_date, sensor, True, cache_dir=cache_dir)
    else:
        logger.info("LAI data fetching skipped (use_lai=False).")

    ds_gvf = None
    ds_ndvi = None
    if sensor in ["VNP", "VJ1"]:
        logger.info(f"Fetching {sensor} NDVI data (13C1) instead of GVF...")
        ds_ndvi = data_fetcher("ndvi", start_date, end_date, sensor, True, cache_dir=cache_dir)
    else:
        logger.info(f"Fetching {sensor} GVF data...")
        ds_gvf = data_fetcher("gvf", start_date, end_date, sensor, True, cache_dir=cache_dir)

    ds_prigent = None
    if prigent_path:
        try:
            ds_prigent = load_prigent_drag_partition(prigent_path)
        except FileNotFoundError:
            logger.warning(f"Prigent drag partition file not found at {prigent_path}. Skipping.")

    ds_results = calculate_drag_partition(
        ds_brdf,
        ds_lai,
        ds_albedo=ds_albedo,
        ds_nbar=ds_nbar,
        ds_gvf=ds_gvf,
        ds_ndvi=ds_ndvi,
        ds_prigent=ds_prigent,
        use_lai=use_lai,
        ndvi_threshold=ndvi_threshold,
        use_gvf_adjustment=use_gvf_adjustment,
        use_ndvi_adjustment=use_ndvi_adjustment,
        vegetation_gamma=vegetation_gamma,
        bare_threshold=bare_threshold,
    )

    # Ensure the output has a time dimension/coordinate for the processed day
    if "time" not in ds_results.coords:
        logger.debug(f"Adding time coordinate from input start_date: {start_date}")
        ds_results = ds_results.expand_dims(time=[datetime.strptime(start_date, "%Y-%m-%d")])

    return ds_results
