import logging
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
try:
    from dask.distributed import Client, get_client
except ImportError:
    Client = None
    get_client = None

from .algorithm import calculate_drag_partition
from .io import get_cmg_data

# Set up a logger for the module
logger = logging.getLogger(__name__)

Sensor = Literal["MODIS", "VIIRS", "NESDIS"]


def run_drag_partition_pipeline(
    start_date: str,
    end_date: str,
    u10_wind: float | xr.DataArray = None,
    sensor: Sensor = "MODIS",
    data_fetcher: Callable[[str, str, str, Sensor, bool], xr.Dataset | None] = get_cmg_data,
    use_lai: bool = True,
    output_dir: str | Path | None = None,
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
        The sensor to use for data retrieval. Can be 'MODIS', 'VIIRS', or 'NESDIS'.
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

        saved_paths = []
        current = dt_start
        while current <= dt_end:
            day_str = current.strftime("%Y-%m-%d")
            out_path = output_dir / f"drag_partition_{sensor}_{day_str}.nc"

            if out_path.exists():
                logger.info(f"Output for {day_str} already exists at {out_path}. Skipping.")
            else:
                logger.info(f"--- Processing Day: {day_str} ---")
                ds_day = run_drag_partition_pipeline(
                    day_str, day_str, u10_wind, sensor, data_fetcher, use_lai, output_dir=None
                )
                logger.info(f"Saving {day_str} result to {out_path}...")
                ds_day.to_netcdf(out_path)

            saved_paths.append(out_path)
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
    ds_brdf = data_fetcher("brdf", start_date, end_date, sensor, False)

    logger.info(f"Fetching {sensor} Albedo data (C3)...")
    ds_albedo = data_fetcher("albedo", start_date, end_date, sensor, True)

    logger.info(f"Fetching {sensor} NBAR data (C4)...")
    ds_nbar = data_fetcher("nbar", start_date, end_date, sensor, True)

    ds_lai = None
    if use_lai:
        logger.info(f"Fetching {sensor} LAI data...")
        ds_lai = data_fetcher("lai", start_date, end_date, sensor, True)
    else:
        logger.info("LAI data fetching skipped (use_lai=False).")

    ds_gvf = None
    if use_lai and sensor == "NESDIS":
        logger.info(f"Fetching {sensor} GVF data...")
        ds_gvf = data_fetcher("gvf", start_date, end_date, sensor, True)

    ds_results = calculate_drag_partition(
        ds_brdf, ds_lai, ds_albedo=ds_albedo, ds_nbar=ds_nbar, ds_gvf=ds_gvf, use_lai=use_lai
    )

    return ds_results
