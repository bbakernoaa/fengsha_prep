import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import xarray as xr

from .algorithm import calculate_drag_partition
from .io import get_cmg_data

# Set up a logger for the module
logger = logging.getLogger(__name__)

Sensor = Literal["MODIS", "VIIRS"]


def run_drag_partition_pipeline(
    start_date: str,
    end_date: str,
    sensor: Sensor = "MODIS",
    data_fetcher: Callable[[str, str, str, Sensor], xr.Dataset] = get_cmg_data,
) -> xr.DataArray:
    """Automated pipeline to fetch data and calculate the effective drag (feff).

    This function implements a hybrid model to estimate the effective drag
    coefficient by partitioning drag between bare soil, green vegetation,
    and non-photosynthetic (brown) vegetation. It fetches Albedo and LAI
    data concurrently to improve performance.

    Parameters
    ----------
    start_date : str
        The start date for the analysis in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the analysis in 'YYYY-MM-DD' format.
    sensor : Sensor, optional
        The sensor to use for data retrieval. Can be 'MODIS' or 'VIIRS'.
        Defaults to 'MODIS'.
    data_fetcher : Callable[[str, str, str, Sensor], xr.Dataset], optional
        A function that retrieves CMG data. Defaults to `get_cmg_data`.
        This parameter allows for dependency injection, primarily for testing.

    Returns
    -------
    xr.DataArray
        A DataArray representing the calculated effective drag (feff).

    References
    ----------
    - Chappell & Webb (2016): DOI 10.1016/j.aeolia.2015.11.001
    - Leung et al. (2023): DOI 10.5194/acp-23-11235-2023
    - Hennen et al. (2023): DOI 10.1016/j.aeolia.2022.100852
    - Guerschman et al. (2009): DOI 10.1016/j.rse.2009.01.006
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        logger.info(f"Submitting concurrent fetch for {sensor} Albedo and LAI data...")
        future_alb = executor.submit(data_fetcher, "albedo", start_date, end_date, sensor)
        future_lai = executor.submit(data_fetcher, "lai", start_date, end_date, sensor)

        logger.info("Waiting for data fetching to complete...")
        ds_alb = future_alb.result()
        ds_lai = future_lai.result()
        logger.info("Data fetching complete.")

    return calculate_drag_partition(ds_alb, ds_lai)
