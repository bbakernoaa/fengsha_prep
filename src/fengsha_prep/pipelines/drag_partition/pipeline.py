import logging
from typing import Callable, Union

import xarray as xr

from .algorithm import calculate_drag_partition
from .io import get_modis_data

# Set up a logger for the module
logger = logging.getLogger(__name__)


def run_drag_partition_pipeline(
    start_date: str,
    end_date: str,
    u10_wind: Union[float, xr.DataArray],
    data_fetcher: Callable[[str, str, str], xr.Dataset] = get_modis_data,
) -> xr.DataArray:
    """Automated pipeline to fetch data and calculate the Drag Partition.

    This function implements a hybrid model to estimate the surface friction
    velocity (us*) by partitioning drag between bare soil, green vegetation,
    and non-photosynthetic (brown) vegetation.

    Parameters
    ----------
    start_date : str
        The start date for the analysis in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the analysis in 'YYYY-MM-DD' format.
    u10_wind : Union[float, xr.DataArray]
        The 10-meter wind speed. Can be a constant (float) or a DataArray
        with dimensions matching the MODIS grid.
    data_fetcher : Callable[[str, str, str], xr.Dataset], optional
        A function that retrieves MODIS data. Defaults to `get_modis_data`.
        This parameter allows for dependency injection, primarily for testing.

    Returns
    -------
    xr.DataArray
        A DataArray representing the calculated surface friction velocity (us*).

    References
    ----------
    - Chappell & Webb (2016): DOI 10.1016/j.aeolia.2015.11.001
    - Leung et al. (2023): DOI 10.5194/acp-23-11235-2023
    - Hennen et al. (2023): DOI 10.1016/j.aeolia.2022.100852
    - Guerschman et al. (2009): DOI 10.1016/j.rse.2009.01.006
    """
    logger.info("Fetching MODIS Albedo (MCD43C3)...")
    ds_alb = data_fetcher("MCD43C3", start_date, end_date)

    logger.info("Fetching MODIS LAI (MCD15A2H)...")
    ds_lai = data_fetcher("MCD15A2H", start_date, end_date)

    return calculate_drag_partition(ds_alb, ds_lai, u10_wind)
