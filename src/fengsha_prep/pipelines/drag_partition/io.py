from typing import Literal

import earthaccess
import xarray as xr

Sensor = Literal["MODIS", "VIIRS"]

PRODUCT_MAP = {
    "MODIS": {"albedo": "MCD43C3", "lai": "MCD15A2H"},
    "VIIRS": {"albedo": "VJ143C3", "lai": "VNP15A2H"},
}


def get_cmg_data(
    product_type: str, start_date: str, end_date: str, sensor: Sensor = "MODIS"
) -> xr.Dataset:
    """Retrieves MODIS or VIIRS CMG data from a NASA DAAC using earthaccess.

    Parameters
    ----------
    product_type : str
        The type of product to retrieve (e.g., "albedo", "lai").
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.
    sensor : Sensor, optional
        The sensor to use ('MODIS' or 'VIIRS'). Defaults to 'MODIS'.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the downloaded and consolidated data.
    """
    short_name = PRODUCT_MAP[sensor][product_type]
    results = earthaccess.search_data(
        short_name=short_name, cloud_hosted=True, temporal=(start_date, end_date)
    )
    # Open the multi-file dataset using xarray
    ds = earthaccess.open_xr(results)
    return ds
