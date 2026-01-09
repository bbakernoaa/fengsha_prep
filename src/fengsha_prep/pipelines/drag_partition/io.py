import earthaccess
import xarray as xr


def get_modis_data(product: str, start_date: str, end_date: str) -> xr.Dataset:
    """Retrieves MODIS CMG data from a NASA DAAC using earthaccess.

    Parameters
    ----------
    product : str
        The short name of the MODIS product (e.g., "MCD43C3", "MCD15A2H").
    start_date : str
        The start date for the data search in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data search in 'YYYY-MM-DD' format.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the downloaded and consolidated MODIS data.
    """
    results = earthaccess.search_data(
        short_name=product, cloud_hosted=True, temporal=(start_date, end_date)
    )
    # Open the multi-file dataset using xarray
    ds = earthaccess.open_xr(results)
    return ds
