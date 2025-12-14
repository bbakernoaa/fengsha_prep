"""
This module provides functions for regridding data.
"""
import xarray as xr
import xesmf as xe
import numpy as np

def regrid_modis_to_rectilinear(ds, var_name, lon_min, lon_max, d_lon, lat_min, lat_max, d_lat):
    """
    Regrids a dataset from a MODIS sinusoidal grid to a rectilinear Gaussian grid.

    Args:
        ds (xarray.Dataset): The input dataset with the data on a sinusoidal grid.
        var_name (str): The name of the variable to regrid.
        lon_min (float): The minimum longitude of the output grid.
        lon_max (float): The maximum longitude of the output grid.
        d_lon (float): The longitude resolution of the output grid.
        lat_min (float): The minimum latitude of the output grid.
        lat_max (float): The maximum latitude of the output grid.
        d_lat (float): The latitude resolution of the output grid.

    Returns:
        xarray.Dataset: The regridded dataset.
    """
    # Create the output grid
    ds_out = xe.util.grid_global(d_lon, d_lat, lon1=lon_min, lat1=lat_min)
    ds_out = ds_out.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    # Create the regridder
    regridder = xe.Regridder(ds, ds_out, 'bilinear')

    # Perform the regridding
    ds_regridded = regridder(ds[var_name])

    return ds_regridded
