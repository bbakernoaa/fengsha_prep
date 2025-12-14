"""
This module provides functions for regridding data.
"""
from typing import Union
import xarray as xr
import xesmf as xe
import numpy as np

def regrid_modis_to_rectilinear(
    ds: xr.Dataset,
    var_name: str,
    lon_min: float,
    lon_max: float,
    d_lon: float,
    lat_min: float,
    lat_max: float,
    d_lat: float,
    method: str = 'bilinear'
) -> xr.DataArray:
    """
    Regrids a dataset from a MODIS sinusoidal grid to a rectilinear grid.

    Args:
        ds (xr.Dataset): Input dataset with data on a sinusoidal grid.
                         It must contain 'lat' and 'lon' as 2D variables.
        var_name (str): The name of the variable to regrid.
        lon_min (float): Minimum longitude of the output grid.
        lon_max (float): Maximum longitude of the output grid.
        d_lon (float): Longitude resolution of the output grid.
        lat_min (float): Minimum latitude of the output grid.
        lat_max (float): Maximum latitude of the output grid.
        d_lat (float): Latitude resolution of the output grid.
        method (str): Regridding method. Defaults to 'bilinear'.
                      Options include 'bilinear', 'conservative', 'nearest_s2d', etc.

    Returns:
        xr.DataArray: The regridded data as an xarray DataArray.
    """
    # Create the output grid directly
    lon_out = np.arange(lon_min, lon_max + d_lon, d_lon)
    lat_out = np.arange(lat_min, lat_max + d_lat, d_lat)

    ds_out = xr.Dataset({
        'lat': (('lat',), lat_out),
        'lon': (('lon',), lon_out),
    })

    # Create the regridder
    regridder = xe.Regridder(ds, ds_out, method)

    # Perform the regridding
    ds_regridded = regridder(ds[var_name])

    return ds_regridded
