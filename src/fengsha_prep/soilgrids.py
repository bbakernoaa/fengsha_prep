"""
This module provides functions to retrieve soil data from SoilGrids.
"""
from pathlib import Path
from typing import Union

import xarray as xr
from soilgrids import SoilGrids


def get_soilgrids_data(
    service_id: str,
    coverage_id: str,
    west: float,
    south: float,
    east: float,
    north: float,
    crs: str,
    output_path: Union[str, Path],
) -> xr.DataArray:
    """
    Retrieves soil data from SoilGrids for a given area and saves it as a compressed NetCDF file.

    Args:
        service_id: The service ID for the soil property.
        coverage_id: The coverage ID for the soil property.
        west: The western boundary of the area.
        south: The southern boundary of the area.
        east: The eastern boundary of the area.
        north: The northern boundary of the area.
        crs: The coordinate reference system.
        output_path: The path to the output NetCDF file.

    Returns:
        The soil data as an xarray DataArray.
    """
    soil_grids = SoilGrids()
    data = soil_grids.get_coverage_data(
        service_id=service_id,
        coverage_id=coverage_id,
        west=west,
        south=south,
        east=east,
        north=north,
        crs=crs,
    )

    # Define compression encoding for NetCDF output
    encoding = {data.name: {"zlib": True, "complevel": 5}}

    # Save to compressed NetCDF
    data.to_netcdf(output_path, encoding=encoding)

    return data
