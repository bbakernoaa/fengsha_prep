"""
This module provides functions to retrieve soil data from SoilGrids.
"""

from soilgrids import SoilGrids

def get_soilgrids_data(service_id, coverage_id, west, south, east, north, crs, output):
    """
    Retrieves soil data from SoilGrids for a given area.

    Args:
        service_id (str): The service ID for the soil property.
        coverage_id (str): The coverage ID for the soil property.
        west (float): The western boundary of the area.
        south (float): The southern boundary of the area.
        east (float): The eastern boundary of the area.
        north (float): The northern boundary of the area.
        crs (str): The coordinate reference system.
        output (str): The path to the output file.

    Returns:
        xarray.DataArray: The soil data as an xarray DataArray.
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
        output=output
    )
    return data
