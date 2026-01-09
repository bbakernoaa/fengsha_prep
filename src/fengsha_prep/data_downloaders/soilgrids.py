"""
This module provides functions to retrieve soil data from SoilGrids.
"""
import asyncio
from pathlib import Path
from typing import Union

import xarray as xr
from soilgrids import SoilGrids


async def get_soilgrids_data_async(
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
    Asynchronously retrieves soil data from SoilGrids for a given area and
    saves it as a compressed NetCDF file.
    This function uses asyncio.to_thread to run the blocking I/O operations
    in a separate thread, making it non-blocking.
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

    def _blocking_io():
        """Helper function to encapsulate blocking I/O."""
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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_netcdf(output_path, encoding=encoding)
        return data

    return await asyncio.to_thread(_blocking_io)


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
    Retrieves soil data from SoilGrids for a given area and saves it as a
    compressed NetCDF file.
    This is a synchronous wrapper for the get_soilgrids_data_async function.
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
    Raises:
        RuntimeError: If called from within a running asyncio event loop.
    """
    return asyncio.run(
        get_soilgrids_data_async(
            service_id=service_id,
            coverage_id=coverage_id,
            west=west,
            south=south,
            east=east,
            north=north,
            crs=crs,
            output_path=output_path,
        )
    )
