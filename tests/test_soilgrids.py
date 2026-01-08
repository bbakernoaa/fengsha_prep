"""
Tests for the soilgrids module.
"""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr

from fengsha_prep.data_downloaders import soilgrids


class TestSoilgrids(unittest.TestCase):
    """
    Tests for the soilgrids module.
    """

    @patch("pathlib.Path.mkdir")
    @patch("xarray.DataArray.to_netcdf")
    @patch("fengsha_prep.data_downloaders.soilgrids.SoilGrids")
    def test_get_soilgrids_data(self, mock_soilgrids, mock_to_netcdf, mock_mkdir):
        """
        Tests the get_soilgrids_data function.
        """
        # Create a mock SoilGrids object
        mock_soilgrids_instance = mock_soilgrids.return_value

        # Create a mock xarray.DataArray
        mock_data = xr.DataArray(
            np.random.rand(2, 2),
            coords={"lat": [0, 1], "lon": [0, 1]},
            dims=["lat", "lon"],
            name="mock_variable",
        )
        mock_soilgrids_instance.get_coverage_data.return_value = mock_data

        # Call the function with mock data
        data = soilgrids.get_soilgrids_data(
            service_id="mock_service",
            coverage_id="mock_coverage",
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs="mock_crs",
            output_path="mock_output.nc",
        )

        # Assert that the SoilGrids class was instantiated
        mock_soilgrids.assert_called_once()

        # Assert that the get_coverage_data method was called with the correct arguments
        mock_soilgrids_instance.get_coverage_data.assert_called_once_with(
            service_id="mock_service",
            coverage_id="mock_coverage",
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs="mock_crs",
        )

        # Assert that the to_netcdf method was called with the correct arguments
        expected_encoding = {"mock_variable": {"zlib": True, "complevel": 5}}
        mock_to_netcdf.assert_called_once_with(
            "mock_output.nc", encoding=expected_encoding
        )

        # Assert that mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert that the function returned the mock data
        xr.testing.assert_equal(data, mock_data)

    @patch("pathlib.Path.mkdir")
    @patch("xarray.DataArray.to_netcdf")
    @patch("fengsha_prep.data_downloaders.soilgrids.SoilGrids")
    def test_get_soilgrids_data_async(self, mock_soilgrids, mock_to_netcdf, mock_mkdir):
        """
        Tests the get_soilgrids_data_async function.
        """
        # Create a mock SoilGrids object
        mock_soilgrids_instance = mock_soilgrids.return_value

        # Create a mock xarray.DataArray
        mock_data = xr.DataArray(
            np.random.rand(2, 2),
            coords={"lat": [0, 1], "lon": [0, 1]},
            dims=["lat", "lon"],
            name="mock_variable",
        )
        mock_soilgrids_instance.get_coverage_data.return_value = mock_data

        # Call the async function
        data = asyncio.run(
            soilgrids.get_soilgrids_data_async(
                service_id="mock_service",
                coverage_id="mock_coverage",
                west=-10,
                south=-10,
                east=10,
                north=10,
                crs="mock_crs",
                output_path="mock_output.nc",
            )
        )

        # Assert that the SoilGrids class was instantiated
        mock_soilgrids.assert_called_once()

        # Assert that the get_coverage_data method was called with the correct arguments
        mock_soilgrids_instance.get_coverage_data.assert_called_once_with(
            service_id="mock_service",
            coverage_id="mock_coverage",
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs="mock_crs",
        )

        # Assert that the to_netcdf method was called with the correct arguments
        expected_encoding = {"mock_variable": {"zlib": True, "complevel": 5}}
        mock_to_netcdf.assert_called_once_with(
            "mock_output.nc", encoding=expected_encoding
        )
        # Assert that mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert that the function returned the mock data
        xr.testing.assert_equal(data, mock_data)


if __name__ == "__main__":
    unittest.main()
