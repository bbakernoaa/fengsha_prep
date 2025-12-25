"""
Tests for the soilgrids module.
"""

import unittest
from unittest.mock import patch
from fengsha_prep.data_downloaders import soilgrids
import xarray as xr
import numpy as np

class TestSoilgrids(unittest.TestCase):
    """
    Tests for the soilgrids module.
    """

    @patch('xarray.DataArray.to_netcdf')
    @patch('fengsha_prep.data_downloaders.soilgrids.SoilGrids')
    def test_get_soilgrids_data(self, mock_soilgrids, mock_to_netcdf):
        """
        Tests the get_soilgrids_data function.
        """
        # Create a mock SoilGrids object
        mock_soilgrids_instance = mock_soilgrids.return_value

        # Create a mock xarray.DataArray
        mock_data = xr.DataArray(
            np.random.rand(2, 2),
            coords={'lat': [0, 1], 'lon': [0, 1]},
            dims=['lat', 'lon'],
            name='mock_variable'
        )
        mock_soilgrids_instance.get_coverage_data.return_value = mock_data

        # Call the function with mock data
        data = soilgrids.get_soilgrids_data(
            service_id='mock_service',
            coverage_id='mock_coverage',
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs='mock_crs',
            output_path='mock_output.nc'
        )

        # Assert that the SoilGrids class was instantiated
        mock_soilgrids.assert_called_once()

        # Assert that the get_coverage_data method was called with the correct arguments
        mock_soilgrids_instance.get_coverage_data.assert_called_once_with(
            service_id='mock_service',
            coverage_id='mock_coverage',
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs='mock_crs',
        )

        # Assert that the to_netcdf method was called with the correct arguments
        expected_encoding = {'mock_variable': {'zlib': True, 'complevel': 5}}
        mock_to_netcdf.assert_called_once_with('mock_output.nc', encoding=expected_encoding)

        # Assert that the function returned the mock data
        xr.testing.assert_equal(data, mock_data)

if __name__ == '__main__':
    unittest.main()
