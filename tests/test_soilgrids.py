"""
Tests for the soilgrids module.
"""

import unittest
from unittest.mock import patch
from src.fengsha_prep import soilgrids

class TestSoilgrids(unittest.TestCase):
    """
    Tests for the soilgrids module.
    """

    @patch('src.fengsha_prep.soilgrids.SoilGrids')
    def test_get_soilgrids_data(self, mock_soilgrids):
        """
        Tests the get_soilgrids_data function.
        """
        # Create a mock SoilGrids object
        mock_soilgrids_instance = mock_soilgrids.return_value
        mock_soilgrids_instance.get_coverage_data.return_value = "Mock data"

        # Call the function with mock data
        data = soilgrids.get_soilgrids_data(
            service_id='mock_service',
            coverage_id='mock_coverage',
            west=-10,
            south=-10,
            east=10,
            north=10,
            crs='mock_crs',
            output='mock_output.tif'
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
            output='mock_output.tif'
        )

        # Assert that the function returned the mock data
        self.assertEqual(data, "Mock data")

if __name__ == '__main__':
    unittest.main()
