"""
Tests for the regrid module.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the xesmf and esmpy modules to avoid installation issues
# This must be done BEFORE importing the regrid module.
sys.modules['xesmf'] = MagicMock()
sys.modules['esmpy'] = MagicMock()

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from fengsha_prep import regrid  # noqa: E402

class TestRegrid(unittest.TestCase):
    """
    Tests for the regrid module.
    """

    @patch('fengsha_prep.regrid.xe')
    def test_regrid_modis_to_rectilinear(self, mock_xe):
        """
        Tests the regrid_modis_to_rectilinear function.
        """
        # Create a dummy sinusoidal dataset
        ds_sinu = xr.Dataset({
            'foo': (('y', 'x'), np.random.rand(10, 20)),
            'lat': (('y', 'x'), np.random.uniform(20, 50, size=(10, 20))),
            'lon': (('y', 'x'), np.random.uniform(-120, -70, size=(10, 20))),
        })

        # Define the output grid
        lon_min, lon_max, d_lon = -110, -80, 1.0
        lat_min, lat_max, d_lat = 30, 45, 1.0

        # Mock the Regridder
        mock_regridder_instance = MagicMock()
        regridded_data = xr.Dataset({
            'foo': (('lat', 'lon'), np.random.rand(16, 31)),
            'lat': ('lat', np.arange(lat_min, lat_max + d_lat, d_lat)),
            'lon': ('lon', np.arange(lon_min, lon_max + d_lon, d_lon)),
        })
        mock_regridder_instance.return_value = regridded_data
        mock_xe.Regridder.return_value = mock_regridder_instance

        # Call the regridding function
        ds_regridded = regrid.regrid_modis_to_rectilinear(
            ds_sinu, 'foo', lon_min, lon_max, d_lon, lat_min, lat_max, d_lat
        )

        # Assert that the output has the correct dimensions
        self.assertIn('lat', ds_regridded.dims)
        self.assertIn('lon', ds_regridded.dims)

        # Assert that the output coordinates are within the specified bounds
        self.assertGreaterEqual(ds_regridded.lon.min(), lon_min)
        self.assertLessEqual(ds_regridded.lon.max(), lon_max)
        self.assertGreaterEqual(ds_regridded.lat.min(), lat_min)
        self.assertLessEqual(ds_regridded.lat.max(), lat_max)

if __name__ == '__main__':
    unittest.main()
