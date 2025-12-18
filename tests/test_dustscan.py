import datetime
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from src.fengsha_prep.DustSCAN import (
    cluster_events,
    detect_dust,
    process_scene,
    _process_scene_sync
)


class TestDustScan(unittest.TestCase):

    def test_detect_dust(self):
        """
        Test the detect_dust function with mock data.
        """
        mock_scene = MagicMock()
        shape = (10, 10)
        b08_data = np.full(shape, 280)
        b10_data = np.full(shape, 290)
        b12_data = np.full(shape, 290)
        b12_data[5:, 5:] = 280

        mock_scene.__getitem__.side_effect = lambda key: {
            'C11': xr.DataArray(b08_data),
            'C13': xr.DataArray(b10_data),
            'C15': xr.DataArray(b12_data)
        }[key]

        thresholds = {
            'diff_12_10': -0.5,
            'diff_10_8': 2.0,
            'temp_10': 280
        }

        dust_mask = detect_dust(mock_scene, 'goes16', thresholds)

        self.assertIsInstance(dust_mask, xr.DataArray)
        self.assertEqual(dust_mask.shape, (10, 10))
        self.assertTrue(np.all(dust_mask.values[5:, 5:] == True))
        self.assertTrue(np.all(dust_mask.values[:5, :5] == False))

    def test_cluster_events_no_dust(self):
        """Test cluster_events with a mask that contains no dust."""
        dust_mask = xr.DataArray(np.zeros((10, 10), dtype=bool), dims=('y', 'x'))
        events = cluster_events(dust_mask, datetime.datetime.now(), 'goes16')
        self.assertEqual(len(events), 0)


class TestAsyncDustScan(unittest.IsolatedAsyncioTestCase):
    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    async def test_process_scene_success(self, mock_sync_processor):
        """
        Test the async process_scene function for a successful run.
        """
        mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
        mock_sat_id = 'goes16'
        mock_thresholds = {'key': 'value'}
        expected_events = [
            {'lat': 34.5, 'lon': -101.2, 'area': 50},
            {'lat': 35.1, 'lon': -102.5, 'area': 120}
        ]
        mock_sync_processor.return_value = expected_events

        events = await process_scene(mock_scn_time, mock_sat_id, mock_thresholds)

        mock_sync_processor.assert_called_once_with(
            mock_scn_time, mock_sat_id, mock_thresholds
        )
        self.assertEqual(events, expected_events)

    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    async def test_process_scene_exception(self, mock_sync_processor):
        """
        Test the async process_scene function when an exception occurs.
        """
        mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
        mock_sat_id = 'goes16'
        mock_thresholds = {'key': 'value'}
        mock_sync_processor.side_effect = Exception("Something went wrong")

        events = await process_scene(mock_scn_time, mock_sat_id, mock_thresholds)

        mock_sync_processor.assert_called_once_with(
            mock_scn_time, mock_sat_id, mock_thresholds
        )
        self.assertIsNone(events)


class TestDustScanIntegration(unittest.TestCase):
    @patch('src.fengsha_prep.DustSCAN.load_scene_data')
    def test_full_pipeline_with_mock_scene(self, mock_load_scene):
        """
        Integration test for the synchronous pipeline (_process_scene_sync)
        using a mock scene.
        """
        mock_scene = MagicMock()
        # Use a higher resolution grid to ensure points are close enough for DBSCAN
        shape = (600, 600)
        lats = np.linspace(30, 35, shape[0])
        lons = np.linspace(-100, -95, shape[1])
        lon2d, lat2d = np.meshgrid(lons, lats)

        coords = {'lat': (('y', 'x'), lat2d), 'lon': (('y', 'x'), lon2d)}
        dims = ['y', 'x']

        b08_data = 280 * np.ones(shape)
        b10_data = 290 * np.ones(shape)
        b12_data = 290 * np.ones(shape)
        # Create a 10x10 pixel patch of dust-like values in the center of the grid
        b12_data[295:305, 295:305] = 280

        mock_scene.__getitem__.side_effect = lambda key: {
            'C11': xr.DataArray(b08_data, coords=coords, dims=dims),
            'C13': xr.DataArray(b10_data, coords=coords, dims=dims),
            'C15': xr.DataArray(b12_data, coords=coords, dims=dims),
        }[key]
        mock_load_scene.return_value = mock_scene

        scn_time = datetime.datetime.now()
        sat_id = 'goes16'
        thresholds = {
            'diff_12_10': -0.5,
            'diff_10_8': 2.0,
            'temp_10': 280
        }

        events = _process_scene_sync(scn_time, sat_id, thresholds)

        self.assertIsNotNone(events)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event['latitude'], 32.5, places=1)
        self.assertAlmostEqual(event['longitude'], -97.5, places=1)
        self.assertEqual(event['area_pixels'], 100)
        self.assertEqual(event['satellite'], 'goes16')


if __name__ == '__main__':
    unittest.main()
