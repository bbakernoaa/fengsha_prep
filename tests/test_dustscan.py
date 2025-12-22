import datetime
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from src.fengsha_prep.DustSCAN import (
    _process_scene_sync,
    cluster_events,
    detect_dust,
    load_scene_data,
    dust_scan_pipeline
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

    def test_cluster_events_with_dust(self):
        """
        Test the cluster_events function with a mock dust mask containing two plumes.
        """
        shape = (500, 500)
        lats = np.linspace(30, 35, shape[0])
        lons = np.linspace(-100, -95, shape[1])
        lon2d, lat2d = np.meshgrid(lons, lats)

        dust_data = np.zeros(shape, dtype=bool)
        # Plume 1
        dust_data[100:120, 100:120] = True
        # Plume 2
        dust_data[400:420, 400:420] = True

        dust_mask = xr.DataArray(
            dust_data,
            coords={'lat': (('y', 'x'), lat2d), 'lon': (('y', 'x'), lon2d)},
            dims=('y', 'x')
        )
        scn_time = datetime.datetime.now()
        events = cluster_events(dust_mask, scn_time, 'goes16')

        self.assertEqual(len(events), 2)
        # Check area, ensuring they are roughly the correct size
        self.assertAlmostEqual(events[0]['area_pixels'], 400, delta=20)
        self.assertAlmostEqual(events[1]['area_pixels'], 400, delta=20)
        # Check that the centroids are in different locations
        self.assertNotAlmostEqual(events[0]['latitude'], events[1]['latitude'])


class TestAsyncDustScan(unittest.IsolatedAsyncioTestCase):
    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    @patch('src.fengsha_prep.DustSCAN.load_scene_data')
    async def test_dust_scan_pipeline_success(self, mock_load_scene, mock_process_sync):
        """
        Test the async dust_scan_pipeline for a successful run.
        """
        mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
        mock_sat_id = 'goes16'
        mock_thresholds = {'key': 'value'}
        mock_scene_obj = MagicMock()
        expected_events = [{'event': 1}]

        mock_load_scene.return_value = mock_scene_obj
        mock_process_sync.return_value = expected_events

        events = await dust_scan_pipeline(mock_scn_time, mock_sat_id, mock_thresholds)

        mock_load_scene.assert_called_once_with(mock_scn_time, mock_sat_id)
        mock_process_sync.assert_called_once_with(
            mock_scene_obj, mock_scn_time, mock_sat_id, mock_thresholds
        )
        self.assertEqual(events, expected_events)

    @patch('src.fengsha_prep.DustSCAN.load_scene_data')
    async def test_dust_scan_pipeline_load_fails(self, mock_load_scene):
        """
        Test the async pipeline when scene loading returns None.
        """
        mock_load_scene.return_value = None
        events = await dust_scan_pipeline(datetime.datetime.now(), 'goes16', {})
        self.assertIsNone(events)

    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    @patch('src.fengsha_prep.DustSCAN.load_scene_data')
    async def test_dust_scan_pipeline_process_fails(self, mock_load_scene, mock_process_sync):
        """
        Test the async pipeline when the synchronous processing part fails.
        """
        mock_load_scene.return_value = MagicMock()
        mock_process_sync.side_effect = Exception("Processing failed")

        events = await dust_scan_pipeline(datetime.datetime.now(), 'goes16', {})
        self.assertIsNone(events)


class TestDustScanIntegration(unittest.TestCase):
    def test_process_scene_sync_integration(self):
        """
        Integration test for the synchronous processing part of the pipeline.
        """
        # Create a mock Scene object with realistic data
        shape = (600, 600)
        lats = np.linspace(30, 35, shape[0])
        lons = np.linspace(-100, -95, shape[1])
        lon2d, lat2d = np.meshgrid(lons, lats)
        coords = {'lat': (('y', 'x'), lat2d), 'lon': (('y', 'x'), lon2d)}
        dims = ['y', 'x']

        b08_data = 280 * np.ones(shape)
        b10_data = 290 * np.ones(shape)
        b12_data = 290 * np.ones(shape)
        b12_data[295:305, 295:305] = 280  # Dust patch

        mock_scene = MagicMock()
        mock_scene.__getitem__.side_effect = lambda key: {
            'C11': xr.DataArray(b08_data, coords=coords, dims=dims),
            'C13': xr.DataArray(b10_data, coords=coords, dims=dims),
            'C15': xr.DataArray(b12_data, coords=coords, dims=dims),
        }[key]

        scn_time = datetime.datetime.now()
        sat_id = 'goes16'
        thresholds = {
            'diff_12_10': -0.5,
            'diff_10_8': 2.0,
            'temp_10': 280
        }

        # Run the synchronous processing function
        events = _process_scene_sync(mock_scene, scn_time, sat_id, thresholds)

        # Assertions
        self.assertIsNotNone(events)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event['latitude'], 32.5, places=1)
        self.assertAlmostEqual(event['longitude'], -97.5, places=1)
        self.assertEqual(event['area_pixels'], 100)
        self.assertEqual(event['satellite'], 'goes16')

    @patch('src.fengsha_prep.DustSCAN.goes_s3')
    def test_load_scene_data_file_not_found(self, mock_goes_s3):
        """
        Test that load_scene_data returns None when S3 files are not found.
        """
        mock_goes_s3.SATELLITE_CONFIG = {'goes16': {}}
        mock_goes_s3.get_s3_path.side_effect = FileNotFoundError
        scn = load_scene_data(datetime.datetime.now(), 'goes16')
        self.assertIsNone(scn)

    @patch('src.fengsha_prep.DustSCAN.glob')
    @patch('src.fengsha_prep.DustSCAN.Scene')
    def test_load_scene_data_local_fallback(self, mock_scene_cls, mock_glob):
        """
        Test the local file loading fallback for a non-GOES satellite.
        """
        mock_glob.glob.return_value = ['data/himawari_file.nc']
        mock_scene_instance = mock_scene_cls.return_value

        scn_time = datetime.datetime.now()
        sat_id = 'himawari8'

        scn = load_scene_data(scn_time, sat_id)

        self.assertIsNotNone(scn)
        mock_glob.glob.assert_called_once()
        mock_scene_cls.assert_called_once_with(filenames=['data/himawari_file.nc'], reader='ahi_hsd')
        mock_scene_instance.load.assert_called_once_with(['B11', 'B13', 'B15'])


if __name__ == '__main__':
    unittest.main()
