import datetime
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from src.fengsha_prep.DustSCAN import (
    cluster_events,
    detect_dust,
    load_scene_data,
    process_scene,
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
    @patch('src.fengsha_prep.DustSCAN.dust_scan_pipeline')
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

    @patch('src.fengsha_prep.DustSCAN.dust_scan_pipeline')
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
    @patch('src.fengsha_prep.DustSCAN.Scene')
    @patch('src.fengsha_prep.DustSCAN.goes_s3')
    def test_full_pipeline_with_mock_s3(self, mock_goes_s3, mock_scene_cls):
        """
        Integration test for the synchronous pipeline with mocked S3 data.
        """
        mock_s3_path = "s3://mock-bucket/mock-data"
        mock_goes_s3.get_s3_path.return_value = mock_s3_path
        mock_goes_s3.SATELLITE_CONFIG = {'goes16': {}}
        mock_goes_s3.SATELLITE_BANDS = {'goes16': ['C11', 'C13', 'C15']}

        mock_scene_instance = mock_scene_cls.return_value
        mock_resampled_scene = mock_scene_instance.resample.return_value

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

        mock_resampled_scene.__getitem__.side_effect = lambda key: {
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

        events = dust_scan_pipeline(scn_time, sat_id, thresholds)

        mock_goes_s3.get_s3_path.assert_called_once_with(sat_id, scn_time)
        mock_scene_cls.assert_called_once_with(reader='abi_l1b', filenames=[mock_s3_path])
        mock_scene_instance.load.assert_called_once_with(['C11', 'C13', 'C15'])
        mock_scene_instance.resample.assert_called_once_with(resampler='native')

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
