import datetime
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from src.fengsha_prep.DustSCAN import detect_dust, process_scene


class TestDustScan(unittest.TestCase):

    def test_detect_dust(self):
        """
        Test the detect_dust function with mock data.
        """
        # Create a mock Scene object
        mock_scene = MagicMock()

        # --- Mock Data Setup ---
        shape = (10, 10)
        b08_data = np.full(shape, 280)
        b10_data = np.full(shape, 290)

        # This initial setup should NOT be detected as dust.
        # Let's set diff_12_10 to be 0, which is not < -0.5.
        b12_data = np.full(shape, 290)

        # Now, create a patch in the bottom-right that SHOULD be detected as dust.
        # This will make diff_12_10 = 280 - 290 = -10, which is < -0.5
        # The other conditions (diff_10_8 > 2.0 and temp_10 > 280) are also met.
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

        # Call the function with the mock data
        dust_mask = detect_dust(mock_scene, 'goes16', thresholds)

        # Check that the output is an xarray DataArray
        self.assertIsInstance(dust_mask, xr.DataArray)

        # Check that the dust mask has the correct shape
        self.assertEqual(dust_mask.shape, (10, 10))

        # Check that the dust mask has the correct values
        self.assertTrue(np.all(dust_mask.values[5:, 5:] == True))
        self.assertTrue(np.all(dust_mask.values[:5, :5] == False))


class TestAsyncDustScan(unittest.IsolatedAsyncioTestCase):
    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    async def test_process_scene_success(self, mock_sync_processor):
        """
        Test the async process_scene function for a successful run.
        """
        # --- Mock Setup ---
        mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
        mock_sat_id = 'goes16'
        mock_thresholds = {'key': 'value'}
        expected_events = [
            {'lat': 34.5, 'lon': -101.2, 'area': 50},
            {'lat': 35.1, 'lon': -102.5, 'area': 120}
        ]
        # The function called by asyncio.to_thread is _process_scene_sync
        mock_sync_processor.return_value = expected_events

        # --- Call the async function ---
        events = await process_scene(mock_scn_time, mock_sat_id, mock_thresholds)

        # --- Assertions ---
        # Check that our synchronous helper was called correctly
        mock_sync_processor.assert_called_once_with(
            mock_scn_time, mock_sat_id, mock_thresholds
        )
        # Check that the events are returned as expected
        self.assertEqual(events, expected_events)

    @patch('src.fengsha_prep.DustSCAN._process_scene_sync')
    async def test_process_scene_exception(self, mock_sync_processor):
        """
        Test the async process_scene function when an exception occurs.
        """
        # --- Mock Setup ---
        mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
        mock_sat_id = 'goes16'
        mock_thresholds = {'key': 'value'}
        mock_sync_processor.side_effect = Exception("Something went wrong")

        # --- Call the async function ---
        # We expect it to catch the exception and return None
        events = await process_scene(mock_scn_time, mock_sat_id, mock_thresholds)

        # --- Assertions ---
        mock_sync_processor.assert_called_once_with(
            mock_scn_time, mock_sat_id, mock_thresholds
        )
        self.assertIsNone(events)


if __name__ == '__main__':
    unittest.main()
