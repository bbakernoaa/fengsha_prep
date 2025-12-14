"""
Tests for the bnu module.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.fengsha_prep import bnu
import os
import shutil

class TestBnu(unittest.TestCase):
    """
    Tests for the bnu module.
    """
    def setUp(self):
        self.output_dir = 'test_bnu_data'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch('src.fengsha_prep.bnu.requests.get')
    @patch('src.fengsha_prep.bnu.tomllib.load')
    def test_get_bnu_data_with_mock_config(self, mock_load, mock_get):
        """
        Tests the get_bnu_data function with a mocked config file.
        """
        # Mock the config file
        mock_config = {
            'bnu_data': {
                'sand_urls': [
                    "http://not-a-real-site.com/sand1.nc",
                    "http://example.com/sand2.nc" # Keep one placeholder to test both paths
                ]
            }
        }
        mock_load.return_value = mock_config

        # Mock the requests.get call
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value.__enter__.return_value = mock_response

        # Call the function
        downloaded_files = bnu.get_bnu_data('sand', output_dir=self.output_dir)

        # Assert that the correct files were "downloaded"
        expected_files = [
            os.path.join(self.output_dir, 'sand1.nc'),
            os.path.join(self.output_dir, 'sand2.nc')
        ]
        self.assertEqual(downloaded_files, expected_files)

        # Check that the real download path was called
        mock_get.assert_called_once_with("http://not-a-real-site.com/sand1.nc", stream=True)

        # Check that a file was created for the placeholder URL
        self.assertTrue(os.path.exists(expected_files[1]))


if __name__ == '__main__':
    unittest.main()
