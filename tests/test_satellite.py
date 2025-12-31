import datetime
import unittest
from unittest.mock import patch

from fengsha_prep.common.satellite import get_s3_path, _parse_s3_timestamp


class TestGoesS3(unittest.TestCase):
    def test_parse_s3_timestamp(self):
        """Test the timestamp parsing from a GOES S3 filename."""
        filename = "OR_ABI-L1b-RadC-M6C13_G16_s20231001501172_e20231001503545_c20231001503598.nc"
        expected_time = datetime.datetime(2023, 4, 10, 15, 1, 17)
        self.assertEqual(_parse_s3_timestamp(filename), expected_time)

    @patch("fengsha_prep.common.satellite.fs")
    def test_get_s3_path_success(self, mock_s3fs):
        """Test that the correct S3 file path is selected."""
        mock_files = [
            "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001500000_e...nc",
            "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001515000_e...nc",
            "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001530000_e...nc",
        ]
        mock_s3fs.ls.side_effect = [FileNotFoundError, mock_files, FileNotFoundError]

        scn_time = datetime.datetime(
            2023, 10, 27, 15, 14, 0
        )  # Closest to the 15:15 file
        sat_id = "goes16"
        expected_path = "s3://" + mock_files[1]

        self.assertEqual(get_s3_path(sat_id, scn_time), expected_path)

    @patch("fengsha_prep.common.satellite.fs")
    def test_get_s3_path_edge_of_hour(self, mock_s3fs):
        """Test S3 path selection when the closest file is in the next hour."""
        files_hour15 = [
            "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001545000_e...nc",
        ]
        files_hour16 = [
            "noaa-goes16/ABI-L1b-RadC/2023/300/16/OR_ABI-L1b-RadC-M6C13_G16_s20233001600000_e...nc",
        ]
        # fs.ls will be called for hour-1, hour, and hour+1
        mock_s3fs.ls.side_effect = [FileNotFoundError, files_hour15, files_hour16]

        scn_time = datetime.datetime(2023, 10, 27, 15, 59, 0)  # Closest to 16:00
        expected_path = "s3://" + files_hour16[0]

        self.assertEqual(get_s3_path("goes16", scn_time), expected_path)

    @patch("fengsha_prep.common.satellite.fs")
    def test_get_s3_path_no_files_found(self, mock_s3fs):
        """Test that FileNotFoundError is raised when no files are in any searched S3 dir."""
        mock_s3fs.ls.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            get_s3_path("goes16", datetime.datetime.now())

    def test_get_s3_path_unsupported_satellite(self):
        """Test that a ValueError is raised for an unsupported satellite."""
        with self.assertRaises(ValueError):
            get_s3_path("landsat8", datetime.datetime.now())
