import datetime
from unittest.mock import AsyncMock

import pytest

from fengsha_prep.common.satellite import _parse_s3_timestamp, get_s3_path


def test_parse_s3_timestamp():
    """Test the timestamp parsing from a GOES S3 filename."""
    filename = (
        "OR_ABI-L1b-RadC-M6C13_G16_s20231001501172_e20231001503545_c20231001503598.nc"
    )
    # Note: Day of year 100 is April 10th in a non-leap year (2023)
    expected_time = datetime.datetime(2023, 4, 10, 15, 1, 17)
    assert _parse_s3_timestamp(filename) == expected_time


@pytest.mark.asyncio
async def test_get_s3_path_success():
    """Test that the correct S3 file path is selected asynchronously."""
    mock_s3 = AsyncMock()
    mock_files = [
        "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001500000_e...nc",
        "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001515000_e...nc",
        "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001530000_e...nc",
    ]
    # The function calls ls for hour-1, hour, and hour+1.
    # We set the side effect to simulate finding nothing, then files, then nothing.
    # The order is preserved by asyncio.gather.
    mock_s3.ls.side_effect = [[], mock_files, []]

    scn_time = datetime.datetime(2023, 10, 27, 15, 14, 0)  # Closest to the 15:15 file
    sat_id = "goes16"
    expected_path = "s3://" + mock_files[1]

    result_path = await get_s3_path(mock_s3, sat_id, scn_time)
    assert result_path == expected_path


@pytest.mark.asyncio
async def test_get_s3_path_edge_of_hour():
    """Test S3 path selection when the closest file is in the next hour."""
    mock_s3 = AsyncMock()
    files_hour15 = [
        "noaa-goes16/ABI-L1b-RadC/2023/300/15/OR_ABI-L1b-RadC-M6C13_G16_s20233001545000_e...nc",
    ]
    files_hour16 = [
        "noaa-goes16/ABI-L1b-RadC/2023/300/16/OR_ABI-L1b-RadC-M6C13_G16_s20233001600000_e...nc",
    ]
    # Simulate finding files in hour 15 and 16. The function searches
    # hour-1, hour, and hour+1 relative to scn_time (15:59), so it will
    # search hours 14, 15, and 16.
    mock_s3.ls.side_effect = [[], files_hour15, files_hour16]

    scn_time = datetime.datetime(2023, 10, 27, 15, 59, 0)  # Closest to 16:00
    expected_path = "s3://" + files_hour16[0]

    result_path = await get_s3_path(mock_s3, "goes16", scn_time)
    assert result_path == expected_path


@pytest.mark.asyncio
async def test_get_s3_path_no_files_found():
    """Test that FileNotFoundError is raised when no files are found."""
    mock_s3 = AsyncMock()
    # s3.ls returns an empty list for all calls
    mock_s3.ls.return_value = []
    with pytest.raises(FileNotFoundError):
        await get_s3_path(mock_s3, "goes16", datetime.datetime.now())


@pytest.mark.asyncio
async def test_get_s3_path_unsupported_satellite():
    """Test that a ValueError is raised for an unsupported satellite."""
    mock_s3 = AsyncMock()
    with pytest.raises(ValueError):
        await get_s3_path(mock_s3, "landsat8", datetime.datetime.now())
