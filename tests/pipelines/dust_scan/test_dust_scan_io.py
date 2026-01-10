import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from satpy import Scene

from fengsha_prep.pipelines.dust_scan.io import (
    _load_scene_from_local,
    _load_scene_from_s3,
    load_scene_data,
)


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.io.asyncio.to_thread")
@patch("fengsha_prep.pipelines.dust_scan.io.s3fs.S3FileSystem")
@patch("fengsha_prep.pipelines.dust_scan.io.satellite")
async def test_load_scene_from_s3_async_success(
    mock_satellite, mock_s3fs, mock_to_thread
):
    """
    Tests the refactored, async-native _load_scene_from_s3 function.
    """
    # --- Setup ---
    mock_s3_instance = mock_s3fs.return_value
    mock_s3_instance.exists = AsyncMock(return_value=True)
    mock_satellite.get_s3_path = AsyncMock(return_value="s3://bucket/file.nc")

    mock_scene = MagicMock(spec=Scene)
    mock_to_thread.return_value = mock_scene
    meta = {"reader": "abi_l1b", "bands": ["C01", "C02", "C03"]}

    # --- Execution ---
    scn_time = datetime.datetime.now()
    scn = await _load_scene_from_s3(scn_time, "goes16", meta)

    # --- Verification ---
    assert scn is mock_scene
    mock_s3fs.assert_called_once_with(asynchronous=True, anon=True)
    mock_s3_instance.exists.assert_awaited_once_with("s3://bucket/file.nc")
    mock_to_thread.assert_awaited_once()


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.io.s3fs.S3FileSystem")
@patch("fengsha_prep.pipelines.dust_scan.io.satellite")
async def test_load_scene_from_s3_async_not_found(mock_satellite, mock_s3fs):
    """
    Tests that the async S3 loader returns None when the file does not exist.
    """
    # --- Setup ---
    mock_s3_instance = mock_s3fs.return_value
    mock_s3_instance.exists = AsyncMock(return_value=False)
    mock_satellite.get_s3_path.return_value = "s3://bucket/file.nc"
    meta = {"reader": "abi_l1b", "bands": ["C01", "C02", "C03"]}

    # --- Execution ---
    scn = await _load_scene_from_s3(datetime.datetime.now(), "goes16", meta)

    # --- Verification ---
    assert scn is None


@patch("fengsha_prep.pipelines.dust_scan.io.glob")
@patch("fengsha_prep.pipelines.dust_scan.io.Scene")
def test_load_scene_from_local_success(mock_scene_cls, mock_glob):
    """
    Tests that _load_scene_from_local successfully loads a scene from the
    local filesystem.
    """
    mock_glob.glob.return_value = ["data/himawari_file.nc"]
    meta = {"reader": "ahi_hsd", "bands": ["B11", "B13", "B15"]}
    mock_scene_instance = mock_scene_cls.return_value
    mock_scene_instance.resample.return_value = mock_scene_instance

    scn = _load_scene_from_local(datetime.datetime.now(), "himawari8", meta)

    assert scn is not None
    mock_scene_cls.assert_called_once_with(
        filenames=["data/himawari_file.nc"], reader="ahi_hsd"
    )
    mock_scene_instance.load.assert_called_once_with(["B11", "B13", "B15"])


@patch("fengsha_prep.pipelines.dust_scan.io.glob")
def test_load_scene_from_local_not_found(mock_glob):
    """
    Tests that _load_scene_from_local returns None when no files are found.
    """
    mock_glob.glob.return_value = []
    meta = {"reader": "ahi_hsd", "bands": ["B11", "B13", "B15"]}
    scn = _load_scene_from_local(datetime.datetime.now(), "himawari8", meta)
    assert scn is None


@pytest.mark.asyncio
@patch(
    "fengsha_prep.pipelines.dust_scan.io._load_scene_from_s3", new_callable=AsyncMock
)
@patch("fengsha_prep.pipelines.dust_scan.io.satellite")
async def test_load_scene_data_dispatches_to_s3(mock_satellite, mock_load_s3):
    """
    Tests that load_scene_data calls the async S3 loader for S3-based satellites.
    """
    mock_satellite.get_satellite_metadata.return_value = {"is_s3": True}
    scn_time = datetime.datetime.now()

    await load_scene_data(scn_time, "goes16")

    mock_load_s3.assert_awaited_once_with(scn_time, "goes16", {"is_s3": True})


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.io.asyncio.to_thread")
@patch("fengsha_prep.pipelines.dust_scan.io._load_scene_from_local")
@patch("fengsha_prep.pipelines.dust_scan.io.satellite")
async def test_load_scene_data_dispatches_to_local(
    mock_satellite, mock_load_local, mock_to_thread
):
    """
    Tests that load_scene_data calls the local loader via to_thread for non-S3
    satellites.
    """
    mock_satellite.get_satellite_metadata.return_value = {"is_s3": False}
    scn_time = datetime.datetime.now()

    await load_scene_data(scn_time, "himawari8", data_dir="test_dir")

    # Verify that the blocking function was called inside a thread
    mock_to_thread.assert_awaited_once()
    # And that the original function was passed to the thread with the correct args
    assert mock_to_thread.call_args[0][0] == mock_load_local
    assert mock_to_thread.call_args[0][1] == scn_time
    assert mock_to_thread.call_args[0][3] == {"is_s3": False}
    assert mock_to_thread.call_args[1]["data_dir"] == "test_dir"
