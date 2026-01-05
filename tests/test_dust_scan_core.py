import datetime
from unittest.mock import MagicMock, patch, ANY, AsyncMock

import numpy as np
import xarray as xr
import pytest
from satpy import Scene

from fengsha_prep.pipelines.dust_scan.core import (
    _process_scene_sync,
    detect_dust,
    cluster_events,
    DEFAULT_THRESHOLDS,
    dust_scan_pipeline,
    load_scene_data,
    _load_scene_from_s3,
    _load_scene_from_local,
)


@pytest.fixture
def mock_scene() -> Scene:
    """Creates a mock Satpy Scene with realistic data for testing."""
    # Create mock data for three different bands (8, 10, and 12 micrometers)
    # The data simulates a dust plume in the center of the scene
    data = np.full((3, 256, 256), 290.0, dtype=np.float32)

    # Dust Plume Simulation (values that should trigger the dust detection algorithm)
    # In the center of the image, create a region where:
    # B12 - B10 < -0.5  (e.g., B12=285, B10=286 -> -1)
    # B10 - B08 > 2.0   (e.g., B10=286, B08=283 -> 3)
    # B10 > 280         (e.g., 286 > 280)
    data[0, 100:150, 100:150] = 283.0  # Band 8
    data[1, 100:150, 100:150] = 286.0  # Band 10
    data[2, 100:150, 100:150] = 285.0  # Band 12

    # Create xarray DataArrays for each band
    lat = np.linspace(30, 40, 256)
    lon = np.linspace(-100, -90, 256)
    coords = {
        "y": lat,
        "x": lon,
        "lat": (("y", "x"), np.full((256, 256), lat[:, None])),
        "lon": (("y", "x"), np.full((256, 256), lon[None, :])),
    }

    bands = {
        "C11": xr.DataArray(data[0], dims=("y", "x"), coords=coords),
        "C13": xr.DataArray(data[1], dims=("y", "x"), coords=coords),
        "C15": xr.DataArray(data[2], dims=("y", "x"), coords=coords),
    }

    # Use a mock for the Satpy Scene object
    scene = Scene()
    scene["C11"] = bands["C11"]
    scene["C13"] = bands["C13"]
    scene["C15"] = bands["C15"]

    return scene


def test_detect_dust_identifies_plume(mock_scene):
    """
    Tests that the detect_dust function correctly identifies a dust plume
    based on the mock data and default thresholds.
    """
    sat_id = "goes16"
    dust_mask = detect_dust(mock_scene, sat_id, DEFAULT_THRESHOLDS)

    # --- Verification ---
    # 1. Check the output type
    assert isinstance(dust_mask, xr.DataArray)

    # 2. Check that dust was detected
    assert dust_mask.sum() > 0

    # 3. Check the detected area is the correct size (50x50 pixels)
    assert dust_mask.sum() == 50 * 50

    # 4. Check for the history attribute (provenance)
    assert "history" in dust_mask.attrs
    assert "goes16" in dust_mask.attrs["history"]


def test_cluster_events_finds_clusters():
    """
    Tests that the cluster_events function can identify a distinct cluster
    from a sample binary dust mask.
    """
    # Create a sample dust mask with a single, large cluster
    mask_data = np.zeros((256, 256), dtype=bool)
    mask_data[100:150, 100:150] = True

    lat = np.linspace(30, 40, 256)
    lon = np.linspace(-100, -90, 256)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    dust_mask = xr.DataArray(
        mask_data,
        dims=("y", "x"),
        coords={"lat": (("y", "x"), lat_grid), "lon": (("y", "x"), lon_grid)},
    )

    scn_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    sat_id = "goes16"
    events = cluster_events(dust_mask, scn_time, sat_id)

    # --- Verification ---
    # 1. Check that one and only one event was detected
    assert len(events) == 1

    # 2. Check the properties of the detected event
    event = events[0]
    assert event["satellite"] == sat_id
    assert event["area_pixels"] == 50 * 50
    # Check that the centroid is within the expected lat/lon range
    assert 34.5 < event["latitude"] < 35.5
    assert -95.5 < event["longitude"] < -94.5


def test_cluster_events_no_dust():
    """
    Tests that cluster_events returns an empty list when there's no dust.
    """
    # Create an empty dust mask
    mask_data = np.zeros((256, 256), dtype=bool)
    lat = np.linspace(30, 40, 256)
    lon = np.linspace(-100, -90, 256)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    dust_mask = xr.DataArray(
        mask_data,
        dims=("y", "x"),
        coords={"lat": (("y", "x"), lat_grid), "lon": (("y", "x"), lon_grid)},
    )

    scn_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    sat_id = "goes16"
    events = cluster_events(dust_mask, scn_time, sat_id)

    # --- Verification ---
    assert len(events) == 0


def test_cluster_events_with_multiple_plumes():
    """
    Tests that cluster_events can distinguish between two separate dust plumes.
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
        coords={"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
        dims=("y", "x"),
    )
    scn_time = datetime.datetime.now()
    events = cluster_events(dust_mask, scn_time, "goes16")

    assert len(events) == 2
    # Check that the areas are approximately correct
    assert abs(events[0]["area_pixels"] - 400) < 20
    assert abs(events[1]["area_pixels"] - 400) < 20
    # Check that the centroids are in different locations
    assert abs(events[0]["latitude"] - events[1]["latitude"]) > 1.0


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core._process_scene_sync")
@patch("fengsha_prep.pipelines.dust_scan.core.load_scene_data", new_callable=AsyncMock)
async def test_dust_scan_pipeline_success(mock_load_scene, mock_process_sync):
    """
    Tests the async dust_scan_pipeline for a successful run.
    """
    mock_scn_time = datetime.datetime(2023, 1, 1, 12, 0)
    mock_sat_id = "goes16"
    mock_thresholds = {"key": "value"}
    mock_scene_obj = MagicMock()
    expected_events = [{"event": 1}]

    # Configure the async mock
    mock_load_scene.return_value = mock_scene_obj
    # The sync mock remains the same
    mock_process_sync.return_value = expected_events

    events = await dust_scan_pipeline(mock_scn_time, mock_sat_id, mock_thresholds)

    mock_load_scene.assert_awaited_once_with(mock_scn_time, mock_sat_id, ANY)
    mock_process_sync.assert_called_once_with(
        mock_scene_obj, mock_scn_time, mock_sat_id, mock_thresholds
    )
    assert events == expected_events


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core.load_scene_data", new_callable=AsyncMock)
async def test_dust_scan_pipeline_load_fails(mock_load_scene):
    """
    Tests the async pipeline when scene loading returns None.
    """
    mock_load_scene.return_value = None
    events = await dust_scan_pipeline(datetime.datetime.now(), "goes16", {})
    assert events == []


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core._process_scene_sync")
@patch("fengsha_prep.pipelines.dust_scan.core.load_scene_data", new_callable=AsyncMock)
async def test_dust_scan_pipeline_process_fails(mock_load_scene, mock_process_sync):
    """
    Tests the async pipeline when the synchronous processing part fails.
    """
    mock_load_scene.return_value = MagicMock()
    mock_process_sync.side_effect = Exception("Processing failed")

    events = await dust_scan_pipeline(datetime.datetime.now(), "goes16", {})
    assert events == []


def test_process_scene_sync_integration():
    """
    Integration test for the synchronous processing part of the pipeline.
    """
    shape = (600, 600)
    lats = np.linspace(30, 35, shape[0])
    lons = np.linspace(-100, -95, shape[1])
    lon2d, lat2d = np.meshgrid(lons, lats)
    coords = {"lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)}
    dims = ["y", "x"]

    b08_data = 280 * np.ones(shape)
    b10_data = 290 * np.ones(shape)
    b12_data = 290 * np.ones(shape)
    b12_data[295:305, 295:305] = 280  # Dust patch

    mock_scene = Scene()
    mock_scene["C11"] = xr.DataArray(b08_data, coords=coords, dims=dims)
    mock_scene["C13"] = xr.DataArray(b10_data, coords=coords, dims=dims)
    mock_scene["C15"] = xr.DataArray(b12_data, coords=coords, dims=dims)

    scn_time = datetime.datetime.now()
    sat_id = "goes16"

    events = _process_scene_sync(mock_scene, scn_time, sat_id, DEFAULT_THRESHOLDS)

    assert events is not None
    assert len(events) == 1
    event = events[0]
    assert abs(event["latitude"] - 32.5) < 0.1
    assert abs(event["longitude"] - -97.5) < 0.1
    assert event["area_pixels"] == 100
    assert event["satellite"] == "goes16"


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core.asyncio.to_thread")
@patch("fengsha_prep.pipelines.dust_scan.core.s3fs.S3FileSystem")
@patch("fengsha_prep.pipelines.dust_scan.core.satellite")
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
@patch("fengsha_prep.pipelines.dust_scan.core.s3fs.S3FileSystem")
@patch("fengsha_prep.pipelines.dust_scan.core.satellite")
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


@patch("fengsha_prep.pipelines.dust_scan.core.glob")
@patch("fengsha_prep.pipelines.dust_scan.core.Scene")
def test_load_scene_from_local_success(mock_scene_cls, mock_glob):
    """
    Tests that _load_scene_from_local successfully loads a scene from the local filesystem.
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


@patch("fengsha_prep.pipelines.dust_scan.core.glob")
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
    "fengsha_prep.pipelines.dust_scan.core._load_scene_from_s3", new_callable=AsyncMock
)
@patch("fengsha_prep.pipelines.dust_scan.core.satellite")
async def test_load_scene_data_dispatches_to_s3(mock_satellite, mock_load_s3):
    """
    Tests that load_scene_data calls the async S3 loader for S3-based satellites.
    """
    mock_satellite.get_satellite_metadata.return_value = {"is_s3": True}
    scn_time = datetime.datetime.now()

    await load_scene_data(scn_time, "goes16")

    mock_load_s3.assert_awaited_once_with(scn_time, "goes16", {"is_s3": True})


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core.asyncio.to_thread")
@patch("fengsha_prep.pipelines.dust_scan.core._load_scene_from_local")
@patch("fengsha_prep.pipelines.dust_scan.core.satellite")
async def test_load_scene_data_dispatches_to_local(
    mock_satellite, mock_load_local, mock_to_thread
):
    """
    Tests that load_scene_data calls the local loader via to_thread for non-S3 satellites.
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


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.core.logging")
@patch("fengsha_prep.pipelines.dust_scan.core._process_scene_sync")
@patch("fengsha_prep.pipelines.dust_scan.core.load_scene_data")
async def test_dust_scan_pipeline_handles_processing_value_error(
    mock_load_scene, mock_process_sync, mock_logging
):
    """
    Tests that the pipeline correctly handles a ValueError during the
    CPU-bound processing stage and logs the appropriate error.
    """
    mock_load_scene.return_value = MagicMock(spec=Scene)
    mock_process_sync.side_effect = ValueError("Invalid data dimensions")

    scn_time = datetime.datetime(2023, 1, 1, 12, 0)
    result = await dust_scan_pipeline(scn_time, "goes16", {})

    assert result == []
    mock_logging.exception.assert_called_once()
    log_message = mock_logging.exception.call_args[0][0]
    assert f"Failed to process scene for {scn_time}" in log_message
