import datetime
from unittest.mock import MagicMock

import numpy as np
import xarray as xr
import pytest
from satpy import Scene
from unittest.mock import patch

from fengsha_prep.pipelines.dust_scan.core import (
    _process_scene_sync,
    detect_dust,
    cluster_events,
    DEFAULT_THRESHOLDS,
    dust_scan_pipeline,
    load_scene_data,
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
    coords = {'y': lat, 'x': lon, 'lat': (('y', 'x'), np.full((256, 256), lat[:, None])),
              'lon': (('y', 'x'), np.full((256, 256), lon[None, :]))}

    bands = {
        'C11': xr.DataArray(data[0], dims=('y', 'x'), coords=coords),
        'C13': xr.DataArray(data[1], dims=('y', 'x'), coords=coords),
        'C15': xr.DataArray(data[2], dims=('y', 'x'), coords=coords)
    }

    # Use a mock for the Satpy Scene object
    scene = MagicMock(spec=Scene)
    scene.__getitem__.side_effect = lambda key: bands[key]
    scene.keys.return_value = bands.keys()

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
    assert 'history' in dust_mask.attrs
    assert 'goes16' in dust_mask.attrs['history']


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
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    dust_mask = xr.DataArray(
        mask_data,
        dims=('y', 'x'),
        coords={'lat': (('y', 'x'), lat_grid), 'lon': (('y', 'x'), lon_grid)}
    )

    scn_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    sat_id = "goes16"
    events = cluster_events(dust_mask, scn_time, sat_id)

    # --- Verification ---
    # 1. Check that one and only one event was detected
    assert len(events) == 1

    # 2. Check the properties of the detected event
    event = events[0]
    assert event['satellite'] == sat_id
    assert event['area_pixels'] == 50 * 50
    # Check that the centroid is within the expected lat/lon range
    assert 34.5 < event['latitude'] < 35.5
    assert -95.5 < event['longitude'] < -94.5


def test_cluster_events_no_dust():
    """
    Tests that cluster_events returns an empty list when there's no dust.
    """
    # Create an empty dust mask
    mask_data = np.zeros((256, 256), dtype=bool)
    lat = np.linspace(30, 40, 256)
    lon = np.linspace(-100, -90, 256)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    dust_mask = xr.DataArray(
        mask_data,
        dims=('y', 'x'),
        coords={'lat': (('y', 'x'), lat_grid), 'lon': (('y', 'x'), lon_grid)}
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
        coords={'lat': (('y', 'x'), lat2d), 'lon': (('y', 'x'), lon2d)},
        dims=('y', 'x')
    )
    scn_time = datetime.datetime.now()
    events = cluster_events(dust_mask, scn_time, 'goes16')

    assert len(events) == 2
    # Check that the areas are approximately correct
    assert abs(events[0]['area_pixels'] - 400) < 20
    assert abs(events[1]['area_pixels'] - 400) < 20
    # Check that the centroids are in different locations
    assert abs(events[0]['latitude'] - events[1]['latitude']) > 1.0


@pytest.mark.asyncio
@patch('fengsha_prep.pipelines.dust_scan.core._process_scene_sync')
@patch('fengsha_prep.pipelines.dust_scan.core.load_scene_data')
async def test_dust_scan_pipeline_success(mock_load_scene, mock_process_sync):
    """
    Tests the async dust_scan_pipeline for a successful run.
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
    assert events == expected_events


@pytest.mark.asyncio
@patch('fengsha_prep.pipelines.dust_scan.core.load_scene_data')
async def test_dust_scan_pipeline_load_fails(mock_load_scene):
    """
    Tests the async pipeline when scene loading returns None.
    """
    mock_load_scene.return_value = None
    events = await dust_scan_pipeline(datetime.datetime.now(), 'goes16', {})
    assert events is None


@pytest.mark.asyncio
@patch('fengsha_prep.pipelines.dust_scan.core._process_scene_sync')
@patch('fengsha_prep.pipelines.dust_scan.core.load_scene_data')
async def test_dust_scan_pipeline_process_fails(mock_load_scene, mock_process_sync):
    """
    Tests the async pipeline when the synchronous processing part fails.
    """
    mock_load_scene.return_value = MagicMock()
    mock_process_sync.side_effect = Exception("Processing failed")

    events = await dust_scan_pipeline(datetime.datetime.now(), 'goes16', {})
    assert events is None


def test_process_scene_sync_integration():
    """
    Integration test for the synchronous processing part of the pipeline.
    """
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

    mock_scene = MagicMock(spec=Scene)
    mock_scene.__getitem__.side_effect = lambda key: {
        'C11': xr.DataArray(b08_data, coords=coords, dims=dims),
        'C13': xr.DataArray(b10_data, coords=coords, dims=dims),
        'C15': xr.DataArray(b12_data, coords=coords, dims=dims),
    }[key]
    mock_scene.keys.return_value = ['C11', 'C13', 'C15']


    scn_time = datetime.datetime.now()
    sat_id = 'goes16'

    events = _process_scene_sync(mock_scene, scn_time, sat_id, DEFAULT_THRESHOLDS)

    assert events is not None
    assert len(events) == 1
    event = events[0]
    assert abs(event['latitude'] - 32.5) < 0.1
    assert abs(event['longitude'] - -97.5) < 0.1
    assert event['area_pixels'] == 100
    assert event['satellite'] == 'goes16'


@patch('fengsha_prep.pipelines.dust_scan.core.goes_s3')
def test_load_scene_data_file_not_found(mock_goes_s3):
    """
    Tests that load_scene_data returns None when S3 files are not found.
    """
    mock_goes_s3.SATELLITE_CONFIG = {'goes16': {}}
    mock_goes_s3.get_s3_path.side_effect = FileNotFoundError
    scn = load_scene_data(datetime.datetime.now(), 'goes16')
    assert scn is None


@patch('fengsha_prep.pipelines.dust_scan.core.glob')
@patch('fengsha_prep.pipelines.dust_scan.core.Scene')
def test_load_scene_data_local_fallback(mock_scene_cls, mock_glob):
    """
    Tests the local file loading fallback for a non-GOES satellite.
    """
    mock_glob.glob.return_value = ['data/himawari_file.nc']
    mock_scene_instance = mock_scene_cls.return_value
    mock_scene_instance.resample.return_value = mock_scene_instance


    scn_time = datetime.datetime.now()
    sat_id = 'himawari8'

    scn = load_scene_data(scn_time, sat_id)

    assert scn is not None
    mock_glob.glob.assert_called_once()
    mock_scene_cls.assert_called_once_with(filenames=['data/himawari_file.nc'], reader='ahi_hsd')
    mock_scene_instance.load.assert_called_once_with(['B11', 'B13', 'B15'])


@pytest.mark.asyncio
@patch('fengsha_prep.pipelines.dust_scan.core.logging')
@patch('fengsha_prep.pipelines.dust_scan.core._process_scene_sync')
@patch('fengsha_prep.pipelines.dust_scan.core.load_scene_data')
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
    result = await dust_scan_pipeline(scn_time, 'goes16', {})

    assert result is None
    mock_logging.error.assert_called_once()
    log_message = mock_logging.error.call_args[0][0]
    assert "Data processing error" in log_message
    assert "Invalid data dimensions" in log_message
