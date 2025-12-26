import asyncio
import datetime
from unittest.mock import patch, ANY

import numpy as np
import pytest
import xarray as xr
from satpy import Scene

from fengsha_prep.pipelines.dust_scan import core


@pytest.fixture
def mock_scene() -> Scene:
    """Creates a mock Satpy Scene for testing."""
    # Define grid dimensions and coordinates
    lon = np.arange(-100, -90, 0.1)
    lat = np.arange(30, 40, 0.1)
    lons, lats = np.meshgrid(lon, lat)

    coords = {'lat': (('y', 'x'), lats), 'lon': (('y', 'x'), lons)}

    # Create mock band data for non-dusty areas
    b10_data = np.full(lons.shape, 270)  # temp_10 < 280 (fails this check)
    b12_data = b10_data + 1
    b08_data = b10_data + 1

    # Create a patch of dust that meets all conditions
    dusty_b10 = 300  # temp_10 > 280
    b10_data[20:40, 20:40] = dusty_b10
    b12_data[20:40, 20:40] = dusty_b10 - 1  # diff_12_10 < -0.5
    b08_data[20:40, 20:40] = dusty_b10 - 3  # diff_10_8 > 2.0

    scene = Scene()
    scene['C11'] = xr.DataArray(b08_data, dims=('y', 'x'), coords=coords, name='C11')
    scene['C13'] = xr.DataArray(b10_data, dims=('y', 'x'), coords=coords, name='C13')
    scene['C15'] = xr.DataArray(b12_data, dims=('y', 'x'), coords=coords, name='C15')

    return scene


def test_detect_dust_identifies_dust(mock_scene):
    """Verify that the dust detection algorithm correctly creates a mask."""
    sat_id = 'goes16'
    thresholds = core.DEFAULT_THRESHOLDS

    dust_mask = core.detect_dust(mock_scene, sat_id, thresholds)

    assert isinstance(dust_mask, xr.DataArray)
    # The mock data is set up to have a 20x20 dust patch
    assert dust_mask.sum().item() == 400
    # Check that a non-dusty area is correctly masked as 0
    assert dust_mask[0, 0].item() == 0
    # Check that a dusty area is correctly masked as 1
    assert dust_mask[20, 20].item() == 1
    assert 'history' in dust_mask.attrs


def test_cluster_events_finds_clusters():
    """Verify that DBSCAN correctly identifies dust plumes from a mask."""
    # Create a high-resolution grid so points are close enough for clustering
    grid_res = 100
    mask_data = np.zeros((grid_res, grid_res), dtype=bool)
    mask_data[10:30, 10:30] = True  # Cluster 1 (20x20=400px)
    mask_data[70:90, 70:90] = True  # Cluster 2 (20x20=400px)

    # Create coordinates over a smaller geographic area
    lat = np.linspace(30, 32, grid_res)
    lon = np.linspace(-100, -98, grid_res)
    lons, lats = np.meshgrid(lon, lat)

    dust_mask = xr.DataArray(
        mask_data,
        dims=('y', 'x'),
        coords={'lat': (('y', 'x'), lats), 'lon': (('y', 'x'), lons)}
    )

    scn_time = datetime.datetime.now(datetime.UTC)
    events = core.cluster_events(dust_mask, scn_time, 'goes16')

    assert len(events) == 2
    # Check properties of one of the events
    event = events[0]
    assert event['datetime'] == scn_time
    assert event['satellite'] == 'goes16'
    assert event['area_pixels'] == 400
    assert isinstance(event['latitude'], float)
    assert isinstance(event['longitude'], float)


def test_cluster_events_no_dust():
    """Verify that no events are returned when the dust mask is empty."""
    lat = np.linspace(30, 35, 10)
    lon = np.linspace(-100, -95, 10)
    lons, lats = np.meshgrid(lon, lat)
    coords = {'lat': (('y', 'x'), lats), 'lon': (('y', 'x'), lons)}

    dust_mask = xr.DataArray(np.zeros((10, 10), dtype=bool), dims=('y', 'x'), coords=coords)
    events = core.cluster_events(dust_mask, datetime.datetime.now(datetime.UTC), 'goes16')
    assert events == []


@patch('fengsha_prep.pipelines.dust_scan.core._load_scene_from_s3')
def test_load_scene_data_for_s3_satellite(mock_load_s3):
    """Verify that S3 data sources are routed to the S3 loader."""
    core.load_scene_data(datetime.datetime.now(datetime.UTC), 'goes16')
    mock_load_s3.assert_called_once()


@patch('fengsha_prep.pipelines.dust_scan.core._load_scene_from_local')
def test_load_scene_data_for_local_satellite(mock_load_local):
    """Verify that local data sources are routed to the local loader with data_dir."""
    core.load_scene_data(datetime.datetime.now(datetime.UTC), 'himawari8', data_dir='/test/dir')
    # Check that data_dir is passed through
    mock_load_local.assert_called_once_with(ANY, 'himawari8', ANY, data_dir='/test/dir')


@pytest.mark.asyncio
@patch('fengsha_prep.pipelines.dust_scan.core.load_scene_data')
async def test_dust_scan_pipeline_integration(mock_load_data, mock_scene):
    """Test the full pipeline orchestration from scene loading to clustering."""
    # The mock for load_scene_data should return the scene directly, not a Future,
    # because asyncio.to_thread will wrap the synchronous function call.
    mock_load_data.return_value = mock_scene

    scn_time = datetime.datetime(2023, 1, 1, 12, 0)
    sat_id = 'goes16'

    # The side effect for to_thread should handle both awaited and regular calls.
    async def async_side_effect(func, *args, **kwargs):
        # We call the function with its arguments. If it's a coroutine, we await it.
        # In this test, `load_scene_data` is a regular function passed to `to_thread`,
        # so it won't be awaited here, but the wrapper will be.
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    with patch('asyncio.to_thread', side_effect=async_side_effect) as mock_to_thread:
        events = await core.dust_scan_pipeline(scn_time, sat_id, core.DEFAULT_THRESHOLDS, data_dir=None)

        # Verify that load_scene_data was called via to_thread.
        mock_load_data.assert_called_once_with(scn_time, sat_id, None)

        # Two calls to `to_thread`: one for loading, one for processing.
        assert mock_to_thread.call_count == 2

        # Verify the final output based on our mock data
        assert len(events) == 1
        assert events[0]['area_pixels'] == 400
        assert events[0]['datetime'] == scn_time
