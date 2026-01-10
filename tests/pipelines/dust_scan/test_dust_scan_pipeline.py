import datetime
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from satpy import Scene

from fengsha_prep.pipelines.dust_scan.algorithm import DEFAULT_THRESHOLDS
from fengsha_prep.pipelines.dust_scan.pipeline import (
    _process_scene_sync,
    dust_scan_pipeline,
)


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.pipeline._process_scene_sync")
@patch(
    "fengsha_prep.pipelines.dust_scan.pipeline.load_scene_data", new_callable=AsyncMock
)
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
@patch(
    "fengsha_prep.pipelines.dust_scan.pipeline.load_scene_data", new_callable=AsyncMock
)
async def test_dust_scan_pipeline_load_fails(mock_load_scene):
    """
    Tests the async pipeline when scene loading returns None.
    """
    mock_load_scene.return_value = None
    events = await dust_scan_pipeline(datetime.datetime.now(), "goes16", {})
    assert events == []


@pytest.mark.asyncio
@patch("fengsha_prep.pipelines.dust_scan.pipeline._process_scene_sync")
@patch(
    "fengsha_prep.pipelines.dust_scan.pipeline.load_scene_data", new_callable=AsyncMock
)
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
@patch("fengsha_prep.pipelines.dust_scan.pipeline.logger")
@patch("fengsha_prep.pipelines.dust_scan.pipeline._process_scene_sync")
@patch("fengsha_prep.pipelines.dust_scan.pipeline.load_scene_data")
async def test_dust_scan_pipeline_handles_processing_value_error(
    mock_load_scene, mock_process_sync, mock_logger
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
    mock_logger.exception.assert_called_once()
    log_message = mock_logger.exception.call_args[0][0]
    assert f"Failed to process scene for {scn_time}" in log_message
