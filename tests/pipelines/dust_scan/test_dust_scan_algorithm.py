import datetime

import numpy as np
import pytest
import xarray as xr
from satpy import Scene

from fengsha_prep.pipelines.dust_scan.algorithm import (
    DEFAULT_THRESHOLDS,
    cluster_events,
    detect_dust,
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
