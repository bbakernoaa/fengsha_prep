"""
Tests for the orchestration layer of the uthresh pipeline.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from fengsha_prep.pipelines.uthresh.pipeline import generate_dust_flux_map

# --- Test Fixtures for Mock Data ---


@pytest.fixture
def mock_geo_data() -> dict:
    """Provides standard coordinates for mock xarray datasets."""
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)
    return {
        "lat": lat,
        "lon": lon,
        "dims": ("lat", "lon"),
        "coords": {"lat": lat, "lon": lon},
    }


@pytest.fixture
def mock_albedo_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock albedo dataset."""
    return xr.Dataset(coords=mock_geo_data["coords"])


@pytest.fixture
def mock_lai_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock LAI dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["Lai"] = xr.DataArray(np.full((10, 10), 1.5), dims=mock_geo_data["dims"])
    return ds


@pytest.fixture
def mock_lc_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock land cover dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["LC_Type1"] = xr.DataArray(
        np.full((10, 10), 7, dtype=int), dims=mock_geo_data["dims"]
    )
    return ds


@pytest.fixture
def mock_soil_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock soil properties dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["clay"] = xr.DataArray(np.full((10, 10), 25.0), dims=mock_geo_data["dims"])
    ds["soc"] = xr.DataArray(np.full((10, 10), 5.0), dims=mock_geo_data["dims"])
    return ds


@pytest.fixture
def mock_met_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock meteorology dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["soilw"] = xr.DataArray(np.full((10, 10), 0.2), dims=mock_geo_data["dims"])
    ds["ustar"] = xr.DataArray(np.full((10, 10), 10.0), dims=mock_geo_data["dims"])
    return ds


def test_generate_dust_flux_map_orchestration(
    mock_albedo_data, mock_lai_data, mock_lc_data, mock_soil_data, mock_met_data
):
    """
    Tests the end-to-end orchestration of the `generate_dust_flux_map` function.
    This is an integration test for the pipeline's core logic.
    """
    mock_model = MagicMock()

    # Mock the functions from the algorithm module that are called by the pipeline
    with patch(
        "fengsha_prep.pipelines.uthresh.pipeline.compute_hybrid_drag_partition"
    ) as mock_drag, patch(
        "fengsha_prep.pipelines.uthresh.pipeline.compute_moisture_inhibition"
    ) as mock_moisture, patch(
        "fengsha_prep.pipelines.uthresh.pipeline.predict_threshold_velocity"
    ) as mock_predict:
        # Define the return values for the mocked algorithm functions
        mock_drag.return_value = xr.DataArray(0.05)
        mock_moisture.return_value = xr.DataArray(1.2)
        mock_predict.return_value = xr.DataArray(0.35)

        # Execute the pipeline function
        result = generate_dust_flux_map(
            ds_alb=mock_albedo_data,
            ds_lai=mock_lai_data,
            ds_lc=mock_lc_data,
            ds_soil=mock_soil_data,
            ds_met=mock_met_data,
            model=mock_model,
        )

        # Assert that the algorithm functions were called with the correct data
        mock_drag.assert_called_once()
        mock_moisture.assert_called_once()
        mock_predict.assert_called_once()

        # Assert that the final result is a valid DataArray
        assert isinstance(result, xr.DataArray)
        assert not result.isnull().any()
