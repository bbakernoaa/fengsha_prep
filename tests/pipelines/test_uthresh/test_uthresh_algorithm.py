"""
Tests for the core algorithms of the uthresh pipeline.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fengsha_prep.pipelines.uthresh.algorithm import (
    compute_hybrid_drag_partition,
    compute_moisture_inhibition,
    predict_threshold_velocity,
    prepare_balanced_training,
)

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
def mock_brdf_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock BRDF dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["BRDF_Albedo_Parameter_Isotropic_Band1"] = xr.DataArray(
        np.full((10, 10), 0.30), dims=mock_geo_data["dims"]
    )
    ds["BRDF_Albedo_Parameter_Volumetric_Band1"] = xr.DataArray(
        np.full((10, 10), 0.1), dims=mock_geo_data["dims"]
    )
    ds["BRDF_Albedo_Parameter_Geometric_Band1"] = xr.DataArray(
        np.full((10, 10), 0.05), dims=mock_geo_data["dims"]
    )
    return ds


@pytest.fixture
def mock_nbar_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock NBAR dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["Nadir_Reflectance_Band1"] = xr.DataArray(
        np.full((10, 10), 0.15), dims=mock_geo_data["dims"]
    )
    return ds


@pytest.fixture
def mock_lai_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock LAI dataset."""
    ds = xr.Dataset(coords=mock_geo_data["coords"])
    ds["Lai"] = xr.DataArray(np.full((10, 10), 1.5), dims=mock_geo_data["dims"])
    return ds


@pytest.fixture
def mock_training_dataframe() -> pd.DataFrame:
    """Creates a mock DataFrame for training."""
    data = {
        "clay": np.random.uniform(5, 40, 1000),
        "igbp": np.random.randint(1, 14, 1000),
    }
    return pd.DataFrame(data)


# --- Unit Tests for Algorithms ---


def test_compute_hybrid_drag_partition(mock_brdf_data, mock_lai_data):
    """Tests the drag partition physics calculation."""
    igbp_class = 7  # Open Shrublands
    result = compute_hybrid_drag_partition(
        mock_brdf_data, mock_lai_data, igbp_class, ds_nbar=None
    )

    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0 and result.mean() < 0.1


def test_compute_moisture_inhibition():
    """Tests the moisture inhibition physics calculation."""
    h_w_dry = compute_moisture_inhibition(moisture=0.1, clay=20, soc=5)
    assert h_w_dry == 1.0

    h_w_wet = compute_moisture_inhibition(moisture=7.0, clay=30, soc=10)
    assert h_w_wet > 1.0


def test_prepare_balanced_training(mock_training_dataframe):
    """Tests the data stratification and sampling logic."""
    df_in = mock_training_dataframe
    df_out = prepare_balanced_training(df_in)

    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) <= len(df_in)
    assert "texture" in df_out.columns


def test_predict_threshold_velocity(mock_geo_data):
    """Tests the ML prediction wrapper."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.full(100, 0.35)

    ds_soil = xr.Dataset(
        {
            "clay": xr.DataArray(np.full((10, 10), 25.0), dims=mock_geo_data["dims"]),
            "soc": xr.DataArray(np.full((10, 10), 5.0), dims=mock_geo_data["dims"]),
            "bdod": xr.DataArray(np.full((10, 10), 1.4), dims=mock_geo_data["dims"]),
        },
        coords=mock_geo_data["coords"],
    )
    R = xr.DataArray(np.full((10, 10), 0.05), dims=mock_geo_data["dims"])
    H = xr.DataArray(np.full((10, 10), 1.2), dims=mock_geo_data["dims"])
    lai = xr.DataArray(
        np.full((10, 10), 1.5),
        dims=mock_geo_data["dims"],
        coords=mock_geo_data["coords"],
    )

    result = predict_threshold_velocity(mock_model, ds_soil, R, H, lai)

    mock_model.predict.assert_called_once()
    assert isinstance(result, xr.DataArray)
    assert result.shape == (10, 10)
    assert (result.values == 0.35).all()
