"""
Unit tests for the uthresh.py module.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch, MagicMock

# Import the functions and classes to be tested from the package
from fengsha_prep.pipelines.uthresh.core import (
    DustDataEngine,
    compute_hybrid_drag_partition,
    compute_moisture_inhibition,
    prepare_balanced_training,
    train_piml_model,
    generate_dust_flux_map,
)

# --- Test Fixtures for Mock Data ---

@pytest.fixture
def mock_geo_data() -> dict:
    """Provides standard coordinates for mock xarray datasets."""
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)
    return {
        'lat': lat,
        'lon': lon,
        'dims': ('lat', 'lon'),
        'coords': {'lat': lat, 'lon': lon}
    }

@pytest.fixture
def mock_albedo_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock albedo dataset."""
    ds = xr.Dataset(coords=mock_geo_data['coords'])
    ds['Albedo_BSW_Band1'] = xr.DataArray(np.full((10, 10), 0.15), dims=mock_geo_data['dims'])
    ds['BRDF_Albedo_Parameter_Isotropic_Band1'] = xr.DataArray(np.full((10, 10), 0.30), dims=mock_geo_data['dims'])
    return ds

@pytest.fixture
def mock_lai_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock LAI dataset."""
    ds = xr.Dataset(coords=mock_geo_data['coords'])
    ds['Lai'] = xr.DataArray(np.full((10, 10), 1.5), dims=mock_geo_data['dims'])
    return ds

@pytest.fixture
def mock_lc_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock land cover dataset."""
    ds = xr.Dataset(coords=mock_geo_data['coords'])
    # IGBP class 7: Open Shrublands
    ds['LC_Type1'] = xr.DataArray(np.full((10, 10), 7, dtype=int), dims=mock_geo_data['dims'])
    return ds

@pytest.fixture
def mock_soil_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock soil properties dataset."""
    ds = xr.Dataset(coords=mock_geo_data['coords'])
    ds['clay'] = xr.DataArray(np.full((10, 10), 25.0), dims=mock_geo_data['dims'])
    ds['soc'] = xr.DataArray(np.full((10, 10), 5.0), dims=mock_geo_data['dims'])
    ds['bdod'] = xr.DataArray(np.full((10, 10), 1.4), dims=mock_geo_data['dims']) # Added bdod
    return ds

@pytest.fixture
def mock_met_data(mock_geo_data: dict) -> xr.Dataset:
    """Creates a mock meteorology dataset."""
    ds = xr.Dataset(coords=mock_geo_data['coords'])
    ds['soilw'] = xr.DataArray(np.full((10, 10), 0.2), dims=mock_geo_data['dims'])
    ds['ustar'] = xr.DataArray(np.full((10, 10), 10.0), dims=mock_geo_data['dims'])
    return ds

@pytest.fixture
def mock_training_dataframe() -> pd.DataFrame:
    """Creates a mock DataFrame for training."""
    data = {
        'clay': np.random.uniform(5, 40, 1000),
        'soc': np.random.uniform(1, 20, 1000),
        'bdod': np.random.uniform(1.2, 1.6, 1000),
        'R_partition': np.random.uniform(0.01, 0.05, 1000),
        'h_w_inhibition': np.random.uniform(1.0, 1.5, 1000),
        'lai': np.random.uniform(0.1, 4.0, 1000),
        'u_eff_target': np.random.uniform(0.2, 0.5, 1000),
        'igbp': np.random.randint(1, 14, 1000),
    }
    return pd.DataFrame(data)

# --- Unit Tests ---

def test_compute_hybrid_drag_partition(mock_albedo_data, mock_lai_data):
    """Tests the drag partition physics calculation."""
    igbp_class = 7 # Open Shrublands
    result = compute_hybrid_drag_partition(mock_albedo_data, mock_lai_data, igbp_class)

    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0 and result.mean() < 0.1

def test_compute_moisture_inhibition():
    """Tests the moisture inhibition physics calculation."""
    h_w_dry = compute_moisture_inhibition(moisture=0.1, clay=20, soc=5)
    assert h_w_dry == 1.0

    h_w_wet = compute_moisture_inhibition(moisture=7.0, clay=30, soc=10)
    assert h_w_wet > 1.0

    moisture = np.array([0.1, 7.0])
    clay = np.array([20, 30])
    soc = np.array([5, 10])
    results = compute_moisture_inhibition(moisture, clay, soc)
    assert isinstance(results, np.ndarray)
    assert results[0] == 1.0
    assert results[1] > 1.0

def test_prepare_balanced_training(mock_training_dataframe):
    """Tests the data stratification and sampling logic."""
    df_in = mock_training_dataframe
    df_out = prepare_balanced_training(df_in)

    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) <= len(df_in)
    assert 'texture' in df_out.columns

@patch('fengsha_prep.pipelines.uthresh.core.XGBRegressor')
def test_train_piml_model(mock_xgb, mock_training_dataframe):
    """Tests the ML model training orchestration."""
    mock_model_instance = MagicMock()
    mock_xgb.return_value = mock_model_instance

    model = train_piml_model(mock_training_dataframe)

    mock_xgb.assert_called_once()
    mock_model_instance.fit.assert_called_once()
    assert model is mock_model_instance

def test_generate_dust_flux_map(mock_albedo_data, mock_lai_data, mock_lc_data, mock_soil_data, mock_met_data, mock_geo_data):
    """Tests the end-to-end flux calculation with a mock model."""
    mock_model = MagicMock()
    # Mock the predict method to return a flat array of threshold values
    mock_model.predict.return_value = np.full(100, 0.35)

    mock_r = xr.DataArray(np.full((10, 10), 0.05), dims=mock_geo_data['dims'], coords=mock_geo_data['coords'])
    mock_h = xr.DataArray(np.full((10, 10), 1.2), dims=mock_geo_data['dims'], coords=mock_geo_data['coords'])

    with patch('fengsha_prep.pipelines.uthresh.core.compute_hybrid_drag_partition', return_value=mock_r):
        with patch('fengsha_prep.pipelines.uthresh.core.compute_moisture_inhibition', return_value=mock_h):
            result = generate_dust_flux_map(
                mock_albedo_data, mock_lai_data, mock_lc_data, mock_soil_data, mock_met_data, mock_model
            )

    # Assert that the model's predict method was called
    mock_model.predict.assert_called_once()

    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.shape == (10, 10)
    assert result.mean() > 0

@patch('fengsha_prep.pipelines.uthresh.core.earthaccess.login')
@patch('fengsha_prep.pipelines.uthresh.core.s3fs.S3FileSystem')
@patch('fengsha_prep.pipelines.uthresh.core.WebCoverageService')
def test_dust_data_engine_mocks(mock_wcs, mock_s3fs, mock_earthaccess):
    """Tests that the DustDataEngine initializes and calls its clients."""
    engine = DustDataEngine()

    mock_earthaccess.assert_called_once()
    mock_s3fs.assert_called_once_with(anon=True)

    mock_wcs_instance = MagicMock()
    mock_wcs.return_value = mock_wcs_instance
    mock_wcs_instance.getCoverage.return_value.read.return_value = b''

    with patch('fengsha_prep.pipelines.uthresh.core.rasterio.open', MagicMock()):
        engine.fetch_soilgrids(lat=35.0, lon=-95.0)

    mock_wcs.assert_called_once()
    assert mock_wcs_instance.getCoverage.call_count > 0
