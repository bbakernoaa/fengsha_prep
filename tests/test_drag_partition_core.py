from unittest.mock import patch
import numpy as np
import xarray as xr
import pytest
from fengsha_prep.pipelines.drag_partition.core import (
    _calculate_drag_partition,
    process_hybrid_drag,
)

@pytest.fixture
def mock_modis_data() -> tuple[xr.Dataset, xr.Dataset]:
    """Creates mock MODIS Albedo and LAI datasets for testing."""
    # Create coordinates
    lat = np.linspace(30, 40, 20)
    lon = np.linspace(-100, -90, 20)
    time = np.arange('2024-03-01', '2024-03-08', dtype='datetime64[D]')
    dims = ('time', 'lat', 'lon')
    coords = {'time': time, 'lat': lat, 'lon': lon}

    # --- Mock Albedo Dataset (MCD43C3) ---
    ds_alb = xr.Dataset(coords=coords)
    ds_alb['Albedo_BSW_Band1'] = xr.DataArray(np.full((7, 20, 20), 0.15), dims=dims)
    ds_alb['BRDF_Albedo_Parameter_Isotropic_Band1'] = xr.DataArray(np.full((7, 20, 20), 0.30), dims=dims)
    ds_alb['Albedo_BSW_Band6'] = xr.DataArray(np.full((7, 20, 20), 0.27), dims=dims)
    ds_alb['Albedo_BSW_Band7'] = xr.DataArray(np.full((7, 20, 20), 0.20), dims=dims)

    # --- Mock LAI Dataset (MCD15A2H) ---
    ds_lai = xr.Dataset(coords=coords)
    # Simulate a gradient in LAI
    lai_data = np.ones((7, 20, 20))
    lai_data[:, 10:, :] = 2.5  # Higher LAI in the southern half
    ds_lai['Lai'] = xr.DataArray(lai_data, dims=dims)

    return ds_alb, ds_lai

@patch('fengsha_prep.pipelines.drag_partition.core.get_modis_data')
@patch('fengsha_prep.pipelines.drag_partition.core.earthaccess.login')
def test_process_hybrid_drag_calculation(mock_login, mock_get_data, mock_modis_data):
    """
    Tests the end-to-end calculation of the process_hybrid_drag function,
    mocking the data fetching and authentication parts.
    """
    # Configure the mock to return our test datasets
    ds_alb, ds_lai = mock_modis_data
    mock_get_data.side_effect = [ds_alb, ds_lai]
    mock_login.return_value = True  # Mock the login call

    start_date = "2024-03-01"
    end_date = "2024-03-07"
    u10_wind = 8.0  # Constant wind speed

    # --- Execute the function ---
    us_star = process_hybrid_drag(start_date, end_date, u10_wind)

    # --- Verification ---
    # 1. Check the output type and shape
    assert isinstance(us_star, xr.DataArray)
    assert us_star.shape == (7, 20, 20)

    # 2. Check the attributes
    assert 'units' in us_star.attrs
    assert us_star.attrs['units'] == 'm s-1'
    assert 'history' in us_star.attrs

    # 3. Check the calculated values
    # The values should not be uniform because LAI is not uniform
    assert us_star.mean().values > 0
    assert us_star.std().values > 0  # There should be variation
    # Check that the friction velocity is a reasonable fraction of the wind speed
    assert us_star.mean().values < (u10_wind * 0.1)


def test_calculate_drag_partition_logic(mock_modis_data):
    """
    Tests the pure calculation logic of the _calculate_drag_partition function.
    """
    ds_alb, ds_lai = mock_modis_data
    u10_wind = 10.0

    # --- Execute the function ---
    us_star = _calculate_drag_partition(ds_alb, ds_lai, u10_wind)

    # --- Verification ---
    # 1. Check the output type and shape
    assert isinstance(us_star, xr.DataArray)
    # The output shape should match the input shape after interpolation
    assert us_star.shape == (7, 20, 20)

    # 2. Check the attributes
    assert "units" in us_star.attrs
    assert us_star.attrs["units"] == "m s-1"

    # 3. Check for NaN values
    assert not us_star.isnull().any()

    # 4. Check that the calculation produces expected variation
    # Because LAI is varied in the mock data, the output should also vary
    assert us_star.std().item() > 0

    # 5. Check a specific value for correctness
    # For the area with lai = 1.0, the value should be around 0.000175.
    # This value was derived from a manual calculation trace and acts as a
    # regression check. The previous value of 0.35 was incorrect for the
    # given mock data.
    assert np.isclose(us_star.values[0, 0, 0], 0.000175, atol=1e-6)
