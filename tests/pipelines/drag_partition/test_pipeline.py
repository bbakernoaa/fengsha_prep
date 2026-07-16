from unittest.mock import MagicMock, ANY

import pytest
import xarray as xr

from fengsha_prep.pipelines.drag_partition.pipeline import run_drag_partition_pipeline


@pytest.fixture
def mock_data_fetcher():
    """Pytest fixture to create a mock data fetcher."""
    mock_fetcher = MagicMock()
    dummy_ds = xr.Dataset(
        {
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), [[0.2]]),
            "BRDF_Albedo_Parameter_Geometric_Band2": (("y", "x"), [[0.05]]),
            "BRDF_Albedo_Parameter_Isotropic_Band6": (("y", "x"), [[0.3]]),
            "BRDF_Albedo_Parameter_Isotropic_Band7": (("y", "x"), [[0.1]]),
            "Nadir_Reflectance_Band1": (("y", "x"), [[0.15]]),
            "Lai": (("y", "x"), [[1.5]]),
        }
    )
    mock_fetcher.return_value = dummy_ds
    return mock_fetcher


def test_run_drag_partition_pipeline_integration_modis(mock_data_fetcher):
    """Integration test for the drag partition pipeline with MODIS sensor."""
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    run_drag_partition_pipeline(
        start_date, end_date, sensor="MODIS", data_fetcher=mock_data_fetcher
    )

    # Assert that the data_fetcher was called 5 times (brdf, albedo, nbar, lai, gvf)
    assert mock_data_fetcher.call_count == 5
    mock_data_fetcher.assert_any_call("brdf", start_date, end_date, "MODIS", False, cache_dir=None)
    mock_data_fetcher.assert_any_call("albedo", start_date, end_date, "MODIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("nbar", start_date, end_date, "MODIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("lai", start_date, end_date, "MODIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("gvf", start_date, end_date, "MODIS", True, cache_dir=None)


def test_run_drag_partition_pipeline_integration_vnp(mock_data_fetcher):
    """Integration test for the drag partition pipeline with VNP sensor."""
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    run_drag_partition_pipeline(
        start_date, end_date, sensor="VNP", data_fetcher=mock_data_fetcher
    )

    # Assert that the data_fetcher was called 3 times (brdf, lai, ndvi)
    assert mock_data_fetcher.call_count == 3
    mock_data_fetcher.assert_any_call("brdf", start_date, end_date, "VNP", False, cache_dir=None)
    mock_data_fetcher.assert_any_call("lai", start_date, end_date, "VNP", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("ndvi", start_date, end_date, "VNP", True, cache_dir=None)


def test_run_drag_partition_pipeline_integration_nesdis(mock_data_fetcher):
    """Integration test for the drag partition pipeline with NESDIS sensor."""
    start_date = "2024-01-01"
    end_date = "2024-01-07"

    # Define a side_effect to return different datasets for different products
    def side_effect(product, start_date_val, end_date_val, sensor_val, is_scaled, cache_dir=None):
        if product == "brdf":
            return xr.Dataset({"Albedo_BSA_M5": (("y", "x"), [[0.15]])})
        elif product == "nbar":
            return xr.Dataset({"Nadir_Reflectance_M5": (("y", "x"), [[0.15]])})
        elif product == "lai":
            return xr.Dataset({"LAI": (("y", "x"), [[0.5]])})
        elif product == "gvf":
            return xr.Dataset({"gvf_4km": (("y", "x"), [[0.2]])})
        return xr.Dataset()

    mock_data_fetcher.side_effect = side_effect

    run_drag_partition_pipeline(
        start_date, end_date, sensor="NESDIS", data_fetcher=mock_data_fetcher
    )

    # Assert that the data_fetcher was called five times (brdf, albedo, nbar, lai, gvf)
    assert mock_data_fetcher.call_count == 5
    mock_data_fetcher.assert_any_call("brdf", start_date, end_date, "NESDIS", False, cache_dir=None)
    mock_data_fetcher.assert_any_call("albedo", start_date, end_date, "NESDIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("nbar", start_date, end_date, "NESDIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("lai", start_date, end_date, "NESDIS", True, cache_dir=None)
    mock_data_fetcher.assert_any_call("gvf", start_date, end_date, "NESDIS", True, cache_dir=None)
