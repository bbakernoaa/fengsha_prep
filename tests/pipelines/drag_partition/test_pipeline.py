from unittest.mock import MagicMock

import pytest
import xarray as xr

from fengsha_prep.pipelines.drag_partition.pipeline import run_drag_partition_pipeline


@pytest.fixture
def mock_data_fetcher():
    """Pytest fixture to create a mock data fetcher."""
    mock_fetcher = MagicMock()
    dummy_ds = xr.Dataset(
        {
            "Albedo_BSW_Band1": (("y", "x"), [[0.15]]),
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), [[0.2]]),
            "Albedo_BSW_Band6": (("y", "x"), [[0.3]]),
            "Albedo_BSW_Band7": (("y", "x"), [[0.1]]),
            "Lai": (("y", "x"), [[1.5]]),
        }
    )
    mock_fetcher.return_value = dummy_ds
    return mock_fetcher


def test_run_drag_partition_pipeline_integration_modis(mock_data_fetcher):
    """Integration test for the drag partition pipeline with MODIS sensor.

    This test verifies the end-to-end orchestration of the pipeline for the
    default MODIS sensor. It uses a mock `data_fetcher` to ensure the correct
    product types are requested.
    """
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    run_drag_partition_pipeline(
        start_date, end_date, sensor="MODIS", data_fetcher=mock_data_fetcher
    )

    # Assert that the data_fetcher was called twice
    assert mock_data_fetcher.call_count == 2
    mock_data_fetcher.assert_any_call("albedo", start_date, end_date, "MODIS")
    mock_data_fetcher.assert_any_call("lai", start_date, end_date, "MODIS")


def test_run_drag_partition_pipeline_integration_viirs(mock_data_fetcher):
    """Integration test for the drag partition pipeline with VIIRS sensor.

    This test verifies that the pipeline correctly requests data for the
    VIIRS sensor when specified.
    """
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    run_drag_partition_pipeline(
        start_date, end_date, sensor="VIIRS", data_fetcher=mock_data_fetcher
    )

    # Assert that the data_fetcher was called twice
    assert mock_data_fetcher.call_count == 2
    mock_data_fetcher.assert_any_call("albedo", start_date, end_date, "VIIRS")
    mock_data_fetcher.assert_any_call("lai", start_date, end_date, "VIIRS")
