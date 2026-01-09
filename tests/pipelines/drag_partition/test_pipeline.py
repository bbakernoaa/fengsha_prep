from unittest.mock import MagicMock

import xarray as xr

from fengsha_prep.pipelines.drag_partition.pipeline import run_drag_partition_pipeline


def test_run_drag_partition_pipeline_integration():
    """Integration test for the drag partition pipeline.

    This test verifies the end-to-end orchestration of the pipeline. It uses
    a mock `data_fetcher` to simulate the I/O operations, ensuring that the
    pipeline correctly calls the data fetching logic and passes the results
    to the algorithm.
    """
    # Create a mock data_fetcher function
    mock_fetcher = MagicMock()

    # Configure the mock to return a dummy xarray Dataset
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

    # Execute the pipeline with the mock fetcher
    start_date = "2024-01-01"
    end_date = "2024-01-07"
    u10_wind = 10.0
    run_drag_partition_pipeline(start_date, end_date, u10_wind, data_fetcher=mock_fetcher)

    # Assert that the data_fetcher was called twice (once for Albedo, once for LAI)
    assert mock_fetcher.call_count == 2
    mock_fetcher.assert_any_call("MCD43C3", start_date, end_date)
    mock_fetcher.assert_any_call("MCD15A2H", start_date, end_date)
