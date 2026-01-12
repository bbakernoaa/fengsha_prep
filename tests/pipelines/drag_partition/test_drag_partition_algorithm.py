import numpy as np
import xarray as xr

from fengsha_prep.pipelines.drag_partition.algorithm import calculate_drag_partition


def test_calculate_drag_partition_pure_logic():
    """Unit test for the drag partition algorithm.

    This test isolates the scientific calculation from any I/O operations by
    providing mock xarray Datasets. It verifies that the core algorithm
    produces the expected output based on a controlled set of inputs.
    """
    # Create mock albedo and LAI datasets. These values are chosen to avoid
    # edge cases where the vegetation cover fraction (sigma_g + sigma_b)
    # sums to 1.0 or more, which would result in a zero drag partition.
    ds_alb = xr.Dataset(
        {
            "Albedo_BSW_Band1": (("y", "x"), np.full((2, 2), 0.15)),
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), np.full((2, 2), 0.2)),
            "Albedo_BSW_Band6": (
                ("y", "x"),
                np.full((2, 2), 0.2),
            ),  # Produces a moderate NDTI
            "Albedo_BSW_Band7": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_lai = xr.Dataset(
        {"Lai": (("y", "x"), np.full((2, 2), 0.5))},  # Produces a moderate sigma_g
        coords={"y": [1, 2], "x": [1, 2]},
    )
    u10_wind = 10.0

    # Execute the algorithm
    result = calculate_drag_partition(ds_alb, ds_lai, u10_wind)

    # Assertions
    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0  # Check for a physically plausible result

    # Check that the history attribute is correctly set
    assert "history" in result.attrs
    assert "Calculated at" in result.attrs["history"]
