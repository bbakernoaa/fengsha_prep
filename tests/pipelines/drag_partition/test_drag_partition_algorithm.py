import numpy as np
import xarray as xr

from fengsha_prep.pipelines.drag_partition.algorithm import calculate_drag_partition


def test_calculate_drag_partition_pure_logic():
    """Unit test for the drag partition algorithm."""
    # Create mock BRDF and LAI datasets.
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), np.full((2, 2), 0.2)),
            "BRDF_Albedo_Parameter_Isotropic_Band6": (("y", "x"), np.full((2, 2), 0.2)),
            "BRDF_Albedo_Parameter_Isotropic_Band7": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_albedo = xr.Dataset(
        {
            "Albedo_BSA_Band1": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_lai = xr.Dataset(
        {"Lai": (("y", "x"), np.full((2, 2), 0.5))},
        coords={"y": [1, 2], "x": [1, 2]},
    )

    # Execute the algorithm
    result = calculate_drag_partition(ds_brdf, ds_lai, ds_albedo=ds_albedo)

    # Assertions
    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0


def test_calculate_drag_partition_with_nbar():
    """Unit test for the drag partition algorithm using NBAR."""
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), np.full((2, 2), 0.2)),
            "BRDF_Albedo_Parameter_Isotropic_Band6": (("y", "x"), np.full((2, 2), 0.2)),
            "BRDF_Albedo_Parameter_Isotropic_Band7": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_nbar = xr.Dataset(
        {
            "Nadir_Reflectance_Band1": (("y", "x"), np.full((2, 2), 0.15)),
            "Nadir_Reflectance_Band6": (("y", "x"), np.full((2, 2), 0.2)),
            "Nadir_Reflectance_Band7": (("y", "x"), np.full((2, 2), 0.1)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_lai = xr.Dataset(
        {"Lai": (("y", "x"), np.full((2, 2), 0.5))},
        coords={"y": [1, 2], "x": [1, 2]},
    )

    # Execute the algorithm
    result = calculate_drag_partition(ds_brdf, ds_lai, ds_nbar=ds_nbar)

    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0

    # Check that attributes are correctly set for feff
    assert result.attrs["long_name"] == "Effective Drag Coefficient"
    assert result.attrs["units"] == "dimensionless"
    assert "history" in result.attrs
    assert "Calculated at" in result.attrs["history"]


def test_calculate_drag_partition_nesdis_vars():
    """Unit test for the drag partition algorithm with NESDIS variables."""
    ds_alb = xr.Dataset(
        {
            "I1_TOC": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_lai = xr.Dataset(
        {"LAI": (("y", "x"), np.full((2, 2), 0.5))},
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_gvf = xr.Dataset(
        {"gvf_4km": (("y", "x"), np.full((2, 2), 0.2))},
        coords={"y": [1, 2], "x": [1, 2]},
    )

    # Execute the algorithm
    result = calculate_drag_partition(ds_alb, ds_lai, ds_gvf=ds_gvf)

    # Assertions
    assert isinstance(result, xr.DataArray)
    assert not np.isnan(result).any()
    assert result.mean() > 0
