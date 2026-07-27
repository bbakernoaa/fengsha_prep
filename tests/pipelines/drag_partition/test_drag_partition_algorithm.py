import numpy as np
import xarray as xr
from fengsha_prep.pipelines.drag_partition.algorithm import calculate_drag_partition

def test_calculate_drag_partition_pure_logic():
    """Unit test for the color-normalized drag partition algorithm."""
    # Create mock BRDF and LAI datasets.
    # A bright surface (f_iso = 0.40) and dark surface (f_iso = 0.15)
    # with identical geometric shadowing (f_geo = 0.05).
    ds_brdf_bright = xr.Dataset(
        {
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), np.full((2, 2), 0.40)),
            "BRDF_Albedo_Parameter_Geometric_Band2": (("y", "x"), np.full((2, 2), 0.05)),
            "BRDF_Albedo_Parameter_Isotropic_Band7": (("y", "x"), np.full((2, 2), 0.30)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    
    ds_brdf_dark = xr.Dataset(
        {
            "BRDF_Albedo_Parameter_Isotropic_Band1": (("y", "x"), np.full((2, 2), 0.15)),
            "BRDF_Albedo_Parameter_Geometric_Band2": (("y", "x"), np.full((2, 2), 0.05)),
            "BRDF_Albedo_Parameter_Isotropic_Band7": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )

    # Execute the algorithm on both
    res_bright = calculate_drag_partition(ds_brdf_bright, use_lai=False)
    res_dark = calculate_drag_partition(ds_brdf_dark, use_lai=False)

    # Assertions
    assert isinstance(res_bright, xr.Dataset)
    assert isinstance(res_dark, xr.Dataset)
    assert "feff" in res_bright
    assert "feff" in res_dark
    
    # Darker soil will have slightly lower R (due to damped coupling),
    # but it shouldn't blow up or be extremely penalized.
    assert res_bright["feff"].mean() > 0.20
    assert res_dark["feff"].mean() > 0.15
    # Verify that we do not have NaNs from edge padding (which rolling-std used to cause)
    assert not np.isnan(res_bright["feff"]).any()

def test_calculate_drag_partition_nesdis_vars():
    """Unit test for the drag partition algorithm with NESDIS variables."""
    ds_alb = xr.Dataset(
        {
            "I1_TOC": (("y", "x"), np.full((2, 2), 0.15)),
        },
        coords={"y": [1, 2], "x": [1, 2]},
    )
    ds_gvf = xr.Dataset(
        {"gvf_4km": (("y", "x"), np.full((2, 2), 0.2))},
        coords={"y": [1, 2], "x": [1, 2]},
    )

    # Execute the algorithm
    result = calculate_drag_partition(ds_alb, ds_gvf=ds_gvf, use_lai=False)

    # Assertions
    assert isinstance(result, xr.Dataset)
    assert not np.isnan(result["feff"]).any()
    assert result["feff"].mean() > 0
