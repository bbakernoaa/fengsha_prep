"""Regression tests for the Prigent et al. static drag-partition blend.

Bug being guarded against: the published Prigent file stores its field as
``PRIGENT_RDRAG``, while the loader only renamed ``PRIGENT_DRAG`` and the
algorithm only looked for ``drag_partition``/``drag`` (case-sensitive). The
blend was therefore silently skipped and bright, smooth desert pixels
(f_geo == 0) fell back to the pure dynamic Raupach value, which is exactly
1.0 -- i.e. nulls were "filled with 1" instead of being blended with Prigent.
"""

import numpy as np
import pytest
import xarray as xr

from fengsha_prep.pipelines.drag_partition.algorithm import calculate_drag_partition
from fengsha_prep.pipelines.drag_partition.io import load_prigent_drag_partition


def _smooth_desert_brdf():
    """Bright, smooth surface: f_iso well retrieved but f_geo == 0 everywhere.

    lat/lon are required for the Prigent alignment path to be exercised.
    """
    return xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.zeros((2, 2), dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )


def test_prigent_blend_replaces_smooth_desert_fallback():
    """f_geo == 0 must take the Prigent value, not the Raupach ceiling of 1.0."""
    ds_brdf = _smooth_desert_brdf()
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)

    ra_bare = res["ra_bare"].values
    # Sanity check on the premise: the dynamic term alone saturates at 1.0 here.
    assert not np.any(np.isclose(ra_bare, 1.0)), (
        "Prigent blend was skipped; smooth desert pixels defaulted to 1.0"
    )
    # Taklamakan tuning only applies at 36-42N/75-90E, so 0.35 passes through.
    np.testing.assert_allclose(ra_bare, 0.35, rtol=1e-5)
    np.testing.assert_allclose(res["feff"].values, 0.35, rtol=1e-5)


def test_prigent_blend_uses_dynamic_term_where_geo_is_retrieved():
    """Where f_geo is confidently retrieved the dynamic Raupach result wins."""
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.full((2, 2), 0.05, dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)

    # f_geo=0.05 is well above the 0.002 hybrid transition, so w_hybrid == 1.
    without = calculate_drag_partition(ds_brdf, use_lai=False)
    np.testing.assert_allclose(res["ra_bare"].values, without["ra_bare"].values, rtol=1e-5)


def test_prigent_invalid_negative_values_are_masked():
    """Prigent's negative sentinel/invalid values must not leak into feff."""
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.zeros((2, 2), dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), -1.2, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)

    feff = res["feff"].values
    finite = feff[np.isfinite(feff)]
    assert finite.size > 0
    assert (finite >= 0.0).all(), f"Negative Prigent values leaked into feff: {finite.min()}"
    assert (finite <= 1.0).all()


def test_load_prigent_drag_partition_renames_published_variable(tmp_path):
    """The loader must normalise the published PRIGENT_RDRAG variable name."""
    src = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [-10.0, 10.0], "lon": [0.0, 1.0]},
    )  # ascending lat on purpose: the loader should flip it to descending.
    path = tmp_path / "prigent.nc"
    src.to_netcdf(path)

    ds = load_prigent_drag_partition(path)

    assert "drag_partition" in ds.data_vars
    assert ds.lat.values[0] > ds.lat.values[-1], "latitude should be descending"


@pytest.mark.parametrize("name", ["PRIGENT_RDRAG", "PRIGENT_DRAG", "Drag_Partition"])
def test_algorithm_finds_prigent_regardless_of_variable_casing(name):
    """calculate_drag_partition must locate the static partition field robustly."""
    ds_brdf = _smooth_desert_brdf()
    ds_prigent = xr.Dataset(
        {name: (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)

    np.testing.assert_allclose(res["ra_bare"].values, 0.35, rtol=1e-5)


def test_no_data_pixels_are_floored_not_nan_or_one():
    """Pixels with no dynamic retrieval and no valid Prigent must floor to ~1e-4.

    The old behaviour left these as NaN (or saturated the dynamic Raupach term
    to exactly 1.0 over the ocean). The model should instead see a near-zero
    drag partition that effectively shuts dust emission off.
    """
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), np.nan, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.full((2, 2), np.nan, dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    # All-invalid Prigent (ocean/ice sentinel) so the static field cannot fill.
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), -1.0, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)

    feff = res["feff"].values
    assert not np.any(np.isnan(feff)), "feff must never contain NaN"
    assert not np.any(feff == 1.0), "no-data pixels must not saturate to 1.0"
    np.testing.assert_allclose(feff, 1e-4, rtol=1e-6)


def test_masked_drag_floor_is_configurable():
    """The floor value can be overridden by the caller."""
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.zeros((2, 2), dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), -1.0, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    res = calculate_drag_partition(
        ds_brdf, ds_prigent=ds_prigent, use_lai=False, masked_drag_floor=5e-4
    )

    np.testing.assert_allclose(res["feff"].values, 5e-4, rtol=1e-6)


def test_near_zero_f_geo_leans_on_prigent_with_wider_blend():
    """A small-but-nonzero f_geo (below the blend width) must be Prigent-dominated.

    Previously the blend ramped over 0 -> 0.002, so f_geo = 0.003 was treated as
    fully dynamic. The default width is now 0.005, so f_geo = 0.003 should sit at
    w_hybrid = 0.6 (i.e. 40% weight on the static Prigent climatology).
    """
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.full((2, 2), 0.003, dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    blended = calculate_drag_partition(ds_brdf, ds_prigent=ds_prigent, use_lai=False)
    dynamic_only = calculate_drag_partition(ds_brdf, use_lai=False)

    # With the wider default, the result is pulled toward Prigent (0.35) and away
    # from the pure dynamic retrieval.
    assert float(blended["ra_bare"].mean()) < float(dynamic_only["ra_bare"].mean())
    # w_hybrid = 0.003 / 0.005 = 0.6 -> ra_bare = 0.6*dynamic + 0.4*0.35
    expected = 0.6 * dynamic_only["ra_bare"].values + 0.4 * 0.35
    np.testing.assert_allclose(blended["ra_bare"].values, expected, rtol=1e-5)


def test_blend_width_is_configurable():
    """A narrow blend width restores the old fully-dynamic behaviour above it."""
    ds_brdf = xr.Dataset(
        {
            "BRDF_Albedo_Parameter1_Band1": (("lat", "lon"), np.full((2, 2), 0.40, dtype="float32")),
            "BRDF_Albedo_Parameter3_Band1": (("lat", "lon"), np.full((2, 2), 0.003, dtype="float32")),
        },
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )
    ds_prigent = xr.Dataset(
        {"PRIGENT_RDRAG": (("lat", "lon"), np.full((2, 2), 0.35, dtype="float32"))},
        coords={"lat": [20.0, 10.0], "lon": [0.0, 1.0]},
    )

    narrow = calculate_drag_partition(
        ds_brdf, ds_prigent=ds_prigent, use_lai=False, f_geo_blend_width=0.002
    )
    dynamic_only = calculate_drag_partition(ds_brdf, use_lai=False)

    # f_geo=0.003 >= width 0.002 -> w_hybrid=1 -> pure dynamic, Prigent unused.
    np.testing.assert_allclose(narrow["ra_bare"].values, dynamic_only["ra_bare"].values, rtol=1e-5)
