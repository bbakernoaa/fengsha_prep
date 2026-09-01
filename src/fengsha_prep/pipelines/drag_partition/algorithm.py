import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr

# Set up logging
logger = logging.getLogger(__name__)


def calculate_drag_partition(
    ds_brdf: xr.Dataset,
    ds_lai: xr.Dataset | None = None,
    ds_albedo: xr.Dataset | None = None,
    ds_nbar: xr.Dataset | None = None,
    ds_gvf: xr.Dataset | None = None,
    ds_ndvi: xr.Dataset | None = None,
    ds_prigent: xr.Dataset | None = None,
    use_lai: bool = True,
    ndvi_threshold: float = 0.35,
    use_gvf_adjustment: bool = False,
    use_ndvi_adjustment: bool = False,
    vegetation_gamma: float = 1.0,
    bare_threshold: float = 0.80,
    masked_drag_floor: float = 1e-4,
    f_geo_blend_width: float = 0.005,
) -> xr.Dataset:
    """Calculates the effective drag (feff) using a hybrid model.

    This is a pure function that encapsulates the scientific logic for
    estimating the effective drag coefficient.

    Parameters
    ----------
    ds_brdf : xr.Dataset
        The BRDF Parameters dataset (e.g., MCD43C1, VJ143C1).
    ds_lai : xr.Dataset, optional
        The LAI dataset (e.g., MCD15A2H, VNP15A2H). Defaults to None.
    ds_albedo : xr.Dataset, optional
        The Albedo dataset (e.g., MCD43C3, VJ143C3). Defaults to None.
    ds_nbar : xr.Dataset, optional
        The NBAR dataset (e.g., MCD43C4, VJ143C4). Defaults to None.
    ds_gvf : xr.Dataset, optional
        The GVF dataset (e.g., GVF-WKL-GLB). Defaults to None.
    ds_ndvi : xr.Dataset, optional
        The NDVI/EVI dataset (e.g., VNP13C1, VJ113C1). Defaults to None.
    ds_prigent : xr.Dataset, optional
        The Prigent et al. Drag Partition dataset. Defaults to None.
    use_lai : bool, optional
        Whether to include the LAI (green vegetation) component in the
        calculation. Defaults to True.
    ndvi_threshold : float, optional
        Threshold for NDVI masking when GVF is not available.
        Defaults to 0.15.
    use_gvf_adjustment : bool, optional
        Whether to apply GVF-based vegetation attenuation when GVF is
        available. Defaults to False because BRDF parameters already encode
        substantial vegetation structural effects.
    use_ndvi_adjustment : bool, optional
        Whether to apply NDVI/EVI-based vegetation attenuation when GVF is
        unavailable. Defaults to False because BRDF parameters already encode
        substantial vegetation structural effects.
    vegetation_gamma : float, optional
        Non-linear exponent applied to lower bare-surface drag values
        (below ``bare_threshold``) to protect transition-zone detail.
        Defaults to 1.0 (no reshaping).
    bare_threshold : float, optional
        Upper threshold used by the non-linear washout protection.
        Values at or above this are left unchanged. Defaults to 0.80.
    masked_drag_floor : float, optional
        Drag partition value assigned to ocean, snow, and no-data pixels.
        A near-zero value effectively shuts dust emission off in the model
        while avoiding NaNs in the output. Defaults to 1e-4.
    f_geo_blend_width : float, optional
        Geometric-scattering parameter value at which the dynamic Raupach
        retrieval fully replaces the static Prigent climatology. Below this
        the geometric term is considered essentially undetected (smooth or
        bright surfaces) and the static field dominates. Defaults to 0.005.

    Returns
    -------
    xr.Dataset
        A Dataset containing:
        - feff: Total effective drag coefficient.
        - ra_bare: Bare surface drag component.
        - f_veg: Vegetation attenuation factor.
    """
    logger.info("Starting drag partition calculation (use_lai=%s)...", use_lai)

    # Align LAI (and others) to BRDF grid (0.05 degree CMG)
    logger.debug("Aligning datasets to BRDF grid...")

    def align_dataset(ds_target, ds_source, name):
        if ds_source is None:
            return None

        # Drop time if it exists to allow alignment by lat/lon only for the current processing day
        if "time" in ds_source.dims:
            logger.debug(f"Dropping time dimension from {name} for alignment.")
            if ds_source.sizes["time"] > 1:
                ds_source = ds_source.isel(time=0, drop=True)
            else:
                ds_source = ds_source.squeeze("time", drop=True)

        # Check for coordinate intersection. If one has lat/lon and other has x/y (mocks), skip alignment.
        source_coords = set(ds_source.coords) | set(ds_source.dims)
        target_coords = set(ds_target.coords) | set(ds_target.dims)

        # If no common spatial coordinates, return as is (useful for tests with x/y mocks)
        if not (source_coords & target_coords & {"lat", "lon", "x", "y"}):
            logger.warning(f"No common spatial coordinates between {name} and target. Skipping orientation check.")
            return ds_source

        # Check if coordinates already match (ignoring precision)
        if "lat" in ds_source.coords and "lon" in ds_source.coords and \
           "lat" in ds_target.coords and "lon" in ds_target.coords:
            if ds_source.lat.size == ds_target.lat.size and ds_source.lon.size == ds_target.lon.size:
                # Only use fast assign_coords if the orientation and values already match
                if np.allclose(ds_source.lat.values[[0, -1]], ds_target.lat.values[[0, -1]], atol=0.1):
                    logger.debug(f"Dataset {name} already matches grid size and orientation. Aligning coordinates.")
                    # Force exact coordinate match to avoid floating point issues later
                    ds_source = ds_source.assign_coords(lat=ds_target.lat, lon=ds_target.lon)
                    return ds_source

        logger.info(f"Interpolating {name} to match BRDF grid...")
        try:
            return ds_source.interp_like(ds_target, method="nearest")
        except Exception as e:
            logger.warning(f"Failed to interpolate {name}: {e}. Returning original.")
            return ds_source

    # Capture time coordinate for the output before squeezing
    time_coords = None
    if "time" in ds_brdf.coords:
        if ds_brdf.sizes.get("time", 1) > 1:
            time_coords = ds_brdf.time.isel(time=[0])
        else:
            time_coords = ds_brdf.time

    # Ensure ds_brdf itself is squeezed of time if it has it
    if "time" in ds_brdf.dims:
        if ds_brdf.sizes["time"] > 1:
            ds_brdf_calc = ds_brdf.isel(time=0, drop=True)
        else:
            ds_brdf_calc = ds_brdf.squeeze("time", drop=True)
    else:
        ds_brdf_calc = ds_brdf

    align_jobs = {
        "LAI": ds_lai,
        "GVF": ds_gvf,
        "NDVI": ds_ndvi,
        "NBAR": ds_nbar,
        "Albedo": ds_albedo,
        "Prigent": ds_prigent,
    }
    aligned: dict = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(align_dataset, ds_brdf_calc, src, name): name
            for name, src in align_jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            aligned[name] = future.result()
    ds_lai    = aligned["LAI"]
    ds_gvf    = aligned["GVF"]
    ds_ndvi   = aligned["NDVI"]
    ds_nbar   = aligned["NBAR"]
    ds_albedo = aligned["Albedo"]
    ds_prigent = aligned["Prigent"]

    # --- DRAG PARTITION CALCULATION ---
    # A. Bare Surface (Chappell & Webb)
    # The shadow ratio (omega_n) is defined as 1 - (BSA / Isotropic)

    # Identify Isotropic (f_iso) and BSA variables
    def get_var(ds, search_terms):
        if ds is None: return None
        for v in ds.data_vars:
            if all(term in v for term in search_terms):
                res = ds[v]
                if "time" in res.dims:
                    if res.sizes["time"] > 1:
                        res = res.isel(time=0, drop=True)
                    else:
                        res = res.squeeze("time", drop=True)
                return res
        return None

    # Use matched spectral bands for Isotropic/Geometric parameters to avoid
    # physically inconsistent red-vs-NIR structural ratios.
    logger.debug("Identifying matched-band f_iso and f_geo BRDF parameters...")

    def has_band_token(var_name: str, token: str) -> bool:
        # Exact token match with separators to avoid collisions (e.g., M1 != M10).
        return re.search(rf"(?:^|[_\-]){re.escape(token)}(?:$|[_\-])", var_name, flags=re.IGNORECASE) is not None

    def find_brdf_component(ds: xr.Dataset, component_terms: list[str], band_token: str):
        if ds is None:
            return None, None
        for var_name in ds.data_vars:
            if all(term.lower() in var_name.lower() for term in component_terms) and has_band_token(var_name, band_token):
                res = ds[var_name]
                if "time" in res.dims:
                    if res.sizes["time"] > 1:
                        res = res.isel(time=0, drop=True)
                    else:
                        res = res.squeeze("time", drop=True)
                return res, var_name
        return None, None

    def get_matched_brdf_pair(ds):
        # Priority order: MODIS Band1 (Red) -> VIIRS M5 -> M7 fallback
        candidates = ["Band1", "M5", "M7"]

        for band_name in candidates:
            iso, iso_var = find_brdf_component(ds, ["Isotropic"], band_name)
            if iso is None:
                iso, iso_var = find_brdf_component(ds, ["Parameter1"], band_name)

            geo, geo_var = find_brdf_component(ds, ["Geometric"], band_name)
            if geo is None:
                geo, geo_var = find_brdf_component(ds, ["Parameter3"], band_name)

            if iso is not None and geo is not None:
                logger.info(
                    "Using matched BRDF band for f_iso/f_geo: %s (f_iso=%s, f_geo=%s)",
                    band_name,
                    iso_var,
                    geo_var,
                )
                return iso, geo

        return None, None

    f_iso, f_geo = get_matched_brdf_pair(ds_brdf)

    # Fallback if matched pair not found: find any Isotropic/Parameter1 variable
    if f_iso is None:
        logger.info("Matched BRDF pair not found. Searching for any Isotropic fallback...")
        for v in ds_brdf.data_vars:
            if "isotropic" in v.lower() or "parameter1" in v.lower():
                f_iso = ds_brdf[v]
                break

        # If still None, check if there are other albedo-like variables
        if f_iso is None:
            for v in ds_brdf.data_vars:
                if any(term in v.lower() for term in ["toc", "reflectance", "albedo", "i1"]):
                    f_iso = ds_brdf[v]
                    break

        # If still None, take the first variable in the dataset
        if f_iso is None and len(ds_brdf.data_vars) > 0:
            first_var = list(ds_brdf.data_vars)[0]
            f_iso = ds_brdf[first_var]

        if f_iso is not None and "time" in f_iso.dims:
            if f_iso.sizes["time"] > 1:
                f_iso = f_iso.isel(time=0, drop=True)
            else:
                f_iso = f_iso.squeeze("time", drop=True)

    if f_geo is None:
        for v in ds_brdf.data_vars:
            if "geometric" in v.lower() or "parameter3" in v.lower():
                f_geo = ds_brdf[v]
                break
        if f_geo is not None and "time" in f_geo.dims:
            if f_geo.sizes["time"] > 1:
                f_geo = f_geo.isel(time=0, drop=True)
            else:
                f_geo = f_geo.squeeze("time", drop=True)

    if f_iso is None:
        logger.error(f"Required Isotropic parameter missing. Available: {list(ds_brdf.data_vars)}")
        raise KeyError(f"Could not find Isotropic parameter in BRDF dataset. Found: {list(ds_brdf.data_vars)}")

    if f_geo is None:
        logger.warning("Required Geometric parameter missing. Using scaled Isotropic fallback.")
        f_geo = f_iso * 0.1  # Heuristic fallback when geometric term is unavailable.

    # Identify NBAR (Nadir Reflectance)
    nbar = None
    if ds_nbar is not None:
        nbar = get_var(ds_nbar, ["Nadir", "Band1"])
        if nbar is None: nbar = get_var(ds_nbar, ["Nadir", "M5"])
        if nbar is None: nbar = get_var(ds_nbar, ["Nadir", "M4"])

    # Detect unscaled NASA data: raw HDF files use integer dtypes (int16/uint16);
    # scaled float data is already in [0, 1]. No compute required.
    if f_iso.dtype.kind in ("i", "u"):
        logger.info("Detected integer (unscaled) BRDF parameters. Applying 0.001 scale factor.")
        f_iso = f_iso * np.float32(0.001)
        f_geo = f_geo * np.float32(0.001)

    # SWIR bands for NDTI (Brown vegetation)
    logger.debug("Identifying SWIR bands for NDTI...")

    band7 = get_var(ds_nbar, ["Nadir", "Band7"])
    if band7 is None: band7 = get_var(ds_brdf, ["Isotropic", "Band7"])
    if band7 is None: band7 = f_iso
    gvf = get_var(ds_gvf, ["GVF"])
    if gvf is None: gvf = get_var(ds_gvf, ['gvf_4km'])

    # Extract LAI if available
    lai = get_var(ds_lai, ["LAI"])
    if lai is None: lai = get_var(ds_lai, ["Lai"])

    ndvi = get_var(ds_ndvi, ["NDVI"])
    if ndvi is None: ndvi = get_var(ds_ndvi, ["EVI"])

    # Case-insensitive lookup: the published Prigent file stores its field as
    # "PRIGENT_RDRAG", so an exact-case search for "drag_partition" silently
    # missed it and disabled the static blend entirely.
    def get_var_ci(ds, search_terms):
        if ds is None: return None
        for v in ds.data_vars:
            if all(term.lower() in v.lower() for term in search_terms):
                res = ds[v]
                if "time" in res.dims:
                    if res.sizes["time"] > 1:
                        res = res.isel(time=0, drop=True)
                    else:
                        res = res.squeeze("time", drop=True)
                return res
        return None

    f_prigent = get_var_ci(ds_prigent, ["drag_partition"])
    if f_prigent is None: f_prigent = get_var_ci(ds_prigent, ["drag"])
    if f_prigent is None: f_prigent = get_var_ci(ds_prigent, ["prigent"])

    # Scale NDVI if raw integer values (e.g. 0-10000 range in HDF).
    if ndvi is not None and ndvi.dtype.kind in ("i", "u"):
        logger.info("Detected integer (unscaled) NDVI/EVI. Applying 0.0001 scale factor.")
        ndvi = ndvi * np.float32(0.0001)

    # Protect against unphysical zero or negative values in BRDF parameters
    # and ensure non-NaN outputs for ratio calculations.
    safe_f_iso = f_iso.where(f_iso >= 0.001, 0.001)
    if "lat" in safe_f_iso.coords: safe_f_iso = safe_f_iso.sortby("lat", ascending=False)
    if "lon" in safe_f_iso.coords: safe_f_iso = safe_f_iso.sortby("lon", ascending=True)

    # Use a small floor for f_geo to avoid NaNs in regions with very smooth surfaces
    # while preserving the dynamic range for hybridization.
    safe_f_geo = f_geo.where(f_geo >= 0.0001, 0.0001)
    if "lat" in safe_f_geo.coords: safe_f_geo = safe_f_geo.sortby("lat", ascending=False)
    if "lon" in safe_f_geo.coords: safe_f_geo = safe_f_geo.sortby("lon", ascending=True)

    lam = 1.25 * safe_f_geo / safe_f_iso

    ds_tmp = xr.Dataset({'f_iso': safe_f_iso, 'f_geo': safe_f_geo})

    feff_r = compute_globally_scaled_raupach_drag(ds_tmp,'f_iso','f_geo','ignore_me', gamma_min=0.001, gamma_max=.25, sigma = 1.45, m_stress=0.16)

    # Determine vegetation cover fraction (f_v) using VAI (LAI) or GVF/NDVI as a proxy,
    # prioritizing unmasked NDVI to prevent dry desert zero-masking issues.
    # if lai is not None:
    #     f_v = (lai / 1.0).clip(0, 1)
    # elif gvf is not None:
    #     f_v = gvf.clip(0, 1)
    # elif ndvi is not None:
    #     f_v = ((ndvi - 0.05) / (0.80 - 0.05)).clip(0, 1)
    # else:
    #     f_v = xr.zeros_like(safe_f_iso)

    # In very bright surfaces f_geo is often zero or unreliable. To improve this,
    # we integrate the Prigent et al. static drag partition (R_bare).
    ra_bare = feff_r  # Default to dynamic calculation
    if ds_prigent is not None and f_prigent is not None:
        # f_prigent was already resampled onto the BRDF grid by the align pass
        # above; the previous guard tested for a literal "drag_partition" key,
        # which silently disabled the blend whenever the source variable was
        # named differently (e.g. the published "PRIGENT_RDRAG").
        if "lat" in f_prigent.coords and "lon" in f_prigent.coords:
            # Apply regional tuning (e.g., Taklamakan) if applicable
            f_prigent = apply_regional_taklamakan_tuning(f_prigent, f_prigent.lat, f_prigent.lon, multiplier=1.12)

            # The static climatology encodes ocean/ice as a negative sentinel
            # (~ -1.0) and leaves gaps as NaN. Mask those so they cannot leak
            # unphysical negative drag ratios into the output.
            f_prigent = f_prigent.where(f_prigent >= 0.0)

            # Bare-soil partitioning hybridization:
            # We integrate the Prigent et al. static drag partition (R_bare) wherever
            # the dynamic BRDF retrieval (f_geo) is zero, NaN, or barely above the
            # detection limit. Bright, smooth desert surfaces routinely return a tiny
            # but non-zero f_geo, which would otherwise let the noisy dynamic term take
            # over too early and default to Smooth (R=1).

            # w=1 uses dynamic Raupach; w=0 uses static Prigent.
            # The ramp spans 0 -> f_geo_blend_width, so "basically zero" f_geo
            # (e.g. < 0.005) is still dominated by the static climatology.
            blend_width = max(float(f_geo_blend_width), 1e-6)
            w_hybrid = (f_geo / blend_width).clip(0, 1).fillna(0.0)

            # Combine the dynamic and static components. Where the static field
            # is unavailable (ocean/ice sentinel or NaN gaps), keep the dynamic
            # retrieval only if f_geo was meaningfully retrieved; otherwise fall
            # back to the near-zero floor, which shuts drag off over the ocean
            # instead of leaking the saturated smooth-surface value (R=1).
            blend = w_hybrid * feff_r + (1.0 - w_hybrid) * f_prigent
            dynamic_only = feff_r.where(w_hybrid > 0, masked_drag_floor)
            ra_bare = blend.fillna(dynamic_only).clip(masked_drag_floor, 1.0)
        else:
            logger.warning("Prigent drag partition lacks lat/lon coordinates; skipping static blend.")
    elif ds_prigent is not None:
        logger.warning(
            "Prigent dataset provided but no drag partition variable found. "
            f"Available: {list(ds_prigent.data_vars)}. Skipping static blend."
        )

    feff = ra_bare
    # E. Snow Masking
    snow = get_var(ds_brdf, ["Percent_Snow"])
    if snow is None: snow = get_var(ds_albedo, ["Percent_Snow"])

    if snow is not None:
        # Fill NaNs with 0 to avoid masking missing metadata as snow.
        is_snow = snow.fillna(0) > 0
        logger.info("Applying Percent_Snow mask (lazy).")
        # Floor snow pixels to near-zero drag rather than NaN so the model
        # simply shuts emission off there.
        feff = feff.where(~is_snow, masked_drag_floor)

    # F. Ocean/Water masking
    # Prefer explicit BRDF land/water classification if available.
    land_water = get_var(ds_brdf, ["Land_Water"])
    if land_water is None:
        land_water = get_var(ds_brdf, ["LandWater"])
    if land_water is None:
        land_water = get_var(ds_brdf, ["Land", "Water"])

    if land_water is not None:
        # MODIS/VIIRS BRDF LandWaterType convention:
        # 0,3,5,6,7 are water classes; 1=land, 2=coastline, 4=ephemeral water.
        # We fill NaNs with 1 (land) to avoid aggressive masking in mixed or missing pixels.
        is_land_like = land_water.fillna(1).isin([1, 2, 4])
        logger.info("Applying BRDF Land_Water_Type ocean mask.")
        # Floor water pixels to near-zero drag rather than NaN so the model
        # effectively shuts dust emission off over the ocean.
        feff = feff.where(is_land_like, masked_drag_floor)
    elif ndvi is not None:
        # Fallback when land/water type is unavailable: mask typical open-water NDVI values.
        logger.info("Land/water type not found. Applying NDVI-based ocean fallback mask (ndvi > 0).")
        feff = feff.where(ndvi > 0, masked_drag_floor)

    # Final safety net: any remaining NaN (no BRDF retrieval and no static
    # Prigent coverage) is floored to near-zero drag so the output is always
    # usable by the model instead of carrying NaN gaps.
    feff = feff.fillna(masked_drag_floor)

    # Create output dataset — skip None variables to avoid object-dtype write errors
    data_vars = {
        k: v for k, v in {
            "feff": feff,
            "ndvi": ndvi,
            "f_iso": f_iso,
            "f_geo": f_geo,
            "lam": lam,
            "ra_bare": ra_bare,
        }.items() if v is not None
    }
    ds_out = xr.Dataset(data_vars=data_vars)

    if time_coords is not None:
        ds_out = ds_out.expand_dims(time=time_coords)

    if "feff" in ds_out:
        ds_out.feff.attrs.update({
            "long_name": "Total Effective Drag Coefficient",
            "units": "dimensionless",
        })

    if "f_iso" in ds_out:
        ds_out.f_iso.attrs.update({
            "long_name": "Isotropic Scattering Parameter",
            "units": "dimensionless",
        })

    if "f_geo" in ds_out:
        ds_out.f_geo.attrs.update({
            "long_name": "Geometric Scattering Parameter",
            "units": "dimensionless",
        })

    if "lam" in ds_out:
        ds_out.lam.attrs.update({
            "long_name": "Frontal Area Index (lambda)",
            "units": "dimensionless",
        })

    if "ra_bare" in ds_out:
        ds_out.ra_bare.attrs.update({
            "long_name": "Bare Soil Shear Stress Ratio",
            "units": "dimensionless",
        })

    ds_out.attrs["history"] = (
        f"Calculated at {np.datetime64('now')} using the hybrid drag partition model."
    )

    logger.info("Drag partition calculation complete.")
    return ds_out


def compute_globally_scaled_raupach_drag(ds, f_iso_var, f_geo_var, qa_var,
                                         gamma_min=0.001, gamma_max=0.25,
                                         c_scale=1.2, sigma=1.45, m_stress=0.16,
                                         beta_min=90.0, beta_max=202.0, k_beta=25.0):
    """
    Computes a globally scaled direct Raupach drag partition factor (F_eff)
    by reformulating beta as a dynamic function of range-corrected optical
    lateral cover to resolve the over-sheltering gradient bias.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing aligned multi-angular BRDF parameter arrays.
    f_iso_var / f_geo_var : str
        Variable names for the isotropic and geometric kernel weights.
    qa_var : str
        Quality Assurance mask variable name.
    gamma_min / gamma_max : float
        Observed bare soil background floor and canopy saturation limits.
    c_scale : float
        Calibration scalar mapping stretched shadow parameters to lateral cover.
    sigma : float
        Roughness element basal-to-frontal area index ratio.
    m_stress : float
        Geometric tuning parameter for spatial shear stress non-uniformity.
    beta_min / beta_max : float
        Dynamic floor and ceiling boundaries for the canopy drag coefficient ratio.
    k_beta : float
        Curvature intensity tracking parameter for the beta transition curve.

    Returns:
    --------
    F_eff : xarray.DataArray
        Unified globally scaled drag partition factor array [0.0 - 1.0].
    """
    # 1. Clear out cloud, aerosol, and snow contamination using QA flags
    #valid_mask = ds[qa_var] == 0
    f_iso = ds[f_iso_var]#.where(valid_mask)
    f_geo = ds[f_geo_var]#.where(valid_mask)

    # 2. Extract raw shadow parameters safely
    f_iso_safe = f_iso.where(f_iso > 1e-4, np.nan)
    gamma_raw = f_geo / f_iso_safe

    # 3. Apply Min-Max Feature Normalization to isolate the canopy structure
    gamma_stretched = (gamma_raw - gamma_min) / (gamma_max - gamma_min)
    gamma_calibrated = gamma_stretched.clip(min=0.0, max=1.0)

    # 4. Map the stretched proxy directly to structural lateral cover (lambda_v)
    lat_cover = c_scale * gamma_calibrated

    # 5. Enforce mathematical boundaries to prevent singular/negative denominators
    lambda_upper_bound = (1.0 / (m_stress * sigma)) - 1e-3
    lat_cover_bounded = lat_cover.clip(min=0.0, max=lambda_upper_bound)

    # 6. Structurally Reformulate Beta as a Dynamic Field Variable
    # Smoothly transit beta based on canopy density to handle porous dryland biomes
    beta_dynamic = beta_min + (beta_max - beta_min) * np.tanh(k_beta * lat_cover_bounded)

    # 7. Evaluate the Globally Scaled Raupach Shear Stress Partition
    term_basal = 1.0 - (m_stress * sigma * lat_cover_bounded)
    term_drag = 1.0 + (m_stress * beta_dynamic * lat_cover_bounded)

    F_eff = (term_basal * term_drag) ** -0.5

    return F_eff.clip(min=0.0, max=1.0)

def apply_regional_taklamakan_tuning(f_eff_r_global, lat_array, lon_array, multiplier=1.10):
    """
    Applies a localized scaling multiplier to the rock drag partition factor
    specifically over the geographic boundaries of the Taklamakan Desert.

    Parameters:
    -----------
    f_eff_r_global : xarray.DataArray
        The calculated baseline global rock drag partition factor array.
    lat_array / lon_array : xarray.DataArray
        Coordinate arrays mapped to the layout of the model grid mesh.
    multiplier : float
        The regional scaling coefficient (e.g., 1.10 for a 10% increase).

    Returns:
    --------
    f_eff_r_tuned : xarray.DataArray
        The regionally tuned rock drag partition factor array, clipped to 1.0.
    """
    # 1. Define the spatial bounding box for the Taklamakan Desert core
    # Latitude: 36N to 42N | Longitude: 75E to 90E
    taklamakan_mask = (
        (lat_array >= 36.0) & (lat_array <= 42.0) &
        (lon_array >= 75.0) & (lon_array <= 90.0)
    )

    # 2. Apply the 10% scaling increase strictly inside the masked domain
    # Outside the mask, the baseline global values remain completely untouched
    f_eff_r_tuned = xr.where(taklamakan_mask, f_eff_r_global * multiplier, f_eff_r_global)

    # 3. Guardrail: Enforce the physical boundary limit (f_eff cannot exceed 1.0)
    return f_eff_r_tuned.clip(max=1.0)