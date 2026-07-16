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
    use_lai: bool = True,
    ndvi_threshold: float = 0.35,
    use_gvf_adjustment: bool = False,
    use_ndvi_adjustment: bool = False,
    vegetation_gamma: float = 1.0,
    bare_threshold: float = 0.80,
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

        # Check if coordinates already match (ignoring precision)
        if "lat" in ds_source.dims and "lat" in ds_target.dims:
            if ds_source.lat.size == ds_target.lat.size and ds_source.lon.size == ds_target.lon.size:
                logger.debug(f"Dataset {name} already matches grid size. Aligning coordinates.")
                # Force exact coordinate match to avoid floating point issues later
                ds_source = ds_source.assign_coords(lat=ds_target.lat, lon=ds_target.lon)
                return ds_source

        logger.info(f"Interpolating {name} to match BRDF grid...")
        return ds_source.interp_like(ds_target, method="nearest")

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
    }
    aligned: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
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

    # Scale NDVI if raw integer values (e.g. 0-10000 range in HDF).
    if ndvi is not None and ndvi.dtype.kind in ("i", "u"):
        logger.info("Detected integer (unscaled) NDVI/EVI. Applying 0.0001 scale factor.")
        ndvi = ndvi * np.float32(0.0001)

    # Protect against unphysical zero or negative values in BRDF parameters
    safe_f_iso = f_iso.where(f_iso >= 0.001)

    # Tiny background micro-roughness fallback for f_geo when it is extremely close to zero or NaN
    # to account for grain-scale aeolian roughness, keeping the bare sand drag close to smooth (R ≈ 0.98).
    safe_f_geo = f_geo.where(f_geo >= 0.0001, 0.0001).fillna(0.0001)

    # Option 2: Normalizing f_iso with color using damped coupling (k=0.5)
    # This prevents darker soils from artificially inflating the estimated roughness.
    f_iso_ref = 0.30
    color_normalized_denominator = np.sqrt(safe_f_iso * f_iso_ref)

    # Calculate solid/rock frontal area index (lam_solid) with a safe clip limit
    lam_solid = (1.25 * safe_f_geo / color_normalized_denominator).clip(0, 2)

    # Bare surface/rock drag partition (feff_r, Raupach-style)
    term2_r = (1 - 0.5 * lam_solid) * (1 + 45 * lam_solid)
    feff_r = 1 / np.sqrt(term2_r.clip(min=0.001))

    # Determine vegetation cover fraction (f_v) using VAI (LAI) or GVF/NDVI as a proxy,
    # prioritizing unmasked NDVI to prevent dry desert zero-masking issues.
    if lai is not None:
        f_v = (lai / 1.0).clip(0, 1)
    elif gvf is not None:
        f_v = gvf.clip(0, 1)
    elif ndvi is not None:
        f_v = ((ndvi - 0.05) / (0.80 - 0.05)).clip(0, 1)
    else:
        f_v = xr.zeros_like(safe_f_iso)

    # Calculate vegetation drag partition (feff_v) using Okin [2008] / Pierre et al. [2014]
    # K is the normalized mean gap length between obstacles
    safe_f_v = f_v.where(f_v >= 0.0001)
    K = 2.0 * (1.0 / safe_f_v - 1.0)
    feff_v = xr.where(f_v >= 0.0001, (K + 1.536) / (K + 4.8), 1.0)

    # Non-linear washout protection for partially vegetated transition zones (applied to feff_r).
    if vegetation_gamma != 1.0:
        normalized_lower = feff_r / bare_threshold
        adjusted_lower = (normalized_lower ** vegetation_gamma) * bare_threshold
        feff_r = xr.where(feff_r >= bare_threshold, feff_r, adjusted_lower)

    # Combine solid (feff_r) and vegetation (feff_v) drag partition factors 
    # using the Leung et al. [2023] cubic weighted mean: F_eff^3 = (1 - f_v) * feff_r^3 + f_v * feff_v^3
    feff = ((1.0 - f_v) * (feff_r ** 3) + f_v * (feff_v ** 3)) ** (1.0 / 3.0)

    # For compatibility with downstream outputs and tests
    ra_bare = feff_r
    lam = lam_solid

    # E. Snow Masking
    snow = get_var(ds_brdf, ["Percent_Snow"])
    if snow is None: snow = get_var(ds_albedo, ["Percent_Snow"])

    if snow is not None:
        # Fill NaNs with 0 to avoid masking missing metadata as snow.
        is_snow = snow.fillna(0) > 0
        logger.info("Applying Percent_Snow mask (lazy).")
        feff = feff.where(~is_snow)

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
        is_land_like = land_water.isin([1, 2, 4])
        logger.info("Applying BRDF Land_Water_Type ocean mask.")
        feff = feff.where(is_land_like)
    elif ndvi is not None:
        # Fallback when land/water type is unavailable: mask typical open-water NDVI values.
        logger.info("Land/water type not found. Applying NDVI-based ocean fallback mask (ndvi > 0).")
        feff = feff.where(ndvi > 0)

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
