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

    ndvi = get_var(ds_ndvi, ["NDVI"])
    if ndvi is None: ndvi = get_var(ds_ndvi, ["EVI"])

    # Scale NDVI if raw integer values (e.g. 0-10000 range in HDF).
    if ndvi is not None and ndvi.dtype.kind in ("i", "u"):
        logger.info("Detected integer (unscaled) NDVI/EVI. Applying 0.0001 scale factor.")
        ndvi = ndvi * np.float32(0.0001)

    # Ratio calculation with safety for zero f_iso
    safe_f_iso = f_iso.where(f_iso >= 0.001)
    ratio = (f_geo.fillna(0.001) / safe_f_iso).clip(0, 2)
    
    lam = 1.25 * ratio
    texture = safe_f_iso.rolling(lat=5, lon=5, center=True).std()
    roughness_proxy = (texture / 0.1).clip(0,0.5)

    # Bare surface drag (Chappell & Webb variant 2)
    term2 = (1 - 0.5 * lam) * (1 + 45 * lam)
    R2 = 1 / np.sqrt(term2.clip(min=0.001))
    roughness_weight = ((R2 - 0.75) / 0.20).clip(0, 1)
    ra_bare = R2 - (roughness_proxy * roughness_weight)

    # Non-linear washout protection for partially vegetated transition zones.
    if vegetation_gamma != 1.0:
        normalized_lower = ra_bare / bare_threshold
        adjusted_lower = (normalized_lower ** vegetation_gamma) * bare_threshold
        ra_bare = xr.where(ra_bare >= bare_threshold, ra_bare, adjusted_lower)

    # Vegetation masking
    if gvf is not None and use_gvf_adjustment:
        logger.info("Applying soft GVF attenuation with dense-vegetation mask (< 0.6)...")
        feff = ra_bare * (1.0 - gvf.clip(0, 1))
        feff = feff.where(gvf < 0.6)
    elif gvf is not None and not use_gvf_adjustment:
        logger.info("GVF adjustment disabled; using BRDF-only drag.")
        feff = ra_bare
    elif ndvi is not None and use_ndvi_adjustment:
        logger.info(f"Applying soft NDVI/EVI attenuation with mask (< {ndvi_threshold})...")
        veg_fraction = ((ndvi - 0.1) / (ndvi_threshold - 0.1)).clip(0, 1)
        feff = ra_bare * (1.0 - veg_fraction)
        feff = feff.where(ndvi < ndvi_threshold)
    elif ndvi is not None and not use_ndvi_adjustment:
        logger.info("NDVI/EVI adjustment disabled; using BRDF-only drag where GVF is unavailable.")
        feff = ra_bare
    else:
        logger.warning("No vegetation mask (GVF or NDVI) applied.")
        feff = ra_bare

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
    data_vars = {k: v for k, v in {"feff": feff, "ndvi": ndvi}.items() if v is not None}
    ds_out = xr.Dataset(data_vars=data_vars)

    if time_coords is not None:
        ds_out = ds_out.expand_dims(time=time_coords)

    if "feff" in ds_out:
        ds_out.feff.attrs.update({
            "long_name": "Total Effective Drag Coefficient",
            "units": "dimensionless",
        })

    ds_out.attrs["history"] = (
        f"Calculated at {np.datetime64('now')} using the hybrid drag partition model."
    )

    logger.info("Drag partition calculation complete.")
    return ds_out
