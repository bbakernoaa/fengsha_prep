import logging
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

    # Red Band (Band1 for MODIS, M5/M4 for VIIRS/VNP/VJ1 - or M7 if requested)
    logger.debug("Identifying f_iso (isotropic) parameter...")
    f_iso = get_var(ds_brdf, ["Isotropic", "Band1"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M7"])  # Prioritize M7 for consistency if M5 is filtered out
    # if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M5"])
    # if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M4"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M7"])
    # if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M5"])
    # if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M4"])

    if f_iso is None:
        logger.error(f"Required Isotropic parameter missing. Available: {list(ds_brdf.data_vars)}")
        raise KeyError(f"Could not find Isotropic parameter in BRDF dataset. Found: {list(ds_brdf.data_vars)}")

    # Identify NBAR (Nadir Reflectance)
    nbar = None
    if ds_nbar is not None:
        nbar = get_var(ds_nbar, ["Nadir", "Band1"])
        if nbar is None: nbar = get_var(ds_nbar, ["Nadir", "M5"])
        if nbar is None: nbar = get_var(ds_nbar, ["Nadir", "M4"])

    # Identify Geometric kernel weight (Parameter 3)
    # This is a direct proxy for surface roughness/shadowing structures
    f_geo = get_var(ds_brdf, ["Geometric", "Band2"])
    if f_geo is None: f_geo = get_var(ds_brdf, ["Geometric", "M7"])
    if f_geo is None: f_geo = get_var(ds_brdf, ["Parameter3", "M7"])

    if f_geo is None:
        logger.warning("Required Geometric parameter missing. Using Isotropic as fallback.")
        f_geo = f_iso * 0.1  # Heuristic fallback

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
    texture = safe_f_iso.rolling(lat=4, lon=4, center=True).std()
    roughness_proxy = (texture / 0.1).clip(0,0.5)

    # Bare surface drag (Chappell & Webb variant 2)
    term2 = (1 - 0.5 * lam) * (1 + 45 * lam)
    R2 = 1 / np.sqrt(term2.clip(min=0.001))
    ra_bare = xr.where(R2 > 0.85, R2 - roughness_proxy, R2)

    # Vegetation masking
    if gvf is not None:
        logger.info("Applying GVF mask (< 0.3)...")
        feff = ra_bare.where(gvf < 0.3)
    elif ndvi is not None:
        logger.info(f"Applying NDVI/EVI mask (< {ndvi_threshold})...")
        feff = ra_bare.where(ndvi < ndvi_threshold)
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
