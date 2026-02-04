import logging
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
    use_lai: bool = True,
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
    use_lai : bool, optional
        Whether to include the LAI (green vegetation) component in the
        calculation. Defaults to True.

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

        # Check if coordinates already match (ignoring precision)
        if "lat" in ds_source.dims and "lat" in ds_target.dims:
            if ds_source.lat.size == ds_target.lat.size and ds_source.lon.size == ds_target.lon.size:
                logger.debug(f"Dataset {name} already matches grid size. Aligning coordinates.")
                # Force exact coordinate match to avoid floating point issues later
                ds_source = ds_source.assign_coords(lat=ds_target.lat, lon=ds_target.lon)
                if "time" in ds_target.dims and "time" not in ds_source.dims:
                    ds_source = ds_source.expand_dims(time=ds_target.time)
                return ds_source

        logger.info(f"Interpolating {name} to match BRDF grid...")
        return ds_source.interp_like(ds_target, method="nearest")

    ds_lai = align_dataset(ds_brdf, ds_lai, "LAI")
    ds_gvf = align_dataset(ds_brdf, ds_gvf, "GVF")
    ds_nbar = align_dataset(ds_brdf, ds_nbar, "NBAR")
    ds_albedo = align_dataset(ds_brdf, ds_albedo, "Albedo")

    # --- DRAG PARTITION CALCULATION ---
    # A. Bare Surface (Chappell & Webb)
    # The shadow ratio (omega_n) is defined as 1 - (BSA / Isotropic)

    # Identify Isotropic (f_iso) and BSA variables
    def get_var(ds, search_terms):
        if ds is None: return None
        for v in ds.data_vars:
            if all(term in v for term in search_terms):
                return ds[v]
        return None

    # Red Band (Band1 for MODIS, M5/M4 for VIIRS)
    logger.debug("Identifying f_iso (isotropic) parameter...")
    f_iso = get_var(ds_brdf, ["Isotropic", "Band1"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M5"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M4"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M5"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M4"])

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

    # SWIR bands for NDTI (Brown vegetation)
    logger.debug("Identifying SWIR bands for NDTI...")
    band6 = get_var(ds_nbar, ["Nadir", "Band6"])
    if band6 is None: band6 = get_var(ds_nbar, ["Nadir", "M10"])
    if band6 is None: band6 = get_var(ds_brdf, ["Isotropic", "Band6"])
    if band6 is None: band6 = get_var(ds_brdf, ["Isotropic", "M10"])
    if band6 is None: band6 = f_iso

    band7 = get_var(ds_nbar, ["Nadir", "Band7"])
    if band7 is None: band7 = get_var(ds_nbar, ["Nadir", "M11"])
    if band7 is None: band7 = get_var(ds_brdf, ["Isotropic", "Band7"])
    if band7 is None: band7 = get_var(ds_brdf, ["Isotropic", "M11"])
    if band7 is None: band7 = f_iso
    gvf = get_var(ds_gvf, ["GVF"])
    if gvf is None: gvf = get_var(ds_gvf, ['gvf_4km'])

    ratio = f_geo.clip(0.001,1) / f_iso
    lam = 1.25 * ratio
    texture = f_iso.rolling(lat=4, lon=4, center=True).std()
    roughness_proxy = (texture / 0.1).clip(0,0.5)

    R = 1 / np.sqrt((1-1.45*0.16 * lam) * (1 + 202*0.16 * lam))
    mask_saturated = R > 0.85
    ra_bare = xr.where(mask_saturated, R - roughness_proxy, R)
    R2 = 1 / np.sqrt((1 - 0.5 * lam) * (1 +45 * lam))
    mask_saturated2 = R2 > 0.85
    ra_bare2 = xr.where(mask_saturated2, R2 - roughness_proxy, R2)
    feff = ra_bare.where(gvf < 0.3)
    feff2 = ra_bare2.where(gvf < 0.3)

    # E. Snow Masking
    snow = get_var(ds_brdf, ["Percent_Snow"])
    if snow is None: snow = get_var(ds_albedo, ["Percent_Snow"])

    if snow is not None:
        logger.info("Applying Percent_Snow mask (threshold > 0)...")
        # Mask out feff where snow is present to suppress dust emission
        feff = feff.where(snow == 0)
        feff2 = feff2.where(snow == 0)

    # Create output dataset
    ds_out = xr.Dataset(
        data_vars={
            "feff": feff,
            "feff2": feff2,
            "ra_bare": ra_bare,
            # 'ratio': f_geo / f_iso.where(f_iso > 0)
            # 'f_geo': f_geo,
            # 'f_iso': f_iso,
        }
    )

    ds_out.feff.attrs.update({
        "long_name": "Total Effective Drag Coefficient",
        "units": "dimensionless",
    })
    ds_out.ra_bare.attrs.update({
        "long_name": "Bare Surface Drag Component (ra_bare)",
        "units": "dimensionless",
    })

    ds_out.attrs["history"] = (
        f"Calculated at {np.datetime64('now')} using the hybrid drag partition model."
    )

    logger.info("Drag partition calculation complete.")
    return ds_out
