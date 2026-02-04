"""
Core algorithms for the uthresh pipeline.
Includes physics-based models and machine learning components.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor

# Set up logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION & PARAMETERS ---
IGBP_B_MAP: dict[int, float] = {
    1: 0.04,
    2: 0.04,
    3: 0.04,
    4: 0.04,
    5: 0.04,
    6: 0.11,
    7: 0.10,
    8: 0.09,
    9: 0.09,
    10: 0.08,
    11: 0.06,
    12: 0.14,
    13: 0.10,
    14: 0.12,
    16: 0.18,
}


# --- STAGE 2: PHYSICS ENGINE (DRAG & MOISTURE) ---


def compute_hybrid_drag_partition(
    ds_brdf: xr.Dataset,
    ds_lai: xr.Dataset,
    igbp_class: int | xr.DataArray,
    ds_albedo: xr.Dataset | None = None,
    ds_gvf: xr.Dataset | None = None,
    ds_nbar: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculates the roughness ratio R (us*/u*) using a hybrid model.

    This model combines the Chappell & Webb (2016) bare soil shadowing
    component with the Leung et al. (2023) vegetation sheltering component.

    Parameters
    ----------
    ds_brdf : xarray.Dataset
        Dataset containing BRDF parameters (f_iso, f_vol, f_geo).
    ds_lai : xarray.Dataset
        Dataset containing 'Lai' or 'LAI' (Leaf Area Index).
    igbp_class : Union[int, xarray.DataArray]
        IGBP land cover class(es).
    ds_albedo : xarray.Dataset, optional
        Dataset containing Albedo parameters (BSA). Defaults to None.
    ds_gvf : xarray.Dataset, optional
        Dataset containing 'gvf_4km'. Defaults to None.
    ds_nbar : xarray.Dataset, optional
        Dataset containing NBAR reflectances. Defaults to None.

    Returns
    -------
    xarray.DataArray
        The calculated drag partition ratio (R).
    """
    logger.info("Computing hybrid drag partition (R)...")

    # Bare component (Shadowing) - Chappell & Webb (2016)
    def get_var(ds, search_terms):
        if ds is None: return None
        for v in ds.data_vars:
            if all(term in v for term in search_terms):
                return ds[v]
        return None

    logger.debug("Identifying f_iso (isotropic) parameter...")
    f_iso = get_var(ds_brdf, ["Isotropic", "Band1"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M5"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Isotropic", "M4"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M5"])
    if f_iso is None: f_iso = get_var(ds_brdf, ["Parameter1", "M4"])

    if f_iso is None:
        logger.error(f"Required Isotropic parameter missing. Available: {list(ds_brdf.data_vars)}")
        raise KeyError(f"Could not find Isotropic parameter in BRDF dataset. Found: {list(ds_brdf.data_vars)}")

    logger.debug("Identifying BSA (albedo) parameter...")
    bsa = get_var(ds_albedo, ["BSA", "Band1"])
    if bsa is None: bsa = get_var(ds_albedo, ["BSA", "M5"])
    if bsa is None: bsa = get_var(ds_albedo, ["BSA", "M4"])
    if bsa is None: bsa = get_var(ds_brdf, ["BSA", "Band1"])
    if bsa is None: bsa = get_var(ds_brdf, ["BSA", "M5"])
    if bsa is None: bsa = get_var(ds_brdf, ["BSA", "M4"])
    if bsa is None:
        logger.warning("BSA missing, falling back to Isotropic (BSA=f_iso)")
        bsa = f_iso

    logger.info("Computing shadow ratio (omega_n)...")
    omega_n = (1.0 - (bsa / f_iso)).clip(0, 1) * 100.0

    # Scale to shadow-volume omega_ns
    logger.info("Scaling to shadow volume (omega_ns)...")
    omega_ns = ((0.0001 - 0.1) * (omega_n - 35.0) / (0.0 - 35.0)) + 0.1
    omega_ns = omega_ns.clip(0.0001, 0.1)
    ra_bare = 0.0311 * np.exp(-omega_ns / 1.131) + 0.007

    # Veg component (Sheltering)
    logger.info("Computing vegetation component...")
    b_param = IGBP_B_MAP.get(igbp_class, 0.1)

    lai_var = "Lai" if "Lai" in ds_lai else "LAI"
    lai = ds_lai[lai_var]

    if ds_gvf is not None and "gvf_4km" in ds_gvf:
        sigma_total = ds_gvf["gvf_4km"].clip(0, 1)
    else:
        sigma_total = (1.0 - np.exp(-0.5 * lai)).clip(0, 1)

    lambda_total = (lai / 2.0) + 0.05
    f_veg = (1.0 - sigma_total) * np.exp(-lambda_total / b_param)

    logger.info("Hybrid drag partition calculation complete.")
    return ra_bare * f_veg


def compute_moisture_inhibition(
    moisture: float | np.ndarray,
    clay: float | np.ndarray,
    soc: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculates the moisture inhibition factor H(w).

    This function implements the relationship from Fecan et al. (1999) with
    an adjustment for Soil Organic Carbon (SOC).

    Parameters
    ----------
    moisture : Union[float, np.ndarray]
        Soil moisture content.
    clay : Union[float, np.ndarray]
        Clay fraction (%).
    soc : Union[float, np.ndarray]
        Soil Organic Carbon content (g/kg or similar unit, scaled in func).

    Returns
    -------
    Union[float, np.ndarray]
        The moisture inhibition factor, where 1.0 means no inhibition.
    """
    w_prime = (0.0014 * clay**2 + 0.17 * clay) * (1 + 0.05 * soc / 100)
    return np.where(
        moisture > w_prime, np.sqrt(1 + 1.21 * (moisture - w_prime) ** 0.68), 1.0
    )


# --- STAGE 3: PIML TRAINING & STRATIFICATION ---


def prepare_balanced_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratifies training data to ensure representativeness.

    The function stratifies by IGBP land cover and soil texture class, then
    samples from each group to create a more balanced training set.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing training data, including 'clay', 'igbp',
        and other features.

    Returns
    -------
    pd.DataFrame
        A new dataframe with balanced samples.
    """
    df["texture"] = pd.cut(
        df["clay"], bins=[0, 15, 30, 100], labels=["Sand", "Loam", "Clay"]
    )
    # Sample equally across land-use/soil combinations
    return df.groupby(["igbp", "texture"], group_keys=False, observed=False).apply(
        lambda x: x.sample(min(len(x), 300))
    )


def train_piml_model(df: pd.DataFrame) -> XGBRegressor:
    """
    Trains the PIML threshold velocity inverter model.

    Uses XGBoost and is designed to be evaluated with GroupKFold to ensure
    the model generalizes well across different IGBP classes.

    Parameters
    ----------
    df : pd.DataFrame
        The prepared training dataframe, including all features and the
        target variable 'u_eff_target'.

    Returns
    -------
    xgboost.XGBRegressor
        The trained XGBoost model.
    """
    features = ["clay", "soc", "bdod", "R_partition", "h_w_inhibition", "lai"]
    X, y = df[features], df["u_eff_target"]  # u_eff at detection time

    # Model: XGBoost optimized for physical non-linearity
    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=7, subsample=0.8
    )

    # Evaluate with GroupKFold to ensure unseen IGBP generalization
    # (Cross-val scoring logic here...)

    model.fit(X, y)
    return model


def predict_threshold_velocity(
    model: XGBRegressor,
    ds_soil: xr.Dataset,
    R: xr.DataArray,
    H: xr.DataArray,
    lai: xr.DataArray,
) -> xr.DataArray:
    """
    Predicts threshold friction velocity (u*t) using the trained PIML model.

    This high-performance version avoids creating an intermediate Pandas DataFrame.
    Instead, it stacks the input DataArrays, constructs a NumPy feature array,
    runs the prediction, and reshapes the result back into a DataArray with
    the original coordinates preserved.

    Parameters
    ----------
    model : xgboost.XGBRegressor
        The trained PIML model.
    ds_soil : xr.Dataset
        Dataset containing soil properties ('clay', 'soc', 'bdod').
    R : xr.DataArray
        The drag partition ratio.
    H : xr.DataArray
        The moisture inhibition factor.
    lai : xr.DataArray
        The Leaf Area Index.

    Returns
    -------
    xr.DataArray
        A DataArray of the predicted threshold friction velocity, with dimensions
        matching the input `lai` DataArray.
    """
    # Ensure all inputs are aligned to the same grid as LAI
    # This is safer and prevents accidental misalignments.
    ds_soil_aligned = ds_soil.broadcast_like(lai)
    R_aligned = R.broadcast_like(lai)
    H_aligned = H.broadcast_like(lai)

    # Define the order of features expected by the model
    feature_arrays = [
        ds_soil_aligned["clay"],
        ds_soil_aligned["soc"],
        ds_soil_aligned["bdod"],
        R_aligned,
        H_aligned,
        lai,
    ]

    # Stack the arrays into a 2D feature matrix (n_points, n_features)
    # The .values call extracts the underlying numpy array.
    feature_matrix = np.vstack([arr.values.ravel() for arr in feature_arrays]).T

    # Run prediction on the raw numpy data
    u_thresh_flat = model.predict(feature_matrix)

    # Reshape the flat prediction array back to the original spatial dimensions
    # and return it as a DataArray with proper coordinates.
    return xr.DataArray(
        u_thresh_flat.reshape(lai.shape),
        coords=lai.coords,
        dims=lai.dims,
        name="threshold_velocity",
    )
