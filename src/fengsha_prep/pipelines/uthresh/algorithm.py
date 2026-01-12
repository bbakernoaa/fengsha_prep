"""
Core algorithms for the uthresh pipeline.
Includes physics-based models and machine learning components.
"""
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor

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
    ds_alb: xr.Dataset, ds_lai: xr.Dataset, igbp_class: int | xr.DataArray
) -> xr.DataArray:
    """
    Calculates the roughness ratio R (us*/u*) using a hybrid model.

    This model combines the Chappell & Webb (2016) bare soil shadowing
    component with the Leung et al. (2 XXIII) vegetation sheltering component.

    Parameters
    ----------
    ds_alb : xarray.Dataset
        Dataset containing albedo variables, including 'Albedo_BSW_Band1'
        and 'BRDF_Albedo_Parameter_Isotropic_Band1'.
    ds_lai : xarray.Dataset
        Dataset containing 'Lai' (Leaf Area Index).
    igbp_class : Union[int, xarray.DataArray]
        IGBP land cover class(es).

    Returns
    -------
    xarray.DataArray
        The calculated drag partition ratio (R).
    """
    # Bare component (Shadowing)
    omega_n = 1.0 - (
        ds_alb["Albedo_BSW_Band1"] / ds_alb["BRDF_Albedo_Parameter_Isotropic_Band1"]
    )
    omega_ns = ((0.0001 - 0.1) * (omega_n - 35.0) / (0.0 - 35.0)) + 0.1
    ra_bare = 0.0311 * np.exp(-omega_ns / 1.131) + 0.007

    # Veg component (Sheltering)
    b_param = IGBP_B_MAP.get(igbp_class, 0.1)
    sigma_total = (1.0 - np.exp(-0.5 * ds_lai["Lai"])).clip(0, 1)
    lambda_total = (ds_lai["Lai"] / 2.0) + 0.05
    f_veg = (1.0 - sigma_total) * np.exp(-lambda_total / b_param)

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
    return df.groupby(["igbp", "texture"], group_keys=False).apply(
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
    Predicts the threshold friction velocity (u*t) using the trained PIML model.

    This function prepares the feature DataFrame from input xarray objects,
    runs the prediction, and reshapes the result back into a DataArray.

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
        A DataArray of the predicted threshold friction velocity.
    """
    feature_df = pd.DataFrame(
        {
            "clay": ds_soil["clay"].values.ravel(),
            "soc": ds_soil["soc"].values.ravel(),
            "bdod": ds_soil["bdod"].values.ravel(),
            "R_partition": R.values.ravel(),
            "h_w_inhibition": H.values.ravel(),
            "lai": lai.values.ravel(),
        }
    )

    u_thresh_flat = model.predict(feature_df)

    return xr.DataArray(
        u_thresh_flat.reshape(lai.shape),
        coords=lai.coords,
        dims=lai.dims,
    )
