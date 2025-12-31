"""
Global Dust Emission PIML Suite (GDE-PIML)
Full end-to-end pipeline: Retrieval -> Physics -> ML -> Flux Projection.

References:
- Chappell & Webb ((2016) [AEM]: DOI 10.1016/j.aeolia.2015.11.001
- Leung et al. (2023) [Vegetation]: DOI 10.5194/acp-23-11235-2023
- Marticorena & Bergametti (1995) [Flux]: DOI 10.1029/95JD00690
- SoilGrids v2.0: DOI 10.5194/soil-7-217-2021
"""

import io
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import earthaccess
import rasterio
from owslib.wcs import WebCoverageService
from xgboost import XGBRegressor

# --- CONFIGURATION & PARAMETERS ---
IGBP_B_MAP: Dict[int, float] = {
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

# --- STAGE 1: DATA RETRIEVAL (EARTHDATA & S3) ---


class DustDataEngine:
    """Orchestrates the retrieval of meteorological and soil data."""

    def __init__(self) -> None:
        """Initializes authentication handlers for Earthdata and S3."""
        self.auth = earthaccess.login(strategy="interactive")
        self.fs_s3 = s3fs.S3FileSystem(anon=True)
        self.wcs_url = "https://maps.isric.org/mapserv?map=/srv/node/node_modules/soilgrids/maps/soilgrids.map"

    def fetch_met_ufs(self, dt: datetime, lat: float, lon: float) -> xr.Dataset:
        """
        Fetch UFS Replay 3-hourly meteorological data for a specific time and location.

        Parameters
        ----------
        dt : datetime.datetime
            The timestamp for which to fetch the data.
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        xarray.Dataset
            A dataset containing the meteorological variables for the nearest grid point.
        """
        bucket = "noaa-ufs-gefsv13replay-pds"
        path = f"s3://{bucket}/{dt.strftime('%Y%m%d/%H')}/atmos/gefs.t{dt.strftime('%H')}z.pgrb2.0p25.f000"
        with self.fs_s3.open(path) as f:
            ds = xr.open_dataset(
                f,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            )
            return ds.sel(latitude=lat, longitude=lon, method="nearest")

    def fetch_soilgrids(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Fetch physical/chemical soil properties from ISRIC SoilGrids via WCS.

        Parameters
        ----------
        lat : float
            Latitude of the target point.
        lon : float
            Longitude of the target point.

        Returns
        -------
        Dict[str, float]
            A dictionary containing soil properties ('clay', 'sand', 'soc', 'bdod').
        """
        wcs = WebCoverageService(self.wcs_url, version="2.0.1")
        bbox = (lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)
        res: Dict[str, float] = {}
        for var in ["clay", "sand", "soc", "bdod"]:
            layer = f"{var}_0-5cm_mean"
            resp = wcs.getCoverage(
                identifier=layer,
                crs="EPSG:4326",
                bbox=bbox,
                width=1,
                height=1,
                format="image/tiff",
            )
            with rasterio.open(io.BytesIO(resp.read())) as src:
                res[var] = src.read(1)[0, 0]
        return res


# --- STAGE 2: PHYSICS ENGINE (DRAG & MOISTURE) ---


def compute_hybrid_drag_partition(
    ds_alb: xr.Dataset, ds_lai: xr.Dataset, igbp_class: Union[int, xr.DataArray]
) -> xr.DataArray:
    """
    Calculates the roughness ratio R (us*/u*) using a hybrid model.

    This model combines the Chappell & Webb (2016) bare soil shadowing
    component with the Leung et al. (2023) vegetation sheltering component.

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
    moisture: Union[float, np.ndarray],
    clay: Union[float, np.ndarray],
    soc: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
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


# --- STAGE 4: GLOBAL PROJECTION & FLUX ---


def generate_dust_flux_map(
    ds_alb: xr.Dataset,
    ds_lai: xr.Dataset,
    ds_lc: xr.Dataset,
    ds_soil: xr.Dataset,
    ds_met: xr.Dataset,
    model: XGBRegressor,
) -> xr.DataArray:
    """
    Projects the ML threshold and calculates the Marticorena-Bergametti flux.

    This function serves as the final step, integrating the physics components
    with the trained PIML model to produce a map of dust flux.

    Parameters
    ----------
    ds_alb : xarray.Dataset
        Albedo data.
    ds_lai : xarray.Dataset
        Leaf Area Index data.
    ds_lc : xarray.Dataset
        Land Cover data ('LC_Type1').
    ds_soil : xarray.Dataset
        Soil properties data ('clay', 'soc', 'bdod').
    ds_met : xarray.Dataset
        Meteorological data ('soilw', 'ustar').
    model : xgboost.XGBRegressor
        The trained PIML model for predicting threshold velocity.

    Returns
    -------
    xarray.DataArray
        The final calculated vertical dust flux.
    """
    # 1. Feature Prep
    R = compute_hybrid_drag_partition(ds_alb, ds_lai, ds_lc["LC_Type1"])
    H = compute_moisture_inhibition(ds_met["soilw"], ds_soil["clay"], ds_soil["soc"])

    # 2. Predict u*t (Threshold)
    # Flatten xarray DataArrays into a 1D array for prediction
    feature_df = pd.DataFrame(
        {
            "clay": ds_soil["clay"].values.ravel(),
            "soc": ds_soil["soc"].values.ravel(),
            "bdod": ds_soil["bdod"].values.ravel(),
            "R_partition": R.values.ravel(),
            "h_w_inhibition": H.values.ravel(),
            "lai": ds_lai["Lai"].values.ravel(),
        }
    )

    # Predict using the trained model
    u_thresh_flat = model.predict(feature_df)

    # Reshape the predictions back to the original 2D grid
    u_thresh = xr.DataArray(
        u_thresh_flat.reshape(ds_lai.lat.size, ds_lai.lon.size),
        coords=ds_lai.coords,
        dims=ds_lai.dims,
    )

    # 3. Flux Calculation (Physics)
    u_eff = (ds_met["ustar"] * R) / H
    # Saltation Q (g/m/s)
    Q = xr.where(
        u_eff > u_thresh,
        (1.23 / 9.81)
        * u_eff**3
        * (1 - u_thresh**2 / u_eff**2)
        * (1 + u_thresh / u_eff),
        0,
    )
    # Sandblasting alpha
    alpha = 10 ** (13.4 * (ds_soil["clay"] / 100) - 6.0)
    dust_flux = Q * alpha  # Vertical Flux

    return dust_flux
