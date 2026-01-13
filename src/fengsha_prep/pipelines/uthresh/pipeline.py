"""
Global Dust Emission PIML Suite (GDE-PIML)
Full end-to-end pipeline: Retrieval -> Physics -> ML -> Flux Projection.

References:
- Chappell & Webb ((2016) [AEM]: DOI 10.1016/j.aeolia.2015.11.001
- Leung et al. (2023) [Vegetation]: DOI 10.5194/acp-23-11235-2023
- Marticorena & Bergametti (1995) [Flux]: DOI 10.1029/95JD00690
- SoilGrids v2.0: DOI 10.5194/soil-7-217-2021
"""

from datetime import datetime
from typing import Optional, Protocol

import xarray as xr
from xgboost import XGBRegressor

from .algorithm import (
    compute_hybrid_drag_partition,
    compute_moisture_inhibition,
    predict_threshold_velocity,
)
from .io import DustDataEngine

# --- PROTOCOL FOR DEPENDENCY INJECTION ---


class DataFetcher(Protocol):
    """Defines the interface for a data fetching object."""

    def fetch_met_ufs(self, dt: datetime, lat: float, lon: float) -> xr.Dataset: ...

    def fetch_soilgrids(self, lat: float, lon: float) -> dict[str, float]: ...


# --- HELPER FUNCTIONS FOR FLUX CALCULATION ---


def _prepare_physical_features(
    ds_alb: xr.Dataset,
    ds_lai: xr.Dataset,
    ds_lc: xr.Dataset,
    ds_soil: xr.Dataset,
    ds_met: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes intermediate physical variables for the flux calculation.

    - R: The drag partition ratio.
    - H: The moisture inhibition factor.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        A tuple containing the R and H DataArrays.
    """
    R = compute_hybrid_drag_partition(ds_alb, ds_lai, ds_lc["LC_Type1"])
    H = compute_moisture_inhibition(ds_met["soilw"], ds_soil["clay"], ds_soil["soc"])
    R.name = "drag_partition_ratio"
    H.name = "moisture_inhibition_factor"
    return R, H


def _calculate_saltation_flux(
    u_eff: xr.DataArray, u_thresh: xr.DataArray
) -> xr.DataArray:
    """
    Calculates the saltation flux (Q) based on Marticorena & Bergametti (1995).

    Parameters
    ----------
    u_eff : xr.DataArray
        The effective friction velocity.
    u_thresh : xr.DataArray
        The threshold friction velocity.

    Returns
    -------
    xr.DataArray
        The calculated saltation flux.
    """
    Q = xr.where(
        u_eff > u_thresh,
        (1.23 / 9.81)
        * u_eff**3
        * (1 - u_thresh**2 / u_eff**2)
        * (1 + u_thresh / u_eff),
        0,
    )
    Q.name = "saltation_flux"
    return Q


def _calculate_sandblasting_efficiency(clay_fraction: xr.DataArray) -> xr.DataArray:
    """
    Calculates the sandblasting efficiency (alpha).

    Parameters
    ----------
    clay_fraction : xr.DataArray
        The clay fraction of the soil (in %).

    Returns
    -------
    xr.DataArray
        The sandblasting efficiency factor.
    """
    alpha = 10 ** (13.4 * (clay_fraction / 100) - 6.0)
    alpha.name = "sandblasting_efficiency"
    return alpha


# --- PIPELINE ORCHESTRATION ---


def generate_dust_flux_map(
    ds_alb: xr.Dataset,
    ds_lai: xr.Dataset,
    ds_lc: xr.Dataset,
    ds_soil: xr.Dataset,
    ds_met: xr.Dataset,
    model: XGBRegressor,
) -> xr.DataArray:
    """
    Orchestrates the physics and ML components to produce a dust flux map.

    This function integrates the physics-based feature preparation with the
    trained PIML model to predict and calculate the final dust flux based on
    the Marticorena & Bergametti (1995) model.

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
    # 1. Feature Prep (Physics)
    R, H = _prepare_physical_features(ds_alb, ds_lai, ds_lc, ds_soil, ds_met)

    # 2. Predict u*t (Machine Learning)
    u_thresh = predict_threshold_velocity(model, ds_soil, R, H, ds_lai["Lai"])

    # 3. Flux Calculation (Physics)
    u_eff = (ds_met["ustar"] * R) / H
    Q = _calculate_saltation_flux(u_eff, u_thresh)
    alpha = _calculate_sandblasting_efficiency(ds_soil["clay"])

    dust_flux = Q * alpha
    dust_flux.name = "vertical_dust_flux"

    return dust_flux


def _run_pipeline_with_fetcher(
    dt: datetime, lat: float, lon: float, data_fetcher: DataFetcher
) -> xr.Dataset:
    """
    Internal helper to run the data fetching part of the pipeline.
    Note: This remains a placeholder for a more complete implementation.
    """
    ds_met = data_fetcher.fetch_met_ufs(dt, lat, lon)
    soil_props = data_fetcher.fetch_soilgrids(lat, lon)
    ds_soil = xr.Dataset(soil_props)
    return xr.merge([ds_met, ds_soil])


def run_uthresh_pipeline(
    dt: datetime,
    lat: float,
    lon: float,
    model: XGBRegressor,
    data_fetcher: Optional[DataFetcher] = None,
) -> xr.Dataset:
    """
    Full end-to-end pipeline for a single point in time and space.

    Orchestrates data fetching and flux calculation. If a `data_fetcher` is not
    provided, a default `DustDataEngine` will be created and used.

    Note: The current implementation is a placeholder and only demonstrates
    the data fetching step. A full implementation would also fetch albedo,
    LAI, and land cover data to call `generate_dust_flux_map`.

    Parameters
    ----------
    dt : datetime.datetime
        The timestamp for the analysis.
    lat : float
        Latitude of the target point.
    lon : float
        Longitude of the target point.
    model : xgboost.XGBRegressor
        The trained PIML model.
    data_fetcher : Optional[DataFetcher]
        An object that provides data fetching methods. If None, a default
        `DustDataEngine` is created. This is useful for dependency injection.

    Returns
    -------
    xarray.Dataset
        A dataset containing the fetched meteorological and soil data.
    """
    if data_fetcher:
        return _run_pipeline_with_fetcher(dt, lat, lon, data_fetcher)

    # If no fetcher is provided, create a default one and manage its lifecycle.
    with DustDataEngine() as engine:
        return _run_pipeline_with_fetcher(dt, lat, lon, engine)
