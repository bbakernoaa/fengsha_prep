import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from satpy import Scene
from sklearn.cluster import DBSCAN

from fengsha_prep.common import satellite

# Set up logging
logger = logging.getLogger(__name__)

# Default thresholds for dust detection, can be overridden.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "diff_12_10": -0.5,
    "diff_10_8": 2.0,
    "temp_10": 280,
}

# --- Performance Tuning Constants for Clustering ---
# If the number of dusty pixels exceeds this, downsample before clustering.
PIXEL_COUNT_THRESHOLD = 75_000
# The factor by which to downsample the dust mask (e.g., a factor of 4
# means a 4x4 grid becomes 1 pixel).
COARSEN_FACTOR = 4


def detect_dust(scn: Scene, sat_id: str, thresholds: dict[str, float]) -> xr.DataArray:
    """Applies a physical algorithm to detect dust in a scene.

    This algorithm is based on the brightness temperature differences between
    infrared channels, a common technique for identifying dust aerosols. The
    resulting DataArray includes a 'history' attribute detailing the
    detection parameters.

    Parameters
    ----------
    scn : Scene
        The Satpy Scene object containing the satellite data.
    sat_id : str
        The identifier of the satellite.
    thresholds : Dict[str, float]
        A dictionary of thresholds used for dust detection.

    Returns
    -------
    xr.DataArray
        A binary dust mask (1=Dust, 0=No Dust) with coordinate information
        and a 'history' attribute documenting the processing.
    """
    logger.info(f"Detecting dust for satellite {sat_id}...")
    meta = satellite.get_satellite_metadata(sat_id)
    if not meta or "bands" not in meta or len(meta["bands"]) < 3:
        logger.error(f"Invalid band configuration for {sat_id}")
        raise ValueError(f"Invalid band configuration for {sat_id}")

    bands = meta["bands"]
    logger.debug(f"Using bands: {bands}")
    b08, b10, b12 = scn[bands[0]], scn[bands[1]], scn[bands[2]]

    logger.debug("Applying spectral thresholds...")
    diff_12_10 = b12 - b10
    diff_10_8 = b10 - b08

    dust_mask = (
        (diff_12_10 < thresholds["diff_12_10"])
        & (diff_10_8 > thresholds["diff_10_8"])
        & (b10 > thresholds["temp_10"])
    )

    dust_pixel_count = int(dust_mask.sum())
    logger.info(f"Dust detection complete. Found {dust_pixel_count} dusty pixels.")

    # --- Add Provenance ---
    history_log = (
        f"Dust mask generated at {datetime.datetime.now(datetime.UTC).isoformat()}Z. "
        f"Satellite: {sat_id}. Thresholds: {thresholds}."
    )
    dust_mask.attrs["history"] = history_log

    return dust_mask


def cluster_events(
    dust_mask: xr.DataArray, scn_time: datetime.datetime, sat_id: str
) -> list[dict[str, Any]]:
    """Identifies distinct dust plumes from a dust mask using DBSCAN.

    For performance on large dust events, this function first checks if the
    number of dusty pixels exceeds a threshold. If so, it downsamples the
    mask by coarsening before running DBSCAN. After clustering, it calculates
    the centroid and area of each dust plume using a vectorized approach with
    pandas for high efficiency.

    Parameters
    ----------
    dust_mask : xr.DataArray
        The binary dust mask DataArray.
    scn_time : datetime.datetime
        The timestamp of the scene.
    sat_id : str
        The identifier of the satellite.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents a detected
        dust event with its properties (centroid, area, etc.). An empty list
        is returned if no dust pixels are found.
    """
    logger.info("Clustering dust events...")
    # --- Performance Optimization: Downsample large masks ---
    # If the number of dusty pixels is very high, coarsening reduces the
    # point count for DBSCAN, preventing memory issues and speeding up the process.
    pixel_count = int(np.sum(dust_mask).item())
    logger.debug(f"Input dust mask has {pixel_count} pixels.")

    if pixel_count > PIXEL_COUNT_THRESHOLD:
        logger.info(
            f"Pixel count ({pixel_count}) exceeds threshold. Coarsening mask..."
        )
        dust_mask = dust_mask.coarsen(
            dim={"x": COARSEN_FACTOR, "y": COARSEN_FACTOR}, boundary="trim"
        ).max()
        logger.debug(f"Coarsened mask has {int(np.sum(dust_mask).item())} pixels.")

    y_indices, x_indices = np.where(dust_mask.values)
    if y_indices.size == 0:
        logger.info("No dusty pixels found. Returning empty event list.")
        return []

    lats = dust_mask.lat.values[y_indices, x_indices]
    lons = dust_mask.lon.values[y_indices, x_indices]
    coords_deg = np.column_stack((lats, lons))
    coords_rad = np.radians(coords_deg)

    # --- Clustering ---
    # Use Haversine for geographic distances and run in parallel (n_jobs=-1).
    earth_radius_km: float = 6371.0
    eps_km: float = 20.0
    eps_rad: float = eps_km / earth_radius_km
    logger.info("Running DBSCAN clustering with Haversine metric...")
    db = DBSCAN(eps=eps_rad, min_samples=10, metric="haversine", n_jobs=-1).fit(
        coords_rad
    )
    labels: np.ndarray = db.labels_

    # Get number of clusters (ignore noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"DBSCAN identified {n_clusters} clusters.")

    # --- Vectorized Centroid Calculation ---
    # This approach is significantly faster than looping through each cluster.
    # It uses pandas to group all clustered points by their label and
    # calculates the mean lat/lon and size of each group in a single operation.
    logger.debug("Calculating event properties (centroid, area)...")
    df = pd.DataFrame(
        {
            "lat": coords_deg[:, 0],
            "lon": coords_deg[:, 1],
            "label": labels,
        }
    )
    # Filter out noise points (label -1)
    df_clusters = df[df["label"] != -1]

    if df_clusters.empty:
        logger.info("No clusters formed after filtering noise. Returning empty event list.")
        return []

    # Group by cluster label and aggregate
    grouped = df_clusters.groupby("label").agg(
        latitude=("lat", "mean"),
        longitude=("lon", "mean"),
        area_pixels=("lat", "size"),
    )

    # --- Format Output ---
    grouped["datetime"] = scn_time
    grouped["satellite"] = sat_id

    events = grouped.reset_index(drop=True).to_dict("records")
    logger.info(f"Clustering complete. {len(events)} events recorded.")
    # Reset index to turn the 'label' group key into a column, then format as records.
    return events
