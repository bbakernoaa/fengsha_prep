import datetime
from typing import Any

import numpy as np
import xarray as xr
from satpy import Scene
from sklearn.cluster import DBSCAN

from fengsha_prep.common import satellite

# Default thresholds for dust detection, can be overridden.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "diff_12_10": -0.5,
    "diff_10_8": 2.0,
    "temp_10": 280,
}


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
    meta = satellite.get_satellite_metadata(sat_id)
    if not meta or "bands" not in meta or len(meta["bands"]) < 3:
        raise ValueError(f"Invalid band configuration for {sat_id}")

    bands = meta["bands"]
    b08, b10, b12 = scn[bands[0]], scn[bands[1]], scn[bands[2]]

    diff_12_10 = b12 - b10
    diff_10_8 = b10 - b08

    dust_mask = (
        (diff_12_10 < thresholds["diff_12_10"])
        & (diff_10_8 > thresholds["diff_10_8"])
        & (b10 > thresholds["temp_10"])
    )

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

    This function takes a binary dust mask, extracts the coordinates of the
    dusty pixels, and uses the DBSCAN clustering algorithm to group them into
    distinct events or plumes based on geographic proximity.

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
        dust event with its properties (centroid, area, etc.).
    """
    # More memory-efficient way to get dusty pixel coordinates, avoiding
    # the creation of large intermediate xarray objects.
    y_indices, x_indices = np.where(dust_mask.values)

    if y_indices.size == 0:
        return []

    # Extract the lat/lon coordinates for the dusty pixels directly.
    # Assumes 'lat' and 'lon' are 2D coordinates in the DataArray.
    lats = dust_mask.lat.values[y_indices, x_indices]
    lons = dust_mask.lon.values[y_indices, x_indices]
    coords_deg = np.column_stack((lats, lons))

    # For accurate geographic clustering, we use the haversine metric, which
    # requires coordinates in radians.
    coords_rad = np.radians(coords_deg)

    # DBSCAN eps is the search radius. For haversine, it's in radians.
    # We want a radius of ~20km to ensure plume continuity. Earth's radius is ~6371 km.
    # eps = 20 km / 6371 km
    earth_radius_km: float = 6371.0
    eps_km: float = 20.0
    eps_rad: float = eps_km / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=10, metric="haversine").fit(coords_rad)
    labels: np.ndarray = db.labels_
    unique_labels: set = set(labels)

    events: list[dict[str, Any]] = []
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask: np.ndarray = labels == k
        cluster_coords_deg: np.ndarray = coords_deg[class_member_mask]

        lat_mean: float = np.mean(cluster_coords_deg[:, 0])
        lon_mean: float = np.mean(cluster_coords_deg[:, 1])
        area_px: int = len(cluster_coords_deg)

        events.append(
            {
                "datetime": scn_time,
                "latitude": float(lat_mean),
                "longitude": float(lon_mean),
                "area_pixels": int(area_px),
                "satellite": sat_id,
            }
        )
    return events
