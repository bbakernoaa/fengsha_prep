import asyncio
import datetime
import glob
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from satpy import Scene
from sklearn.cluster import DBSCAN

# ==========================================
# CONFIGURATION
# ==========================================
SAT_ID = 'goes16'
REGION = 'meso'
START_TIME = datetime.datetime(2023, 4, 1, 18, 0)
END_TIME = datetime.datetime(2023, 4, 1, 20, 0)
OUTPUT_CSV = 'dust_events_catalog.csv'
THRESHOLDS = {
    'diff_12_10': -0.5,
    'diff_10_8': 2.0,
    'temp_10': 280
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_file_pattern(sat_id: str) -> str:
    """
    Gets the satpy reader name for a given satellite.

    Args:
        sat_id: The identifier of the satellite (e.g., 'goes16').

    Returns:
        The corresponding satpy reader name.

    Raises:
        ValueError: If the satellite is not supported.
    """
    if 'goes' in sat_id:
        return 'abi_l1b'
    elif 'himawari' in sat_id:
        return 'ahi_hsd'
    elif 'seviri' in sat_id:
        return 'seviri_l1b_native'
    else:
        raise ValueError("Unsupported satellite.")


def load_scene_data(scn_time: datetime.datetime, sat_id: str) -> Optional[Scene]:
    """
    Loads and preprocesses satellite data for a single timestamp.

    This function finds the relevant satellite data file(s) for a given time,
    loads them into a Satpy Scene object, loads the required bands, and
    resamples the scene to a common grid.

    Args:
        scn_time: The timestamp for which to load data.
        sat_id: The identifier of the satellite.

    Returns:
        A preprocessed Satpy Scene object, or None if no files are found.
    """
    files = glob.glob(f'data/*{scn_time.strftime("%Y%j%H%M")}*.nc')
    if not files:
        logging.warning(f"No files found for {scn_time}")
        return None

    reader = get_file_pattern(sat_id)
    scn = Scene(filenames=files, reader=reader)

    if 'goes' in sat_id:
        bands = ['C11', 'C13', 'C15']
    elif 'himawari' in sat_id:
        bands = ['B11', 'B13', 'B15']
    else:
        raise NotImplementedError(f"Band mapping for {sat_id} not implemented.")

    scn.load(bands)
    return scn.resample(resampler='native')


def detect_dust(scn: Scene, sat_id: str, thresholds: Dict[str, float]) -> xr.DataArray:
    """
    Applies a physical algorithm to detect dust in a scene.

    This algorithm is based on the brightness temperature differences between
    infrared channels, a common technique for identifying dust aerosols.

    Args:
        scn: The Satpy Scene object containing the satellite data.
        sat_id: The identifier of the satellite.
        thresholds: A dictionary of thresholds used for dust detection.

    Returns:
        An xarray DataArray representing the binary dust mask (1=Dust, 0=No Dust).
    """
    if 'goes' in sat_id:
        b08 = scn['C11']
        b10 = scn['C13']
        b12 = scn['C15']
    else:
        raise NotImplementedError(f"Dust detection for {sat_id} not implemented.")

    diff_12_10 = b12 - b10
    diff_10_8 = b10 - b08

    dust_mask = (
            (diff_12_10 < thresholds['diff_12_10']) &
            (diff_10_8 > thresholds['diff_10_8']) &
            (b10 > thresholds['temp_10'])
    )
    return dust_mask


def cluster_events(dust_mask: xr.DataArray, scn_time: datetime.datetime, sat_id: str) -> List[Dict[str, Any]]:
    """
    Identifies distinct dust plumes from a dust mask using DBSCAN.

    This function takes a binary dust mask, extracts the coordinates of the
    dusty pixels, and uses the DBSCAN clustering algorithm to group them into
    distinct events or plumes.

    Args:
        dust_mask: The binary dust mask DataArray.
        scn_time: The timestamp of the scene.
        sat_id: The identifier of the satellite.

    Returns:
        A list of dictionaries, where each dictionary represents a detected
        dust event with its properties (centroid, area, etc.).
    """
    valid_pixels = dust_mask.where(dust_mask, drop=True)
    if valid_pixels.size == 0:
        return []

    # This is the critical fix: use geographic lat/lon coordinates for clustering.
    lats = valid_pixels.lat.values
    lons = valid_pixels.lon.values
    coords_deg = np.column_stack((lats, lons))

    # For accurate geographic clustering, we use the haversine metric, which
    # requires coordinates in radians.
    coords_rad = np.radians(coords_deg)

    # DBSCAN eps is the search radius. For haversine, it's in radians.
    # We want a radius of ~5km. Earth's radius is ~6371 km.
    # eps = 5 km / 6371 km
    earth_radius_km = 6371
    eps_km = 5
    eps_rad = eps_km / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine').fit(coords_rad)
    labels = db.labels_
    unique_labels = set(labels)

    events = []
    for k in unique_labels:
        if k == -1:  # -1 is noise in DBSCAN
            continue

        class_member_mask = (labels == k)
        # We use the original degree coordinates for calculating the centroid
        cluster_coords_deg = coords_deg[class_member_mask]

        # Calculate the centroid of the cluster in degrees
        lat_mean = np.mean(cluster_coords_deg[:, 0])
        lon_mean = np.mean(cluster_coords_deg[:, 1])
        area_px = len(cluster_coords_deg)

        events.append({
            'datetime': scn_time,
            'latitude': float(lat_mean),
            'longitude': float(lon_mean),
            'area_pixels': int(area_px),
            'satellite': sat_id
        })
    return events


async def process_scene(scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]) -> Optional[List[Dict[str, Any]]]:
    """
    Orchestrates the processing of a single satellite scene asynchronously.

    This function handles the loading of data, detection of dust, and
    clustering of dust events for a single timestamp. Heavy lifting is done
    in a separate thread to avoid blocking the asyncio event loop.

    Args:
        scn_time: The timestamp of the scene to process.
        sat_id: The identifier of the satellite.
        thresholds: A dictionary of thresholds for dust detection.

    Returns:
        A list of detected dust events, or None if an error occurs.
    """
    try:
        # Run synchronous, CPU-bound code in a separate thread
        return await asyncio.to_thread(
            _process_scene_sync, scn_time, sat_id, thresholds
        )
    except Exception as e:
        logging.error(f"Error processing {scn_time}: {e}")
        return None


def _process_scene_sync(
    scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]
) -> Optional[List[Dict[str, Any]]]:
    """Synchronous helper function for scene processing."""
    scn = load_scene_data(scn_time, sat_id)
    if scn is None:
        return None

    dust_mask = detect_dust(scn, sat_id, thresholds)
    events = cluster_events(dust_mask, scn_time, sat_id)
    return events


async def main() -> None:
    """
    Main execution loop for concurrent dust detection.

    This function iterates through a time range, creates concurrent tasks for
    processing each satellite scene, and saves the detected dust events to a CSV.
    It uses a semaphore to limit the number of concurrent processes to avoid
    overwhelming the system.
    """
    # Create a semaphore to limit concurrency to a reasonable number
    semaphore = asyncio.Semaphore(10)
    all_events: List[Dict[str, Any]] = []
    tasks = []
    current_time = START_TIME
    logging.info(f"Starting analysis for {SAT_ID} with bounded concurrency...")

    async def worker(scn_time: datetime.datetime):
        """Acquires semaphore and runs the scene processing."""
        async with semaphore:
            logging.info(f"Processing {scn_time}...")
            events = await process_scene(scn_time, SAT_ID, THRESHOLDS)
            if events:
                all_events.extend(events)
                logging.info(f"  Found {len(events)} dust plumes at {scn_time}.")

    while current_time <= END_TIME:
        task = asyncio.create_task(worker(current_time))
        tasks.append(task)
        current_time += datetime.timedelta(minutes=15)

    await asyncio.gather(*tasks)

    if all_events:
        df = pd.DataFrame(all_events)
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Success! Saved {len(df)} events to {OUTPUT_CSV}")
    else:
        logging.info("No dust events detected in this period.")


if __name__ == "__main__":
    asyncio.run(main())
