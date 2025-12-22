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

from . import goes_s3

# Default thresholds for dust detection, can be overridden.
DEFAULT_THRESHOLDS = {
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
    if sat_id in goes_s3.SATELLITE_CONFIG:
        s3_path = None
        try:
            s3_path = goes_s3.get_s3_path(sat_id, scn_time)
            bands = goes_s3.SATELLITE_BANDS.get(sat_id)
            if not bands:
                raise ValueError(f"Band mapping for {sat_id} not implemented.")

            # Satpy automatically uses s3fs for s3:// paths
            scn = Scene(reader="abi_l1b", filenames=[s3_path])
            scn.load(bands)
            return scn.resample(resampler='native')
        except FileNotFoundError:
            log_msg = f"No files found on S3 for {scn_time}"
            if s3_path:
                log_msg += f" at {s3_path}"
            logging.warning(log_msg)
            return None
        except Exception as e:
            logging.error(f"Error loading GOES data for {scn_time}: {e}")
            return None

    # Fallback to local file glob for other satellites
    files = glob.glob(f'data/*{scn_time.strftime("%Y%j%H%M")}*.nc')
    if not files:
        logging.warning(f"No local files found for {scn_time}")
        return None

    reader = get_file_pattern(sat_id)
    scn = Scene(filenames=files, reader=reader)

    if 'himawari' in sat_id:
        bands = ['B11', 'B13', 'B15']
    elif 'seviri' in sat_id:
        bands = ['IR_87', 'IR_108', 'IR_120']
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
    if sat_id in goes_s3.SATELLITE_CONFIG:
        bands = goes_s3.SATELLITE_BANDS.get(sat_id)
        if not bands or len(bands) < 3:
            raise ValueError(f"Invalid band configuration for {sat_id}")
        b08, b10, b12 = scn[bands[0]], scn[bands[1]], scn[bands[2]]
    elif 'himawari' in sat_id:
        b08, b10, b12 = scn['B11'], scn['B13'], scn['B15']
    elif 'seviri' in sat_id:
        b08, b10, b12 = scn['IR_87'], scn['IR_108'], scn['IR_120']
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
    # Stack the spatial dimensions and drop non-dusty pixels to get a clean
    # list of coordinates. This is robust to non-contiguous dust plumes.
    stacked_mask = dust_mask.stack(points=('y', 'x'))
    valid_pixels = stacked_mask.where(stacked_mask, drop=True)

    if valid_pixels.size == 0:
        return []

    lats = valid_pixels.lat.values
    lons = valid_pixels.lon.values
    coords_deg = np.column_stack((lats, lons))

    # For accurate geographic clustering, we use the haversine metric, which
    # requires coordinates in radians.
    coords_rad = np.radians(coords_deg)

    # DBSCAN eps is the search radius. For haversine, it's in radians.
    # We want a radius of ~20km to ensure plume continuity. Earth's radius is ~6371 km.
    # eps = 20 km / 6371 km
    earth_radius_km = 6371
    eps_km = 20
    eps_rad = eps_km / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine').fit(coords_rad)
    labels = db.labels_
    unique_labels = set(labels)

    events = []
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask = (labels == k)
        cluster_coords_deg = coords_deg[class_member_mask]

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


def dust_scan_pipeline(
    scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]
) -> Optional[List[Dict[str, Any]]]:
    """Synchronous pipeline for scene processing."""
    scn = load_scene_data(scn_time, sat_id)
    if scn is None:
        return None

    dust_mask = detect_dust(scn, sat_id, thresholds)
    events = cluster_events(dust_mask, scn_time, sat_id)
    return events


async def process_scene(scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]) -> Optional[List[Dict[str, Any]]]:
    """
    Orchestrates the processing of a single satellite scene asynchronously.
    """
    try:
        return await asyncio.to_thread(
            dust_scan_pipeline, scn_time, sat_id, thresholds
        )
    except Exception as e:
        logging.error(f"Error processing {scn_time}: {e}")
        return None


async def run_dust_scan_in_period(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    sat_id: str,
    output_csv: str,
    thresholds: Dict[str, float] = None,
    concurrency_limit: int = 10
) -> None:
    """
    Main execution loop for concurrent dust detection over a time period.

    Args:
        start_time: The start of the time range to analyze.
        end_time: The end of the time range to analyze.
        sat_id: The identifier of the satellite (e.g., 'goes16').
        output_csv: Path to save the resulting CSV file.
        thresholds: Dictionary of thresholds for dust detection. Uses DEFAULT_THRESHOLDS if None.
        concurrency_limit: The maximum number of concurrent scene processing tasks.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    semaphore = asyncio.Semaphore(concurrency_limit)
    all_events: List[Dict[str, Any]] = []
    tasks = []
    current_time = start_time

    logging.info(f"Starting analysis for {sat_id} from {start_time} to {end_time}...")

    async def worker(scn_time: datetime.datetime):
        """Acquires semaphore and runs the scene processing."""
        async with semaphore:
            logging.info(f"Processing {scn_time}...")
            events = await process_scene(scn_time, sat_id, thresholds)
            if events:
                all_events.extend(events)
                logging.info(f"  Found {len(events)} dust plumes at {scn_time}.")

    while current_time <= end_time:
        task = asyncio.create_task(worker(current_time))
        tasks.append(task)
        current_time += datetime.timedelta(minutes=15)

    await asyncio.gather(*tasks)

    if all_events:
        df = pd.DataFrame(all_events)
        df.to_csv(output_csv, index=False)
        logging.info(f"Success! Saved {len(df)} events to {output_csv}")
    else:
        logging.info("No dust events detected in this period.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Scan satellite data for dust events.")
    parser.add_argument(
        '--sat',
        type=str,
        default='goes16',
        help="Satellite ID (e.g., 'goes16')."
    )
    parser.add_argument(
        '--start',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M'),
        required=True,
        help="Start time in YYYY-MM-DDTHH:MM format."
    )
    parser.add_argument(
        '--end',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M'),
        required=True,
        help="End time in YYYY-MM-DDTHH:MM format."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dust_events.csv',
        help="Output CSV file path."
    )
    args = parser.parse_args()

    asyncio.run(run_dust_scan_in_period(
        start_time=args.start,
        end_time=args.end,
        sat_id=args.sat,
        output_csv=args.output
    ))
