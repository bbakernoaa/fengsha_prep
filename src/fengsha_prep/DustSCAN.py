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
DEFAULT_THRESHOLDS: Dict[str, float] = {
    'diff_12_10': -0.5,
    'diff_10_8': 2.0,
    'temp_10': 280
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_file_pattern(sat_id: str) -> str:
    """Gets the satpy reader name for a given satellite.

    Parameters
    ----------
    sat_id : str
        The identifier of the satellite (e.g., 'goes16', 'himawari8').

    Returns
    -------
    str
        The corresponding satpy reader name.

    Raises
    ------
    ValueError
        If the satellite is not supported.
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
    """Loads and preprocesses satellite data for a single timestamp.

    This function finds the relevant satellite data file(s) for a given time,
    loads them into a Satpy Scene object, loads the required bands, and
    resamples the scene to a common grid. It supports loading GOES data from
    AWS S3 or other satellites from a local filesystem.

    Parameters
    ----------
    scn_time : datetime.datetime
        The timestamp for which to load data.
    sat_id : str
        The identifier of the satellite.

    Returns
    -------
    Optional[Scene]
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
        bands: List[str] = ['B11', 'B13', 'B15']
    elif 'seviri' in sat_id:
        bands = ['IR_87', 'IR_108', 'IR_120']
    else:
        raise NotImplementedError(f"Band mapping for {sat_id} not implemented.")

    scn.load(bands)
    return scn.resample(resampler='native')


def detect_dust(scn: Scene, sat_id: str, thresholds: Dict[str, float]) -> xr.DataArray:
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

    # --- Add Provenance ---
    history_log = (
        f"Dust mask generated at {datetime.datetime.utcnow().isoformat()}Z. "
        f"Satellite: {sat_id}. Thresholds: {thresholds}."
    )
    dust_mask.attrs['history'] = history_log

    return dust_mask


def cluster_events(dust_mask: xr.DataArray, scn_time: datetime.datetime, sat_id: str) -> List[Dict[str, Any]]:
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
    earth_radius_km: float = 6371.0
    eps_km: float = 20.0
    eps_rad: float = eps_km / earth_radius_km

    db = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine').fit(coords_rad)
    labels: np.ndarray = db.labels_
    unique_labels: set = set(labels)

    events: List[Dict[str, Any]] = []
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask: np.ndarray = (labels == k)
        cluster_coords_deg: np.ndarray = coords_deg[class_member_mask]

        lat_mean: float = np.mean(cluster_coords_deg[:, 0])
        lon_mean: float = np.mean(cluster_coords_deg[:, 1])
        area_px: int = len(cluster_coords_deg)

        events.append({
            'datetime': scn_time,
            'latitude': float(lat_mean),
            'longitude': float(lon_mean),
            'area_pixels': int(area_px),
            'satellite': sat_id
        })
    return events


def _process_scene_sync(
    scn: Scene, scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Synchronous (CPU-bound) part of the processing pipeline.

    Parameters
    ----------
    scn : Scene
        The loaded Satpy scene.
    scn_time : datetime.datetime
        The timestamp of the scene.
    sat_id : str
        The identifier of the satellite.
    thresholds : Dict[str, float]
        The thresholds for dust detection.

    Returns
    -------
    List[Dict[str, Any]]
        A list of detected dust events.
    """
    dust_mask = detect_dust(scn, sat_id, thresholds)
    return cluster_events(dust_mask, scn_time, sat_id)


async def dust_scan_pipeline(
    scn_time: datetime.datetime, sat_id: str, thresholds: Dict[str, float]
) -> Optional[List[Dict[str, Any]]]:
    """Orchestrates the processing of a single satellite scene asynchronously.

    This function separates the I/O-bound data loading from the CPU-bound
    data processing, running each in a separate thread to avoid blocking the
    asyncio event loop.

    Parameters
    ----------
    scn_time : datetime.datetime
        The timestamp of the scene to process.
    sat_id : str
        The identifier of the satellite.
    thresholds : Dict[str, float]
        The thresholds for dust detection.

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        A list of detected dust events, or None if an error occurs or no
        data is found.
    """
    try:
        # I/O-bound: Load satellite data. This is a blocking operation,
        # so we run it in a thread pool to avoid stalling the event loop.
        scn = await asyncio.to_thread(load_scene_data, scn_time, sat_id)
        if scn is None:
            # This can happen if no files are found for the given time.
            return None
    except Exception as e:
        logging.error(f"Error loading data for {scn_time}: {e}")
        return None

    try:
        # CPU-bound: Process the loaded data. This is also blocking, so it
        # runs in the thread pool as well.
        return await asyncio.to_thread(
            _process_scene_sync, scn, scn_time, sat_id, thresholds
        )
    except Exception as e:
        logging.error(f"Error processing scene for {scn_time}: {e}")
        return None


async def run_dust_scan_in_period(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    sat_id: str,
    output_csv: str,
    thresholds: Optional[Dict[str, float]] = None,
    concurrency_limit: int = 10
) -> None:
    """Main execution loop for concurrent dust detection over a time period.

    Parameters
    ----------
    start_time : datetime.datetime
        The start of the time range to analyze.
    end_time : datetime.datetime
        The end of the time range to analyze.
    sat_id : str
        The identifier of the satellite (e.g., 'goes16').
    output_csv : str
        Path to save the resulting CSV file.
    thresholds : Optional[Dict[str, float]], optional
        Dictionary of thresholds for dust detection. Uses DEFAULT_THRESHOLDS
        if None, by default None.
    concurrency_limit : int, optional
        The maximum number of concurrent scene processing tasks, by default 10.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    semaphore = asyncio.Semaphore(concurrency_limit)
    all_events: List[Dict[str, Any]] = []
    tasks: List[asyncio.Task] = []
    current_time = start_time

    logging.info(f"Starting analysis for {sat_id} from {start_time} to {end_time}...")

    async def worker(scn_time: datetime.datetime, thresholds_dict: Dict[str, float]):
        """Acquires semaphore and runs the scene processing."""
        async with semaphore:
            logging.info(f"Processing {scn_time}...")
            events = await dust_scan_pipeline(scn_time, sat_id, thresholds_dict)
            if events:
                all_events.extend(events)
                logging.info(f"  Found {len(events)} dust plumes at {scn_time}.")

    while current_time <= end_time:
        task = asyncio.create_task(worker(current_time, thresholds))
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
