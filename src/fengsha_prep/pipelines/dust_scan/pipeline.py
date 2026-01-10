import asyncio
import datetime
import logging
from typing import Any

import pandas as pd
from satpy import Scene

from .algorithm import (
    DEFAULT_THRESHOLDS,
    cluster_events,
    detect_dust,
)
from .io import load_scene_data

# Set up a logger for the module
logger = logging.getLogger(__name__)


def _process_scene_sync(
    scn: Scene,
    scn_time: datetime.datetime,
    sat_id: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
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
    scn_time: datetime.datetime,
    sat_id: str,
    thresholds: dict[str, float],
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Orchestrates the processing of a single satellite scene.

    This function separates I/O-bound data loading from CPU-bound data
    processing, running each in a separate thread to avoid blocking the
    asyncio event loop.

    Returns
    -------
    List[Dict[str, Any]]
        A list of detected dust events, or an empty list if an error occurs
        or no data is found.
    """
    try:
        # Await the natively async data loading function.
        scn = await load_scene_data(scn_time, sat_id, data_dir)
        if scn is None:
            # This is an expected outcome if no files are found.
            logger.debug(f"No data available for {scn_time}, skipping.")
            return []

        # CPU-bound: Process the loaded data.
        events = await asyncio.to_thread(
            _process_scene_sync, scn, scn_time, sat_id, thresholds
        )
        return events

    except Exception:
        # Catch any unexpected errors during the entire process.
        logger.exception(f"Failed to process scene for {scn_time}")
        return []


async def run_dust_scan_in_period(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    sat_id: str,
    output_csv: str,
    thresholds: dict[str, float] | None = None,
    concurrency_limit: int = 10,
    data_dir: str | None = None,
) -> None:
    """
    Main execution loop for concurrent dust detection over a time period.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks: list[asyncio.Task] = []
    current_time = start_time

    logger.info(
        "Scanning %s from %s to %s with concurrency %d...",
        sat_id,
        start_time,
        end_time,
        concurrency_limit,
    )

    async def worker(scn_time: datetime.datetime):
        async with semaphore:
            return await dust_scan_pipeline(
                scn_time, sat_id, thresholds, data_dir=data_dir
            )

    while current_time <= end_time:
        tasks.append(asyncio.create_task(worker(current_time)))
        current_time += datetime.timedelta(minutes=15)

    # Process tasks as they complete to keep memory usage flat
    all_events: list[dict[str, Any]] = []
    for future in asyncio.as_completed(tasks):
        events = await future
        if events:
            all_events.extend(events)
            logger.info(f"Found {len(events)} dust plumes at {events[0]['datetime']}.")

    if all_events:
        df = pd.DataFrame(all_events).sort_values(by="datetime").reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"✅ Success! Saved {len(df)} total events to {output_csv}")
    else:
        logger.info("No dust events detected in this period.")
