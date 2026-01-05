"""
Utilities for handling satellite data.
"""

import asyncio
import datetime
from typing import Dict, Any

import s3fs

SATELLITE_METADATA: Dict[str, Dict[str, Any]] = {
    "goes16": {
        "bucket": "noaa-goes16",
        "product": "ABI-L1b-RadC",
        "bands": ["C11", "C13", "C15"],
        "reader": "abi_l1b",
        "is_s3": True,
    },
    "goes17": {
        "bucket": "noaa-goes17",
        "product": "ABI-L1b-RadC",
        "bands": ["C11", "C13", "C15"],
        "reader": "abi_l1b",
        "is_s3": True,
    },
    "goes18": {
        "bucket": "noaa-goes18",
        "product": "ABI-L1b-RadC",
        "bands": ["C11", "C13", "C15"],
        "reader": "abi_l1b",
        "is_s3": True,
    },
    "himawari": {
        "bands": ["B11", "B13", "B15"],
        "reader": "ahi_hsd",
        "is_s3": False,
    },
    "seviri": {
        "bands": ["IR_87", "IR_108", "IR_120"],
        "reader": "seviri_l1b_native",
        "is_s3": False,
    },
    "modis": {
        "bands": ["29", "31", "32"],
        "reader": "modis_l1b",
        "is_s3": False,
    },
}


def get_satellite_metadata(sat_id: str) -> Dict[str, Any]:
    """
    Retrieves metadata for a given satellite ID.

    This function handles variations in satellite names by checking for common
    substrings for certain satellite families.

    Parameters
    ----------
    sat_id : str
        The identifier of the satellite (e.g., 'goes16', 'himawari8').

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the satellite's metadata, or an empty dict
        if the satellite is not supported.
    """
    if sat_id in SATELLITE_METADATA:
        return SATELLITE_METADATA[sat_id]
    if "himawari" in sat_id:
        return SATELLITE_METADATA["himawari"]
    if "seviri" in sat_id:
        return SATELLITE_METADATA["seviri"]
    if "modis" in sat_id:
        return SATELLITE_METADATA["modis"]
    return {}


def _parse_s3_timestamp(filename: str) -> datetime.datetime:
    """Extracts the timestamp from a GOES satellite data filename."""
    parts = filename.split("_")
    timestamp_str = parts[3][1:-1]
    return datetime.datetime.strptime(timestamp_str, "%Y%j%H%M%S")


async def get_s3_path(
    s3: s3fs.S3FileSystem, sat_id: str, scn_time: datetime.datetime
) -> str:
    """
    Asynchronously finds the S3 path for the GOES file closest to a given timestamp.

    This function searches the S3 directory for the given hour, as well as the
    preceding and succeeding hours concurrently, to ensure the chronologically
    closest file is found.

    Args:
        s3: An asynchronous S3FileSystem instance.
        sat_id: The identifier of the satellite (e.g., 'goes16').
        scn_time: The target timestamp.

    Returns:
        The full S3 path to the closest satellite data file.

    Raises:
        ValueError: If the satellite is not supported for S3.
        FileNotFoundError: If no files are found in any of the searched directories.
    """
    config = SATELLITE_METADATA.get(sat_id)
    if not config or "bucket" not in config:
        raise ValueError(f"Unsupported S3 satellite: {sat_id}")

    async def _list_dir(hour_offset: int):
        search_time = scn_time + datetime.timedelta(hours=hour_offset)
        bucket = config["bucket"]
        product = config["product"]
        s3_dir = (
            f"s3://{bucket}/{product}/{search_time.strftime('%Y/%j/%H')}/"
        )
        try:
            return await s3.ls(s3_dir)
        except FileNotFoundError:
            return []

    # Concurrently search the target hour, the one before, and the one after.
    search_tasks = [_list_dir(offset) for offset in [-1, 0, 1]]
    dir_contents = await asyncio.gather(*search_tasks)
    all_files = [item for sublist in dir_contents for item in sublist]

    if not all_files:
        raise FileNotFoundError(f"No files found for {sat_id} around {scn_time}")

    closest_file = min(
        all_files, key=lambda f: abs(_parse_s3_timestamp(f) - scn_time)
    )

    return f"s3://{closest_file}"
