"""
Utilities for handling GOES satellite data from AWS S3.
"""
import datetime
import s3fs
from typing import List, Dict

SATELLITE_CONFIG: Dict[str, Dict[str, str]] = {
    "goes16": {"bucket": "noaa-goes16", "product": "ABI-L1b-RadC"},
    "goes17": {"bucket": "noaa-goes17", "product": "ABI-L1b-RadC"},
    "goes18": {"bucket": "noaa-goes18", "product": "ABI-L1b-RadC"},
}

SATELLITE_BANDS: Dict[str, List[str]] = {
    "goes16": ["C11", "C13", "C15"],
    "goes17": ["C11", "C13", "C15"],
    "goes18": ["C11", "C13", "C15"],
}

# Initialize the S3 file system object
fs = s3fs.S3FileSystem(anon=True)


def _parse_s3_timestamp(filename: str) -> datetime.datetime:
    """Extracts the timestamp from a GOES satellite data filename."""
    parts = filename.split('_')
    # Extract the 'sYYYYJJJHHMMSSd' part, where 'd' is tenths of a second
    timestamp_str = parts[3][1:-1]
    return datetime.datetime.strptime(timestamp_str, '%Y%j%H%M%S')


def get_s3_path(sat_id: str, scn_time: datetime.datetime) -> str:
    """
    Finds the S3 path for the GOES file closest to a given timestamp.

    This function searches the S3 directory for the given hour, as well as the
    preceding and succeeding hours, to ensure the chronologically closest file
    is found, especially for timestamps near the edge of an hour.

    Args:
        sat_id: The identifier of the satellite (e.g., 'goes16').
        scn_time: The target timestamp.

    Returns:
        The full S3 path to the closest satellite data file.

    Raises:
        ValueError: If the satellite is not supported.
        FileNotFoundError: If no files are found in any of the searched directories.
    """
    config = SATELLITE_CONFIG.get(sat_id)
    if not config:
        raise ValueError(f"Unsupported satellite: {sat_id}")

    all_files = []
    for hour_offset in [-1, 0, 1]:
        search_time = scn_time + datetime.timedelta(hours=hour_offset)
        bucket = config["bucket"]
        product = config["product"]
        year = search_time.strftime("%Y")
        day_of_year = search_time.strftime("%j")
        hour = search_time.strftime("%H")

        s3_dir = f"s3://{bucket}/{product}/{year}/{day_of_year}/{hour}/"

        try:
            all_files.extend(fs.ls(s3_dir))
        except FileNotFoundError:
            continue

    if not all_files:
        raise FileNotFoundError(f"No files found for {sat_id} around {scn_time}")

    # Find the file with the timestamp closest to scn_time
    closest_file = min(
        all_files,
        key=lambda f: abs(_parse_s3_timestamp(f) - scn_time)
    )

    return f"s3://{closest_file}"
