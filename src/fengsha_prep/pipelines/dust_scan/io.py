import asyncio
import datetime
import glob
import logging
from typing import Any

import s3fs
from satpy import Scene

from fengsha_prep.common import satellite


async def _load_scene_from_s3(
    scn_time: datetime.datetime, sat_id: str, meta: dict[str, Any]
) -> Scene | None:
    """Asynchronously loads a single satellite scene from an AWS S3 bucket."""
    s3 = s3fs.S3FileSystem(asynchronous=True, anon=True)
    try:
        s3_path = await satellite.get_s3_path(s3, sat_id, scn_time)
        if not await s3.exists(s3_path):
            logging.debug(f"No S3 file found for {sat_id} at {scn_time}")
            return None

        # Satpy's Scene is blocking, so we run it in a thread.
        # The key is that the file check (`s3.exists`) is non-blocking.
        def _blocking_load():
            scn = Scene(reader=meta["reader"], filenames=[s3_path])
            scn.load(meta["bands"])
            return scn.resample(resampler="native")

        return await asyncio.to_thread(_blocking_load)

    except Exception as e:
        logging.error(f"Error loading GOES data for {scn_time} from S3: {e}")
        return None


def _load_scene_from_local(
    scn_time: datetime.datetime,
    sat_id: str,
    meta: dict[str, Any],
    data_dir: str | None = None,
) -> Scene | None:
    """Loads a single satellite scene from the local filesystem."""
    try:
        search_dir = data_dir if data_dir is not None else "data"
        files = glob.glob(f"{search_dir}/*{scn_time.strftime('%Y%j%H%M')}*.*")
        if not files:
            logging.debug(
                f"No local files found for {sat_id} at {scn_time} in {search_dir}"
            )
            return None

        scn = Scene(filenames=files, reader=meta["reader"])
        scn.load(meta["bands"])
        return scn.resample(resampler="native")
    except Exception as e:
        logging.error(f"Error loading local data for {scn_time}: {e}")
        return None


async def load_scene_data(
    scn_time: datetime.datetime, sat_id: str, data_dir: str | None = None
) -> Scene | None:
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
    data_dir : Optional[str]
        The directory to search for local data files. Defaults to 'data'.

    Returns
    -------
    Optional[Scene]
        A preprocessed Satpy Scene object, or None if no files are found.
    """
    meta = satellite.get_satellite_metadata(sat_id)
    if not meta:
        logging.error(f"Unsupported satellite: {sat_id}")
        return None

    if meta.get("is_s3", False):
        return await _load_scene_from_s3(scn_time, sat_id, meta)
    else:
        # Local file I/O is also blocking, so run it in a thread.
        return await asyncio.to_thread(
            _load_scene_from_local, scn_time, sat_id, meta, data_dir=data_dir
        )
