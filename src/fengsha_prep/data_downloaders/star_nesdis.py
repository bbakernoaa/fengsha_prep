"""
Utility to retrieve and process data from NOAA STAR NESDIS VIIRS BRDF/Albedo.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)


async def fetch_star_nesdis_file_list(url: str) -> list[str]:
    """
    Scrapes the NOAA STAR NESDIS directory listing for NetCDF files.

    Args:
        url: The URL of the directory listing.

    Returns:
        A list of full URLs to the NetCDF files.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()
                # Find all links ending in .nc
                file_links = re.findall(r'href="([^"]+\.nc)"', html)
                return [urljoin(url, link) for link in file_links]
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching file list from {url}: {e}")
        return []


async def download_file(
    url: str, filepath: Path, session: aiohttp.ClientSession | None = None
) -> Path | None:
    """
    Asynchronously downloads a single file.

    Args:
        url: URL to download.
        filepath: Path to save the file.
        session: Optional aiohttp.ClientSession.

    Returns:
        Path to the downloaded file or None if failed.
    """
    managed_session = False
    if session is None:
        session = aiohttp.ClientSession()
        managed_session = True

    try:
        logger.info(f"Downloading {url} to {filepath}...")
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            filepath.write_bytes(content)
            logger.info(f"Download complete: {filepath}")
            return filepath
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading {url}: {e}")
        return None
    finally:
        if managed_session:
            await session.close()


async def download_star_nesdis_files(
    urls: list[str], output_dir: Path, concurrency_limit: int = 5
) -> list[Path]:
    """
    Asynchronously downloads a list of files concurrently.

    Args:
        urls: List of URLs to download.
        output_dir: Directory to save the files.
        concurrency_limit: Maximum number of concurrent downloads.

    Returns:
        List of paths to the downloaded files.
    """
    if not urls:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []

    async def worker(session: aiohttp.ClientSession, url: str, filepath: Path):
        async with semaphore:
            return await download_file(url, filepath, session)

    async with aiohttp.ClientSession() as session:
        for url in urls:
            filename = Path(urlparse(url).path).name
            filepath = output_dir / filename
            task = asyncio.create_task(worker(session, url, filepath))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    return [res for res in results if res is not None]


def load_and_regrid_star_nesdis_data(
    filepath: Path | str, target_resolution: float = 0.027
) -> xr.Dataset:
    """
    Loads a STAR NESDIS NetCDF file, adds coordinates, and regrids to EPSG:4326.

    The input files are assumed to be on a global 21600x43200 1km grid.

    Args:
        filepath: Path to the NetCDF file.
        target_resolution: Target resolution in degrees (default 0.027 ~ 3km).

    Returns:
        Regridded xarray Dataset.
    """
    with xr.open_dataset(filepath) as ds:
        # Expected dimensions for 1km global grid
        expected_rows = 21600
        expected_cols = 43200

        if (
            ds.sizes.get("row") != expected_rows
            or ds.sizes.get("column") != expected_cols
        ):
            logger.warning(
                f"Dataset dimensions {ds.sizes} do not match expected "
                f"({expected_rows}, {expected_cols}). Regridding might be incorrect."
            )

        # Create coordinates based on actual dimensions
        n_rows = ds.sizes.get("row")
        n_cols = ds.sizes.get("column")

        # Centers of pixels, assuming global coverage
        lat = np.linspace(90 - (180 / n_rows) / 2, -90 + (180 / n_rows) / 2, n_rows)
        lon = np.linspace(
            -180 + (360 / n_cols) / 2, 180 - (360 / n_cols) / 2, n_cols
        )

        ds = ds.assign_coords(row=lat, column=lon)
        ds = ds.rename({"row": "lat", "column": "lon"})

        # Set CRS
        ds.rio.write_crs("EPSG:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

        # Regrid using rioxarray
        # We use reproject to change resolution and ensure it's on a clean grid
        ds_regridded = ds.rio.reproject(
            "EPSG:4326",
            resolution=target_resolution,
        )

        # Ensure all variables are float32
        for var in ds_regridded.data_vars:
            if ds_regridded[var].dtype != np.float32:
                ds_regridded[var] = ds_regridded[var].astype(np.float32)

        return ds_regridded


async def process_star_nesdis_data(
    url: str, output_path: Path | str, target_resolution: float = 0.027
) -> Path:
    """
    Full pipeline: download, regrid, save as float32, and cleanup.

    Args:
        url: URL of the file to process.
        output_path: Where to save the regridded file.
        target_resolution: Target resolution in degrees.

    Returns:
        Path to the processed file.
    """
    output_path = Path(output_path)
    filename = Path(urlparse(url).path).name
    temp_file = Path(f"temp_{filename}")

    try:
        # 1. Download
        await download_file(url, temp_file)

        # 2. Load and regrid
        ds_regridded = load_and_regrid_star_nesdis_data(
            temp_file, target_resolution=target_resolution
        )

        # 3. Save as float32 (handled in load_and_regrid_star_nesdis_data)
        ds_regridded.to_netcdf(output_path)
        ds_regridded.close()
        logger.info(f"Saved processed data to {output_path}")

        return output_path
    finally:
        # 4. Cleanup
        if temp_file.exists():
            os.remove(temp_file)
            logger.info(f"Cleaned up temporary file {temp_file}")
