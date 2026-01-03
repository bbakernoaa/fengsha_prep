"""
This module provides functions to retrieve soil data from the BNU soil dataset.
"""

import asyncio
import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

# Set up a logger for the module
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads the configuration from a TOML file.
    If a path is provided, it's used. Otherwise, it defaults to the
    internal config.toml file.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


async def _download_file(
    session: aiohttp.ClientSession, url: str, filepath: Path
) -> Optional[Path]:
    """
    Asynchronously downloads a single file and saves it.

    Returns the filepath on success, None on failure.
    """
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


async def _download_files_concurrently(
    urls: List[str], output_dir: Path, concurrency_limit: int
) -> List[Path]:
    """
    Asynchronously downloads a list of files concurrently.

    This is the core, testable download logic, decoupled from configuration.

    Args:
        urls: A list of URLs to download.
        output_dir: The directory where the downloaded files will be saved.
        concurrency_limit: The maximum number of concurrent downloads.

    Returns:
        A list of file paths for the downloaded data.
    """
    if not urls:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []

    async def worker(session: aiohttp.ClientSession, url: str, filepath: Path):
        """Acquires semaphore and runs the download task."""
        async with semaphore:
            return await _download_file(session, url, filepath)

    async with aiohttp.ClientSession() as session:
        for url in urls:
            filename = url.split("/")[-1]
            filepath = output_dir / filename
            task = asyncio.create_task(worker(session, url, filepath))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
    successful_downloads = [res for res in results if res is not None]
    return successful_downloads


async def get_bnu_data_async(
    data_type: str,
    output_dir: str = "bnu_data",
    concurrency_limit: int = 10,
    config_path: Optional[Path] = None,
) -> List[Path]:
    """
    Asynchronously retrieves soil data from the BNU soil dataset by downloading
    it concurrently from URLs specified in the configuration file.

    This function acts as a configuration-aware wrapper around the core
    download logic.

    Args:
        data_type: The type of data to retrieve (e.g., 'sand', 'silt', 'clay').
        output_dir: The directory where the downloaded files will be saved.
        concurrency_limit: The maximum number of concurrent downloads.
        config_path: Optional path to a custom configuration file.

    Returns:
        A list of file paths for the downloaded data.
    """
    config = load_config(config_path)
    urls = config.get("bnu_data", {}).get(f"{data_type}_urls", [])

    if not urls:
        logger.warning(f"No URLs found for data type: {data_type}")
        return []

    return await _download_files_concurrently(
        urls=urls,
        output_dir=Path(output_dir),
        concurrency_limit=concurrency_limit,
    )


def get_bnu_data(
    data_type: str,
    output_dir: str = "bnu_data",
    concurrency_limit: int = 10,
    config_path: Optional[Path] = None,
) -> List[Path]:
    """
    Retrieves soil data from the BNU soil dataset by downloading it
    concurrently from URLs specified in the configuration file.

    This is a synchronous wrapper around the asynchronous `get_bnu_data_async`
    function.

    Args:
        data_type: The type of data to retrieve (e.g., 'sand', 'silt', 'clay').
        output_dir: The directory where the downloaded files will be saved.
        concurrency_limit: The maximum number of concurrent downloads.
        config_path: Optional path to a custom configuration file.

    Returns:
        A list of file paths for the downloaded data.
    """
    return asyncio.run(
        get_bnu_data_async(data_type, output_dir, concurrency_limit, config_path)
    )


if __name__ == "__main__":
    # Configure basic logging to see the output when the script is run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Example of how to run the synchronous function
    logger.info("--- Downloading Sand Data ---")
    # Example with default concurrency limit
    sand_files = get_bnu_data("sand")
    logger.info(f"Downloaded sand files: {sand_files}")

    logger.info("\n--- Downloading Silt Data (with a limit of 5) ---")
    # Example with a custom concurrency limit
    silt_files = get_bnu_data("silt", concurrency_limit=5)
    logger.info(f"Downloaded silt files: {silt_files}")
