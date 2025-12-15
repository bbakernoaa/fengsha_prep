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

# Use pathlib for more robust path handling
CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config() -> Dict[str, Any]:
    """Loads the configuration from the config.toml file."""
    with open(CONFIG_PATH, "rb") as f:
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


async def get_bnu_data(
    data_type: str, output_dir: str = "bnu_data"
) -> List[Path]:
    """
    Asynchronously retrieves soil data from the BNU soil dataset by downloading
    it concurrently from URLs specified in the configuration file.

    Args:
        data_type: The type of data to retrieve (e.g., 'sand', 'silt', 'clay').
        output_dir: The directory where the downloaded files will be saved.

    Returns:
        A list of file paths for the downloaded data.
    """
    config = load_config()
    urls = config.get("bnu_data", {}).get(f"{data_type}_urls", [])

    if not urls:
        logger.warning(f"No URLs found for data type: {data_type}")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = []
    placeholder_files: List[Path] = []

    async with aiohttp.ClientSession() as session:
        for url in urls:
            filename = url.split("/")[-1]
            filepath = output_path / filename

            # Skip downloading from placeholder URLs, but create dummy files for testing
            if "example.com" in url:
                logger.info(f"Skipping placeholder URL: {url}")
                filepath.write_text(f"This is a dummy file for {filename}")
                placeholder_files.append(filepath)
                continue

            task = asyncio.create_task(_download_file(session, url, filepath))
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    successful_downloads = [res for res in results if res is not None]

    return placeholder_files + successful_downloads


if __name__ == "__main__":
    # Configure basic logging to see the output when the script is run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Example of how to run the asynchronous function
    async def main():
        logger.info("--- Downloading Sand Data ---")
        sand_files = await get_bnu_data("sand")
        logger.info(f"Downloaded sand files: {sand_files}")

        logger.info("\n--- Downloading Silt Data ---")
        silt_files = await get_bnu_data("silt")
        logger.info(f"Downloaded silt files: {silt_files}")

    asyncio.run(main())
