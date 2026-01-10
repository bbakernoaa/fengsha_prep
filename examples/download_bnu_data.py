"""
Example script for downloading BNU soil data.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing the main package
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.fengsha_prep.data_downloaders.bnu import get_bnu_data  # noqa: E402

# Configure basic logging to see the output from the downloader and this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrates how to use the get_bnu_data function."""
    logger.info("--- Downloading Sand Data ---")
    # Example with default concurrency limit
    sand_files = get_bnu_data("sand")
    logger.info(f"Downloaded sand files: {sand_files}")

    logger.info("\n--- Downloading Silt Data (with a limit of 5) ---")
    # Example with a custom concurrency limit
    silt_files = get_bnu_data("silt", concurrency_limit=5)
    logger.info(f"Downloaded silt files: {silt_files}")


if __name__ == "__main__":
    main()
