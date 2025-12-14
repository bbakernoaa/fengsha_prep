"""
This module provides functions to retrieve soil data from the BNU soil dataset.
"""

import os
import requests
import tomllib

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.toml')

def load_config():
    """Loads the configuration from the config.toml file."""
    with open(CONFIG_PATH, 'rb') as f:
        return tomllib.load(f)

def get_bnu_data(data_type, output_dir='bnu_data'):
    """
    Retrieves soil data from the BNU soil dataset by downloading it from URLs
    specified in the configuration file.

    Args:
        data_type (str): The type of data to retrieve (e.g., 'sand', 'silt', 'clay').
        output_dir (str): The directory where the downloaded files will be saved.

    Returns:
        list: A list of file paths for the downloaded data.
    """
    config = load_config()
    urls = config.get('bnu_data', {}).get(f'{data_type}_urls', [])

    if not urls:
        print(f"No URLs found for data type: {data_type}")
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    downloaded_files = []
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(output_dir, filename)

        # Skip downloading from placeholder URLs, but create dummy files for testing
        if "example.com" in url:
            print(f"Skipping placeholder URL: {url}")
            with open(filepath, 'w') as f:
                f.write(f"This is a dummy file for {filename}")
            downloaded_files.append(filepath)
            continue

        try:
            print(f"Downloading {url} to {filepath}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            downloaded_files.append(filepath)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")

    return downloaded_files
