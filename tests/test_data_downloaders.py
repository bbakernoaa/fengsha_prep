
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fengsha_prep.data_downloaders.bnu import (
    _download_files_concurrently,
    bnu_downloader,
)


@pytest.mark.asyncio
async def test_download_files_concurrently_independent_of_config():
    """
    Tests that the core download function can be called with a simple list of
    URLs and is independent of any configuration file.
    """
    # Arrange
    urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
    output_dir = "test_output"
    mock_write_bytes = MagicMock()

    # 1. Mock the response object that the `get` context manager yields.
    #    It has an async `read` method and a sync `raise_for_status` method.
    mock_response = AsyncMock()
    mock_response.read.return_value = b"file content"
    mock_response.raise_for_status = MagicMock()

    # 2. Mock the async context manager that `session.get()` returns.
    get_context_manager = AsyncMock()
    get_context_manager.__aenter__.return_value = mock_response

    # 3. Mock the session instance.
    mock_session_instance = AsyncMock()
    #    `get` is a regular method that returns the context manager.
    mock_session_instance.get = MagicMock(return_value=get_context_manager)

    # 4. Mock the `aiohttp.ClientSession` factory, which is also an async context manager.
    mock_session_factory = AsyncMock()
    mock_session_factory.__aenter__.return_value = mock_session_instance

    # Act
    with patch("aiohttp.ClientSession", return_value=mock_session_factory):
        with patch("pathlib.Path.write_bytes", mock_write_bytes):
            downloaded_files = await _download_files_concurrently(
                urls, output_dir, concurrency_limit=2
            )

    # Assert
    assert len(downloaded_files) == 2
    for i, path in enumerate(downloaded_files):
        assert isinstance(path, Path)
        assert path.name == f"file{i+1}.txt"

    # Verify that the session was used to download the files
    assert mock_session_instance.get.call_count == 2
    mock_session_instance.get.assert_any_call(urls[0])
    mock_session_instance.get.assert_any_call(urls[1])

    # Verify that write_bytes was called for each file
    assert mock_write_bytes.call_count == 2


@pytest.mark.asyncio
async def test_bnu_downloader_loads_config_and_calls_downloader():
    """
    Tests that the bnu_downloader wrapper function correctly loads the
    configuration and calls the downloader.
    """
    # Arrange
    mock_config = {
        "bnu_data": {
            "sand_urls": [
                "http://example.com/sand1.txt",
                "http://example.com/sand2.txt",
            ]
        }
    }
    expected_urls = mock_config["bnu_data"]["sand_urls"]
    output_dir = "test_bnu_data"
    concurrency_limit = 5

    # Mock the configuration loader and the downloader
    with patch(
        "fengsha_prep.data_downloaders.bnu.load_config",
        return_value=mock_config,
    ) as mock_load_config:
        with patch(
            "fengsha_prep.data_downloaders.bnu._download_files_concurrently",
            new_callable=AsyncMock,
        ) as mock_downloader:
            # Act
            await bnu_downloader("sand", output_dir, concurrency_limit)

            # Assert
            mock_load_config.assert_called_once()
            mock_downloader.assert_awaited_once_with(
                expected_urls, output_dir, concurrency_limit
            )
