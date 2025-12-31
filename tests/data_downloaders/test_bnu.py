
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fengsha_prep.data_downloaders.bnu import _download_files_concurrently


@pytest.mark.asyncio
async def test_download_files_concurrently_success(tmp_path: Path):
    """
    Verify that `_download_files_concurrently` successfully "downloads"
    files to the specified directory.
    """
    # Arrange
    urls = [
        "http://example.com/file1.txt",
        "http://example.com/file2.txt",
        "http://example.com/file3.txt",
    ]
    output_dir = tmp_path / "downloads"
    expected_files = [output_dir / "file1.txt", output_dir / "file2.txt", output_dir / "file3.txt"]

    # Act
    # We patch the `_download_file` function to avoid actual downloads and to control the return value
    with patch(
        "fengsha_prep.data_downloaders.bnu._download_file", new_callable=AsyncMock
    ) as mock_download:
        # We need to make sure the mock returns the path so the final list is populated
        async def side_effect(session, url, filepath):
            filepath.touch()  # Create a dummy file to check for existence
            return filepath

        mock_download.side_effect = side_effect
        result = await _download_files_concurrently(urls, output_dir, concurrency_limit=5)

    # Assert
    assert len(result) == len(urls)
    assert all(file.exists() for file in expected_files)
    assert set(result) == set(expected_files)
    assert mock_download.call_count == len(urls)


@pytest.mark.asyncio
async def test_download_files_concurrently_partial_failure(tmp_path: Path):
    """
    Verify that `_download_files_concurrently` returns only the paths of
    successfully downloaded files when some downloads fail.
    """
    # Arrange
    urls = [
        "http://example.com/success1.txt",
        "http://example.com/failure.txt",
        "http://example.com/success2.txt",
    ]
    output_dir = tmp_path / "downloads"
    successful_files = [output_dir / "success1.txt", output_dir / "success2.txt"]

    # Mock the _download_file to simulate a failure for a specific URL
    async def selective_download(session, url, filepath):
        if "failure" in url:
            return None  # Simulate a download failure
        filepath.touch()
        return filepath

    # Act
    with patch(
        "fengsha_prep.data_downloaders.bnu._download_file", new=selective_download
    ):
        result = await _download_files_concurrently(urls, output_dir, concurrency_limit=5)

    # Assert
    assert len(result) == 2
    assert set(result) == set(successful_files)
    assert all(file.exists() for file in successful_files)
    assert not (output_dir / "failure.txt").exists()


@pytest.mark.asyncio
async def test_download_files_concurrently_no_urls(tmp_path: Path):
    """
    Verify that `_download_files_concurrently` returns an empty list
    when given no URLs.
    """
    # Arrange
    urls = []
    output_dir = tmp_path / "downloads"

    # Act
    result = await _download_files_concurrently(urls, output_dir, concurrency_limit=5)

    # Assert
    assert result == []
    assert not output_dir.exists()
