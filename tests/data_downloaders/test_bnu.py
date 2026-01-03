import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fengsha_prep.data_downloaders.bnu import (
    _download_files_concurrently,
    get_bnu_data,
    get_bnu_data_async,
)


@patch("fengsha_prep.data_downloaders.bnu.load_config")
@patch(
    "fengsha_prep.data_downloaders.bnu._download_files_concurrently",
    new_callable=AsyncMock,
)
def test_get_bnu_data_success(
    mock_download_concurrently: AsyncMock, mock_load_config, tmp_path: Path
):
    """
    Verify `get_bnu_data` (sync) calls download logic with correct params.
    """
    # Arrange
    data_type = "sand"
    output_dir = tmp_path / "bnu_output"
    mock_urls = ["http://example.com/sand1.nc", "http://example.com/sand2.nc"]
    mock_load_config.return_value = {"bnu_data": {f"{data_type}_urls": mock_urls}}
    expected_paths = [output_dir / url.split("/")[-1] for url in mock_urls]
    mock_download_concurrently.return_value = expected_paths

    # Act
    result = get_bnu_data(data_type, output_dir=str(output_dir))

    # Assert
    mock_load_config.assert_called_once()
    # The sync wrapper calls the async function, which calls the downloader.
    # So we check if the downloader was awaited.
    mock_download_concurrently.assert_awaited_once_with(
        urls=mock_urls,
        output_dir=Path(output_dir),
        concurrency_limit=10,  # Default value
    )
    assert result == expected_paths


@patch("fengsha_prep.data_downloaders.bnu.load_config")
@patch(
    "fengsha_prep.data_downloaders.bnu._download_files_concurrently",
    new_callable=AsyncMock,
)
def test_get_bnu_data_no_urls(
    mock_download_concurrently: AsyncMock, mock_load_config
):
    """
    Verify `get_bnu_data` (sync) returns empty list if no URLs are found.
    """
    # Arrange
    data_type = "unobtanium"
    # Return a config that is valid but doesn't contain the requested data_type
    mock_load_config.return_value = {"bnu_data": {"sand_urls": ["some_url"]}}

    # Act
    result = get_bnu_data(data_type)

    # Assert
    mock_load_config.assert_called_once()
    mock_download_concurrently.assert_not_awaited()
    assert result == []


@pytest.mark.asyncio
@patch("fengsha_prep.data_downloaders.bnu.load_config")
@patch(
    "fengsha_prep.data_downloaders.bnu._download_files_concurrently",
    new_callable=AsyncMock,
)
async def test_get_bnu_data_async_success(
    mock_download_concurrently: AsyncMock, mock_load_config, tmp_path: Path
):
    """
    Verify `get_bnu_data_async` calls download logic with correct params.
    """
    # Arrange
    data_type = "silt"
    output_dir = tmp_path / "bnu_output_async"
    mock_urls = ["http://example.com/silt1.nc"]
    mock_load_config.return_value = {"bnu_data": {f"{data_type}_urls": mock_urls}}
    expected_paths = [output_dir / url.split("/")[-1] for url in mock_urls]
    mock_download_concurrently.return_value = expected_paths

    # Act
    result = await get_bnu_data_async(data_type, output_dir=str(output_dir))

    # Assert
    mock_load_config.assert_called_once()
    mock_download_concurrently.assert_awaited_once_with(
        urls=mock_urls,
        output_dir=Path(output_dir),
        concurrency_limit=10,
    )
    assert result == expected_paths


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


@pytest.mark.asyncio
async def test_download_files_respects_concurrency_limit(tmp_path: Path):
    """
    Verify that the download function correctly limits the number of
    concurrent operations.
    """
    # Arrange
    urls = [f"http://example.com/file{i}" for i in range(10)]
    output_dir = tmp_path / "downloads"
    concurrency_limit = 3
    active_tasks = 0
    max_active_tasks = 0
    lock = asyncio.Lock()
    pause_event = asyncio.Event()

    async def mock_download_side_effect(session, url, filepath):
        nonlocal active_tasks, max_active_tasks
        async with lock:
            active_tasks += 1
            max_active_tasks = max(max_active_tasks, active_tasks)
        await pause_event.wait()
        async with lock:
            active_tasks -= 1
        filepath.touch()
        return filepath

    # Act
    with patch(
        "fengsha_prep.data_downloaders.bnu._download_file",
        new_callable=AsyncMock,
        side_effect=mock_download_side_effect,
    ) as mock_download:
        # Create a task to run the download function in the background
        download_task = asyncio.create_task(
            _download_files_concurrently(urls, output_dir, concurrency_limit)
        )
        # Give the event loop a moment to start the tasks
        await asyncio.sleep(0.1)
        # At this point, the first batch of tasks should be running and paused
        # Un-pause the tasks to let them complete
        pause_event.set()
        # Wait for the main download task to finish
        await download_task

    # Assert
    assert max_active_tasks == concurrency_limit
    assert mock_download.call_count == len(urls)
