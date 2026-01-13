"""
Tests for the I/O operations of the uthresh pipeline.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import s3fs
import xarray as xr

from fengsha_prep.pipelines.uthresh.io import (
    AsyncDustDataEngine,
    DustDataEngine,
)


# --- Tests for the Synchronous Wrapper ---


@pytest.mark.asyncio
async def test_dust_data_engine_context_manager():
    """
    Tests that the synchronous DustDataEngine works as a context manager.
    This test needs `@pytest.mark.asyncio` to provide a running event loop
    for the underlying `aiohttp.ClientSession`.
    """
    # We need to properly close the session created inside the context manager
    # To do this without real network calls, we can patch the session's close method
    with patch("aiohttp.ClientSession.close", new_callable=AsyncMock) as mock_close:
        with DustDataEngine() as engine:
            assert engine._async_engine is not None
            assert engine._http_session is not None
            assert not engine._http_session.closed
    # The __exit__ method uses `loop.create_task` which is fire-and-forget.
    # We need to yield control to the event loop to allow the task to run.
    await asyncio.sleep(0)
    # After exiting, the close method should have been awaited
    mock_close.assert_awaited_once()


def test_dust_data_engine_raises_error_if_not_in_context():
    """Tests that using the engine outside a `with` block raises a RuntimeError."""
    engine = DustDataEngine()
    with pytest.raises(RuntimeError, match="must be used as a context manager"):
        engine.fetch_met_ufs(datetime.now(), 0, 0)


# --- Tests for the Asynchronous Engine ---


@pytest.mark.asyncio
async def test_async_engine_fetch_met_ufs():
    """Tests the UFS meteorology data fetching logic."""
    # Mock the s3 filesystem and the file object it returns
    mock_s3_fs = AsyncMock(spec=s3fs.S3FileSystem)
    mock_s3_file = MagicMock()
    # The return of `open` is an async context manager
    mock_s3_context = AsyncMock()
    mock_s3_context.__aenter__.return_value = mock_s3_file
    mock_s3_fs.open.return_value = mock_s3_context

    # Mock the return value of xarray.open_dataset
    mock_ds = xr.Dataset({"t2m": (("latitude", "longitude"), [[273.15]])})
    mock_ds = mock_ds.assign_coords(latitude=[0], longitude=[0])

    # Instantiate the async engine
    # We don't need a real http_session for this test
    engine = AsyncDustDataEngine(http_session=AsyncMock(), s3_filesystem=mock_s3_fs)

    # Patch asyncio.to_thread since xarray.open_dataset is blocking
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = mock_ds.sel(
            latitude=0, longitude=0, method="nearest"
        )
        dt = datetime(2023, 1, 1, 12)
        result_ds = await engine.fetch_met_ufs(dt, lat=0.1, lon=0.1)

    # Assertions
    # `open` is a sync method returning an async context manager
    mock_s3_fs.open.assert_called_once()
    mock_to_thread.assert_awaited_once()
    assert isinstance(result_ds, xr.Dataset)
    assert "t2m" in result_ds
    assert result_ds.t2m.item() == 273.15


@pytest.mark.asyncio
async def test_async_engine_fetch_soilgrids_concurrently():
    """Tests the SoilGrids fetching logic with a mocked aiohttp session."""
    # Mock the aiohttp session and its response
    mock_http_session = MagicMock()
    # The mock response needs `raise_for_status` (sync) and an async `read`
    mock_response = AsyncMock()
    mock_response.read.return_value = b"some_tiff_bytes"
    mock_response.raise_for_status = MagicMock()  # This is a sync method

    # The context manager that `get` returns
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_response
    mock_http_session.get.return_value = mock_context_manager

    # Instantiate the async engine
    engine = AsyncDustDataEngine(
        http_session=mock_http_session, s3_filesystem=MagicMock()
    )

    # Use a nested patch for rasterio.open, which is called inside asyncio.to_thread
    mock_rasterio_file = MagicMock()
    mock_rasterio_file.read.return_value = [[42.0]]
    mock_rasterio_context = MagicMock()
    mock_rasterio_context.__enter__.return_value = mock_rasterio_file

    with (
        patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        patch(
            "fengsha_prep.pipelines.uthresh.io.rasterio.open",
            return_value=mock_rasterio_context,
        ),
    ):
        # Make `to_thread` just return a value directly
        mock_to_thread.return_value = 42.0
        result = await engine.fetch_soilgrids_concurrently(lat=35.0, lon=-95.0)

    # Assertions
    # Should be called once for each variable: clay, sand, soc, bdod
    assert mock_http_session.get.call_count == 4
    assert len(result) == 4
    assert result["clay"] == 42.0
    assert result["sand"] == 42.0
    assert result["soc"] == 42.0
    assert result["bdod"] == 42.0
