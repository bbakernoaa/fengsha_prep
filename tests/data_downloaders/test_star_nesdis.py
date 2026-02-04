import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from fengsha_prep.data_downloaders.star_nesdis import (
    fetch_star_nesdis_file_list,
    load_and_regrid_star_nesdis_data,
    process_star_nesdis_data,
)


@pytest.mark.asyncio
async def test_fetch_star_nesdis_file_list():
    mock_html = """
    <html>
    <body>
    <a href="file1.nc">file1.nc</a>
    <a href="file2.nc">file2.nc</a>
    <a href="image.jpg">image.jpg</a>
    </body>
    </html>
    """
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.text.return_value = mock_html
        mock_response.raise_for_status = lambda: None
        mock_get.return_value.__aenter__.return_value = mock_response

        url = "https://example.com/data/"
        file_list = await fetch_star_nesdis_file_list(url)

        assert len(file_list) == 2
        assert "https://example.com/data/file1.nc" in file_list
        assert "https://example.com/data/file2.nc" in file_list


def test_load_and_regrid_star_nesdis_data(tmp_path):
    rows, cols = 100, 200
    data = np.random.rand(rows, cols).astype(np.float64)
    ds = xr.Dataset({"BRDF_QC": (("row", "column"), data)})
    filepath = tmp_path / "test.nc"
    ds.to_netcdf(filepath)

    ds_regridded = load_and_regrid_star_nesdis_data(filepath, target_resolution=1.0)

    assert "x" in ds_regridded.dims
    assert "y" in ds_regridded.dims
    assert ds_regridded.BRDF_QC.dtype == np.float32


@pytest.mark.asyncio
async def test_process_star_nesdis_data(tmp_path):
    url = "https://example.com/test.nc"
    output_path = tmp_path / "processed.nc"

    # Mock download_file
    async def mock_download(url, filepath, session=None):
        rows, cols = 100, 200
        ds = xr.Dataset({"data": (("row", "column"), np.random.rand(rows, cols))})
        ds.to_netcdf(filepath)
        return Path(filepath)

    target = "fengsha_prep.data_downloaders.star_nesdis.download_file"
    with patch(target, side_effect=mock_download):
        result = await process_star_nesdis_data(url, output_path, target_resolution=1.0)

        assert result == output_path
        assert output_path.exists()

        # Verify output content
        ds_result = xr.open_dataset(output_path)
        assert ds_result.data.dtype == np.float32

        # Verify cleanup (the temp file name depends on the URL)
        temp_file = Path("temp_test.nc")
        assert not await asyncio.to_thread(temp_file.exists)
