"""
Tests for the I/O operations of the uthresh pipeline.
"""
from unittest.mock import MagicMock, patch

from fengsha_prep.pipelines.uthresh.io import DustDataEngine


@patch("fengsha_prep.pipelines.uthresh.io.earthaccess.login")
@patch("fengsha_prep.pipelines.uthresh.io.s3fs.S3FileSystem")
@patch("fengsha_prep.pipelines.uthresh.io.WebCoverageService")
def test_dust_data_engine_initialization(mock_wcs, mock_s3fs, mock_earthaccess):
    """Tests that the DustDataEngine initializes its clients correctly."""
    engine = DustDataEngine()

    mock_earthaccess.assert_called_once_with(strategy="interactive")
    mock_s3fs.assert_called_once_with(anon=True)
    assert "maps.isric.org" in engine.wcs_url


@patch("fengsha_prep.pipelines.uthresh.io.earthaccess.login", MagicMock())
@patch("fengsha_prep.pipelines.uthresh.io.s3fs.S3FileSystem", MagicMock())
@patch("fengsha_prep.pipelines.uthresh.io.WebCoverageService")
def test_fetch_soilgrids(mock_wcs):
    """Tests the soilgrids fetching logic with a mocked WCS client."""
    mock_wcs_instance = MagicMock()
    # Mock the response from the WCS service
    mock_wcs_instance.getCoverage.return_value.read.return_value = b""
    mock_wcs.return_value = mock_wcs_instance

    engine = DustDataEngine()

    # Use a nested patch for rasterio.open
    with patch("fengsha_prep.pipelines.uthresh.io.rasterio.open", MagicMock()):
        result = engine.fetch_soilgrids(lat=35.0, lon=-95.0)

    # Verify that the WCS client was called for each variable
    assert mock_wcs_instance.getCoverage.call_count == 4
    assert "clay" in result
    assert "sand" in result
    assert "soc" in result
    assert "bdod" in result
