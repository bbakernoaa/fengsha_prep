"""
Tests for the bnu module.
"""
import shutil
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.fengsha_prep import bnu


class TestBnu(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the bnu module.
    """

    def setUp(self):
        self.output_dir = Path("test_bnu_data")
        self.output_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @patch("src.fengsha_prep.bnu.tomllib.load")
    @patch("aiohttp.ClientSession", new_callable=MagicMock)
    async def test_get_bnu_data_with_mock_config(
        self, MockClientSession: MagicMock, mock_load: MagicMock
    ):
        """
        Tests the get_bnu_data function with a correctly mocked aiohttp session.
        This version uses a more explicit mocking strategy to avoid TypeErrors.
        """
        # 1. Mock the config file loading
        mock_load.return_value = {
            "bnu_data": {
                "sand_urls": [
                    "http://not-a-real-site.com/sand1.nc",
                    "http://example.com/sand2.nc",  # Placeholder to test both paths
                ]
            }
        }

        # 2. Create the mock response object.
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.read.return_value = b"test content"

        # This is the async context manager that session.get() returns.
        mock_response_context = AsyncMock()
        mock_response_context.__aenter__.return_value = mock_response

        # 3. Create the mock session object.
        mock_session = AsyncMock()
        # CRITICAL FIX: session.get is a regular method returning an async context manager.
        # We use a MagicMock here to prevent it from being a coroutine itself.
        mock_session.get = MagicMock(return_value=mock_response_context)

        # 4. Create the top-level async context manager for `aiohttp.ClientSession()`
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        MockClientSession.return_value = mock_session_context

        # Call the asynchronous function
        downloaded_files = await bnu.get_bnu_data(
            "sand", output_dir=str(self.output_dir)
        )

        # Sort paths to ensure consistent order for comparison
        downloaded_files.sort(key=lambda p: p.name)

        # Assert that the correct files were "downloaded"
        expected_files = [
            self.output_dir / "sand1.nc",
            self.output_dir / "sand2.nc",
        ]
        expected_files.sort(key=lambda p: p.name)
        self.assertEqual([str(p) for p in downloaded_files], [str(p) for p in expected_files])

        # Check that the download URL was called
        mock_session.get.assert_called_once_with("http://not-a-real-site.com/sand1.nc")

        # Check that files were created and have the correct content
        sand1_path = self.output_dir / "sand1.nc"
        sand2_path = self.output_dir / "sand2.nc"
        self.assertTrue(sand1_path.exists())
        self.assertEqual(sand1_path.read_bytes(), b"test content")
        self.assertTrue(sand2_path.exists())
        self.assertIn("dummy file", sand2_path.read_text())


if __name__ == "__main__":
    unittest.main()
