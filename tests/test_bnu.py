"""
Tests for the bnu module.
"""

import asyncio
import shutil
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fengsha_prep.data_downloaders import bnu


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

    @patch("fengsha_prep.data_downloaders.bnu.tomllib.load")
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
                    "http://not-a-real-site.com/sand2.nc",
                ]
            }
        }

        # 2. Create the mock response object.
        mock_response = AsyncMock()
        # `raise_for_status` is a synchronous method on the response object.
        mock_response.raise_for_status = MagicMock(return_value=None)
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
        self.assertEqual(
            [str(p) for p in downloaded_files], [str(p) for p in expected_files]
        )

        # Check that both URLs were called
        self.assertEqual(mock_session.get.call_count, 2)
        mock_session.get.assert_any_call("http://not-a-real-site.com/sand1.nc")
        mock_session.get.assert_any_call("http://not-a-real-site.com/sand2.nc")

        # Check that files were created and have the correct content
        sand1_path = self.output_dir / "sand1.nc"
        sand2_path = self.output_dir / "sand2.nc"
        self.assertTrue(sand1_path.exists())
        self.assertEqual(sand1_path.read_bytes(), b"test content")
        self.assertTrue(sand2_path.exists())
        self.assertEqual(sand2_path.read_bytes(), b"test content")

    @patch("fengsha_prep.data_downloaders.bnu.tomllib.load")
    @patch("fengsha_prep.data_downloaders.bnu._download_file")
    async def test_get_bnu_data_respects_concurrency_limit(
        self, mock_download_file: AsyncMock, mock_load: MagicMock
    ):
        """
        Tests that get_bnu_data limits the number of concurrent downloads.
        """
        # 1. Configure Mocks
        num_urls = 10
        concurrency_limit = 3
        mock_load.return_value = {
            "bnu_data": {
                "sand_urls": [f"http://test.com/file{i}.nc" for i in range(num_urls)]
            }
        }
        # Make the mock function return a value
        mock_download_file.return_value = Path("mock_file.nc")

        # 2. Setup Concurrency Tracking
        active_tasks = 0
        max_active_tasks = 0
        lock = asyncio.Lock()
        # This event will be used to pause the downloads and inspect the state
        pause_event = asyncio.Event()

        async def side_effect_to_track_concurrency(*args, **kwargs):
            nonlocal active_tasks, max_active_tasks
            async with lock:
                active_tasks += 1
                max_active_tasks = max(max_active_tasks, active_tasks)

            # Wait until the event is set, to simulate a download in progress
            await pause_event.wait()

            async with lock:
                active_tasks -= 1
            return Path("mock_file.nc")

        mock_download_file.side_effect = side_effect_to_track_concurrency

        # 3. Run the Test
        # Start get_bnu_data but don't wait for it to complete yet.
        task = asyncio.create_task(
            bnu.get_bnu_data(
                "sand",
                output_dir=str(self.output_dir),
                concurrency_limit=concurrency_limit,
            )
        )

        # Allow some time for the tasks to start and hit the pause_event
        await asyncio.sleep(0.1)

        # At this point, the number of active tasks should not exceed the limit.
        # This is the core assertion for the concurrency limit.
        self.assertLessEqual(max_active_tasks, concurrency_limit)

        # Unpause the tasks to let them finish
        pause_event.set()
        await task

        # A final check to ensure the peak concurrency was exactly the limit and all tasks ran
        self.assertEqual(max_active_tasks, concurrency_limit)
        self.assertEqual(mock_download_file.call_count, num_urls)


if __name__ == "__main__":
    unittest.main()
