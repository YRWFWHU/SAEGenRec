"""Tests for image downloader."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from saegenrec.data.schemas import ITEM_METADATA_FEATURES


class TestImageDownloader:
    @pytest.fixture
    def metadata_dir(self, tmp_path):
        """Create item_metadata dataset on disk."""
        metadata = Dataset.from_dict(
            {
                "item_id": ["i1", "i2", "i3"],
                "title": ["Item One", "Item Two", "Item Three"],
                "brand": ["A", "B", "C"],
                "categories": [["c1"], ["c2"], []],
                "description": ["d1", "d2", "d3"],
                "price": [9.99, 19.99, None],
                "image_url": ["http://example.com/1.jpg", "http://example.com/2.jpg", ""],
            },
            features=ITEM_METADATA_FEATURES,
        )
        metadata.save_to_disk(str(tmp_path / "item_metadata"))
        return tmp_path / "item_metadata"

    @patch("saegenrec.data.processors.images.requests.get")
    def test_successful_download(self, mock_get, metadata_dir, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"fake image data"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from saegenrec.data.processors.images import download_images

        output_dir = tmp_path / "images"
        stats = download_images(metadata_dir, output_dir)
        assert stats["downloaded"] == 2
        assert stats["failed"] == 0
        assert (output_dir / "i1.jpg").exists()

    @patch("saegenrec.data.processors.images.requests.get")
    def test_skip_existing(self, mock_get, metadata_dir, tmp_path):
        output_dir = tmp_path / "images"
        output_dir.mkdir()
        (output_dir / "i1.jpg").write_bytes(b"existing")

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from saegenrec.data.processors.images import download_images

        stats = download_images(metadata_dir, output_dir)
        assert stats["skipped_existing"] == 1
        assert stats["downloaded"] == 1

    @patch("saegenrec.data.processors.images.requests.get")
    def test_error_skip(self, mock_get, metadata_dir, tmp_path):
        mock_get.side_effect = Exception("Network error")

        from saegenrec.data.processors.images import download_images

        output_dir = tmp_path / "images"
        stats = download_images(metadata_dir, output_dir, max_retries=1)
        assert stats["failed"] == 2
        assert len(stats["errors"]) == 2

    def test_empty_url_skipped(self, metadata_dir, tmp_path):
        """Items with empty image_url should not be attempted."""
        from saegenrec.data.processors.images import download_images

        with patch("saegenrec.data.processors.images.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [b"data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            output_dir = tmp_path / "images"
            stats = download_images(metadata_dir, output_dir)
            assert stats["total_urls"] == 2
