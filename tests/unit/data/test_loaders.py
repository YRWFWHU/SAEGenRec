"""Tests for Amazon2015 and Amazon2023 data loaders."""

import json
from pathlib import Path

import pytest

from saegenrec.data.loaders.amazon2015 import Amazon2015Loader
from saegenrec.data.loaders.amazon2023 import Amazon2023Loader
from saegenrec.data.loaders.base import LOADER_REGISTRY, get_loader


class TestLoaderRegistry:
    def test_amazon2015_registered(self):
        assert "amazon2015" in LOADER_REGISTRY

    def test_amazon2023_registered(self):
        assert "amazon2023" in LOADER_REGISTRY

    def test_get_loader_valid(self):
        loader = get_loader("amazon2015")
        assert isinstance(loader, Amazon2015Loader)

    def test_get_loader_invalid(self):
        with pytest.raises(ValueError, match="Unknown dataset loader"):
            get_loader("nonexistent")


class TestAmazon2015Loader:
    def test_load_interactions(self, amazon2015_raw_dir):
        loader = Amazon2015Loader()
        ds = loader.load_interactions(amazon2015_raw_dir)
        assert len(ds) == 15
        assert set(ds.column_names) == {
            "user_id", "item_id", "timestamp", "rating", "review_text", "review_summary",
        }

    def test_field_mapping(self, amazon2015_raw_dir):
        loader = Amazon2015Loader()
        ds = loader.load_interactions(amazon2015_raw_dir)
        row = ds[0]
        assert isinstance(row["user_id"], str)
        assert isinstance(row["timestamp"], int)
        assert isinstance(row["rating"], float)

    def test_dedup(self, amazon2015_raw_dir):
        """Duplicate (user, item, timestamp) triples should be deduplicated."""
        dup_record = {
            "reviewerID": "u1", "asin": "i1", "reviewText": "Duplicate",
            "overall": 5.0, "summary": "Dup", "unixReviewTime": 1000,
        }
        review_file = amazon2015_raw_dir / "TestCat.json"
        with open(review_file, "a") as f:
            f.write(json.dumps(dup_record) + "\n")

        loader = Amazon2015Loader()
        ds = loader.load_interactions(amazon2015_raw_dir)
        assert len(ds) == 15

    def test_missing_fields(self, tmp_path):
        """Missing optional fields should be filled with empty strings."""
        data_dir = tmp_path / "MissingFields"
        data_dir.mkdir()
        review_file = data_dir / "MissingFields.json"
        record = {"reviewerID": "u1", "asin": "i1", "unixReviewTime": 100, "overall": 5.0}
        with open(review_file, "w") as f:
            f.write(json.dumps(record) + "\n")

        loader = Amazon2015Loader()
        ds = loader.load_interactions(data_dir)
        assert len(ds) == 1
        assert ds[0]["review_text"] == ""
        assert ds[0]["review_summary"] == ""

    def test_empty_file(self, tmp_path):
        data_dir = tmp_path / "EmptyFile"
        data_dir.mkdir()
        (data_dir / "EmptyFile.json").write_text("")
        loader = Amazon2015Loader()
        ds = loader.load_interactions(data_dir)
        assert len(ds) == 0

    def test_file_not_found(self, tmp_path):
        loader = Amazon2015Loader()
        with pytest.raises(FileNotFoundError):
            loader.load_interactions(tmp_path / "NoSuchDir")

    def test_load_item_metadata(self, amazon2015_raw_dir):
        loader = Amazon2015Loader()
        ds = loader.load_item_metadata(amazon2015_raw_dir)
        assert len(ds) == 5
        assert "categories" in ds.column_names
        assert isinstance(ds[0]["categories"], list)

    def test_metadata_price_none(self, amazon2015_raw_dir):
        loader = Amazon2015Loader()
        ds = loader.load_item_metadata(amazon2015_raw_dir)
        item3 = [r for r in ds if r["item_id"] == "i3"][0]
        assert item3["price"] is None


class TestAmazon2023Loader:
    def test_load_interactions(self, amazon2023_raw_dir):
        loader = Amazon2023Loader()
        ds = loader.load_interactions(amazon2023_raw_dir)
        assert len(ds) == 15

    def test_timestamp_conversion(self, amazon2023_raw_dir):
        """Timestamps should be converted from ms to seconds."""
        loader = Amazon2023Loader()
        ds = loader.load_interactions(amazon2023_raw_dir)
        row = ds[0]
        assert row["timestamp"] == 1000

    def test_field_mapping(self, amazon2023_raw_dir):
        loader = Amazon2023Loader()
        ds = loader.load_interactions(amazon2023_raw_dir)
        row = ds[0]
        assert row["review_summary"] != ""

    def test_load_item_metadata(self, amazon2023_raw_dir):
        loader = Amazon2023Loader()
        ds = loader.load_item_metadata(amazon2023_raw_dir)
        assert len(ds) == 5

    def test_price_parsing(self, amazon2023_raw_dir):
        """Prices like '$29.99' should be parsed to float."""
        loader = Amazon2023Loader()
        ds = loader.load_item_metadata(amazon2023_raw_dir)
        item4 = [r for r in ds if r["item_id"] == "i4"][0]
        assert item4["price"] == pytest.approx(29.99, abs=0.01)

    def test_empty_images(self, amazon2023_raw_dir):
        """Empty images list should result in empty image_url."""
        loader = Amazon2023Loader()
        ds = loader.load_item_metadata(amazon2023_raw_dir)
        item3 = [r for r in ds if r["item_id"] == "i3"][0]
        assert item3["image_url"] == ""

    def test_description_join(self, amazon2023_raw_dir):
        """Description list should be joined with spaces."""
        loader = Amazon2023Loader()
        ds = loader.load_item_metadata(amazon2023_raw_dir)
        item1 = [r for r in ds if r["item_id"] == "i1"][0]
        assert item1["description"] == "Desc 1"
