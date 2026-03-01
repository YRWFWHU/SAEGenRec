"""Tests for K-core iterative filtering."""

import pytest
from datasets import Dataset

from saegenrec.data.processors.kcore import kcore_filter
from saegenrec.data.schemas import INTERACTIONS_FEATURES


class TestKcoreFilter:
    def test_no_filtering_needed(self, synthetic_interactions_dataset):
        """All users/items have 3 interactions each — threshold=3 should keep all."""
        filtered, stats = kcore_filter(synthetic_interactions_dataset, threshold=3)
        assert len(filtered) == 15
        assert stats["raw_interactions"] == 15
        assert stats["filtered_interactions"] == 15

    def test_filter_removes_sparse(self):
        """Users/items with fewer interactions than threshold should be removed."""
        data = {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3", "i1"],
            "timestamp": [100, 200, 300, 150],
            "rating": [5.0, 4.0, 3.0, 4.0],
            "review_text": ["a", "b", "c", "d"],
            "review_summary": ["s1", "s2", "s3", "s4"],
        }
        ds = Dataset.from_dict(data, features=INTERACTIONS_FEATURES)
        filtered, stats = kcore_filter(ds, threshold=3)
        assert stats["filtered_interactions"] < 4

    def test_threshold_boundary(self, synthetic_interactions_dataset):
        """Threshold exactly matching count should keep records."""
        filtered, stats = kcore_filter(synthetic_interactions_dataset, threshold=3)
        assert stats["filtered_interactions"] == 15

    def test_high_threshold_empty(self, synthetic_interactions_dataset):
        """Very high threshold should result in empty dataset."""
        filtered, stats = kcore_filter(synthetic_interactions_dataset, threshold=100)
        assert stats["filtered_interactions"] == 0
        assert stats["num_users"] == 0

    def test_convergence(self):
        """Filter should converge: iterative removal until stable."""
        data = {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u3"],
            "item_id": ["i1", "i2", "i1", "i3", "i1", "i2", "i3"],
            "timestamp": [100, 200, 300, 400, 500, 600, 700],
            "rating": [5.0] * 7,
            "review_text": [""] * 7,
            "review_summary": [""] * 7,
        }
        ds = Dataset.from_dict(data, features=INTERACTIONS_FEATURES)
        filtered, stats = kcore_filter(ds, threshold=2)
        assert stats["kcore_iterations"] >= 1
        # After convergence, all remaining users/items should have >= threshold interactions
        if stats["filtered_interactions"] > 0:
            df = filtered.to_pandas()
            assert (df.groupby("user_id").size() >= 2).all()
            assert (df.groupby("item_id").size() >= 2).all()

    def test_stats_output(self, synthetic_interactions_dataset):
        _, stats = kcore_filter(synthetic_interactions_dataset, threshold=3)
        assert "raw_interactions" in stats
        assert "filtered_interactions" in stats
        assert "kcore_threshold" in stats
        assert "kcore_iterations" in stats
        assert "num_users" in stats
        assert "num_items" in stats

    def test_threshold_one(self, synthetic_interactions_dataset):
        """Threshold=1 should keep everything."""
        filtered, stats = kcore_filter(synthetic_interactions_dataset, threshold=1)
        assert stats["filtered_interactions"] == 15
