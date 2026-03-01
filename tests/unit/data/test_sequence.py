"""Tests for sequence builder."""

import pytest
from datasets import Dataset

from saegenrec.data.processors.sequence import build_sequences
from saegenrec.data.schemas import INTERACTIONS_FEATURES


class TestBuildSequences:
    def test_basic_build(self, synthetic_interactions_dataset):
        seqs, user_map, item_map, stats = build_sequences(synthetic_interactions_dataset)
        assert len(seqs) == 3
        assert len(user_map) == 3
        assert len(item_map) == 5

    def test_time_ordering(self, synthetic_interactions_dataset):
        """Each user's sequence should be sorted by timestamp."""
        seqs, _, _, _ = build_sequences(synthetic_interactions_dataset)
        for row in seqs:
            timestamps = row["timestamps"]
            assert timestamps == sorted(timestamps), "Timestamps must be non-decreasing"

    def test_id_mapping_bijectivity(self, synthetic_interactions_dataset):
        """ID mappings should be bijective (one-to-one)."""
        _, user_map, item_map, _ = build_sequences(synthetic_interactions_dataset)

        user_orig = user_map["original_id"]
        user_mapped = user_map["mapped_id"]
        assert len(set(user_orig)) == len(user_orig)
        assert len(set(user_mapped)) == len(user_mapped)

        item_orig = item_map["original_id"]
        item_mapped = item_map["mapped_id"]
        assert len(set(item_orig)) == len(item_orig)
        assert len(set(item_mapped)) == len(item_mapped)

    def test_id_mapping_starts_from_zero(self, synthetic_interactions_dataset):
        _, user_map, item_map, _ = build_sequences(synthetic_interactions_dataset)
        assert min(user_map["mapped_id"]) == 0
        assert min(item_map["mapped_id"]) == 0

    def test_review_field_preservation(self, synthetic_interactions_dataset):
        """Review text and summary fields should be preserved in sequences."""
        seqs, _, _, _ = build_sequences(synthetic_interactions_dataset)
        for row in seqs:
            assert len(row["review_texts"]) == len(row["item_ids"])
            assert len(row["review_summaries"]) == len(row["item_ids"])

    def test_sequence_lengths_consistent(self, synthetic_interactions_dataset):
        """All parallel arrays in each sequence should have the same length."""
        seqs, _, _, _ = build_sequences(synthetic_interactions_dataset)
        for row in seqs:
            n = len(row["item_ids"])
            assert len(row["timestamps"]) == n
            assert len(row["ratings"]) == n
            assert len(row["review_texts"]) == n
            assert len(row["review_summaries"]) == n

    def test_stats_output(self, synthetic_interactions_dataset):
        _, _, _, stats = build_sequences(synthetic_interactions_dataset)
        assert "avg_seq_length" in stats
        assert "num_users" in stats
        assert "num_items" in stats
        assert stats["avg_seq_length"] == 5.0

    def test_empty_interactions(self):
        empty = Dataset.from_dict(
            {k: [] for k in INTERACTIONS_FEATURES}, features=INTERACTIONS_FEATURES
        )
        seqs, user_map, item_map, stats = build_sequences(empty)
        assert len(seqs) == 0
        assert len(user_map) == 0
        assert len(item_map) == 0

    def test_dedup_within_build(self):
        """Duplicate (user, item, timestamp) should be deduplicated."""
        data = {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i1", "i2"],
            "timestamp": [100, 100, 200],
            "rating": [5.0, 4.0, 3.0],
            "review_text": ["a", "b", "c"],
            "review_summary": ["s1", "s2", "s3"],
        }
        ds = Dataset.from_dict(data, features=INTERACTIONS_FEATURES)
        seqs, _, _, _ = build_sequences(ds)
        assert len(seqs[0]["item_ids"]) == 2
