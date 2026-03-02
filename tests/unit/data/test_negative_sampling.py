"""Tests for negative sampling processor."""

import pytest
from datasets import Dataset

from saegenrec.data.schemas import INTERIM_SAMPLE_FEATURES, NEGATIVE_SAMPLE_FEATURES


@pytest.fixture
def item_titles_map():
    """mapped_id -> title mapping for 10 items."""
    return {i: f"Item {i}" for i in range(10)}


@pytest.fixture
def all_item_ids():
    """Global item ID list (10 items)."""
    return list(range(10))


@pytest.fixture
def synthetic_interim_samples_dataset():
    """Small InterimSample dataset for testing."""
    return Dataset.from_dict(
        {
            "user_id": [0, 0, 1, 1, 2],
            "history_item_ids": [[0, 1], [0, 1, 2], [3, 4], [3, 4, 5], [6, 7]],
            "history_item_titles": [
                ["Item 0", "Item 1"],
                ["Item 0", "Item 1", "Item 2"],
                ["Item 3", "Item 4"],
                ["Item 3", "Item 4", "Item 5"],
                ["Item 6", "Item 7"],
            ],
            "target_item_id": [2, 3, 5, 6, 8],
            "target_item_title": ["Item 2", "Item 3", "Item 5", "Item 6", "Item 8"],
        },
        features=INTERIM_SAMPLE_FEATURES,
    )


@pytest.fixture
def user_interacted_items():
    """user_id -> set of all interacted item_ids (from full UserSequence)."""
    return {
        0: {0, 1, 2, 3},
        1: {3, 4, 5, 6},
        2: {6, 7, 8},
    }


class TestBuildUserInteractedItems:
    def test_basic(self, synthetic_user_sequences_dataset):
        from saegenrec.data.processors.negative_sampling import (
            build_user_interacted_items,
        )

        result = build_user_interacted_items(synthetic_user_sequences_dataset)
        assert result[0] == {0, 1, 2, 3, 4}
        assert result[1] == {0, 1, 2, 3, 4}
        assert result[2] == {0, 1, 2, 3, 4}


class TestSampleNegatives:
    def test_correct_count(
        self,
        synthetic_interim_samples_dataset,
        user_interacted_items,
        all_item_ids,
        item_titles_map,
    ):
        """Each sample should have exactly num_negatives negative items."""
        from saegenrec.data.processors.negative_sampling import sample_negatives

        result_ds, stats = sample_negatives(
            synthetic_interim_samples_dataset,
            user_interacted_items,
            all_item_ids,
            item_titles_map,
            num_negatives=3,
            seed=42,
        )
        assert len(result_ds) == 5
        for row in result_ds:
            assert len(row["negative_item_ids"]) == 3
            assert len(row["negative_item_titles"]) == 3

    def test_no_overlap_with_history(
        self,
        synthetic_interim_samples_dataset,
        user_interacted_items,
        all_item_ids,
        item_titles_map,
    ):
        """Negative items must not be in user's interaction history."""
        from saegenrec.data.processors.negative_sampling import sample_negatives

        result_ds, _ = sample_negatives(
            synthetic_interim_samples_dataset,
            user_interacted_items,
            all_item_ids,
            item_titles_map,
            num_negatives=3,
            seed=42,
        )
        for row in result_ds:
            uid = row["user_id"]
            interacted = user_interacted_items[uid]
            for neg_id in row["negative_item_ids"]:
                assert neg_id not in interacted

    def test_seed_reproducibility(
        self,
        synthetic_interim_samples_dataset,
        user_interacted_items,
        all_item_ids,
        item_titles_map,
    ):
        """Same seed should produce identical results."""
        from saegenrec.data.processors.negative_sampling import sample_negatives

        ds1, _ = sample_negatives(
            synthetic_interim_samples_dataset,
            user_interacted_items,
            all_item_ids,
            item_titles_map,
            num_negatives=3,
            seed=42,
        )
        ds2, _ = sample_negatives(
            synthetic_interim_samples_dataset,
            user_interacted_items,
            all_item_ids,
            item_titles_map,
            num_negatives=3,
            seed=42,
        )
        for i in range(len(ds1)):
            assert ds1[i]["negative_item_ids"] == ds2[i]["negative_item_ids"]

    def test_degraded_sampling(self, item_titles_map):
        """When available negatives < num_negatives, sample all available + warn."""
        from saegenrec.data.processors.negative_sampling import sample_negatives

        samples = Dataset.from_dict(
            {
                "user_id": [0],
                "history_item_ids": [[0]],
                "history_item_titles": [["Item 0"]],
                "target_item_id": [1],
                "target_item_title": ["Item 1"],
            },
            features=INTERIM_SAMPLE_FEATURES,
        )
        user_interacted = {0: {0, 1, 2, 3, 4, 5, 6, 7, 8}}
        small_all_ids = list(range(10))

        result_ds, stats = sample_negatives(
            samples,
            user_interacted,
            small_all_ids,
            item_titles_map,
            num_negatives=5,
            seed=42,
        )
        neg_ids = result_ds[0]["negative_item_ids"]
        assert len(neg_ids) == 1
        assert neg_ids[0] == 9
        assert stats["num_negatives_warnings"] > 0

    def test_output_schema(
        self,
        synthetic_interim_samples_dataset,
        user_interacted_items,
        all_item_ids,
        item_titles_map,
    ):
        """Output should match NEGATIVE_SAMPLE_FEATURES schema."""
        from saegenrec.data.processors.negative_sampling import sample_negatives

        result_ds, _ = sample_negatives(
            synthetic_interim_samples_dataset,
            user_interacted_items,
            all_item_ids,
            item_titles_map,
            num_negatives=3,
            seed=42,
        )
        assert set(result_ds.column_names) == set(NEGATIVE_SAMPLE_FEATURES)
