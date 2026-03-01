"""Tests for sliding window data augmentation."""

import pytest
from datasets import Dataset

from saegenrec.data.processors.augment import convert_eval_split, sliding_window_augment
from saegenrec.data.schemas import USER_SEQUENCES_FEATURES
from saegenrec.data.tokenizers.passthrough import PassthroughTokenizer


@pytest.fixture
def passthrough_tokenizer():
    return PassthroughTokenizer(num_items=10)


@pytest.fixture
def item_titles():
    return {i: f"Item {i}" for i in range(10)}


class TestSlidingWindow:
    def test_sample_count(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Each user with N items generates N-1 samples."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        # 3 users * (5-1) = 12 samples
        assert len(ds) == 12

    def test_truncation(self, passthrough_tokenizer, item_titles):
        """History should be truncated to max_seq_len."""
        data = {
            "user_id": [0],
            "item_ids": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "timestamps": [list(range(100, 200, 10))],
            "ratings": [[5.0] * 10],
            "review_texts": [[""] * 10],
            "review_summaries": [[""] * 10],
        }
        seqs = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        ds = sliding_window_augment(seqs, passthrough_tokenizer, item_titles, max_seq_len=3)

        for row in ds:
            assert len(row["history_item_ids"]) <= 3

    def test_min_sequence(self, passthrough_tokenizer, item_titles):
        """A sequence of length 2 should produce exactly 1 sample."""
        data = {
            "user_id": [0],
            "item_ids": [[0, 1]],
            "timestamps": [[100, 200]],
            "ratings": [[5.0, 4.0]],
            "review_texts": [["a", "b"]],
            "review_summaries": [["s1", "s2"]],
        }
        seqs = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        ds = sliding_window_augment(seqs, passthrough_tokenizer, item_titles, max_seq_len=20)
        assert len(ds) == 1
        assert ds[0]["history_item_ids"] == [0]
        assert ds[0]["target_item_id"] == 1

    def test_history_range(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """History length should be in [1, max_seq_len]."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        for row in ds:
            h_len = len(row["history_item_ids"])
            assert 1 <= h_len <= 20

    def test_tokens_present(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Token fields should be populated."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        for row in ds:
            assert len(row["history_item_tokens"]) == len(row["history_item_ids"])
            assert len(row["target_item_tokens"]) == 1

    def test_skip_single_item(self, passthrough_tokenizer, item_titles):
        """Sequences with only 1 item should produce no samples."""
        data = {
            "user_id": [0],
            "item_ids": [[5]],
            "timestamps": [[100]],
            "ratings": [[5.0]],
            "review_texts": [["a"]],
            "review_summaries": [["s"]],
        }
        seqs = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        ds = sliding_window_augment(seqs, passthrough_tokenizer, item_titles, max_seq_len=20)
        assert len(ds) == 0


class TestConvertEvalSplit:
    def test_single_item_sequence(self, passthrough_tokenizer, item_titles):
        """LOO eval sequences have 1 item — should produce 1 sample with empty history."""
        data = {
            "user_id": [0, 1],
            "item_ids": [[4], [3]],
            "timestamps": [[500], [400]],
            "ratings": [[5.0], [4.0]],
            "review_texts": [["a"], ["b"]],
            "review_summaries": [["s1"], ["s2"]],
        }
        seqs = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        ds = convert_eval_split(seqs, passthrough_tokenizer, item_titles, max_seq_len=20)
        assert len(ds) == 2
        assert ds[0]["target_item_id"] == 4
        assert ds[0]["history_item_ids"] == []

    def test_multi_item_sequence(self, passthrough_tokenizer, item_titles):
        """TO eval sequences may have multiple items."""
        data = {
            "user_id": [0],
            "item_ids": [[0, 1, 2]],
            "timestamps": [[100, 200, 300]],
            "ratings": [[5.0, 4.0, 3.0]],
            "review_texts": [["a", "b", "c"]],
            "review_summaries": [["s1", "s2", "s3"]],
        }
        seqs = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        ds = convert_eval_split(seqs, passthrough_tokenizer, item_titles, max_seq_len=20)
        assert len(ds) == 1
        assert ds[0]["history_item_ids"] == [0, 1]
        assert ds[0]["target_item_id"] == 2
