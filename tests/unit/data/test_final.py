"""Tests for final training data generation."""

import pytest
from datasets import Dataset

from saegenrec.data.processors.augment import sliding_window_augment
from saegenrec.data.schemas import TRAINING_SAMPLE_FEATURES, USER_SEQUENCES_FEATURES
from saegenrec.data.tokenizers.passthrough import PassthroughTokenizer


@pytest.fixture
def passthrough_tokenizer():
    return PassthroughTokenizer(num_items=10)


@pytest.fixture
def item_titles():
    return {i: f"Item {i}" for i in range(10)}


class TestFinalDataGeneration:
    def test_schema_compliance(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Output should match TRAINING_SAMPLE_FEATURES schema."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        assert set(ds.column_names) == set(TRAINING_SAMPLE_FEATURES)

    def test_item_text_fields(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Each sample should have item titles."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        for row in ds:
            assert len(row["history_item_titles"]) == len(row["history_item_ids"])
            assert isinstance(row["target_item_title"], str)

    def test_item_token_fields(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Each sample should have item tokens."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        for row in ds:
            assert len(row["history_item_tokens"]) == len(row["history_item_ids"])
            for tokens in row["history_item_tokens"]:
                assert len(tokens) == passthrough_tokenizer.token_length
            assert len(row["target_item_tokens"]) == passthrough_tokenizer.token_length

    def test_hf_dataset_format(self, synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles):
        """Output should be a valid HuggingFace Dataset."""
        ds = sliding_window_augment(
            synthetic_user_sequences_dataset, passthrough_tokenizer, item_titles, max_seq_len=20
        )
        assert hasattr(ds, "save_to_disk")
        assert hasattr(ds, "to_pandas")
        assert len(ds) > 0
