"""Tests for final (legacy) training data generation with tokenizer."""

import pytest
from datasets import Dataset

from saegenrec.data.processors.final import generate_final_data
from saegenrec.data.schemas import (
    ID_MAP_FEATURES,
    TRAINING_SAMPLE_FEATURES,
)
from saegenrec.data.tokenizers.passthrough import PassthroughTokenizer


@pytest.fixture
def passthrough_tokenizer():
    return PassthroughTokenizer(num_items=10)


class TestFinalDataGeneration:
    def _run_generate(self, tmp_path, synthetic_user_sequences_dataset,
                      synthetic_item_metadata_dataset, tokenizer):
        interim_dir = tmp_path / "interim"
        interim_dir.mkdir()
        synthetic_user_sequences_dataset.save_to_disk(str(interim_dir / "user_sequences"))
        synthetic_item_metadata_dataset.save_to_disk(str(interim_dir / "item_metadata"))
        item_id_map = Dataset.from_dict(
            {"original_id": ["i1", "i2", "i3", "i4", "i5"],
             "mapped_id": [0, 1, 2, 3, 4]},
            features=ID_MAP_FEATURES,
        )
        item_id_map.save_to_disk(str(interim_dir / "item_id_map"))

        from saegenrec.data.processors.split import split_data
        train_seqs, valid_seqs, test_seqs, _ = split_data(
            synthetic_user_sequences_dataset, strategy="loo"
        )

        output_dir = tmp_path / "processed"
        stats = generate_final_data(
            user_sequences_dir=interim_dir / "user_sequences",
            item_metadata_dir=interim_dir / "item_metadata",
            item_id_map_dir=interim_dir / "item_id_map",
            train_sequences=train_seqs,
            valid_sequences=valid_seqs,
            test_sequences=test_seqs,
            tokenizer=tokenizer,
            max_seq_len=20,
            output_dir=output_dir,
            split_strategy="loo",
        )
        from datasets import load_from_disk
        train_ds = load_from_disk(str(output_dir / "train"))
        return train_ds, stats

    def test_schema_compliance(self, tmp_path, synthetic_user_sequences_dataset,
                               synthetic_item_metadata_dataset, passthrough_tokenizer):
        """Output should match TRAINING_SAMPLE_FEATURES schema."""
        train_ds, _ = self._run_generate(
            tmp_path, synthetic_user_sequences_dataset,
            synthetic_item_metadata_dataset, passthrough_tokenizer,
        )
        assert set(train_ds.column_names) == set(TRAINING_SAMPLE_FEATURES)

    def test_item_text_fields(self, tmp_path, synthetic_user_sequences_dataset,
                              synthetic_item_metadata_dataset, passthrough_tokenizer):
        """Each sample should have item titles."""
        train_ds, _ = self._run_generate(
            tmp_path, synthetic_user_sequences_dataset,
            synthetic_item_metadata_dataset, passthrough_tokenizer,
        )
        for row in train_ds:
            assert len(row["history_item_titles"]) == len(row["history_item_ids"])
            assert isinstance(row["target_item_title"], str)

    def test_item_token_fields(self, tmp_path, synthetic_user_sequences_dataset,
                               synthetic_item_metadata_dataset, passthrough_tokenizer):
        """Each sample should have item tokens via tokenizer."""
        train_ds, _ = self._run_generate(
            tmp_path, synthetic_user_sequences_dataset,
            synthetic_item_metadata_dataset, passthrough_tokenizer,
        )
        for row in train_ds:
            assert len(row["history_item_tokens"]) == len(row["history_item_ids"])
            assert len(row["target_item_tokens"]) == passthrough_tokenizer.token_length

    def test_hf_dataset_format(self, tmp_path, synthetic_user_sequences_dataset,
                               synthetic_item_metadata_dataset, passthrough_tokenizer):
        """Output should be a valid HuggingFace Dataset."""
        train_ds, _ = self._run_generate(
            tmp_path, synthetic_user_sequences_dataset,
            synthetic_item_metadata_dataset, passthrough_tokenizer,
        )
        assert hasattr(train_ds, "save_to_disk")
        assert hasattr(train_ds, "to_pandas")
        assert len(train_ds) > 0
