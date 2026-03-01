"""Integration tests for pipeline orchestrator and final data generation."""

import json

import pytest
from datasets import Dataset

from saegenrec.data.config import PipelineConfig
from saegenrec.data.processors.final import generate_final_data
from saegenrec.data.processors.sequence import save_interim
from saegenrec.data.schemas import (
    ID_MAP_FEATURES,
    INTERACTIONS_FEATURES,
    ITEM_METADATA_FEATURES,
    USER_SEQUENCES_FEATURES,
)


class TestSaveInterim:
    def test_save_and_load(self, tmp_path, synthetic_interactions_dataset,
                           synthetic_user_sequences_dataset,
                           synthetic_item_metadata_dataset):
        user_map = Dataset.from_dict(
            {"original_id": ["u1", "u2", "u3"], "mapped_id": [0, 1, 2]},
            features=ID_MAP_FEATURES,
        )
        item_map = Dataset.from_dict(
            {"original_id": ["i1", "i2", "i3", "i4", "i5"],
             "mapped_id": [0, 1, 2, 3, 4]},
            features=ID_MAP_FEATURES,
        )
        stats = {"num_users": 3, "num_items": 5}

        output_dir = tmp_path / "interim" / "test" / "cat"
        save_interim(
            output_dir,
            synthetic_interactions_dataset,
            synthetic_user_sequences_dataset,
            synthetic_item_metadata_dataset,
            user_map,
            item_map,
            stats,
        )

        assert (output_dir / "interactions").exists()
        assert (output_dir / "user_sequences").exists()
        assert (output_dir / "item_metadata").exists()
        assert (output_dir / "user_id_map").exists()
        assert (output_dir / "item_id_map").exists()
        assert (output_dir / "stats.json").exists()

        with open(output_dir / "stats.json") as f:
            loaded_stats = json.load(f)
        assert loaded_stats["num_users"] == 3


class TestGenerateFinalData:
    def test_end_to_end(self, tmp_path, synthetic_user_sequences_dataset,
                        synthetic_item_metadata_dataset):
        """Generate final data from sequences → train/valid/test TrainingSample datasets."""
        # Save required datasets to disk
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

        # Split sequences for LOO
        from saegenrec.data.processors.split import split_data
        train_seqs, valid_seqs, test_seqs, _ = split_data(
            synthetic_user_sequences_dataset, strategy="loo"
        )

        output_dir = tmp_path / "processed"
        from saegenrec.data.tokenizers.passthrough import PassthroughTokenizer
        tokenizer = PassthroughTokenizer(num_items=5)

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

        assert (output_dir / "train").exists()
        assert (output_dir / "valid").exists()
        assert (output_dir / "test").exists()
        assert (output_dir / "stats.json").exists()
        assert stats["train_samples"] > 0
        assert stats["valid_samples"] == 3
        assert stats["test_samples"] == 3


class TestPipelineOrchestrator:
    def test_full_pipeline(self, tmp_path, amazon2015_raw_dir):
        """Run full pipeline end-to-end with synthetic Amazon2015 data."""
        from saegenrec.data.pipeline import run_pipeline

        config = PipelineConfig()
        config.dataset.name = "amazon2015"
        config.dataset.category = "TestCat"
        config.dataset.raw_dir = str(amazon2015_raw_dir.parent.parent)
        config.processing.kcore_threshold = 3
        config.processing.split_strategy = "loo"
        config.processing.max_seq_len = 10
        config.output.interim_dir = str(tmp_path / "interim")
        config.output.processed_dir = str(tmp_path / "processed")

        stats = run_pipeline(config)

        assert stats["train_samples"] > 0
        assert (tmp_path / "interim" / "amazon2015" / "TestCat" / "interactions").exists()
        assert (tmp_path / "processed" / "amazon2015" / "TestCat" / "loo" / "train").exists()

    def test_selective_steps(self, tmp_path, amazon2015_raw_dir):
        """Run only load and filter steps."""
        from saegenrec.data.pipeline import run_pipeline

        config = PipelineConfig()
        config.dataset.name = "amazon2015"
        config.dataset.category = "TestCat"
        config.dataset.raw_dir = str(amazon2015_raw_dir.parent.parent)
        config.processing.kcore_threshold = 3
        config.output.interim_dir = str(tmp_path / "interim")
        config.output.processed_dir = str(tmp_path / "processed")

        stats = run_pipeline(config, steps=["load", "filter"])

        assert "raw_interactions" in stats
        assert "filtered_interactions" in stats
        assert (tmp_path / "interim" / "amazon2015" / "TestCat" / "interactions").exists()
