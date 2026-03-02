"""Integration tests for pipeline orchestrator and two-stage architecture."""

import json

import pytest
from datasets import Dataset

from saegenrec.data.config import PipelineConfig
from saegenrec.data.processors.sequence import save_interim
from saegenrec.data.schemas import (
    ID_MAP_FEATURES,
    INTERIM_SAMPLE_FEATURES,
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


class TestTwoStageAugment:
    """Test Stage 2 augment output — no tokenizer, INTERIM_SAMPLE_FEATURES schema."""

    def test_end_to_end(self, tmp_path, synthetic_user_sequences_dataset,
                        synthetic_item_metadata_dataset):
        """Generate augmented data from sequences → train/valid/test InterimSample datasets."""
        from saegenrec.data.processors.augment import (
            convert_eval_split,
            sliding_window_augment,
        )
        from saegenrec.data.processors.split import split_data

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

        train_seqs, valid_seqs, test_seqs, _ = split_data(
            synthetic_user_sequences_dataset, strategy="loo"
        )

        item_titles = {i: f"Item {i}" for i in range(5)}
        train_ds = sliding_window_augment(train_seqs, item_titles, max_seq_len=20)
        valid_ds = convert_eval_split(valid_seqs, item_titles, max_seq_len=20)
        test_ds = convert_eval_split(test_seqs, item_titles, max_seq_len=20)

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        train_ds.save_to_disk(str(output_dir / "train"))
        valid_ds.save_to_disk(str(output_dir / "valid"))
        test_ds.save_to_disk(str(output_dir / "test"))

        assert (output_dir / "train").exists()
        assert (output_dir / "valid").exists()
        assert (output_dir / "test").exists()
        assert len(train_ds) > 0
        assert len(valid_ds) == 3
        assert len(test_ds) == 3

        assert set(train_ds.column_names) == set(INTERIM_SAMPLE_FEATURES)
        assert "history_item_tokens" not in train_ds.column_names
        assert "target_item_tokens" not in train_ds.column_names


class TestPipelineOrchestrator:
    def _make_config(self, tmp_path, amazon2015_raw_dir) -> PipelineConfig:
        config = PipelineConfig()
        config.dataset.name = "amazon2015"
        config.dataset.category = "TestCat"
        config.dataset.raw_dir = str(amazon2015_raw_dir.parent.parent)
        config.processing.kcore_threshold = 3
        config.processing.split_strategy = "loo"
        config.processing.max_seq_len = 10
        config.output.interim_dir = str(tmp_path / "interim")
        config.output.processed_dir = str(tmp_path / "processed")
        return config

    def test_full_pipeline(self, tmp_path, amazon2015_raw_dir):
        """Run full pipeline end-to-end with synthetic Amazon2015 data (two-stage)."""
        from saegenrec.data.pipeline import run_pipeline

        config = self._make_config(tmp_path, amazon2015_raw_dir)
        stats = run_pipeline(config)

        stage1_dir = tmp_path / "interim" / "amazon2015" / "TestCat"
        stage2_dir = stage1_dir / "loo"

        assert (stage1_dir / "interactions").exists()
        assert (stage1_dir / "user_sequences").exists()
        assert (stage1_dir / "item_metadata").exists()
        assert (stage1_dir / "stats.json").exists()

        assert (stage2_dir / "train_sequences").exists()
        assert (stage2_dir / "train").exists()
        assert (stage2_dir / "valid").exists()
        assert (stage2_dir / "test").exists()
        assert (stage2_dir / "stats.json").exists()

        assert stats["train_samples"] > 0

        from datasets import load_from_disk

        train_ds = load_from_disk(str(stage2_dir / "train"))
        assert "history_item_tokens" not in train_ds.column_names
        assert "target_item_tokens" not in train_ds.column_names

    def test_selective_steps_stage1(self, tmp_path, amazon2015_raw_dir):
        """Run only Stage 1 steps (load, filter, sequence)."""
        from saegenrec.data.pipeline import STAGE1_STEPS, STAGE2_STEPS, run_pipeline

        assert STAGE1_STEPS == ["load", "filter", "sequence"]
        assert STAGE2_STEPS == ["split", "augment", "negative_sampling"]

        config = self._make_config(tmp_path, amazon2015_raw_dir)
        stats = run_pipeline(config, steps=STAGE1_STEPS)

        stage1_dir = tmp_path / "interim" / "amazon2015" / "TestCat"
        assert (stage1_dir / "interactions").exists()
        assert (stage1_dir / "user_sequences").exists()
        assert "raw_interactions" in stats

        stage2_dir = stage1_dir / "loo"
        assert not (stage2_dir / "train").exists()

    def test_incremental_stage2_only(self, tmp_path, amazon2015_raw_dir):
        """Run Stage 1 first, then only Stage 2 — Stage 1 artifacts should be reused."""
        from saegenrec.data.pipeline import STAGE1_STEPS, STAGE2_STEPS, run_pipeline

        config = self._make_config(tmp_path, amazon2015_raw_dir)
        stage1_dir = tmp_path / "interim" / "amazon2015" / "TestCat"
        stage2_dir = stage1_dir / "loo"

        run_pipeline(config, steps=STAGE1_STEPS)
        assert (stage1_dir / "user_sequences").exists()
        stage1_stat_mtime = (stage1_dir / "stats.json").stat().st_mtime

        run_pipeline(config, steps=STAGE2_STEPS)
        assert (stage2_dir / "train").exists()
        assert (stage1_dir / "stats.json").stat().st_mtime == stage1_stat_mtime

    def test_incremental_negative_sampling_only(self, tmp_path, amazon2015_raw_dir):
        """Run full pipeline, then re-run only negative_sampling step."""
        from datasets import load_from_disk

        from saegenrec.data.pipeline import run_pipeline

        config = self._make_config(tmp_path, amazon2015_raw_dir)
        stage2_dir = tmp_path / "interim" / "amazon2015" / "TestCat" / "loo"

        run_pipeline(config)
        first_train = load_from_disk(str(stage2_dir / "train"))

        config.processing.seed = 123
        run_pipeline(config, steps=["negative_sampling"])
        second_train = load_from_disk(str(stage2_dir / "train"))

        assert len(first_train) == len(second_train)
        assert "negative_item_ids" in second_train.column_names

    def test_missing_prerequisites_error(self, tmp_path):
        """Running Stage 2 without Stage 1 artifacts should raise FileNotFoundError."""
        from saegenrec.data.pipeline import run_pipeline

        config = PipelineConfig()
        config.dataset.name = "amazon2015"
        config.dataset.category = "TestCat"
        config.output.interim_dir = str(tmp_path / "interim")
        config.output.processed_dir = str(tmp_path / "processed")

        with pytest.raises(FileNotFoundError, match="Stage 1 artifacts not found"):
            run_pipeline(config, steps=["split", "augment"])

    def test_empty_data_after_filter(self, tmp_path):
        """Pipeline should handle gracefully when K-core filter removes all data."""
        from saegenrec.data.pipeline import run_pipeline
        from saegenrec.data.schemas import INTERACTIONS_FEATURES

        config = PipelineConfig()
        config.dataset.name = "amazon2015"
        config.dataset.category = "TestCat"
        config.processing.kcore_threshold = 100
        config.output.interim_dir = str(tmp_path / "interim")
        config.output.processed_dir = str(tmp_path / "processed")

        stage1_dir = tmp_path / "interim" / "amazon2015" / "TestCat"
        stage1_dir.mkdir(parents=True)

        sparse_ds = Dataset.from_dict(
            {
                "user_id": ["u1"],
                "item_id": ["i1"],
                "timestamp": [100],
                "rating": [5.0],
                "review_text": ["text"],
                "review_summary": ["summary"],
            },
            features=INTERACTIONS_FEATURES,
        )
        sparse_ds.save_to_disk(str(stage1_dir / "raw_interactions"))

        stats = run_pipeline(config, steps=["filter"])

        assert stats["filtered_interactions"] == 0
