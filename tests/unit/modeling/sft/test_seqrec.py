"""Tests for SeqRec SFT task builder."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from saegenrec.data.schemas import INTERIM_SAMPLE_FEATURES, SFT_FEATURES, USER_SEQUENCES_FEATURES
from saegenrec.modeling.sft.seqrec import SeqRecTaskBuilder

TEMPLATES = {
    "seqrec": [
        {
            "instruction": "Predict next item.",
            "input_template": "History: {history_sids}",
            "output_template": "{target_sid}",
        },
        {
            "instruction": "What comes next?",
            "input_template": "Sequence: {history_sids}",
            "output_template": "{target_sid}",
        },
    ]
}


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    f = tmp_path / "templates.yaml"
    f.write_text(yaml.dump(TEMPLATES))
    return f


@pytest.fixture
def stage2_dir(tmp_path: Path, mock_train_sequences_dataset: Dataset) -> Path:
    d = tmp_path / "stage2"
    d.mkdir()
    mock_train_sequences_dataset.save_to_disk(str(d / "train_sequences"))
    return d


def _make_config(template_file: Path, **overrides) -> dict:
    cfg = {
        "template_file": str(template_file),
        "max_history_len": 20,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


class TestSeqRecTaskBuilder:
    def test_task_type(self):
        assert SeqRecTaskBuilder().task_type == "seqrec"

    def test_build_produces_alpaca_format(
        self, stage2_dir, template_file, mock_sid_map, tmp_path
    ):
        builder = SeqRecTaskBuilder()
        config = _make_config(template_file)
        ds = builder.build(tmp_path, stage2_dir, mock_sid_map, config)

        assert ds.features == SFT_FEATURES
        assert len(ds) == 3  # 3 users all have len > 1
        for row in ds:
            assert row["task_type"] == "seqrec"
            assert len(row["instruction"]) > 0
            assert len(row["input"]) > 0
            assert "<s_" in row["output"]

    def test_history_truncation(
        self, tmp_path, template_file, mock_sid_map
    ):
        long_seq = Dataset.from_dict(
            {
                "user_id": [0],
                "item_ids": [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
                "timestamps": [list(range(10))],
                "ratings": [[1.0] * 10],
                "review_texts": [[""] * 10],
                "review_summaries": [[""] * 10],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        stage2 = tmp_path / "stage2_trunc"
        stage2.mkdir()
        long_seq.save_to_disk(str(stage2 / "train_sequences"))

        config = _make_config(template_file, max_history_len=3)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2, mock_sid_map, config)

        assert len(ds) == 1
        sid_tokens_in_input = ds[0]["input"]
        sid_count = sid_tokens_in_input.count("<s_a_")
        assert sid_count == 3

    def test_sid_substitution(self, stage2_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2_dir, mock_sid_map, config)

        for row in ds:
            assert "<s_" in row["input"]
            assert "<s_" in row["output"]

    def test_template_randomization(self, stage2_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file, seed=0)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2_dir, mock_sid_map, config)
        instructions = {row["instruction"] for row in ds}
        assert len(instructions) >= 1

    def test_skip_single_interaction(self, tmp_path, template_file, mock_sid_map, caplog):
        single = Dataset.from_dict(
            {
                "user_id": [0, 1],
                "item_ids": [[0], [0, 1]],
                "timestamps": [[100], [100, 200]],
                "ratings": [[5.0], [5.0, 4.0]],
                "review_texts": [["r0"], ["r0", "r1"]],
                "review_summaries": [["s0"], ["s0", "s1"]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        stage2 = tmp_path / "stage2_single"
        stage2.mkdir()
        single.save_to_disk(str(stage2 / "train_sequences"))

        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2, mock_sid_map, config)
        assert len(ds) == 1

    def test_build_from_augmented_data(self, tmp_path, template_file, mock_sid_map):
        """When augmented train/ exists, use it instead of train_sequences."""
        augmented = Dataset.from_dict(
            {
                "user_id": [0, 0, 1, 1, 1],
                "history_item_ids": [[0], [0, 1], [2, 3], [1, 2, 3], [0, 1, 2, 3]],
                "history_item_titles": [
                    ["T0"], ["T0", "T1"], ["T2", "T3"], ["T1", "T2", "T3"],
                    ["T0", "T1", "T2", "T3"],
                ],
                "target_item_id": [1, 2, 4, 4, 4],
                "target_item_title": ["T1", "T2", "T4", "T4", "T4"],
            },
            features=INTERIM_SAMPLE_FEATURES,
        )
        stage2 = tmp_path / "stage2_aug"
        stage2.mkdir()
        augmented.save_to_disk(str(stage2 / "train"))

        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2, mock_sid_map, config)

        assert len(ds) == 5
        for row in ds:
            assert row["task_type"] == "seqrec"
            assert "<s_" in row["input"]
            assert "<s_" in row["output"]

    def test_augmented_preferred_over_sequences(self, tmp_path, template_file, mock_sid_map):
        """When both train/ and train_sequences/ exist, train/ is preferred."""
        augmented = Dataset.from_dict(
            {
                "user_id": [0, 0],
                "history_item_ids": [[0, 1], [0, 1, 2]],
                "history_item_titles": [["T0", "T1"], ["T0", "T1", "T2"]],
                "target_item_id": [2, 3],
                "target_item_title": ["T2", "T3"],
            },
            features=INTERIM_SAMPLE_FEATURES,
        )
        sequences = Dataset.from_dict(
            {
                "user_id": [0],
                "item_ids": [[0, 1, 2, 3]],
                "timestamps": [[100, 200, 300, 400]],
                "ratings": [[5.0, 4.0, 3.0, 2.0]],
                "review_texts": [["r0", "r1", "r2", "r3"]],
                "review_summaries": [["s0", "s1", "s2", "s3"]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        stage2 = tmp_path / "stage2_both"
        stage2.mkdir()
        augmented.save_to_disk(str(stage2 / "train"))
        sequences.save_to_disk(str(stage2 / "train_sequences"))

        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(tmp_path, stage2, mock_sid_map, config)

        assert len(ds) == 2  # from augmented, not 1 from sequences
