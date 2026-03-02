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


class TestSeqRecEvalSplit:
    """Tests for LOO eval split reconstruction (_build_eval_split)."""

    @pytest.fixture
    def loo_stage2(self, tmp_path: Path) -> Path:
        """Simulate LOO split output: train has full history, valid/test have single item."""
        stage2 = tmp_path / "stage2_loo"
        stage2.mkdir()

        # train_sequences: user history (all items except last two)
        # User 0: full seq [0,1,2,3,4] → train=[0,1,2], valid_item=3, test_item=4
        # User 1: full seq [1,2,3]     → train=[1],     valid_item=2, test_item=3
        # User 2: full seq [0,2,4]     → train=[0],     valid_item=2, test_item=4
        train_seqs = Dataset.from_dict(
            {
                "user_id": [0, 1, 2],
                "item_ids": [[0, 1, 2], [1], [0]],
                "timestamps": [[100, 200, 300], [150], [110]],
                "ratings": [[5.0, 4.0, 3.0], [3.0], [4.0]],
                "review_texts": [["r0", "r1", "r2"], ["r5"], ["r8"]],
                "review_summaries": [["s0", "s1", "s2"], ["s5"], ["s8"]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        valid_seqs = Dataset.from_dict(
            {
                "user_id": [0, 1, 2],
                "item_ids": [[3], [2], [2]],
                "timestamps": [[400], [250], [410]],
                "ratings": [[2.0], [5.0], [3.0]],
                "review_texts": [["r3"], ["r6"], ["r9"]],
                "review_summaries": [["s3"], ["s6"], ["s9"]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        test_seqs = Dataset.from_dict(
            {
                "user_id": [0, 1, 2],
                "item_ids": [[4], [3], [4]],
                "timestamps": [[500], [350], [510]],
                "ratings": [[1.0], [4.0], [5.0]],
                "review_texts": [["r4"], ["r7"], ["r10"]],
                "review_summaries": [["s4"], ["s7"], ["s10"]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        train_seqs.save_to_disk(str(stage2 / "train_sequences"))
        valid_seqs.save_to_disk(str(stage2 / "valid_sequences"))
        test_seqs.save_to_disk(str(stage2 / "test_sequences"))
        return stage2

    def test_valid_split_uses_train_history(
        self, loo_stage2, template_file, mock_sid_map, tmp_path
    ):
        """Valid split: history = train items, target = valid hold-out item."""
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(
            tmp_path, loo_stage2, mock_sid_map, config, split="valid"
        )

        assert ds.features == SFT_FEATURES
        assert len(ds) == 3
        for row in ds:
            assert row["task_type"] == "seqrec"
            assert "<s_" in row["input"]
            assert "<s_" in row["output"]

    def test_test_split_includes_valid_item_in_history(
        self, loo_stage2, template_file, mock_sid_map, tmp_path
    ):
        """Test split: history = train items + valid item, target = test item."""
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(
            tmp_path, loo_stage2, mock_sid_map, config, split="test"
        )

        assert ds.features == SFT_FEATURES
        assert len(ds) == 3
        for row in ds:
            assert row["task_type"] == "seqrec"
            assert "<s_" in row["input"]
            assert "<s_" in row["output"]

    def test_valid_history_content_matches_train_items(
        self, loo_stage2, template_file, mock_sid_map, tmp_path
    ):
        """Verify the valid split history actually contains train sequence SIDs."""
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(
            tmp_path, loo_stage2, mock_sid_map, config, split="valid"
        )
        sid_lookup = {r["item_id"]: r["sid_tokens"] for r in mock_sid_map}

        # User 0: train=[0,1,2], valid_target=3
        # Find the sample whose output matches target=3's SID
        target_3_sid = sid_lookup[3]
        user0_sample = [r for r in ds if r["output"] == target_3_sid]
        assert len(user0_sample) == 1
        inp = user0_sample[0]["input"]
        assert sid_lookup[0] in inp
        assert sid_lookup[1] in inp
        assert sid_lookup[2] in inp

    def test_test_history_includes_valid_holdout(
        self, loo_stage2, template_file, mock_sid_map, tmp_path
    ):
        """Verify test split history includes both train items and valid hold-out."""
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(
            tmp_path, loo_stage2, mock_sid_map, config, split="test"
        )
        sid_lookup = {r["item_id"]: r["sid_tokens"] for r in mock_sid_map}

        # User 0: train=[0,1,2], valid_item=3, test_target=4
        # User 2: train=[0],     valid_item=2, test_target=4
        # Both target item 4 — find the one with longer history (User 0)
        target_4_sid = sid_lookup[4]
        user0_candidates = [r for r in ds if r["output"] == target_4_sid]
        assert len(user0_candidates) == 2
        user0_sample = max(user0_candidates, key=lambda r: len(r["input"]))
        inp = user0_sample["input"]
        assert sid_lookup[0] in inp
        assert sid_lookup[1] in inp
        assert sid_lookup[2] in inp
        assert sid_lookup[3] in inp  # valid hold-out included

    def test_eval_split_missing_train_sequences_returns_empty(
        self, tmp_path, template_file, mock_sid_map
    ):
        stage2 = tmp_path / "stage2_empty"
        stage2.mkdir()
        config = _make_config(template_file)
        ds = SeqRecTaskBuilder().build(
            tmp_path, stage2, mock_sid_map, config, split="valid"
        )
        assert len(ds) == 0

    def test_eval_split_history_truncation(
        self, tmp_path, template_file, mock_sid_map
    ):
        """Eval split respects max_history_len truncation."""
        stage2 = tmp_path / "stage2_trunc_eval"
        stage2.mkdir()

        train_seqs = Dataset.from_dict(
            {
                "user_id": [0],
                "item_ids": [[0, 1, 2, 3, 4, 0, 1, 2]],
                "timestamps": [list(range(8))],
                "ratings": [[1.0] * 8],
                "review_texts": [[""] * 8],
                "review_summaries": [[""] * 8],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        valid_seqs = Dataset.from_dict(
            {
                "user_id": [0],
                "item_ids": [[3]],
                "timestamps": [[100]],
                "ratings": [[5.0]],
                "review_texts": [[""]],
                "review_summaries": [[""]],
            },
            features=USER_SEQUENCES_FEATURES,
        )
        train_seqs.save_to_disk(str(stage2 / "train_sequences"))
        valid_seqs.save_to_disk(str(stage2 / "valid_sequences"))

        config = _make_config(template_file, max_history_len=3)
        ds = SeqRecTaskBuilder().build(
            tmp_path, stage2, mock_sid_map, config, split="valid"
        )
        assert len(ds) == 1
        sid_count = ds[0]["input"].count("<s_a_")
        assert sid_count == 3
