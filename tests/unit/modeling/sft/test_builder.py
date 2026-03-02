"""Tests for SFTDatasetBuilder orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.builder import SFTDatasetBuilder

TEMPLATES = {
    "seqrec": [
        {
            "instruction": "Predict next.",
            "input_template": "H: {history_sids}",
            "output_template": "{target_sid}",
        },
    ],
    "item2index": [
        {
            "instruction": "Title to SID.",
            "input_template": "T: {title}",
            "output_template": "{sid_tokens}",
        },
    ],
    "index2item": [
        {
            "instruction": "SID to title.",
            "input_template": "S: {sid_tokens}",
            "output_template": "{title}",
        },
    ],
}


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    f = tmp_path / "templates.yaml"
    f.write_text(yaml.dump(TEMPLATES))
    return f


@pytest.fixture
def stage1_dir(
    tmp_path: Path,
    mock_item_metadata_dataset: Dataset,
    mock_item_id_map_dataset: Dataset,
) -> Path:
    d = tmp_path / "stage1"
    d.mkdir()
    mock_item_metadata_dataset.save_to_disk(str(d / "item_metadata"))
    mock_item_id_map_dataset.save_to_disk(str(d / "item_id_map"))
    return d


@pytest.fixture
def stage2_dir(tmp_path: Path, mock_train_sequences_dataset: Dataset) -> Path:
    d = tmp_path / "stage2"
    d.mkdir()
    mock_train_sequences_dataset.save_to_disk(str(d / "train_sequences"))
    return d


def _make_config(template_file: Path, **overrides) -> dict:
    cfg = {
        "tasks": ["seqrec", "item2index", "index2item"],
        "template_file": str(template_file),
        "max_history_len": 20,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


class TestSFTDatasetBuilder:
    def test_multi_task_merge(
        self, stage1_dir, stage2_dir, template_file, mock_sid_map, tmp_path
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_config(template_file)

        ds = SFTDatasetBuilder().build(
            stage1_dir, stage2_dir, mock_sid_map, output_dir, config
        )

        task_types = set(ds["task_type"])
        assert "seqrec" in task_types
        assert "item2index" in task_types
        assert "index2item" in task_types
        assert len(ds) == 3 + 5 + 5  # 3 seqrec + 5 item2index + 5 index2item

    def test_task_weights_sampling(
        self, stage1_dir, stage2_dir, template_file, mock_sid_map, tmp_path
    ):
        output_dir = tmp_path / "output_w"
        output_dir.mkdir()
        config = _make_config(template_file, task_weights={"item2index": 0.4})

        ds = SFTDatasetBuilder().build(
            stage1_dir, stage2_dir, mock_sid_map, output_dir, config
        )

        item2index_count = sum(1 for r in ds if r["task_type"] == "item2index")
        assert item2index_count < 5
        assert item2index_count >= 1

    def test_output_schema_validation(
        self, stage1_dir, stage2_dir, template_file, mock_sid_map, tmp_path
    ):
        output_dir = tmp_path / "output_s"
        output_dir.mkdir()
        config = _make_config(template_file)

        ds = SFTDatasetBuilder().build(
            stage1_dir, stage2_dir, mock_sid_map, output_dir, config
        )

        assert ds.features == SFT_FEATURES
        for row in ds:
            assert isinstance(row["task_type"], str)
            assert isinstance(row["instruction"], str)
            assert isinstance(row["input"], str)
            assert isinstance(row["output"], str)

    def test_output_saved_to_disk(
        self, stage1_dir, stage2_dir, template_file, mock_sid_map, tmp_path
    ):
        output_dir = tmp_path / "output_d"
        output_dir.mkdir()
        config = _make_config(template_file)

        SFTDatasetBuilder().build(
            stage1_dir, stage2_dir, mock_sid_map, output_dir, config
        )

        assert (output_dir / "sft_data").exists()
