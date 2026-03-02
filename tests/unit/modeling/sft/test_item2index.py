"""Tests for Item2Index and Index2Item SFT task builders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.index2item import Index2ItemTaskBuilder
from saegenrec.modeling.sft.item2index import Item2IndexTaskBuilder

TEMPLATES = {
    "item2index": [
        {
            "instruction": "Predict semantic ID.",
            "input_template": "Title: {title}",
            "output_template": "{sid_tokens}",
        },
    ],
    "index2item": [
        {
            "instruction": "Predict item title.",
            "input_template": "SID: {sid_tokens}",
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


def _make_config(template_file: Path, **overrides) -> dict:
    cfg = {"template_file": str(template_file), "seed": 42}
    cfg.update(overrides)
    return cfg


class TestItem2IndexTaskBuilder:
    def test_task_type(self):
        assert Item2IndexTaskBuilder().task_type == "item2index"

    def test_all_items_covered(self, stage1_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file)
        ds = Item2IndexTaskBuilder().build(stage1_dir, tmp_path, mock_sid_map, config)

        assert len(ds) == 5
        assert ds.features == SFT_FEATURES
        for row in ds:
            assert row["task_type"] == "item2index"

    def test_output_contains_sid_tokens(self, stage1_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file)
        ds = Item2IndexTaskBuilder().build(stage1_dir, tmp_path, mock_sid_map, config)

        for row in ds:
            assert "<s_" in row["output"]
            assert "Item" in row["input"] or "Title" in row["input"]


class TestIndex2ItemTaskBuilder:
    def test_task_type(self):
        assert Index2ItemTaskBuilder().task_type == "index2item"

    def test_all_items_covered(self, stage1_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file)
        ds = Index2ItemTaskBuilder().build(stage1_dir, tmp_path, mock_sid_map, config)

        assert len(ds) == 5
        assert ds.features == SFT_FEATURES
        for row in ds:
            assert row["task_type"] == "index2item"

    def test_output_contains_title(self, stage1_dir, template_file, mock_sid_map, tmp_path):
        config = _make_config(template_file)
        ds = Index2ItemTaskBuilder().build(stage1_dir, tmp_path, mock_sid_map, config)

        titles = {"Item Zero", "Item One", "Item Two", "Item Three", "Item Four"}
        for row in ds:
            assert row["output"] in titles
            assert "<s_" in row["input"]
