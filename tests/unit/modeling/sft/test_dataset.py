"""Tests for SFT dataset loading and SID token injection."""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.collator import convert_to_conversational
from saegenrec.modeling.sft.dataset import add_sid_tokens_to_tokenizer, load_sft_dataset


@pytest.fixture
def mock_sft_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "task_type": ["seqrec", "item2index", "seqrec"],
            "instruction": [
                "Predict next item.",
                "Map title to SID.",
                "Predict next item.",
            ],
            "input": [
                "History: <s_a_0> <s_b_1>",
                "Title: Item One",
                "History: <s_a_1> <s_b_0>",
            ],
            "output": ["<s_a_0><s_b_2>", "<s_a_1><s_b_1>", "<s_a_0><s_b_1>"],
        },
        features=SFT_FEATURES,
    )


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TestConvertToConversational:
    def test_structure(self, mock_sft_dataset):
        ds = convert_to_conversational(mock_sft_dataset)
        assert "messages" in ds.column_names
        assert "task_type" in ds.column_names
        assert "instruction" not in ds.column_names

    def test_messages_format(self, mock_sft_dataset):
        ds = convert_to_conversational(mock_sft_dataset)
        row = ds[0]
        assert len(row["messages"]) == 2
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert "Predict next item." in row["messages"][0]["content"]

    def test_empty_input(self):
        ds = Dataset.from_dict(
            {
                "task_type": ["seqrec"],
                "instruction": ["Predict next."],
                "input": [""],
                "output": ["X"],
            },
            features=SFT_FEATURES,
        )
        conv = convert_to_conversational(ds)
        assert conv[0]["messages"][0]["content"] == "Predict next."


class TestAddSIDTokens:
    def test_adds_tokens(self, tokenizer, mock_sid_map):
        original_size = len(tokenizer)
        new_tokens = add_sid_tokens_to_tokenizer(tokenizer, mock_sid_map)
        assert len(tokenizer) > original_size
        assert len(new_tokens) > 0
        for tok in new_tokens:
            assert tok.startswith("<") and tok.endswith(">")


class TestLoadSFTDataset:
    def test_load_and_convert(self, mock_sft_dataset, tmp_path: Path):
        save_dir = tmp_path / "sft_data"
        mock_sft_dataset.save_to_disk(str(save_dir))

        ds = load_sft_dataset(save_dir)
        assert "messages" in ds.column_names
        assert len(ds) == 3

    def test_load_with_task_filter(self, mock_sft_dataset, tmp_path: Path):
        save_dir = tmp_path / "sft_data"
        mock_sft_dataset.save_to_disk(str(save_dir))

        ds = load_sft_dataset(save_dir, tasks=["seqrec"])
        assert len(ds) == 2

    def test_load_with_split(self, mock_sft_dataset, tmp_path: Path):
        save_dir = tmp_path / "sft_data" / "train"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        mock_sft_dataset.save_to_disk(str(save_dir))

        ds = load_sft_dataset(tmp_path / "sft_data", split="train")
        assert len(ds) == 3
