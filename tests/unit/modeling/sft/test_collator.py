"""Tests for SFT data conversion utilities."""

from __future__ import annotations

from datasets import Dataset

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.collator import convert_to_conversational


class TestConvertToConversational:
    def test_basic_conversion(self):
        ds = Dataset.from_dict(
            {
                "task_type": ["seqrec", "item2index"],
                "instruction": ["Predict next item.", "Map title to SID."],
                "input": ["History: A B", "Title: Item"],
                "output": ["C", "SID_X"],
            },
            features=SFT_FEATURES,
        )
        result = convert_to_conversational(ds)

        assert "messages" in result.column_names
        assert "task_type" in result.column_names
        assert len(result) == 2

        row = result[0]
        assert row["messages"][0]["role"] == "user"
        assert "Predict next item." in row["messages"][0]["content"]
        assert "History: A B" in row["messages"][0]["content"]
        assert row["messages"][1]["role"] == "assistant"
        assert row["messages"][1]["content"] == "C"

    def test_preserves_task_type(self):
        ds = Dataset.from_dict(
            {
                "task_type": ["index2item"],
                "instruction": ["SID to title."],
                "input": ["SID: X"],
                "output": ["Item Name"],
            },
            features=SFT_FEATURES,
        )
        result = convert_to_conversational(ds)
        assert result[0]["task_type"] == "index2item"

    def test_empty_input_field(self):
        ds = Dataset.from_dict(
            {
                "task_type": ["seqrec"],
                "instruction": ["Predict next."],
                "input": [""],
                "output": ["X"],
            },
            features=SFT_FEATURES,
        )
        result = convert_to_conversational(ds)
        assert result[0]["messages"][0]["content"] == "Predict next."
