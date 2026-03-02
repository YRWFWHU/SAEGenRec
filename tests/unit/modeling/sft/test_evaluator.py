"""Tests for RecEvaluator."""

from __future__ import annotations

import pytest
from datasets import Dataset

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.collator import convert_to_conversational
from saegenrec.modeling.sft.evaluator import RecEvaluator, RecEvalResult


@pytest.fixture
def eval_seqrec_dataset() -> Dataset:
    """SeqRec eval samples in conversational format."""
    raw = Dataset.from_dict(
        {
            "task_type": ["seqrec", "seqrec"],
            "instruction": ["Predict next item."] * 2,
            "input": ["History: A B", "History: C D"],
            "output": ["<s_a_0><s_b_1><s_c_2><s_d_3>", "<s_a_0><s_b_2><s_c_3><s_d_1>"],
        },
        features=SFT_FEATURES,
    )
    return convert_to_conversational(raw)


class TestRecEvalResult:
    def test_defaults(self):
        r = RecEvalResult()
        assert r.step == 0
        assert r.num_samples == 0
        assert r.metrics == {}


class TestRecEvaluator:
    def test_init(self, mock_sid_map):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        evaluator = RecEvaluator(tokenizer, mock_sid_map)
        assert len(evaluator.all_sids) == 5
        assert evaluator.max_new_tokens == 32

    def test_compute_metrics_perfect(self, mock_sid_map):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        evaluator = RecEvaluator(tokenizer, mock_sid_map, eval_top_k=[1, 5, 10])

        predictions = [
            "<s_a_0><s_b_1><s_c_2><s_d_3>",
            "<s_a_0><s_b_2><s_c_3><s_d_1>",
        ]
        ground_truths = [
            "<s_a_0><s_b_1><s_c_2><s_d_3>",
            "<s_a_0><s_b_2><s_c_3><s_d_1>",
        ]

        metrics = evaluator._compute_metrics(predictions, ground_truths)
        assert metrics["hr@1"] == 1.0
        assert metrics["hr@10"] == 1.0

    def test_compute_metrics_no_match(self, mock_sid_map):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        evaluator = RecEvaluator(tokenizer, mock_sid_map, eval_top_k=[1, 10])

        predictions = ["wrong_output", "also_wrong"]
        ground_truths = [
            "<s_a_0><s_b_1><s_c_2><s_d_3>",
            "<s_a_0><s_b_2><s_c_3><s_d_1>",
        ]

        metrics = evaluator._compute_metrics(predictions, ground_truths)
        assert metrics["hr@1"] == 0.0
        assert metrics["hr@10"] == 0.0

    def test_normalize_sid(self, mock_sid_map):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        evaluator = RecEvaluator(tokenizer, mock_sid_map)
        assert evaluator._normalize_sid("<s_a_0> <s_b_1>") == "<s_a_0><s_b_1>"
        assert evaluator._normalize_sid("  <s_a_0><s_b_1>  ") == "<s_a_0><s_b_1>"

    def test_empty_eval_dataset(self, mock_sid_map):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
        )
        evaluator = RecEvaluator(tokenizer, mock_sid_map)

        empty_ds = Dataset.from_dict(
            {"messages": [], "task_type": []}
        )
        from unittest.mock import MagicMock

        model = MagicMock()
        result = evaluator.evaluate(model, empty_ds, step=0)
        assert result.num_samples == 0
