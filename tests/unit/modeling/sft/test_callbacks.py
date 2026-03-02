"""Tests for RecMetricsCallback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from transformers import TrainerControl, TrainerState, TrainingArguments

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.callbacks import RecMetricsCallback
from saegenrec.modeling.sft.collator import convert_to_conversational
from saegenrec.modeling.sft.evaluator import RecEvalResult, RecEvaluator


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=RecEvaluator)
    evaluator.evaluate.return_value = RecEvalResult(
        step=100,
        num_samples=10,
        num_valid_predictions=8,
        metrics={"hr@1": 0.5, "hr@5": 0.7, "hr@10": 0.8, "ndcg@5": 0.6, "ndcg@10": 0.65},
    )
    return evaluator


@pytest.fixture
def eval_dataset():
    raw = Dataset.from_dict(
        {
            "task_type": ["seqrec"],
            "instruction": ["Predict."],
            "input": ["History: A"],
            "output": ["B"],
        },
        features=SFT_FEATURES,
    )
    return convert_to_conversational(raw)


@pytest.fixture
def callback(mock_evaluator, eval_dataset):
    return RecMetricsCallback(
        evaluator=mock_evaluator,
        eval_dataset=eval_dataset,
        rec_eval_steps=100,
        eval_batch_size=4,
    )


class TestRecMetricsCallback:
    def test_skips_non_eval_step(self, callback, mock_evaluator):
        state = TrainerState(global_step=50)
        args = MagicMock(spec=TrainingArguments)
        control = TrainerControl()

        callback.on_step_end(args, state, control, model=MagicMock())
        mock_evaluator.evaluate.assert_not_called()

    def test_evaluates_at_rec_eval_step(self, callback, mock_evaluator):
        state = TrainerState(global_step=100)
        state.log_history = []
        args = MagicMock(spec=TrainingArguments)
        args.output_dir = "/tmp/test_output"
        control = TrainerControl()

        callback.on_step_end(args, state, control, model=MagicMock())
        mock_evaluator.evaluate.assert_called_once()

    def test_tracks_best_hr10(self, callback, mock_evaluator):
        state = TrainerState(global_step=100)
        state.log_history = []
        args = MagicMock(spec=TrainingArguments)
        args.output_dir = "/tmp/test_output"
        control = TrainerControl()

        callback.on_step_end(args, state, control, model=MagicMock())
        assert callback.best_hr10 == 0.8
        assert callback.best_step == 100

    def test_updates_best_on_improvement(self, callback, mock_evaluator):
        state = TrainerState(global_step=100)
        state.log_history = []
        args = MagicMock(spec=TrainingArguments)
        args.output_dir = "/tmp/test_output"
        control = TrainerControl()

        callback.on_step_end(args, state, control, model=MagicMock())
        assert callback.best_hr10 == 0.8

        mock_evaluator.evaluate.return_value = RecEvalResult(
            step=200, num_samples=10, num_valid_predictions=9,
            metrics={"hr@1": 0.6, "hr@5": 0.8, "hr@10": 0.9, "ndcg@5": 0.7, "ndcg@10": 0.75},
        )
        state.global_step = 200
        callback.on_step_end(args, state, control, model=MagicMock())
        assert callback.best_hr10 == 0.9
        assert callback.best_step == 200

    def test_saves_history_on_train_end(self, callback, tmp_path: Path):
        callback.eval_history = [
            {"step": 100, "hr@10": 0.8},
            {"step": 200, "hr@10": 0.9},
        ]
        args = MagicMock(spec=TrainingArguments)
        args.output_dir = str(tmp_path)
        state = TrainerState()
        control = TrainerControl()

        callback.on_train_end(args, state, control)

        history_path = tmp_path / "rec_eval_history.json"
        assert history_path.exists()
        with open(history_path) as f:
            history = json.load(f)
        assert len(history) == 2

    def test_skips_step_zero(self, callback, mock_evaluator):
        state = TrainerState(global_step=0)
        args = MagicMock(spec=TrainingArguments)
        control = TrainerControl()

        callback.on_step_end(args, state, control, model=MagicMock())
        mock_evaluator.evaluate.assert_not_called()
