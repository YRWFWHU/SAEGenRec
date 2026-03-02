"""Training callbacks for recommendation metrics evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .evaluator import RecEvalResult, RecEvaluator


class RecMetricsCallback(TrainerCallback):
    """Periodically evaluates recommendation metrics during training.

    Runs greedy generation on validation SeqRec samples every
    ``rec_eval_steps`` global steps, computes HR@K/NDCG@K, and tracks
    the best checkpoint by HR@10.
    """

    def __init__(
        self,
        evaluator: RecEvaluator,
        eval_dataset: Dataset,
        rec_eval_steps: int = 500,
        eval_batch_size: int = 4,
    ):
        self.evaluator = evaluator
        self.eval_dataset = eval_dataset
        self.rec_eval_steps = rec_eval_steps
        self.eval_batch_size = eval_batch_size

        self.best_hr10: float = 0.0
        self.best_step: int = 0
        self.best_checkpoint_path: str | None = None
        self.eval_history: list[dict] = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.global_step % self.rec_eval_steps != 0 or state.global_step == 0:
            return

        result = self.evaluator.evaluate(
            model=model,
            eval_dataset=self.eval_dataset,
            step=state.global_step,
            batch_size=self.eval_batch_size,
        )

        self._log_metrics(result, state, args)
        self._track_best(result, state, args)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info(
            "Best HR@10: {:.4f} at step {} (checkpoint: {})",
            self.best_hr10,
            self.best_step,
            self.best_checkpoint_path,
        )

        history_path = Path(args.output_dir) / "rec_eval_history.json"
        with open(history_path, "w") as f:
            json.dump(self.eval_history, f, indent=2)
        logger.info("Saved rec eval history to {}", history_path)

    def _log_metrics(self, result: RecEvalResult, state: TrainerState, args: TrainingArguments):
        """Log recommendation metrics to the trainer's logger."""
        log_dict = {f"rec/{k}": v for k, v in result.metrics.items()}
        log_dict["rec/num_valid_predictions"] = result.num_valid_predictions
        log_dict["rec/num_samples"] = result.num_samples

        for key, value in log_dict.items():
            state.log_history.append({"step": state.global_step, key: value})

        self.eval_history.append({"step": result.step, **result.metrics})

    def _track_best(self, result: RecEvalResult, state: TrainerState, args: TrainingArguments):
        """Track best model by HR@10."""
        hr10 = result.metrics.get("hr@10", 0.0)
        if hr10 > self.best_hr10:
            self.best_hr10 = hr10
            self.best_step = state.global_step

            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            self.best_checkpoint_path = str(checkpoint_dir)

            logger.info(
                "New best HR@10: {:.4f} at step {} (checkpoint: {})",
                hr10,
                state.global_step,
                checkpoint_dir,
            )
