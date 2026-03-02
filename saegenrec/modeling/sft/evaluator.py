"""Recommendation evaluator — greedy generation + HR@K / NDCG@K calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from datasets import Dataset
from loguru import logger
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class RecEvalResult:
    """Container for recommendation evaluation results."""

    step: int = 0
    num_samples: int = 0
    num_valid_predictions: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class RecEvaluator:
    """Evaluates recommendation quality via greedy generation.

    Given a validation/test set of SeqRec samples, generates the model's
    prediction and compares against ground truth SIDs to compute HR@K
    and NDCG@K.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sid_map: Dataset,
        eval_top_k: list[int] | None = None,
        max_new_tokens: int = 32,
    ):
        self.tokenizer = tokenizer
        self.eval_top_k = eval_top_k or [1, 5, 10]
        self.max_new_tokens = max_new_tokens

        self.sid_to_item: dict[str, int] = {}
        for row in sid_map:
            self.sid_to_item[row["sid_tokens"]] = row["item_id"]

        self.all_sids: set[str] = set(self.sid_to_item.keys())

    def evaluate(
        self,
        model: PreTrainedModel,
        eval_dataset: Dataset,
        step: int = 0,
        batch_size: int = 4,
        logits_processor: list | None = None,
        num_beams: int = 1,
    ) -> RecEvalResult:
        """Run evaluation on SeqRec samples.

        Args:
            model: The model to evaluate.
            eval_dataset: Dataset in conversational format (``messages`` + ``task_type``).
            step: Current training step (for logging).
            batch_size: Batch size for generation.
            logits_processor: Optional list of LogitsProcessors (e.g. constrained decoding).
            num_beams: Number of beams for beam search (1 = greedy).

        Returns:
            RecEvalResult with HR@K, NDCG@K metrics.
        """
        seqrec_samples = [row for row in eval_dataset if row["task_type"] == "seqrec"]

        if not seqrec_samples:
            logger.warning("No SeqRec samples found in eval dataset, skipping rec eval")
            return RecEvalResult(step=step)

        model.eval()
        device = next(model.parameters()).device

        predictions: list[str] = []
        ground_truths: list[str] = []

        for i in range(0, len(seqrec_samples), batch_size):
            batch = seqrec_samples[i : i + batch_size]
            prompts = []
            targets = []

            for sample in batch:
                messages = sample["messages"]
                user_msg = [m for m in messages if m["role"] == "user"]
                assistant_msg = [m for m in messages if m["role"] == "assistant"]

                prompt_messages = user_msg
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_text)
                targets.append(assistant_msg[0]["content"] if assistant_msg else "")

            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "num_beams": num_beams,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if logits_processor:
                generate_kwargs["logits_processor"] = logits_processor

            with torch.no_grad():
                outputs = model.generate(**encoded, **generate_kwargs)

            for j, output in enumerate(outputs):
                input_len = encoded["input_ids"][j].shape[0]
                generated_ids = output[input_len:]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                predictions.append(generated_text)
                ground_truths.append(targets[j])

        metrics = self._compute_metrics(predictions, ground_truths)
        num_valid = sum(1 for p in predictions if self._normalize_sid(p) in self.all_sids)

        result = RecEvalResult(
            step=step,
            num_samples=len(seqrec_samples),
            num_valid_predictions=num_valid,
            metrics=metrics,
        )

        logger.info(
            "RecEval@step {}: samples={}, valid_preds={}/{}, {}",
            step,
            result.num_samples,
            num_valid,
            len(predictions),
            {k: f"{v:.4f}" for k, v in metrics.items()},
        )

        return result

    def _compute_metrics(
        self, predictions: list[str], ground_truths: list[str]
    ) -> dict[str, float]:
        """Compute HR@K and NDCG@K."""
        metrics: dict[str, float] = {}

        for k in self.eval_top_k:
            hits = 0
            ndcg_sum = 0.0

            for pred, gt in zip(predictions, ground_truths):
                pred_sid = self._normalize_sid(pred)
                gt_sid = self._normalize_sid(gt)

                if pred_sid == gt_sid and pred_sid:
                    hits += 1
                    ndcg_sum += 1.0 / math.log2(2)

            n = len(predictions) if predictions else 1
            metrics[f"hr@{k}"] = hits / n
            metrics[f"ndcg@{k}"] = ndcg_sum / n

        return metrics

    def _normalize_sid(self, text: str) -> str:
        """Normalize SID text for comparison."""
        return text.replace(" ", "").strip()
