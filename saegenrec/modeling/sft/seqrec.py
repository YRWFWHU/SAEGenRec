"""SeqRec (sequence recommendation) SFT task builder."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from datasets import Dataset, load_from_disk
from loguru import logger

from saegenrec.data.schemas import SFT_FEATURES

from .base import SFTTaskBuilder, register_sft_task


@register_sft_task("seqrec")
class SeqRecTaskBuilder(SFTTaskBuilder):
    @property
    def task_type(self) -> str:
        return "seqrec"

    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        config: dict[str, Any],
        split: str | None = None,
    ) -> Dataset:
        templates = self.load_templates(config["template_file"])
        max_history_len = config.get("max_history_len", 20)
        seed = config.get("seed", 42)
        rng = random.Random(seed)

        sid_lookup: dict[int, str] = {row["item_id"]: row["sid_tokens"] for row in sid_map}

        split_name = split or "train"

        if split_name in ("valid", "test"):
            return self._build_eval_split(
                stage2_dir, split_name, sid_lookup, templates, max_history_len, rng
            )

        augmented_path = stage2_dir / split_name
        if augmented_path.exists():
            data = load_from_disk(str(augmented_path))
            if "history_item_ids" in data.column_names:
                return self._build_from_augmented(
                    data, sid_lookup, templates, max_history_len, rng
                )

        sequences_path = stage2_dir / f"{split_name}_sequences"
        if sequences_path.exists():
            sequences = load_from_disk(str(sequences_path))
            return self._build_from_sequences(
                sequences, sid_lookup, templates, max_history_len, rng
            )

        logger.warning(
            "No data found for split '{}' at {} or {}, returning empty dataset",
            split_name,
            augmented_path,
            sequences_path,
        )
        return self._empty_dataset()

    @staticmethod
    def _empty_dataset() -> Dataset:
        return Dataset.from_dict(
            {k: [] for k in ("task_type", "instruction", "input", "output")},
            features=SFT_FEATURES,
        )

    def _build_eval_split(
        self,
        stage2_dir: Path,
        split: str,
        sid_lookup: dict[int, str],
        templates: list[dict],
        max_history_len: int,
        rng: random.Random,
    ) -> Dataset:
        """Build eval samples by reconstructing full sequences from LOO split parts.

        In LOO split, valid/test sequences only store the hold-out item.
        This method reconstructs the full (history, target) pair:
          - valid: history = train items,                  target = valid item
          - test:  history = train items + valid item,     target = test item
        """
        train_seq_path = stage2_dir / "train_sequences"
        if not train_seq_path.exists():
            logger.warning(
                "train_sequences not found at {}, cannot build eval split", train_seq_path
            )
            return self._empty_dataset()

        train_seqs = load_from_disk(str(train_seq_path))
        user_train_history: dict[int, list[int]] = {
            row["user_id"]: list(row["item_ids"]) for row in train_seqs
        }

        user_valid_item: dict[int, int] = {}
        if split == "test":
            valid_seq_path = stage2_dir / "valid_sequences"
            if valid_seq_path.exists():
                valid_seqs = load_from_disk(str(valid_seq_path))
                for row in valid_seqs:
                    items = row["item_ids"]
                    if items:
                        user_valid_item[row["user_id"]] = items[-1]

        target_seq_path = stage2_dir / f"{split}_sequences"
        if not target_seq_path.exists():
            logger.warning(
                "{}_sequences not found at {}, cannot build eval split",
                split,
                target_seq_path,
            )
            return self._empty_dataset()

        target_seqs = load_from_disk(str(target_seq_path))
        targets: dict[int, int] = {}
        for row in target_seqs:
            items = row["item_ids"]
            if items:
                targets[row["user_id"]] = items[-1]

        records: list[dict[str, str]] = []
        skipped = 0
        for uid, target_id in targets.items():
            history = list(user_train_history.get(uid, []))

            if split == "test" and uid in user_valid_item:
                history.append(user_valid_item[uid])

            if not history or target_id not in sid_lookup:
                skipped += 1
                continue

            unmapped = [iid for iid in history if iid not in sid_lookup]
            if unmapped:
                skipped += 1
                continue

            if len(history) > max_history_len:
                history = history[-max_history_len:]

            history_sids = " ".join(sid_lookup[iid] for iid in history)
            target_sid = sid_lookup[target_id]

            tpl = rng.choice(templates)
            records.append(
                {
                    "task_type": self.task_type,
                    "instruction": tpl["instruction"],
                    "input": tpl["input_template"].format(history_sids=history_sids),
                    "output": tpl["output_template"].format(target_sid=target_sid),
                }
            )

        if skipped:
            logger.info(
                "Eval split '{}': skipped {} samples (unmapped IDs or empty history)",
                split,
                skipped,
            )
        logger.info("Eval split '{}': built {} seqrec samples", split, len(records))

        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in SFT_FEATURES},
            features=SFT_FEATURES,
        )

    def _build_from_augmented(
        self,
        train_data: Dataset,
        sid_lookup: dict[int, str],
        templates: list[dict],
        max_history_len: int,
        rng: random.Random,
    ) -> Dataset:
        """Build from sliding-window augmented data (history_item_ids → target_item_id)."""
        records: list[dict[str, str]] = []
        skipped = 0
        for row in train_data:
            history: list[int] = row["history_item_ids"]
            target: int = row["target_item_id"]

            if not history or target not in sid_lookup:
                skipped += 1
                continue

            unmapped = [iid for iid in history if iid not in sid_lookup]
            if unmapped:
                skipped += 1
                continue

            if len(history) > max_history_len:
                history = history[-max_history_len:]

            history_sids = " ".join(sid_lookup[iid] for iid in history)
            target_sid = sid_lookup[target]

            tpl = rng.choice(templates)
            records.append(
                {
                    "task_type": self.task_type,
                    "instruction": tpl["instruction"],
                    "input": tpl["input_template"].format(history_sids=history_sids),
                    "output": tpl["output_template"].format(target_sid=target_sid),
                }
            )

        if skipped:
            logger.info("Skipped {} samples with unmapped item IDs", skipped)

        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in SFT_FEATURES},
            features=SFT_FEATURES,
        )

    def _build_from_sequences(
        self,
        train_sequences: Dataset,
        sid_lookup: dict[int, str],
        templates: list[dict],
        max_history_len: int,
        rng: random.Random,
    ) -> Dataset:
        """Fallback: build from raw train_sequences (one sample per user)."""
        records: list[dict[str, str]] = []
        for row in train_sequences:
            item_ids: list[int] = row["item_ids"]
            if len(item_ids) <= 1:
                logger.warning(
                    "Skipping user {} with {} interaction(s)",
                    row["user_id"],
                    len(item_ids),
                )
                continue

            history = item_ids[:-1]
            if len(history) > max_history_len:
                history = history[-max_history_len:]
            target = item_ids[-1]

            history_sids = " ".join(sid_lookup[iid] for iid in history)
            target_sid = sid_lookup[target]

            tpl = rng.choice(templates)
            records.append(
                {
                    "task_type": self.task_type,
                    "instruction": tpl["instruction"],
                    "input": tpl["input_template"].format(history_sids=history_sids),
                    "output": tpl["output_template"].format(target_sid=target_sid),
                }
            )

        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in SFT_FEATURES},
            features=SFT_FEATURES,
        )
