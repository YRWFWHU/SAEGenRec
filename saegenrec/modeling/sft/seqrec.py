"""SeqRec (sequence recommendation) SFT task builder."""

from __future__ import annotations

import random
from pathlib import Path
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
    ) -> Dataset:
        templates = self.load_templates(config["template_file"])
        max_history_len = config.get("max_history_len", 20)
        seed = config.get("seed", 42)
        rng = random.Random(seed)

        sid_lookup: dict[int, str] = {row["item_id"]: row["sid_tokens"] for row in sid_map}

        augmented_path = stage2_dir / "train"
        if augmented_path.exists():
            train_data = load_from_disk(str(augmented_path))
            return self._build_from_augmented(train_data, sid_lookup, templates, max_history_len, rng)

        logger.warning(
            "Augmented train data not found at {}, falling back to train_sequences",
            augmented_path,
        )
        train_sequences = load_from_disk(str(stage2_dir / "train_sequences"))
        return self._build_from_sequences(train_sequences, sid_lookup, templates, max_history_len, rng)

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
