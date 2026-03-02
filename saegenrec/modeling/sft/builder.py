"""SFT dataset builder — orchestrates multi-task SFT data generation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from datasets import Dataset, concatenate_datasets
from loguru import logger

from .base import get_sft_task_builder


class SFTDatasetBuilder:
    """Builds a merged, shuffled SFT dataset from multiple task builders."""

    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        output_dir: Path,
        config: dict[str, Any],
        splits: list[str] | None = None,
    ) -> Dataset | dict[str, Dataset]:
        """Build SFT dataset(s) for the given split(s).

        Args:
            stage1_dir: Stage 1 data directory (item metadata, ID map).
            stage2_dir: Stage 2 data directory (train/valid/test sequences).
            sid_map: HF Dataset with item_id, codes, sid_tokens columns.
            output_dir: Directory to save the built SFT dataset(s).
            config: SFT builder config dict.
            splits: List of splits to build (e.g. ["train", "valid", "test"]).
                When ``None``, builds a single dataset from train data
                (legacy behaviour) and returns a ``Dataset``.  When provided,
                returns ``dict[str, Dataset]``.
        """
        if splits is None:
            return self._build_single(stage1_dir, stage2_dir, sid_map, output_dir, config)

        results: dict[str, Dataset] = {}
        for split in splits:
            logger.info("Building SFT data for split '{}'", split)
            ds = self._build_single(
                stage1_dir, stage2_dir, sid_map, output_dir, config, split=split
            )
            results[split] = ds
        return results

    def _build_single(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        output_dir: Path,
        config: dict[str, Any],
        split: str | None = None,
    ) -> Dataset:
        tasks: list[str] = config["tasks"]
        task_weights: dict[str, float] = config.get("task_weights", {})
        seed: int = config.get("seed", 42)

        task_datasets: list[Dataset] = []
        for task_name in tasks:
            builder = get_sft_task_builder(task_name)

            if split is not None and task_name == "seqrec":
                ds = builder.build(stage1_dir, stage2_dir, sid_map, config, split=split)
            else:
                ds = builder.build(stage1_dir, stage2_dir, sid_map, config)

            logger.info(
                "Task '{}' (split={}) produced {} samples",
                task_name,
                split or "default",
                len(ds),
            )

            if task_name in task_weights and 0 < task_weights[task_name] < 1.0:
                n_samples = max(1, int(len(ds) * task_weights[task_name]))
                rng = random.Random(seed)
                indices = rng.sample(range(len(ds)), n_samples)
                ds = ds.select(indices)
                logger.info(
                    "Task '{}' subsampled to {} samples (weight={})",
                    task_name,
                    len(ds),
                    task_weights[task_name],
                )

            task_datasets.append(ds)

        merged = concatenate_datasets(task_datasets)
        merged = merged.shuffle(seed=seed)

        if split is not None:
            save_path = output_dir / "sft_data" / split
        else:
            save_path = output_dir / "sft_data"
        merged.save_to_disk(str(save_path))
        logger.info("Saved {} SFT samples to {}", len(merged), save_path)

        return merged
