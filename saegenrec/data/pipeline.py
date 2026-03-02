"""Pipeline orchestrator — two-stage architecture with step execution."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from loguru import logger

from saegenrec.data.config import PipelineConfig

# Ensure concrete implementations are registered
import saegenrec.data.loaders.amazon2015  # noqa: F401
import saegenrec.data.loaders.amazon2023  # noqa: F401
from saegenrec.data.loaders.base import get_loader
from saegenrec.data.processors.augment import convert_eval_split, sliding_window_augment
from saegenrec.data.processors.kcore import kcore_filter
from saegenrec.data.processors.sequence import build_sequences, save_interim
from saegenrec.data.processors.split import split_data

STAGE1_STEPS = ["load", "filter", "sequence"]
STAGE2_STEPS = ["split", "augment", "negative_sampling"]
ALL_STEPS = STAGE1_STEPS + STAGE2_STEPS
LEGACY_STEPS = ["generate", "embed"]


def _build_item_titles(item_metadata: Dataset, item_id_map: Dataset) -> dict[int, str]:
    """Build mapped_id → title lookup from metadata and ID map."""
    orig_to_mapped = dict(zip(item_id_map["original_id"], item_id_map["mapped_id"]))
    item_titles: dict[int, str] = {}
    for row in item_metadata:
        mapped_id = orig_to_mapped.get(row["item_id"])
        if mapped_id is not None:
            item_titles[mapped_id] = row["title"]
    return item_titles


def _validate_prerequisites(steps: list[str], stage1_dir: Path, stage2_dir: Path) -> None:
    """Check that required upstream artifacts exist for the requested steps."""
    has_stage1 = any(s in steps for s in STAGE1_STEPS)
    has_stage2 = any(s in steps for s in STAGE2_STEPS)

    if has_stage2 and not has_stage1:
        if not (stage1_dir / "user_sequences").exists():
            raise FileNotFoundError(
                f"Stage 1 artifacts not found at {stage1_dir}. "
                f"Run Stage 1 first (steps: {STAGE1_STEPS})."
            )

    if "augment" in steps and "split" not in steps:
        if not (stage2_dir / "train_sequences").exists():
            raise FileNotFoundError(
                f"Split artifacts not found at {stage2_dir}. Run 'split' step first."
            )

    if "negative_sampling" in steps and "augment" not in steps:
        if not (stage2_dir / "train").exists():
            raise FileNotFoundError(
                f"Augment artifacts not found at {stage2_dir}. Run 'augment' step first."
            )


def run_pipeline(config: PipelineConfig, steps: list[str] | None = None) -> dict:
    """Run the data processing pipeline.

    Stage 1 (data filtering): load → filter → sequence
      Output: data/interim/{dataset}/{category}/
    Stage 2 (data splitting): split → augment
      Output: data/interim/{dataset}/{category}/{split_strategy}/

    Args:
        config: Pipeline configuration.
        steps: Specific steps to run. If None, runs ALL_STEPS.

    Returns:
        Combined statistics from all executed steps.
    """
    steps = steps or ALL_STEPS
    stage1_stats: dict = {}
    stage2_stats: dict = {}

    stage1_dir = config.output.interim_path(config.dataset.name, config.dataset.category)
    stage2_dir = stage1_dir / config.processing.split_strategy

    _validate_prerequisites(steps, stage1_dir, stage2_dir)

    # Keep legacy dir for generate/embed backward compatibility
    legacy_dir = config.output.processed_path(
        config.dataset.name,
        config.dataset.category,
        config.processing.split_strategy,
    )

    # ── Variables populated across steps ──
    interactions: Dataset | None = None
    item_metadata: Dataset | None = None
    filtered: Dataset | None = None
    user_sequences: Dataset | None = None
    item_id_map: Dataset | None = None
    train_seqs: Dataset | None = None
    valid_seqs: Dataset | None = None
    test_seqs: Dataset | None = None

    # ═══════════════════════════════════════
    # Stage 1: load → filter → sequence
    # ═══════════════════════════════════════

    if "load" in steps:
        logger.info("=== Step: Load ===")
        loader = get_loader(config.dataset.name)
        data_path = config.dataset.data_path

        interactions = loader.load_interactions(data_path)
        item_metadata = loader.load_item_metadata(data_path)

        stage1_dir.mkdir(parents=True, exist_ok=True)
        interactions.save_to_disk(str(stage1_dir / "raw_interactions"))
        item_metadata.save_to_disk(str(stage1_dir / "item_metadata"))
        stage1_stats["raw_interactions"] = len(interactions)

    if "filter" in steps:
        logger.info("=== Step: K-core Filter ===")
        if interactions is None:
            interactions = load_from_disk(str(stage1_dir / "raw_interactions"))

        filtered, filter_stats = kcore_filter(interactions, config.processing.kcore_threshold)
        stage1_stats.update(filter_stats)

        filtered.save_to_disk(str(stage1_dir / "interactions"))

        if len(filtered) == 0:
            logger.warning(
                "K-core filter removed all interactions. Pipeline will produce empty datasets."
            )

    if "sequence" in steps:
        logger.info("=== Step: Sequence Build ===")
        if filtered is None:
            filtered = load_from_disk(str(stage1_dir / "interactions"))
        if item_metadata is None:
            item_metadata = load_from_disk(str(stage1_dir / "item_metadata"))

        user_sequences, user_id_map, item_id_map_ds, seq_stats = build_sequences(filtered)
        item_id_map = item_id_map_ds
        stage1_stats.update(seq_stats)

        save_interim(
            stage1_dir,
            filtered,
            user_sequences,
            item_metadata,
            user_id_map,
            item_id_map,
            {
                **stage1_stats,
                "dataset_name": config.dataset.name,
                "category": config.dataset.category,
            },
        )

    # ═══════════════════════════════════════
    # Stage 2: split → augment → negative_sampling
    # ═══════════════════════════════════════

    if "split" in steps:
        logger.info("=== Step: Data Split ===")
        if user_sequences is None:
            user_sequences = load_from_disk(str(stage1_dir / "user_sequences"))

        train_seqs, valid_seqs, test_seqs, split_stats = split_data(
            user_sequences,
            strategy=config.processing.split_strategy,
            ratio=config.processing.split_ratio,
        )
        stage2_stats.update(split_stats)

        stage2_dir.mkdir(parents=True, exist_ok=True)
        train_seqs.save_to_disk(str(stage2_dir / "train_sequences"))
        valid_seqs.save_to_disk(str(stage2_dir / "valid_sequences"))
        test_seqs.save_to_disk(str(stage2_dir / "test_sequences"))

    if "augment" in steps:
        logger.info("=== Step: Augment ===")
        if train_seqs is None:
            train_seqs = load_from_disk(str(stage2_dir / "train_sequences"))
        if valid_seqs is None:
            valid_seqs = load_from_disk(str(stage2_dir / "valid_sequences"))
        if test_seqs is None:
            test_seqs = load_from_disk(str(stage2_dir / "test_sequences"))

        if item_metadata is None:
            item_metadata = load_from_disk(str(stage1_dir / "item_metadata"))
        if item_id_map is None:
            item_id_map = load_from_disk(str(stage1_dir / "item_id_map"))

        item_titles = _build_item_titles(item_metadata, item_id_map)

        train_ds = sliding_window_augment(train_seqs, item_titles, config.processing.max_seq_len)
        valid_ds = convert_eval_split(valid_seqs, item_titles, config.processing.max_seq_len)
        test_ds = convert_eval_split(test_seqs, item_titles, config.processing.max_seq_len)

        stage2_dir.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(stage2_dir / "train"))
        valid_ds.save_to_disk(str(stage2_dir / "valid"))
        test_ds.save_to_disk(str(stage2_dir / "test"))

        train_history_lens = [len(h) for h in train_ds["history_item_ids"]]
        avg_history = (
            sum(train_history_lens) / len(train_history_lens) if train_history_lens else 0.0
        )

        stage2_stats.update(
            {
                "split_strategy": config.processing.split_strategy,
                "split_ratio": (
                    config.processing.split_ratio
                    if config.processing.split_strategy == "to"
                    else None
                ),
                "max_seq_len": config.processing.max_seq_len,
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "test_samples": len(test_ds),
                "avg_history_length": round(avg_history, 2),
            }
        )

        stats_path = stage2_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stage2_stats, f, indent=4)

        logger.info(
            f"Augment complete: {len(train_ds)} train, "
            f"{len(valid_ds)} valid, {len(test_ds)} test → {stage2_dir}"
        )

    if "negative_sampling" in steps:
        logger.info("=== Step: Negative Sampling ===")
        from saegenrec.data.processors.negative_sampling import (
            build_user_interacted_items,
            sample_negatives,
        )

        if user_sequences is None:
            user_sequences = load_from_disk(str(stage1_dir / "user_sequences"))
        if item_id_map is None:
            item_id_map = load_from_disk(str(stage1_dir / "item_id_map"))
        if item_metadata is None:
            item_metadata = load_from_disk(str(stage1_dir / "item_metadata"))

        user_interacted = build_user_interacted_items(user_sequences)
        all_item_ids = list(item_id_map["mapped_id"])
        item_titles = _build_item_titles(item_metadata, item_id_map)

        for split_name in ("train", "valid", "test"):
            split_ds = load_from_disk(str(stage2_dir / split_name))
            neg_ds, neg_stats = sample_negatives(
                split_ds,
                user_interacted,
                all_item_ids,
                item_titles,
                num_negatives=config.processing.num_negatives,
                seed=config.processing.seed,
            )
            neg_ds.save_to_disk(str(stage2_dir / split_name))
            stage2_stats.update({f"neg_{split_name}_{k}": v for k, v in neg_stats.items()})

        stats_path = stage2_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stage2_stats, f, indent=4)

        logger.info("Negative sampling complete for all splits")

    # ═══════════════════════════════════════
    # Legacy steps (backward compatibility)
    # ═══════════════════════════════════════

    if "generate" in steps:
        logger.info("=== Step: Final Data Generation (legacy) ===")
        from saegenrec.data.processors.final import generate_final_data
        from saegenrec.data.tokenizers.base import get_tokenizer
        import saegenrec.data.tokenizers.passthrough  # noqa: F401

        if train_seqs is None:
            train_seqs = load_from_disk(str(stage2_dir / "train_sequences"))
        if valid_seqs is None:
            valid_seqs = load_from_disk(str(stage2_dir / "valid_sequences"))
        if test_seqs is None:
            test_seqs = load_from_disk(str(stage2_dir / "test_sequences"))

        num_items = stage1_stats.get("num_items", 0)
        if num_items == 0:
            _item_id_map = load_from_disk(str(stage1_dir / "item_id_map"))
            num_items = len(_item_id_map)

        tokenizer = get_tokenizer(
            config.tokenizer.name, num_items=num_items, **config.tokenizer.params
        )

        final_stats = generate_final_data(
            user_sequences_dir=stage1_dir / "user_sequences",
            item_metadata_dir=stage1_dir / "item_metadata",
            item_id_map_dir=stage1_dir / "item_id_map",
            train_sequences=train_seqs,
            valid_sequences=valid_seqs,
            test_sequences=test_seqs,
            tokenizer=tokenizer,
            max_seq_len=config.processing.max_seq_len,
            output_dir=legacy_dir,
            split_strategy=config.processing.split_strategy,
        )
        stage2_stats.update(final_stats)

    if "embed" in steps and config.embedding.enabled:
        logger.info("=== Step: Text Embedding ===")
        from saegenrec.data.embeddings.text import generate_text_embeddings

        generate_text_embeddings(
            item_metadata_dir=stage1_dir / "item_metadata",
            item_id_map_dir=stage1_dir / "item_id_map",
            output_dir=stage1_dir,
            model_name=config.embedding.model_name,
            text_fields=config.embedding.text_fields,
            batch_size=config.embedding.batch_size,
            device=config.embedding.device,
        )

    logger.success("Pipeline completed successfully")
    return {**stage1_stats, **stage2_stats}
