"""Pipeline orchestrator — sequences step execution."""

from __future__ import annotations

from datasets import load_from_disk
from loguru import logger

from saegenrec.data.config import PipelineConfig

# Ensure concrete implementations are registered
import saegenrec.data.loaders.amazon2015  # noqa: F401
import saegenrec.data.loaders.amazon2023  # noqa: F401
from saegenrec.data.loaders.base import get_loader
from saegenrec.data.processors.kcore import kcore_filter
from saegenrec.data.processors.sequence import build_sequences, save_interim
from saegenrec.data.processors.split import split_data
from saegenrec.data.tokenizers.base import get_tokenizer
import saegenrec.data.tokenizers.passthrough  # noqa: F401

ALL_STEPS = ["load", "filter", "sequence", "split", "augment", "generate", "embed"]


def run_pipeline(config: PipelineConfig, steps: list[str] | None = None) -> dict:
    """Run the data processing pipeline.

    Args:
        config: Pipeline configuration.
        steps: Specific steps to run. If None, runs all steps.

    Returns:
        Combined statistics from all executed steps.
    """
    steps = steps or ALL_STEPS
    all_stats: dict = {}

    interim_dir = config.output.interim_path(config.dataset.name, config.dataset.category)
    processed_dir = config.output.processed_path(
        config.dataset.name, config.dataset.category, config.processing.split_strategy
    )

    if "load" in steps:
        logger.info("=== Step: Load ===")
        loader = get_loader(config.dataset.name)
        data_path = config.dataset.data_path

        interactions = loader.load_interactions(data_path)
        item_metadata = loader.load_item_metadata(data_path)

        interim_dir.mkdir(parents=True, exist_ok=True)
        interactions.save_to_disk(str(interim_dir / "raw_interactions"))
        item_metadata.save_to_disk(str(interim_dir / "item_metadata"))
        all_stats["raw_interactions"] = len(interactions)

    if "filter" in steps:
        logger.info("=== Step: K-core Filter ===")
        if "load" not in steps:
            interactions = load_from_disk(str(interim_dir / "raw_interactions"))

        filtered, filter_stats = kcore_filter(interactions, config.processing.kcore_threshold)
        all_stats.update(filter_stats)

        filtered.save_to_disk(str(interim_dir / "interactions"))

    if "sequence" in steps:
        logger.info("=== Step: Sequence Build ===")
        if "filter" not in steps:
            filtered = load_from_disk(str(interim_dir / "interactions"))
        if "load" not in steps and "filter" not in steps:
            item_metadata = load_from_disk(str(interim_dir / "item_metadata"))

        user_sequences, user_id_map, item_id_map, seq_stats = build_sequences(filtered)
        all_stats.update(seq_stats)

        save_interim(
            interim_dir,
            filtered,
            user_sequences,
            item_metadata,
            user_id_map,
            item_id_map,
            {**all_stats, "dataset_name": config.dataset.name, "category": config.dataset.category},
        )

    if "split" in steps:
        logger.info("=== Step: Data Split ===")
        if "sequence" not in steps:
            user_sequences = load_from_disk(str(interim_dir / "user_sequences"))

        train_seqs, valid_seqs, test_seqs, split_stats = split_data(
            user_sequences,
            strategy=config.processing.split_strategy,
            ratio=config.processing.split_ratio,
        )
        all_stats.update(split_stats)

        processed_dir.mkdir(parents=True, exist_ok=True)
        train_seqs.save_to_disk(str(processed_dir / "train_sequences"))
        valid_seqs.save_to_disk(str(processed_dir / "valid_sequences"))
        test_seqs.save_to_disk(str(processed_dir / "test_sequences"))

    if "generate" in steps:
        logger.info("=== Step: Final Data Generation ===")
        if "split" not in steps:
            train_seqs = load_from_disk(str(processed_dir / "train_sequences"))
            valid_seqs = load_from_disk(str(processed_dir / "valid_sequences"))
            test_seqs = load_from_disk(str(processed_dir / "test_sequences"))

        from saegenrec.data.processors.final import generate_final_data

        num_items = all_stats.get("num_items", 0)
        if num_items == 0:
            item_id_map = load_from_disk(str(interim_dir / "item_id_map"))
            num_items = len(item_id_map)

        tokenizer = get_tokenizer(config.tokenizer.name, num_items=num_items, **config.tokenizer.params)

        final_stats = generate_final_data(
            user_sequences_dir=interim_dir / "user_sequences",
            item_metadata_dir=interim_dir / "item_metadata",
            item_id_map_dir=interim_dir / "item_id_map",
            train_sequences=train_seqs,
            valid_sequences=valid_seqs,
            test_sequences=test_seqs,
            tokenizer=tokenizer,
            max_seq_len=config.processing.max_seq_len,
            output_dir=processed_dir,
            split_strategy=config.processing.split_strategy,
        )
        all_stats.update(final_stats)

    if "embed" in steps and config.embedding.enabled:
        logger.info("=== Step: Text Embedding ===")
        from saegenrec.data.embeddings.text import generate_text_embeddings

        generate_text_embeddings(
            item_metadata_dir=interim_dir / "item_metadata",
            item_id_map_dir=interim_dir / "item_id_map",
            output_dir=interim_dir,
            model_name=config.embedding.model_name,
            text_fields=config.embedding.text_fields,
            batch_size=config.embedding.batch_size,
            device=config.embedding.device,
        )

    logger.success("Pipeline completed successfully")
    return all_stats
