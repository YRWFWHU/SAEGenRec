"""Final training data generation — attach item tokens and text info."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from loguru import logger

from saegenrec.data.processors.augment import convert_eval_split, sliding_window_augment
from saegenrec.data.tokenizers.base import ItemTokenizer


def generate_final_data(
    user_sequences_dir: Path,
    item_metadata_dir: Path,
    item_id_map_dir: Path,
    train_sequences: Dataset,
    valid_sequences: Dataset,
    test_sequences: Dataset,
    tokenizer: ItemTokenizer,
    max_seq_len: int,
    output_dir: Path,
    split_strategy: str,
) -> dict:
    """Generate final training data with item tokens and text info.

    Returns:
        Statistics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_metadata = load_from_disk(str(item_metadata_dir))
    item_id_map = load_from_disk(str(item_id_map_dir))

    # Build original_id → mapped_id lookup
    orig_to_mapped = dict(zip(item_id_map["original_id"], item_id_map["mapped_id"]))
    # Build mapped_id → title lookup
    item_titles: dict[int, str] = {}
    for row in item_metadata:
        mapped_id = orig_to_mapped.get(row["item_id"])
        if mapped_id is not None:
            item_titles[mapped_id] = row["title"]

    # Generate training samples (sliding window augmentation)
    train_ds = sliding_window_augment(
        train_sequences, tokenizer, item_titles, max_seq_len
    )
    train_ds.save_to_disk(str(output_dir / "train"))

    # Convert validation and test sets
    valid_ds = convert_eval_split(valid_sequences, tokenizer, item_titles, max_seq_len)
    valid_ds.save_to_disk(str(output_dir / "valid"))

    test_ds = convert_eval_split(test_sequences, tokenizer, item_titles, max_seq_len)
    test_ds.save_to_disk(str(output_dir / "test"))

    # Compute stats
    train_history_lens = [len(h) for h in train_ds["history_item_ids"]]
    avg_history = (
        sum(train_history_lens) / len(train_history_lens) if train_history_lens else 0.0
    )

    stats = {
        "split_strategy": split_strategy,
        "split_ratio": None if split_strategy == "loo" else None,
        "max_seq_len": max_seq_len,
        "train_users": len(set(train_ds["user_id"])),
        "valid_users": len(valid_ds),
        "test_users": len(test_ds),
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
        "excluded_users": 0,
        "tokenizer": tokenizer.__class__.__name__.lower().replace("tokenizer", ""),
        "avg_history_length": round(avg_history, 2),
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(
        f"Final data generated: {len(train_ds)} train, "
        f"{len(valid_ds)} valid, {len(test_ds)} test samples → {output_dir}"
    )

    return stats
