"""User interaction sequence builder and interim data persistence."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from loguru import logger

from saegenrec.data.schemas import (
    ID_MAP_FEATURES,
    USER_SEQUENCES_FEATURES,
)


def build_sequences(
    interactions: Dataset,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Build time-sorted user interaction sequences from filtered interactions.

    Returns:
        (user_sequences_dataset, user_id_map_dataset, item_id_map_dataset, stats)
    """
    df = interactions.to_pandas()

    if len(df) == 0:
        empty_seqs = Dataset.from_dict(
            {k: [] for k in USER_SEQUENCES_FEATURES}, features=USER_SEQUENCES_FEATURES
        )
        empty_map = Dataset.from_dict({k: [] for k in ID_MAP_FEATURES}, features=ID_MAP_FEATURES)
        return empty_seqs, empty_map, empty_map, {"avg_seq_length": 0.0}

    # Dedup by (user_id, item_id, timestamp)
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="first")

    # Sort by user then timestamp
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Build ID mappings
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_id_map = {orig: idx for idx, orig in enumerate(unique_users)}
    item_id_map = {orig: idx for idx, orig in enumerate(unique_items)}

    user_id_map_ds = Dataset.from_dict(
        {
            "original_id": list(user_id_map.keys()),
            "mapped_id": list(user_id_map.values()),
        },
        features=ID_MAP_FEATURES,
    )

    item_id_map_ds = Dataset.from_dict(
        {
            "original_id": list(item_id_map.keys()),
            "mapped_id": list(item_id_map.values()),
        },
        features=ID_MAP_FEATURES,
    )

    # Build per-user sequences
    sequences = {
        "user_id": [],
        "item_ids": [],
        "timestamps": [],
        "ratings": [],
        "review_texts": [],
        "review_summaries": [],
    }

    for user_orig, group in df.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        mapped_user_id = user_id_map[user_orig]

        sequences["user_id"].append(mapped_user_id)
        sequences["item_ids"].append([item_id_map[iid] for iid in group["item_id"].tolist()])
        sequences["timestamps"].append(group["timestamp"].tolist())
        sequences["ratings"].append(group["rating"].tolist())
        sequences["review_texts"].append(group["review_text"].tolist())
        sequences["review_summaries"].append(group["review_summary"].tolist())

    user_sequences_ds = Dataset.from_dict(sequences, features=USER_SEQUENCES_FEATURES)

    seq_lengths = [len(ids) for ids in sequences["item_ids"]]
    stats = {
        "num_users": len(seq_lengths),
        "num_items": len(unique_items),
        "avg_seq_length": sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0.0,
        "min_seq_length": min(seq_lengths) if seq_lengths else 0,
        "max_seq_length": max(seq_lengths) if seq_lengths else 0,
    }

    logger.info(
        f"Built {stats['num_users']} user sequences, avg length {stats['avg_seq_length']:.2f}"
    )

    return user_sequences_ds, user_id_map_ds, item_id_map_ds, stats


def save_interim(
    output_dir: Path,
    interactions: Dataset,
    user_sequences: Dataset,
    item_metadata: Dataset,
    user_id_map: Dataset,
    item_id_map: Dataset,
    stats: dict,
) -> None:
    """Save all interim data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions.save_to_disk(str(output_dir / "interactions"))
    user_sequences.save_to_disk(str(output_dir / "user_sequences"))
    item_metadata.save_to_disk(str(output_dir / "item_metadata"))
    user_id_map.save_to_disk(str(output_dir / "user_id_map"))
    item_id_map.save_to_disk(str(output_dir / "item_id_map"))

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Saved interim data to {output_dir}")
