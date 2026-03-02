"""Data splitting strategies: Leave-One-Out (LOO) and Temporal Order (TO)."""

from __future__ import annotations

from datasets import Dataset
from loguru import logger

from saegenrec.data.schemas import USER_SEQUENCES_FEATURES


def split_data(
    user_sequences: Dataset,
    strategy: str = "loo",
    ratio: list[float] | None = None,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Split user sequences into train/valid/test sets.

    Args:
        user_sequences: UserSequence dataset.
        strategy: "loo" or "to".
        ratio: Split ratio for TO strategy (default: [0.8, 0.1, 0.1]).

    Returns:
        (train_dataset, valid_dataset, test_dataset, stats)
    """
    if strategy == "loo":
        return _split_loo(user_sequences)
    elif strategy == "to":
        ratio = ratio or [0.8, 0.1, 0.1]
        return _split_to(user_sequences, ratio)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def _split_loo(user_sequences: Dataset) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Leave-One-Out split: last→test, second-last→valid, rest→train."""
    train_data = {k: [] for k in USER_SEQUENCES_FEATURES}
    valid_data = {k: [] for k in USER_SEQUENCES_FEATURES}
    test_data = {k: [] for k in USER_SEQUENCES_FEATURES}
    excluded_users = 0

    for row in user_sequences:
        seq_len = len(row["item_ids"])

        if seq_len < 3:
            excluded_users += 1
            continue

        uid = row["user_id"]

        # Test: last item
        test_data["user_id"].append(uid)
        test_data["item_ids"].append(row["item_ids"][-1:])
        test_data["timestamps"].append(row["timestamps"][-1:])
        test_data["ratings"].append(row["ratings"][-1:])
        test_data["review_texts"].append(row["review_texts"][-1:])
        test_data["review_summaries"].append(row["review_summaries"][-1:])

        # Valid: second-last item
        valid_data["user_id"].append(uid)
        valid_data["item_ids"].append(row["item_ids"][-2:-1])
        valid_data["timestamps"].append(row["timestamps"][-2:-1])
        valid_data["ratings"].append(row["ratings"][-2:-1])
        valid_data["review_texts"].append(row["review_texts"][-2:-1])
        valid_data["review_summaries"].append(row["review_summaries"][-2:-1])

        # Train: rest
        train_data["user_id"].append(uid)
        train_data["item_ids"].append(row["item_ids"][:-2])
        train_data["timestamps"].append(row["timestamps"][:-2])
        train_data["ratings"].append(row["ratings"][:-2])
        train_data["review_texts"].append(row["review_texts"][:-2])
        train_data["review_summaries"].append(row["review_summaries"][:-2])

    stats = {
        "split_strategy": "loo",
        "split_ratio": None,
        "train_users": len(train_data["user_id"]),
        "valid_users": len(valid_data["user_id"]),
        "test_users": len(test_data["user_id"]),
        "excluded_users": excluded_users,
    }

    logger.info(
        f"LOO split: {stats['train_users']} users (excluded {excluded_users} with <3 interactions)"
    )

    return (
        Dataset.from_dict(train_data, features=USER_SEQUENCES_FEATURES),
        Dataset.from_dict(valid_data, features=USER_SEQUENCES_FEATURES),
        Dataset.from_dict(test_data, features=USER_SEQUENCES_FEATURES),
        stats,
    )


def _split_to(
    user_sequences: Dataset,
    ratio: list[float],
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Temporal Order split: global timestamp-based partition."""
    # Collect all (user_id, idx_in_seq, timestamp) triples
    all_entries = []
    for row in user_sequences:
        uid = row["user_id"]
        for idx, ts in enumerate(row["timestamps"]):
            all_entries.append((uid, idx, ts))

    all_entries.sort(key=lambda x: x[2])

    total = len(all_entries)
    train_end = int(total * ratio[0])
    valid_end = int(total * (ratio[0] + ratio[1]))

    train_set = set()
    valid_set = set()
    test_set = set()

    for i, (uid, idx, ts) in enumerate(all_entries):
        if i < train_end:
            train_set.add((uid, idx))
        elif i < valid_end:
            valid_set.add((uid, idx))
        else:
            test_set.add((uid, idx))

    def _build_split(user_sequences: Dataset, index_set: set) -> Dataset:
        data = {k: [] for k in USER_SEQUENCES_FEATURES}
        for row in user_sequences:
            uid = row["user_id"]
            indices = [idx for idx in range(len(row["item_ids"])) if (uid, idx) in index_set]
            if not indices:
                continue
            data["user_id"].append(uid)
            data["item_ids"].append([row["item_ids"][i] for i in indices])
            data["timestamps"].append([row["timestamps"][i] for i in indices])
            data["ratings"].append([row["ratings"][i] for i in indices])
            data["review_texts"].append([row["review_texts"][i] for i in indices])
            data["review_summaries"].append([row["review_summaries"][i] for i in indices])
        return Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)

    train_ds = _build_split(user_sequences, train_set)
    valid_ds = _build_split(user_sequences, valid_set)
    test_ds = _build_split(user_sequences, test_set)

    stats = {
        "split_strategy": "to",
        "split_ratio": ratio,
        "train_users": len(train_ds),
        "valid_users": len(valid_ds),
        "test_users": len(test_ds),
        "excluded_users": 0,
    }

    logger.info(
        f"TO split ({ratio}): train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}"
    )

    return train_ds, valid_ds, test_ds, stats
