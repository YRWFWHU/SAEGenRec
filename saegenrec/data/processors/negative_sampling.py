"""Negative sampling for re-ranking tasks."""

from __future__ import annotations

from datasets import Dataset
from loguru import logger
import numpy as np

from saegenrec.data.schemas import NEGATIVE_SAMPLE_FEATURES


def build_user_interacted_items(
    user_sequences: Dataset,
) -> dict[int, set[int]]:
    """Build {user_id: {item_id, ...}} mapping from full user sequences."""
    result: dict[int, set[int]] = {}
    for row in user_sequences:
        result[row["user_id"]] = set(row["item_ids"])
    return result


def sample_negatives(
    samples: Dataset,
    user_interacted_items: dict[int, set[int]],
    all_item_ids: list[int],
    item_titles: dict[int, str],
    num_negatives: int = 99,
    seed: int | None = 42,
) -> tuple[Dataset, dict]:
    """Sample negative items for each sample, excluding user's interaction history.

    Returns:
        (Dataset with NEGATIVE_SAMPLE_FEATURES, stats dict)
    """
    rng = np.random.default_rng(seed)
    all_item_set = set(all_item_ids)

    output: dict[str, list] = {k: [] for k in NEGATIVE_SAMPLE_FEATURES}
    num_warnings = 0

    for row in samples:
        uid = row["user_id"]
        interacted = user_interacted_items.get(uid, set())
        candidates = list(all_item_set - interacted)

        if len(candidates) < num_negatives:
            num_warnings += 1
            if len(candidates) == 0:
                logger.warning(f"User {uid}: no available negative items, returning empty list")
                neg_ids = []
            else:
                logger.warning(
                    f"User {uid}: only {len(candidates)} negatives available "
                    f"(requested {num_negatives})"
                )
                neg_ids = candidates
        else:
            chosen_indices = rng.choice(len(candidates), size=num_negatives, replace=False)
            neg_ids = [candidates[i] for i in chosen_indices]

        output["user_id"].append(uid)
        output["history_item_ids"].append(row["history_item_ids"])
        output["history_item_titles"].append(row["history_item_titles"])
        output["target_item_id"].append(row["target_item_id"])
        output["target_item_title"].append(row["target_item_title"])
        output["negative_item_ids"].append(neg_ids)
        output["negative_item_titles"].append([item_titles.get(nid, "") for nid in neg_ids])

    stats = {
        "num_negatives_requested": num_negatives,
        "num_negatives_warnings": num_warnings,
        "total_samples": len(samples),
    }

    logger.info(
        f"Negative sampling: {len(samples)} samples, "
        f"{num_negatives} negatives/sample, {num_warnings} warnings"
    )

    return Dataset.from_dict(output, features=NEGATIVE_SAMPLE_FEATURES), stats
