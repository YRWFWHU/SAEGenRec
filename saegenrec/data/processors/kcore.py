"""K-core iterative filtering."""

from __future__ import annotations

from datasets import Dataset
from loguru import logger

from saegenrec.data.schemas import INTERACTIONS_FEATURES


def kcore_filter(interactions: Dataset, threshold: int = 5) -> tuple[Dataset, dict]:
    """Apply iterative K-core filtering on interactions.

    Removes users and items with fewer than `threshold` interactions,
    iterating until convergence.

    Returns:
        (filtered_dataset, statistics_dict)
    """
    df = interactions.to_pandas()
    raw_count = len(df)
    iteration = 0

    while True:
        iteration += 1
        user_counts = df.groupby("user_id")["item_id"].transform("count")
        item_counts = df.groupby("item_id")["user_id"].transform("count")
        mask = (user_counts >= threshold) & (item_counts >= threshold)

        if mask.all():
            break

        df = df[mask].copy()

        if len(df) == 0:
            logger.warning(
                f"K-core filtering resulted in empty dataset at iteration {iteration} "
                f"(threshold={threshold})"
            )
            break

    filtered_count = len(df)
    num_users = df["user_id"].nunique() if filtered_count > 0 else 0
    num_items = df["item_id"].nunique() if filtered_count > 0 else 0

    logger.info(
        f"K-core filter (k={threshold}): {raw_count} → {filtered_count} interactions "
        f"({num_users} users, {num_items} items) in {iteration} iterations"
    )

    stats = {
        "raw_interactions": raw_count,
        "filtered_interactions": filtered_count,
        "kcore_threshold": threshold,
        "kcore_iterations": iteration,
        "num_users": num_users,
        "num_items": num_items,
    }

    if filtered_count == 0:
        filtered_ds = Dataset.from_dict(
            {k: [] for k in INTERACTIONS_FEATURES}, features=INTERACTIONS_FEATURES
        )
    else:
        filtered_ds = Dataset.from_pandas(df, features=INTERACTIONS_FEATURES, preserve_index=False)

    return filtered_ds, stats
