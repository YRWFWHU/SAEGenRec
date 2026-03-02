"""Sliding window data augmentation and validation/test set conversion."""

from __future__ import annotations

from datasets import Dataset
from loguru import logger

from saegenrec.data.schemas import INTERIM_SAMPLE_FEATURES


def sliding_window_augment(
    train_sequences: Dataset,
    item_titles: dict[int, str],
    max_seq_len: int = 20,
) -> Dataset:
    """Generate (history, target) training samples via sliding window.

    For each user sequence [i1, i2, ..., iN], generates N-1 samples:
      sample k: history=[i1,...,ik][-max_seq_len:], target=i_{k+1}
    """
    samples: dict[str, list] = {k: [] for k in INTERIM_SAMPLE_FEATURES}

    for row in train_sequences:
        uid = row["user_id"]
        item_ids = row["item_ids"]
        seq_len = len(item_ids)

        if seq_len < 2:
            continue

        for target_pos in range(1, seq_len):
            history = item_ids[:target_pos]
            if len(history) > max_seq_len:
                history = history[-max_seq_len:]
            target_id = item_ids[target_pos]

            samples["user_id"].append(uid)
            samples["history_item_ids"].append(history)
            samples["history_item_titles"].append([item_titles.get(iid, "") for iid in history])
            samples["target_item_id"].append(target_id)
            samples["target_item_title"].append(item_titles.get(target_id, ""))

    logger.info(f"Sliding window: generated {len(samples['user_id'])} training samples")
    return Dataset.from_dict(samples, features=INTERIM_SAMPLE_FEATURES)


def convert_eval_split(
    eval_sequences: Dataset,
    item_titles: dict[int, str],
    max_seq_len: int = 20,
) -> Dataset:
    """Convert validation/test sequences to InterimSample format.

    No augmentation — each user produces exactly one sample:
    history = all items except last (truncated to max_seq_len), target = last item.
    """
    samples: dict[str, list] = {k: [] for k in INTERIM_SAMPLE_FEATURES}

    for row in eval_sequences:
        uid = row["user_id"]
        item_ids = row["item_ids"]

        if len(item_ids) < 1:
            continue

        if len(item_ids) == 1:
            samples["user_id"].append(uid)
            samples["history_item_ids"].append([])
            samples["history_item_titles"].append([])
            samples["target_item_id"].append(item_ids[0])
            samples["target_item_title"].append(item_titles.get(item_ids[0], ""))
        else:
            history = item_ids[:-1]
            if len(history) > max_seq_len:
                history = history[-max_seq_len:]
            target_id = item_ids[-1]

            samples["user_id"].append(uid)
            samples["history_item_ids"].append(history)
            samples["history_item_titles"].append([item_titles.get(iid, "") for iid in history])
            samples["target_item_id"].append(target_id)
            samples["target_item_title"].append(item_titles.get(target_id, ""))

    logger.info(f"Converted {len(samples['user_id'])} eval samples")
    return Dataset.from_dict(samples, features=INTERIM_SAMPLE_FEATURES)
