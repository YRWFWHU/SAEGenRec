"""SFT dataset — loads HF Dataset, adds SID tokens, prepares for SFTTrainer."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_from_disk
from loguru import logger
from transformers import PreTrainedTokenizer

from .collator import convert_to_conversational


def add_sid_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizer,
    sid_map: Dataset,
) -> list[str]:
    """Extract unique SID tokens from the sid_map and add them to the tokenizer.

    Returns the list of newly added tokens.
    """
    all_sid_tokens: set[str] = set()
    for row in sid_map:
        tokens = row["sid_tokens"].split("><")
        for t in tokens:
            t = t.strip()
            if not t.startswith("<"):
                t = "<" + t
            if not t.endswith(">"):
                t = t + ">"
            all_sid_tokens.add(t)

    new_tokens = sorted(all_sid_tokens)
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    logger.info("Added {} SID tokens to tokenizer (vocab size: {})", num_added, len(tokenizer))
    return new_tokens


def load_sft_dataset(
    data_dir: str | Path,
    tasks: list[str] | None = None,
    split: str | None = None,
) -> Dataset:
    """Load SFT dataset from disk, optionally filtering by tasks.

    Args:
        data_dir: Root SFT data directory.  If *split* is given, loads
            ``data_dir/{split}``; otherwise loads ``data_dir`` directly.
        tasks: If provided, filter to only these task types.
        split: Optional split name (``train``, ``valid``, ``test``).

    Returns:
        HF Dataset in conversational format (``messages`` + ``task_type``).
    """
    load_path = Path(data_dir) / split if split else Path(data_dir)
    ds = load_from_disk(str(load_path))
    logger.info("Loaded {} samples from {}", len(ds), load_path)

    if tasks is not None:
        ds = ds.filter(lambda row: row["task_type"] in tasks)
        logger.info("Filtered to tasks {}: {} samples", tasks, len(ds))

    ds = convert_to_conversational(ds)
    return ds
