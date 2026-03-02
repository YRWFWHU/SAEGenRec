"""ItemTokenizer ABC, registry, and SID map builder (T011)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import string

from datasets import Dataset, load_from_disk
from loguru import logger
import torch

from saegenrec.data.schemas import SID_MAP_FEATURES
from saegenrec.modeling.tokenizers.collision import resolve_collisions

ITEM_TOKENIZER_REGISTRY: dict[str, type[ItemTokenizer]] = {}


def register_item_tokenizer(name: str):
    """Decorator that registers an ItemTokenizer subclass under *name*."""

    def decorator(cls: type[ItemTokenizer]) -> type[ItemTokenizer]:
        ITEM_TOKENIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_item_tokenizer(name: str, **kwargs) -> ItemTokenizer:
    """Instantiate a registered ItemTokenizer by name."""
    if name not in ITEM_TOKENIZER_REGISTRY:
        available = list(ITEM_TOKENIZER_REGISTRY.keys())
        raise ValueError(f"Unknown item tokenizer: '{name}'. Available: {available}")
    return ITEM_TOKENIZER_REGISTRY[name](**kwargs)


class ItemTokenizer(ABC):
    """Abstract base class for item tokenizers that map embeddings to SIDs."""

    @abstractmethod
    def train(
        self,
        semantic_embeddings_dir: Path | str,
        collaborative_embeddings_dir: Path | str | None,
        config: dict,
    ) -> dict: ...

    @abstractmethod
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def save(self, path: Path | str) -> None: ...

    @abstractmethod
    def load(self, path: Path | str) -> None: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def codebook_size(self) -> int: ...

    def generate(
        self,
        semantic_embeddings_dir: Path | str,
        collaborative_embeddings_dir: Path | str | None,
        output_dir: Path | str,
        config: dict,
    ) -> Dataset:
        """Full pipeline: train → encode → resolve collisions → build SID map → save."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self.train(semantic_embeddings_dir, collaborative_embeddings_dir, config)
        logger.info(f"Training complete: {stats}")

        ds = load_from_disk(str(semantic_embeddings_dir))
        item_ids = ds["item_id"]
        embeddings = torch.tensor(ds["embedding"], dtype=torch.float32)

        raw_codes = self.encode(embeddings)

        strategy = config.get("collision_strategy", "append_level")
        resolved_codes = resolve_collisions(raw_codes, strategy=strategy)

        sid_map = _build_sid_map(
            item_ids,
            resolved_codes,
            token_format=config.get("sid_token_format", "<s_{level}_{code}>"),
            begin_token=config.get("sid_begin_token", ""),
            end_token=config.get("sid_end_token", ""),
        )

        sid_map.save_to_disk(str(output_dir / "item_sid_map"))
        self.save(output_dir / "tokenizer_model")
        logger.info(f"Saved SID map ({len(sid_map)} items) to {output_dir}")

        return sid_map


def _build_sid_map(
    item_ids: list[int],
    codes: list[list[int]],
    token_format: str = "<s_{level}_{code}>",
    begin_token: str = "",
    end_token: str = "",
) -> Dataset:
    """Build a HuggingFace Dataset with SID tokens for each item."""
    level_names = list(string.ascii_lowercase)
    records: dict[str, list] = {"item_id": [], "codes": [], "sid_tokens": []}

    for item_id, item_codes in zip(item_ids, codes):
        tokens = [
            token_format.format(level=level_names[i], code=c) for i, c in enumerate(item_codes)
        ]
        sid_str = begin_token + "".join(tokens) + end_token
        records["item_id"].append(item_id)
        records["codes"].append(list(item_codes))
        records["sid_tokens"].append(sid_str)

    return Dataset.from_dict(records, features=SID_MAP_FEATURES)
