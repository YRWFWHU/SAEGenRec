"""ItemTokenizer abstract base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

TOKENIZER_REGISTRY: dict[str, type[ItemTokenizer]] = {}


def register_tokenizer(name: str):
    """Decorator to register an ItemTokenizer implementation."""

    def decorator(cls: type[ItemTokenizer]):
        TOKENIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_tokenizer(name: str, **kwargs) -> ItemTokenizer:
    """Get an ItemTokenizer instance by name."""
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer: {name}. Available: {list(TOKENIZER_REGISTRY.keys())}"
        )
    return TOKENIZER_REGISTRY[name](**kwargs)


class ItemTokenizer(ABC):
    """Abstract interface for item tokenization.

    Converts item IDs (contiguous integers) to discrete token sequences.
    Does not constrain embedding source or quantization method.
    """

    @abstractmethod
    def tokenize(self, item_id: int) -> list[int]:
        """Convert an item ID to a discrete token sequence."""
        ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> int:
        """Convert a discrete token sequence back to an item ID."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Token vocabulary size."""
        ...

    @property
    @abstractmethod
    def token_length(self) -> int:
        """Token sequence length per item."""
        ...

    def tokenize_batch(self, item_ids: list[int]) -> list[list[int]]:
        """Batch tokenize (default: sequential calls; subclasses may override)."""
        return [self.tokenize(item_id) for item_id in item_ids]
