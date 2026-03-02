"""CollaborativeEmbedder abstract base class and registry.

Custom collaborative embedders can be registered via the decorator::

    from saegenrec.data.embeddings.collaborative.base import (
        CollaborativeEmbedder,
        register_collaborative_embedder,
    )

    @register_collaborative_embedder("my-model")
    class MyModelEmbedder(CollaborativeEmbedder):
        def generate(self, data_dir, output_dir, config):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

COLLABORATIVE_EMBEDDER_REGISTRY: dict[str, type[CollaborativeEmbedder]] = {}


def register_collaborative_embedder(name: str):
    """Decorator to register a CollaborativeEmbedder implementation."""

    def decorator(cls: type[CollaborativeEmbedder]):
        COLLABORATIVE_EMBEDDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_collaborative_embedder(name: str, **kwargs) -> CollaborativeEmbedder:
    """Get a CollaborativeEmbedder instance by registered name."""
    if name not in COLLABORATIVE_EMBEDDER_REGISTRY:
        available = list(COLLABORATIVE_EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown collaborative embedder: '{name}'. Available: {available}"
        )
    return COLLABORATIVE_EMBEDDER_REGISTRY[name](**kwargs)


class CollaborativeEmbedder(ABC):
    """Abstract interface for collaborative embedding generation.

    Trains sequential recommendation models on user interaction sequences
    and extracts item embeddings from learned nn.Embedding weights.
    """

    @abstractmethod
    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        """Train model and extract collaborative embeddings from Stage 2 split data.

        Args:
            data_dir: Directory containing train/valid/test_sequences/ and item_id_map/.
            output_dir: Directory to save item_collaborative_embeddings/.
            config: Embedding configuration dictionary.

        Returns:
            HuggingFace Dataset with item_id and embedding columns.
        """
        ...
