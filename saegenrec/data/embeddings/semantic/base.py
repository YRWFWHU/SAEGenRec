"""SemanticEmbedder abstract base class and registry.

Custom semantic embedders can be registered via the decorator::

    from saegenrec.data.embeddings.semantic.base import (
        SemanticEmbedder,
        register_semantic_embedder,
    )

    @register_semantic_embedder("my-custom-embedder")
    class MyCustomEmbedder(SemanticEmbedder):
        def generate(self, data_dir, output_dir, config):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

SEMANTIC_EMBEDDER_REGISTRY: dict[str, type[SemanticEmbedder]] = {}


def register_semantic_embedder(name: str):
    """Decorator to register a SemanticEmbedder implementation."""

    def decorator(cls: type[SemanticEmbedder]):
        SEMANTIC_EMBEDDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_semantic_embedder(name: str, **kwargs) -> SemanticEmbedder:
    """Get a SemanticEmbedder instance by registered name."""
    if name not in SEMANTIC_EMBEDDER_REGISTRY:
        available = list(SEMANTIC_EMBEDDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown semantic embedder: '{name}'. Available: {available}"
        )
    return SEMANTIC_EMBEDDER_REGISTRY[name](**kwargs)


class SemanticEmbedder(ABC):
    """Abstract interface for semantic embedding generation.

    Encodes item metadata text fields into dense vector representations
    using pre-trained language models.
    """

    @abstractmethod
    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        """Generate semantic embeddings from Stage 1 item metadata.

        Args:
            data_dir: Directory containing item_metadata/ and item_id_map/.
            output_dir: Directory to save item_semantic_embeddings/.
            config: Embedding configuration dictionary.

        Returns:
            HuggingFace Dataset with item_id and embedding columns.
        """
        ...
