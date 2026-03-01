"""DatasetLoader abstract base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {}


def register_loader(name: str):
    """Decorator to register a DatasetLoader implementation."""

    def decorator(cls: type[DatasetLoader]):
        LOADER_REGISTRY[name] = cls
        return cls

    return decorator


def get_loader(name: str) -> DatasetLoader:
    """Get a DatasetLoader instance by name."""
    if name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset loader: {name}. Available: {list(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[name]()


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Each raw data format (e.g., Amazon2015, Amazon2023) MUST implement this interface.
    All implementations MUST map raw fields to the unified output schema.
    """

    @abstractmethod
    def load_interactions(self, data_dir: Path) -> Dataset:
        """Load raw interactions and convert to unified format.

        Returns:
            HuggingFace Dataset with schema:
                user_id (string), item_id (string), timestamp (int64),
                rating (float32), review_text (string), review_summary (string)
        """
        ...

    @abstractmethod
    def load_item_metadata(self, data_dir: Path) -> Dataset:
        """Load item metadata and convert to unified format.

        Returns:
            HuggingFace Dataset with schema:
                item_id (string), title (string), brand (string),
                categories (Sequence(string)), description (string),
                price (float32), image_url (string)
        """
        ...
