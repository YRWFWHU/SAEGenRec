"""GenRecModel abstract base class and model registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset

GENREC_MODEL_REGISTRY: dict[str, type[GenRecModel]] = {}


def register_genrec_model(name: str):
    """Decorator to register a GenRecModel implementation by name."""

    def decorator(cls: type[GenRecModel]) -> type[GenRecModel]:
        GENREC_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_genrec_model(name: str, **kwargs) -> GenRecModel:
    """Instantiate a registered GenRecModel by name."""
    if name not in GENREC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown GenRec model: '{name}'. Available: {list(GENREC_MODEL_REGISTRY.keys())}"
        )
    return GENREC_MODEL_REGISTRY[name](**kwargs)


class GenRecModel(ABC):
    """Abstract base class for generative recommendation models."""

    @abstractmethod
    def train(self, dataset: Dataset, training_args: dict) -> dict: ...

    @abstractmethod
    def generate(self, input_text: str | list[str], **kwargs) -> list[str]: ...

    @abstractmethod
    def evaluate(self, dataset: Dataset, metrics: list[str] | None = None) -> dict[str, float]: ...

    @abstractmethod
    def save_pretrained(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Path, **kwargs) -> GenRecModel: ...
