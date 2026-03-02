"""SFT task builder base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset
import yaml

SFT_TASK_REGISTRY: dict[str, type[SFTTaskBuilder]] = {}


def register_sft_task(name: str):
    """Decorator to register an SFT task builder class."""

    def wrapper(cls: type[SFTTaskBuilder]) -> type[SFTTaskBuilder]:
        SFT_TASK_REGISTRY[name] = cls
        return cls

    return wrapper


def get_sft_task_builder(name: str) -> SFTTaskBuilder:
    """Instantiate an SFT task builder by registered name."""
    if name not in SFT_TASK_REGISTRY:
        raise ValueError(
            f"Unknown SFT task: '{name}'. Available: {list(SFT_TASK_REGISTRY.keys())}"
        )
    return SFT_TASK_REGISTRY[name]()


class SFTTaskBuilder(ABC):
    """Abstract base for SFT task builders."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type identifier (e.g. 'seqrec')."""

    @abstractmethod
    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        config: dict[str, Any],
    ) -> Dataset:
        """Build SFT dataset for this task type."""

    def load_templates(self, template_file: str | Path) -> list[dict[str, str]]:
        """Load prompt templates from YAML for this task_type."""
        with open(template_file) as f:
            all_templates = yaml.safe_load(f)
        templates = all_templates.get(self.task_type)
        if not templates:
            raise ValueError(
                f"No templates found for task_type '{self.task_type}' in {template_file}"
            )
        return templates
