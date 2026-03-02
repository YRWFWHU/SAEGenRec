"""Pipeline configuration dataclasses with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str = "amazon2015"
    category: str = "Baby"
    raw_dir: str = "data/raw"

    @property
    def data_path(self) -> Path:
        """Raw data path: {raw_dir}/{name_mapped}/{category}/"""
        name_map = {"amazon2015": "Amazon2015", "amazon2023": "Amazon2023"}
        if self.name not in name_map:
            raise ValueError(
                f"Unknown dataset name: {self.name}. Available: {list(name_map.keys())}"
            )
        return Path(self.raw_dir) / name_map[self.name] / self.category


@dataclass
class ProcessingConfig:
    kcore_threshold: int = 5
    split_strategy: str = "loo"
    split_ratio: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    max_seq_len: int = 20
    num_negatives: int = 99
    seed: int | None = 42

    def __post_init__(self):
        if self.split_strategy not in ("loo", "to"):
            raise ValueError(f"split_strategy must be 'loo' or 'to', got '{self.split_strategy}'")
        if self.split_strategy == "to" and abs(sum(self.split_ratio) - 1.0) > 1e-6:
            raise ValueError(f"split_ratio must sum to 1.0, got {sum(self.split_ratio)}")
        if self.kcore_threshold < 1:
            raise ValueError(f"kcore_threshold must be >= 1, got {self.kcore_threshold}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.num_negatives < 1:
            raise ValueError(f"num_negatives must be >= 1, got {self.num_negatives}")


@dataclass
class TokenizerConfig:
    name: str = "passthrough"
    params: dict = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    enabled: bool = False
    model_name: str = "all-MiniLM-L6-v2"
    text_fields: list[str] = field(default_factory=lambda: ["title", "brand", "categories"])
    batch_size: int = 256
    device: str = "cpu"


@dataclass
class OutputConfig:
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"

    def interim_path(self, dataset_name: str, category: str) -> Path:
        return Path(self.interim_dir) / dataset_name / category

    def processed_path(self, dataset_name: str, category: str, split_strategy: str) -> Path:
        return Path(self.processed_dir) / dataset_name / category / split_strategy


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Path | str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        processing=ProcessingConfig(**raw.get("processing", {})),
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
