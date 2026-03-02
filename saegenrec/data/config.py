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
    """Legacy embedding config (deprecated — use SemanticEmbeddingConfig)."""

    enabled: bool = False
    model_name: str = "all-MiniLM-L6-v2"
    text_fields: list[str] = field(default_factory=lambda: ["title", "brand", "categories"])
    batch_size: int = 256
    device: str = "cpu"


@dataclass
class SemanticEmbeddingConfig:
    enabled: bool = False
    name: str = "sentence-transformer"
    model_name: str = "all-MiniLM-L6-v2"
    text_fields: list[str] = field(
        default_factory=lambda: ["title", "brand", "description", "price"]
    )
    normalize: bool = False
    batch_size: int = 256
    device: str = "cpu"


@dataclass
class CollaborativeEmbeddingConfig:
    enabled: bool = False
    name: str = "sasrec"
    loss_type: str = "CE"
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    max_seq_len: int = 50
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 200
    eval_top_k: list[int] = field(default_factory=lambda: [10, 20])
    device: str = "auto"
    seed: int = 42


@dataclass
class ItemTokenizerConfig:
    enabled: bool = False
    name: str = "rqvae"
    num_codebooks: int = 4
    codebook_size: int = 256
    collision_strategy: str = "append_level"
    sid_token_format: str = "<s_{level}_{code}>"
    sid_begin_token: str = "<|sid_begin|>"
    sid_end_token: str = "<|sid_end|>"
    params: dict = field(default_factory=dict)


@dataclass
class SFTBuilderConfig:
    enabled: bool = False
    tasks: list[str] = field(default_factory=lambda: ["seqrec", "item2index", "index2item"])
    task_weights: dict[str, float] = field(default_factory=dict)
    template_file: str = "configs/templates/sft_prompts.yaml"
    max_history_len: int = 20
    seed: int = 42


@dataclass
class OutputConfig:
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"

    def interim_path(self, dataset_name: str, category: str) -> Path:
        return Path(self.interim_dir) / dataset_name / category

    def processed_path(self, dataset_name: str, category: str, split_strategy: str) -> Path:
        return Path(self.processed_dir) / dataset_name / category / split_strategy

    def modeling_path(self, dataset_name: str, category: str) -> Path:
        """Output path for modeling artifacts (tokenize/build-sft), without split_strategy."""
        return Path(self.processed_dir) / dataset_name / category


@dataclass
class SFTTrainingEnabled:
    """Lightweight wrapper indicating whether SFT training is enabled in the YAML."""

    enabled: bool = False


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    semantic_embedding: SemanticEmbeddingConfig = field(default_factory=SemanticEmbeddingConfig)
    collaborative_embedding: CollaborativeEmbeddingConfig = field(
        default_factory=CollaborativeEmbeddingConfig
    )
    item_tokenizer: ItemTokenizerConfig = field(default_factory=ItemTokenizerConfig)
    sft_builder: SFTBuilderConfig = field(default_factory=SFTBuilderConfig)
    sft_training: SFTTrainingEnabled = field(default_factory=SFTTrainingEnabled)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Path | str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    import warnings

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    sft_training_raw = raw.get("sft_training", {})
    sft_training_enabled = SFTTrainingEnabled(enabled=sft_training_raw.get("enabled", False))

    cfg = PipelineConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        processing=ProcessingConfig(**raw.get("processing", {})),
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        semantic_embedding=SemanticEmbeddingConfig(**raw.get("semantic_embedding", {})),
        collaborative_embedding=CollaborativeEmbeddingConfig(
            **raw.get("collaborative_embedding", {})
        ),
        item_tokenizer=ItemTokenizerConfig(**raw.get("item_tokenizer", {})),
        sft_builder=SFTBuilderConfig(**raw.get("sft_builder", {})),
        sft_training=sft_training_enabled,
        output=OutputConfig(**raw.get("output", {})),
    )

    if cfg.embedding.enabled and "semantic_embedding" not in raw:
        warnings.warn(
            "The 'embedding' config section is deprecated. "
            "Use 'semantic_embedding' instead. "
            "Auto-migrating: semantic_embedding.enabled=True with legacy settings.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg.semantic_embedding.enabled = True
        cfg.semantic_embedding.model_name = cfg.embedding.model_name
        cfg.semantic_embedding.text_fields = cfg.embedding.text_fields
        cfg.semantic_embedding.batch_size = cfg.embedding.batch_size
        cfg.semantic_embedding.device = cfg.embedding.device

    return cfg
