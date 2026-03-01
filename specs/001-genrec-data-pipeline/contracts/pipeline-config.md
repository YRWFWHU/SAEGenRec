# Contract: Pipeline Configuration

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## YAML Configuration Schema

```yaml
# configs/default.yaml — 流水线默认配置
dataset:
  name: "amazon2015"              # 数据集版本: "amazon2015" | "amazon2023"
  category: "Baby"                # 数据集类目
  raw_dir: "data/raw"             # 原始数据根目录（相对于项目根）

processing:
  kcore_threshold: 5              # K-core 过滤最小交互次数阈值
  split_strategy: "loo"           # 划分策略: "loo" | "to"
  split_ratio: [0.8, 0.1, 0.1]   # TO 划分比例 (仅 split_strategy="to" 时生效)
  max_seq_len: 20                 # 滑动窗口最长序列长度

tokenizer:
  name: "passthrough"             # Tokenizer 名称（对应 registry key）
  params: {}                      # Tokenizer 构造参数

embedding:
  enabled: false                  # 是否生成文本 embedding
  model_name: "all-MiniLM-L6-v2" # 预训练模型名称
  text_fields:                    # 拼接的元数据字段
    - "title"
    - "brand"
    - "categories"
  batch_size: 256                 # 推理 batch size
  device: "cpu"                   # 推理设备: "cpu" | "cuda"

output:
  interim_dir: "data/interim"     # 中间数据输出目录（相对于项目根）
  processed_dir: "data/processed" # 最终数据输出目录（相对于项目根）
```

## Dataclass Definition

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    name: str = "amazon2015"
    category: str = "Baby"
    raw_dir: str = "data/raw"

    @property
    def data_path(self) -> Path:
        """原始数据完整路径: {raw_dir}/{name_mapped}/{category}/"""
        name_map = {"amazon2015": "Amazon2015", "amazon2023": "Amazon2023"}
        return Path(self.raw_dir) / name_map[self.name] / self.category


@dataclass
class ProcessingConfig:
    kcore_threshold: int = 5
    split_strategy: str = "loo"
    split_ratio: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    max_seq_len: int = 20

    def __post_init__(self):
        if self.split_strategy not in ("loo", "to"):
            raise ValueError(f"split_strategy must be 'loo' or 'to', got '{self.split_strategy}'")
        if self.split_strategy == "to" and abs(sum(self.split_ratio) - 1.0) > 1e-6:
            raise ValueError(f"split_ratio must sum to 1.0, got {sum(self.split_ratio)}")
        if self.kcore_threshold < 1:
            raise ValueError(f"kcore_threshold must be >= 1, got {self.kcore_threshold}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")


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
        """中间数据完整路径。"""
        return Path(self.interim_dir) / dataset_name / category

    def processed_path(self, dataset_name: str, category: str, split_strategy: str) -> Path:
        """最终数据完整路径。"""
        return Path(self.processed_dir) / dataset_name / category / split_strategy


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
```

## Config Loading

```python
import yaml
from pathlib import Path


def load_config(config_path: Path) -> PipelineConfig:
    """从 YAML 文件加载流水线配置。"""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        processing=ProcessingConfig(**raw.get("processing", {})),
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
```

## CLI Entry Point

```python
@app.command()
def process(
    config: Path = typer.Argument(..., help="YAML 配置文件路径"),
    steps: list[str] = typer.Option(
        None, "--step", "-s",
        help="指定运行的步骤 (load, filter, sequence, split, augment, generate, embed)",
    ),
):
    """运行数据处理流水线。"""
    ...
```

## Validation Rules

1. 所有路径参数 MUST 支持相对路径（相对于项目根）和绝对路径
2. `split_ratio` 仅在 `split_strategy="to"` 时被使用
3. `tokenizer.params` 中的参数 MUST 与对应 Tokenizer 类的 `__init__` 参数匹配
4. `embedding.text_fields` 中的字段名 MUST 存在于 ItemMetadata schema 中
5. 所有 dataclass 字段 MUST 有合理的默认值，使得 `PipelineConfig()` 可直接使用
