# Contract: GenRecModel 抽象接口

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02

## Interface Definition

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


@dataclass
class GenRecConfig:
    """生成式推荐模型配置。"""
    base_model_name: str = "Qwen/Qwen2.5-0.5B"
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] | None = None
    training_strategy: str = "sft"  # "sft" | "rl" (预留)
    sid_tokens_path: str | None = None


class GenRecModel(ABC):
    """生成式推荐模型抽象接口。

    遵循 HuggingFace 设计哲学：
    - 通过配置指定 base model、LoRA 参数、训练策略
    - 方法签名与 HuggingFace Trainer 风格一致
    - 支持通过注册表机制选择不同实现

    本期仅定义接口，不实现训练逻辑。
    """

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        training_args: dict,
    ) -> dict:
        """训练模型。

        Args:
            dataset: SFT 数据集（Alpaca 格式）。
            training_args: 训练参数（兼容 HuggingFace TrainingArguments）。

        Returns:
            训练统计信息。
        """
        ...

    @abstractmethod
    def generate(
        self,
        input_text: str | list[str],
        **kwargs,
    ) -> list[str]:
        """生成推荐结果。

        Args:
            input_text: 输入文本（prompt）。
            **kwargs: 生成参数（max_new_tokens, temperature 等）。

        Returns:
            生成的文本列表。
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """评估模型。

        Args:
            dataset: 评估数据集。
            metrics: 评估指标列表。

        Returns:
            指标名 → 指标值映射。
        """
        ...

    @abstractmethod
    def save_pretrained(self, path: Path) -> None:
        """保存模型（HuggingFace 风格）。"""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Path, **kwargs) -> "GenRecModel":
        """加载模型（HuggingFace 风格）。"""
        ...
```

## Registry

```python
GENREC_MODEL_REGISTRY: dict[str, type[GenRecModel]] = {}


def register_genrec_model(name: str):
    """装饰器：注册 GenRecModel 实现。"""
    def decorator(cls: type[GenRecModel]):
        GENREC_MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_genrec_model(name: str, **kwargs) -> GenRecModel:
    """根据名称获取 GenRecModel 实例。"""
    if name not in GENREC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown GenRec model: '{name}'. "
            f"Available: {list(GENREC_MODEL_REGISTRY.keys())}"
        )
    return GENREC_MODEL_REGISTRY[name](**kwargs)
```

## Constrained Decoding (同包)

约束解码模块位于 `saegenrec/modeling/decoding/`，与 GenRecModel 配合使用：

```python
from transformers import LogitsProcessor


class SIDConstrainedLogitsProcessor(LogitsProcessor):
    """基于 SID Prefix Trie 的约束解码。

    在每步解码时，将不在 Trie 有效前缀中的 token logits 设为 -inf，
    确保生成的 SID token 序列对应真实物品。

    兼容 HuggingFace transformers.generate() 的 logits_processor 参数。
    """

    def __init__(self, trie: "SIDTrie", sid_begin_token_id: int, sid_end_token_id: int):
        ...

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ...
```

```python
class SIDTrie:
    """SID 前缀树。从 item_sid_map 构建，用于约束解码。"""

    def __init__(self):
        self.root: dict = {}

    def insert(self, token_ids: list[int]) -> None: ...
    def search_prefix(self, prefix: list[int]) -> list[int]: ...

    @classmethod
    def from_sid_map(cls, sid_map: Dataset, tokenizer) -> "SIDTrie": ...
```

## Scope

本期 (004) 实现：
- GenRecModel ABC + 注册表
- SIDTrie 前缀树数据结构
- SIDConstrainedLogitsProcessor（兼容 HuggingFace LogitsProcessor）

以下留作后续迭代：
- 具体模型实现（SFT Trainer、RL Trainer）
- 词表扩展（SID special tokens 注入 LLM tokenizer）
- 完整推理管道
- 评估管道（HR@K, NDCG@K）

## Invariants

1. `GenRecModel` 子类 MUST 可通过 `register_genrec_model` 装饰器注册
2. `from_pretrained` 加载的模型 MUST 与 `save_pretrained` 保存的模型行为一致
3. 接口不依赖任何具体训练框架（PyTorch、DeepSpeed 等为实现细节）
