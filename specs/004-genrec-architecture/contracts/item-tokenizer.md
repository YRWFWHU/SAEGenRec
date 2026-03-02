# Contract: ItemTokenizer 抽象接口 (v2)

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02
**Supersedes**: `specs/001-genrec-data-pipeline/contracts/item-tokenizer.md`

## Interface Definition

```python
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from datasets import Dataset


class ItemTokenizer(ABC):
    """物品 Tokenizer 抽象接口。

    将物品 embedding 映射为层次化离散码（SID）。
    支持训练、编码、碰撞消解的完整流程。

    ABC 的 generate 方法同时接收语义和协同 embedding 路径，
    具体如何使用由各实现自行决定（FR-007）。
    """

    @abstractmethod
    def train(
        self,
        semantic_embeddings_dir: Path,
        collaborative_embeddings_dir: Path | None,
        config: dict,
    ) -> dict:
        """训练 tokenizer。

        Args:
            semantic_embeddings_dir: 语义 embedding HF Dataset 目录。
            collaborative_embeddings_dir: 协同 embedding HF Dataset 目录（可选）。
            config: 训练配置字典。

        Returns:
            训练统计信息（损失、码本利用率等）。
        """
        ...

    @abstractmethod
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """将 embedding 批量编码为离散码。

        Args:
            embeddings: (N, D) float tensor。

        Returns:
            (N, num_codebooks) int tensor，每个值 ∈ [0, codebook_size)。
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """保存训练好的 tokenizer 到磁盘。"""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """从磁盘加载训练好的 tokenizer。"""
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        """码本层数。"""
        ...

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """每层码本大小。"""
        ...

    def generate(
        self,
        semantic_embeddings_dir: Path,
        collaborative_embeddings_dir: Path | None,
        output_dir: Path,
        config: dict,
    ) -> Dataset:
        """完整流程：训练 → 编码 → 碰撞消解 → 保存 SID map。

        提供默认实现，子类可覆写。

        Args:
            semantic_embeddings_dir: 语义 embedding 目录。
            collaborative_embeddings_dir: 协同 embedding 目录（可选）。
            output_dir: 输出目录（保存 item_sid_map/）。
            config: 完整配置字典。

        Returns:
            item_sid_map HuggingFace Dataset。
        """
        from saegenrec.modeling.tokenizers.collision import resolve_collisions

        train_stats = self.train(
            semantic_embeddings_dir, collaborative_embeddings_dir, config
        )
        logger.info(f"Training complete: {train_stats}")

        embeddings_ds = load_from_disk(str(semantic_embeddings_dir))
        embeddings_tensor = torch.tensor(embeddings_ds["embedding"])
        item_ids = embeddings_ds["item_id"]

        raw_codes = self.encode(embeddings_tensor)

        resolved_codes = resolve_collisions(
            raw_codes,
            strategy=config.get("collision_strategy", "append_level"),
            **config.get("collision_params", {}),
        )

        sid_map = _build_sid_map(
            item_ids, resolved_codes,
            token_format=config.get("sid_token_format", "<s_{level}_{code}>"),
            begin_token=config.get("sid_begin_token", "<|sid_begin|>"),
            end_token=config.get("sid_end_token", "<|sid_end|>"),
        )

        sid_map.save_to_disk(str(output_dir / "item_sid_map"))
        self.save(output_dir / "tokenizer_model")

        return sid_map
```

## Registry

```python
ITEM_TOKENIZER_REGISTRY: dict[str, type[ItemTokenizer]] = {}


def register_item_tokenizer(name: str):
    """装饰器：注册 ItemTokenizer 实现。"""
    def decorator(cls: type[ItemTokenizer]):
        ITEM_TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator


def get_item_tokenizer(name: str, **kwargs) -> ItemTokenizer:
    """根据名称获取 ItemTokenizer 实例。"""
    if name not in ITEM_TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown item tokenizer: '{name}'. "
            f"Available: {list(ITEM_TOKENIZER_REGISTRY.keys())}"
        )
    return ITEM_TOKENIZER_REGISTRY[name](**kwargs)
```

## Implementations

### RQVAETokenizer

```python
@register_item_tokenizer("rqvae")
class RQVAETokenizer(ItemTokenizer):
    """RQ-VAE Tokenizer：MLP 编码器 + 残差向量量化 + MLP 解码器。

    使用 PyTorch Lightning 训练。仅消费语义 embedding。
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        **kwargs,
    ): ...

    def train(self, semantic_embeddings_dir, collaborative_embeddings_dir, config) -> dict: ...
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### RQKMeansTokenizer

```python
@register_item_tokenizer("rqkmeans")
class RQKMeansTokenizer(ItemTokenizer):
    """RQ-KMeans Tokenizer：逐层残差 KMeans 聚类。

    无神经网络训练，CPU 可运行。仅消费语义 embedding。
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        use_constrained: bool = False,
        kmeans_niter: int = 20,
        use_gpu: bool = False,
        **kwargs,
    ): ...

    def train(self, semantic_embeddings_dir, collaborative_embeddings_dir, config) -> dict: ...
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

## Collision Resolution

```python
def resolve_collisions(
    codes: torch.Tensor,
    strategy: str = "append_level",
    **kwargs,
) -> list[list[int]]:
    """碰撞消解：确保每个物品获得唯一 SID。

    Args:
        codes: (N, num_codebooks) int tensor。
        strategy: "append_level" | "sinkhorn"。
        **kwargs: 策略特定参数。

    Returns:
        消解后的 codes 列表，每个元素长度可能不同（append_level）或固定（sinkhorn）。
    """
    ...
```

## Output Schema

```python
SID_MAP_FEATURES = Features({
    "item_id": Value("int32"),
    "codes": Sequence(Value("int32")),
    "sid_tokens": Value("string"),
})
```

## Invariants

1. `len(set(tuple(c) for c in all_codes)) == len(all_codes)` — 碰撞消解后全局唯一
2. `all(0 <= code < codebook_size for code in codes[i])` — 码值范围有效（前 num_codebooks 层）
3. `len(item_sid_map) == len(item_id_map)` — 每个物品都有 SID
4. 训练后 `save → load → encode` 结果与训练后直接 `encode` 一致
