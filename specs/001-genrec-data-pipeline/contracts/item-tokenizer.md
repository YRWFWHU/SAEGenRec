# Contract: ItemTokenizer 抽象接口

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## Interface Definition

```python
from abc import ABC, abstractmethod


class ItemTokenizer(ABC):
    """物品 Tokenizer 抽象接口。
    
    将物品 ID（连续整数）转换为一组离散 token 表示。
    不限定 embedding 来源和量化方法，具体实现自行决定。
    """

    @abstractmethod
    def tokenize(self, item_id: int) -> list[int]:
        """将物品 ID 转换为离散 token 序列。

        Args:
            item_id: 映射后的物品整数 ID。

        Returns:
            离散 token 序列（整数列表）。长度由实现决定。
        """
        ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> int:
        """将离散 token 序列还原为物品 ID。

        Args:
            tokens: 离散 token 序列。

        Returns:
            物品整数 ID。

        Raises:
            ValueError: token 序列无法还原为有效物品 ID。
        """
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """token 词表大小。"""
        ...

    @property
    @abstractmethod
    def token_length(self) -> int:
        """每个物品的 token 序列长度。"""
        ...

    def tokenize_batch(self, item_ids: list[int]) -> list[list[int]]:
        """批量 tokenize（默认实现逐个调用，子类可覆写优化）。"""
        return [self.tokenize(item_id) for item_id in item_ids]
```

## Registry

```python
TOKENIZER_REGISTRY: dict[str, type[ItemTokenizer]] = {}


def register_tokenizer(name: str):
    """装饰器：注册 ItemTokenizer 实现。"""
    def decorator(cls: type[ItemTokenizer]):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator


def get_tokenizer(name: str, **kwargs) -> ItemTokenizer:
    """根据名称获取 ItemTokenizer 实例。"""
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer: {name}. "
            f"Available: {list(TOKENIZER_REGISTRY.keys())}"
        )
    return TOKENIZER_REGISTRY[name](**kwargs)
```

## Implementations

### PassthroughTokenizer

```python
@register_tokenizer("passthrough")
class PassthroughTokenizer(ItemTokenizer):
    """透传 Tokenizer：将物品整数 ID 直接作为单个 token 返回。"""

    def __init__(self, num_items: int):
        self._num_items = num_items

    def tokenize(self, item_id: int) -> list[int]:
        return [item_id]

    def detokenize(self, tokens: list[int]) -> int:
        if len(tokens) != 1:
            raise ValueError(f"Expected 1 token, got {len(tokens)}")
        return tokens[0]

    @property
    def vocab_size(self) -> int:
        return self._num_items

    @property
    def token_length(self) -> int:
        return 1
```

## Future Implementations (out of scope)

以下实现不在本流水线范围内，但接口设计 MUST 支持：

- **RQVAETokenizer**: 基于 RQ-VAE 量化的 Tokenizer，消费预计算的文本/协同 embedding
- **PQTokenizer**: 基于乘积量化的 Tokenizer
- **KMeansTokenizer**: 基于多层 K-Means 聚类的 Tokenizer

## Invariants

1. `tokenize(detokenize(tokens)) == tokens`（round-trip 一致性）
2. `detokenize(tokenize(item_id)) == item_id`（round-trip 一致性）
3. `len(tokenize(item_id)) == token_length`（输出长度固定）
4. `all(0 <= t < vocab_size for t in tokenize(item_id))`（token 值在词表范围内）
