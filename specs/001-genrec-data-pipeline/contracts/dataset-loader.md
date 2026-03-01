# Contract: DatasetLoader 抽象接口

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## Interface Definition

```python
from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset


class DatasetLoader(ABC):
    """数据集加载器抽象基类。
    
    每种原始数据格式（如 Amazon2015、Amazon2023）MUST 实现此接口。
    所有实现 MUST 将原始字段映射为统一的输出 schema。
    """

    @abstractmethod
    def load_interactions(self, data_dir: Path) -> Dataset:
        """加载原始交互记录并转换为统一格式。

        Args:
            data_dir: 包含原始数据文件的目录路径。

        Returns:
            HuggingFace Dataset，schema:
                - user_id: string     — 用户原始 ID
                - item_id: string     — 物品原始 ID
                - timestamp: int64    — Unix 时间戳（秒）
                - rating: float32     — 评分
                - review_text: string — 评论正文
                - review_summary: string — 评论标题/摘要

        Raises:
            FileNotFoundError: 数据文件不存在
            ValueError: 数据格式不符合预期
        """
        ...

    @abstractmethod
    def load_item_metadata(self, data_dir: Path) -> Dataset:
        """加载物品元数据并转换为统一格式。

        Args:
            data_dir: 包含元数据文件的目录路径。

        Returns:
            HuggingFace Dataset，schema:
                - item_id: string               — 物品原始 ID
                - title: string                  — 商品标题
                - brand: string                  — 品牌
                - categories: Sequence(string)   — 品类列表
                - description: string            — 商品描述
                - price: float32                 — 价格（可为 null）
                - image_url: string              — 图片 URL（可为 null）

        Raises:
            FileNotFoundError: 元数据文件不存在
            ValueError: 数据格式不符合预期
        """
        ...
```

## Registry

```python
LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {}


def register_loader(name: str):
    """装饰器：注册 DatasetLoader 实现。"""
    def decorator(cls: type[DatasetLoader]):
        LOADER_REGISTRY[name] = cls
        return cls
    return decorator


def get_loader(name: str) -> DatasetLoader:
    """根据名称获取 DatasetLoader 实例。"""
    if name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset loader: {name}. "
            f"Available: {list(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[name]()
```

## Implementations

### Amazon2015Loader

```python
@register_loader("amazon2015")
class Amazon2015Loader(DatasetLoader):
    """Amazon 2015 数据集加载器。
    
    文件格式：每行一个 JSON 对象（.json 扩展名但实际为 JSON Lines）。
    评论文件：{Category}.json
    元数据文件：meta_{Category}.json
    """
    ...
```

**Field Mapping**:

| 统一字段 | Amazon2015 原始字段 |
|----------|-------------------|
| `user_id` | `reviewerID` |
| `item_id` | `asin` |
| `timestamp` | `unixReviewTime` |
| `rating` | `overall` |
| `review_text` | `reviewText` |
| `review_summary` | `summary` |
| `title` | `title` |
| `brand` | `brand` |
| `categories` | `categories[0]` |
| `description` | `description` |
| `price` | `price` |
| `image_url` | `imUrl` |

### Amazon2023Loader

```python
@register_loader("amazon2023")
class Amazon2023Loader(DatasetLoader):
    """Amazon 2023 数据集加载器。
    
    文件格式：JSON Lines（.jsonl 扩展名）。
    评论文件：{Category}.jsonl
    元数据文件：meta_{Category}.jsonl
    """
    ...
```

**Field Mapping**:

| 统一字段 | Amazon2023 原始字段 |
|----------|-------------------|
| `user_id` | `user_id` |
| `item_id` | `parent_asin` |
| `timestamp` | `timestamp // 1000`（毫秒→秒） |
| `rating` | `rating` |
| `review_text` | `text` |
| `review_summary` | `title` |
| `title` | `title`（元数据） |
| `brand` | `store` |
| `categories` | `categories` |
| `description` | `' '.join(description)` |
| `price` | `price`（解析字符串为 float） |
| `image_url` | `images[0]["large"]` |

## Invariants

1. 所有实现 MUST 返回符合统一 schema 的 Dataset，不可包含额外字段
2. 缺失字段 MUST 填充为空字符串（string）或 None（数值类型）
3. 时间戳 MUST 统一为 Unix 秒精度（int64）
4. 同一 `(user_id, item_id, timestamp)` 三元组存在重复时 MUST 去重（保留第一条）
