# Data Model: 生成式推荐数据处理流水线

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## Entity Relationship Overview

```
RawInteraction ──(load)──> Interaction ──(k-core filter)──> FilteredInteraction
                                                                    │
RawItemMetadata ──(load)──> ItemMetadata                           │
                                │                                   │
                                ▼                                   ▼
                         TextEmbedding              UserSequence ──(split)──> SplitSequences
                                │                                                   │
                                ▼                                                   ▼
                         ItemTokenizer ◄───── PassthroughTokenizer     SlidingWindowSamples
                                │                                                   │
                                └──────────────────────┬────────────────────────────┘
                                                       ▼
                                                TrainingSample
```

## Entities

### 1. RawInteraction（原始交互记录）

数据源中的原始评论/交互记录，格式因数据集版本而异。

**Amazon2015 字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `reviewerID` | `string` | 用户原始 ID |
| `asin` | `string` | 商品 ASIN |
| `reviewText` | `string` | 评论正文 |
| `overall` | `float` | 评分 (1-5) |
| `summary` | `string` | 评论摘要/标题 |
| `unixReviewTime` | `int` | Unix 时间戳（秒） |

**Amazon2023 字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | `string` | 用户原始 ID |
| `parent_asin` | `string` | 商品父 ASIN |
| `text` | `string` | 评论正文 |
| `rating` | `float` | 评分 (1-5) |
| `title` | `string` | 评论标题 |
| `timestamp` | `int` | 时间戳（毫秒） |

### 2. Interaction（统一交互记录）

DatasetLoader 输出的标准化交互记录。所有数据集版本的原始数据经 Loader 转换后统一为此格式。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `user_id` | `Value("string")` | 用户原始 ID |
| `item_id` | `Value("string")` | 物品原始 ID |
| `timestamp` | `Value("int64")` | Unix 时间戳（秒，统一精度） |
| `rating` | `Value("float32")` | 评分 |
| `review_text` | `Value("string")` | 评论正文 |
| `review_summary` | `Value("string")` | 评论标题/摘要 |

**Validation Rules**:
- `user_id` 和 `item_id` 不可为空
- `timestamp` > 0
- `rating` ∈ [1.0, 5.0]
- Amazon2023 的毫秒时间戳在 Loader 中转换为秒

**Uniqueness**: `(user_id, item_id, timestamp)` 三元组唯一。存在重复时保留第一条。

### 3. ItemMetadata（物品元数据）

DatasetLoader 输出的标准化物品元数据。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `item_id` | `Value("string")` | 物品原始 ID |
| `title` | `Value("string")` | 商品标题 |
| `brand` | `Value("string")` | 品牌 |
| `categories` | `Sequence(Value("string"))` | 品类列表 |
| `description` | `Value("string")` | 商品描述 |
| `price` | `Value("float32")` | 价格（可为 null） |
| `image_url` | `Value("string")` | 图片 URL（可为 null） |

**Field Mapping (Amazon2015)**:
- `item_id` ← `asin`
- `brand` ← `brand`
- `categories` ← `categories[0]`（取第一层列表）
- `description` ← `description`
- `image_url` ← `imUrl`

**Field Mapping (Amazon2023)**:
- `item_id` ← `parent_asin`
- `brand` ← `store`
- `categories` ← `categories`
- `description` ← `' '.join(description)` （description 是 list）
- `image_url` ← `images[0].large`（取第一张图的 large URL）

**Uniqueness**: `item_id` 唯一。

### 4. IDMap（ID 映射）

用户和物品的原始 ID 到连续整数 ID 的映射。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `original_id` | `Value("string")` | 原始 ID |
| `mapped_id` | `Value("int32")` | 连续整数 ID（从 0 开始） |

**Validation Rules**:
- `mapped_id` 从 0 开始连续递增
- 双射关系：原始 ID 和映射 ID 均唯一

### 5. UserSequence（用户交互序列）

K-core 过滤后，按时间排序的用户交互序列。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `user_id` | `Value("int32")` | 映射后的用户 ID |
| `item_ids` | `Sequence(Value("int32"))` | 按时间排序的物品 ID 序列 |
| `timestamps` | `Sequence(Value("int64"))` | 对应的时间戳序列 |
| `ratings` | `Sequence(Value("float32"))` | 对应的评分序列 |
| `review_texts` | `Sequence(Value("string"))` | 对应的评论正文序列 |
| `review_summaries` | `Sequence(Value("string"))` | 对应的评论摘要序列 |

**Validation Rules**:
- `item_ids`, `timestamps`, `ratings`, `review_texts`, `review_summaries` 长度一致
- `timestamps` 严格非递减排序
- K-core 过滤后，每个用户的序列长度 ≥ K（默认 5）
- `user_id` 唯一

### 6. TrainingSample（训练样本）

滑动窗口增强后的最终训练样本。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `user_id` | `Value("int32")` | 用户 ID |
| `history_item_ids` | `Sequence(Value("int32"))` | 历史物品 ID 序列 |
| `history_item_tokens` | `Sequence(Sequence(Value("int32")))` | 历史物品的 token 序列（每个物品一组 token） |
| `history_item_titles` | `Sequence(Value("string"))` | 历史物品标题序列 |
| `target_item_id` | `Value("int32")` | 目标物品 ID |
| `target_item_tokens` | `Sequence(Value("int32"))` | 目标物品的 token 序列 |
| `target_item_title` | `Value("string")` | 目标物品标题 |

**Validation Rules**:
- `history_item_ids` 长度 ∈ [1, max_seq_len]（默认 max_seq_len=20）
- `history_item_tokens` 长度与 `history_item_ids` 一致
- `history_item_titles` 长度与 `history_item_ids` 一致
- 训练集样本：每个用户可产生多条样本（滑动窗口增强）
- 验证/测试集样本：每个用户一条样本（LOO）或按时间划分的多条样本（TO）

### 7. TextEmbedding（物品文本 Embedding）

使用预训练语言模型对物品元数据文本生成的向量表示。

**HuggingFace Dataset Schema**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `item_id` | `Value("int32")` | 映射后的物品 ID |
| `embedding` | `Array2D(shape=(1, dim), dtype="float32")` | 文本 embedding 向量 |

**Validation Rules**:
- `item_id` 唯一，覆盖所有通过 K-core 过滤的物品
- `embedding` 维度由预训练模型决定（如 `all-MiniLM-L6-v2` → 384 维）
- 向量为 L2 归一化后的结果

## State Transitions

### 数据流水线状态流

```
Raw Data (immutable)
    │
    ▼
[DatasetLoader.load_interactions()]  ──→  Interaction Dataset
[DatasetLoader.load_item_metadata()]  ──→  ItemMetadata Dataset
    │
    ▼
[K-core Filter]  ──→  Filtered Interaction Dataset
                       + Statistics (原始数/过滤后数/用户数/物品数)
    │
    ▼
[Sequence Builder]  ──→  UserSequence Dataset
                          + UserIDMap Dataset
                          + ItemIDMap Dataset
                          + Statistics (平均序列长度)
    │
    ▼
[Data Splitter]  ──→  Train/Valid/Test UserSequence Datasets
                       + Statistics (各集样本数)
    │
    ▼
[Sliding Window]  ──→  Train TrainingSample Dataset (augmented)
                        Valid/Test TrainingSample Dataset (直接转换)
    │
    ▼
[Final Generator]  ──→  TrainingSample Datasets with item tokens + text info
    │                     (data/processed/ 目录)
    │
    ├──(optional)──→  [Text Embedding Generator]  ──→  TextEmbedding Dataset
    │                                                    (data/interim/ 目录)
    └──(optional)──→  [Image Downloader]  ──→  Image files
                                                (data/external/images/ 目录)
```

### 配置状态

```
YAML Config File
    │
    ▼
[Load & Parse]  ──→  PipelineConfig (dataclass)
                      ├── dataset: DatasetConfig
                      │   ├── name: str          ("amazon2015" | "amazon2023")
                      │   ├── category: str      ("Baby", "All_Beauty", ...)
                      │   └── raw_dir: Path
                      ├── processing: ProcessingConfig
                      │   ├── kcore_threshold: int     (default: 5)
                      │   ├── split_strategy: str      ("loo" | "to")
                      │   ├── split_ratio: tuple       (default: (0.8, 0.1, 0.1))
                      │   └── max_seq_len: int         (default: 20)
                      ├── tokenizer: TokenizerConfig
                      │   ├── name: str                 (default: "passthrough")
                      │   └── params: dict             (default: {})
                      ├── embedding: EmbeddingConfig
                      │   ├── enabled: bool            (default: false)
                      │   ├── model_name: str          (default: "all-MiniLM-L6-v2")
                      │   ├── text_fields: list[str]   (default: ["title", "brand", "categories"])
                      │   └── batch_size: int          (default: 256)
                      └── output: OutputConfig
                          ├── interim_dir: Path
                          └── processed_dir: Path
```
