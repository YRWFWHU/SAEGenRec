# 数据管道架构

## 概述

数据管道将原始 Amazon 评论数据逐步转换为序列推荐训练样本。管道采用**两阶段架构**，由 `pipeline.py` 编排：

- **阶段 1（数据过滤）**: `load → filter → sequence`，输出到 `data/interim/{dataset}/{category}/`
- **阶段 2（数据划分）**: `split → augment → negative_sampling`，输出到 `data/interim/{dataset}/{category}/{split_strategy}/`

两阶段通过磁盘解耦，切换划分策略（LOO/TO）无需重跑阶段 1。

## 处理流程

```
Raw Data (.json/.jsonl)
    │
    ▼
╔═══════════════════════════════════════════════════════════════╗
║ 阶段 1: 数据过滤 → data/interim/{dataset}/{category}/        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Step 1: Load                                                 ║
║    DatasetLoader.load_interactions() → INTERACTIONS_FEATURES   ║
║    DatasetLoader.load_item_metadata() → ITEM_METADATA_FEATURES ║
║                         │                                     ║
║                         ▼                                     ║
║  Step 2: Filter                                               ║
║    kcore_filter(interactions, k=5) → 稠密化交互子集           ║
║                         │                                     ║
║                         ▼                                     ║
║  Step 3: Sequence                                             ║
║    build_sequences(filtered) → 用户行为序列 + ID 映射表       ║
║                                                               ║
╚════════════════════════════╤══════════════════════════════════╝
                             │ 持久化到磁盘
                             ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 阶段 2: 数据划分 → data/interim/{dataset}/{category}/{strategy}/     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Step 4: Split                                                        ║
║    split_data(sequences, "loo"/"to") → train/valid/test 序列          ║
║                         │                                             ║
║                         ▼                                             ║
║  Step 5: Augment                                                      ║
║    sliding_window_augment(train) → InterimSample                      ║
║    convert_eval_split(valid/test) → InterimSample                     ║
║                         │                                             ║
║                         ▼                                             ║
║  Step 6: Negative Sampling                                            ║
║    sample_negatives(samples) → NegativeSample                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

遗留步骤（显式调用）:
  generate — 附加 tokenizer 生成 TrainingSample（含 token 字段）
  embed    — 生成商品文本嵌入向量
```

## 各步骤详解

### Step 1: Load — 数据加载

**模块**: `saegenrec.data.loaders`

通过注册表模式加载原始数据，将不同格式的源数据映射到统一 schema。

| 加载器 | 注册名 | 交互字段映射 | 元数据字段映射 |
|--------|--------|-------------|---------------|
| `Amazon2015Loader` | `amazon2015` | `reviewerID` → `user_id`, `asin` → `item_id`, `unixReviewTime` → `timestamp` | `asin` → `item_id`, `imUrl` → `image_url`, `brand` → `brand` |
| `Amazon2023Loader` | `amazon2023` | `user_id` → `user_id`, `parent_asin` → `item_id`, `timestamp`(ms) → `timestamp`(s) | `parent_asin` → `item_id`, `store` → `brand`, `images[0].large` → `image_url` |

**处理细节**:

- 交互数据按 `(user_id, item_id, timestamp)` 三元组去重
- 空 `user_id` 或 `item_id` 的记录被跳过
- Amazon 2015 metadata 文件可能是 Python dict 字面量格式，加载器自动回退到 `ast.literal_eval` 解析

### Step 2: Filter — K-core 过滤

**模块**: `saegenrec.data.processors.kcore`

迭代删除交互数少于 `k` 次的用户和商品，直到所有保留的用户和商品都满足阈值。

**算法**:

```
REPEAT:
    计算每个 user 的交互次数 user_counts
    计算每个 item 的交互次数 item_counts
    保留 user_counts >= k AND item_counts >= k 的交互
    IF 没有行被删除: BREAK
```

**配置**: `processing.kcore_threshold`（默认 5）

### Step 3: Sequence — 序列构建

**模块**: `saegenrec.data.processors.sequence`

将过滤后的扁平交互记录转换为按时间排序的用户行为序列。

**关键操作**:

1. 为每个唯一用户和商品分配连续整数 ID（0-indexed）
2. 按 `(user_id, timestamp)` 排序
3. 按用户分组，聚合为序列字段（`item_ids`, `timestamps`, `ratings`, ...）
4. 持久化 ID 映射表和中间数据到 `data/interim/`

### Step 4: Split — 数据划分

**模块**: `saegenrec.data.processors.split`

支持两种策略：

#### Leave-One-Out (LOO)

每个用户的最后一条交互 → 测试集，倒数第二条 → 验证集，其余 → 训练集。要求用户至少有 3 条交互，否则被排除。

#### Temporal Order (TO)

按全局时间戳排序所有交互，按比例（默认 `0.8:0.1:0.1`）划分到 train/valid/test。

### Step 5: Augment — 数据增强

**模块**: `saegenrec.data.processors.augment`

augment 步骤与 tokenizer 完全解耦，输出 `INTERIM_SAMPLE_FEATURES` schema（不含 token 字段）。

#### 滑动窗口增强（训练集）

对每个用户的训练序列 `[i₁, i₂, ..., iₙ]`，生成 `n-1` 个样本：

- 样本 k: `history = [i₁, ..., iₖ][-max_seq_len:]`, `target = iₖ₊₁`

#### 评估集转换（验证/测试集）

每个用户产生恰好一个样本，target 为序列中唯一的目标商品。

#### InterimSample 格式

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `history_item_ids` | Sequence(int32) | 历史商品 ID 序列 |
| `history_item_titles` | Sequence(string) | 历史商品标题 |
| `target_item_id` | int32 | 目标商品 ID |
| `target_item_title` | string | 目标商品标题 |

### Step 6: Negative Sampling — 负采样

**模块**: `saegenrec.data.processors.negative_sampling`

为每条样本（train/valid/test）随机采样用户未交互过的商品作为负样本，用于重排序任务。

**关键特性**:

- 使用 `numpy.random.Generator` + 随机种子，保证可复现
- 负样本严格排除用户的完整交互历史
- 可用负样本不足时降级采样并记录 WARNING

**配置**: `processing.num_negatives`（默认 99）, `processing.seed`（默认 42）

#### NegativeSample 格式

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `history_item_ids` | Sequence(int32) | 历史商品 ID 序列 |
| `history_item_titles` | Sequence(string) | 历史商品标题 |
| `target_item_id` | int32 | 目标商品 ID |
| `target_item_title` | string | 目标商品标题 |
| `negative_item_ids` | Sequence(int32) | 负样本商品 ID 列表 |
| `negative_item_titles` | Sequence(string) | 负样本商品标题列表 |

### 遗留步骤: Generate + Embed

#### Generate（遗留）

**模块**: `saegenrec.data.processors.final`

在 InterimSample 基础上通过 `ItemTokenizer` 附加 token 字段，生成 `TRAINING_SAMPLE_FEATURES` schema。需显式通过 `--step generate` 调用。

#### Embed（可选）

**模块**: `saegenrec.data.embeddings.text`

使用 sentence-transformers 模型将商品文本信息编码为稠密向量。

- 默认模型: `all-MiniLM-L6-v2`
- 拼接字段: `title`, `brand`, `categories`（可配置）
- 支持 CPU/CUDA 设备
- 分批处理（默认 batch_size=256）

## 数据 Schema 总览

管道使用 HuggingFace `Features` 定义严格的 schema，在每个阶段验证数据合规性。

### Interactions

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | string | 原始用户标识 |
| `item_id` | string | 原始商品标识 |
| `timestamp` | int64 | Unix 时间戳（秒） |
| `rating` | float32 | 评分 |
| `review_text` | string | 评论正文 |
| `review_summary` | string | 评论摘要 |

### UserSequences

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `item_ids` | Sequence(int32) | 按时间排序的商品 ID 序列 |
| `timestamps` | Sequence(int64) | 对应时间戳序列 |
| `ratings` | Sequence(float32) | 对应评分序列 |
| `review_texts` | Sequence(string) | 对应评论序列 |
| `review_summaries` | Sequence(string) | 对应摘要序列 |

### ItemMetadata

| 字段 | 类型 | 说明 |
|------|------|------|
| `item_id` | string | 原始商品标识 |
| `title` | string | 商品标题 |
| `brand` | string | 品牌 |
| `categories` | Sequence(string) | 分类标签 |
| `description` | string | 商品描述 |
| `price` | float32 | 价格（可为 null） |
| `image_url` | string | 商品图片 URL |

### IDMap

| 字段 | 类型 | 说明 |
|------|------|------|
| `original_id` | string | 原始 ID |
| `mapped_id` | int32 | 映射后连续整数 ID |

### InterimSample

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `history_item_ids` | Sequence(int32) | 历史商品 ID 序列 |
| `history_item_titles` | Sequence(string) | 历史商品标题 |
| `target_item_id` | int32 | 目标商品 ID |
| `target_item_title` | string | 目标商品标题 |

### NegativeSample

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `history_item_ids` | Sequence(int32) | 历史商品 ID 序列 |
| `history_item_titles` | Sequence(string) | 历史商品标题 |
| `target_item_id` | int32 | 目标商品 ID |
| `target_item_title` | string | 目标商品标题 |
| `negative_item_ids` | Sequence(int32) | 负样本商品 ID 列表 |
| `negative_item_titles` | Sequence(string) | 负样本商品标题列表 |

### TrainingSample（遗留）

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int32 | 映射后用户 ID |
| `history_item_ids` | Sequence(int32) | 历史商品 ID 序列 |
| `history_item_tokens` | Sequence(Sequence(int32)) | 历史商品 token 序列 |
| `history_item_titles` | Sequence(string) | 历史商品标题 |
| `target_item_id` | int32 | 目标商品 ID |
| `target_item_tokens` | Sequence(int32) | 目标商品 token 序列 |
| `target_item_title` | string | 目标商品标题 |

### TextEmbedding

| 字段 | 类型 | 说明 |
|------|------|------|
| `item_id` | int32 | 映射后商品 ID |
| `embedding` | Sequence(float32) | 文本嵌入向量 |

## 扩展指南

### 添加新的数据加载器

```python
from saegenrec.data.loaders.base import DatasetLoader, register_loader

@register_loader("my_dataset")
class MyDatasetLoader(DatasetLoader):
    def load_interactions(self, data_dir):
        # 返回符合 INTERACTIONS_FEATURES schema 的 Dataset
        ...

    def load_item_metadata(self, data_dir):
        # 返回符合 ITEM_METADATA_FEATURES schema 的 Dataset
        ...
```

### 添加新的 ItemTokenizer

```python
from saegenrec.data.tokenizers.base import ItemTokenizer, register_tokenizer

@register_tokenizer("my_tokenizer")
class MyTokenizer(ItemTokenizer):
    def tokenize(self, item_id: int) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def token_length(self) -> int: ...
```
