# Contract: Data Schemas

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## 存储目录结构

```text
data/
├── raw/                                    # 原始不可变数据
│   ├── Amazon2015/
│   │   ├── Baby/
│   │   │   ├── Baby.json                   # 评论数据 (JSON Lines)
│   │   │   └── meta_Baby.json              # 物品元数据 (JSON Lines)
│   │   ├── Beauty/
│   │   └── ...
│   └── Amazon2023/
│       ├── All_Beauty/
│       │   ├── All_Beauty.jsonl             # 评论数据 (JSONL)
│       │   └── meta_All_Beauty.jsonl        # 物品元数据 (JSONL)
│       └── ...
│
├── interim/                                # 中间转换数据
│   └── {dataset_name}/{category}/
│       ├── interactions/                   # HF Dataset: 统一交互记录
│       ├── user_sequences/                 # HF Dataset: 用户交互序列
│       ├── item_metadata/                  # HF Dataset: 物品元数据
│       ├── user_id_map/                    # HF Dataset: 用户 ID 映射
│       ├── item_id_map/                    # HF Dataset: 物品 ID 映射
│       ├── text_embeddings/                # HF Dataset: 文本 embedding (可选)
│       └── stats.json                      # 处理统计信息
│
├── processed/                              # 最终建模用数据
│   └── {dataset_name}/{category}/{split_strategy}/
│       ├── train/                          # HF Dataset: 训练集
│       ├── valid/                          # HF Dataset: 验证集
│       ├── test/                           # HF Dataset: 测试集
│       └── stats.json                      # 划分与增强统计信息
│
└── external/                               # 第三方数据
    └── images/                             # 下载的商品图片
        └── {dataset_name}/{category}/
            ├── {item_id}.jpg
            └── ...
```

## HuggingFace Dataset Schemas

### Interactions Dataset (`interim/{name}/{cat}/interactions/`)

```python
from datasets import Features, Value

INTERACTIONS_FEATURES = Features({
    "user_id": Value("string"),
    "item_id": Value("string"),
    "timestamp": Value("int64"),
    "rating": Value("float32"),
    "review_text": Value("string"),
    "review_summary": Value("string"),
})
```

### UserSequences Dataset (`interim/{name}/{cat}/user_sequences/`)

```python
from datasets import Features, Sequence, Value

USER_SEQUENCES_FEATURES = Features({
    "user_id": Value("int32"),
    "item_ids": Sequence(Value("int32")),
    "timestamps": Sequence(Value("int64")),
    "ratings": Sequence(Value("float32")),
    "review_texts": Sequence(Value("string")),
    "review_summaries": Sequence(Value("string")),
})
```

### ItemMetadata Dataset (`interim/{name}/{cat}/item_metadata/`)

```python
ITEM_METADATA_FEATURES = Features({
    "item_id": Value("string"),
    "title": Value("string"),
    "brand": Value("string"),
    "categories": Sequence(Value("string")),
    "description": Value("string"),
    "price": Value("float32"),
    "image_url": Value("string"),
})
```

### IDMap Datasets (`interim/{name}/{cat}/user_id_map/`, `item_id_map/`)

```python
ID_MAP_FEATURES = Features({
    "original_id": Value("string"),
    "mapped_id": Value("int32"),
})
```

### TrainingSample Datasets (`processed/{name}/{cat}/{strategy}/train|valid|test/`)

```python
TRAINING_SAMPLE_FEATURES = Features({
    "user_id": Value("int32"),
    "history_item_ids": Sequence(Value("int32")),
    "history_item_tokens": Sequence(Sequence(Value("int32"))),
    "history_item_titles": Sequence(Value("string")),
    "target_item_id": Value("int32"),
    "target_item_tokens": Sequence(Value("int32")),
    "target_item_title": Value("string"),
})
```

### TextEmbedding Dataset (`interim/{name}/{cat}/text_embeddings/`)

```python
from datasets import Array2D

TEXT_EMBEDDING_FEATURES = Features({
    "item_id": Value("int32"),
    "embedding": Array2D(shape=(1, None), dtype="float32"),
})
```

## Statistics Files

### `interim/{name}/{cat}/stats.json`

```json
{
    "dataset_name": "amazon2015",
    "category": "Baby",
    "raw_interactions": 160792,
    "filtered_interactions": 53888,
    "kcore_threshold": 5,
    "kcore_iterations": 4,
    "num_users": 8765,
    "num_items": 6432,
    "avg_seq_length": 6.15,
    "min_seq_length": 5,
    "max_seq_length": 142
}
```

### `processed/{name}/{cat}/{strategy}/stats.json`

```json
{
    "split_strategy": "loo",
    "split_ratio": null,
    "max_seq_len": 20,
    "train_users": 8765,
    "valid_users": 8765,
    "test_users": 8765,
    "train_samples": 45123,
    "valid_samples": 8765,
    "test_samples": 8765,
    "excluded_users": 0,
    "tokenizer": "passthrough",
    "avg_history_length": 5.15
}
```

## Data Integrity Invariants

1. `data/raw/` 中的文件 MUST 保持不可变（只读）
2. `data/interim/` 和 `data/processed/` 中的所有 HF Dataset MUST 可通过 `datasets.load_from_disk()` 加载
3. 所有 Dataset MUST 包含对应 Features 定义中的全部字段，不可有额外字段
4. ID 映射 MUST 满足双射：`len(unique(original_id)) == len(unique(mapped_id)) == len(dataset)`
5. 用户序列中的 `item_ids` MUST 使用映射后的整数 ID
6. 训练/验证/测试集之间 MUST 无样本重叠
7. 时间维度零泄露：训练集所有时间戳 ≤ 验证集所有时间戳 ≤ 测试集所有时间戳（TO 策略）
8. 统计信息 MUST 在每次处理后自动生成，与实际数据一致
