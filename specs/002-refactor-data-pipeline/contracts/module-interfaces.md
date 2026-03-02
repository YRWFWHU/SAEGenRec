# Module Interface Contracts

**Feature**: 002-refactor-data-pipeline  
**Date**: 2026-03-02

## 1. `saegenrec.data.processors.augment` (Modified)

### `sliding_window_augment`

```python
def sliding_window_augment(
    train_sequences: Dataset,       # UserSequence schema
    item_titles: dict[int, str],    # mapped_id → title
    max_seq_len: int = 20,
) -> Dataset:
    """
    通过滑动窗口从训练序列生成 (history, target) 样本对。
    
    变更: 移除 tokenizer 参数，输出 INTERIM_SAMPLE_FEATURES schema。
    """
```

### `convert_eval_split`

```python
def convert_eval_split(
    eval_sequences: Dataset,        # UserSequence schema
    item_titles: dict[int, str],    # mapped_id → title
    max_seq_len: int = 20,
) -> Dataset:
    """
    将验证/测试序列转为 InterimSample 格式。
    
    变更: 移除 tokenizer 参数，输出 INTERIM_SAMPLE_FEATURES schema。
    """
```

## 2. `saegenrec.data.processors.negative_sampling` (New)

### `sample_negatives`

```python
def sample_negatives(
    samples: Dataset,                           # INTERIM_SAMPLE_FEATURES
    user_interacted_items: dict[int, set[int]], # user_id → 完整交互 item_id 集合
    all_item_ids: list[int],                    # 全局商品 ID 列表
    item_titles: dict[int, str],                # mapped_id → title
    num_negatives: int = 99,
    seed: int | None = 42,
) -> tuple[Dataset, dict]:
    """
    为每条样本采样负样本。
    
    Returns: (NEGATIVE_SAMPLE_FEATURES dataset, stats dict)
    """
```

### `build_user_interacted_items`

```python
def build_user_interacted_items(
    user_sequences: Dataset,  # UserSequence schema (split 前的完整序列)
) -> dict[int, set[int]]:
    """构建 {user_id: {item_id, ...}} 映射。"""
```

## 3. `saegenrec.data.pipeline` (Modified)

```python
STAGE1_STEPS = ["load", "filter", "sequence"]
STAGE2_STEPS = ["split", "augment", "negative_sampling"]
ALL_STEPS = STAGE1_STEPS + STAGE2_STEPS
LEGACY_STEPS = ["generate", "embed"]

def run_pipeline(config: PipelineConfig, steps: list[str] | None = None) -> dict:
    """
    变更:
    - 默认步骤 = ALL_STEPS (不含 generate/embed)
    - 阶段 1 输出到 interim_dir/{dataset}/{category}/
    - 阶段 2 输出到 interim_dir/{dataset}/{category}/{split_strategy}/
    - augment 不依赖 tokenizer
    - 新增 negative_sampling 步骤
    """
```

## 4. `saegenrec.data.config` (Modified)

```python
@dataclass
class ProcessingConfig:
    kcore_threshold: int = 5
    split_strategy: str = "loo"
    split_ratio: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    max_seq_len: int = 20
    num_negatives: int = 99      # NEW
    seed: int | None = 42         # NEW

    # Validation:
    # - split_ratio 仅在 split_strategy == "to" 时验证（须 sum=1.0）
    # - split_strategy == "loo" 时 split_ratio 被忽略
```

## 5. `saegenrec.data.schemas` (Modified)

```python
# NEW
INTERIM_SAMPLE_FEATURES = Features({
    "user_id": Value("int32"),
    "history_item_ids": Sequence(Value("int32")),
    "history_item_titles": Sequence(Value("string")),
    "target_item_id": Value("int32"),
    "target_item_title": Value("string"),
})

# NEW
NEGATIVE_SAMPLE_FEATURES = Features({
    "user_id": Value("int32"),
    "history_item_ids": Sequence(Value("int32")),
    "history_item_titles": Sequence(Value("string")),
    "target_item_id": Value("int32"),
    "target_item_title": Value("string"),
    "negative_item_ids": Sequence(Value("int32")),
    "negative_item_titles": Sequence(Value("string")),
})
```

所有现有 Features 定义保持不变。
