# Data Model: 重构数据处理管道

**Feature**: 002-refactor-data-pipeline  
**Date**: 2026-03-02

## Entities

### 1. Interaction（保持不变）

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| user_id | string | NOT NULL | 原始用户 ID |
| item_id | string | NOT NULL | 原始商品 ID |
| timestamp | int64 | NOT NULL, ≥ 0 | Unix 时间戳（秒） |
| rating | float32 | [1.0, 5.0] | 评分 |
| review_text | string | 可为空字符串 | 评论正文 |
| review_summary | string | 可为空字符串 | 评论摘要 |

**唯一性**: `(user_id, item_id, timestamp)` 去重  
**阶段**: 阶段 1 · load/filter 步骤

### 2. UserSequence（保持不变）

| Field | Type | Constraints |
|-------|------|-------------|
| user_id | int32 | NOT NULL, ≥ 0（映射后） |
| item_ids | Sequence[int32] | NOT NULL, len ≥ 1 |
| timestamps | Sequence[int64] | len == len(item_ids) |
| ratings | Sequence[float32] | len == len(item_ids) |
| review_texts | Sequence[string] | len == len(item_ids) |
| review_summaries | Sequence[string] | len == len(item_ids) |

**阶段**: 阶段 1 · sequence 步骤

### 3. ItemMetadata（保持不变）

| Field | Type | Constraints |
|-------|------|-------------|
| item_id | string | NOT NULL |
| title | string | 可为空字符串 |
| brand | string | 可为空字符串 |
| categories | Sequence[string] | 可为空列表 |
| description | string | 可为空字符串 |
| price | float32 | 可为 None |
| image_url | string | 可为空字符串 |

**阶段**: 阶段 1 · load 步骤

### 4. IDMap（保持不变）

| Field | Type | Constraints |
|-------|------|-------------|
| original_id | string | NOT NULL, UNIQUE |
| mapped_id | int32 | NOT NULL, UNIQUE, ≥ 0 |

**阶段**: 阶段 1 · sequence 步骤

### 5. InterimSample（新增）

augment 步骤产出的样本，不含 token 字段。

| Field | Type | Constraints |
|-------|------|-------------|
| user_id | int32 | NOT NULL, ≥ 0 |
| history_item_ids | Sequence[int32] | NOT NULL, len ≤ max_seq_len |
| history_item_titles | Sequence[string] | len == len(history_item_ids) |
| target_item_id | int32 | NOT NULL, ≥ 0 |
| target_item_title | string | 可为空字符串 |

**验证规则**:
- 训练集：通过滑动窗口从 UserSequence 生成多条样本（seq_len < 2 的用户跳过）
- 验证/测试集：每用户一条样本

**阶段**: 阶段 2 · augment 步骤

### 6. NegativeSample（新增）

InterimSample 扩展，附加负采样信息。

| Field | Type | Constraints |
|-------|------|-------------|
| user_id | int32 | NOT NULL, ≥ 0 |
| history_item_ids | Sequence[int32] | 继承自 InterimSample |
| history_item_titles | Sequence[string] | 继承自 InterimSample |
| target_item_id | int32 | 继承自 InterimSample |
| target_item_title | string | 继承自 InterimSample |
| negative_item_ids | Sequence[int32] | len ≤ num_negatives，无重复 |
| negative_item_titles | Sequence[string] | len == len(negative_item_ids) |

**验证规则**:
- `negative_item_ids` 中所有 ID 不在该用户的完整交互历史中
- 当可用负样本 < `num_negatives` 时，len 可小于 num_negatives，需记录警告

**阶段**: 阶段 2 · negative_sampling 步骤

### 7. PipelineConfig（修改）

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| processing.num_negatives | int | 99 | **新增** 每样本负采样数（≥ 1） |
| processing.seed | int \| None | 42 | **新增** 随机种子（None=不固定） |

其余字段保持不变。

## Entity Relationships — 两阶段数据流

```
阶段 1（数据过滤）→ data/interim/{dataset}/{category}/
═══════════════════════════════════════════════════════

Interaction ──[load]──> raw_interactions, item_metadata
                         │
                    [filter: kcore]
                         │
                    interactions
                         │
                    [sequence: build]
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        UserSequence   IDMap     IDMap        stats.json
        (user_seqs)  (user_map) (item_map)

阶段 2（数据划分）→ data/interim/{dataset}/{category}/{split_strategy}/
═══════════════════════════════════════════════════════════════════════

UserSequence ──[split: loo/to]──> train_seq, valid_seq, test_seq
                                       │
                 ItemMetadata ──────────┤
                                       │
                              [augment: sliding_window + item_titles]
                                       │
                              train, valid, test (InterimSample)
                                       │
                              [negative_sampling]
                                       │
                              train, valid, test (NegativeSample)
                                       │
                                  stats.json
```

## Negative Sampling Edge Cases

```
available_negatives >= num_negatives? ──Yes──> sample(num_negatives)
         │No
         ▼
available_negatives > 0? ──Yes──> sample(all_available) + WARNING
         │No
         ▼
negative_item_ids=[] + WARNING
```
