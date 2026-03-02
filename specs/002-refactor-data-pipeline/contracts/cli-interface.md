# CLI & Make Interface Contract

**Feature**: 002-refactor-data-pipeline  
**Date**: 2026-03-02

## CLI Command: `process`

```bash
python -m saegenrec.dataset process <CONFIG> [OPTIONS]
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CONFIG` | Path | Yes | YAML 配置文件路径 |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--step` / `-s` | string (可多次) | 全部步骤 | 要运行的步骤名称 |
| `--dataset` | string | 取自 CONFIG | 覆盖 `dataset.name` |
| `--category` | string | 取自 CONFIG | 覆盖 `dataset.category` |
| `--kcore` | int | 取自 CONFIG | 覆盖 `processing.kcore_threshold` |
| `--max-seq-len` | int | 取自 CONFIG | 覆盖 `processing.max_seq_len` |
| `--num-negatives` | int | 取自 CONFIG | 覆盖 `processing.num_negatives` |
| `--split-strategy` | string | 取自 CONFIG | 覆盖 `processing.split_strategy` |
| `--split-ratio` | float list | 取自 CONFIG | 覆盖 `processing.split_ratio`（仅 `to` 策略生效） |
| `--seed` | int | 取自 CONFIG | 覆盖 `processing.seed` |

### Valid Step Names

阶段 1: `load`, `filter`, `sequence`  
阶段 2: `split`, `augment`, `negative_sampling`  
遗留（显式调用）: `generate`, `embed`

### Default Steps

`["load", "filter", "sequence", "split", "augment", "negative_sampling"]`

## Makefile Targets

所有参数均从 `CONFIG` 指定的 YAML 文件中读取。如需临时覆盖个别参数，可直接使用 CLI `--option`。

### `make data-filter` — 阶段 1

```makefile
CONFIG ?= configs/default.yaml

data-filter:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) \
		--step load --step filter --step sequence
```

### `make data-split` — 阶段 2

```makefile
CONFIG ?= configs/default.yaml

data-split:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) \
		--step split --step augment --step negative_sampling
```

### Usage Examples

```bash
# 阶段 1：使用默认配置
make data-filter

# 阶段 1：指定配置文件
make data-filter CONFIG=configs/amazon2015_beauty.yaml

# 阶段 2：使用默认配置
make data-split

# 阶段 2：指定配置文件
make data-split CONFIG=configs/amazon2023_fashion.yaml

# 临时覆盖个别参数（通过 CLI --option）
python -m saegenrec.dataset process configs/default.yaml \
    --step negative_sampling --num-negatives 49

# 两阶段连续运行同一配置
make data-filter CONFIG=configs/amazon2015_beauty.yaml && \
make data-split CONFIG=configs/amazon2015_beauty.yaml
```

## Disk Output Contract

### 阶段 1 输出: `{interim_dir}/{dataset}/{category}/`

```
├── raw_interactions/        # HF Dataset (Arrow)
├── item_metadata/           # HF Dataset (Arrow)
├── interactions/            # HF Dataset (Arrow)
├── user_sequences/          # HF Dataset (Arrow)
├── user_id_map/             # HF Dataset (Arrow)
├── item_id_map/             # HF Dataset (Arrow)
└── stats.json               # 阶段 1 统计
```

### 阶段 2 输出: `{interim_dir}/{dataset}/{category}/{split_strategy}/`

```
├── train_sequences/         # HF Dataset (Arrow)
├── valid_sequences/         # HF Dataset (Arrow)
├── test_sequences/          # HF Dataset (Arrow)
├── train/                   # HF Dataset (Arrow) — NegativeSample schema
├── valid/                   # HF Dataset (Arrow) — NegativeSample schema
├── test/                    # HF Dataset (Arrow) — NegativeSample schema
└── stats.json               # 阶段 2 统计
```

### stats.json — 阶段 1

```json
{
    "dataset_name": "string",
    "category": "string",
    "raw_interactions": "int",
    "filtered_interactions": "int",
    "kcore_threshold": "int",
    "kcore_iterations": "int",
    "num_users": "int",
    "num_items": "int",
    "avg_seq_length": "float",
    "min_seq_length": "int",
    "max_seq_length": "int"
}
```

### stats.json — 阶段 2

```json
{
    "split_strategy": "string",
    "split_ratio": "[float] | null",
    "train_users": "int",
    "valid_users": "int",
    "test_users": "int",
    "excluded_users": "int",
    "max_seq_len": "int",
    "train_samples": "int",
    "valid_samples": "int",
    "test_samples": "int",
    "avg_history_length": "float",
    "neg_train_num_negatives_requested": "int",
    "neg_train_num_negatives_warnings": "int",
    "neg_train_total_samples": "int",
    "neg_valid_num_negatives_requested": "int",
    "neg_valid_num_negatives_warnings": "int",
    "neg_valid_total_samples": "int",
    "neg_test_num_negatives_requested": "int",
    "neg_test_num_negatives_warnings": "int",
    "neg_test_total_samples": "int"
}
```
