# Quickstart: 重构数据处理管道

**Feature**: 002-refactor-data-pipeline  
**Date**: 2026-03-02

## Prerequisites

1. Python 3.11，依赖已安装：`pip install -e .`
2. 原始数据已放在 `data/raw/` 下

## 两阶段工作流

### 阶段 1：数据过滤（load → filter → sequence）

```bash
# 使用 Make（参数从配置文件读取）
make data-filter CONFIG=configs/examples/amazon2015_beauty.yaml

# 使用默认配置
make data-filter

# 使用 CLI（可临时覆盖参数）
python -m saegenrec.dataset process configs/default.yaml \
    --step load --step filter --step sequence \
    --kcore 10
```

输出: `data/interim/amazon2015/Beauty/`（user_sequences, id_maps, stats.json 等）

### 阶段 2：数据划分（split → augment → negative_sampling）

```bash
# LOO 划分
make data-split CONFIG=configs/examples/amazon2015_beauty.yaml

# 切换 TO 划分（使用不同配置文件，无需重跑阶段 1）
make data-split CONFIG=configs/examples/amazon2015_beauty.yaml --split-strategy to --split-ratio 0.8,0.1,0.1
```

输出: `data/interim/amazon2015/Beauty/loo/` 和 `data/interim/amazon2015/Beauty/to/`

### 单步重跑

```bash
# 仅重跑负采样
python -m saegenrec.dataset process configs/default.yaml \
    --step negative_sampling --num-negatives 49
```

## 查看输出

```python
from datasets import load_from_disk

train = load_from_disk("data/interim/amazon2015/Beauty/loo/train")
print(train.column_names)
# ['user_id', 'history_item_ids', 'history_item_titles',
#  'target_item_id', 'target_item_title',
#  'negative_item_ids', 'negative_item_titles']

assert "history_item_tokens" not in train.column_names
```

## 运行测试

```bash
make test
python -m pytest tests/unit/data/test_negative_sampling.py -v
```

## 配置示例

```yaml
dataset:
  name: "amazon2015"
  category: "Baby"
  raw_dir: "data/raw"

processing:
  kcore_threshold: 5
  split_strategy: "loo"
  split_ratio: [0.8, 0.1, 0.1]
  max_seq_len: 20
  num_negatives: 99
  seed: 42

output:
  interim_dir: "data/interim"
  processed_dir: "data/processed"
```
