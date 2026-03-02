# Quickstart: 生成式推荐架构

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02

## 前置条件

确保已完成 Stage 1 + Stage 2 + Embedding 步骤：

```bash
# Stage 1: 数据加载、过滤、序列构建
make data-filter CONFIG=configs/default.yaml

# Stage 2: 数据划分、增强、负采样
make data-split CONFIG=configs/default.yaml

# Embedding: 语义 + 协同
make data-embed CONFIG=configs/default.yaml
```

## 配置

在 `configs/default.yaml` 中添加：

```yaml
item_tokenizer:
  enabled: true
  name: "rqvae"              # 或 "rqkmeans"
  num_codebooks: 4
  codebook_size: 256
  collision_strategy: "append_level"
  params:
    hidden_dim: 256
    latent_dim: 64
    lr: 0.001
    epochs: 100
    batch_size: 512
    device: "cuda"

sft_builder:
  enabled: true
  tasks: ["seqrec", "item2index", "index2item"]
  template_file: "configs/templates/sft_prompts.yaml"
  max_history_len: 20
  seed: 42
```

## 运行

### 方式一：管道步骤（推荐）

```bash
# 仅 tokenize
make data-tokenize CONFIG=configs/default.yaml

# 仅 SFT 数据构建
make data-build-sft CONFIG=configs/default.yaml

# 一键执行 tokenize + SFT 构建
python -m saegenrec.dataset process configs/default.yaml \
  --step tokenize --step build-sft
```

### 方式二：独立 CLI 命令

```bash
# Tokenize
python -m saegenrec.dataset tokenize configs/default.yaml

# Build SFT data
python -m saegenrec.dataset build-sft configs/default.yaml

# 强制覆盖已有结果
python -m saegenrec.dataset tokenize configs/default.yaml --force
```

### 方式三：使用 RQ-KMeans (CPU)

```yaml
item_tokenizer:
  enabled: true
  name: "rqkmeans"
  num_codebooks: 4
  codebook_size: 256
  collision_strategy: "append_level"
  params:
    use_constrained: false
    kmeans_niter: 20
    use_gpu: false
```

## 输出

所有模型产物输出到 `data/processed/`（CCDS "最终建模用数据集"）：

```text
data/processed/{dataset}/{category}/
├── item_sid_map/          # HF Dataset: item_id, codes, sid_tokens
├── tokenizer_model/       # 训练好的 tokenizer 模型权重
└── sft_data/              # HF Dataset: task_type, instruction, input, output
```

## 验证

```python
from datasets import load_from_disk

# 检查 SID map
sid_map = load_from_disk("data/processed/amazon2015/Beauty/item_sid_map")
print(f"Items: {len(sid_map)}")
print(f"Sample: {sid_map[0]}")

# 检查唯一性
codes_set = set(tuple(c) for c in sid_map["codes"])
assert len(codes_set) == len(sid_map), "SID collision detected!"

# 检查 SFT 数据
sft_data = load_from_disk("data/processed/amazon2015/Beauty/sft_data")
print(f"SFT samples: {len(sft_data)}")
print(f"Task types: {set(sft_data['task_type'])}")
print(f"Sample:\n{sft_data[0]}")
```

## 代码结构

核心模块位于 `saegenrec/modeling/`：

```text
saegenrec/modeling/
├── tokenizers/     # ItemTokenizer (RQ-VAE, RQ-KMeans, 碰撞消解)
├── sft/            # SFT 数据构建 (SeqRec, Item2Index, Index2Item)
├── genrec/         # LLM 模型接口 (HuggingFace 风格 ABC)
└── decoding/       # 约束解码 (SID Trie, Constrained LogitsProcessor)
```

## 扩展新 Tokenizer

```python
from saegenrec.modeling.tokenizers.base import ItemTokenizer, register_item_tokenizer

@register_item_tokenizer("my-tokenizer")
class MyTokenizer(ItemTokenizer):
    def train(self, semantic_embeddings_dir, collaborative_embeddings_dir, config):
        ...
    def encode(self, embeddings):
        ...
    # ... 实现其余抽象方法
```

## 扩展新 SFT 任务

```python
from saegenrec.modeling.sft.base import SFTTaskBuilder, register_sft_task

@register_sft_task("my-task")
class MyTaskBuilder(SFTTaskBuilder):
    @property
    def task_type(self):
        return "my-task"
    def build(self, data_dir, sid_map, config):
        ...
```

新增 prompt 模板只需编辑 `configs/templates/sft_prompts.yaml`，无需修改代码。
