# Quickstart: Embedding 模块

**Branch**: `003-embedding-module` | **Date**: 2026-03-02

## 前提条件

- Stage 1 数据已生成: `data/interim/{dataset}/{category}/` 含 `item_metadata/`, `item_id_map/`
- (协同 embedding) Stage 2 split 已完成: `data/interim/{dataset}/{category}/{split_strategy}/` 含 `train_sequences/`, `valid_sequences/`, `test_sequences/`

## 1. 生成语义 Embedding

```bash
# 方式一：独立 CLI 命令
python -m saegenrec.dataset embed-semantic configs/default.yaml

# 方式二：流水线集成（需配置 semantic_embedding.enabled: true）
python -m saegenrec.dataset process configs/default.yaml --step embed
```

输出: `data/interim/amazon2015/Baby/item_semantic_embeddings/`

## 2. 生成协同 Embedding

```bash
# 方式一：独立 CLI 命令
python -m saegenrec.dataset embed-collaborative configs/default.yaml

# 方式二：流水线集成（需配置 collaborative_embedding.enabled: true）
python -m saegenrec.dataset process configs/default.yaml --step embed
```

输出: `data/interim/amazon2015/Baby/loo/item_collaborative_embeddings/`

## 3. 配置示例

在 `configs/default.yaml` 中添加:

```yaml
semantic_embedding:
  enabled: true
  name: "sentence-transformer"
  model_name: "all-MiniLM-L6-v2"
  text_fields: ["title", "brand", "description", "price"]
  normalize: false
  device: "cpu"

collaborative_embedding:
  enabled: true
  name: "sasrec"
  hidden_size: 64
  num_epochs: 200
  device: "auto"
```

## 4. 强制重新生成

```bash
python -m saegenrec.dataset embed-semantic configs/default.yaml --force
python -m saegenrec.dataset embed-collaborative configs/default.yaml --force
```

## 5. 加载生成的 Embedding

```python
from datasets import load_from_disk

semantic = load_from_disk("data/interim/amazon2015/Baby/item_semantic_embeddings")
print(f"Items: {len(semantic)}, Dim: {len(semantic[0]['embedding'])}")

collaborative = load_from_disk("data/interim/amazon2015/Baby/loo/item_collaborative_embeddings")
print(f"Items: {len(collaborative)}, Dim: {len(collaborative[0]['embedding'])}")
```

## 6. 自定义 Embedder 扩展

```python
from saegenrec.data.embeddings.semantic.base import SemanticEmbedder, register_semantic_embedder

@register_semantic_embedder("my-custom-embedder")
class MyEmbedder(SemanticEmbedder):
    def generate(self, data_dir, output_dir, config):
        # 自定义 embedding 生成逻辑
        ...
```

配置中指定: `semantic_embedding.name: "my-custom-embedder"`
