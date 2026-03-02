# Data Model: Embedding 模块

**Branch**: `003-embedding-module` | **Date**: 2026-03-02

## Entity Relationship Overview

```
                  Stage 1 产出                          Stage 2 产出
              ┌───────────────┐                   ┌──────────────────┐
              │ item_metadata │                   │ train_sequences  │
              │ item_id_map   │                   │ valid_sequences  │
              └──────┬────────┘                   │ test_sequences   │
                     │                            └────────┬─────────┘
          ┌──────────┴──────────┐                          │
          ▼                     │                          ▼
  SemanticEmbedder              │              CollaborativeEmbedder
  (编码型, 无训练)               │              (训练型, Lightning)
          │                     │                          │
          ▼                     │                          ▼
  item_semantic_embeddings/     │       {split}/item_collaborative_embeddings/
  (Stage 1 同级)                │              (Stage 2 同级)
                                │
                   ┌────────────┘
                   ▼
            下游 ItemTokenizer
            (独立模块, 消费 embedding)
```

## Entities

### 1. SemanticEmbedder（语义 Embedder ABC）

物品语义 embedding 生成器的抽象基类。

**抽象方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `generate` | `(data_dir: Path, output_dir: Path, config: dict) → Dataset` | 从 Stage 1 数据生成语义 embedding |

**注册表**:

| 组件 | 名称 | 说明 |
|------|------|------|
| Registry | `SEMANTIC_EMBEDDER_REGISTRY: dict[str, type[SemanticEmbedder]]` | 全局注册表 |
| Decorator | `@register_semantic_embedder("name")` | 注册装饰器 |
| Factory | `get_semantic_embedder(name: str, **kwargs) → SemanticEmbedder` | 工厂函数 |

**默认实现**: `SentenceTransformerEmbedder` (注册名: `"sentence-transformer"`)

### 2. CollaborativeEmbedder（协同 Embedder ABC）

物品协同 embedding 生成器的抽象基类。

**抽象方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `generate` | `(data_dir: Path, output_dir: Path, config: dict) → Dataset` | 从 Stage 2 划分数据训练模型并提取 embedding |

**注册表**:

| 组件 | 名称 | 说明 |
|------|------|------|
| Registry | `COLLABORATIVE_EMBEDDER_REGISTRY: dict[str, type[CollaborativeEmbedder]]` | 全局注册表 |
| Decorator | `@register_collaborative_embedder("name")` | 注册装饰器 |
| Factory | `get_collaborative_embedder(name: str, **kwargs) → CollaborativeEmbedder` | 工厂函数 |

**默认实现**: `SASRecEmbedder` (注册名: `"sasrec"`)

### 3. SemanticEmbeddingDataset（语义 Embedding 输出）

语义 embedder 的输出 Dataset。

**HuggingFace Dataset Schema (SEMANTIC_EMBEDDING_FEATURES)**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `item_id` | `Value("int32")` | 映射后的物品 ID（来自 item_id_map） |
| `embedding` | `Sequence(Value("float32"))` | 语义 embedding 向量 |

**存储路径**: `data/interim/{dataset}/{category}/item_semantic_embeddings/`

**Validation Rules**:
- `item_id` 唯一
- 覆盖所有通过 K-core 过滤且在 item_metadata 中有记录的物品
- `embedding` 维度由预训练模型决定（如 `all-MiniLM-L6-v2` → 384 维）
- 默认不做 L2 归一化（`normalize=False`）

### 4. CollaborativeEmbeddingDataset（协同 Embedding 输出）

协同 embedder 的输出 Dataset。

**HuggingFace Dataset Schema (COLLABORATIVE_EMBEDDING_FEATURES)**:

| 字段 | HF Feature Type | 说明 |
|------|----------------|------|
| `item_id` | `Value("int32")` | 映射后的物品 ID |
| `embedding` | `Sequence(Value("float32"))` | 协同 embedding 向量 |

**存储路径**: `data/interim/{dataset}/{category}/{split_strategy}/item_collaborative_embeddings/`

**Validation Rules**:
- `item_id` 唯一
- 覆盖 item_id_map 中的所有物品（包括 padding item 0 if used）
- `embedding` 维度由模型配置的 `hidden_size` 决定
- 来自训练后模型 `nn.Embedding` 层的权重

### 5. SemanticEmbeddingConfig（语义 Embedding 配置）

**dataclass 字段**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | 是否在流水线 embed 步骤中运行 |
| `name` | `str` | `"sentence-transformer"` | 注册表中的实现名称 |
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | sentence-transformers 模型名 |
| `text_fields` | `list[str]` | `["title", "brand", "description", "price"]` | 拼接的文本字段 |
| `normalize` | `bool` | `False` | 是否 L2 归一化 |
| `batch_size` | `int` | `256` | 编码批大小 |
| `device` | `str` | `"cpu"` | 推理设备 |

### 6. CollaborativeEmbeddingConfig（协同 Embedding 配置）

**dataclass 字段**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | 是否在流水线 embed 步骤中运行 |
| `name` | `str` | `"sasrec"` | 注册表中的实现名称 |
| `hidden_size` | `int` | `64` | 模型隐藏层 / embedding 维度 |
| `num_layers` | `int` | `2` | Transformer / RNN 层数 |
| `num_heads` | `int` | `1` | 注意力头数（SASRec 专用） |
| `max_seq_len` | `int` | `50` | 训练时最大序列长度 |
| `dropout` | `float` | `0.2` | Dropout 比率 |
| `learning_rate` | `float` | `0.001` | 学习率 |
| `batch_size` | `int` | `256` | 训练批大小 |
| `num_epochs` | `int` | `200` | 训练轮数 |
| `eval_top_k` | `list[int]` | `[10, 20]` | 评估指标的 K 值列表 |
| `device` | `str` | `"auto"` | 训练设备（auto 自动检测 GPU） |
| `seed` | `int` | `42` | 随机种子 |

### 7. SASRec 模型（nn.Module）

序列推荐模型 SASRec 的 PyTorch 实现（参考 RecBole 实现）。

**模型结构**:

| 组件 | 说明 |
|------|------|
| `item_embedding` | `nn.Embedding(num_items + 1, hidden_size, padding_idx=0)` — 物品 embedding 层，padding 0 |
| `position_embedding` | `nn.Embedding(max_seq_len, hidden_size)` — 位置 embedding |
| `attention_layers` | `nn.ModuleList` of `SASRecBlock`（自注意力 + FFN + LayerNorm + Dropout） |
| `output_layer` | 无额外参数，直接用 item_embedding 权重做预测（tied weights） |

**输入/输出**:

| 阶段 | 输入 | 输出 |
|------|------|------|
| 训练 | `(seq: [B, L], pos: [B, L], neg: [B, L])` | BPR loss (scalar) |
| 评估 | `(seq: [B, L], candidates: [B, C])` | scores `[B, C]` |
| 提取 | 无 | `item_embedding.weight.data[1:]` → `[num_items, hidden_size]` |

**Embedding 提取方式**: 训练完成后，直接取 `model.item_embedding.weight.data[1:]`（跳过 padding index 0），得到 `[num_items, hidden_size]` 的 embedding 矩阵。

## State Transitions

### Embedding 生成数据流

```
Stage 1 完成                   Stage 2 完成 (split)
    │                              │
    ├── item_metadata              ├── train_sequences
    ├── item_id_map                ├── valid_sequences
    │                              └── test_sequences
    │                                      │
    ▼                                      ▼
[SemanticEmbedder.generate()]    [CollaborativeEmbedder.generate()]
    │                                      │
    │  输入: item_metadata,                │  输入: train/valid/test_sequences,
    │        item_id_map                   │        item_id_map
    │                                      │
    │  处理: 文本拼接 → 模型编码           │  处理: 序列训练 → 权重提取
    │                                      │
    ▼                                      ▼
item_semantic_embeddings/         {split}/item_collaborative_embeddings/
  (HuggingFace Dataset)             (HuggingFace Dataset)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
            统计信息输出
            (物品数, 维度, 耗时)
```

### Skip / Force 状态

```
检查输出目录
    │
    ├── 目录不存在 → 执行生成
    │
    ├── 目录已存在 + --force → 清除旧数据 → 执行生成
    │
    └── 目录已存在 + 无 --force → 跳过 (log info)
```

### 配置结构更新

```
PipelineConfig (dataclass)
├── dataset: DatasetConfig               (不变)
├── processing: ProcessingConfig         (不变)
├── tokenizer: TokenizerConfig           (不变)
├── embedding: EmbeddingConfig           (保留, deprecated)
├── semantic_embedding: SemanticEmbeddingConfig    (新增)
├── collaborative_embedding: CollaborativeEmbeddingConfig  (新增)
└── output: OutputConfig                 (不变)
```
