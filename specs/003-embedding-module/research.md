# Research: Embedding 模块

**Branch**: `003-embedding-module` | **Date**: 2026-03-02

## R1: 协同 Embedder 训练框架选择

### Decision: PyTorch Lightning

### Rationale

对比三种候选方案：

| 维度 | 自定义训练循环 | HuggingFace Trainer | PyTorch Lightning |
|------|---------------|--------------------|--------------------|
| **Constitution 合规** | ✅ "兼容的训练循环" | ✅ 首选方案 | ⚠️ 需豁免（功能重叠） |
| **序列推荐适配** | ✅ 完全灵活 | ❌ 预设 NLP 范式，排序指标和序列数据格式需大量适配 | ✅ 模型/数据/训练解耦，适合自定义评估 |
| **训练基础设施** | ❌ 手写设备管理/日志/checkpoint | ✅ 内置 | ✅ 内置 |
| **参考实现先例** | Align3GR, MiniOneRec 均使用 | 无参考实现使用 | 序列推荐社区广泛使用 |
| **样板代码量** | 高（~200 行训练循环样板） | 低 | 低 |
| **新依赖引入** | 无 | 无 | pytorch-lightning |

选择 Lightning 的核心原因：
1. HuggingFace Trainer 的数据流预设（tokenized text batches）与序列推荐模型的输入格式（用户交互 ID 序列）不匹配，强行适配引入不必要的复杂度
2. 自定义训练循环虽然灵活但需手写大量样板代码（设备管理、梯度累积、日志、早停、checkpoint），违反简洁性原则
3. Lightning 的 `LightningModule` + `Trainer` 将模型逻辑与训练基础设施解耦，与 embedding 模块的 ABC 设计哲学一致
4. 用户在功能描述中明确指定

### Alternatives Considered

- **自定义训练循环**: 参考实现均采用，但这些项目是一次性实验脚本而非可复用框架。对于需要支持多种模型（SASRec、GRU4Rec 等）的注册表系统，标准化训练基础设施的价值更高。
- **HuggingFace Trainer**: 作为 constitution 首选方案，对 NLP 分类/生成任务最优；但 `compute_metrics` 回调预设 prediction-label 范式，序列推荐的 full-ranking 评估（对全部物品排序后计算 HR/NDCG）需要绕过其预设逻辑。

## R2: 语义 Embedding 模型选择

### Decision: sentence-transformers (默认 all-MiniLM-L6-v2)，通过配置可替换

### Rationale

- 当前代码库已使用 `sentence-transformers`，无需引入新依赖
- `all-MiniLM-L6-v2` (384 维) 在速度和质量间取得良好平衡
- Align3GR 使用 `sentence-t5-base` (768 维)，MiniOneRec 使用 Qwen
- 通过 SemanticEmbedder 注册表支持未来替换为其他模型

### Alternatives Considered

- **T5-based (sentence-t5-base)**: Align3GR 使用，质量更高但速度慢 3-5x
- **LLaMA / Qwen**: MiniOneRec 使用，需要 GPU 推理，不适合作为默认选项

## R3: L2 归一化默认关闭

### Decision: 默认关闭，通过 `normalize: bool = False` 配置

### Rationale

- Align3GR 论文中语义 embedding 不做 L2 归一化，直接送入 RQ-VAE 进行量化
- 现有 `text.py` 代码中 `normalize_embeddings=True`（L2 归一化），这是旧行为
- 新实现默认对齐 Align3GR 方案：关闭归一化
- 保留配置开关以兼容需要归一化的场景

### Alternatives Considered

- **默认开启**: 与旧代码一致，但与 Align3GR 参考实现不一致

## R4: SASRec 模型实现方案

### Decision: 自实现 SASRec 作为 nn.Module，参考 RecBole 实现

### Rationale

- RecBole 提供了经过广泛验证的 SASRec 实现，模型架构清晰（MultiHeadAttention + PositionwiseFeedForward + LayerNorm），可作为可靠的参考蓝本
- 不直接依赖 RecBole 框架（避免引入其数据格式和训练流程的耦合），而是参考其模型架构代码自实现为独立的 nn.Module
- 自实现的 nn.Module 可直接配合 PyTorch Lightning 使用
- 从 `nn.Embedding` 层提取权重的逻辑简单直接：`model.item_embedding.weight.data`
- RecBole 的 SASRec 实现包含完整的 causal mask、position embedding 和 BPR loss 逻辑，参考价值高于其他简化实现

### Alternatives Considered

- **直接依赖 RecBole**: 框架耦合过重，引入不必要的数据格式约束
- **参考 MiniOneRec**: 实现较为简化，缺少部分细节（如 causal attention mask 处理）
- **HuggingFace PreTrainedModel**: Constitution 推荐，但 SASRec 不是 NLP 模型，强行继承 `PreTrainedModel` 增加不必要的复杂度

## R5: 评估指标实现

### Decision: 自实现 Hit Rate@K 和 NDCG@K，参考 MiniOneRec 的 GPU 加速评估

### Rationale

所有参考实现均自实现这两个指标，逻辑简单：
- **Hit Rate@K**: `(target_rank < K).float().mean()`
- **NDCG@K**: `(1 / log2(target_rank + 2)).mean()` (仅对 rank < K 的样本)
- MiniOneRec 使用 CUDA tensors 加速 full-ranking 评估
- 不引入 torchmetrics 等额外依赖

### Alternatives Considered

- **torchmetrics**: 通用指标库，但推荐系统的 full-ranking 评估有特殊需求（需对全部物品排序），torchmetrics 的 retrieval metrics 预设不匹配
- **RecBole evaluator**: 过重，需引入整个框架

## R6: 配置结构设计

### Decision: 扩展现有 PipelineConfig，新增 SemanticEmbeddingConfig 和 CollaborativeEmbeddingConfig

### Rationale

```yaml
semantic_embedding:
  enabled: true
  name: "sentence-transformer"      # 注册表名
  model_name: "all-MiniLM-L6-v2"
  text_fields: ["title", "brand", "description", "price"]
  normalize: false
  batch_size: 256
  device: "cpu"

collaborative_embedding:
  enabled: false
  name: "sasrec"                     # 注册表名
  hidden_size: 64
  num_layers: 2
  num_heads: 1
  max_seq_len: 50
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 256
  num_epochs: 200
  eval_top_k: [10, 20]
  device: "auto"
  seed: 42
```

- 保留现有 `embedding` 配置节用于向后兼容（deprecated），新增 `semantic_embedding` 和 `collaborative_embedding`
- 通用参数（name, device）放在顶层，模型特有参数通过 `params: {}` 传递或内联
- `enabled` 控制是否在流水线 embed 步骤中自动运行

### Alternatives Considered

- **单一 embedding 配置节**: 两种 embedder 差异过大（一个编码、一个训练），合并反而增加复杂度
