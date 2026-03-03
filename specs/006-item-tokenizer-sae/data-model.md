# Data Model: SAE Item Tokenizer

**Feature Branch**: `006-item-tokenizer-sae`  
**Date**: 2026-03-01

## Entities

### 1. JumpReLU SAE 模型参数

核心 SAE 模型，包含编码器/解码器权重和 JumpReLU 阈值。

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `W_enc` | `(d_in, d_sae)` | `nn.Parameter` | 编码器权重矩阵 |
| `b_enc` | `(d_sae,)` | `nn.Parameter` | 编码器偏置 |
| `W_dec` | `(d_sae, d_in)` | `nn.Parameter` | 解码器权重矩阵 |
| `b_dec` | `(d_in,)` | `nn.Parameter` | 解码器偏置（输入预处理用） |
| `log_threshold` | `(d_sae,)` | `nn.Parameter` | JumpReLU 阈值的对数（训练参数，`threshold = exp(log_threshold)`） |

### 2. SAE 超参数配置

通过配置文件传递，保存在模型旁供复现。

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `d_in` | `int` | 自动推断 | 输入嵌入维度（从数据集自动获取） |
| `d_sae` | `int` | `8192` | SAE 隐藏维度 / 概念词表大小 |
| `top_k` | `int` | `8` | 每个物品选取的激活特征数 |
| `learning_rate` | `float` | `1e-3` | Adam 优化器学习率 |
| `epochs` | `int` | `50` | 训练轮数 |
| `batch_size` | `int` | `256` | 训练 batch 大小 |
| `l0_coefficient` | `float` | `1e-3` | L0 稀疏性损失系数 |
| `jumprelu_bandwidth` | `float` | `0.05` | JumpReLU STE 梯度的 bandwidth 参数 |
| `jumprelu_init_threshold` | `float` | `0.01` | JumpReLU 阈值初始值 |
| `device` | `str` | `"cpu"` | 训练设备 |
| `seed` | `int` | `42` | 随机种子 |

### 3. SAETokenizer 属性

`ItemTokenizer` 实现的运行时属性。

| Property | Type | Maps to | Description |
|----------|------|---------|-------------|
| `num_codebooks` | `int` | `top_k` | SID 中的 code 数量 |
| `codebook_size` | `int` | `d_sae` | 概念词表大小（每个 code 的取值范围） |

### 4. 训练输出指标

`train()` 方法返回的指标字典。

| Field | Type | Description |
|-------|------|-------------|
| `final_mse_loss` | `float` | 最终 epoch 的平均 MSE 重构损失 |
| `final_l0_loss` | `float` | 最终 epoch 的平均 L0 稀疏性损失 |
| `final_total_loss` | `float` | 最终 epoch 的总损失 |
| `mean_l0` | `float` | 最终 epoch 每个样本的平均激活特征数 |
| `vocab_utilization` | `float` | 概念词表利用率（被至少一个物品使用的概念数 / d_sae） |
| `num_dead_features` | `int` | 训练结束时从未被激活的特征数 |

### 5. 输入数据（已有，来自上游 Embedding 模块）

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | `int` | 物品内部 ID |
| `embedding` | `list[float]` | 物品文本嵌入向量（维度 = `d_in`） |

### 6. 输出 SID Map（已有 schema: `SID_MAP_FEATURES`）

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | `int32` | 物品内部 ID |
| `codes` | `Sequence(int32)` | SAE top_k 特征索引（按激活值降序排列） |
| `sid_tokens` | `string` | 格式化 SID 字符串（如 `<|sid_begin|><s_a_3071><s_b_1547>...<|sid_end|>`） |

## Relationships

```
YAML Config (item_tokenizer section)
  │
  ├── name: "sae"
  ├── num_codebooks: 8     → maps to top_k
  ├── codebook_size: 8192  → maps to d_sae
  └── params: {lr, epochs, batch_size, l0_coefficient, ...}
         │
         ▼
SAETokenizer (ItemTokenizer implementation)
  ├── owns → JumpReLU SAE Model (W_enc, W_dec, b_enc, b_dec, log_threshold)
  ├── reads → Semantic Embeddings (HF Dataset: item_id, embedding)
  ├── produces → SID Map (HF Dataset: item_id, codes, sid_tokens)
  └── saves/loads → Model artifacts (safetensors + hparams.json)
```

## State Transitions

```
SAETokenizer Lifecycle:
  INIT → TRAINING → TRAINED → ENCODING → SAVED
                                  ↑
  INIT → LOADING ─────────────────┘

  INIT: 构造函数接收 d_sae, top_k 等参数
  TRAINING: train() — 从嵌入数据训练 JumpReLU SAE
  TRAINED: 模型就绪，可执行 encode()
  ENCODING: encode() — 前向传播 + top_k 选取
  SAVED: save() — 权重和配置持久化到磁盘
  LOADING: load() — 从磁盘恢复模型状态 → TRAINED
```

## Validation Rules

- `d_sae` MUST > `top_k`（概念词表大小必须大于选取数量）
- `d_sae` MUST > 0 且为正整数
- `top_k` MUST > 0 且为正整数
- `learning_rate` MUST > 0
- `epochs` MUST > 0
- `batch_size` MUST > 0
- `l0_coefficient` MUST >= 0
- `jumprelu_bandwidth` MUST > 0
- `jumprelu_init_threshold` MUST > 0
- 输入嵌入维度 `d_in` 在 `train()` 时自动推断，`encode()` 时验证一致性
- `encode()` 调用前 MUST 确保模型已训练（`train()`）或已加载（`load()`）
