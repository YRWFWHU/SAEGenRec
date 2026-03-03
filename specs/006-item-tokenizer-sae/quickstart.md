# Quickstart: SAE Item Tokenizer

## 前置条件

1. 已完成数据 pipeline 处理（`make data-filter && make data-split`）
2. 已生成语义嵌入（`make data-embed-semantic`）

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

新增依赖（`pyproject.toml` 已更新）：
- `safetensors`（模型权重保存格式）

### 2. 修改配置文件

编辑 `configs/default.yaml`，将 `item_tokenizer` 段修改为：

```yaml
item_tokenizer:
  enabled: true
  name: "sae"
  num_codebooks: 8       # top_k: 每个物品的 SID code 数量
  codebook_size: 8192    # d_sae: 概念词表大小
  params:
    learning_rate: 0.001
    epochs: 50
    batch_size: 256
    device: "cuda"
    l0_coefficient: 0.001
```

### 3. 运行 Tokenize

```bash
make data-tokenize
# 等价于: python -m saegenrec.dataset tokenize configs/default.yaml
```

训练过程中会输出日志：
```
Training JumpReLU SAE: d_in=384, d_sae=8192, top_k=8
Epoch  1/50 | MSE: 0.4521 | L0: 0.0032 | Mean active: 45.2
Epoch 10/50 | MSE: 0.1234 | L0: 0.0015 | Mean active: 12.3
Epoch 50/50 | MSE: 0.0456 | L0: 0.0008 | Mean active: 8.7
Training complete: {'final_mse_loss': 0.0456, 'mean_l0': 8.7, 'vocab_utilization': 0.45}
Codebook utilization: 3686/8192 (45.0%)
Saved SID map (12345 items) to data/processed/amazon2015/Beauty
```

### 4. 验证输出

```bash
# 检查生成的 SID map
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed/amazon2015/Beauty/item_sid_map')
print(f'Items: {len(ds)}')
print(f'Sample codes: {ds[0][\"codes\"]}')
print(f'Sample SID: {ds[0][\"sid_tokens\"]}')
print(f'Codes per item: {len(ds[0][\"codes\"])}')
"
```

预期输出：
```
Items: 12345
Sample codes: [3071, 1547, 941, 7587, 7639, 3383, 6576, 5411]
Sample SID: <|sid_begin|><s_a_3071><s_b_1547><s_c_941><s_d_7587><s_e_7639><s_f_3383><s_g_6576><s_h_5411><|sid_end|>
Codes per item: 8
```

### 5. 超参数调优

常用调参方向：

| 参数 | 效果 | 建议范围 |
|------|------|----------|
| `codebook_size` (d_sae) | 增大 → 更细粒度的概念，更少碰撞 | 2048 ~ 16384 |
| `num_codebooks` (top_k) | 增大 → 更长的 SID，更多信息 | 4 ~ 16 |
| `l0_coefficient` | 增大 → 更稀疏，减少 → 更密集 | 1e-4 ~ 1e-2 |
| `epochs` | 增大 → 更低的重构损失 | 20 ~ 100 |

### 6. 与 RQ-VAE 对比实验

切换回 RQ-VAE 只需修改 `name`：

```yaml
item_tokenizer:
  name: "rqvae"    # 切换为 RQ-VAE
  # name: "sae"    # 切换为 SAE
```

## 目录结构

Tokenize 完成后的输出：

```
data/processed/amazon2015/Beauty/
├── item_sid_map/          # HF Dataset: item_id, codes, sid_tokens
└── tokenizer_model/       # SAE 模型权重
    ├── sae_weights.safetensors   # 模型权重
    └── hparams.json              # 超参数配置
```
