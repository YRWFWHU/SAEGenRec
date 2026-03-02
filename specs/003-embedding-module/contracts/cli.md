# CLI Contracts: Embedding 模块

**Branch**: `003-embedding-module` | **Date**: 2026-03-02

## 1. 流水线集成 (embed 步骤)

**命令**: `python -m saegenrec.dataset process <config> --step embed`

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | Path (必需) | YAML 配置文件路径 |
| `--step embed` | str | 指定运行 embed 步骤 |
| `--force` | flag | 强制覆盖已存在的 embedding 输出 |

**行为**:
- 根据配置中 `semantic_embedding.enabled` 和 `collaborative_embedding.enabled` 决定运行哪些 embedder
- 语义 embedding 依赖 Stage 1 数据（item_metadata, item_id_map）
- 协同 embedding 依赖 Stage 2 数据（train/valid/test_sequences）
- 两种 embedding 按顺序生成（先语义后协同）
- 各自独立保存到对应目录

**输出** (stdout/log):
```
=== Step: Embed (Semantic) ===
Encoding 3,857 items with all-MiniLM-L6-v2 on cpu
Saved 3,857 semantic embeddings to data/interim/amazon2015/Baby/item_semantic_embeddings/
Stats: items=3857, dim=384, elapsed=45.2s

=== Step: Embed (Collaborative) ===
Training SASRec (hidden=64, layers=2, epochs=200)
Epoch 1/200: HR@10=0.0512, NDCG@10=0.0234
...
Epoch 200/200: HR@10=0.3821, NDCG@10=0.2156
Saved 3,857 collaborative embeddings to data/interim/amazon2015/Baby/loo/item_collaborative_embeddings/
Stats: items=3857, dim=64, elapsed=312.5s
```

**退出码**: 0 成功, 1 失败

---

## 2. 独立语义 Embedding 命令

**命令**: `python -m saegenrec.dataset embed-semantic <config> [options]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | Path (必需) | — | YAML 配置文件路径 |
| `--force` | flag | False | 强制覆盖已存在的输出 |
| `--model-name` | str | None | 覆盖 semantic_embedding.model_name |
| `--device` | str | None | 覆盖 semantic_embedding.device |

**前置条件**: Stage 1 完成（item_metadata, item_id_map 存在）

**输出路径**: `data/interim/{dataset}/{category}/item_semantic_embeddings/`

**错误处理**:
- Stage 1 数据缺失 → 错误信息 + exit 1
- 输出已存在且无 --force → info 提示跳过 + exit 0

---

## 3. 独立协同 Embedding 命令

**命令**: `python -m saegenrec.dataset embed-collaborative <config> [options]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | Path (必需) | — | YAML 配置文件路径 |
| `--force` | flag | False | 强制覆盖已存在的输出 |
| `--device` | str | None | 覆盖 collaborative_embedding.device |
| `--num-epochs` | int | None | 覆盖 collaborative_embedding.num_epochs |

**前置条件**: Stage 2 完成（train/valid/test_sequences 存在）

**输出路径**: `data/interim/{dataset}/{category}/{split_strategy}/item_collaborative_embeddings/`

**错误处理**:
- Stage 2 数据缺失 → 错误信息 + exit 1
- GPU 不可用但配置指定 GPU → warning + 回退 CPU
- 输出已存在且无 --force → info 提示跳过 + exit 0

---

## 4. YAML 配置契约

```yaml
# 新增配置节（与现有 dataset/processing/tokenizer/output 并列）

semantic_embedding:
  enabled: false                              # 流水线 embed 步骤是否运行
  name: "sentence-transformer"                # 注册表中的实现名称
  model_name: "all-MiniLM-L6-v2"             # 预训练模型
  text_fields:                                # 拼接的元数据文本字段
    - "title"
    - "brand"
    - "description"
    - "price"
  normalize: false                            # L2 归一化 (默认关闭, 对齐 Align3GR)
  batch_size: 256                             # 编码批大小
  device: "cpu"                               # 推理设备

collaborative_embedding:
  enabled: false                              # 流水线 embed 步骤是否运行
  name: "sasrec"                              # 注册表中的实现名称
  hidden_size: 64                             # 模型隐藏层 / embedding 维度
  num_layers: 2                               # Transformer 层数
  num_heads: 1                                # 注意力头数
  max_seq_len: 50                             # 训练最大序列长度
  dropout: 0.2                                # Dropout
  learning_rate: 0.001                        # 学习率
  batch_size: 256                             # 训练批大小
  num_epochs: 200                             # 训练轮数
  eval_top_k: [10, 20]                        # 评估指标 K 值
  device: "auto"                              # 训练设备 (auto 自动检测)
  seed: 42                                    # 随机种子
```

**向后兼容**: 旧 `embedding` 配置节保留但标记为 deprecated。如果 `semantic_embedding` 未配置但 `embedding.enabled=true`，系统自动降级到旧行为并输出 deprecation warning。
