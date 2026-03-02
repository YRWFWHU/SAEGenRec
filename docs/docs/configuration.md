# 配置参考

管道的所有行为通过 YAML 配置文件控制。配置文件对应 `PipelineConfig` dataclass 层级结构。

## 完整配置示例

```yaml
dataset:
  name: "amazon2015"          # 数据集名称（amazon2015 | amazon2023）
  category: "Beauty"          # 品类名（与 data/raw/ 下的目录名对应）
  raw_dir: "data/raw"         # 原始数据根目录

processing:
  kcore_threshold: 5          # K-core 过滤阈值
  split_strategy: "loo"       # 划分策略（loo | to）
  split_ratio: [0.8, 0.1, 0.1]  # TO 策略的划分比例（仅 to 模式生效）
  max_seq_len: 20             # 训练样本最大历史长度
  num_negatives: 99           # 每条样本的负采样数量
  seed: 42                    # 负采样随机种子（null 为不固定）

tokenizer:
  name: "passthrough"         # tokenizer 注册名
  params: {}                  # tokenizer 初始化参数

semantic_embedding:
  enabled: true               # 是否启用语义嵌入
  name: "sentence-transformer" # 语义 embedder 注册名
  model_name: "all-MiniLM-L6-v2"  # sentence-transformers 模型名
  text_fields:                # 拼接为嵌入输入的元数据字段
    - "title"
    - "brand"
    - "description"
    - "price"
  normalize: false            # L2 归一化（默认关闭）
  batch_size: 256             # 嵌入计算批次大小
  device: "cpu"               # 计算设备（cpu | cuda）

collaborative_embedding:
  enabled: true               # 是否启用协同嵌入
  name: "sasrec"              # 协同 embedder 注册名
  loss_type: "CE"             # 损失函数（CE | BPR）
  hidden_size: 64             # Transformer 隐藏层维度
  num_layers: 2               # Transformer 层数
  num_heads: 2                # 多头注意力头数
  max_seq_len: 20             # 模型输入序列最大长度
  dropout: 0.5                # Dropout 比率
  learning_rate: 0.001        # 学习率
  batch_size: 256             # 训练批次大小
  num_epochs: 200             # 训练轮数
  eval_top_k: [10, 20]        # 评估 Top-K 列表
  device: "auto"              # 计算设备（auto | cpu | cuda）
  seed: 42                    # 随机种子

item_tokenizer:
  enabled: false                # 是否启用物品 Tokenization
  name: "rqvae"                 # tokenizer 注册名（rqvae | rqkmeans）
  num_codebooks: 4              # 码本层数
  codebook_size: 256            # 每层码本大小
  collision_strategy: "append_level"  # 碰撞消解策略
  params:
    hidden_dim: 256             # RQ-VAE 隐藏层维度
    latent_dim: 64              # RQ-VAE 潜空间维度
    lr: 0.001                   # 学习率
    epochs: 100                 # 训练轮数
    batch_size: 512             # 训练批次大小
    device: "cuda"              # 计算设备

sft_builder:
  enabled: false                # 是否启用 SFT 数据构建
  tasks:                        # 启用的 SFT 任务列表
    - "seqrec"
    - "item2index"
    - "index2item"
  template_file: "configs/templates/sft_prompts.yaml"  # Prompt 模板文件
  max_history_len: 20           # SeqRec 任务最大历史长度
  seed: 42                      # 模板随机采样种子

output:
  interim_dir: "data/interim"       # 中间数据输出目录
  processed_dir: "data/processed"   # 最终数据输出目录
```

## 各节详解

### `dataset` — 数据源配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"amazon2015"` | 数据集名称，决定使用哪个加载器。可选值: `amazon2015`, `amazon2023` |
| `category` | string | `"Baby"` | 品类名，必须与 `data/raw/{DatasetDir}/{category}/` 的目录名一致 |
| `raw_dir` | string | `"data/raw"` | 原始数据根目录的相对路径 |

**数据路径推导规则**: `{raw_dir}/{name→映射}/{category}/`

| `name` | 映射后目录名 |
|--------|-------------|
| `amazon2015` | `Amazon2015` |
| `amazon2023` | `Amazon2023` |

### `processing` — 处理参数

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `kcore_threshold` | int | `5` | ≥ 1 | K-core 过滤的最低交互次数 |
| `split_strategy` | string | `"loo"` | `loo` 或 `to` | 数据划分策略 |
| `split_ratio` | list[float] | `[0.8, 0.1, 0.1]` | 总和为 1.0（仅 `to` 模式校验） | train:valid:test 比例 |
| `max_seq_len` | int | `20` | ≥ 1 | 滑动窗口生成训练样本时的最大历史长度 |
| `num_negatives` | int | `99` | ≥ 1 | 每条样本的负采样数量 |
| `seed` | int \| None | `42` | — | 负采样随机种子，`null` 表示不固定 |

**划分策略对比**:

| | LOO | TO |
|---|---|---|
| 原理 | 每用户最后一条 → test，倒数第二条 → valid | 按全局时间戳比例划分 |
| 适用场景 | 推荐系统标准评测 | 时序敏感场景 |
| 用户最少交互数 | 3（少于 3 条的用户被排除） | 无限制 |
| `split_ratio` | 不使用 | 必须提供 |

### `tokenizer` — 商品 tokenizer

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"passthrough"` | tokenizer 注册名。`passthrough` 直接返回商品 ID 作为单 token |
| `params` | dict | `{}` | 传递给 tokenizer 构造函数的额外参数 |

### `semantic_embedding` — 语义嵌入

使用预训练语言模型对商品元数据文本字段提取语义 embedding。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用语义嵌入 |
| `name` | string | `"sentence-transformer"` | 语义 embedder 注册名 |
| `model_name` | string | `"all-MiniLM-L6-v2"` | HuggingFace 上的 sentence-transformers 模型名 |
| `text_fields` | list[string] | `["title", "brand", "description", "price"]` | 拼接为模型输入文本的元数据字段。`price` 为数值时自动转为文本 |
| `normalize` | bool | `false` | 是否对输出 embedding 做 L2 归一化 |
| `batch_size` | int | `256` | 每批编码的商品数 |
| `device` | string | `"cpu"` | 计算设备。GPU 加速使用 `"cuda"` |

**常用模型**:

| 模型 | 维度 | 说明 |
|------|------|------|
| `all-MiniLM-L6-v2` | 384 | 轻量级，适合快速验证 |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 高质量中文 + 英文嵌入，需 GPU |

### `collaborative_embedding` — 协同嵌入

通过训练序列推荐模型，从学习到的 `nn.Embedding` 权重中提取协同过滤 embedding。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用协同嵌入 |
| `name` | string | `"sasrec"` | 协同 embedder 注册名 |
| `loss_type` | string | `"CE"` | 损失函数类型。`"CE"` = CrossEntropy，`"BPR"` = BPR |
| `hidden_size` | int | `64` | Transformer 隐藏层维度（即输出 embedding 维度） |
| `num_layers` | int | `2` | Self-Attention 层数 |
| `num_heads` | int | `2` | 多头注意力头数 |
| `max_seq_len` | int | `50` | 模型输入序列最大长度 |
| `dropout` | float | `0.5` | Dropout 比率 |
| `learning_rate` | float | `0.001` | Adam 学习率 |
| `batch_size` | int | `256` | 训练批次大小 |
| `num_epochs` | int | `200` | 训练轮数 |
| `eval_top_k` | list[int] | `[10, 20]` | 评估指标的 Top-K 值列表 |
| `device` | string | `"auto"` | 计算设备。`"auto"` 自动选择可用 GPU |
| `seed` | int | `42` | 随机种子 |

**注意**: `max_seq_len` 与 `processing.max_seq_len` 独立。前者控制协同模型训练时截断序列的长度，后者控制滑动窗口增强时生成训练样本的历史长度。协同 embedder 直接消费 Stage 2 的 split 数据（`train_sequences/` 等），不受 `processing.max_seq_len` 影响。

**训练过程**:

- 每个 epoch 输出 `train_loss`
- 在验证集和测试集上输出 `Hit Rate@K` 和 `NDCG@K` 指标
- 使用梯度裁剪（`gradient_clip_val=5.0`）提高训练稳定性

### `item_tokenizer` — 物品 Tokenization

将语义 embedding 映射为层次化语义 ID（SID）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用物品 Tokenization |
| `name` | string | `"rqvae"` | tokenizer 注册名。可选值: `rqvae`, `rqkmeans` |
| `num_codebooks` | int | `4` | 码本层数（即 SID 基础长度） |
| `codebook_size` | int | `256` | 每层码本中的向量数量 |
| `collision_strategy` | string | `"append_level"` | 碰撞消解策略。`append_level` 追加层级去重，`sinkhorn` Sinkhorn 重分配 |
| `sid_token_format` | string | `"<s_{level}_{code}>"` | SID token 字符串格式 |
| `sid_begin_token` | string | `"<\|sid_begin\|>"` | SID 序列起始界定符 |
| `sid_end_token` | string | `"<\|sid_end\|>"` | SID 序列结束界定符 |
| `params` | dict | `{}` | 传递给具体 tokenizer 的额外参数 |

**`params` 字段（RQ-VAE）**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | int | `256` | MLP 编码器/解码器隐藏层维度 |
| `latent_dim` | int | `64` | 潜空间维度 |
| `lr` | float | `0.001` | 学习率 |
| `epochs` | int | `100` | 训练轮数 |
| `batch_size` | int | `512` | 训练批次大小 |
| `device` | string | `"cuda"` | 计算设备 |
| `ema_decay` | float | `0.99` | EMA 码本更新的衰减率 |
| `dead_code_threshold` | int | `2` | 死码替换的使用次数阈值 |

**内置 Tokenizer 对比**:

| | RQ-VAE | RQ-KMeans |
|---|---|---|
| 训练方式 | MLP 编码器 + 反向传播 | 逐层 KMeans 聚类 |
| 需要 GPU | 是（推荐） | 否 |
| 码本质量 | 高（可学习编码器） | 中（固定投影） |
| 训练速度 | 较慢 | 较快 |
| 均衡约束 | EMA + 死码替换 | 均衡 KMeans（FAISS） |

### `sft_builder` — SFT 数据构建

将推荐数据 + SID 映射转换为 LLM 指令微调数据。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用 SFT 数据构建 |
| `tasks` | list[string] | `["seqrec", "item2index", "index2item"]` | 启用的 SFT 任务类型列表 |
| `task_weights` | dict[string, float] | `{}` | 任务采样权重（0~1），空表示不降采样 |
| `template_file` | string | `"configs/templates/sft_prompts.yaml"` | Prompt 模板文件路径 |
| `max_history_len` | int | `20` | SeqRec 任务中用户历史序列的最大长度 |
| `seed` | int | `42` | 模板随机采样和数据 shuffle 的种子 |

**内置 SFT 任务**:

| 任务 | 注册名 | 数据来源 | 说明 |
|------|--------|---------|------|
| 序列推荐 | `seqrec` | `train/`（滑动窗口增强数据） | 历史 SID 序列 → 目标 SID |
| 物品→SID | `item2index` | `item_metadata/` + `item_sid_map/` | 物品标题 → SID |
| SID→物品 | `index2item` | `item_metadata/` + `item_sid_map/` | SID → 物品标题 |

### `output` — 输出路径

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interim_dir` | string | `"data/interim"` | 中间数据目录。实际路径: `{interim_dir}/{dataset_name}/{category}/` |
| `processed_dir` | string | `"data/processed"` | 最终数据目录。实际路径: `{processed_dir}/{dataset_name}/{category}/{split_strategy}/` |

## 现有示例配置

| 文件 | 数据集 | 品类 |
|------|--------|------|
| `configs/default.yaml` | amazon2015 / Baby | 通用默认 |
| `configs/examples/amazon2015_baby.yaml` | amazon2015 | Baby |
| `configs/examples/amazon2015_beauty.yaml` | amazon2015 | Beauty |
| `configs/examples/amazon2023_beauty.yaml` | amazon2023 | All_Beauty |
| `configs/examples/amazon2023_fashion.yaml` | amazon2023 | Amazon_Fashion |

## 配置兼容性

旧版 `embedding` 配置节仍可使用，但已被标记为 **deprecated**。如果配置中存在 `embedding.enabled: true` 且未提供 `semantic_embedding` 节，系统将自动迁移设置并发出 `DeprecationWarning`。建议迁移到 `semantic_embedding`。

## 创建自定义配置

只需指定与默认值不同的字段即可：

```yaml
dataset:
  name: "amazon2023"
  category: "Books"

processing:
  kcore_threshold: 10
  max_seq_len: 50

semantic_embedding:
  enabled: true
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  device: "cuda"

collaborative_embedding:
  enabled: true
  loss_type: "BPR"
  num_epochs: 100
```

未指定的字段自动使用默认值。
