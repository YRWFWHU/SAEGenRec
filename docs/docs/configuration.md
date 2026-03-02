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

embedding:
  enabled: false              # 是否启用文本嵌入步骤
  model_name: "all-MiniLM-L6-v2"  # sentence-transformers 模型名
  text_fields:                # 拼接为嵌入输入的元数据字段
    - "title"
    - "brand"
    - "categories"
  batch_size: 256             # 嵌入计算批次大小
  device: "cpu"               # 计算设备（cpu | cuda）

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

### `embedding` — 文本嵌入

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否在管道中执行嵌入步骤 |
| `model_name` | string | `"all-MiniLM-L6-v2"` | HuggingFace 上的 sentence-transformers 模型名 |
| `text_fields` | list[string] | `["title", "brand", "categories"]` | 拼接为模型输入文本的元数据字段 |
| `batch_size` | int | `256` | 每批编码的商品数 |
| `device` | string | `"cpu"` | 计算设备。GPU 加速使用 `"cuda"` |

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

## 创建自定义配置

只需指定与默认值不同的字段即可：

```yaml
dataset:
  name: "amazon2023"
  category: "Books"

processing:
  kcore_threshold: 10
  max_seq_len: 50

embedding:
  enabled: true
  device: "cuda"
```

未指定的字段自动使用默认值。
