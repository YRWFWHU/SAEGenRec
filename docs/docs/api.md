# API 参考

## 模块结构

```
saegenrec.data
├── config          管道配置 dataclasses
├── pipeline        管道编排器
├── schemas         HuggingFace Dataset Feature 定义
├── loaders         数据加载器
│   ├── base        DatasetLoader ABC + 注册表
│   ├── amazon2015  Amazon 2015 加载器
│   └── amazon2023  Amazon 2023 加载器
├── processors      数据处理器
│   ├── kcore       K-core 过滤
│   ├── sequence    序列构建
│   ├── split       数据划分
│   ├── augment     滑动窗口增强
│   ├── final       最终数据生成
│   └── images      图片下载
├── tokenizers      商品 tokenizer
│   ├── base        ItemTokenizer ABC + 注册表
│   └── passthrough 恒等 tokenizer
└── embeddings
    └── text        文本嵌入生成
```

---

## `saegenrec.data.config`

### `load_config(config_path) → PipelineConfig`

从 YAML 文件加载管道配置。

```python
from saegenrec.data.config import load_config

cfg = load_config("configs/examples/amazon2015_beauty.yaml")
print(cfg.dataset.name)       # "amazon2015"
print(cfg.dataset.data_path)  # Path("data/raw/Amazon2015/Beauty")
```

### 配置 Dataclasses

| 类 | 字段 | 说明 |
|----|------|------|
| `DatasetConfig` | `name`, `category`, `raw_dir` | 数据源定义 |
| `ProcessingConfig` | `kcore_threshold`, `split_strategy`, `split_ratio`, `max_seq_len` | 处理参数 |
| `TokenizerConfig` | `name`, `params` | tokenizer 选择 |
| `EmbeddingConfig` | `enabled`, `model_name`, `text_fields`, `batch_size`, `device` | 嵌入配置 |
| `OutputConfig` | `interim_dir`, `processed_dir` | 输出路径 |
| `PipelineConfig` | `dataset`, `processing`, `tokenizer`, `embedding`, `output` | 顶层配置 |

---

## `saegenrec.data.pipeline`

### `run_pipeline(config, steps=None) → dict`

执行数据处理管道。

**参数**:

- `config` (`PipelineConfig`): 管道配置
- `steps` (`list[str] | None`): 要执行的步骤列表。`None` 表示全部执行

**可用步骤**: `"load"`, `"filter"`, `"sequence"`, `"split"`, `"augment"`, `"generate"`, `"embed"`

**返回**: 包含所有步骤统计信息的字典

```python
from saegenrec.data.config import load_config
from saegenrec.data.pipeline import run_pipeline

cfg = load_config("configs/examples/amazon2015_beauty.yaml")
stats = run_pipeline(cfg, steps=["load", "filter"])
print(stats["filtered_interactions"])  # 198502
```

---

## `saegenrec.data.loaders`

### `DatasetLoader` (ABC)

所有数据加载器的抽象基类。

**抽象方法**:

#### `load_interactions(data_dir: Path) → Dataset`

加载原始交互数据，返回符合 `INTERACTIONS_FEATURES` schema 的 Dataset。

#### `load_item_metadata(data_dir: Path) → Dataset`

加载商品元数据，返回符合 `ITEM_METADATA_FEATURES` schema 的 Dataset。

### `register_loader(name: str)`

装饰器，将 `DatasetLoader` 子类注册到全局注册表。

```python
@register_loader("my_format")
class MyLoader(DatasetLoader):
    ...
```

### `get_loader(name: str) → DatasetLoader`

按注册名获取加载器实例。

```python
loader = get_loader("amazon2015")
interactions = loader.load_interactions(Path("data/raw/Amazon2015/Beauty"))
```

### 内置加载器

| 注册名 | 类 | 交互文件 | 元数据文件 |
|--------|-----|---------|-----------|
| `amazon2015` | `Amazon2015Loader` | `{Category}.json` | `meta_{Category}.json` |
| `amazon2023` | `Amazon2023Loader` | `{Category}.jsonl` | `meta_{Category}.jsonl` |

---

## `saegenrec.data.processors`

### `kcore_filter(interactions, threshold=5) → (Dataset, dict)`

对交互数据执行迭代 K-core 过滤。

**参数**:

- `interactions` (`Dataset`): 输入交互数据
- `threshold` (`int`): 最低交互次数

**返回**: `(过滤后 Dataset, 统计字典)`

统计字典字段: `raw_interactions`, `filtered_interactions`, `kcore_threshold`, `kcore_iterations`, `num_users`, `num_items`

---

### `build_sequences(interactions) → (Dataset, Dataset, Dataset, dict)`

从过滤后的交互构建用户行为序列。

**返回**: `(user_sequences, user_id_map, item_id_map, stats)`

- `user_sequences`: `USER_SEQUENCES_FEATURES` schema
- `user_id_map` / `item_id_map`: `ID_MAP_FEATURES` schema

---

### `save_interim(output_dir, interactions, user_sequences, item_metadata, user_id_map, item_id_map, stats)`

将所有中间数据持久化到磁盘。

---

### `split_data(user_sequences, strategy="loo", ratio=None) → (Dataset, Dataset, Dataset, dict)`

将用户序列划分为 train/valid/test 集。

**参数**:

- `strategy`: `"loo"`（Leave-One-Out）或 `"to"`（Temporal Order）
- `ratio`: TO 策略的 `[train, valid, test]` 比例

---

### `sliding_window_augment(train_sequences, tokenizer, item_titles, max_seq_len=20) → Dataset`

通过滑动窗口从训练序列生成 `(history, target)` 样本。

**返回**: `TRAINING_SAMPLE_FEATURES` schema 的 Dataset

---

### `convert_eval_split(eval_sequences, tokenizer, item_titles, max_seq_len=20) → Dataset`

将验证/测试序列转换为 `TrainingSample` 格式（每用户一个样本，无增强）。

---

### `generate_final_data(...) → dict`

最终数据生成的入口函数，内部调用 `sliding_window_augment` 和 `convert_eval_split`。

---

### `download_images(item_metadata_dir, output_dir, max_workers=4, timeout=10)`

从商品元数据中的 URL 批量下载商品图片。

---

## `saegenrec.data.tokenizers`

### `ItemTokenizer` (ABC)

商品 tokenizer 抽象基类，将连续整数商品 ID 转换为离散 token 序列。

**抽象方法/属性**:

| 方法/属性 | 签名 | 说明 |
|----------|------|------|
| `tokenize` | `(item_id: int) → list[int]` | ID → token 序列 |
| `detokenize` | `(tokens: list[int]) → int` | token 序列 → ID |
| `vocab_size` | `→ int` (property) | 词表大小 |
| `token_length` | `→ int` (property) | 每个商品的 token 序列长度 |
| `tokenize_batch` | `(item_ids: list[int]) → list[list[int]]` | 批量 tokenize（可覆写优化） |

### `register_tokenizer(name: str)` / `get_tokenizer(name: str, **kwargs) → ItemTokenizer`

注册和获取 tokenizer 的辅助函数，用法与 `register_loader` / `get_loader` 类似。

### 内置 Tokenizer

| 注册名 | 类 | 行为 |
|--------|-----|------|
| `passthrough` | `PassthroughTokenizer` | 恒等映射，`tokenize(x) = [x]`，`vocab_size = num_items` |

---

## `saegenrec.data.embeddings`

### `generate_text_embeddings(item_metadata_dir, item_id_map_dir, output_dir, model_name, text_fields, batch_size, device)`

使用 sentence-transformers 为商品生成文本嵌入向量。

**流程**:

1. 加载 `item_metadata` 和 `item_id_map`
2. 对每个商品，将 `text_fields` 指定的字段拼接为文本
3. 分批通过 sentence-transformers 模型编码
4. 输出 `TEXT_EMBEDDING_FEATURES` schema 的 Dataset

---

## `saegenrec.data.schemas`

定义管道各阶段的 HuggingFace `Features` schema 常量。

| 常量 | 用途 |
|------|------|
| `INTERACTIONS_FEATURES` | 原始/过滤后交互 |
| `USER_SEQUENCES_FEATURES` | 用户行为序列 |
| `ITEM_METADATA_FEATURES` | 商品元数据 |
| `ID_MAP_FEATURES` | ID 映射表 |
| `TRAINING_SAMPLE_FEATURES` | 最终训练样本 |
| `TEXT_EMBEDDING_FEATURES` | 文本嵌入向量 |

---

## CLI 入口

### `saegenrec.dataset`

```bash
# 运行完整管道
python -m saegenrec.dataset process <config.yaml> [--step <step_name>]...

# 下载商品图片
python -m saegenrec.dataset download-images <config.yaml>
```
