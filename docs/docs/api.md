# API 参考

## 模块结构

```
saegenrec.data
├── config              管道配置 dataclasses
├── pipeline            管道编排器（两阶段 + 嵌入 + 建模架构）
├── schemas             HuggingFace Dataset Feature 定义
├── loaders             数据加载器
│   ├── base            DatasetLoader ABC + 注册表
│   ├── amazon2015      Amazon 2015 加载器
│   └── amazon2023      Amazon 2023 加载器
├── processors          数据处理器
│   ├── kcore           K-core 过滤
│   ├── sequence        序列构建
│   ├── split           数据划分
│   ├── augment         滑动窗口增强（tokenizer 无关）
│   ├── negative_sampling  负采样
│   ├── final           遗留: 最终数据生成（含 tokenizer）
│   └── images          图片下载
├── tokenizers          商品 tokenizer（遗留，用于 generate 步骤）
│   ├── base            ItemTokenizer ABC + 注册表
│   └── passthrough     恒等 tokenizer
└── embeddings          商品嵌入生成
    ├── text            遗留: 文本嵌入（deprecated）
    ├── semantic        语义嵌入子系统
    │   ├── base            SemanticEmbedder ABC + 注册表
    │   └── sentence_transformer  sentence-transformers 实现
    └── collaborative   协同嵌入子系统
        ├── base            CollaborativeEmbedder ABC + 注册表
        ├── sasrec          SASRec embedder 实现
        └── models          模型定义
            ├── sasrec_model    SASRec nn.Module
            └── metrics         推荐评估指标（Hit Rate, NDCG）

saegenrec.modeling
├── tokenizers          物品 Tokenization（embedding → SID）
│   ├── base            ItemTokenizer ABC + 注册表
│   ├── collision       碰撞消解策略
│   ├── rqvae           RQ-VAE tokenizer
│   ├── rqkmeans        RQ-KMeans tokenizer（FAISS）
│   └── models
│       └── rqvae_model     RQVAEModel（PyTorch Lightning）
├── sft                 SFT 数据构建
│   ├── base            SFTTaskBuilder ABC + 注册表
│   ├── builder         SFTDatasetBuilder 编排器
│   ├── seqrec          序列推荐任务
│   ├── item2index      物品→SID 任务
│   └── index2item      SID→物品 任务
├── genrec              生成式推荐模型
│   ├── base            GenRecModel ABC + 注册表
│   └── config          GenRecConfig dataclass
└── decoding            约束解码
    ├── trie            SIDTrie 前缀树
    └── constrained     SIDConstrainedLogitsProcessor
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
| `ProcessingConfig` | `kcore_threshold`, `split_strategy`, `split_ratio`, `max_seq_len`, `num_negatives`, `seed` | 处理参数 |
| `TokenizerConfig` | `name`, `params` | tokenizer 选择 |
| `EmbeddingConfig` | `enabled`, `model_name`, `text_fields`, `batch_size`, `device` | 遗留嵌入配置（deprecated） |
| `SemanticEmbeddingConfig` | `enabled`, `name`, `model_name`, `text_fields`, `normalize`, `batch_size`, `device` | 语义嵌入配置 |
| `CollaborativeEmbeddingConfig` | `enabled`, `name`, `loss_type`, `hidden_size`, `num_layers`, `num_heads`, `max_seq_len`, `dropout`, `learning_rate`, `batch_size`, `num_epochs`, `eval_top_k`, `device`, `seed` | 协同嵌入配置 |
| `ItemTokenizerConfig` | `enabled`, `name`, `num_codebooks`, `codebook_size`, `collision_strategy`, `sid_token_format`, `sid_begin_token`, `sid_end_token`, `params` | 物品 Tokenization 配置 |
| `SFTBuilderConfig` | `enabled`, `tasks`, `task_weights`, `template_file`, `max_history_len`, `seed` | SFT 数据构建配置 |
| `OutputConfig` | `interim_dir`, `processed_dir` | 输出路径 |
| `PipelineConfig` | `dataset`, `processing`, `tokenizer`, `embedding`, `semantic_embedding`, `collaborative_embedding`, `item_tokenizer`, `sft_builder`, `output` | 顶层配置 |

---

## `saegenrec.data.pipeline`

### 管道常量

| 常量 | 值 | 说明 |
|------|------|------|
| `STAGE1_STEPS` | `["load", "filter", "sequence"]` | 阶段 1 步骤 |
| `STAGE2_STEPS` | `["split", "augment", "negative_sampling"]` | 阶段 2 步骤 |
| `ALL_STEPS` | `STAGE1_STEPS + STAGE2_STEPS` | 所有标准步骤 |
| `LEGACY_STEPS` | `["generate", "embed"]` | 遗留步骤 |

### `run_pipeline(config, steps=None) → dict`

执行数据处理管道。

**参数**:

- `config` (`PipelineConfig`): 管道配置
- `steps` (`list[str] | None`): 要执行的步骤列表。`None` 表示执行 `ALL_STEPS`

**可用步骤**: `"load"`, `"filter"`, `"sequence"`, `"split"`, `"augment"`, `"negative_sampling"`, `"embed"`, `"tokenize"`, `"build-sft"`, `"generate"`（遗留）

**返回**: 包含所有步骤统计信息的字典

**前置条件检查**: 增量运行时会自动验证前置步骤的产物是否存在。例如仅运行 `split` 步骤时，会检查 `user_sequences/` 是否已存在于磁盘。

```python
from saegenrec.data.config import load_config
from saegenrec.data.pipeline import run_pipeline

cfg = load_config("configs/examples/amazon2015_beauty.yaml")

# 仅运行阶段 1
stats = run_pipeline(cfg, steps=["load", "filter", "sequence"])

# 仅运行阶段 2（需先完成阶段 1）
stats = run_pipeline(cfg, steps=["split", "augment", "negative_sampling"])
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
- `ratio`: TO 策略的 `[train, valid, test]` 比例（LOO 时忽略）

---

### `sliding_window_augment(train_sequences, item_titles, max_seq_len=20) → Dataset`

通过滑动窗口从训练序列生成 `(history, target)` 样本。

**返回**: `INTERIM_SAMPLE_FEATURES` schema 的 Dataset（不含 token 字段）

---

### `convert_eval_split(eval_sequences, item_titles, max_seq_len=20) → Dataset`

将验证/测试序列转换为 `InterimSample` 格式（每用户一个样本，无增强）。

**返回**: `INTERIM_SAMPLE_FEATURES` schema 的 Dataset

---

### `build_user_interacted_items(user_sequences) → dict[int, set[int]]`

从用户序列构建用户交互物品映射。

**参数**:

- `user_sequences` (`Dataset`): 用户序列数据

**返回**: `{user_id: {item_id_1, item_id_2, ...}}` 映射

---

### `sample_negatives(samples, user_interacted_items, all_item_ids, item_titles, num_negatives=99, seed=42) → (Dataset, dict)`

为每条样本采样负样本。

**参数**:

- `samples` (`Dataset`): `INTERIM_SAMPLE_FEATURES` schema 的输入样本
- `user_interacted_items` (`dict[int, set[int]]`): 用户交互历史映射
- `all_item_ids` (`list[int]`): 全部商品 ID 列表
- `item_titles` (`dict[int, str]`): 商品 ID → 标题映射
- `num_negatives` (`int`): 每条样本的负采样数
- `seed` (`int | None`): 随机种子

**返回**: `(NegativeSample Dataset, 统计字典)`

---

### `generate_final_data(...) → dict`

遗留的最终数据生成入口函数。内部调用 `sliding_window_augment` 和 `convert_eval_split` 后，通过 tokenizer 附加 token 字段，产出 `TRAINING_SAMPLE_FEATURES` schema。

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

嵌入模块包含两个解耦的子系统，各自拥有独立的 ABC 和注册表。

### `SemanticEmbedder` (ABC)

语义嵌入器抽象基类。使用预训练语言模型对商品元数据文本字段提取语义 embedding。

**抽象方法**:

#### `generate(data_dir: Path, output_dir: Path, config: dict) → Dataset`

从阶段 1 的 `item_metadata/` 和 `item_id_map/` 生成语义 embedding。

- **输入**: `data_dir` 下的 `item_metadata/`、`item_id_map/`
- **输出**: 符合 `SEMANTIC_EMBEDDING_FEATURES` schema 的 Dataset，保存到 `output_dir/item_semantic_embeddings/`

### `register_semantic_embedder(name: str)` / `get_semantic_embedder(name: str) → SemanticEmbedder`

注册和获取语义 embedder 的辅助函数。

### 内置语义 Embedder

| 注册名 | 类 | 说明 |
|--------|-----|------|
| `sentence-transformer` | `SentenceTransformerEmbedder` | 基于 `sentence-transformers` 库 |

---

### `CollaborativeEmbedder` (ABC)

协同嵌入器抽象基类。训练序列推荐模型，从 `nn.Embedding` 权重提取协同过滤 embedding。

**抽象方法**:

#### `generate(data_dir: Path, output_dir: Path, config: dict) → Dataset`

从阶段 2 的 `train_sequences/`、`valid_sequences/`、`test_sequences/` 和 `item_id_map/` 训练模型并提取 embedding。

- **输入**: `data_dir` 下的 split 数据和 ID 映射
- **输出**: 符合 `COLLABORATIVE_EMBEDDING_FEATURES` schema 的 Dataset，保存到 `output_dir/item_collaborative_embeddings/`

### `register_collaborative_embedder(name: str)` / `get_collaborative_embedder(name: str) → CollaborativeEmbedder`

注册和获取协同 embedder 的辅助函数。

### 内置协同 Embedder

| 注册名 | 类 | 模型 | 说明 |
|--------|-----|------|------|
| `sasrec` | `SASRecEmbedder` | SASRec | 自注意力序列推荐，对齐 RecBole。使用 PyTorch Lightning 训练 |

---

### 遗留 API

#### `generate_text_embeddings(item_metadata_dir, item_id_map_dir, output_dir, model_name, text_fields, batch_size, device)`

> **Deprecated**: 请使用 `SemanticEmbedder` 替代。

使用 sentence-transformers 为商品生成文本嵌入向量。

---

## `saegenrec.data.embeddings.collaborative.models`

### `SASRec` (`nn.Module`)

SASRec 自注意力序列推荐模型实现，对齐 RecBole 架构。

**构造参数**: `num_items`, `hidden_size`, `max_seq_len`, `num_layers`, `num_heads`, `dropout`

**关键方法**:

| 方法 | 说明 |
|------|------|
| `forward(item_seq)` | 前向传播，返回序列表示 |
| `bpr_loss(seq, pos, neg)` | 计算 BPR 损失 |
| `ce_loss(seq, pos)` | 计算 CrossEntropy 损失 |
| `predict(item_seq)` | 预测所有物品的分数（用于评估） |

### 推荐评估指标

| 函数 | 说明 |
|------|------|
| `hit_rate_at_k(scores, targets, k)` | 计算 Hit Rate@K |
| `ndcg_at_k(scores, targets, k)` | 计算 NDCG@K |

---

## `saegenrec.data.schemas`

定义管道各阶段的 HuggingFace `Features` schema 常量。

| 常量 | 用途 |
|------|------|
| `INTERACTIONS_FEATURES` | 原始/过滤后交互 |
| `USER_SEQUENCES_FEATURES` | 用户行为序列 |
| `ITEM_METADATA_FEATURES` | 商品元数据 |
| `ID_MAP_FEATURES` | ID 映射表 |
| `INTERIM_SAMPLE_FEATURES` | augment 输出的中间样本（不含 token） |
| `NEGATIVE_SAMPLE_FEATURES` | 负采样后的样本 |
| `TRAINING_SAMPLE_FEATURES` | 遗留: 含 token 的最终训练样本 |
| `TEXT_EMBEDDING_FEATURES` | 遗留: 文本嵌入向量 |
| `SEMANTIC_EMBEDDING_FEATURES` | 语义嵌入向量 |
| `COLLABORATIVE_EMBEDDING_FEATURES` | 协同过滤嵌入向量 |
| `SID_MAP_FEATURES` | 物品 SID 映射（item_id, codes, sid_tokens） |
| `SFT_FEATURES` | SFT 指令数据（task_type, instruction, input, output） |

---

## CLI 入口

### `saegenrec.dataset`

```bash
# 运行完整管道
python -m saegenrec.dataset process <config.yaml> [--step <step_name>]... [--force]

# 可用 CLI 覆盖参数
python -m saegenrec.dataset process <config.yaml> \
    --dataset <name> \
    --category <cat> \
    --kcore <threshold> \
    --max-seq-len <len> \
    --split-strategy <loo|to> \
    --split-ratio <train> <valid> <test> \
    --num-negatives <n> \
    --seed <s>

# 仅生成语义嵌入
python -m saegenrec.dataset embed-semantic <config.yaml> [--force]

# 仅生成协同嵌入
python -m saegenrec.dataset embed-collaborative <config.yaml> [--force]

# 物品 Tokenization（embedding → SID）
python -m saegenrec.dataset tokenize <config.yaml> [--force]

# 构建 SFT 指令数据
python -m saegenrec.dataset build-sft <config.yaml> [--force]

# 下载商品图片
python -m saegenrec.dataset download-images <config.yaml>
```

**`--force` 标志**: 强制覆盖已存在的结果。默认行为是检测到已有结果时跳过。

---

## `saegenrec.modeling` — 建模子系统

公共 API 通过 `saegenrec.modeling` 导出：

```python
from saegenrec.modeling import ItemTokenizer, SFTDatasetBuilder, GenRecModel, SIDTrie
```

---

### `saegenrec.modeling.tokenizers`

#### `ItemTokenizer` (ABC)

物品 Tokenization 的抽象基类，将 embedding 映射为层次化 SID。

**抽象方法/属性**:

| 方法/属性 | 签名 | 说明 |
|----------|------|------|
| `train` | `(semantic_embeddings_dir, collaborative_embeddings_dir, config) → dict` | 训练量化模型 |
| `encode` | `(embeddings: Tensor) → Tensor` | 将 embedding 编码为离散码 |
| `save` | `(path) → None` | 保存模型参数 |
| `load` | `(path) → None` | 加载模型参数 |
| `num_codebooks` | `→ int` (property) | 码本层数 |
| `codebook_size` | `→ int` (property) | 每层码本大小 |

**默认方法**:

#### `generate(semantic_embeddings_dir, collaborative_embeddings_dir, output_dir, config) → Dataset`

完整管道：train → encode → 碰撞消解 → 构建 SID map → 保存。

```python
from saegenrec.modeling.tokenizers.base import get_item_tokenizer

tokenizer = get_item_tokenizer("rqvae")
sid_map = tokenizer.generate(
    semantic_embeddings_dir="data/interim/amazon2015/Beauty/item_semantic_embeddings",
    collaborative_embeddings_dir=None,
    output_dir="data/processed/amazon2015/Beauty",
    config={"num_codebooks": 4, "codebook_size": 256, ...},
)
```

#### `register_item_tokenizer(name)` / `get_item_tokenizer(name, **kwargs)`

注册和获取 ItemTokenizer 的辅助函数。

#### 内置 Tokenizer

| 注册名 | 类 | 说明 |
|--------|-----|------|
| `rqvae` | `RQVAETokenizer` | MLP 编码器 + 残差向量量化，PyTorch Lightning 训练 |
| `rqkmeans` | `RQKMeansTokenizer` | 逐层残差 KMeans，基于 FAISS + k-means-constrained |

---

#### `resolve_collisions(codes, strategy) → list[list[int]]`

碰撞消解函数。将 `(N, num_codebooks)` 的 codes 张量处理为唯一的 SID 列表。

**策略**: `"append_level"`（追加消歧层）、`"sinkhorn"`（当前回退到 append_level）。

---

### `saegenrec.modeling.tokenizers.models`

#### `RQVAEModel` (`pl.LightningModule`)

残差向量量化变分自编码器。

**构造参数**: `input_dim`, `hidden_dim`, `latent_dim`, `num_codebooks`, `codebook_size`, `ema_decay`, `dead_code_threshold`

**防坍缩机制**:

- **数据初始化**: 首个 batch 时用编码器输出初始化码本
- **EMA 更新**: 指数移动平均更新码本向量
- **死码替换**: 使用次数低于阈值的码本条目从当前 batch 重采样

---

### `saegenrec.modeling.sft`

#### `SFTTaskBuilder` (ABC)

SFT 任务构建器抽象基类。

**抽象方法/属性**:

| 方法/属性 | 签名 | 说明 |
|----------|------|------|
| `task_type` | `→ str` (property) | 任务类型标识 |
| `build` | `(stage1_dir, stage2_dir, sid_map, config) → Dataset` | 构建 SFT 数据集 |

**内置方法**:

| 方法 | 说明 |
|------|------|
| `load_templates(template_file)` | 从 YAML 加载当前 task_type 的 prompt 模板 |

#### `register_sft_task(name)` / `get_sft_task_builder(name)`

注册和获取 SFTTaskBuilder 的辅助函数。

#### 内置任务

| 注册名 | 类 | 说明 |
|--------|-----|------|
| `seqrec` | `SeqRecTaskBuilder` | 序列推荐：优先使用滑动窗口增强数据 |
| `item2index` | `Item2IndexTaskBuilder` | 物品标题 → SID |
| `index2item` | `Index2ItemTaskBuilder` | SID → 物品标题 |

#### `SFTDatasetBuilder`

多任务 SFT 数据编排器。

```python
from saegenrec.modeling.sft.builder import SFTDatasetBuilder

builder = SFTDatasetBuilder()
merged_ds = builder.build(stage1_dir, stage2_dir, sid_map, output_dir, config)
```

**功能**: 调用各 task builder → 合并 → shuffle → 保存。支持 `task_weights` 降采样。

---

### `saegenrec.modeling.genrec`

#### `GenRecModel` (ABC)

生成式推荐模型抽象基类。

**抽象方法**:

| 方法 | 签名 | 说明 |
|------|------|------|
| `train` | `(dataset, training_args) → dict` | 训练模型 |
| `generate` | `(input_text, **kwargs) → list[str]` | 生成推荐结果 |
| `evaluate` | `(dataset, metrics) → dict[str, float]` | 评估模型 |
| `save_pretrained` | `(path) → None` | 保存模型 |
| `from_pretrained` | `(path, **kwargs) → GenRecModel` | 加载模型（classmethod） |

#### `register_genrec_model(name)` / `get_genrec_model(name, **kwargs)`

注册和获取 GenRecModel 的辅助函数。

#### `GenRecConfig`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_model_name` | str | `"Qwen/Qwen2.5-0.5B"` | 基座模型名 |
| `lora_enabled` | bool | `True` | 是否启用 LoRA |
| `lora_r` | int | `16` | LoRA 秩 |
| `lora_alpha` | int | `32` | LoRA alpha |
| `lora_target_modules` | list[str] \| None | `None` | LoRA 目标模块 |
| `training_strategy` | str | `"sft"` | 训练策略（sft / rl） |
| `sid_tokens_path` | str \| None | `None` | SID token 文件路径 |

---

### `saegenrec.modeling.decoding`

#### `SIDTrie`

基于前缀树的 SID 搜索结构，用于约束解码。

| 方法 | 签名 | 说明 |
|------|------|------|
| `insert` | `(token_ids: list[int]) → None` | 插入一个 SID 序列 |
| `search_prefix` | `(prefix: list[int]) → list[int]` | 给定前缀返回合法的下一步 token ID |
| `from_sid_map` | `(sid_map, tokenizer) → SIDTrie` | 从 item_sid_map 构建（classmethod） |

```python
from saegenrec.modeling.decoding.trie import SIDTrie
from datasets import load_from_disk
from transformers import AutoTokenizer

sid_map = load_from_disk("data/processed/amazon2015/Beauty/item_sid_map")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.5B")
trie = SIDTrie.from_sid_map(sid_map, tokenizer)
```

#### `SIDConstrainedLogitsProcessor`

HuggingFace `LogitsProcessor` 子类，在 LLM 生成时将非法 SID token 的 logits 设为 `-inf`。

```python
from saegenrec.modeling.decoding.constrained import SIDConstrainedLogitsProcessor

processor = SIDConstrainedLogitsProcessor(
    trie=trie,
    sid_begin_token_id=tokenizer.convert_tokens_to_ids("<|sid_begin|>"),
    sid_end_token_id=tokenizer.convert_tokens_to_ids("<|sid_end|>"),
)
```
