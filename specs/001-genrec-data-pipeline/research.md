# Research: 生成式推荐数据处理流水线

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## R1: YAML 配置与 dataclasses 的桥接方案

**Decision**: 使用纯 dataclasses 定义配置结构，配合 PyYAML 加载 YAML 文件后通过 `dacite` 或手工解析实例化 dataclass。

**Rationale**: Constitution 要求"配置管理: dataclasses 或 HuggingFace TrainingArguments 风格"，spec 要求 YAML 驱动（FR-014）。dataclasses 提供类型检查和默认值，YAML 提供用户友好的配置文件格式。两者结合是最自然的方案：YAML 负责序列化，dataclass 负责运行时类型安全。

不引入 `dacite` 第三方库以保持依赖简洁。使用一个简单的递归函数将嵌套 dict 映射到嵌套 dataclass，利用 `__dataclass_fields__` 和类型注解做转换。

**Alternatives considered**:
- `pydantic`: 功能更强大但引入额外依赖，违反简洁性原则
- `OmegaConf` + `hydra`: 科研项目常用但过于重量级，与 HF 生态风格不符
- 纯 `dataclasses` 无 YAML: 配置不够用户友好，需要在代码中修改参数
- `dacite` 库: 功能合适但引入额外依赖；手工解析逻辑简单，约 20 行代码即可实现

## R2: HuggingFace Datasets Arrow 格式的中间数据存储方案

**Decision**: 中间数据（`data/interim/`）和最终数据（`data/processed/`）均使用 `datasets.Dataset.save_to_disk()` 存储为 Arrow 格式。每个数据集版本 + 类目的处理结果存放在独立子目录中。

**Rationale**: Constitution I 要求"数据加载 MUST 优先使用 `datasets` 库或兼容的 Arrow 格式"。Arrow 格式支持内存映射（memory-mapped）读取，对大规模数据高效；与 HuggingFace 生态无缝集成；保留列类型信息。

目录布局：
```
data/interim/{dataset_version}/{category}/
├── interactions/        # Dataset: 过滤后的交互记录
├── user_sequences/      # Dataset: 用户交互序列
├── item_metadata/       # Dataset: 物品元数据
├── user_id_map/         # Dataset: 用户 ID 映射
├── item_id_map/         # Dataset: 物品 ID 映射
└── text_embeddings/     # Dataset: 物品文本 embedding (可选)

data/processed/{dataset_version}/{category}/{split_strategy}/
├── train/               # Dataset: 训练集（滑动窗口增强后）
├── valid/               # Dataset: 验证集
└── test/                # Dataset: 测试集
```

**Alternatives considered**:
- CSV/TSV（RecBole 风格）: 参考实现常用但不符合 Constitution 的 HF 优先原则
- Parquet: HF Datasets 底层也支持，但 `save_to_disk` 的 Arrow 格式保留更多元数据（schema、feature types）
- JSON Lines: 参考实现中也常用，但大规模数据下 I/O 效率低于 Arrow

## R3: 可插拔数据集 Loader 的设计模式

**Decision**: 使用抽象基类（ABC）定义 `DatasetLoader` 接口，每个数据集版本（Amazon2015、Amazon2023）实现具体子类。通过 registry dict 按名称注册和查找 Loader。

**Rationale**: FR-001 要求"新增数据集仅需添加对应的处理脚本，不修改核心流水线代码"。ABC + registry 是 Python 中最常见的可插拔模式：

```python
class DatasetLoader(ABC):
    @abstractmethod
    def load_interactions(self, data_dir: Path) -> Dataset: ...
    
    @abstractmethod
    def load_item_metadata(self, data_dir: Path) -> Dataset: ...

LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {}
```

新增 Loader 只需继承 ABC 并注册到 registry。参考 MiniOneRec 和 LLMRank 的模式：每种数据集有独立的处理脚本，但缺少统一接口。本设计在此基础上增加了接口抽象。

**Alternatives considered**:
- 函数式（每个 Loader 一个函数）: 更简单但缺少接口约束，难以确保输出格式一致
- Plugin 系统（`importlib`）: 过于复杂，违反简洁性原则
- 工厂方法模式: 本质上与 registry + ABC 等价，但 registry 更显式

## R4: K-core 迭代过滤的实现策略

**Decision**: 使用 pandas 实现迭代 K-core 过滤。每轮迭代计算用户和物品的交互计数，过滤掉低于阈值的实体，重复直到无变化（收敛）。

**Rationale**: 参考 LLMRank 的 `filter_inters()` 和 MiniOneRec 的 `k_core_filtering_json2csv_style()` 均采用迭代过滤直到稳定的方式。pandas 的 `groupby().transform('count')` 能高效计算计数。

```python
while True:
    user_counts = df.groupby('user_id')['item_id'].transform('count')
    item_counts = df.groupby('item_id')['user_id'].transform('count')
    mask = (user_counts >= k) & (item_counts >= k)
    if mask.all():
        break
    df = df[mask]
```

对 10 万条数据，通常 3-5 轮迭代即可收敛，性能无忧。

**Alternatives considered**:
- 图算法（NetworkX）: 概念上更优雅但引入不必要的依赖
- 纯 NumPy: 需要手工管理索引映射，代码可读性差
- SQL (DuckDB): 功能强大但引入新依赖

## R5: Amazon2015 JSON 格式的解析策略

**Decision**: 使用逐行 `json.loads()` 解析 Amazon2015 的 `.json` 文件（实际为 JSON Lines 格式，每行一个 JSON 对象）。

**Rationale**: 实际检查原始文件发现，Amazon2015 的 `.json` 文件虽然扩展名为 `.json`，但每行是一个独立的 JSON 对象（NDJSON/JSON Lines 格式），而非标准 JSON 数组。这与 Amazon2023 的 `.jsonl` 文件格式本质相同，仅扩展名不同。

所有参考实现（LLMRank、MiniOneRec）均采用逐行解析方式处理这类文件。

**Alternatives considered**:
- `pd.read_json(lines=True)`: 可行但对大文件一次性加载到内存
- `datasets.load_dataset('json')`: HF 原生支持但对字段映射不够灵活
- 流式解析（`ijson`）: 对非标准 JSON 无需如此

## R6: 文本 Embedding 模型的选择

**Decision**: 使用 `sentence-transformers` 库加载预训练模型，默认模型为 `all-MiniLM-L6-v2`（轻量级、质量适中）。支持通过配置指定其他模型（如 `text-embedding-3-small`、`bge-base-en-v1.5`）。

**Rationale**: FR-011 要求"使用预训练语言模型对物品元数据文本生成向量表示"。`sentence-transformers` 是 HuggingFace 生态中最常用的文本 embedding 工具，API 简洁（`model.encode(texts)`），支持 batch 推理和 GPU 加速。

参考 MiniOneRec 的 `amazon_text2emb.py` 使用 Transformers 库的 `AutoModel` + mean pooling，但 `sentence-transformers` 封装了这些细节，更符合简洁性原则。

embedding 存储为 NumPy 数组（`.npy`），与 HuggingFace Datasets 兼容（可作为 `Array2D` 特征类型存储在 Dataset 中）。

**Alternatives considered**:
- 原生 Transformers + mean pooling: 参考 MiniOneRec 的方式，更灵活但代码更多
- OpenAI API: 质量好但需要 API key、产生费用、不可本地复现
- TF-IDF / BM25: 非深度学习方法，表示能力较弱

## R7: 滑动窗口数据增强的实现策略

**Decision**: 在 pandas DataFrame 或 HuggingFace Dataset 上实现滑动窗口。对每个用户的训练序列，生成 (history, target) 样本对。history 从左截断到 max_seq_len。

**Rationale**: FR-008 明确定义了滑动窗口逻辑。参考实现中 Align3GR 使用 `arr[i:i+seq_len]` 模式，MiniOneRec 在数据生成时直接截断历史。

关键实现点：
- 每个用户序列 `[i_1, i_2, ..., i_N]` 生成 N-1 个样本
- 第 k 个样本: history = `[i_1, ..., i_k][-max_seq_len:]`, target = `i_{k+1}`
- history 长度范围 `[1, max_seq_len]`

使用 Python list 操作 + `datasets.Dataset.from_dict()` 批量构建输出 Dataset，避免逐条 append 的性能问题。

**Alternatives considered**:
- NumPy 向量化: 对变长序列不友好
- 训练时在线生成（DataLoader）: 将增强推迟到训练阶段，但不符合流水线"预处理"的定位
- 多进程并行: 初版不需要，10 万条数据单进程足够（SC-001: <10 分钟）

## R8: LOO vs TO 划分的统一接口设计

**Decision**: 定义统一的 `split_data()` 函数，接受策略参数（`"loo"` 或 `"to"`），返回 `(train, valid, test)` 三个 Dataset。LOO 按用户级别划分，TO 按全局时间戳划分。

**Rationale**: FR-006/FR-007 分别定义了两种划分策略。两者输入相同（用户交互序列 Dataset）、输出相同（三个 Dataset），适合用策略模式统一接口。

LOO 实现：
- 每个用户最后 1 条 → test，倒数第 2 条 → valid，其余 → train
- 交互总数 < 3 的用户排除（Acceptance Scenario 2.3）

TO 实现：
- 所有交互按时间戳排序
- 按比例（默认 8:1:1）分配到 train/valid/test
- 划分点基于总交互数的百分位

**Alternatives considered**:
- 独立函数（`split_loo()`, `split_to()`）: 更简单但接口不统一
- 策略类（`SplitStrategy` ABC）: 过度设计，两种策略用 if-else 足矣

## R9: 物品 Tokenizer 的依赖注入方式

**Decision**: 最终数据生成步骤（`final.py`）接受一个 `ItemTokenizer` 实例作为参数。默认使用 `PassthroughTokenizer`。通过配置文件指定 Tokenizer 类名，在流水线编排器中实例化并注入。

**Rationale**: FR-009/FR-010 定义了接口和默认实现。依赖注入使得最终数据生成不依赖具体的 Tokenizer 实现，符合模块化原则（Constitution V）。

实例化方式：配置文件中指定 `tokenizer.class_name: "PassthroughTokenizer"` 和可选的 `tokenizer.params`，流水线编排器根据类名从 registry 中查找并实例化。

**Alternatives considered**:
- 工厂函数: 本质等价，但 registry 更显式
- 硬编码默认: 违反可配置原则
- 配置中指定完整 Python 路径（`saegenrec.data.tokenizers.passthrough.PassthroughTokenizer`）: 过于冗长

## R10: 依赖版本管理策略

**Decision**: 在 `pyproject.toml` 中添加以下新依赖（不指定精确版本，使用兼容范围）：

```toml
"datasets>=2.14",
"pyyaml>=6.0",
"torch>=2.0",
"transformers>=4.30",
"sentence-transformers>=2.2",
"requests>=2.28",
```

**Rationale**: Constitution 要求使用 `pyproject.toml + pip` 管理依赖。不指定精确版本以避免与用户环境冲突，但设置最低版本确保 API 兼容性。`torch` 版本范围较宽是因为用户可能已安装特定 CUDA 版本的 PyTorch。

`datasets>=2.14` 确保支持 `Dataset.save_to_disk()` 的最新格式。`transformers>=4.30` 确保支持较新的模型架构。

**Alternatives considered**:
- 精确版本锁定: 更可复现但过于严格，可能与用户环境冲突
- 不设最低版本: 可能引入 API 不兼容的旧版本
