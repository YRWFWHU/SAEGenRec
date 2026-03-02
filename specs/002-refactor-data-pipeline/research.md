# Research: 重构数据处理管道

**Feature**: 002-refactor-data-pipeline  
**Date**: 2026-03-02

## R1: 两阶段架构设计

### Decision
将管道拆为两个独立阶段：
- **阶段 1（数据过滤）**：`load → filter → sequence`，输出到 `data/interim/{dataset}/{category}/`
- **阶段 2（数据划分）**：`split → augment → negative_sampling`，输出到 `data/interim/{dataset}/{category}/{split_strategy}/`

两阶段通过磁盘上的 Arrow 数据集解耦。阶段 2 读取阶段 1 的 `user_sequences`、`item_metadata`、`item_id_map` 等产物。

### Rationale
- 阶段 1（K-core 过滤 + 序列构建）是计算密集型操作，对于大数据集耗时较长
- 阶段 2（划分 + 增强 + 负采样）参数化程度高（split_strategy、max_seq_len、num_negatives），需频繁迭代
- 分离后切换划分策略无需重跑过滤，按 `{split_strategy}/` 隔离输出目录支持多策略结果共存
- 对应两个独立 Make 目标 `make data-filter` / `make data-split`

### Alternatives Considered
1. **单一管道 + `--step` 控制**: 虽然功能等价，但 Make 接口不清晰，用户需理解内部步骤依赖关系
2. **三阶段拆分（过滤/划分/负采样）**: 过细粒度，增加管理复杂度；负采样与 augment 紧耦合（都在 split_strategy 目录下）

## R2: augment 步骤解耦 tokenizer

### Decision
重构 `sliding_window_augment` 和 `convert_eval_split`，移除 `tokenizer` 参数。新函数签名仅接受 `train_sequences: Dataset`、`item_titles: dict[int, str]` 和 `max_seq_len: int`。输出 schema 使用新定义的 `INTERIM_SAMPLE_FEATURES`（不含 `*_tokens` 字段）。

### Rationale
- 当前 `sliding_window_augment` 在循环内调用 `tokenizer.tokenize_batch`，将 tokenization 与样本构建耦合
- 解耦后，augment 产出的样本仅包含 ID 和文本信息，tokenization 延迟到下游 `generate` 步骤
- 保留现有 `final.py` 不变，使其可以在未来对 interim 数据做 tokenization
- 新 schema 包含 `history_item_titles` 和 `target_item_title`（从 item_metadata 查找），满足 FR-004

### Alternatives Considered
1. **保留 tokenizer 参数但传入 passthrough**: 语义不清晰，augment 仍依赖 tokenizer 接口
2. **完全移除文本信息，延迟到 generate**: 违反 FR-004

## R3: 负采样实现策略

### Decision
新增 `saegenrec/data/processors/negative_sampling.py`，核心函数 `sample_negatives()`。对每条样本（train/valid/test），从全局商品集合中均匀随机采样 `num_negatives` 个该用户未交互过的商品 ID。使用 `numpy.random.Generator` 配合指定 seed 确保可复现。

### Rationale
- 均匀随机采样是推荐系统负采样的标准基线策略
- `numpy.random.Generator`（而非全局 `np.random.seed`）遵循现代最佳实践，避免全局状态污染
- 逐用户构建排除集合（`set(user_interacted_items)`），用 `rng.choice` 从候选集中采样
- 当可用负样本 < `num_negatives` 时，采样所有可用负样本并记录警告

### Alternatives Considered
1. **基于流行度的负采样**: spec 明确指出当前仅需均匀随机
2. **使用 Python `random` 模块**: numpy 在大规模采样场景下显著更快
3. **预计算全局候选集**: 对大数据集内存占用过高

## R4: 新 Schema 设计

### Decision
在 `schemas.py` 中新增：

```python
INTERIM_SAMPLE_FEATURES = Features({
    "user_id": Value("int32"),
    "history_item_ids": Sequence(Value("int32")),
    "history_item_titles": Sequence(Value("string")),
    "target_item_id": Value("int32"),
    "target_item_title": Value("string"),
})

NEGATIVE_SAMPLE_FEATURES = Features({
    "user_id": Value("int32"),
    "history_item_ids": Sequence(Value("int32")),
    "history_item_titles": Sequence(Value("string")),
    "target_item_id": Value("int32"),
    "target_item_title": Value("string"),
    "negative_item_ids": Sequence(Value("int32")),
    "negative_item_titles": Sequence(Value("string")),
})
```

### Rationale
- `INTERIM_SAMPLE_FEATURES` 是 augment 输出格式，不含 `*_tokens` 字段（满足 FR-002）
- `NEGATIVE_SAMPLE_FEATURES` 扩展 `negative_item_ids` 和 `negative_item_titles`
- 保留现有 `TRAINING_SAMPLE_FEATURES`（含 tokens）供 `final.py` 继续使用

### Alternatives Considered
1. **直接修改 TRAINING_SAMPLE_FEATURES**: 破坏现有 `final.py` 和相关测试
2. **使用单一 schema 加可选字段**: HF Datasets Features 不支持可选字段

## R5: 配置扩展与 Make 目标设计

### Decision
`ProcessingConfig` 新增 `num_negatives: int = 99` 和 `seed: int | None = 42`。

Makefile 新增两个目标，均仅接受 `CONFIG` 变量指定 YAML 配置文件路径：
- `make data-filter CONFIG=configs/xxx.yaml`: 运行阶段 1（load → filter → sequence）
- `make data-split CONFIG=configs/xxx.yaml`: 运行阶段 2（split → augment → negative_sampling）

CLI `process` 命令新增可选参数（`--num-negatives`、`--seed`、`--split-ratio` 等）用于临时覆盖 YAML 配置值。

### Rationale
- 两个 Make 目标精确匹配两阶段语义
- 所有参数从 YAML 配置文件读取，Make 命令简洁；CLI `--option` 仅用于临时覆盖
- 不同数据集/场景通过独立配置文件管理（如 `configs/examples/amazon2015_beauty.yaml`）
- 默认值与 spec 假设一致（num_negatives=99，seed=42）

### Alternatives Considered
1. **Make 变量逐一暴露参数**: 与 YAML 配置重复，命令冗长
2. **仅通过 YAML 配置无 Make 目标**: 缺少便捷入口

## R6: 输出目录结构

### Decision
```
data/interim/{dataset}/{category}/
├── raw_interactions/       # load 输出
├── item_metadata/          # load 输出
├── interactions/           # filter 输出
├── user_sequences/         # sequence 输出
├── user_id_map/            # sequence 输出
├── item_id_map/            # sequence 输出
├── stats.json              # 阶段 1 统计
├── loo/                    # 阶段 2 (LOO 策略)
│   ├── train_sequences/
│   ├── valid_sequences/
│   ├── test_sequences/
│   ├── train/              # augment + negative_sampling 后
│   ├── valid/
│   ├── test/
│   └── stats.json          # 阶段 2 统计
└── to/                     # 阶段 2 (TO 策略，可并存)
    ├── ...
    └── stats.json
```

### Rationale
- 阶段 1 产物（过滤、序列）与策略无关，放在 `{category}/` 根目录
- 阶段 2 产物按 `{split_strategy}/` 隔离，不同策略可共存
- 各阶段独立 `stats.json`，不互相覆盖

### Alternatives Considered
1. **统一根目录**: 切换策略时覆盖旧结果
2. **更深层嵌套**: 增加路径复杂度
