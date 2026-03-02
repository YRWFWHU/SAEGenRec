# Implementation Plan: 重构数据处理管道 — 解耦 Tokenizer 并新增负采样

**Branch**: `002-refactor-data-pipeline` | **Date**: 2026-03-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-refactor-data-pipeline/spec.md`

## Summary

将现有数据预处理管道从 `load → filter → sequence → split → augment → generate → embed` 重构为两阶段架构：**阶段 1（数据过滤）**：`load → filter → sequence`，输出到 `data/interim/{dataset}/{category}/`；**阶段 2（数据划分）**：`split → augment → negative_sampling`，输出到 `data/interim/{dataset}/{category}/{split_strategy}/`。核心变更：(1) augment 步骤与 tokenizer 解耦；(2) 新增 negative_sampling 步骤；(3) 两阶段可独立运行，切换划分策略无需重跑过滤。现有 `generate` 和 `embed` 步骤保留为遗留可选步骤。

## Technical Context

**Language/Version**: Python 3.11 (`pyproject.toml`: `~=3.11.0`)  
**Primary Dependencies**: HuggingFace Datasets ≥2.14, pandas, numpy, loguru, typer, PyYAML ≥6.0  
**Storage**: HuggingFace Datasets Arrow 格式（本地磁盘）  
**Testing**: pytest  
**Target Platform**: Linux（本地开发 + 服务器训练）  
**Project Type**: Library / CLI 工具  
**Performance Goals**: Amazon 2015 Beauty（~200 万交互）全流程 < 5 分钟  
**Constraints**: 最小修改原则，复用现有 DatasetLoader 注册表、K-core 过滤、序列构建等模块  
**Scale/Scope**: 单数据集处理，用户量级 ~10 万，商品量级 ~10 万

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 状态 | 说明 |
|------|------|------|
| I. HuggingFace 设计哲学 | ✅ PASS | 所有输出使用 HF Datasets Arrow 格式，数据加载使用 `datasets` 库 |
| II. 可复现性 | ✅ PASS | 负采样支持随机种子（`numpy.random.Generator`），管道通过 YAML 配置驱动 |
| III. 测试驱动 | ✅ PASS | 新增模块需编写对应单元测试，使用合成数据 |
| IV. 双环境分离 | ✅ PASS | 通过 YAML 配置文件区分环境，无硬编码路径 |
| V. 模块化与可组合性 | ✅ PASS | augment 与 tokenizer 解耦，negative_sampling 为独立模块，两阶段可独立运行 |
| VI. 版本控制 | ✅ PASS | 在 feature branch `002-refactor-data-pipeline` 上开发 |
| VII. 简洁性 | ✅ PASS | 最小修改原则，复用现有模块，不引入新依赖 |
| 技术栈约束 | ✅ PASS | 未引入任何非规定技术栈工具 |
| CCDS 目录结构 | ✅ PASS | 输出到 `data/interim/`，源码在 `saegenrec/`，测试在 `tests/` |

**Gate 结论**: 无违规，可进入 Phase 0。

### Phase 1 后复查

| 原则 | 状态 | 说明 |
|------|------|------|
| I. HuggingFace 设计哲学 | ✅ PASS | 新增 schema (`INTERIM_SAMPLE_FEATURES`, `NEGATIVE_SAMPLE_FEATURES`) 使用 HF Features |
| II. 可复现性 | ✅ PASS | `numpy.random.Generator` + seed 配置确保负采样可复现 |
| III. 测试驱动 | ✅ PASS | 新增 `test_negative_sampling.py`，修改 `test_augment.py` 适配无 tokenizer 签名 |
| V. 模块化与可组合性 | ✅ PASS | 两阶段架构 + 独立模块设计，阶段间通过磁盘 Arrow 文件解耦 |
| VII. 简洁性 | ✅ PASS | 仅新增 1 个源文件 + 1 个测试文件，复用全部现有模块 |
| CCDS 目录结构 | ✅ PASS | 新文件均在规定目录内，无新增顶层目录 |

## Project Structure

### Documentation (this feature)

```text
specs/002-refactor-data-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
saegenrec/
├── data/
│   ├── config.py               # [MODIFY] 新增 num_negatives, seed 配置
│   ├── pipeline.py             # [MODIFY] 两阶段架构重构，解耦 tokenizer
│   ├── schemas.py              # [MODIFY] 新增无 token 的 InterimSample / NegativeSample schema
│   ├── loaders/                # [NO CHANGE]
│   │   ├── base.py
│   │   ├── amazon2015.py
│   │   └── amazon2023.py
│   ├── processors/
│   │   ├── augment.py          # [MODIFY] 移除 tokenizer 依赖
│   │   ├── negative_sampling.py # [NEW] 负采样处理器
│   │   ├── kcore.py            # [NO CHANGE]
│   │   ├── sequence.py         # [NO CHANGE]
│   │   ├── split.py            # [NO CHANGE]
│   │   ├── final.py            # [NO CHANGE] 保留供后续 tokenization 使用
│   │   └── images.py           # [NO CHANGE]
│   ├── tokenizers/             # [NO CHANGE] 保留但不在预处理管道中使用
│   └── embeddings/             # [NO CHANGE]
├── dataset.py                  # [MODIFY] CLI 适配新步骤列表和覆盖参数

tests/
└── unit/
    └── data/
        ├── conftest.py         # [MODIFY] 新增负采样相关 fixture
        ├── test_augment.py     # [MODIFY] 适配无 tokenizer 的 augment
        ├── test_negative_sampling.py  # [NEW] 负采样单元测试
        └── test_pipeline.py    # [MODIFY] 适配新管道步骤

configs/
├── default.yaml                # [MODIFY] 新增 num_negatives, seed
└── examples/                   # [MODIFY] 更新示例配置

Makefile                        # [MODIFY] 新增 data-filter / data-split 参数化目标
```

**Structure Decision**: 遵循现有 CCDS 布局，所有源码变更在 `saegenrec/data/` 下，新增文件仅 `processors/negative_sampling.py` 和 `tests/unit/data/test_negative_sampling.py`。

## Complexity Tracking

无违规项，无需记录。
