# Implementation Plan: 生成式推荐数据处理流水线

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-genrec-data-pipeline/spec.md`

## Summary

设计并实现一个模块化的数据处理流水线，将 Amazon 评论原始数据（2015/2023 两种格式）处理为用于训练 LLM 生成式推荐模型的标准化数据集。流水线包括：原始数据加载与清洗（可插拔的数据集 Loader）、K-core 过滤、用户交互序列构建、LOO/TO 数据划分、滑动窗口数据增强、物品 Tokenizer 接口（含透传默认实现）、文本 embedding 生成工具，以及最终训练数据输出。所有中间和最终数据均以 HuggingFace Datasets 格式存储，全部处理参数通过 YAML 配置文件驱动。

## Technical Context

**Language/Version**: Python 3.11（`pyproject.toml`: `requires-python = "~=3.11.0"`）
**Primary Dependencies**: HuggingFace Datasets（数据存储与加载）、pandas（数据处理）、PyTorch + Transformers（文本 embedding 生成）、sentence-transformers（可选的 embedding 便捷接口）、typer（CLI）、loguru（日志）、tqdm（进度条）、PyYAML（配置加载）、dataclasses（配置定义）
**Storage**: 文件系统 — HuggingFace Datasets Arrow 格式（`data/interim/`、`data/processed/`）
**Testing**: pytest
**Target Platform**: Linux（本地开发 + 服务器训练，单机运行）
**Project Type**: Library + CLI（数据处理流水线，作为 `saegenrec` 包的子模块）
**Performance Goals**: 10 万条交互记录全流程处理 < 10 分钟（SC-001）
**Constraints**: 单机处理、相同配置和随机种子下 100% 可复现（SC-002）、数据零泄露（SC-003）
**Scale/Scope**: ~100K-1M 交互记录，Amazon2015（9 个类目）和 Amazon2023（7 个类目）多类目数据集

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Constitution 原则 | 状态 | 说明 |
|---|------------------|------|------|
| I | HuggingFace 设计哲学优先 | ✅ PASS | FR-005/FR-012 要求中间和最终数据以 HuggingFace Datasets 格式存储；数据加载使用 `datasets` 库；本阶段不涉及模型定义，无需 `PreTrainedModel` |
| II | 可复现性 | ✅ PASS | FR-014 要求 YAML 配置驱动所有参数；SC-002 要求 100% 可复现；数据处理流程确定性（不涉及随机操作，排序基于时间戳） |
| III | 测试驱动 | ✅ PASS | SC-007 要求所有核心模块有单元测试，覆盖率 ≥80%；测试使用小规模合成数据 |
| IV | 双环境分离 | ✅ PASS | 配置文件驱动数据路径和参数，无硬编码路径；`config.py` 已定义路径常量 |
| V | 模块化与可组合性 | ✅ PASS | FR-001 可插拔数据集 Loader；FR-009 ItemTokenizer 抽象接口；各处理步骤独立模块 |
| VI | 版本控制 | ✅ PASS | Feature branch `001-genrec-data-pipeline`；大文件（数据集）通过 `.gitignore` 排除 |
| VII | 简洁性 | ✅ PASS | YAGNI: 默认 PassthroughTokenizer、具体量化实现延迟；文本 embedding 为工具功能；图片下载 P3 优先级 |

**技术栈合规性**:

| 约束项 | 状态 | 说明 |
|--------|------|------|
| Python 3.10+ | ✅ | 使用 3.11 |
| HuggingFace Datasets | ✅ | 中间/最终数据格式 |
| 配置管理: dataclasses | ✅ | YAML 加载到 dataclasses |
| pytest | ✅ | 测试框架 |
| ruff | ✅ | 已配置 |
| pyproject.toml + pip | ✅ | 已有 |
| CCDS 目录结构 | ✅ | 遵循现有目录结构 |

**⚠ 需补充的依赖**（当前 `pyproject.toml` 中缺失）:
- `datasets` — HuggingFace Datasets 库
- `pyyaml` — YAML 配置加载
- `torch` — PyTorch（文本 embedding 生成所需）
- `transformers` — HuggingFace Transformers（文本 embedding 生成所需）
- `sentence-transformers` — 便捷的文本 embedding 接口（可选）
- `requests` — 图片下载（P3）

## Project Structure

### Documentation (this feature)

```text
specs/001-genrec-data-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── dataset-loader.md
│   ├── item-tokenizer.md
│   ├── pipeline-config.md
│   └── data-schemas.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
saegenrec/
├── __init__.py               # existing
├── config.py                 # existing — 路径常量
├── dataset.py                # existing — 重构为流水线 CLI 入口
├── data/                     # NEW — 数据处理包
│   ├── __init__.py
│   ├── pipeline.py           # 流水线编排器
│   ├── config.py             # 流水线配置 dataclasses
│   ├── loaders/              # 可插拔数据集加载器
│   │   ├── __init__.py
│   │   ├── base.py           # DatasetLoader 抽象基类
│   │   ├── amazon2015.py     # Amazon2015 JSON 格式解析
│   │   └── amazon2023.py     # Amazon2023 JSONL 格式解析
│   ├── processors/           # 核心处理步骤
│   │   ├── __init__.py
│   │   ├── kcore.py          # K-core 迭代过滤
│   │   ├── sequence.py       # 用户交互序列构建 + ID 映射
│   │   ├── split.py          # LOO / TO 数据划分
│   │   ├── augment.py        # 滑动窗口数据增强
│   │   └── final.py          # 最终训练数据生成
│   ├── tokenizers/           # 物品 Tokenizer
│   │   ├── __init__.py
│   │   ├── base.py           # ItemTokenizer 抽象接口
│   │   └── passthrough.py    # PassthroughTokenizer 默认实现
│   └── embeddings/           # Embedding 工具
│       ├── __init__.py
│       └── text.py           # 文本 embedding 生成
├── features.py               # existing (placeholder)
├── plots.py                  # existing (placeholder)
└── modeling/                 # existing (placeholder)
    ├── train.py
    └── predict.py

tests/
├── conftest.py               # 全局 fixtures
├── unit/
│   ├── conftest.py           # 合成测试数据 fixtures
│   └── data/
│       ├── test_loaders.py   # DatasetLoader 测试
│       ├── test_kcore.py     # K-core 过滤测试
│       ├── test_sequence.py  # 序列构建测试
│       ├── test_split.py     # 数据划分测试
│       ├── test_augment.py   # 滑动窗口测试
│       ├── test_final.py     # 最终数据生成测试
│       └── test_tokenizers.py # ItemTokenizer 测试

configs/                      # NEW — 实验配置文件
├── default.yaml              # 默认流水线配置
└── examples/
    ├── amazon2015_baby.yaml  # Amazon2015 Baby 示例配置
    └── amazon2023_beauty.yaml # Amazon2023 All_Beauty 示例配置

scripts/                      # NEW — 入口脚本
└── run_pipeline.sh           # 流水线运行脚本
```

**Structure Decision**: 采用 CCDS 标准结构，数据处理模块置于 `saegenrec/data/` 子包内。`configs/` 和 `scripts/` 为 CCDS 扩展目录。所有新文件严格放置在 CCDS 规定的目录中。

## Post-Design Constitution Re-Check

*Phase 1 设计完成后的合规性复核。*

| # | Constitution 原则 | 状态 | 设计验证 |
|---|------------------|------|----------|
| I | HuggingFace 设计哲学优先 | ✅ PASS | 中间/最终数据均使用 `datasets.Dataset.save_to_disk()`/`load_from_disk()`；data-schemas.md 定义了完整的 HF Features schema；不涉及模型定义 |
| II | 可复现性 | ✅ PASS | pipeline-config.md 定义了完整的 YAML 配置 schema（含所有默认值）；数据处理流程无随机操作；stats.json 记录完整处理统计 |
| III | 测试驱动 | ✅ PASS | 项目结构中为每个核心模块定义了对应测试文件；item-tokenizer.md 定义了可测试的 round-trip invariants |
| IV | 双环境分离 | ✅ PASS | 所有路径通过 PipelineConfig 配置，支持相对和绝对路径；CLI 入口通过 YAML 文件驱动 |
| V | 模块化与可组合性 | ✅ PASS | dataset-loader.md 和 item-tokenizer.md 定义了 ABC + registry 的可插拔接口；各处理步骤独立模块，通过 pipeline.py 编排 |
| VI | 版本控制 | ✅ PASS | 所有数据文件在 `data/` 目录下，已被 `.gitignore` 排除；代码结构清晰，适合 feature branch 工作流 |
| VII | 简洁性 | ✅ PASS | 无不必要的抽象层；registry 模式简单直接；config loading 用 ~20 行代码实现，未引入额外依赖（dacite/pydantic/hydra）|

**CCDS 目录结构验证**:
- `saegenrec/data/` → 源代码包内子模块 ✅
- `tests/unit/data/` → 测试代码 ✅
- `configs/` → CCDS 扩展，已在 Constitution 中声明 ✅
- `scripts/` → CCDS 扩展，已在 Constitution 中声明 ✅
- `data/raw/`, `data/interim/`, `data/processed/`, `data/external/` → CCDS 标准 ✅

**结论**: 设计完全符合 Constitution 所有原则和技术栈约束，无违规项。

## Complexity Tracking

> 无 Constitution 违规项需要辩护。所有设计决策均符合 Constitution 原则。
