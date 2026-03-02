# Implementation Plan: 生成式推荐架构

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-genrec-architecture/spec.md`

## Summary

设计并实现生成式推荐架构的核心模块栈：ItemTokenizer（RQ-VAE / RQ-KMeans）、SFT 数据构建器、LLM 模型接口（HuggingFace 风格）、约束解码。四个子模块统一放在 `saegenrec/modeling/` 下，形成生成式推荐的完整模型包。数据处理管道 (`saegenrec/data/`) 负责原始数据到 embedding 的前序流程，模型包消费 embedding 并产出最终建模数据到 `data/processed/`。

## Technical Context

**Language/Version**: Python 3.11 (pyproject.toml `requires-python = "~=3.11.0"`)
**Primary Dependencies**: PyTorch ≥2.0, PyTorch Lightning ≥2.0, HuggingFace Datasets ≥2.14, Transformers ≥4.30, scikit-learn, FAISS (新增), Typer (CLI)
**Storage**: HuggingFace Datasets (Arrow 格式) + YAML/JSON 模板文件
**Testing**: pytest
**Target Platform**: Linux (本地单 GPU 开发 + 服务器多 GPU 训练)
**Project Type**: Python library + CLI
**Performance Goals**: ~10K 物品的 tokenization 在 10 分钟内完成 (SC-001)
**Constraints**: 碰撞消解后 SID 唯一性 100% (SC-002)，码本利用率 ≥50% (SC-003)
**Scale/Scope**: 1K~100K 物品（Amazon K-core 过滤后）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 状态 | 说明 |
|------|------|------|
| I. HuggingFace 设计哲学 | ✅ PASS | ItemTokenizer/SFTTaskBuilder 均使用 ABC + 注册表模式；GenRecModel 提供 `from_pretrained`/`save_pretrained`；输出为 HF Dataset |
| II. 可复现性 | ✅ PASS | 所有训练参数通过 YAML 配置驱动；随机种子可配置；RQ-VAE 使用 PyTorch Lightning 自带的种子管理 |
| III. 测试驱动 | ✅ PASS | 每个模块设计对应单元测试；使用小规模合成 embedding 数据测试 |
| IV. 双环境分离 | ✅ PASS | 通过配置文件区分 device/路径；RQ-KMeans 可在 CPU 运行 |
| V. 模块化与可组合性 | ✅ PASS | tokenizers/sft/genrec/decoding 四个子包独立可替换；通过注册表组合 |
| VI. 版本控制 | ✅ PASS | 遵循 feature branch 工作流 |
| VII. 简洁性 | ✅ PASS | 复用现有注册表模式；不引入新抽象层；FAISS 是唯一新依赖且必要（KMeans 性能） |

**Gate Result**: ALL PASS — 无需 Complexity Tracking。

## Project Structure

### Documentation (this feature)

```text
specs/004-genrec-architecture/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── item-tokenizer.md
│   ├── sft-task-builder.md
│   └── genrec-model.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
saegenrec/
├── data/                            # 数据加载与前序处理（已有，本期不修改）
│   ├── config.py                    # 扩展：新增 ItemTokenizerConfig, SFTBuilderConfig
│   ├── pipeline.py                  # 扩展：新增 tokenize, build-sft 管道步骤
│   ├── schemas.py                   # 扩展：新增 SID_MAP_FEATURES, SFT_FEATURES
│   ├── loaders/                     # 已有
│   ├── processors/                  # 已有
│   └── embeddings/                  # 已有
│
├── modeling/                        # 生成式推荐核心模块包
│   ├── __init__.py
│   ├── tokenizers/                  # ItemTokenizer 子系统
│   │   ├── __init__.py
│   │   ├── base.py                  # ItemTokenizer ABC + 注册表
│   │   ├── rqvae.py                 # RQVAETokenizer 实现
│   │   ├── rqkmeans.py              # RQKMeansTokenizer 实现
│   │   ├── collision.py             # 碰撞消解策略（Sinkhorn / AppendLevel）
│   │   └── models/
│   │       └── rqvae_model.py       # RQ-VAE PyTorch Lightning Module
│   │
│   ├── sft/                         # SFT 数据构建子系统
│   │   ├── __init__.py
│   │   ├── base.py                  # SFTTaskBuilder ABC + 注册表
│   │   ├── seqrec.py                # SeqRec 任务构建器
│   │   ├── item2index.py            # Item2Index 任务构建器
│   │   ├── index2item.py            # Index2Item 任务构建器
│   │   └── builder.py               # 多任务混合编排器（SFTDatasetBuilder）
│   │
│   ├── genrec/                      # LLM 模型本体（HuggingFace 风格）
│   │   ├── __init__.py
│   │   ├── base.py                  # GenRecModel ABC + 注册表
│   │   └── config.py                # GenRecConfig dataclass
│   │
│   └── decoding/                    # 约束解码
│       ├── __init__.py
│       ├── trie.py                  # SID Prefix Trie（前缀树）
│       └── constrained.py           # 约束解码策略（Trie-constrained beam search）
│
├── dataset.py                       # 扩展：新增 tokenize, build-sft CLI 命令

configs/
├── default.yaml                     # 扩展：新增 item_tokenizer, sft_builder 配置段
└── templates/                       # 新增目录
    └── sft_prompts.yaml             # SFT prompt 模板文件

tests/
└── unit/
    ├── modeling/                     # 新增
    │   ├── tokenizers/
    │   │   ├── test_rqvae.py
    │   │   ├── test_rqkmeans.py
    │   │   └── test_collision.py
    │   ├── sft/
    │   │   ├── test_seqrec.py
    │   │   ├── test_item2index.py
    │   │   └── test_builder.py
    │   └── decoding/
    │       ├── test_trie.py
    │       └── test_constrained.py
    └── data/                         # 已有
```

### 数据流与输出路径

```text
data/interim/{dataset}/{category}/           ← data pipeline 产物（已有）
  ├── item_metadata/
  ├── item_id_map/
  ├── user_sequences/
  ├── {split_strategy}/
  │   ├── train_sequences/
  │   └── ...
  ├── item_semantic_embeddings/              ← embed 步骤产物
  └── item_collaborative_embeddings/

data/processed/{dataset}/{category}/         ← modeling 产物（本期新增）
  ├── item_sid_map/                          ← tokenize 步骤产物
  ├── tokenizer_model/                       ← 训练好的 tokenizer 权重
  └── sft_data/                              ← build-sft 步骤产物
```

**Structure Decision**: 生成式推荐的四个核心子系统（tokenizers、sft、genrec、decoding）统一放在 `saegenrec/modeling/` 下，形成完整模型包。`saegenrec/data/` 专注于原始数据到 embedding 的前序处理。模型包的输出写入 `data/processed/`，符合 CCDS 的"最终建模用数据集"定位。原有 `modeling/` 下的 CCDS 占位文件（`train.py`、`predict.py`）将被替换。
