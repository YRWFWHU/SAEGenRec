# Implementation Plan: Embedding 模块（语义 + 协同）

**Branch**: `003-embedding-module` | **Date**: 2026-03-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-embedding-module/spec.md`

## Summary

设计并实现一个独立的 embedding 模块，包含两个解耦的子系统：语义 embedder（编码型，使用预训练语言模型提取物品元数据语义向量）和协同 embedder（训练型，通过序列推荐模型训练提取物品协同向量）。两个子系统各自拥有独立的 ABC 和注册表，遵循与现有 tokenizer/loader 模块相同的注册表模式，与 tokenizer 模块零耦合。现有 `text_embeddings` 代码将迁移重构为语义 embedder 的默认实现。

## Technical Context

**Language/Version**: Python 3.11 (requires-python ~=3.11.0)
**Primary Dependencies**: sentence-transformers >=2.2, torch >=2.0, transformers >=4.30, datasets >=2.14, typer, loguru
**New Dependencies**: pytorch-lightning (需 constitution 豁免评审，见下文)
**Storage**: HuggingFace Datasets (Arrow format) → data/interim/
**Testing**: pytest
**Target Platform**: Linux (本地开发 + GPU 服务器训练)
**Project Type**: Library + CLI
**Performance Goals**: 语义 10k items < 5 min CPU; 协同 50k interactions < 30 min GPU
**Constraints**: 单机运行，支持 CPU 回退，可复现（固定 seed）
**Scale/Scope**: Amazon Baby (~20k items, ~160k interactions) 为基准数据集

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*
*Post-design re-check: ✅ 通过 (2026-03-02). PyTorch Lightning 豁免已记录于 Complexity Tracking.*

| 原则 | 状态 | 说明 |
|------|------|------|
| I. HuggingFace 设计哲学 | ⚠️ 需豁免 | 规格要求 PyTorch Lightning 训练协同 embedder，与 constitution "训练 MUST 基于 HuggingFace Trainer 或兼容的训练循环" 存在冲突。见 Complexity Tracking 豁免记录。数据存储使用 HuggingFace Datasets ✅ |
| II. 可复现性 | ✅ 通过 | 配置文件驱动所有参数，随机种子通过配置指定并全局设置，输出统计信息 |
| III. 测试驱动 | ✅ 通过 | 每个核心模块需有单元测试：ABC、注册表、语义 embedder、协同 embedder、CLI |
| IV. 双环境分离 | ✅ 通过 | 通过配置区分设备（cpu/cuda），脚本化启动 |
| V. 模块化与可组合 | ✅ 通过 | ABC + 注册表模式，模块间通过接口交互，与 tokenizer 零耦合 |
| VI. 版本控制 | ✅ 通过 | Feature branch 003-embedding-module |
| VII. 简洁性 | ✅ 通过 | 复用已有注册表模式，不引入无必要的抽象层 |
| 技术栈约束 | ⚠️ 需豁免 | PyTorch Lightning 不在批准的技术栈中（与 HuggingFace Trainer 功能重叠），需技术评审 |
| CCDS 目录结构 | ✅ 通过 | 代码在 saegenrec/ 包内，数据在 data/interim/，配置在 configs/ |

## Project Structure

### Documentation (this feature)

```text
specs/003-embedding-module/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── cli.md           # CLI commands contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
saegenrec/
├── data/
│   ├── config.py                          # 扩展: SemanticEmbeddingConfig, CollaborativeEmbeddingConfig
│   ├── pipeline.py                        # 更新: embed 步骤集成新 embedder
│   ├── schemas.py                         # 扩展: SEMANTIC/COLLABORATIVE_EMBEDDING_FEATURES
│   ├── embeddings/
│   │   ├── __init__.py                    # 新建: 导出两套注册表 + ABC
│   │   ├── semantic/
│   │   │   ├── __init__.py                # 导出 SemanticEmbedder ABC + registry
│   │   │   ├── base.py                    # SemanticEmbedder ABC + SEMANTIC_EMBEDDER_REGISTRY
│   │   │   └── sentence_transformer.py    # 默认实现 (迁移自 text.py)
│   │   └── collaborative/
│   │       ├── __init__.py                # 导出 CollaborativeEmbedder ABC + registry
│   │       ├── base.py                    # CollaborativeEmbedder ABC + COLLABORATIVE_EMBEDDER_REGISTRY
│   │       ├── sasrec.py                  # SASRec 实现
│   │       └── models/
│   │           ├── __init__.py
│   │           ├── sasrec_model.py        # SASRec 模型定义 (nn.Module)
│   │           └── metrics.py             # Hit Rate, NDCG 计算
│   ├── tokenizers/                        # 不修改 — 零耦合
│   │   ├── base.py
│   │   ├── passthrough.py
│   │   └── __init__.py
│   └── loaders/                           # 不修改
│       └── ...
├── dataset.py                             # 扩展: embed-semantic, embed-collaborative 命令

tests/
└── unit/
    └── data/
        └── embeddings/
            ├── test_semantic_base.py      # SemanticEmbedder ABC + registry 测试
            ├── test_collaborative_base.py # CollaborativeEmbedder ABC + registry 测试
            ├── test_sentence_transformer.py # SentenceTransformerEmbedder 测试
            └── test_sasrec_embedder.py    # SASRecEmbedder 训练+提取测试

configs/
└── default.yaml                           # 扩展: semantic_embedding + collaborative_embedding 配置节
```

**Structure Decision**: 在现有 `saegenrec/data/embeddings/` 下创建 `semantic/` 和 `collaborative/` 子包，各自包含独立的 ABC、注册表和实现。现有 `text.py` 的逻辑迁移到 `semantic/sentence_transformer.py`，原文件删除。此结构使两个子系统在文件系统层面也物理解耦，同时遵循 CCDS 约束将源代码保持在 `saegenrec/` 包内。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| PyTorch Lightning（技术栈外依赖） | 协同 embedder 需要训练序列推荐模型，Lightning 提供自动 GPU 管理、梯度累积、日志回调、checkpoint 保存等训练基础设施，减少样板代码。用户在功能描述中明确指定。 | 自定义训练循环：可行但需手动实现设备管理、梯度累积、日志、早停等逻辑，增加维护负担。HuggingFace Trainer：设计偏向 NLP 任务的训练范式（tokenized text input），序列推荐模型的数据格式（用户交互序列）和评估方式（排序指标）与其预设范式不匹配，适配成本高。 |
