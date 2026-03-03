# Implementation Plan: SAE Item Tokenizer

**Branch**: `006-item-tokenizer-sae` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-item-tokenizer-sae/spec.md`

## Summary

为 saegenrec 项目新增基于 JumpReLU Sparse Autoencoder (SAE) 的 ItemTokenizer 实现。SAE 在物品文本嵌入上训练，学习稀疏概念表示；编码时选取 top_k 个激活最强的概念特征索引作为物品的语义 ID (SID)。实现参考 SAELens 的 JumpReLU 架构，但在项目内部独立实现，不引入外部依赖。通过现有的 `ItemTokenizer` 注册表和配置体系无缝集成到已有的 tokenize 流程中。

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: torch>=2.0, datasets>=2.14（均为已有依赖，无需新增）  
**Storage**: safetensors 格式保存模型权重 + JSON 保存超参数配置  
**Testing**: pytest  
**Target Platform**: Linux（本地单 GPU 调试 + 服务器训练）  
**Project Type**: Library + CLI  
**Performance Goals**: 训练时间与 RQ-VAE 同量级；encode 为纯前向传播，性能不是瓶颈  
**Constraints**: 遵循 Constitution 的 HuggingFace 设计哲学、可复现性和 CCDS 目录结构  
**Scale/Scope**: Amazon Beauty 数据集（约 10k-50k 物品嵌入），d_sae 最大 16384

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. HuggingFace 设计哲学优先 | ✅ PASS | 使用 dataclass config, HF Datasets 输入格式; SAE 是实用工具组件（同 RQVAEModel 使用 pl.LightningModule），非 LLM 主模型 |
| II. 可复现性 | ✅ PASS | 所有超参数通过 YAML 配置驱动，模型权重可保存/加载，训练指标有日志 |
| III. 测试驱动 | ✅ PASS | 规划了 JumpReLU SAE 模型测试和 SAETokenizer 集成测试 |
| IV. 双环境分离 | ✅ PASS | 通过配置文件区分环境，复用已有 `make data-tokenize` 命令 |
| V. 模块化与可组合性 | ✅ PASS | 遵循 `ItemTokenizer` 抽象接口，通过 `@register_item_tokenizer("sae")` 注册，可替换 |
| VI. 版本控制 | ✅ PASS | 在 feature branch `006-item-tokenizer-sae` 上开发 |
| VII. 简洁性 | ✅ PASS | 项目内独立实现轻量 SAE（~200 行），不引入 SAELens 依赖；简单 PyTorch 训练循环 |

**Post-Phase-1 Re-check**:

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. HuggingFace 设计哲学优先 | ✅ PASS | Config 使用 dataclass，数据用 HF Datasets，与 RQ-VAE tokenizer 保持一致 |
| V. 模块化与可组合性 | ✅ PASS | JumpReLU SAE 模型独立于 tokenizer 封装，可单独测试和复用 |
| VII. 简洁性 | ✅ PASS | 仅实现核心 JumpReLU 前向/反向 + 简洁训练循环，无多余抽象 |

## Project Structure

### Documentation (this feature)

```text
specs/006-item-tokenizer-sae/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 data model
├── quickstart.md        # Phase 1 quickstart guide
├── contracts/
│   └── cli-commands.md  # CLI/config interface contract
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
saegenrec/
└── modeling/
    └── tokenizers/
        ├── __init__.py              # (modify) 导出 SAETokenizer
        ├── sae.py                   # (new) SAETokenizer — ItemTokenizer 实现
        └── models/
            ├── __init__.py          # (existing or new)
            └── jumprelu_sae.py      # (new) JumpReLU SAE 模型（训练版 + 推理版）

tests/
└── unit/
    └── modeling/
        └── tokenizers/
            ├── test_sae_tokenizer.py    # (new) SAETokenizer 集成测试
            └── test_jumprelu_sae.py     # (new) JumpReLU SAE 模型单元测试
```

**Structure Decision**: 遵循项目已有模式——tokenizer 实现放在 `saegenrec/modeling/tokenizers/` 下（同 `rqvae.py`, `rqkmeans.py`），底层模型放在 `models/` 子目录下（同 `models/rqvae_model.py`）。

## Complexity Tracking

> 无 Constitution 违规，此段为空。
