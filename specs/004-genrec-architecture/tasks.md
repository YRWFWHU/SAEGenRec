# Tasks: 生成式推荐架构

**Input**: Design documents from `/specs/004-genrec-architecture/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included — Constitution III（测试驱动）要求每个核心模块有对应单元测试。

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 创建 `saegenrec/modeling/` 包结构、添加依赖、扩展配置和 schema

- [X] T001 Create modeling/ package directory structure with `__init__.py` files for `saegenrec/modeling/`, `saegenrec/modeling/tokenizers/`, `saegenrec/modeling/tokenizers/models/`, `saegenrec/modeling/sft/`, `saegenrec/modeling/genrec/`, `saegenrec/modeling/decoding/`
- [X] T002 Remove CCDS placeholder files `saegenrec/modeling/train.py` and `saegenrec/modeling/predict.py`
- [X] T003 Add `faiss-cpu` and `k-means-constrained` dependencies to `pyproject.toml` (both required)
- [X] T004 [P] Add `ItemTokenizerConfig` and `SFTBuilderConfig` dataclasses to `saegenrec/data/config.py`; extend `PipelineConfig` with `item_tokenizer` and `sft_builder` fields; add `OutputConfig.modeling_path(dataset_name, category)` method (不含 split_strategy，用于 tokenize/build-sft 输出); update `load_config()` to parse new sections
- [X] T005 [P] Add `SID_MAP_FEATURES` and `SFT_FEATURES` to `saegenrec/data/schemas.py`
- [X] T006 [P] Create test directory structure: `tests/unit/modeling/tokenizers/`, `tests/unit/modeling/sft/`, `tests/unit/modeling/decoding/` with `conftest.py` providing shared fixtures (synthetic embeddings, mock SID map)

**Checkpoint**: 项目结构就绪，可开始各 User Story 的实现。

---

## Phase 2: User Story 1 — 物品 Tokenization (Priority: P1) 🎯 MVP

**Goal**: 将物品 embedding 映射为层次化语义 ID（SID），支持 RQ-VAE 和 RQ-KMeans 两种 tokenizer，碰撞消解后确保 SID 唯一。

**Independent Test**: 给定一组合成 embedding，运行 ItemTokenizer 训练和推理，验证输出的 SID 满足唯一性约束且码本利用率 ≥50%。

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T007 [P] [US1] Unit test for ItemTokenizer ABC contract (registry, get_item_tokenizer, abstract methods) in `tests/unit/modeling/tokenizers/test_base.py`
- [X] T008 [P] [US1] Unit test for collision resolution (append_level uniqueness, sinkhorn convergence, edge cases: all-same embeddings, items < codebook_size) in `tests/unit/modeling/tokenizers/test_collision.py`
- [X] T009 [P] [US1] Unit test for RQ-VAE tokenizer (train on synthetic embeddings, encode shape, save/load roundtrip, codebook utilization) in `tests/unit/modeling/tokenizers/test_rqvae.py`
- [X] T010 [P] [US1] Unit test for RQ-KMeans tokenizer (train on synthetic embeddings, encode shape, save/load roundtrip, constrained mode) in `tests/unit/modeling/tokenizers/test_rqkmeans.py`

### Implementation for User Story 1

- [X] T011 [US1] Implement `ItemTokenizer` ABC, registry (`register_item_tokenizer`, `get_item_tokenizer`), SID token builder helper (`_build_sid_map`), and default `generate()` method in `saegenrec/modeling/tokenizers/base.py`
- [X] T012 [US1] Implement collision resolution strategies (`resolve_collisions` dispatcher, `_append_level_resolve`, `_sinkhorn_resolve`) in `saegenrec/modeling/tokenizers/collision.py`
- [X] T013 [US1] Implement `RQVAEModel` PyTorch Lightning Module (MLP encoder, `ResidualVectorQuantizer` with per-layer codebooks, MLP decoder, training/validation steps with recon+quant loss, codebook utilization logging) in `saegenrec/modeling/tokenizers/models/rqvae_model.py`
- [X] T014 [US1] Implement `RQVAETokenizer` (train with PL Trainer, encode via forward pass, save/load checkpoint) in `saegenrec/modeling/tokenizers/rqvae.py`
- [X] T015 [US1] Implement `RQKMeansTokenizer` (FAISS KMeans residual clustering, optional `KMeansConstrained`, save/load centroids) in `saegenrec/modeling/tokenizers/rqkmeans.py`

**Checkpoint**: `ItemTokenizer` 子系统完整可用。可对合成 embedding 数据运行 `RQVAETokenizer` 和 `RQKMeansTokenizer`，输出唯一 SID map。

---

## Phase 3: User Story 2 — SFT 数据构建 (Priority: P1)

**Goal**: 将序列推荐数据 + SID 映射转换为 Alpaca 格式的 SFT 指令微调数据，支持 SeqRec、Item2Index、Index2Item 三种任务类型，每种 ≥5 个 prompt 模板。

**Independent Test**: 给定 mock `item_sid_map` 和 `train_sequences`，运行 SFTDatasetBuilder，验证输出的指令数据格式正确、每种任务类型都有数据。

**Dependencies**: 可与 US1 并行实现（代码无交叉）；运行时依赖 US1 的 `item_sid_map` 输出。

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T016 [P] [US2] Unit test for SFTTaskBuilder ABC contract (registry, template loading from YAML) in `tests/unit/modeling/sft/test_base.py`
- [X] T017 [P] [US2] Unit test for SeqRecTaskBuilder (history truncation, SID substitution, template randomization, skip user with single interaction and log warning) in `tests/unit/modeling/sft/test_seqrec.py`
- [X] T018 [P] [US2] Unit test for Item2IndexTaskBuilder and Index2ItemTaskBuilder (all items covered, SID ↔ title mapping) in `tests/unit/modeling/sft/test_item2index.py`
- [X] T019 [P] [US2] Unit test for SFTDatasetBuilder orchestrator (multi-task merge, task_weights sampling, output schema validation) in `tests/unit/modeling/sft/test_builder.py`

### Implementation for User Story 2

- [X] T020 [US2] Create SFT prompt template file with ≥5 templates per task type (seqrec, item2index, index2item) in `configs/templates/sft_prompts.yaml`
- [X] T021 [US2] Implement `SFTTaskBuilder` ABC, registry (`register_sft_task`, `get_sft_task_builder`), default `load_templates()` method in `saegenrec/modeling/sft/base.py`
- [X] T022 [P] [US2] Implement `SeqRecTaskBuilder` (load train_sequences, map item_ids to SIDs, truncate history to max_history_len, random template fill) in `saegenrec/modeling/sft/seqrec.py`
- [X] T023 [P] [US2] Implement `Item2IndexTaskBuilder` (load item_metadata, map each item to SID, generate title→SID records) in `saegenrec/modeling/sft/item2index.py`
- [X] T024 [P] [US2] Implement `Index2ItemTaskBuilder` (reverse of Item2Index: SID→title records) in `saegenrec/modeling/sft/index2item.py`
- [X] T025 [US2] Implement `SFTDatasetBuilder` orchestrator (iterate enabled tasks, call builders, apply task_weights, merge, shuffle, save as HF Dataset) in `saegenrec/modeling/sft/builder.py`

**Checkpoint**: `SFT` 子系统完整可用。可对 mock 数据运行 `SFTDatasetBuilder`，输出包含三种任务的混合 Alpaca 格式数据集。

---

## Phase 4: User Story 3 — 管道集成 (Priority: P2)

**Goal**: 一键运行 tokenization + SFT 数据构建，集成到现有数据管道中，提供 CLI 命令和 Makefile 目标。

**Independent Test**: 运行 `python -m saegenrec.dataset process <config.yaml> --step tokenize --step build-sft`，验证端到端输出到 `data/processed/` 正确。

**Dependencies**: 依赖 US1 和 US2 完成。

### Implementation for User Story 3

- [X] T026 [US3] Add `tokenize` and `build-sft` steps to `saegenrec/data/pipeline.py`: extend `ALL_STEPS` list, add prerequisite validation (embedding exists for tokenize, item_sid_map exists for build-sft), implement step logic calling `get_item_tokenizer().generate()` and `SFTDatasetBuilder().build()`, output to `data/processed/` via `OutputConfig.processed_path()`
- [X] T027 [US3] Add `tokenize` CLI command to `saegenrec/dataset.py`: load config, validate prerequisites (embedding exists), call `get_item_tokenizer().generate()`, support `--force` flag
- [X] T028 [US3] Add `build-sft` CLI command to `saegenrec/dataset.py`: load config, validate prerequisites (item_sid_map exists), call `SFTDatasetBuilder().build()`, support `--force` flag
- [X] T029 [P] [US3] Add `data-tokenize` and `data-build-sft` Makefile targets in `Makefile`
- [X] T030 [P] [US3] Update `configs/default.yaml` with `item_tokenizer` and `sft_builder` configuration sections (disabled by default)
- [X] T031 [US3] Integration test: run `--step tokenize --step build-sft` on synthetic data, verify `data/processed/` contains valid `item_sid_map/` and `sft_data/` HF Datasets in `tests/unit/modeling/test_pipeline_integration.py`

**Checkpoint**: 端到端管道可用。研究者可通过 `make data-tokenize && make data-build-sft` 或 `--step tokenize --step build-sft` 一键运行。

---

## Phase 5: User Story 4 — 模型接口定义 + 约束解码 (Priority: P3)

**Goal**: 定义 `GenRecModel` ABC（HuggingFace 风格），实现 SID Prefix Trie 和约束解码 LogitsProcessor，为后续 LLM 微调预留扩展点。

**Independent Test**: 构建 SIDTrie → 验证前缀搜索返回正确候选；SIDConstrainedLogitsProcessor 屏蔽无效 token。

**Dependencies**: SIDTrie 依赖 `item_sid_map` 的 schema（T005），但实现上独立。

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T032 [P] [US4] Unit test for SIDTrie (insert, search_prefix, from_sid_map factory, empty trie edge case) in `tests/unit/modeling/decoding/test_trie.py`
- [X] T033 [P] [US4] Unit test for SIDConstrainedLogitsProcessor (valid tokens not masked, invalid tokens set to -inf, batch support) in `tests/unit/modeling/decoding/test_constrained.py`
- [X] T034 [P] [US4] Unit test for GenRecModel ABC (registry, abstract method enforcement, GenRecConfig dataclass) in `tests/unit/modeling/genrec/test_base.py`

### Implementation for User Story 4

- [X] T035 [P] [US4] Implement `GenRecConfig` dataclass in `saegenrec/modeling/genrec/config.py`
- [X] T036 [P] [US4] Implement `GenRecModel` ABC + registry (`register_genrec_model`, `get_genrec_model`) with `train`, `generate`, `evaluate`, `save_pretrained`, `from_pretrained` method signatures in `saegenrec/modeling/genrec/base.py`
- [X] T037 [P] [US4] Implement `SIDTrie` (nested dict trie, `insert`, `search_prefix`, `from_sid_map` class method) in `saegenrec/modeling/decoding/trie.py`
- [X] T038 [US4] Implement `SIDConstrainedLogitsProcessor` (extends `transformers.LogitsProcessor`, uses SIDTrie to mask invalid tokens during generation) in `saegenrec/modeling/decoding/constrained.py`

**Checkpoint**: 模型接口和约束解码就绪。GenRecModel 可被继承注册；SIDTrie + LogitsProcessor 可从 `item_sid_map` 构建并约束 LLM 生成。

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 最终验证、清理和文档

- [X] T039 [P] Run all tests via `pytest tests/unit/modeling/` — ensure 100% pass
- [X] T040 [P] Run `ruff check` and `ruff format` on all new files
- [X] T041 Run quickstart.md validation: execute the complete flow on a small synthetic dataset (create embeddings → tokenize → build-sft → verify outputs)
- [X] T042 Review `saegenrec/modeling/__init__.py` files: add convenience re-exports for public API (`ItemTokenizer`, `SFTDatasetBuilder`, `GenRecModel`, `SIDTrie`)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **US1 Tokenization (Phase 2)**: Depends on Setup; no other story dependencies
- **US2 SFT (Phase 3)**: Depends on Setup; can be implemented in **parallel** with US1 (code in separate files)
- **US3 Pipeline (Phase 4)**: Depends on **both** US1 and US2 completion
- **US4 Model+Decoding (Phase 5)**: Depends on Setup; can be implemented in **parallel** with US1/US2
- **Polish (Phase 6)**: Depends on all phases complete

### User Story Dependencies

```text
         Setup (Phase 1)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  US1(P1)   US2(P1)   US4(P3)    ← 三者可并行
    │         │         │
    └────┬────┘         │
         ▼              │
       US3(P2)          │
         │              │
         └──────┬───────┘
                ▼
           Polish (Phase 6)
```

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- ABC/registry before concrete implementations
- Concrete implementations (marked [P]) can run in parallel
- Orchestrator/integration after all components ready

### Parallel Opportunities

- **Phase 1**: T004, T005, T006 can run in parallel
- **Phase 2 (US1)**: T007–T010 tests in parallel; after T011, T014 and T015 can run in parallel
- **Phase 3 (US2)**: T016–T019 tests in parallel; after T021, T022/T023/T024 in parallel
- **Phase 2+3+5**: US1, US2, US4 can be developed in parallel after Setup
- **Phase 5 (US4)**: T032–T034 tests in parallel; T035/T036/T037 in parallel

---

## Parallel Example: US1 + US2 Concurrent

```text
# After Phase 1 Setup completes, launch in parallel:

# Stream A: US1 Tokenization
Task T007-T010: Write all tokenizer tests (parallel)
Task T011: ItemTokenizer ABC
Task T012: Collision resolution
Task T013: RQ-VAE Lightning module
Task T014-T015: RQVAETokenizer + RQKMeansTokenizer (parallel)

# Stream B: US2 SFT (same time as Stream A)
Task T016-T019: Write all SFT tests (parallel)
Task T020: Prompt templates YAML
Task T021: SFTTaskBuilder ABC
Task T022-T024: SeqRec + Item2Index + Index2Item (parallel)
Task T025: SFTDatasetBuilder orchestrator

# After both streams complete:
Task T026-T030: US3 Pipeline Integration
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: US1 — ItemTokenizer
3. **STOP and VALIDATE**: 用合成 embedding 运行 tokenize，验证 SID 唯一性和码本利用率
4. 输出可用 `item_sid_map/` 即为 MVP

### Incremental Delivery

1. Setup → Foundation ready
2. US1 (Tokenization) → 验证 SID map 正确 → **MVP!**
3. US2 (SFT) → 验证 SFT 数据格式正确 → 可开始 LLM 微调实验
4. US3 (Pipeline) → 一键端到端 → 研究者工作流完整
5. US4 (Model+Decoding) → 为后续 LLM 训练/推理预留接口
6. Polish → 全面测试 + 文档验证

---

## Summary

| Metric | Value |
|--------|-------|
| Total tasks | 42 |
| Phase 1 (Setup) | 6 tasks |
| Phase 2 (US1 Tokenization) | 9 tasks (4 tests + 5 impl) |
| Phase 3 (US2 SFT) | 10 tasks (4 tests + 6 impl) |
| Phase 4 (US3 Pipeline) | 6 tasks (1 integration test + 5 impl) |
| Phase 5 (US4 Model+Decoding) | 7 tasks (3 tests + 4 impl) |
| Phase 6 (Polish) | 4 tasks |
| Parallel opportunities | US1/US2/US4 fully parallelizable after Setup |
| MVP scope | Setup + US1 (15 tasks) |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Constitution III 要求测试先行：先写测试确认 FAIL，再实现代码
- 所有输出到 `data/processed/`（CCDS "最终建模用数据集"）
- `faiss-cpu` 和 `k-means-constrained` 均为新增必需依赖
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
