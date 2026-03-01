# Tasks: 生成式推荐数据处理流水线

**Input**: Design documents from `/specs/001-genrec-data-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: SC-007 要求所有核心模块单元测试覆盖率 ≥80%，因此每个 User Story 均包含测试任务。

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Source**: `saegenrec/data/` (数据处理子包)
- **Tests**: `tests/unit/data/`
- **Configs**: `configs/`
- **Scripts**: `scripts/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目初始化、依赖管理、目录结构创建

- [x] T001 Add new dependencies (datasets, pyyaml, torch, transformers, sentence-transformers, requests) to pyproject.toml
- [x] T002 Create data processing package directory structure: saegenrec/data/__init__.py, saegenrec/data/loaders/__init__.py, saegenrec/data/processors/__init__.py, saegenrec/data/tokenizers/__init__.py, saegenrec/data/embeddings/__init__.py
- [x] T003 [P] Create configs/ directory with configs/default.yaml and configs/examples/ subdirectory
- [x] T004 [P] Create scripts/ directory with scripts/run_pipeline.sh
- [x] T005 [P] Create test directory structure: tests/unit/data/ with tests/unit/__init__.py and tests/unit/data/__init__.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 核心抽象接口、配置系统、HF Schema 定义 — 所有 User Story 的共同基础

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Implement pipeline configuration dataclasses (PipelineConfig, DatasetConfig, ProcessingConfig, TokenizerConfig, EmbeddingConfig, OutputConfig) with YAML loading in saegenrec/data/config.py — per contracts/pipeline-config.md
- [x] T007 [P] Implement DatasetLoader abstract base class with registry (LOADER_REGISTRY, register_loader decorator, get_loader function) in saegenrec/data/loaders/base.py — per contracts/dataset-loader.md
- [x] T008 [P] Implement ItemTokenizer abstract base class with registry (TOKENIZER_REGISTRY, register_tokenizer decorator, get_tokenizer function) in saegenrec/data/tokenizers/base.py — per contracts/item-tokenizer.md
- [x] T009 [P] Define HuggingFace Dataset Feature schemas (INTERACTIONS_FEATURES, USER_SEQUENCES_FEATURES, ITEM_METADATA_FEATURES, ID_MAP_FEATURES, TRAINING_SAMPLE_FEATURES, TEXT_EMBEDDING_FEATURES) in saegenrec/data/schemas.py — per contracts/data-schemas.md
- [x] T010 [P] Create shared test fixtures (synthetic interaction records, item metadata, user sequences) in tests/unit/data/conftest.py and tests/conftest.py
- [x] T011 [P] Write unit tests for config loading, validation, and defaults in tests/unit/data/test_config.py

**Checkpoint**: Foundation ready — 抽象接口、配置系统、schema 定义就绪，user story 实现可以开始

---

## Phase 3: User Story 1 — 原始数据清洗与序列构建 (Priority: P1) 🎯 MVP

**Goal**: 将 Amazon 原始数据加载、K-core 过滤、构建用户交互序列、生成 ID 映射，输出到 data/interim/

**Independent Test**: 提供小规模合成 Amazon 数据，运行后验证 interim 输出的格式、排序、过滤效果

### Tests for User Story 1

- [x] T012 [P] [US1] Write unit tests for Amazon2015Loader (field mapping, dedup, missing fields, empty file) in tests/unit/data/test_loaders.py
- [x] T013 [P] [US1] Write unit tests for Amazon2023Loader (JSONL parsing, timestamp ms→s conversion, field mapping) in tests/unit/data/test_loaders.py
- [x] T014 [P] [US1] Write unit tests for K-core filter (convergence, threshold boundary, empty result warning) in tests/unit/data/test_kcore.py
- [x] T015 [P] [US1] Write unit tests for sequence builder (time ordering, ID mapping bijectivity, review field preservation) in tests/unit/data/test_sequence.py

### Implementation for User Story 1

- [x] T016 [P] [US1] Implement Amazon2015Loader (load_interactions, load_item_metadata with field mapping per contracts/dataset-loader.md) in saegenrec/data/loaders/amazon2015.py
- [x] T017 [P] [US1] Implement Amazon2023Loader (load_interactions, load_item_metadata with field mapping per contracts/dataset-loader.md) in saegenrec/data/loaders/amazon2023.py
- [x] T018 [US1] Implement K-core iterative filter (pandas groupby, convergence loop, statistics output) in saegenrec/data/processors/kcore.py
- [x] T019 [US1] Implement sequence builder (dedup, time-sort, user/item ID mapping, UserSequence Dataset construction) in saegenrec/data/processors/sequence.py
- [x] T020 [US1] Implement interim data persistence (save interactions, user_sequences, item_metadata, id_maps, stats.json to data/interim/) in saegenrec/data/processors/sequence.py

**Checkpoint**: US1 完成 — data/interim/ 中可生成完整的中间数据（交互序列 + ID 映射 + 元数据 + 统计），可独立验证

---

## Phase 4: User Story 2 — 序列数据划分 (Priority: P1)

**Goal**: 将中间数据中的用户交互序列按 LOO 或 TO 策略划分为 train/valid/test

**Independent Test**: 使用 US1 产出的中间数据，分别运行 LOO 和 TO 划分，验证划分正确性和零泄露

### Tests for User Story 2

- [x] T021 [P] [US2] Write unit tests for LOO split (last→test, second-last→valid, rest→train, exclude <3 interactions) in tests/unit/data/test_split.py
- [x] T022 [P] [US2] Write unit tests for TO split (global time ordering, ratio correctness, zero leakage) in tests/unit/data/test_split.py

### Implementation for User Story 2

- [x] T023 [US2] Implement LOO split strategy (per-user last/second-last extraction, exclude users with <3 interactions) in saegenrec/data/processors/split.py
- [x] T024 [US2] Implement TO split strategy (global timestamp sort, ratio-based partition) in saegenrec/data/processors/split.py
- [x] T025 [US2] Implement unified split_data() entry point (strategy dispatch, statistics output, persistence) in saegenrec/data/processors/split.py

**Checkpoint**: US2 完成 — train/valid/test UserSequence Datasets 可正确生成，零泄露验证通过

---

## Phase 5: User Story 3 — 滑动窗口数据增强 (Priority: P1)

**Goal**: 对训练集执行滑动窗口生成 (history, target) 样本对，支持可配置 max_seq_len

**Independent Test**: 输入已划分的训练序列，验证样本数 = sum(len-1)，历史长度 ≤ max_seq_len

### Tests for User Story 3

- [x] T026 [P] [US3] Write unit tests for sliding window (sample count = N-1, truncation at max_seq_len, min sequence length=2, boundary cases) in tests/unit/data/test_augment.py

### Implementation for User Story 3

- [x] T027 [US3] Implement sliding window augmentation (per-user window generation, left truncation, batch Dataset construction) in saegenrec/data/processors/augment.py
- [x] T028 [US3] Implement validation/test set conversion to TrainingSample format (no augmentation, direct history+target extraction) in saegenrec/data/processors/augment.py

**Checkpoint**: US3 完成 — 训练集增强后的 TrainingSample Dataset 可生成，验证/测试集转换正确

---

## Phase 6: User Story 4 — 生成 LLM 训练数据 (Priority: P2)

**Goal**: 将增强后的数据转换为最终训练格式（含 item tokens + text info），输出到 data/processed/

**Independent Test**: 使用 PassthroughTokenizer 运行最终数据生成，验证输出格式含 token 和 text 信息

### Tests for User Story 4

- [x] T029 [P] [US4] Write unit tests for PassthroughTokenizer (tokenize/detokenize round-trip, vocab_size, token_length) in tests/unit/data/test_tokenizers.py
- [x] T030 [P] [US4] Write unit tests for final data generator (TrainingSample schema compliance, item text + token fields, HF Dataset format) in tests/unit/data/test_final.py
- [x] T031 [P] [US4] Write unit tests for text embedding generator (field concatenation, output shape, batch processing) in tests/unit/data/test_embeddings.py

### Implementation for User Story 4

- [x] T032 [P] [US4] Implement PassthroughTokenizer in saegenrec/data/tokenizers/passthrough.py — per contracts/item-tokenizer.md
- [x] T033 [US4] Implement final training data generator (attach item tokens + item titles to TrainingSample, save to data/processed/) in saegenrec/data/processors/final.py
- [x] T034 [US4] Implement text embedding generator (sentence-transformers model loading, field concatenation, batch encode, save to data/interim/) in saegenrec/data/embeddings/text.py

**Checkpoint**: US4 完成 — data/processed/ 中生成完整的 HF Dataset 格式训练数据，含 item tokens 和 text info

---

## Phase 7: User Story 5 — 商品图片下载 (Priority: P3)

**Goal**: 批量下载商品图片到 data/external/images/，支持断点续传和错误跳过

**Independent Test**: 提供含图片 URL 的小规模元数据，运行后验证图片正确保存、失败记录、断点续传

### Tests for User Story 5

- [x] T035 [P] [US5] Write unit tests for image downloader (successful download, URL failure skip, resume existing, error logging) in tests/unit/data/test_images.py

### Implementation for User Story 5

- [x] T036 [US5] Implement image downloader (requests-based download, item_id filename, skip existing, error logging, progress bar) in saegenrec/data/processors/images.py

**Checkpoint**: US5 完成 — 图片下载功能可独立运行

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 流水线编排、CLI 入口、配置文件、文档

- [x] T037 Implement pipeline orchestrator (step sequencing: load→filter→sequence→split→augment→generate, optional embed/download, step selection via --step flag) in saegenrec/data/pipeline.py
- [x] T038 Refactor CLI entry point to integrate pipeline orchestrator (typer command: process with config path and --step options) in saegenrec/dataset.py
- [x] T039 [P] Create default YAML config file in configs/default.yaml — per contracts/pipeline-config.md
- [x] T040 [P] Create example config for Amazon2015 Baby in configs/examples/amazon2015_baby.yaml
- [x] T041 [P] Create example config for Amazon2023 All_Beauty in configs/examples/amazon2023_beauty.yaml
- [x] T042 [P] Create pipeline run script in scripts/run_pipeline.sh
- [x] T043 Implement statistics output (stats.json generation for interim and processed stages, log summary) in saegenrec/data/pipeline.py
- [x] T044 Add pipeline Makefile targets (data-process, data-embed, data-download-images) to Makefile
- [x] T045 Run ruff format and ruff check on all new files, fix any linting issues
- [x] T046 Run full test suite (pytest tests/unit/data/ -v --cov=saegenrec/data --cov-report=term-missing) and verify ≥80% coverage
- [x] T047 Validate quickstart.md workflow end-to-end with Amazon2015 Baby dataset

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — No dependencies on other stories
- **US2 (Phase 4)**: Depends on US1 (Phase 3) — needs user_sequences from data/interim/
- **US3 (Phase 5)**: Depends on US2 (Phase 4) — needs split train/valid/test sequences
- **US4 (Phase 6)**: Depends on US3 (Phase 5) — needs augmented TrainingSample data; ItemTokenizer interface from Phase 2
- **US5 (Phase 7)**: Depends on Foundational (Phase 2) only — independent of US1-US4 (uses raw metadata directly)
- **Polish (Phase 8)**: Depends on US1-US4 completion (US5 optional)

### User Story Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational) ──────────────────────┐
    │                                         │
    ▼                                         ▼
Phase 3 (US1: Load+Filter+Sequence)    Phase 7 (US5: Images) [independent]
    │
    ▼
Phase 4 (US2: Split)
    │
    ▼
Phase 5 (US3: Sliding Window)
    │
    ▼
Phase 6 (US4: Final Generation)
    │
    ▼
Phase 8 (Polish)
```

### Within Each User Story

- Tests MUST be written FIRST, ensure they FAIL before implementation
- Abstract interfaces before concrete implementations
- Core logic before persistence/IO
- Story complete before moving to next priority

### Parallel Opportunities

- Phase 1: T003, T004, T005 can run in parallel
- Phase 2: T007, T008, T009, T010, T011 can all run in parallel
- Phase 3: T012-T015 (all tests) in parallel; T016, T017 (both loaders) in parallel
- Phase 4: T021, T022 (LOO/TO tests) in parallel
- Phase 5: T026 can start while Phase 4 implementation completes
- Phase 6: T029, T030, T031 (all tests) in parallel; T032 in parallel with T034
- Phase 7: Can run entirely in parallel with Phases 3-6 (after Phase 2)
- Phase 8: T039, T040, T041, T042 can all run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests in parallel (should FAIL initially):
Task: "T012 [P] [US1] Write unit tests for Amazon2015Loader in tests/unit/data/test_loaders.py"
Task: "T013 [P] [US1] Write unit tests for Amazon2023Loader in tests/unit/data/test_loaders.py"
Task: "T014 [P] [US1] Write unit tests for K-core filter in tests/unit/data/test_kcore.py"
Task: "T015 [P] [US1] Write unit tests for sequence builder in tests/unit/data/test_sequence.py"

# Then launch both loaders in parallel:
Task: "T016 [P] [US1] Implement Amazon2015Loader in saegenrec/data/loaders/amazon2015.py"
Task: "T017 [P] [US1] Implement Amazon2023Loader in saegenrec/data/loaders/amazon2023.py"

# Sequential (depends on loaders):
Task: "T018 [US1] Implement K-core filter in saegenrec/data/processors/kcore.py"
Task: "T019 [US1] Implement sequence builder in saegenrec/data/processors/sequence.py"
Task: "T020 [US1] Implement interim data persistence in saegenrec/data/processors/sequence.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T011)
3. Complete Phase 3: User Story 1 (T012-T020)
4. **STOP and VALIDATE**: 用小规模 Amazon2015 Baby 数据验证 data/interim/ 输出
5. 确认中间数据格式正确后继续

### Incremental Delivery

1. Setup + Foundational → 基础设施就绪
2. US1 (Load+Filter+Sequence) → 验证 interim 数据 → **MVP!**
3. US2 (Split) → 验证 LOO/TO 划分正确性
4. US3 (Sliding Window) → 验证增强样本数量和格式
5. US4 (Final Generation) → 验证 processed 输出 → **核心流水线完成!**
6. US5 (Images) → 多模态扩展（可选）
7. Polish → CLI 集成、配置文件、Makefile targets

### Sequential Implementation (Single Developer)

按 Phase 顺序执行：1 → 2 → 3 → 4 → 5 → 6 → (7 optional) → 8

每个 Phase 完成后运行对应测试验证，确保前一阶段稳定后再开始下一阶段。

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Write tests first, verify they FAIL, then implement
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- US2→US3→US4 是严格顺序依赖链，US5 可与 US1-US4 并行
