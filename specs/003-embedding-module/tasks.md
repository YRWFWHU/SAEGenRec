# Tasks: Embedding 模块（语义 + 协同）

**Input**: Design documents from `/specs/003-embedding-module/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md

**Tests**: 包含测试任务（Constitution 原则 III 要求每个核心模块有单元测试）。

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 创建包结构、新增依赖、定义 schema

- [X] T001 Create embedding module package structure: `saegenrec/data/embeddings/__init__.py`, `saegenrec/data/embeddings/semantic/__init__.py`, `saegenrec/data/embeddings/semantic/base.py`, `saegenrec/data/embeddings/collaborative/__init__.py`, `saegenrec/data/embeddings/collaborative/base.py`, `saegenrec/data/embeddings/collaborative/models/__init__.py`
- [X] T002 [P] Add `pytorch-lightning` dependency to `pyproject.toml`
- [X] T003 [P] Add `SEMANTIC_EMBEDDING_FEATURES` and `COLLABORATIVE_EMBEDDING_FEATURES` schemas to `saegenrec/data/schemas.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 两套 ABC + 注册表 + 配置 dataclass — 所有 User Story 的基础

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 [P] Implement SemanticEmbedder ABC with `SEMANTIC_EMBEDDER_REGISTRY`, `@register_semantic_embedder`, and `get_semantic_embedder` in `saegenrec/data/embeddings/semantic/base.py`
- [X] T005 [P] Implement CollaborativeEmbedder ABC with `COLLABORATIVE_EMBEDDER_REGISTRY`, `@register_collaborative_embedder`, and `get_collaborative_embedder` in `saegenrec/data/embeddings/collaborative/base.py`
- [X] T006 Create top-level `saegenrec/data/embeddings/__init__.py` exporting both ABCs and registries from semantic and collaborative subpackages
- [X] T007 [P] Add `SemanticEmbeddingConfig` and `CollaborativeEmbeddingConfig` dataclasses to `saegenrec/data/config.py`
- [X] T008 Update `PipelineConfig` and `load_config()` in `saegenrec/data/config.py` to include `semantic_embedding` and `collaborative_embedding` sections, keeping old `embedding` section as deprecated
- [X] T009 [P] Unit test for SemanticEmbedder ABC + registry (register, get, unknown name error) in `tests/unit/data/embeddings/test_semantic_base.py`
- [X] T010 [P] Unit test for CollaborativeEmbedder ABC + registry (register, get, unknown name error) in `tests/unit/data/embeddings/test_collaborative_base.py`

**Checkpoint**: Foundation ready — both ABCs, registries, and configs are in place

---

## Phase 3: User Story 1 — 语义 Embedding 生成 (Priority: P1) 🎯 MVP

**Goal**: 研究者可对 Stage 1 数据运行语义 embedding 生成，使用 sentence-transformers 模型对物品元数据文本提取语义向量并保存为 HuggingFace Dataset

**Independent Test**: 用小规模 Stage 1 中间数据运行 `SentenceTransformerEmbedder.generate()`，验证输出 Dataset 的物品数和向量维度正确

### Implementation for User Story 1

- [X] T011 [US1] Implement `SentenceTransformerEmbedder` in `saegenrec/data/embeddings/semantic/sentence_transformer.py` — migrate logic from `saegenrec/data/embeddings/text.py`, register as `"sentence-transformer"`, support: text field concatenation with price-to-text conversion, configurable L2 normalization (default off), zero vector for all-empty items, skip items missing from item_metadata with warning, skip-if-exists / --force, output stats (items, dim, elapsed)
- [X] T012 [US1] Export `SentenceTransformerEmbedder` in `saegenrec/data/embeddings/semantic/__init__.py` via noqa import to trigger registration
- [X] T013 [US1] Delete old `saegenrec/data/embeddings/text.py`
- [X] T014 [US1] Unit test for `SentenceTransformerEmbedder` in `tests/unit/data/embeddings/test_sentence_transformer.py` — test: normal generation, missing item skip + warning, all-empty-text zero vector, price numeric-to-text, normalize on/off, skip-if-exists behavior

**Checkpoint**: 语义 embedding 生成功能完整可用

---

## Phase 4: User Story 2 — 协同 Embedding 生成 (Priority: P1)

**Goal**: 研究者可对 Stage 2 划分数据训练 SASRec 模型并提取物品协同 embedding，训练中输出评估指标

**Independent Test**: 用小规模 Stage 2 数据运行 `SASRecEmbedder.generate()`，验证训练完成、指标输出、embedding Dataset 维度与配置一致

### Implementation for User Story 2

- [X] T015 [P] [US2] Implement Hit Rate@K and NDCG@K metrics in `saegenrec/data/embeddings/collaborative/models/metrics.py` — GPU-accelerated full-ranking evaluation (reference RecBole/MiniOneRec)
- [X] T016 [P] [US2] Implement SASRec model as `nn.Module` in `saegenrec/data/embeddings/collaborative/models/sasrec_model.py` — reference RecBole: item_embedding (padding_idx=0), position_embedding, SASRecBlock (MultiHeadAttention + FFN + LayerNorm + Dropout), causal mask, BPR loss, tied weights for prediction
- [X] T017 [US2] Implement `SASRecEmbedder` as `CollaborativeEmbedder` in `saegenrec/data/embeddings/collaborative/sasrec.py` — register as `"sasrec"`, PyTorch Lightning LightningModule wrapping SASRec model, DataModule for loading train/valid/test sequences, epoch-end evaluation callback logging HR@K and NDCG@K, extract embedding from `item_embedding.weight.data[1:]` after training, save as HuggingFace Dataset, skip-if-exists / --force, output stats, seed management for reproducibility
- [X] T018 [US2] Export `SASRecEmbedder` in `saegenrec/data/embeddings/collaborative/__init__.py` via noqa import to trigger registration
- [X] T019 [P] [US2] Unit test for metrics in `tests/unit/data/embeddings/test_metrics.py` — verify HR@K and NDCG@K correctness with known rankings
- [X] T020 [US2] Unit test for `SASRecEmbedder` in `tests/unit/data/embeddings/test_sasrec_embedder.py` — test: model forward shape, BPR loss computation, embedding extraction shape, generate() end-to-end with synthetic data (small num_epochs), skip-if-exists behavior

**Checkpoint**: 协同 embedding 生成功能完整可用

---

## Phase 5: User Story 3 — 流水线集成与 CLI 独立调用 (Priority: P2)

**Goal**: 研究者可通过流水线 embed 步骤或独立 CLI 命令调用 embedding 生成

**Independent Test**: 分别通过 `--step embed` 和独立 CLI 命令运行 embedding 生成，验证两种方式均可正常工作

### Implementation for User Story 3

- [X] T021 [US3] Add `embed-semantic` CLI command in `saegenrec/dataset.py` — args: config (required), --force, --model-name, --device; load config, call SemanticEmbedder.generate(), handle errors (Stage 1 missing → exit 1, exists + no force → skip)
- [X] T022 [US3] Add `embed-collaborative` CLI command in `saegenrec/dataset.py` — args: config (required), --force, --device, --num-epochs; load config, call CollaborativeEmbedder.generate(), handle errors (Stage 2 missing → exit 1, GPU unavailable → CPU fallback warning)
- [X] T023 [US3] Update `embed` step in `saegenrec/data/pipeline.py` — replace old `text_embeddings` logic with new embedder system: call SemanticEmbedder if `semantic_embedding.enabled`, call CollaborativeEmbedder if `collaborative_embedding.enabled`, pass --force flag, add prerequisite check (embed depends on split for collaborative)
- [X] T024 [US3] Add `--force` flag to `process` command in `saegenrec/dataset.py` and thread it through to pipeline embed step
- [X] T025 [US3] Update `configs/default.yaml` with `semantic_embedding` and `collaborative_embedding` sections per contracts/cli.md

**Checkpoint**: 流水线和独立 CLI 两种调用方式均可工作

---

## Phase 6: User Story 4 — 自定义 Embedder 扩展 (Priority: P3)

**Goal**: 研究者通过继承 ABC + 注册装饰器即可扩展新的 embedder，无需修改核心代码

**Independent Test**: 实现并注册一个 mock SemanticEmbedder，通过配置指定后验证系统正确调用

### Implementation for User Story 4

- [X] T026 [US4] Verify extensibility: add a docstring example in `saegenrec/data/embeddings/semantic/base.py` and `saegenrec/data/embeddings/collaborative/base.py` showing custom embedder registration pattern
- [X] T027 [US4] Verify error message for unknown embedder name: ensure `get_semantic_embedder` and `get_collaborative_embedder` raise `ValueError` listing all available registered names (already tested in T009/T010, verify wording matches spec)

**Checkpoint**: 可扩展性验证完成

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 向后兼容、解耦验证、文档

- [X] T028 [P] Add deprecation handling in `saegenrec/data/config.py`: if old `embedding.enabled=true` but `semantic_embedding` not configured, auto-migrate to semantic_embedding config with deprecation warning
- [X] T029 [P] Verify zero coupling: ensure no imports between `saegenrec/data/embeddings/` and `saegenrec/data/tokenizers/` — can be a simple grep check or test assertion
- [ ] T030 Run `specs/003-embedding-module/quickstart.md` validation end-to-end on Amazon Baby data

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2). No dependencies on other stories.
- **US2 (Phase 4)**: Depends on Foundational (Phase 2). No dependencies on other stories. Can run in parallel with US1.
- **US3 (Phase 5)**: Depends on US1 AND US2 completion (integrates both embedder types into CLI/pipeline)
- **US4 (Phase 6)**: Depends on Foundational (Phase 2). Can run after Phase 2 but logically verified after US1/US2.
- **Polish (Phase 7)**: Depends on all prior phases

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 — No dependencies on other stories
- **User Story 2 (P1)**: Can start after Phase 2 — No dependencies on other stories, can run in parallel with US1
- **User Story 3 (P2)**: Depends on US1 AND US2 completion — integrates both subsystems
- **User Story 4 (P3)**: Depends on Phase 2 — extensibility is inherent in ABC design

### Within Each User Story

- Implementation before tests (tests verify the implementation)
- Models before embedder (SASRec nn.Module before SASRecEmbedder)
- Core logic before export/registration wiring

### Parallel Opportunities

- T002 and T003 in parallel (different files)
- T004 and T005 in parallel (different subpackages)
- T007, T009, T010 in parallel with each other
- T015 and T016 in parallel (metrics and model are independent files)
- **US1 and US2 entire phases can run in parallel** after Phase 2

---

## Parallel Example: User Story 2

```text
# Launch metrics and model in parallel (different files, no dependencies):
T015: "Implement HR@K and NDCG@K metrics in collaborative/models/metrics.py"
T016: "Implement SASRec model in collaborative/models/sasrec_model.py"

# Then sequentially:
T017: "Implement SASRecEmbedder in collaborative/sasrec.py" (depends on T015, T016)
T018: "Export SASRecEmbedder in collaborative/__init__.py"
T019: "Unit test for metrics" (can parallel with T017 after T015)
T020: "Unit test for SASRecEmbedder" (depends on T017)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1 (语义 Embedding)
4. **STOP and VALIDATE**: Test `embed-semantic` independently
5. Semantic embedding is immediately usable by downstream tokenizers

### Incremental Delivery

1. Setup + Foundational → ABC + registries ready
2. Add User Story 1 → 语义 embedding 可独立使用 (MVP!)
3. Add User Story 2 → 协同 embedding 可独立使用
4. Add User Story 3 → 流水线集成 + CLI 命令
5. Add User Story 4 → 可扩展性验证
6. Polish → 向后兼容 + 解耦验证

### Parallel Strategy

US1 和 US2 相互独立，完成 Phase 2 后可并行开发：

1. 完成 Setup + Foundational
2. 并行启动:
   - Path A: US1 (语义 embedder — 编码型，较简单)
   - Path B: US2 (协同 embedder — 训练型，较复杂)
3. 两者完成后 → US3 (集成)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 和 US2 均为 P1 优先级但相互独立，推荐先完成 US1（更简单）再做 US2
- 协同 embedder (US2) 是本 feature 的最复杂部分：SASRec 模型 + Lightning 训练 + 评估指标
- 旧 `text.py` 必须在 US1 完成后删除（T013），确保迁移无遗漏
- 所有 embedding 输出使用 `SEMANTIC_EMBEDDING_FEATURES` / `COLLABORATIVE_EMBEDDING_FEATURES` schema
