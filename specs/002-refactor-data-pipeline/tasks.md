# Tasks: 重构数据处理管道 — 解耦 Tokenizer 并新增负采样

**Input**: Design documents from `/specs/002-refactor-data-pipeline/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: 配置扩展和 schema 定义 — 所有 User Story 的共享基础设施

**⚠️ CRITICAL**: 所有 User Story 的实现依赖此阶段完成

- [x] T001 [P] 扩展 `ProcessingConfig`，新增 `num_negatives` 和 `seed` 字段 — `saegenrec/data/config.py`
  - 新增 `num_negatives: int = 99` 和 `seed: int | None = 42`
  - 保留现有 `__post_init__` 验证逻辑不变（`split_ratio` 仅 TO 时校验）
  - 新增验证：`num_negatives >= 1`

- [x] T002 [P] 新增 `INTERIM_SAMPLE_FEATURES` 和 `NEGATIVE_SAMPLE_FEATURES` — `saegenrec/data/schemas.py`
  - `INTERIM_SAMPLE_FEATURES`: user_id, history_item_ids, history_item_titles, target_item_id, target_item_title
  - `NEGATIVE_SAMPLE_FEATURES`: 继承上述 + negative_item_ids, negative_item_titles
  - 保留所有现有 Features 定义不变

- [x] T003 [P] 更新默认配置和示例配置 — `configs/default.yaml`, `configs/examples/*.yaml`
  - `default.yaml`: processing 节新增 `num_negatives: 99` 和 `seed: 42`
  - 各 examples 配置同步更新

**Checkpoint**: 配置与 schema 就绪，可开始各 User Story 的实现

---

## Phase 2: User Story 1 — 端到端数据预处理（无 Tokenizer）(Priority: P1) 🎯 MVP

**Goal**: 将 augment 与 tokenizer 解耦，重构管道为两阶段架构，所有输出不含 token 字段

**Independent Test**: 使用 Amazon 2015 Beauty 配置运行管道，验证 `data/interim/` 下产出完整序列数据且不包含 token 字段

### 测试（先写测试，确认失败后再实现）

- [x] T004 [P] [US1] 修改 augment 测试，移除 tokenizer 依赖 — `tests/unit/data/test_augment.py`
  - 移除 `passthrough_tokenizer` fixture
  - `sliding_window_augment` 和 `convert_eval_split` 调用中移除 `tokenizer` 参数
  - 断言输出 schema 匹配 `INTERIM_SAMPLE_FEATURES`（无 `*_tokens` 字段）
  - 保留所有现有测试用例的逻辑（sample_count, truncation, min_sequence 等）

- [x] T005 [P] [US1] 修改 pipeline 测试，适配两阶段架构 — `tests/unit/data/test_pipeline.py`
  - `TestGenerateFinalData` 适配新 schema（无 tokens）
  - `TestPipelineOrchestrator.test_full_pipeline` 验证两阶段输出目录结构
  - `test_selective_steps` 验证 `STAGE1_STEPS` 和 `STAGE2_STEPS` 常量
  - 验证阶段 1 输出到 `{interim_dir}/{dataset}/{category}/`
  - 验证阶段 2 输出到 `{interim_dir}/{dataset}/{category}/{split_strategy}/`
  - 新增 `test_empty_data_after_filter`: 验证 K-core 过滤后无数据时管道优雅终止，输出空数据集和警告日志（满足 Constitution III 空数据处理要求）

### 实现

- [x] T006 [US1] 重构 `sliding_window_augment` 和 `convert_eval_split`，移除 tokenizer — `saegenrec/data/processors/augment.py`
  - 移除 `tokenizer: ItemTokenizer` 参数
  - 移除 `tokenizer.tokenize_batch` 调用
  - 输出使用 `INTERIM_SAMPLE_FEATURES` schema
  - 保留 `item_titles` 参数用于填充文本字段
  - 移除对 `TRAINING_SAMPLE_FEATURES` 的导入（改用 `INTERIM_SAMPLE_FEATURES`）

- [x] T007 [US1] 重构 `run_pipeline` 为两阶段架构 — `saegenrec/data/pipeline.py`
  - 定义常量：`STAGE1_STEPS = ["load", "filter", "sequence"]`，`STAGE2_STEPS = ["split", "augment"]`（Phase 3 T013 追加 `negative_sampling`），`ALL_STEPS = STAGE1_STEPS + STAGE2_STEPS`，`LEGACY_STEPS = ["generate", "embed"]`
  - 阶段 1 输出路径: `config.output.interim_path(dataset, category)`
  - 阶段 2 输出路径: `config.output.interim_path(dataset, category) / split_strategy`
  - augment 步骤调用移除 tokenizer 参数
  - 阶段 2 的 split/augment 步骤从阶段 1 输出目录读取 `user_sequences`、`item_metadata` 等
  - 各阶段独立生成 `stats.json`
  - 保留 `generate` 和 `embed` 为 `LEGACY_STEPS`

- [x] T008 [US1] 更新 CLI `process` 命令，支持新参数覆盖 — `saegenrec/dataset.py`
  - 新增可选参数: `--dataset`, `--category`, `--kcore`, `--max-seq-len`, `--num-negatives`, `--split-strategy`, `--split-ratio`, `--seed`
  - 参数覆盖 YAML 配置中对应值
  - 更新 `--step` 帮助文本，列出所有有效步骤名

- [x] T009 [US1] 新增 Makefile 目标 `data-filter` 和 `data-split` — `Makefile`
  - `data-filter`: `$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) --step load --step filter --step sequence`
  - `data-split`: `$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) --step split --step augment`（Phase 3 T013 追加 `--step negative_sampling`）
  - `CONFIG ?= configs/default.yaml`
  - 保留现有 `data-process` 目标不变

- [x] T010 [US1] 更新 conftest fixture — `tests/unit/data/conftest.py`
  - 更新 `sample_config_path` fixture：YAML 中加入 `num_negatives` 和 `seed`

**Checkpoint**: 此时 User Story 1 应完全可运行：两阶段管道、无 tokenizer、输出结构正确

---

## Phase 3: User Story 2 — 重排序负采样 (Priority: P2)

**Goal**: 为每条样本自动采样负样本，支持可配置数量和可复现性

**Independent Test**: 运行负采样步骤，验证每条样本附带指定数量的负样本且不在用户交互历史中

### 测试（先写测试，确认失败后再实现）

- [x] T011 [P] [US2] 新建负采样单元测试 — `tests/unit/data/test_negative_sampling.py`
  - `TestBuildUserInteractedItems`: 验证从 UserSequence 构建 user→items 映射
  - `TestSampleNegatives`:
    - 验证每条样本产出 `num_negatives` 个负样本 ID
    - 验证负样本不在用户交互历史中
    - 验证 seed 可复现性（同 seed 两次运行结果一致）
    - 验证可用负样本不足时的降级行为（采样所有可用 + 警告）
    - 验证输出 schema 匹配 `NEGATIVE_SAMPLE_FEATURES`
  - 新增 conftest fixture: `synthetic_interim_samples_dataset`（INTERIM_SAMPLE_FEATURES schema）

### 实现

- [x] T012 [US2] 实现负采样处理器 — `saegenrec/data/processors/negative_sampling.py`（新建文件）
  - `build_user_interacted_items(user_sequences: Dataset) -> dict[int, set[int]]`
  - `sample_negatives(samples, user_interacted_items, all_item_ids, item_titles, num_negatives, seed) -> tuple[Dataset, dict]`
  - 使用 `numpy.random.Generator` + seed 确保可复现
  - 负样本不足时降级采样并记录 WARNING
  - 输出使用 `NEGATIVE_SAMPLE_FEATURES` schema
  - stats dict 包含: num_negatives_requested, num_negatives_warnings

- [x] T013 [US2] 在 `run_pipeline` 中集成 `negative_sampling` 步骤 — `saegenrec/data/pipeline.py` + `Makefile`
  - 将 `STAGE2_STEPS` 扩展为 `["split", "augment", "negative_sampling"]`，同步更新 `ALL_STEPS`
  - 在 augment 之后执行 negative_sampling
  - 对 train/valid/test 三个 split 分别调用 `sample_negatives`
  - 将 negative_sampling 统计写入阶段 2 的 `stats.json`
  - 从阶段 1 输出读取 `user_sequences` 和 `item_id_map` 构建排除集合和全局商品列表
  - 更新 Makefile `data-split` 目标：追加 `--step negative_sampling`

- [x] T014 [US2] 新增 conftest 负采样相关 fixture — `tests/unit/data/conftest.py`
  - `synthetic_interim_samples_dataset`: 基于 INTERIM_SAMPLE_FEATURES 的合成数据
  - `synthetic_all_item_ids`: 全局商品 ID 列表
  - `synthetic_item_titles_map`: mapped_id → title 映射

**Checkpoint**: 此时 User Story 2 应完全可运行：负采样正确、可复现、降级处理完备

---

## Phase 4: User Story 3 — 灵活的步骤选择与增量运行 (Priority: P3)

**Goal**: 支持单独运行管道中的某些步骤而不重跑前序步骤

**Independent Test**: 先运行完整管道，然后仅指定 `--step negative_sampling` 重新运行，验证只有负采样结果被更新

- [x] T015 [US3] 增量运行集成测试 — `tests/unit/data/test_pipeline.py`
  - 验证 `--step split --step augment` 仅刷新划分和增强结果
  - 验证 `--step negative_sampling` 仅刷新负采样结果
  - 验证前序阶段输出不变（通过文件修改时间或内容比较）

- [x] T016 [US3] 确保 `run_pipeline` 支持跨阶段增量运行 — `saegenrec/data/pipeline.py`
  - 当指定的步骤仅包含阶段 2 步骤时，从磁盘加载阶段 1 产物
  - 当指定的步骤仅包含部分阶段 2 步骤时，从磁盘加载已有的阶段 2 产物
  - 验证前序产物存在，不存在时报错提示先运行对应阶段

**Checkpoint**: 所有 User Story 均可独立运行和测试

---

## Phase 5: Polish & Cross-Cutting

**Purpose**: 清理和验证

- [x] T017 运行全量测试套件，确保无回归 — `make test`
- [x] T018 [P] 运行 linter 检查 — `make lint`
- [x] T019 验证 quickstart.md 中的命令可正确执行

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Foundational)**: 无依赖，立即开始
- **Phase 2 (US1)**: 依赖 Phase 1 完成
- **Phase 3 (US2)**: 依赖 Phase 2 完成（T007 pipeline 重构后才能集成 negative_sampling）
- **Phase 4 (US3)**: 依赖 Phase 2 + Phase 3 完成
- **Phase 5 (Polish)**: 依赖所有 Phase 完成

### Within Each Phase

- 标记 [P] 的任务可并行执行
- 测试任务先于实现任务（但标记 [P] 时可与其他测试并行）
- Phase 2 内部: T004/T005 → T006 → T007 → T008/T009/T010
- Phase 3 内部: T011 → T012 → T013/T014

### Parallel Opportunities

```
Phase 1:  T001 ─┐
          T002 ─┤─ 同时执行
          T003 ─┘

Phase 2:  T004 ─┬─ 测试先行（并行）
          T005 ─┘
            ↓
          T006 ─── augment 重构
            ↓
          T007 ─── pipeline 重构
            ↓
          T008 ─┬─ 并行
          T009 ─┤
          T010 ─┘

Phase 3:  T011 ─── 测试先行
            ↓
          T012 ─── negative_sampling 实现
            ↓
          T013 ─┬─ 并行
          T014 ─┘
```

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Phase 1: Foundational (T001–T003)
2. Complete Phase 2: User Story 1 (T004–T010)
3. **STOP and VALIDATE**: 运行 `make data-filter && make data-split`，验证两阶段输出（此时 data-split 仅含 split + augment，不含 negative_sampling）
4. 确认 schema 中无 `*_tokens` 字段

### Incremental Delivery

1. Phase 1 → 配置就绪
2. Phase 2 → 两阶段管道可用（MVP）
3. Phase 3 → 负采样可用
4. Phase 4 → 增量运行可用
5. Phase 5 → 质量验证

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
