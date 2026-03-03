# Tasks: SAE Item Tokenizer

**Input**: Design documents from `/specs/006-item-tokenizer-sae/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included — Constitution III 要求每个核心模块有对应单元测试。

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 依赖添加和目录准备

- [x] T001 Add `safetensors` dependency to `pyproject.toml`
- [x] T002 [P] Ensure `saegenrec/modeling/tokenizers/models/__init__.py` exists (create if missing)

---

## Phase 2: Foundational (JumpReLU SAE Model)

**Purpose**: 实现 JumpReLU SAE 核心模型——所有 User Story 的前置依赖

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Implement `JumpReLU` and `Step` custom autograd functions in `saegenrec/modeling/tokenizers/models/jumprelu_sae.py` — 参考 SAELens 的 `references/SAELens/sae_lens/saes/jumprelu_sae.py` 中的 `JumpReLU`（forward: `x * (x > threshold)`）和 `Step`（forward: `(x > threshold)`）自定义 autograd 类，包含 rectangle 函数的 STE 梯度
- [x] T004 Implement `JumpReLUSAE` nn.Module in `saegenrec/modeling/tokenizers/models/jumprelu_sae.py` — 包含 `__init__(d_in, d_sae, jumprelu_init_threshold, jumprelu_bandwidth)`、`encode(x) → sparse activations`、`decode(feature_acts) → reconstruction`、`forward(x) → (sae_out, feature_acts, hidden_pre)` 方法。参数：`W_enc (d_in, d_sae)`, `W_dec (d_sae, d_in)`, `b_enc (d_sae,)`, `b_dec (d_in,)`, `log_threshold (d_sae,)`
- [x] T005 Implement training loss computation in `saegenrec/modeling/tokenizers/models/jumprelu_sae.py` — 添加 `compute_loss(x, sae_out, feature_acts, hidden_pre, l0_coefficient)` 方法或独立函数，计算 `total_loss = MSE(x, sae_out) + l0_coefficient * L0_loss(hidden_pre, threshold, bandwidth)`
- [x] T006 [P] Write unit tests for JumpReLU SAE model in `tests/unit/modeling/tokenizers/test_jumprelu_sae.py` — 验证：前向传播输出 shape 正确、梯度可计算、JumpReLU 稀疏性（大部分激活为零）、encode 输出稀疏激活、decode 恢复到输入维度、loss 计算正确性

**Checkpoint**: JumpReLU SAE 模型可独立实例化、前向传播、计算损失，所有单元测试通过

---

## Phase 3: User Story 1 — 使用 SAE 训练生成物品语义 ID (Priority: P1) 🎯 MVP

**Goal**: 实现 SAETokenizer，支持从文本嵌入训练 JumpReLU SAE 并生成 SID map

**Independent Test**: 准备合成嵌入数据，配置 `item_tokenizer.name: sae`，调用 `train()` + `encode()`，验证输出 shape 和 code 范围

### Implementation for User Story 1

- [x] T007 [US1] Implement `SAETokenizer` class skeleton in `saegenrec/modeling/tokenizers/sae.py` — 继承 `ItemTokenizer`，添加 `@register_item_tokenizer("sae")` 装饰器，实现 `__init__(num_codebooks, codebook_size, **kwargs)` 构造函数（将 `num_codebooks` 映射到 `top_k`，`codebook_size` 映射到 `d_sae`），实现 `num_codebooks` 和 `codebook_size` 属性
- [x] T008 [US1] Implement `SAETokenizer.train()` in `saegenrec/modeling/tokenizers/sae.py` — 从 `semantic_embeddings_dir` 加载 HF Dataset 获取嵌入向量，自动推断 `d_in`，构建 `JumpReLUSAE` 模型，使用 PyTorch DataLoader + Adam 优化器执行训练循环（for epoch → for batch → forward → loss.backward → step），训练结束后计算并返回指标字典（final_mse_loss, final_l0_loss, mean_l0, vocab_utilization, num_dead_features）
- [x] T009 [US1] Implement `SAETokenizer.encode()` in `saegenrec/modeling/tokenizers/sae.py` — 将输入嵌入通过训练好的 SAE 编码，从稀疏激活中选取 top_k 个最大值的索引，按激活值降序排列，返回 `(N, top_k)` 的 `torch.long` tensor
- [x] T010 [US1] Register SAETokenizer and update exports in `saegenrec/modeling/tokenizers/__init__.py` — 添加 `from saegenrec.modeling.tokenizers.sae import SAETokenizer` 和 `__all__` 条目
- [x] T011 [P] [US1] Write integration tests for SAETokenizer train+encode in `tests/unit/modeling/tokenizers/test_sae_tokenizer.py` — 验证：使用合成嵌入数据训练完成后 encode 输出 shape 为 `(N, top_k)`、code 范围 `[0, d_sae)`、训练指标字典包含必要键、通过 `get_item_tokenizer("sae")` 可正确实例化

**Checkpoint**: SAETokenizer 可通过 `make data-tokenize` 使用 `name: sae` 配置完成训练并生成 SID map

---

## Phase 4: User Story 2 — 通过配置文件灵活调整 SAE 超参数 (Priority: P2)

**Goal**: 支持通过 `params.d_sae`/`params.top_k` 覆盖 `codebook_size`/`num_codebooks`，以及传递训练超参数

**Independent Test**: 使用不同的 d_sae 和 top_k 值配置，验证输出维度匹配

### Implementation for User Story 2

- [x] T012 [US2] Implement parameter priority logic in `SAETokenizer.__init__()` in `saegenrec/modeling/tokenizers/sae.py` — 当 `kwargs` 中包含 `d_sae` 时优先于 `codebook_size`，当包含 `top_k` 时优先于 `num_codebooks`；添加参数验证（`d_sae > top_k > 0`）
- [x] T013 [P] [US2] Add SAE tokenizer example config (commented) to `configs/default.yaml` — 在现有 `item_tokenizer` 段下方添加注释示例，展示 SAE 配置方式
- [x] T014 [P] [US2] Write parameter override tests in `tests/unit/modeling/tokenizers/test_sae_tokenizer.py` — 验证：`params.d_sae` 覆盖 `codebook_size`、`params.top_k` 覆盖 `num_codebooks`、默认值生效、`d_sae < top_k` 时抛出 ValueError

**Checkpoint**: 用户可通过修改 YAML 配置的 `d_sae`/`top_k`/训练参数，无需改代码即可执行不同配置的 SAE 训练

---

## Phase 5: User Story 3 — SAE 模型的持久化与复用 (Priority: P3)

**Goal**: 训练好的 SAE 可保存到磁盘并在后续实验中加载复用

**Independent Test**: 训练 → save → 新建 tokenizer → load → encode，验证输出 bit-exact 一致

### Implementation for User Story 3

- [x] T015 [US3] Implement `SAETokenizer.save()` in `saegenrec/modeling/tokenizers/sae.py` — 使用 `safetensors.torch.save_file()` 保存模型权重（W_enc, W_dec, b_enc, b_dec, log_threshold），使用 JSON 保存超参数（d_in, d_sae, top_k, jumprelu_bandwidth, jumprelu_init_threshold）
- [x] T016 [US3] Implement `SAETokenizer.load()` in `saegenrec/modeling/tokenizers/sae.py` — 从 JSON 读取超参数重建 `JumpReLUSAE` 模型，使用 `safetensors.torch.load_file()` 恢复权重
- [x] T017 [P] [US3] Write save/load round-trip tests in `tests/unit/modeling/tokenizers/test_sae_tokenizer.py` — 验证：save + load 后 encode 输出与保存前 bit-exact 一致（`torch.equal`）、超参数正确恢复、load 后 `num_codebooks` 和 `codebook_size` 属性正确

**Checkpoint**: SAE 模型可持久化和复用，save/load 保证编码一致性

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 输入验证、边界情况处理和文档

- [x] T018 Add input validation and edge case handling in `saegenrec/modeling/tokenizers/sae.py` — `d_sae < top_k` 报错、`d_in` 不匹配时报错并提示实际维度、`encode()` 在未训练/未加载时报错
- [x] T019 [P] Add training progress logging with loguru in `saegenrec/modeling/tokenizers/sae.py` — 每 epoch 输出 MSE loss、L0 loss、mean active features；训练完成后输出 vocab utilization
- [x] T020 Run quickstart.md validation — 按照 `specs/006-item-tokenizer-sae/quickstart.md` 步骤执行端到端验证

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (Phase 3): Depends on Phase 2 — no dependencies on other stories
  - US2 (Phase 4): Depends on Phase 3 (needs SAETokenizer constructor)
  - US3 (Phase 5): Depends on Phase 3 (needs trained model for save/load)
- **Polish (Phase 6)**: Depends on all user stories being complete

### Within Each Phase

- Models/autograd functions before higher-level wrappers
- Core implementation before tests (Constitution III: tests verify behavior)
- Tests marked [P] can run in parallel with other [P] tasks in the same phase

### Parallel Opportunities

- T001, T002: Can run in parallel (different files)
- T005, T006: T006 (tests) can be written in parallel with T005 (loss computation) since tests target the model as a whole
- T010, T011: __init__.py update and tests can be written in parallel
- T013, T014: Config example and parameter tests can run in parallel
- T015, T016 → T017: save and load implementation sequential, then tests in parallel

---

## Parallel Example: Phase 2 (Foundational)

```
# Sequential core implementation:
T003 → T004 → T005

# Then tests can run:
T006 (parallel with Phase 3 start if Phase 2 core is done)
```

## Parallel Example: User Story 1

```
# Sequential implementation:
T007 → T008 → T009 → T010

# Tests can be written alongside T010:
T011 [P] (can start once T009 is complete)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational — JumpReLU SAE model (T003-T006)
3. Complete Phase 3: User Story 1 — train + encode (T007-T011)
4. **STOP and VALIDATE**: Test SAETokenizer with `make data-tokenize` using synthetic data
5. SAE tokenizer is functional — can generate SID maps

### Incremental Delivery

1. Setup + Foundational → JumpReLU SAE model ready
2. Add User Story 1 → SAE 可训练和编码 → MVP!
3. Add User Story 2 → 配置灵活性 → 超参数搜索就绪
4. Add User Story 3 → 模型持久化 → 实验复用就绪
5. Polish → 生产就绪

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Total tasks: 20
- Tasks per story: US1=5, US2=3, US3=3, Foundational=4, Setup=2, Polish=3
- SAELens 参考代码位于 `references/SAELens/sae_lens/saes/jumprelu_sae.py`
- 已有 tokenizer 模式参考: `saegenrec/modeling/tokenizers/rqvae.py` 和 `rqkmeans.py`
