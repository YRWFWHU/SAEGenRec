# Feature Specification: SAE Item Tokenizer

**Feature Branch**: `006-item-tokenizer-sae`  
**Created**: 2026-03-01  
**Status**: Draft  
**Input**: User description: "添加ItemTokenizer，使用text embedding训练一个SAE，SAE的训练参考SAELens。参数通过配置文件配置，可选配置是d_sae和Topk。使用JumpReLU SAE模型。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 使用 SAE 训练生成物品语义 ID (Priority: P1)

研究者希望使用 JumpReLU SAE 将物品的文本嵌入向量映射为稀疏语义 ID (SID)，以替代现有的 RQ-VAE 或 RQ-KMeans 方案。研究者在配置文件中将 `item_tokenizer.name` 设为 `"sae"`，设置 `d_sae`（SAE 隐藏维度 / 概念词表大小）和 `top_k`（每个物品选取的激活特征数），然后运行已有的 `make tokenize` 命令即可完成 SAE 训练与 SID 生成。

**Why this priority**: 这是核心功能，没有 SAE 训练与编码能力，后续所有基于 SAE SID 的实验都无法进行。

**Independent Test**: 准备一组物品文本嵌入向量，配置 `item_tokenizer.name: sae`，运行 tokenize 流程，验证输出的 SID map 格式正确、每个物品恰好生成 `top_k` 个 code。

**Acceptance Scenarios**:

1. **Given** 已有语义嵌入数据集（HuggingFace Dataset 格式，含 `item_id` 和 `embedding` 列），**When** 用户配置 `item_tokenizer.name: sae, d_sae: 8192, top_k: 8` 并执行 tokenize 命令，**Then** 系统训练 JumpReLU SAE，生成每个物品的 8 个 code（取自 0~8191 的概念词表），输出 SID map 至指定目录。
2. **Given** SAE 已训练完成，**When** 用户对新的嵌入向量调用 `encode()`，**Then** 返回 shape 为 `(N, top_k)` 的整数 tensor，每行包含 top_k 个激活最强的特征索引。
3. **Given** SAE 模型已保存到磁盘，**When** 用户调用 `load()` 重新加载并再次 `encode()` 相同嵌入向量，**Then** 输出结果与保存前完全一致。

---

### User Story 2 - 通过配置文件灵活调整 SAE 超参数 (Priority: P2)

研究者希望通过 YAML 配置文件控制 SAE 的关键参数（`d_sae`、`top_k`）以及训练参数（学习率、epoch 数、batch size 等），无需修改代码即可进行超参数搜索。

**Why this priority**: 灵活的配置是高效实验迭代的基础，但依赖 P1 的核心训练能力。

**Independent Test**: 修改配置文件中的 `d_sae` 和 `top_k` 值，验证训练输出的 SID 维度符合预期。

**Acceptance Scenarios**:

1. **Given** 配置文件中 `item_tokenizer.params.d_sae: 4096, item_tokenizer.params.top_k: 16`，**When** 执行 tokenize，**Then** SAE 隐藏维度为 4096，每个物品 SID 包含 16 个 code。
2. **Given** 配置文件中未指定 `d_sae` 或 `top_k`，**When** 执行 tokenize，**Then** 使用合理默认值（d_sae=8192, top_k=8）。

---

### User Story 3 - SAE 模型的持久化与复用 (Priority: P3)

研究者希望训练好的 SAE 模型可以保存到磁盘并在后续实验中复用，避免重复训练。

**Why this priority**: 提高实验效率，但不影响核心功能。

**Independent Test**: 训练 SAE 后保存，在另一次运行中加载并编码，验证结果一致。

**Acceptance Scenarios**:

1. **Given** SAE 训练完成，**When** 调用 `save(path)`，**Then** 模型权重和配置参数保存至指定路径。
2. **Given** 已保存的 SAE 模型，**When** 调用 `load(path)`，**Then** 模型恢复到训练完成时的状态，encode 输出一致。

---

### Edge Cases

- 当 `d_sae < top_k` 时，系统应报错（概念词表大小必须大于选取数量）。
- 当嵌入维度 (`d_in`) 与 SAE 期望输入维度不匹配时，系统应报错并提示实际维度。
- 当所有物品嵌入完全相同时，SAE 应仍能正常训练并输出相同的 SID。
- 当物品数量极少（如 < 10）时，训练应仍能完成（可能收敛质量下降但不崩溃）。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 提供名为 `"sae"` 的 `ItemTokenizer` 实现，通过 `@register_item_tokenizer("sae")` 注册到已有的 tokenizer 注册表中。
- **FR-002**: SAE tokenizer MUST 使用 JumpReLU 激活函数，参考 SAELens 的 `JumpReLUTrainingSAE` 实现训练版本，参考 `JumpReLUSAE` 实现推理版本。
- **FR-003**: SAE tokenizer MUST 接受 `d_sae`（SAE 隐藏维度/概念数量）和 `top_k`（选取的最大激活特征数）作为核心配置参数。
- **FR-004**: `train()` 方法 MUST 从语义嵌入数据集中加载嵌入向量，自动推断输入维度 `d_in`，训练 JumpReLU SAE 直到收敛或达到指定 epoch 数。
- **FR-005**: `encode()` 方法 MUST 将嵌入向量通过训练好的 SAE 编码，选取激活值最高的 `top_k` 个特征索引作为 SID code，返回 shape 为 `(N, top_k)` 的整数 tensor。
- **FR-006**: `train()` 方法 MUST 返回训练指标字典，至少包含最终重构损失 (reconstruction loss) 和稀疏性指标 (L0 / 平均激活特征数)。
- **FR-007**: SAE tokenizer MUST 符合已有的 `ItemTokenizer` 抽象接口，包括 `train()`, `encode()`, `save()`, `load()`, `num_codebooks`, `codebook_size` 属性。
- **FR-008**: `num_codebooks` 属性 MUST 返回 `top_k` 值（对应 SID 中的 code 数量）；`codebook_size` 属性 MUST 返回 `d_sae` 值（对应概念词表大小）。
- **FR-009**: 训练参数（learning_rate, epochs, batch_size, l0_coefficient 等）MUST 通过配置文件的 `item_tokenizer.params` 字段传递。

### Key Entities

- **JumpReLU SAE Model**: 核心模型，包含编码器权重 `W_enc (d_in, d_sae)`、解码器权重 `W_dec (d_sae, d_in)`、偏置 `b_enc`, `b_dec`、以及 JumpReLU 阈值参数 `threshold (d_sae,)`。输入物品嵌入 `(d_in,)`，输出稀疏特征激活 `(d_sae,)`。
- **SAE Tokenizer**: `ItemTokenizer` 的具体实现，封装 JumpReLU SAE 的训练、编码、保存/加载逻辑。核心属性：`d_sae`, `top_k`, `d_in`（训练时自动推断）。
- **SID Code**: 每个物品经 SAE 编码后选取的 `top_k` 个特征索引，取值范围为 `[0, d_sae)`，构成物品的语义 ID。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: SAE tokenizer 能够成功训练并生成 SID map，与 RQ-VAE tokenizer 使用相同的 CLI 命令和配置接口。
- **SC-002**: 训练后的 SAE 重构损失（MSE）收敛到合理范围（相对于输入嵌入的范数）。
- **SC-003**: 每个物品的 SID 恰好包含 `top_k` 个 code，code 值均在 `[0, d_sae)` 范围内。
- **SC-004**: 概念词表利用率（即实际被至少一个物品使用的概念数 / d_sae）合理（应 > 10%）。
- **SC-005**: `save()` + `load()` 后 `encode()` 输出与保存前完全一致（bit-exact）。

## Assumptions

- 输入的语义嵌入数据集已通过上游 embedding 模块生成，格式为 HuggingFace Dataset，包含 `item_id` (int) 和 `embedding` (list[float]) 列。
- JumpReLU SAE 的实现参考 SAELens 的架构，但不直接依赖 SAELens 包——在项目内部实现独立的 JumpReLU SAE 模型。
- 默认超参数：`d_sae=8192`, `top_k=8`, `learning_rate=1e-3`, `epochs=50`, `batch_size=256`。
- SAE 训练使用标准的 MSE 重构损失 + L0 稀疏性惩罚（JumpReLU 的 Step 函数 STE 梯度）。
