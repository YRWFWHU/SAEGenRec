# Feature Specification: 生成式推荐架构

**Feature Branch**: `004-genrec-architecture`  
**Created**: 2026-03-02  
**Status**: Draft  
**Input**: 设计生成式推荐架构，包含 ItemTokenizer（RQ-VAE / RQ-KMeans）、LLM 训练数据构建器、以及生成式推荐本体。使用 HuggingFace 设计哲学，重点放在前两部分。

## Clarifications

### Session 2026-03-02

- Q: RQ-VAE tokenizer 默认消费哪种 embedding 作为输入？ → A: ABC 接口同时暴露语义和协同两种 embedding 入口，具体使用逻辑由实现者自行编写；内置的 RQ-VAE 和 RQ-KMeans 仅使用语义 embedding
- Q: SFT prompt 模板应如何存储和管理？ → A: 外部 YAML/JSON 模板文件，运行时加载（新增模板不需改代码）
- Q: 碰撞消解应采用哪种策略？ → A: 可配置，Sinkhorn 重分配和追加层级两种策略均实现，由配置选择
- Q: Item2Index / Index2Item 等非序列 SFT 任务的数据来源？ → A: 使用全部物品（K-core 过滤后），SeqRec 等序列任务仅用训练集交互

## User Scenarios & Testing

### User Story 1 — 物品 Tokenization：将物品映射为层次化语义 ID (Priority: P1)

研究者拥有已生成的语义 embedding 和/或协同 embedding，需要通过 ItemTokenizer 将物品映射为离散的层次化语义 ID（SID），以便后续 LLM 能以自然语言形式生成推荐结果。

**Why this priority**: SID 是生成式推荐的基础表示。没有 SID，LLM 无法将推荐结果映射回具体商品。这是整个管道的前置依赖。

**Independent Test**: 给定一组物品 embedding，运行 ItemTokenizer 训练和推理，验证输出的 SID 满足唯一性约束且碰撞率可接受。

**Acceptance Scenarios**:

1. **Given** 已有 `item_semantic_embeddings/` 数据（内置实现仅需语义 embedding），**When** 运行 `make data-tokenize` 或 `python -m saegenrec.dataset tokenize <config.yaml>`，**Then** 生成 `item_sid_map/` 数据集，包含每个物品的层次化 SID 编码
2. **Given** 选择 RQ-VAE tokenizer 且配置 4 层 × 256 码本，**When** 训练完成后推理所有物品，**Then** 碰撞率经过消解后为 0%，每个物品对应唯一 SID
3. **Given** 选择 RQ-KMeans tokenizer，**When** 训练完成，**Then** 输出相同格式的 `item_sid_map/`，可与 RQ-VAE 互换使用
4. **Given** 已存在 `item_sid_map/`，**When** 再次运行 tokenize，**Then** 默认跳过；使用 `--force` 强制重新训练

---

### User Story 2 — LLM 训练数据构建：将序列推荐数据转换为 SFT 指令数据 (Priority: P1)

研究者需要将序列推荐数据（用户交互历史 + SID）转换为 LLM 可消费的 SFT（Supervised Fine-Tuning）指令微调数据，支持多种推荐任务模板。

**Why this priority**: 与 ItemTokenizer 同优先级，两者共同构成生成式推荐的数据准备核心。SFT 数据的质量直接决定 LLM 微调效果。

**Independent Test**: 给定 `item_sid_map/` 和用户交互序列，运行数据构建器，验证输出的指令数据格式正确、任务分布合理。

**Acceptance Scenarios**:

1. **Given** 已有 `item_sid_map/` 和 `train_sequences/`，**When** 运行 `make data-build-sft` 或 `python -m saegenrec.dataset build-sft <config.yaml>`，**Then** 生成多种 SFT 任务的指令数据集
2. **Given** 配置启用 SeqRec 和 Item2Index 两种任务，**When** 构建完成，**Then** 输出包含这两种任务的混合数据集，每条记录包含 instruction、input、output 三个字段
3. **Given** 用户交互序列长度超过配置的最大历史长度，**When** 构建 SeqRec 数据，**Then** 自动截断为配置长度
4. **Given** 已存在 SFT 数据，**When** 再次运行，**Then** 默认跳过；使用 `--force` 强制重新构建

---

### User Story 3 — 管道集成：一键运行从 tokenization 到 SFT 数据生成 (Priority: P2)

研究者希望通过管道步骤或 Makefile 一键运行 tokenization + SFT 数据构建，无需手动拆分步骤。

**Why this priority**: 提升研究者工作效率，但功能上依赖 P1 的两个子系统。

**Independent Test**: 运行完整管道 `--step tokenize --step build-sft`，验证端到端输出正确。

**Acceptance Scenarios**:

1. **Given** 已完成 Stage 1 + Stage 2 + Embedding，**When** 运行 `python -m saegenrec.dataset process <config.yaml> --step tokenize --step build-sft`，**Then** 按依赖顺序执行 tokenization 和 SFT 数据构建
2. **Given** tokenize 步骤的前置产物（embedding）不存在，**When** 尝试运行，**Then** 提示用户先完成 embed 步骤

---

### User Story 4 — 生成式推荐模型接口定义 (Priority: P3)

定义生成式推荐模型（GenRec Model）的抽象接口，使研究者能以 HuggingFace 设计哲学（`from_pretrained` / `AutoModel` 模式）加载和使用不同的 SFT / RL 配置。

**Why this priority**: 本期重点是 ItemTokenizer 和数据构建。模型接口定义为后续 SFT / RL 训练预留扩展点，当前仅设计接口不实现训练逻辑。

**Independent Test**: 定义 ABC 和注册表，验证接口可被继承且不依赖具体训练框架。

**Acceptance Scenarios**:

1. **Given** 定义了 `GenRecModel` ABC，**When** 研究者实现子类并注册，**Then** 可通过配置文件指定使用哪个模型
2. **Given** 模型接口定义了 `train`、`generate`、`evaluate` 方法签名，**When** 查看接口文档，**Then** 签名与 HuggingFace Trainer 风格一致

---

### Edge Cases

- 所有物品 embedding 相同（退化情况）时，码本分配应仍能完成且报告高碰撞率警告
- 物品数量少于码本大小时，tokenizer 应正常工作且码本利用率降低
- SFT 构建时用户交互序列仅 1 条记录，应跳过该用户并记录警告
- embedding 文件缺失时，tokenize 步骤应给出明确错误提示

## Requirements

### Functional Requirements

#### ItemTokenizer 子系统

- **FR-001**: 系统 MUST 提供 `ItemTokenizer` ABC 和注册表，支持通过装饰器注册不同 tokenizer 实现
- **FR-002**: 系统 MUST 实现 `RQVAETokenizer`，使用 MLP 编码器 + 残差向量量化（RQ）+ MLP 解码器将物品 embedding 映射为多层离散码
- **FR-003**: 系统 MUST 实现 `RQKMeansTokenizer`，使用逐层残差 KMeans 聚类将物品 embedding 映射为多层离散码（无神经网络训练）
- **FR-004**: 两种 tokenizer MUST 支持可配置的码本层数（默认 4 层）和每层码本大小（默认 256）
- **FR-005**: 系统 MUST 在推理后执行碰撞消解，确保每个物品获得唯一 SID。支持两种可配置策略：Sinkhorn 重分配（SID 层数固定）和追加层级去重（SID 层数可变）
- **FR-006**: 系统 MUST 输出 `item_sid_map/` HuggingFace Dataset，schema 包含 `item_id`（int32）、`codes`（Sequence(int32)，每层一个码本索引）和 `sid_tokens`（string，SID token 拼接字符串）
- **FR-007**: `ItemTokenizer` ABC 的 `generate` 方法 MUST 同时接收语义 embedding 和协同 embedding 的路径参数，具体如何使用由各实现自行决定。内置的 RQ-VAE 和 RQ-KMeans 仅使用语义 embedding
- **FR-008**: RQ-VAE tokenizer MUST 使用 PyTorch Lightning 训练，输出训练损失（重建损失、量化损失）和码本利用率指标
- **FR-009**: RQ-KMeans tokenizer MUST 支持均衡约束（Constrained KMeans），保证码本利用率
- **FR-010**: 系统 MUST 将 SID 特殊 token（如 `<s_a_42>`）以配置化格式生成，用于后续 LLM 词表扩展

#### LLM 训练数据构建器子系统

- **FR-011**: 系统 MUST 提供 `SFTTaskBuilder` ABC 和注册表，支持通过装饰器注册不同 SFT 任务类型
- **FR-012**: 系统 MUST 实现以下核心 SFT 任务类型：
  - **SeqRec**（序列推荐）：给定用户历史交互 SID 序列，预测下一个物品 SID。数据来源仅限训练集交互
  - **Item2Index**（物品→SID）：给定物品标题/描述，预测物品 SID。数据来源为全部物品（K-core 过滤后）
  - **Index2Item**（SID→物品）：给定物品 SID，预测物品标题/描述。数据来源为全部物品（K-core 过滤后）
- **FR-013**: 每种 SFT 任务 MUST 支持多个 prompt 模板，存储在外部 YAML/JSON 模板文件中并在运行时加载，新增模板无需修改代码。构建时随机采样模板以增加多样性
- **FR-014**: SFT 数据 MUST 使用 Alpaca 指令格式（instruction / input / output 三字段），以兼容主流 LLM 微调框架
- **FR-015**: 系统 MUST 支持配置启用/禁用各种 SFT 任务类型，以及控制各任务的采样比例
- **FR-016**: 系统 MUST 输出 SFT 数据集为 HuggingFace Dataset 格式，包含 `task_type`、`instruction`、`input`、`output` 字段

#### 生成式推荐模型接口（仅定义）

- **FR-017**: 系统 MUST 定义 `GenRecModel` ABC 和注册表，预留 `train`、`generate`、`evaluate` 方法签名
- **FR-018**: 接口设计 MUST 遵循 HuggingFace 设计哲学，支持通过配置指定 base model、LoRA 参数、训练策略（SFT / RL）

#### 管道集成

- **FR-019**: 系统 MUST 支持 `tokenize` 作为管道步骤，位于 `embed` 之后
- **FR-020**: 系统 MUST 支持 `build-sft` 作为管道步骤，位于 `tokenize` 之后
- **FR-021**: 系统 MUST 提供独立 CLI 命令 `tokenize` 和 `build-sft`，以及对应 Makefile 目标
- **FR-022**: 所有步骤 MUST 支持 `--force` 覆盖已有结果，默认跳过

### Key Entities

- **ItemTokenizer**: 将物品 embedding 映射为层次化离散码（SID）的转换器。拥有独立 ABC + 注册表
- **SID（Semantic ID）**: 物品的层次化离散编码表示，由多层码本索引组成（如 `<s_a_42><s_b_103><s_c_7><s_d_255>`），可作为 LLM 的特殊 token
- **Codebook（码本）**: 向量量化中的离散码向量集合，每层一个码本
- **SFTTaskBuilder**: 将推荐数据转换为特定 SFT 任务指令数据的构建器。拥有独立 ABC + 注册表
- **SFT Dataset**: LLM 微调用的指令数据集，Alpaca 格式
- **GenRecModel**: 生成式推荐模型的抽象接口（本期仅定义，不实现训练）

### Assumptions

- 输入 embedding 来自前序 embed 步骤的 `item_semantic_embeddings/` 和/或 `item_collaborative_embeddings/`
- 物品数量通常在 1K~100K 级别（Amazon 评论数据集 K-core 过滤后）
- SID 特殊 token 的格式参考 OpenOneRec 的 `<s_{level}_{code}>` 模式，带有 `<|sid_begin|>` 和 `<|sid_end|>` 界定符
- RQ-VAE 训练使用 GPU，RQ-KMeans 可在 CPU 上运行
- SFT 数据构建在 CPU 上运行，不需要 GPU
- 当前不实现 User Tokenization（用户 SID），留作后续迭代

### Out of Scope

- LLM 微调训练和推理（仅定义模型接口，不实现训练循环）
- RL（强化学习）训练流程
- User Tokenization（用户 SID 生成）
- 在线推理和服务化部署
- 多模态输入（图片、视频等）

## Success Criteria

### Measurable Outcomes

- **SC-001**: 研究者可在 10 分钟内完成一个中等规模数据集（~10K 物品）的 ItemTokenizer 训练和 SID 生成
- **SC-002**: RQ-VAE / RQ-KMeans 生成的 SID 碰撞消解后，所有物品获得唯一编码（碰撞率 0%）
- **SC-003**: 码本利用率（被使用的码本向量占比）达到 50% 以上
- **SC-004**: SFT 数据构建覆盖至少 3 种推荐任务类型，每种任务至少 5 个 prompt 模板
- **SC-005**: 生成的 SFT 数据可直接被 HuggingFace `transformers.Trainer` 或同类框架消费，无需额外格式转换
- **SC-006**: 新增 tokenizer 或 SFT 任务类型可通过注册表机制在 20 行代码内完成扩展
