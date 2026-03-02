# Feature Specification: Embedding 模块（语义 + 协同）

**Feature Branch**: `003-embedding-module`  
**Created**: 2026-03-02  
**Status**: Clarified  
**Input**: User description: "设计一个 embedding 模块，包含两个解耦的子系统：语义 embedder 和协同 embedder，各自拥有独立的 ABC + 注册表，与现有 tokenizer 模块完全解耦。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 语义 Embedding 生成 (Priority: P1)

研究者对已完成 Stage 1 处理的数据集运行语义 embedding 生成，系统使用预训练语言模型（sentence-transformers）对物品元数据的指定文本字段（默认 title、brand、description、price）拼接后提取语义向量，保存到中间数据同级路径的 `item_semantic_embeddings/` 目录。仅对通过 K-core 过滤且在 item_metadata 中有记录的物品生成 embedding，缺失物品跳过并记录警告。完成后输出统计信息（物品数、维度、耗时）。

**Why this priority**: 语义 embedding 是下游 ItemTokenizer（如 RQ-VAE）和生成式推荐模型的核心输入，生成过程简单（无需训练），是最快可交付的独立价值切片。

**Independent Test**: 提供一份小规模已完成 Stage 1 的中间数据（含 item_metadata 和 item_id_map），运行语义 embedding 生成命令，验证输出 Dataset 的物品数和向量维度正确。

**Acceptance Scenarios**:

1. **Given** data/interim/{dataset}/{category}/ 中存在 item_metadata 和 item_id_map（Stage 1 产出）, **When** 研究者运行语义 embedding 生成, **Then** item_semantic_embeddings/ 目录中生成包含 item_id 和 embedding 的 HuggingFace Dataset，物品数等于 K-core 过滤后且在 item_metadata 中有记录的物品数
2. **Given** item_id_map 中存在某物品 ID 但 item_metadata 中无该物品记录, **When** 生成语义 embedding, **Then** 该物品被跳过，系统记录一条 warning 级别日志
3. **Given** 某物品在 item_metadata 中所有指定文本字段均为空, **When** 生成语义 embedding, **Then** 该物品的 embedding 为零向量（维度与模型输出一致）
4. **Given** item_metadata 中 price 字段为数值类型, **When** 拼接文本字段, **Then** price 被转为文本格式参与拼接（如 29.99 → "29.99"）
5. **Given** item_semantic_embeddings/ 目录已存在, **When** 研究者再次运行, **Then** 默认跳过生成并提示已存在；使用 --force 时覆盖重新生成
6. **Given** 配置中设置 L2 归一化为关闭（默认）, **When** 生成语义 embedding, **Then** 输出向量未经 L2 归一化

---

### User Story 2 - 协同 Embedding 生成 (Priority: P1)

研究者对已完成 Stage 2 划分的数据集运行协同 embedding 生成，系统使用 PyTorch Lightning 在用户交互序列上训练序列推荐模型（如 SASRec），从训练后模型的 nn.Embedding 权重中提取物品协同 embedding，保存到 `data/interim/{dataset}/{category}/{split_strategy}/item_collaborative_embeddings/` 目录。训练过程中每个 epoch 和训练完成后输出推荐评估指标（Hit Rate、NDCG 等）。

**Why this priority**: 协同 embedding 捕获用户行为交互信号，是序列推荐中与语义 embedding 互补的核心信号来源。与 User Story 1 同等重要但相互独立。

**Independent Test**: 提供一份小规模 Stage 1 中间数据（含 user_sequences 和 item_id_map），运行协同 embedding 生成命令，验证训练完成、指标输出正确、embedding Dataset 维度与模型配置一致。

**Acceptance Scenarios**:

1. **Given** data/interim/{dataset}/{category}/{split_strategy}/ 中存在 train_sequences、valid_sequences、test_sequences（Stage 2 产出）及 Stage 1 的 item_id_map, **When** 研究者运行协同 embedding 生成（默认使用 SASRec）, **Then** item_collaborative_embeddings/ 目录中生成包含 item_id 和 embedding 的 HuggingFace Dataset，物品数等于 item_id_map 中的物品总数
2. **Given** 训练过程中, **When** 每个 epoch 结束, **Then** 系统输出当前 epoch 的推荐指标（至少包含 Hit Rate@K 和 NDCG@K）
3. **Given** 训练完成, **When** 提取 embedding, **Then** embedding 从训练后模型的 nn.Embedding 层权重直接提取，维度与模型配置的 hidden_size 一致
4. **Given** 配置中指定使用 GRU4Rec 模型, **When** 运行协同 embedding 生成, **Then** 系统通过注册表加载 GRU4Rec 模型实现并完成训练和 embedding 提取
5. **Given** item_collaborative_embeddings/ 目录已存在, **When** 研究者再次运行, **Then** 默认跳过生成并提示已存在；使用 --force 时覆盖重新生成

---

### User Story 3 - 流水线集成与 CLI 独立调用 (Priority: P2)

研究者可以通过两种方式调用 embedding 生成：(1) 作为数据处理流水线的 embed 步骤自动运行；(2) 作为独立 CLI 命令按需运行。两种方式使用相同的配置和相同的底层逻辑。

**Why this priority**: 灵活的调用方式提升研究效率——流水线模式适合端到端实验，独立 CLI 适合调试和迭代。但核心 embedding 生成逻辑（P1）是前提。

**Independent Test**: 分别通过流水线 `--step embed` 和独立 CLI 命令运行语义/协同 embedding 生成，验证两种方式产出的结果一致。

**Acceptance Scenarios**:

1. **Given** 完整的 YAML 配置文件, **When** 研究者运行 `python -m saegenrec.dataset process <config> --step embed`, **Then** 按配置生成语义和/或协同 embedding
2. **Given** 研究者仅需生成语义 embedding, **When** 运行独立 CLI 命令 `python -m saegenrec.dataset embed-semantic <config>`, **Then** 仅运行语义 embedding 生成，不触发协同 embedding 或其他流水线步骤
3. **Given** 研究者仅需生成协同 embedding, **When** 运行独立 CLI 命令 `python -m saegenrec.dataset embed-collaborative <config>`, **Then** 仅运行协同 embedding 生成
4. **Given** 配置中同时启用语义和协同 embedding, **When** 通过流水线 embed 步骤运行, **Then** 两种 embedding 顺序生成，各自独立保存到对应目录

---

### User Story 4 - 自定义 Embedder 扩展 (Priority: P3)

研究者实现自定义的 SemanticEmbedder 或 CollaborativeEmbedder 子类（如使用不同的预训练模型或不同的序列推荐架构），通过注册表注册后即可被系统调用，无需修改核心模块代码。

**Why this priority**: 可扩展性是研究工具的重要特性，但基础实现（P1）和集成（P2）优先。

**Independent Test**: 实现一个自定义 SemanticEmbedder，通过注册表注册，在配置中指定该实现名称，运行 embedding 生成，验证系统正确调用自定义实现。

**Acceptance Scenarios**:

1. **Given** 研究者实现了继承 SemanticEmbedder ABC 的自定义类并注册到注册表, **When** 配置中指定该实现名称, **Then** 系统使用该自定义实现生成语义 embedding
2. **Given** 研究者实现了继承 CollaborativeEmbedder ABC 的自定义类并注册到注册表, **When** 配置中指定该实现名称, **Then** 系统使用该自定义实现生成协同 embedding
3. **Given** 配置中指定了未注册的 embedder 名称, **When** 运行 embedding 生成, **Then** 系统抛出明确的错误信息，列出所有可用的注册名称

---

### Edge Cases

- 当 Stage 1 中间数据不存在时（item_metadata 或 item_id_map 缺失），语义 embedder 应给出清晰的错误提示，指引用户先运行 Stage 1
- 当 Stage 2 划分数据不存在时（train/valid/test_sequences 缺失），协同 embedder 应给出清晰的错误提示，指引用户先运行 split 步骤
- 当 item_metadata 中所有物品的所有文本字段均为空时，语义 embedding 全部为零向量，系统应发出警告
- 当训练数据中用户序列过短（所有用户序列长度 ≤ 1）导致无法有效训练时，协同 embedder 应发出警告
- 当 GPU 不可用但配置中指定 GPU 时，系统应自动回退到 CPU 并记录警告
- 当 --force 与已存在的输出目录组合使用时，旧数据应被完全清除后重新生成，不保留部分旧文件

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 提供 `SemanticEmbedder` 抽象基类（ABC），核心方法签名为 `generate(data_dir: Path, output_dir: Path, config: dict) → Dataset`，与 CollaborativeEmbedder 对称设计
- **FR-002**: 系统 MUST 提供 `CollaborativeEmbedder` 抽象基类（ABC），核心方法签名为 `generate(data_dir: Path, output_dir: Path, config: dict) → Dataset`，与 SemanticEmbedder 对称设计
- **FR-003**: 语义 embedder 和协同 embedder MUST 各自拥有独立的注册表（registry）和注册装饰器，遵循与现有 tokenizer 模块（`TOKENIZER_REGISTRY` / `@register_tokenizer`）相同的注册表模式
- **FR-004**: embedding 模块 MUST 与现有 tokenizer 模块完全解耦——不存在代码导入依赖或运行时耦合
- **FR-005**: 默认语义 embedder 实现 MUST 使用 sentence-transformers 预训练模型对物品元数据的指定文本字段（默认 title、brand、description、price）拼接后提取语义 embedding
- **FR-006**: 语义 embedder MUST 支持 L2 归一化的可配置开关，默认关闭（对齐 Align3GR 方案）
- **FR-007**: 语义 embedder MUST 仅对通过 K-core 过滤且在 item_metadata 中有记录的物品生成 embedding；item_id_map 中存在但 item_metadata 中缺失的物品 MUST 跳过并记录 warning
- **FR-008**: 语义 embedder MUST 对所有指定文本字段均为空的物品生成零向量（维度与模型输出一致）
- **FR-009**: 语义 embedder MUST 将 price 等数值字段转为文本格式后拼接（如 29.99 → "29.99"）
- **FR-010**: 每种序列推荐模型（SASRec、GRU4Rec 等）MUST 作为独立的 CollaborativeEmbedder 实现注册到 COLLABORATIVE_EMBEDDER_REGISTRY，各自封装完整的训练和 embedding 提取逻辑，使用 PyTorch Lightning 进行训练
- **FR-011**: 协同 embedder MUST 从训练后模型的 nn.Embedding 层权重中直接提取物品协同 embedding
- **FR-012**: 协同 embedder MUST 在训练过程中每个 epoch 结束和训练完成后输出推荐评估指标（至少包含 Hit Rate@K 和 NDCG@K）
- **FR-013**: 协同 embedder MUST 消费 Stage 2 已划分的训练/验证/测试序列（train_sequences、valid_sequences、test_sequences）以及 Stage 1 的 item_id_map；embed 步骤 MUST 在 split 步骤之后运行
- **FR-014**: 语义 embedding MUST 保存到 `data/interim/{dataset}/{category}/item_semantic_embeddings/`（Stage 1 同级，不依赖划分策略）；协同 embedding MUST 保存到 `data/interim/{dataset}/{category}/{split_strategy}/item_collaborative_embeddings/`（Stage 2 同级，因不同划分策略产出不同 embedding）
- **FR-015**: 系统 MUST 支持作为流水线 embed 步骤和独立 CLI 命令两种调用方式
- **FR-016**: 当目标输出目录已存在时，系统 MUST 默认跳过生成；提供 `--force` 参数强制覆盖
- **FR-017**: embedding 生成完成后 MUST 输出统计信息，至少包含：物品数、embedding 维度、耗时
- **FR-018**: 当前版本 MUST NOT 实现断点续传功能
- **FR-019**: 当前版本 MUST NOT 实现用户语义 embedding，留作后续迭代
- **FR-020**: 现有 `saegenrec/data/embeddings/text.py` 中的 `generate_text_embeddings` 功能 MUST 迁移重构为 SemanticEmbedder ABC 的默认实现（注册名如 `sentence-transformer`），废弃旧 `text_embeddings/` 输出路径，统一使用 `item_semantic_embeddings/`

### Key Entities

- **SemanticEmbedder（语义 Embedder）**: 编码型 embedding 生成器的抽象接口。使用预训练语言模型将物品元数据文本编码为稠密向量。核心方法 `generate(data_dir, output_dir, config) → Dataset`。拥有独立的注册表，支持可插拔实现。
- **CollaborativeEmbedder（协同 Embedder）**: 训练型 embedding 生成器的抽象接口。通过在用户交互序列上训练序列推荐模型来学习物品协同 embedding。核心方法 `generate(data_dir, output_dir, config) → Dataset`，与 SemanticEmbedder 对称。每种模型架构（SASRec、GRU4Rec 等）各自作为独立的 CollaborativeEmbedder 实现注册，封装完整的训练和提取逻辑。
- **SemanticEmbeddingDataset（语义 Embedding 数据集）**: 语义 embedder 的输出产物，包含 item_id 和对应的语义 embedding 向量。以 HuggingFace Dataset 格式存储在 `item_semantic_embeddings/` 目录。
- **CollaborativeEmbeddingDataset（协同 Embedding 数据集）**: 协同 embedder 的输出产物，包含 item_id 和对应的协同 embedding 向量。以 HuggingFace Dataset 格式存储在 `{split_strategy}/item_collaborative_embeddings/` 目录（因不同划分策略产出不同 embedding）。
- **SEMANTIC_EMBEDDER_REGISTRY（语义 Embedder 注册表）**: 注册 SemanticEmbedder 实现的全局字典，通过 `@register_semantic_embedder("name")` 装饰器注册，通过 `get_semantic_embedder(name)` 工厂函数获取实例。
- **COLLABORATIVE_EMBEDDER_REGISTRY（协同 Embedder 注册表）**: 注册 CollaborativeEmbedder 实现的全局字典，通过 `@register_collaborative_embedder("name")` 装饰器注册，通过 `get_collaborative_embedder(name)` 工厂函数获取实例。

## Clarifications

### Session 2026-03-02

- Q: SASRec、GRU4Rec 等模型是各自作为独立的 CollaborativeEmbedder 实现注册，还是由一个通用 CollaborativeEmbedder 内部维护模型注册表？ → A: 每种模型（SASRec、GRU4Rec 等）作为独立的 CollaborativeEmbedder 实现注册到 COLLABORATIVE_EMBEDDER_REGISTRY，注册表模式与 tokenizer 类似但两个模块完全独立。
- Q: 协同 embedder 训练数据来源——内部自行划分 Stage 1 数据，还是消费 Stage 2 已划分的数据？ → A: 消费 Stage 2 已划分的 train/valid/test_sequences，embed 步骤必须在 split 之后运行。
- Q: 现有 text_embeddings 代码（saegenrec/data/embeddings/text.py）和旧输出路径如何处理？ → A: 将现有代码迁移重构为 SemanticEmbedder ABC 的默认实现（如 SentenceTransformerEmbedder），废弃旧 text_embeddings/ 路径，统一使用 item_semantic_embeddings/。
- Q: 协同 embedding 存储路径是否应区分划分策略（LOO vs TO 产出不同 embedding）？ → A: 是，协同 embedding 存储在 `data/interim/{dataset}/{category}/{split_strategy}/item_collaborative_embeddings/`，跟随 Stage 2 目录层级。

## Assumptions

- Stage 1 中间数据格式稳定，embedding 模块可直接依赖 `user_sequences`、`item_id_map`、`item_metadata` 的 HuggingFace Dataset schema
- sentence-transformers 库已在项目依赖中（当前代码已使用）
- PyTorch Lightning 和 PyTorch 将作为新增依赖
- GPU 可用性不可预设，系统需支持 CPU 回退
- 推荐评估指标（Hit Rate、NDCG）的 K 值通过配置指定，默认值参照领域惯例（如 K=10, 20）
- 协同 embedder 的模型训练超参数（学习率、batch size、epoch 数、hidden size 等）通过配置文件指定

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 语义 embedding 生成可以在单机 CPU 上对 10,000 个物品在 5 分钟内完成（使用默认 all-MiniLM-L6-v2 模型）
- **SC-002**: 协同 embedding 生成可以在单机 GPU 上对 50,000 条交互记录在 30 分钟内完成训练并提取 embedding
- **SC-003**: 所有 embedding 生成步骤的输出结果在相同配置和随机种子下 100% 可复现
- **SC-004**: 新增一种 SemanticEmbedder 实现仅需继承抽象基类、用装饰器注册，无需修改 embedding 模块的其他代码
- **SC-005**: 新增一种 CollaborativeEmbedder 实现（含新模型架构）仅需继承抽象基类、用装饰器注册，无需修改 embedding 模块的其他代码
- **SC-006**: embedding 模块与 tokenizer 模块之间零代码导入依赖——删除任一模块不影响另一模块的正常运行
- **SC-007**: 协同 embedder 训练产出的推荐指标（Hit Rate@10、NDCG@10）在 Amazon Baby 数据集上达到同类方法的合理范围（Hit Rate@10 > 0.3）
