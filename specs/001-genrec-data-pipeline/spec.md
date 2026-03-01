# Feature Specification: 生成式推荐数据处理流水线

**Feature Branch**: `001-genrec-data-pipeline`  
**Created**: 2026-03-01  
**Status**: Clarified  
**Input**: User description: "设计生成式推荐数据处理模块，用于将 raw_data 处理为序列推荐中间数据，并最终处理为用于训练 LLM 的数据。支持多模态扩展（图片下载）、LOO/TO 数据划分、滑动窗口数据增强、可配置序列长度，以及 SID 编码器接口。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 原始数据清洗与序列构建 (Priority: P1)

研究者将 Amazon 评论原始数据（JSON/JSONL 格式）处理为按时间排序的用户交互序列，产出标准化的中间数据。该流程包括：加载原始评论与商品元数据、K-core 过滤（去除交互次数不足的用户和物品）、构建用户-物品交互序列（按时间排序）、生成用户/物品 ID 映射，并保留每条交互的评论信息（评论正文、评论标题/摘要、评分），输出标准化的中间数据文件。

**Why this priority**: 这是整个数据流水线的基础，没有中间数据，后续所有步骤都无法进行。

**Independent Test**: 提供一份小规模 Amazon 原始数据，运行处理流程后检查输出的中间数据文件格式正确、用户序列按时间排序、K-core 过滤生效。

**Acceptance Scenarios**:

1. **Given** data/raw/ 中存在 Amazon2015 Baby 类目的评论文件和元数据文件, **When** 研究者指定该类目运行原始数据处理, **Then** data/interim/ 中生成包含所有通过 K-core 过滤的用户交互序列文件（每条交互保留评论正文、评论标题/摘要和评分）、用户 ID 映射文件和物品 ID 映射文件
2. **Given** 原始数据中存在交互次数少于 K-core 阈值的用户或物品, **When** 运行 K-core 过滤, **Then** 这些用户和物品被排除，且过滤过程迭代执行直到所有实体满足阈值
3. **Given** 同一用户在原始数据中存在多条交互记录, **When** 构建交互序列, **Then** 该用户的交互按时间戳升序排列
4. **Given** 原始数据为 Amazon2023 JSONL 格式, **When** 研究者使用 Amazon2023 对应的处理脚本运行处理流程, **Then** 系统能正确解析 JSONL 格式并产出与 Amazon2015 处理脚本相同结构的中间数据
5. **Given** 需要支持一种新的数据集格式, **When** 研究者添加对应的处理脚本, **Then** 无需修改核心流水线代码即可完成新数据集的处理

---

### User Story 2 - 序列数据划分 (Priority: P1)

研究者将中间数据中的用户交互序列按照指定策略划分为训练集、验证集和测试集。支持两种划分方式：Leave-One-Out（LOO，每个用户最后一个交互为测试、倒数第二个为验证）和时间比例划分（TO，按时间戳将全部交互按比例划分）。

**Why this priority**: 数据划分是评估推荐模型性能的基础，直接影响实验的科学性和可比性。

**Independent Test**: 使用已生成的中间数据，分别运行 LOO 和 TO 划分策略，验证划分结果的正确性。

**Acceptance Scenarios**:

1. **Given** 已生成的中间交互序列数据, **When** 研究者选择 LOO 划分策略, **Then** 每个用户的最后一个交互被分配到测试集，倒数第二个到验证集，其余到训练集
2. **Given** 已生成的中间交互序列数据, **When** 研究者选择 TO 时间划分策略并指定比例（默认 8:1:1）, **Then** 所有交互按时间戳排序后按比例分配到训练集、验证集和测试集
3. **Given** 使用 LOO 划分策略, **When** 某用户的交互总数不足（少于 3 条）, **Then** 该用户被排除出数据集
4. **Given** 任一划分策略, **When** 划分完成, **Then** 训练集、验证集和测试集之间不存在数据泄露（时间维度上训练集早于验证集早于测试集）

---

### User Story 3 - 滑动窗口数据增强 (Priority: P1)

研究者对训练集序列执行滑动窗口切分，生成用于训练的 (历史序列, 目标物品) 样本对。可配置最长序列长度（默认 20），用于截断历史序列。对于每个用户的训练序列，从第 2 个交互开始逐步向后滑动，每一步以当前交互为目标物品、前面的交互为历史序列，历史序列超过最长序列长度时从左侧截断。

**Why this priority**: 滑动窗口是序列推荐中标准的数据增强方法，直接影响训练数据的数量和质量。

**Independent Test**: 输入一组已划分的训练序列，运行滑动窗口增强，验证输出样本对的历史长度范围和数量。

**Acceptance Scenarios**:

1. **Given** 一个长度为 N 的用户训练序列, **When** 执行滑动窗口, **Then** 生成 N-1 个 (历史序列, 目标物品) 样本对，其中第 i 个样本的目标物品为序列中第 i+1 个交互
2. **Given** 最长序列长度设为 20，某用户训练序列长度为 30, **When** 执行滑动窗口, **Then** 所有样本对的历史序列长度不超过 20，较早的交互被截断
3. **Given** 一个长度为 2 的用户训练序列, **When** 执行滑动窗口, **Then** 生成 1 个样本对（1 个历史物品 + 1 个目标物品）
4. **Given** 自定义最长序列长度（如 10）, **When** 执行滑动窗口, **Then** 所有样本对的历史序列长度在 [1, 10] 范围内

---

### User Story 4 - 生成 LLM 训练数据 (Priority: P2)

研究者将划分和增强后的序列数据转换为 LLM 可训练的最终格式。该过程可选使用物品 Tokenizer 将物品 ID 转换为一组离散 token 表示。物品 Tokenizer 是一个通用抽象，其具体实现可消费预计算的 embedding（文本 embedding 或协同 embedding）并通过量化方法（RQ-VAE、PQ 等）生成离散 token。本流水线提供文本 embedding 生成的工具功能（用预训练语言模型对物品元数据文本做向量化），协同 embedding 的生成需要独立的模型训练流程，不在本流水线范围内。最终数据输出到 data/processed/ 目录。

**Why this priority**: 这是数据处理的最终目标，但依赖前述步骤的完成。物品 Tokenizer 当前仅定义接口、不实现具体逻辑。文本 embedding 生成作为工具功能提供。

**Independent Test**: 输入已增强的训练序列和物品元数据，运行最终数据生成（使用默认的透传 Tokenizer），验证输出文件格式正确。

**Acceptance Scenarios**:

1. **Given** 已完成滑动窗口增强的训练/验证/测试数据, **When** 研究者运行最终数据生成, **Then** data/processed/ 中生成包含物品文本信息和交互序列的训练数据文件
2. **Given** 物品 Tokenizer 接口已定义但未实现具体量化逻辑, **When** 未指定具体 Tokenizer, **Then** 系统使用默认的透传 Tokenizer（将整数 ID 直接作为单个 token），不影响流程正常执行
3. **Given** 用户实现了自定义物品 Tokenizer（如基于语义 embedding 的 RQ-VAE Tokenizer）, **When** 通过配置指定该 Tokenizer, **Then** 系统使用指定的 Tokenizer 将物品 ID 转换为一组离散 token
4. **Given** 最终数据生成完成, **When** 查看输出文件, **Then** 每个样本包含用户历史交互序列和目标物品，其中每个物品同时包含文本信息（如 title）和离散 token 标识，下游训练任务可选择使用其中之一或组合使用

---

### User Story 5 - 商品图片下载 (Priority: P3)

研究者根据原始数据中商品元数据的图片 URL 批量下载商品图片到本地，以支持后续多模态推荐模型的训练。

**Why this priority**: 多模态扩展是未来需求，当前 MVP 可以不包含图片数据。

**Independent Test**: 提供一个包含图片 URL 的小规模元数据文件，运行图片下载功能，验证图片正确保存到指定目录。

**Acceptance Scenarios**:

1. **Given** 商品元数据中包含图片 URL（Amazon2015 的 `imUrl` 或 Amazon2023 的 `images` 字段）, **When** 研究者运行图片下载功能, **Then** 图片被下载并以商品 ID 为文件名保存到 data/external/images/ 目录
2. **Given** 某些图片 URL 无效或下载失败, **When** 下载过程中遇到错误, **Then** 系统记录失败的商品 ID 和错误信息，跳过该图片继续处理其余项
3. **Given** 部分图片已存在于本地, **When** 重新运行下载, **Then** 已存在的图片不被重复下载（支持断点续传）

---

### Edge Cases

- 当原始数据文件为空或格式损坏时，系统应给出清晰的错误提示并中止处理
- 当 K-core 过滤后无用户或物品剩余时（阈值过高），系统应发出警告并报告过滤统计
- 当某用户的训练序列长度仅为 1（只有一个交互）时，无法生成有效样本对，系统应跳过该用户
- 当原始数据中存在重复交互记录（相同用户-物品-时间戳）时，系统应去重处理
- 当不同版本 Amazon 数据的字段名不一致时（如 `reviewerID` vs `user_id`），系统应通过统一的字段映射处理

## Clarifications

### Session 2026-03-01

- Q: 物品 Tokenizer 依赖的 embedding（文本/协同）由谁生成？是否需要独立的 ItemEmbedder 抽象接口？ → A: 不需要独立的 ItemEmbedder 接口。文本 embedding 是简单的工具步骤（用预训练模型对中间数据中的文本信息做转换），本流水线提供该工具功能。协同 embedding 需要在序列推荐数据上训练模型后提取，属于独立模块，不在本流水线范围内。ItemTokenizer 实现自行负责加载所需的预计算 embedding 文件。
- Q: 生成文本 embedding 时使用哪些物品元数据文本字段？ → A: 默认拼接 title + brand + categories，支持通过配置自定义字段组合。
- Q: 最终训练数据中物品是否同时包含文本信息和 token 标识？ → A: 是，每个物品同时包含原始文本信息（如 title）和 token 标识，由下游训练任务决定使用哪个。
- Q: 原始数据清洗阶段需要保留哪些评论(review)字段到中间数据？ → A: 保留评论正文（reviewText/text）、评论标题/摘要（summary/title）和评分（overall/rating）。
- Q: 每种数据集是否需要独立的处理脚本？ → A: 是，每种数据集（如 Amazon2015、Amazon2023）MUST 有对应的处理脚本，封装该数据集特有的格式解析和字段映射逻辑。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 为每种数据集提供独立的处理脚本（如 Amazon2015 脚本、Amazon2023 脚本），封装该数据集特有的格式解析和字段映射逻辑，输出统一的中间数据格式。新增数据集仅需添加对应的处理脚本，不修改核心流水线代码
- **FR-002**: 系统 MUST 提供 K-core 过滤功能，支持配置最小交互次数阈值（默认为 5），迭代过滤直到所有用户和物品均满足阈值
- **FR-003**: 系统 MUST 将通过过滤的交互记录构建为按时间戳升序排列的用户交互序列
- **FR-004**: 系统 MUST 生成用户 ID 映射（原始 ID → 连续整数）和物品 ID 映射（原始 ID → 连续整数），映射从 0 开始
- **FR-005**: 系统 MUST 将中间数据（交互序列、ID 映射、物品元数据、评论信息）以 HuggingFace Datasets 兼容格式存储到 data/interim/。交互记录中 MUST 保留评论正文（Amazon2015: reviewText / Amazon2023: text）、评论标题/摘要（Amazon2015: summary / Amazon2023: title）和评分（Amazon2015: overall / Amazon2023: rating）
- **FR-006**: 系统 MUST 支持 Leave-One-Out (LOO) 划分策略：每个用户最后一个交互为测试集，倒数第二个为验证集，其余为训练集
- **FR-007**: 系统 MUST 支持时间比例划分 (TO) 策略：按全局时间戳排序后按配置比例（默认 8:1:1）划分训练/验证/测试集
- **FR-008**: 系统 MUST 对训练集序列执行滑动窗口数据增强，从每个用户序列的第 2 个交互开始逐步滑动生成 (历史序列, 目标物品) 样本对，支持配置最长序列长度（默认 20）用于截断历史序列
- **FR-009**: 系统 MUST 定义物品 Tokenizer 的抽象接口，包含：tokenize 方法（物品 ID → 离散 token 序列）和 detokenize 方法（离散 token 序列 → 物品 ID）。该接口 MUST 不限定 embedding 来源（语义/协同/混合）和量化方法（RQ-VAE/PQ/K-means 等），具体实现自行负责加载所需的预计算 embedding
- **FR-010**: 系统 MUST 提供默认的透传 Tokenizer（PassthroughTokenizer），将物品整数 ID 直接作为单个 token 返回
- **FR-011**: 系统 MUST 提供文本 embedding 生成工具，使用预训练语言模型对物品元数据文本生成向量表示，默认拼接 title + brand + categories 字段，支持通过配置自定义字段组合，存储到 data/interim/ 目录下供 ItemTokenizer 实现消费
- **FR-012**: 系统 MUST 将最终训练数据（包含交互序列和物品标识）存储到 data/processed/，格式兼容 HuggingFace Datasets
- **FR-013**: 系统 MUST 支持根据商品元数据中的图片 URL 批量下载图片，支持断点续传和错误跳过
- **FR-014**: 系统 MUST 通过配置文件（YAML）驱动全部处理参数，包括：数据集名称与版本、K-core 阈值、划分策略与比例、最长序列长度、物品 Tokenizer 选择
- **FR-015**: 系统 MUST 在处理过程中输出关键统计信息：原始交互数、过滤后交互数、用户数、物品数、平均序列长度、训练/验证/测试样本数

### Key Entities

- **Interaction（交互记录）**: 一次用户-物品交互事件，包含用户标识、物品标识、时间戳、评分、评论正文（reviewText/text）和评论标题/摘要（summary/title）。是原始数据到序列构建的基本单元。
- **UserSequence（用户交互序列）**: 单个用户的按时间排序的物品交互序列，是序列推荐的核心数据结构。
- **ItemMetadata（物品元数据）**: 物品的描述性信息，包含标题、品类、品牌、描述、图片 URL 等。用于生成 LLM 训练数据中的物品文本表示。
- **TrainingSample（训练样本）**: 最终用于 LLM 训练的单条数据，包含历史交互序列和目标物品。每个物品同时携带文本信息（如 title）和离散 token 标识，下游训练任务决定使用哪种表示。
- **ItemTokenizer（物品 Tokenizer）**: 将物品整数 ID 转换为一组离散 token 的抽象接口。具体实现可消费预计算的 embedding（文本 embedding 由本流水线工具生成，协同 embedding 由外部训练流程提供），并通过量化方法（RQ-VAE、PQ、K-means 等）生成离散 token。支持可插拔实现。
- **ItemTextEmbedding（物品文本 Embedding）**: 使用预训练语言模型对物品元数据文本（默认拼接 title + brand + categories，可配置）生成的向量表示。作为中间产物存储在 data/interim/，供 ItemTokenizer 实现消费。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 完整的数据处理流水线（从原始数据到最终训练数据）可以在单机上对 10 万条交互记录在 10 分钟内完成处理
- **SC-002**: 所有数据处理步骤的输出结果在相同配置和随机种子下 100% 可复现
- **SC-003**: LOO 和 TO 两种划分策略产出的训练/验证/测试集之间零数据泄露（验证：测试集中所有交互的时间戳均不早于训练集中最晚的时间戳）
- **SC-004**: 滑动窗口增强后的训练样本数量相比原始训练序列数量增加至少 3 倍（基于平均序列长度 > 15 的数据集）
- **SC-005**: 系统支持 Amazon2015 和 Amazon2023 两种格式的数据集，无需修改核心处理逻辑
- **SC-006**: 新增一种物品 Tokenizer 实现仅需继承抽象接口并实现 tokenize/detokenize 方法，无需修改数据处理流水线的其他代码
- **SC-007**: 所有核心模块（数据加载、K-core 过滤、序列构建、数据划分、滑动窗口、最终数据生成）均有对应的单元测试，测试覆盖率不低于 80%
