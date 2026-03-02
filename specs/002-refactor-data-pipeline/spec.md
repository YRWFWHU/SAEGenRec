# Feature Specification: 重构数据处理管道 — 解耦 Tokenizer 并新增负采样

**Feature Branch**: `002-refactor-data-pipeline`  
**Created**: 2026-03-02  
**Status**: Draft  
**Input**: User description: "帮我修改当前的数据加载模块，我理想的数据流是，首先将raw data处理为序列推荐的形式，包括k-core过滤，滑动窗口增强，LOO，TO划分，以及对于重排序的负样本采样，将这些数据保存在/interim路径下，在这一阶段不涉及tokenizer。"

## Clarifications

### Session 2026-03-02

- Q: "为不同的数据集编写不同的脚本"具体指什么形式？ → A: 为每个数据集/场景编写独立的 YAML 配置文件，通过 `make data-filter CONFIG=...` / `make data-split CONFIG=...` 调用，无需每个数据集单独编写脚本。
- Q: 阶段 2（split → augment → negative_sampling）输出是否按 split_strategy 隔离目录？ → A: 是，阶段 2 输出到 `data/interim/{dataset}/{category}/{split_strategy}/`，不同划分策略的结果共存。
- Q: Make 目标如何体现两阶段分离？ → A: 拆为两个独立 Make 目标：`make data-filter`（阶段 1：load → filter → sequence）和 `make data-split`（阶段 2：split → augment → negative_sampling），均通过 `CONFIG` 指定 YAML 配置文件。
- Q: stats.json 如何组织？ → A: 各阶段独立 stats.json：阶段 1 写入 `{dataset}/{category}/stats.json`（过滤统计），阶段 2 写入 `{dataset}/{category}/{split_strategy}/stats.json`（划分与采样统计）。
- Q: split_ratio 与 split_strategy 的关系？ → A: `split_ratio` 仅在 `split_strategy=to` 时生效，LOO 策略不使用该参数。配置文件中可保留 `split_ratio` 字段但 LOO 时被忽略。

## Constraints

- **最小修改原则**: 在现有代码基础上做最小改动，复用已有的 DatasetLoader 注册表、K-core 过滤、序列构建等模块。
- **Make 命令接口**: 用户通过两个 `make` 目标调用管道：`make data-filter` 和 `make data-split`，所有参数从 `CONFIG` 指定的 YAML 文件中读取，无需在 Make 命令中逐一传参。示例：`make data-filter CONFIG=configs/amazon2015_beauty.yaml`，`make data-split CONFIG=configs/amazon2023_fashion.yaml`。如需临时覆盖个别参数，可直接使用 CLI `--option`。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 端到端数据预处理（无 Tokenizer）(Priority: P1)

作为研究人员，我希望将原始 Amazon 评论数据一键处理为序列推荐所需的标准格式（包含 K-core 过滤、序列构建、数据划分、滑动窗口增强），所有结果保存在 `data/interim/` 下，整个过程不涉及任何 tokenizer 操作，以便后续灵活选择不同的 tokenization 方案。

**Why this priority**: 这是管道的核心价值 —— 将数据处理与 tokenization 解耦，使同一份预处理数据可被多种下游任务复用。

**Independent Test**: 使用 Amazon 2015 Beauty 配置运行管道，验证 `data/interim/` 下产出完整的序列数据（过滤后交互、用户序列、train/valid/test 划分、增强后训练样本），且不包含任何 token 相关字段。

**Acceptance Scenarios**:

1. **Given** 原始 Amazon 2015 Beauty 数据在 `data/raw/` 下，**When** 运行预处理管道并指定 LOO 划分策略，**Then** 在 `data/interim/{dataset}/{category}/` 下产出过滤后交互、用户序列、ID 映射（阶段 1），在 `data/interim/{dataset}/{category}/loo/` 下产出 train/valid/test 序列划分及滑动窗口增强后的训练样本（阶段 2），所有数据不包含 token 字段。
2. **Given** 原始数据已就位，**When** 运行预处理管道并指定 TO 划分策略和 `[0.8, 0.1, 0.1]` 比例，**Then** 数据按全局时间戳比例划分，输出结构与 LOO 一致。
3. **Given** 预处理管道完成，**When** 检查输出目录，**Then** 所有 HuggingFace Dataset 的 schema 中不包含 `*_tokens` 字段。

---

### User Story 2 - 重排序负采样 (Priority: P2)

作为研究人员，我希望在数据预处理阶段对每条训练/评估样本自动采样负样本（用户未交互过的商品），以支持重排序任务的对比学习，负样本数量可通过配置控制。

**Why this priority**: 负采样是重排序任务的关键数据准备步骤，直接影响模型训练效果，但依赖于 P1 的序列构建和划分结果。

**Independent Test**: 在 P1 预处理完成后，运行负采样步骤，验证每条样本附带了指定数量的负样本商品 ID，且所有负样本确实未出现在该用户的交互历史中。

**Acceptance Scenarios**:

1. **Given** 预处理管道已完成序列构建和划分，**When** 运行负采样步骤并设定 `num_negatives=99`，**Then** 每条训练/验证/测试样本附带 99 个负样本商品 ID。
2. **Given** 采样结果，**When** 检查任意样本的负样本列表，**Then** 其中没有任何商品出现在该用户的历史交互序列中。
3. **Given** 配置中设置了随机种子 `seed=42`，**When** 使用相同种子运行两次负采样，**Then** 两次产出的负样本完全一致。

---

### User Story 3 - 灵活的步骤选择与增量运行 (Priority: P3)

作为研究人员，我希望能够单独运行管道中的某些步骤（如仅重新执行负采样而不重跑 K-core 过滤），以节省重复计算时间。

**Why this priority**: 大型数据集的 K-core 过滤和序列构建耗时较长，支持增量运行能显著提升迭代效率。

**Independent Test**: 先运行完整管道，然后仅指定 `--step negative_sampling` 重新运行，验证只有负采样结果被更新，其他中间数据保持不变。

**Acceptance Scenarios**:

1. **Given** 完整管道已运行过，**When** 仅指定 `--step split --step augment` 重新运行，**Then** 只有划分和增强结果被刷新，K-core 过滤和序列构建结果不变。
2. **Given** 完整管道已运行过，**When** 仅指定 `--step negative_sampling` 并更改 `num_negatives` 配置，**Then** 仅负采样结果被更新。

---

### Edge Cases

- 当 K-core 过滤后商品数量过少（如 < `num_negatives`），负采样如何处理？应采样所有可用的非交互商品并记录警告。
- 当用户交互了几乎所有商品，可用负样本不足 `num_negatives` 时，应采样所有可用负样本并在样本中记录实际采样数量。
- 当原始数据为空或 K-core 过滤后无数据时，管道应优雅终止并输出空数据集和警告日志。
- 滑动窗口增强对序列长度 < 2 的用户不产出样本。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 管道 MUST 分为两个阶段执行：**阶段 1（数据过滤）**：`load → filter → sequence`，输出保存到 `data/interim/{dataset}/{category}/`；**阶段 2（数据划分）**：`split → augment → negative_sampling`，输出保存到 `data/interim/{dataset}/{category}/{split_strategy}/`。两阶段可独立运行，切换划分策略时无需重跑阶段 1。
- **FR-002**: 管道 MUST 在整个预处理阶段不引入任何 tokenizer 操作，输出 schema 中不包含 `*_tokens` 字段。
- **FR-003**: `augment` 步骤 MUST 使用滑动窗口方式从训练序列生成 `(history_item_ids, target_item_id)` 样本对，历史序列长度不超过 `max_seq_len`。
- **FR-004**: 增强后的训练样本 MUST 包含商品标题等文本信息（从 item_metadata 中查询），用于后续 LLM 训练。
- **FR-005**: `negative_sampling` 步骤 MUST 为每条样本（train/valid/test）随机采样 `num_negatives` 个该用户未交互过的商品 ID。
- **FR-006**: 负采样 MUST 支持设置随机种子以保证可复现性。
- **FR-007**: 管道 MUST 支持 LOO 和 TO 两种划分策略，通过配置切换。LOO 策略不使用 `split_ratio` 参数；TO 策略 MUST 接受 `split_ratio`（默认 `[0.8, 0.1, 0.1]`，须满足 sum=1.0）。
- **FR-008**: 管道 MUST 支持通过 `--step` 参数选择性运行单个或多个步骤。
- **FR-011**: 管道 MUST 通过两个 `make` 目标调用：`make data-filter`（阶段 1）和 `make data-split`（阶段 2），均通过 `CONFIG` 变量指定 YAML 配置文件路径（默认 `configs/default.yaml`），所有管道参数从配置文件读取。CLI `--option` 可用于临时覆盖配置中的个别参数。
- **FR-009**: 每个步骤的输出 MUST 以 HuggingFace Datasets（Arrow 格式）存储。
- **FR-010**: 管道 MUST 输出处理统计信息：阶段 1 将过滤统计（过滤前后交互数、用户数、商品数、序列长度）写入 `data/interim/{dataset}/{category}/stats.json`；阶段 2 将划分与采样统计（样本数、负采样数量、警告数）写入 `data/interim/{dataset}/{category}/{split_strategy}/stats.json`。

### Key Entities

- **Interaction**: 用户-商品交互记录（user_id, item_id, timestamp, rating, review_text, review_summary）
- **UserSequence**: 按时间排序的用户行为序列（user_id, item_ids, timestamps, ratings, ...）
- **InterimSample**: augment 步骤产出的样本（user_id, history_item_ids, history_item_titles, target_item_id, target_item_title），不含 token 字段
- **NegativeSample**: InterimSample 扩展，附加负采样信息（negative_item_ids, negative_item_titles）
- **IDMap**: 原始 ID 与连续整数 ID 的映射关系
- **ItemMetadata**: 商品元数据（title, brand, categories, description, price, image_url）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 预处理管道能在 5 分钟内完成 Amazon 2015 Beauty 数据集（约 200 万条交互）的全流程处理（load → negative_sampling）。
- **SC-002**: 输出数据的 schema 中不包含任何 `*_tokens` 字段，确认 tokenizer 完全解耦。
- **SC-003**: 负采样结果中 100% 的负样本商品未出现在对应用户的交互历史中。
- **SC-004**: 使用相同配置和随机种子运行两次管道，所有输出数据完全一致（位级别相同）。
- **SC-005**: 管道的核心模块单元测试覆盖率 ≥ 80%。
- **SC-006**: 支持通过 `--step` 参数单独运行任意步骤，耗时仅为该步骤本身的处理时间（不重复执行前序步骤）。
- **SC-007**: 当负样本数量不足 `num_negatives` 时，管道输出警告日志并记录实际采样数量，不中断执行。

## Assumptions

- 原始数据已下载并放置在 `data/raw/` 对应目录下。
- `data/interim/` 目录用于存放所有预处理结果，替代当前 `data/processed/` 中的部分功能。
- 负采样默认数量为 99（可通过配置覆盖），采用均匀随机采样策略。
- 当前阶段仅支持均匀随机负采样，基于流行度的采样策略可在后续迭代中扩展。
- 现有 `DatasetLoader` 注册表机制和 K-core 过滤逻辑保持不变，仅调整管道编排和输出结构。
- 在现有代码基础上做最小改动，优先复用已有模块。
