<!--
Sync Impact Report
==================
- 版本变更: 1.0.0 → 1.1.0 (新增 CCDS 目录结构约束)
- 修改章节:
  - 技术栈约束: 新增 CCDS 目录结构规范
- 新增章节: 无
- 删除章节: 无
- 模板同步状态:
  - .specify/templates/plan-template.md ✅ 无需修改（Constitution Check 为动态填充）
  - .specify/templates/spec-template.md ✅ 无需修改
  - .specify/templates/tasks-template.md ✅ 无需修改
  - .specify/templates/checklist-template.md ✅ 无需修改
- 受影响的现有文件:
  - specs/001-genrec-framework-arch/plan.md ⚠ 项目结构章节需与 CCDS 对齐验证
- 延迟项: 无
-->

# SAEGenRec Constitution

## Core Principles

### I. HuggingFace 设计哲学优先

所有模型、数据处理和训练流程 MUST 遵循 HuggingFace 生态的设计范式：

- 模型 MUST 继承或兼容 `PreTrainedModel` 抽象，提供 `from_pretrained` / `save_pretrained` 接口
- 配置 MUST 使用独立的 Config 类（继承 `PretrainedConfig`），与模型代码解耦
- 训练 MUST 基于 HuggingFace `Trainer` 或兼容的训练循环，支持标准回调机制
- 数据加载 MUST 优先使用 `datasets` 库或兼容的 Arrow 格式
- 新组件 MUST 可通过 HuggingFace 标准接口（Pipeline、Auto Classes）无缝集成

**理由**: 遵循成熟的开源框架设计降低协作门槛，确保与社区工具链的互操作性，便于研究成果的发布与复用。

### II. 可复现性（不可妥协）

每一次实验 MUST 能够被完整复现：

- 所有实验 MUST 通过配置文件（YAML/JSON）驱动，禁止硬编码超参数
- 随机种子 MUST 在配置中显式指定并在训练开始时全局设置（Python、NumPy、PyTorch、CUDA）
- 每次实验 MUST 记录完整的环境信息：依赖版本、GPU 型号、配置快照
- 数据集版本 MUST 明确标识（哈希值或版本号），禁止使用未版本化的数据
- 模型 checkpoint MUST 与对应的配置文件一起保存

**理由**: 可复现性是科研工作的基本要求，不可复现的实验结果不具备学术价值。

### III. 测试驱动

本地正确性测试 MUST 在提交代码前通过：

- 每个核心模块（模型、数据处理、评估指标）MUST 有对应的单元测试
- 测试 MUST 使用小规模合成数据，确保可在本地 单GPU 环境下快速运行（单个测试 < 30秒）
- 模型测试 MUST 至少验证：前向传播形状正确、梯度可计算、`save/load` 一致性
- 数据处理测试 MUST 验证：输入输出格式、边界条件、空数据处理
- 所有测试 MUST 在 `git push` 前通过

**理由**: 本地快速测试能在早期发现逻辑错误，避免浪费昂贵的 GPU 训练资源。

### IV. 双环境分离

本地开发与服务器训练 MUST 明确分离：

- 本地环境专用于：代码编写、单元测试、小规模调试（单 GPU）
- 服务器环境专用于：大规模训练、超参数搜索、完整评估
- 训练脚本 MUST 通过配置文件区分环境，禁止在代码中硬编码服务器路径或设备信息
- 本地与服务器 MUST 使用相同的代码库（通过 GitHub 同步），仅配置不同
- 服务器训练任务 MUST 使用脚本化的启动方式（shell script 或 Makefile），禁止手动逐条命令执行

**理由**: 环境分离确保本地开发的高效性和服务器资源的合理利用，统一代码库避免环境差异导致的不可复现问题。

### V. 模块化与可组合性

框架的各组件 MUST 可独立替换和组合：

- 模型架构、编码器、解码器、推荐策略 MUST 作为独立模块实现
- 模块间的交互 MUST 通过明确定义的接口（抽象基类或 Protocol），禁止隐式耦合
- 新增实验变体 SHOULD 仅需替换或组合现有模块，不应修改核心框架代码
- 每个模块 MUST 可独立实例化和测试，不依赖全局状态
- 配置系统 MUST 支持通过组合不同模块配置来定义完整实验

**理由**: 科研实验需要频繁探索不同组件的组合，模块化设计大幅降低实验迭代成本。

### VI. 版本控制与实验追踪

所有代码和实验 MUST 通过结构化的版本控制管理：

- 代码 MUST 使用 GitHub 进行版本控制，遵循 feature branch 工作流
- 每个实验性功能 MUST 在独立分支上开发，通过 PR 合并到主分支
- 提交信息 MUST 遵循约定式提交（Conventional Commits）格式
- 实验结果（指标、日志）SHOULD 与对应的代码版本（commit hash）关联
- 大文件（数据集、模型 checkpoint）MUST 不直接提交到 Git 仓库，使用 `.gitignore` 排除

**理由**: 结构化的版本控制是多人协作和实验追溯的基础，GitHub 工作流保证代码质量。

### VII. 简洁性

设计和实现 MUST 遵循最小必要原则：

- YAGNI（You Aren't Gonna Need It）：不实现当前实验不需要的功能
- 优先使用 HuggingFace 和 PyTorch 生态的现有工具，避免重复造轮子
- 每个抽象层 MUST 有明确的存在理由，不引入无实际价值的中间层
- 复杂度 MUST 被证明是必要的：新增抽象需说明为何更简单的方案不可行
- 配置项 MUST 有合理的默认值，常见实验场景应能开箱即用

**理由**: 科研代码的核心价值在于实验思路而非工程复杂度，简洁设计加速迭代并降低维护成本。

## 技术栈约束

本项目的技术选型 MUST 遵循以下约束：

- **语言**: Python 3.10+
- **深度学习框架**: PyTorch（通过 HuggingFace Transformers/Accelerate 使用）
- **模型框架**: HuggingFace Transformers（模型定义、训练、推理）
- **分布式训练**: HuggingFace Accelerate
- **数据处理**: HuggingFace Datasets + pandas（必要时）
- **配置管理**: dataclasses 或 HuggingFace TrainingArguments 风格
- **测试框架**: pytest
- **代码质量**: ruff（linting + formatting）
- **依赖管理**: pyproject.toml + pip
- **版本控制**: Git + GitHub

禁止引入与上述技术栈功能重叠的替代方案，除非经过明确的技术评审并记录在案。

### 目录结构约束

项目顶层目录 MUST 遵循 Cookiecutter Data Science（CCDS）推荐的文件结构：

```text
├── data/
│   ├── external/       <- 第三方数据源
│   ├── interim/        <- 中间转换数据
│   ├── processed/      <- 最终建模用数据集
│   └── raw/            <- 原始不可变数据
├── docs/               <- 项目文档
├── models/             <- 训练后的模型、预测结果或模型摘要
├── notebooks/          <- Jupyter notebooks
├── references/         <- 参考资料、数据字典、论文等
├── reports/            <- 生成的分析报告
│   └── figures/        <- 报告用图表
├── saegenrec/          <- 项目源代码包
├── tests/              <- 测试代码
├── configs/            <- 实验配置文件（CCDS 扩展）
├── scripts/            <- 训练/评估入口脚本（CCDS 扩展）
├── pyproject.toml      <- 项目配置与依赖
├── Makefile            <- 便捷命令
└── README.md           <- 项目说明
```

- 新增文件 MUST 放置在上述对应目录中，禁止在项目根目录下随意创建新的顶层目录
- `configs/` 和 `scripts/` 是对 CCDS 标准结构的扩展，用于满足实验配置管理和训练脚本入口的需求
- `data/raw/` 中的数据 MUST 保持不可变，所有处理后的数据放在 `data/processed/`
- `notebooks/` 命名规范：`{序号}-{作者缩写}-{简短描述}`，如 `1.0-yrw-data-exploration`

## 开发工作流

日常开发 MUST 遵循以下流程：

1. **本地开发阶段**:
   - 从 `master` 分支创建 feature branch
   - 编写/修改代码，同步更新对应测试
   - 本地运行 `pytest` 确保所有测试通过
   - 使用小规模数据进行本地调试验证

2. **代码提交阶段**:
   - 确保所有测试通过后 `git push` 到远程分支
   - 创建 Pull Request，描述变更内容和实验目的
   - 代码合并到 `master`

3. **服务器训练阶段**:
   - 在服务器上 `git pull` 最新代码
   - 通过配置文件指定训练参数和数据路径
   - 使用标准化脚本启动训练任务
   - 训练完成后记录实验结果并关联 commit hash

4. **结果回收阶段**:
   - 分析实验结果，整理关键指标
   - 将分析结论记录到对应的实验文档中
   - 根据结果决定下一步实验方向

## Governance

本 Constitution 是 SAEGenRec 项目的最高指导文件，所有开发实践 MUST 与之保持一致：

- 修订本 Constitution MUST 提供修订理由、影响范围说明和迁移计划
- 版本号遵循语义化版本（SemVer）：MAJOR（原则删除/不兼容变更）、MINOR（新增原则/实质性扩展）、PATCH（措辞/格式修正）
- 所有 PR MUST 验证变更是否符合 Constitution 原则
- 与 Constitution 冲突的代码 MUST 在合并前解决冲突或申请原则豁免
- 运行时开发指南参见项目 `README.md` 和 `docs/` 目录

**Version**: 1.1.0 | **Ratified**: 2026-02-28 | **Last Amended**: 2026-03-01
