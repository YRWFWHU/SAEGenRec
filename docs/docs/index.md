# SAEGenRec

生成式推荐系统研究框架 — 从原始 Amazon 商品评论数据到 LLM 可消费的 SFT 训练数据，覆盖数据预处理、商品 embedding、语义 ID 生成、指令数据构建的完整链路。

## 项目目标

为生成式推荐系统研究提供一套**可复现、可扩展**的工具链，覆盖：

1. **数据管道** — 原始数据 → K-core 过滤 → 序列构建 → 数据划分 → 滑动窗口增强 → 负采样
2. **Embedding 生成** — 语义 embedding（sentence-transformers）+ 协同 embedding（SASRec）
3. **物品 Tokenization** — 将 embedding 映射为层次化语义 ID（RQ-VAE / RQ-KMeans）
4. **SFT 数据构建** — 将推荐数据转换为 Alpaca 格式指令微调数据（SeqRec / Item2Index / Index2Item）
5. **生成式推荐模型** — GenRecModel ABC + 约束解码基础设施（SIDTrie + ConstrainedLogitsProcessor）

## 核心能力

| 能力 | 说明 |
|------|------|
| 两阶段管道 | 数据过滤与数据划分解耦，切换划分策略无需重跑过滤 |
| 多格式数据加载 | Amazon 2015（JSON）、Amazon 2023（JSONL），通过注册表模式可扩展 |
| K-core 稠密化 | 迭代过滤低频用户/商品，确保交互密度 |
| 序列构建 | 按时间排序构建用户行为序列，分配连续整数 ID |
| 数据划分 | Leave-One-Out（LOO）和 Temporal Order（TO）两种策略 |
| 滑动窗口增强 | 从用户序列生成多个 (history, target) 训练样本 |
| 负采样 | 为每条样本采样未交互商品作为负样本，支持可复现种子 |
| 语义 Embedding | 基于预训练语言模型对商品元数据提取语义向量 |
| 协同 Embedding | 通过 SASRec 训练提取协同过滤向量 |
| 物品 Tokenization | RQ-VAE / RQ-KMeans 将 embedding 映射为层次化离散 SID |
| 碰撞消解 | append_level / sinkhorn 策略确保所有物品 SID 唯一 |
| SFT 数据构建 | 多任务指令数据（SeqRec + Item2Index + Index2Item），Alpaca 格式 |
| 约束解码 | 基于前缀树（SIDTrie）的 logits processor，确保 LLM 生成有效 SID |
| YAML 配置驱动 | 所有参数通过 YAML 文件控制，支持 CLI 覆盖 |

## 技术栈

- **Python 3.11** + **HuggingFace Datasets**（Arrow 列式存储）
- **sentence-transformers** / **PyTorch** 用于语义 embedding
- **PyTorch Lightning** 用于协同 embedding 训练（SASRec）和 RQ-VAE 训练
- **FAISS** 用于 RQ-KMeans 聚类
- **transformers** 用于约束解码（LogitsProcessor）
- **typer** CLI 框架 + **loguru** 日志
- **pytest** 测试

## 导航

- [快速开始](getting-started.md) — 环境搭建与首次运行
- [数据管道](data-pipeline.md) — 管道架构与处理流程
- [建模子系统](modeling.md) — 物品 Tokenization、SFT 数据构建、约束解码
- [配置参考](configuration.md) — YAML 配置项详解
- [API 参考](api.md) — 模块与函数接口文档
