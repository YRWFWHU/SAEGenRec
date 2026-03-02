# SAEGenRec

Generative Recommendation 数据处理管道 — 将 Amazon 商品评论数据集转换为大语言模型（LLM）可消费的训练数据，并生成多种商品 embedding 表示。

## 项目目标

为生成式推荐系统研究提供一套**可复现、可扩展**的数据预处理工具链，覆盖从原始评论数据到 LLM 训练样本及多模态 embedding 的完整链路。

## 核心能力

| 能力 | 说明 |
|------|------|
| 两阶段管道 | 数据过滤与数据划分解耦，切换划分策略无需重跑过滤 |
| 多格式数据加载 | Amazon 2015（JSON）、Amazon 2023（JSONL），通过注册表模式可扩展 |
| K-core 稠密化 | 迭代过滤低频用户/商品，确保交互密度 |
| 序列构建 | 按时间排序构建用户行为序列，分配连续整数 ID |
| 数据划分 | Leave-One-Out（LOO）和 Temporal Order（TO）两种策略 |
| 滑动窗口增强 | 从用户序列生成多个 (history, target) 训练样本（tokenizer 无关） |
| 负采样 | 为每条样本采样未交互商品作为负样本，支持可复现种子 |
| 语义 Embedding | 基于预训练语言模型（sentence-transformers）对商品元数据文本字段提取语义向量，支持可配置 L2 归一化 |
| 协同 Embedding | 通过 PyTorch Lightning 训练序列推荐模型（SASRec），从 `nn.Embedding` 权重提取协同过滤向量，支持 BPR / CE 损失函数 |
| YAML 配置驱动 | 所有参数通过 YAML 文件控制，支持 CLI 覆盖 |

## 技术栈

- **Python 3.11** + **HuggingFace Datasets**（Arrow 列式存储）
- **sentence-transformers** / **PyTorch** 用于语义 embedding
- **PyTorch Lightning** 用于协同 embedding 训练（SASRec）
- **typer** CLI 框架 + **loguru** 日志
- **pytest** 测试

## 导航

- [快速开始](getting-started.md) — 环境搭建与首次运行
- [数据管道](data-pipeline.md) — 管道架构与处理流程
- [配置参考](configuration.md) — YAML 配置项详解
- [API 参考](api.md) — 模块与函数接口文档
