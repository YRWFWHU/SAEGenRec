# SAEGenRec

Generative Recommendation 数据处理管道 — 将 Amazon 商品评论数据集转换为大语言模型（LLM）可消费的训练数据。

## 项目目标

为生成式推荐系统研究提供一套**可复现、可扩展**的数据预处理工具链，覆盖从原始评论数据到 LLM 训练样本的完整链路。

## 核心能力

| 能力 | 说明 |
|------|------|
| 多格式数据加载 | Amazon 2015（JSON）、Amazon 2023（JSONL），通过注册表模式可扩展 |
| K-core 稠密化 | 迭代过滤低频用户/商品，确保交互密度 |
| 序列构建 | 按时间排序构建用户行为序列，分配连续整数 ID |
| 数据划分 | Leave-One-Out（LOO）和 Temporal Order（TO）两种策略 |
| 滑动窗口增强 | 从用户序列生成多个 (history, target) 训练样本 |
| 文本嵌入 | 基于 sentence-transformers 生成商品文本向量 |
| YAML 配置驱动 | 所有参数通过 YAML 文件控制，支持环境间快速切换 |

## 技术栈

- **Python 3.11** + **HuggingFace Datasets**（Arrow 列式存储）
- **sentence-transformers** / **PyTorch** 用于文本嵌入
- **typer** CLI 框架 + **loguru** 日志
- **pytest** 测试（100 个用例，94% 覆盖率）

## 导航

- [快速开始](getting-started.md) — 环境搭建与首次运行
- [数据管道](data-pipeline.md) — 管道架构与处理流程
- [配置参考](configuration.md) — YAML 配置项详解
- [API 参考](api.md) — 模块与函数接口文档
