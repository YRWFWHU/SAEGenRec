# SAEGenRec

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Generative Recommendation 数据处理管道，将 Amazon 商品评论数据集转换为 LLM 训练数据。

## 功能概览

- **多版本数据加载**：支持 Amazon 2015（JSON）和 Amazon 2023（JSONL）格式
- **K-core 迭代过滤**：移除低频用户和商品，保证数据稠密性
- **用户交互序列构建**：按时间排序生成用户行为序列
- **数据划分**：Leave-One-Out（LOO）和 Temporal Order（TO）两种策略
- **滑动窗口增强**：生成 (history, target) 训练样本
- **文本嵌入**：基于 sentence-transformers 生成商品文本嵌入
- **可扩展架构**：通过注册表模式支持自定义 DatasetLoader 和 ItemTokenizer

## 快速开始

```bash
# 创建环境
conda create --name saegenrec python=3.11 -y
conda activate saegenrec
pip install -e .

# 运行数据管道
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml
```

## 项目结构

```
├── configs/                  <- YAML 管道配置文件
│   ├── default.yaml
│   └── examples/
├── data/
│   ├── raw/                  <- 原始数据（不可变）
│   ├── interim/              <- 中间处理结果
│   └── processed/            <- 最终训练数据
├── docs/                     <- mkdocs 文档
├── saegenrec/
│   ├── config.py             <- 全局配置
│   ├── dataset.py            <- CLI 入口（typer）
│   └── data/                 <- 数据处理模块
│       ├── config.py          <- 管道配置 dataclasses
│       ├── pipeline.py        <- 管道编排器
│       ├── schemas.py         <- HuggingFace Dataset schemas
│       ├── loaders/           <- 数据加载器（Amazon2015 / Amazon2023）
│       ├── processors/        <- 数据处理器（kcore / sequence / split / augment / final / images）
│       ├── tokenizers/        <- 商品 tokenizer（passthrough）
│       └── embeddings/        <- 文本嵌入生成
├── scripts/                  <- Shell 脚本
├── specs/                    <- 功能规格文档
├── tests/                    <- 单元测试（100 个，94% 覆盖率）
├── Makefile                  <- 自动化命令
└── pyproject.toml            <- 项目元数据与依赖
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `make data-process` | 使用默认配置运行完整管道 |
| `make data-embed` | 仅运行文本嵌入步骤 |
| `make test` | 运行全部测试 |
| `make lint` | 代码风格检查 |
| `make format` | 自动格式化代码 |

## 文档

```bash
cd docs && mkdocs serve
```

详细文档请参阅 `docs/` 目录。

--------
