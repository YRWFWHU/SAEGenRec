# 快速开始

## 环境搭建

### 1. 创建 Conda 环境

```bash
make create_environment
conda activate saegenrec
```

### 2. 安装依赖

```bash
make requirements
# 等价于: pip install -e .
```

### 3. 验证安装

```bash
make test
# 应输出: 100 passed
```

## 准备原始数据

将 Amazon 评论数据放入 `data/raw/` 目录，遵循以下命名约定：

### Amazon 2015 格式

```
data/raw/Amazon2015/{Category}/{Category}.json
data/raw/Amazon2015/{Category}/meta_{Category}.json
```

文件为每行一个 JSON 对象的 `.json` 文件（metadata 文件可能是 Python dict 字面量格式，加载器会自动处理）。

### Amazon 2023 格式

```
data/raw/Amazon2023/{Category}/{Category}.jsonl
data/raw/Amazon2023/{Category}/meta_{Category}.jsonl
```

文件为标准 JSON Lines 格式。

## 首次运行

### 使用示例配置

```bash
# 处理 Amazon 2015 Beauty 数据集
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml

# 处理 Amazon 2023 Fashion 数据集
python -m saegenrec.dataset process configs/examples/amazon2023_fashion.yaml
```

### 指定步骤运行

```bash
# 仅执行加载和过滤
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml \
    --step load --step filter

# 仅执行文本嵌入（需先完成前序步骤）
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml \
    --step embed
```

### 使用 Makefile

```bash
make data-process           # 使用 default.yaml 运行完整管道
make data-embed             # 仅文本嵌入步骤
make data-download-images   # 下载商品图片
```

## 输出结构

运行完成后，数据按以下结构组织：

```
data/
├── interim/{dataset}/{category}/          <- 中间数据
│   ├── raw_interactions/                  <- 原始交互（HuggingFace Dataset）
│   ├── interactions/                      <- K-core 过滤后交互
│   ├── user_sequences/                    <- 用户行为序列
│   ├── item_metadata/                     <- 商品元数据
│   ├── user_id_map/                       <- 原始 → 连续整数 ID 映射
│   ├── item_id_map/                       <- 同上
│   └── stats.json                         <- 处理统计
└── processed/{dataset}/{category}/{strategy}/  <- 最终训练数据
    ├── train/                             <- 训练集（TrainingSample 格式）
    ├── valid/                             <- 验证集
    ├── test/                              <- 测试集
    ├── train_sequences/                   <- 训练序列
    ├── valid_sequences/                   <- 验证序列
    ├── test_sequences/                    <- 测试序列
    └── stats.json                         <- 最终统计
```

所有中间和最终数据均以 HuggingFace Datasets（Apache Arrow 格式）存储，支持内存映射和高效随机访问。

## 下一步

- 了解 [管道架构](data-pipeline.md) 中每个步骤的工作原理
- 查看 [配置参考](configuration.md) 定制处理参数
- 阅读 [API 参考](api.md) 了解如何扩展加载器或 tokenizer
