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
# 处理 Amazon 2015 Beauty 数据集（全流程）
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml

# 处理 Amazon 2023 Fashion 数据集
python -m saegenrec.dataset process configs/examples/amazon2023_fashion.yaml
```

### 两阶段分步运行

管道拆分为两个独立阶段，切换划分策略无需重跑数据过滤：

```bash
# 阶段 1: 数据过滤（load → filter → sequence）
make data-filter CONFIG=configs/examples/amazon2015_beauty.yaml

# 阶段 2: 数据划分（split → augment → negative_sampling）
make data-split CONFIG=configs/examples/amazon2015_beauty.yaml
```

### 指定步骤运行

```bash
# 仅执行加载和过滤
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml \
    --step load --step filter

# 仅执行负采样（需先完成前序步骤）
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml \
    --step negative_sampling

# 仅执行文本嵌入（需先完成前序步骤）
python -m saegenrec.dataset process configs/examples/amazon2015_beauty.yaml \
    --step embed
```

### CLI 参数覆盖

可在命令行临时覆盖配置文件中的参数：

```bash
# 使用 TO 策略 + 自定义比例
python -m saegenrec.dataset process configs/default.yaml \
    --split-strategy to --split-ratio 0.8 0.1 0.1

# 修改负采样数量
python -m saegenrec.dataset process configs/default.yaml \
    --num-negatives 199 --seed 0
```

### 使用 Makefile

```bash
make data-filter             # 阶段 1: 数据过滤（使用 default.yaml）
make data-split              # 阶段 2: 数据划分 + 负采样
make data-process            # 遗留: 完整管道
make data-embed              # 仅文本嵌入步骤
make data-download-images    # 下载商品图片
```

所有 Make 目标通过 `CONFIG` 变量指定配置文件（默认 `configs/default.yaml`）：

```bash
make data-filter CONFIG=configs/examples/amazon2023_beauty.yaml
make data-split CONFIG=configs/examples/amazon2023_beauty.yaml
```

## 输出结构

运行完成后，数据按以下结构组织：

```
data/interim/{dataset}/{category}/               ← 阶段 1 输出
├── raw_interactions/                             ← 原始交互（HuggingFace Dataset）
├── interactions/                                 ← K-core 过滤后交互
├── user_sequences/                               ← 用户行为序列
├── item_metadata/                                ← 商品元数据
├── user_id_map/                                  ← 原始 → 连续整数 ID 映射
├── item_id_map/                                  ← 同上
├── stats.json                                    ← 阶段 1 统计
│
└── {split_strategy}/                             ← 阶段 2 输出（按划分策略分目录）
    ├── train_sequences/                          ← 训练序列
    ├── valid_sequences/                          ← 验证序列
    ├── test_sequences/                           ← 测试序列
    ├── train/                                    ← 训练集（InterimSample / NegativeSample）
    ├── valid/                                    ← 验证集
    ├── test/                                     ← 测试集
    └── stats.json                                ← 阶段 2 统计
```

所有中间数据均以 HuggingFace Datasets（Apache Arrow 格式）存储，支持内存映射和高效随机访问。

## 下一步

- 了解 [管道架构](data-pipeline.md) 中每个步骤的工作原理
- 查看 [配置参考](configuration.md) 定制处理参数
- 阅读 [API 参考](api.md) 了解如何扩展加载器或 tokenizer
