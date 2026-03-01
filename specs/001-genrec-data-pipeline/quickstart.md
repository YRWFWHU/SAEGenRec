# Quickstart: 生成式推荐数据处理流水线

**Branch**: `001-genrec-data-pipeline` | **Date**: 2026-03-01

## 前置条件

1. Python 3.11 环境已就绪
2. 项目已安装：`pip install -e .`
3. 原始数据已放置在 `data/raw/` 目录下

## 安装

```bash
# 克隆仓库并切换到 feature 分支
git clone <repo-url> && cd saegenrec
git checkout 001-genrec-data-pipeline

# 安装依赖
pip install -e .
```

## 快速运行

### 使用默认配置处理 Amazon2015 Baby 数据集

```bash
# 运行完整流水线（LOO 划分 + 透传 Tokenizer）
python -m saegenrec.dataset process configs/default.yaml
```

### 使用自定义配置

```bash
# 编辑配置文件
cp configs/examples/amazon2015_baby.yaml configs/my_config.yaml
# 修改 my_config.yaml 中的参数...

# 运行
python -m saegenrec.dataset process configs/my_config.yaml
```

### 分步运行

```bash
# 仅加载和过滤
python -m saegenrec.dataset process configs/default.yaml --step load --step filter --step sequence

# 仅划分和增强
python -m saegenrec.dataset process configs/default.yaml --step split --step augment --step generate

# 生成文本 embedding（可选，需要 GPU 加速）
python -m saegenrec.dataset process configs/default.yaml --step embed
```

## 配置文件示例

### Amazon2015 Baby (LOO 划分)

```yaml
# configs/examples/amazon2015_baby.yaml
dataset:
  name: "amazon2015"
  category: "Baby"

processing:
  kcore_threshold: 5
  split_strategy: "loo"
  max_seq_len: 20

tokenizer:
  name: "passthrough"
```

### Amazon2023 All_Beauty (TO 划分)

```yaml
# configs/examples/amazon2023_beauty.yaml
dataset:
  name: "amazon2023"
  category: "All_Beauty"

processing:
  kcore_threshold: 5
  split_strategy: "to"
  split_ratio: [0.8, 0.1, 0.1]
  max_seq_len: 20

embedding:
  enabled: true
  model_name: "all-MiniLM-L6-v2"
  device: "cuda"
```

## 输出说明

运行完成后，数据将存储在以下位置：

```
data/interim/amazon2015/Baby/
├── interactions/        # 统一格式的交互记录
├── user_sequences/      # 用户交互序列
├── item_metadata/       # 物品元数据
├── user_id_map/         # 用户 ID 映射
├── item_id_map/         # 物品 ID 映射
└── stats.json           # 处理统计

data/processed/amazon2015/Baby/loo/
├── train/               # 训练集（滑动窗口增强后）
├── valid/               # 验证集
├── test/                # 测试集
└── stats.json           # 划分统计
```

## 在代码中使用输出数据

```python
from datasets import load_from_disk

# 加载训练集
train_ds = load_from_disk("data/processed/amazon2015/Baby/loo/train")
print(f"训练样本数: {len(train_ds)}")
print(f"样本示例: {train_ds[0]}")

# 加载物品元数据
metadata = load_from_disk("data/interim/amazon2015/Baby/item_metadata")
print(f"物品数: {len(metadata)}")

# 加载 ID 映射
item_map = load_from_disk("data/interim/amazon2015/Baby/item_id_map")
```

## 运行测试

```bash
# 运行所有数据处理相关测试
python -m pytest tests/unit/data/ -v

# 运行特定模块测试
python -m pytest tests/unit/data/test_kcore.py -v
```

## 添加新数据集支持

1. 在 `saegenrec/data/loaders/` 下创建新的 Loader 文件
2. 继承 `DatasetLoader` 抽象基类
3. 实现 `load_interactions()` 和 `load_item_metadata()` 方法
4. 使用 `@register_loader("your_dataset_name")` 装饰器注册
5. 在 `configs/` 下创建对应的配置文件

## 添加新 ItemTokenizer 实现

1. 在 `saegenrec/data/tokenizers/` 下创建新的 Tokenizer 文件
2. 继承 `ItemTokenizer` 抽象基类
3. 实现 `tokenize()`、`detokenize()`、`vocab_size`、`token_length`
4. 使用 `@register_tokenizer("your_tokenizer_name")` 装饰器注册
5. 在配置文件中指定 `tokenizer.name: "your_tokenizer_name"`
