# 建模子系统

## 概述

建模子系统（`saegenrec.modeling`）承接数据管道的产物，实现生成式推荐的核心建模组件：

```
item_semantic_embeddings/    ──→  ItemTokenizer  ──→  item_sid_map/
                                                            │
train/ + item_metadata/      ──→  SFTDatasetBuilder  ──→  sft_data/
                                                            │
                                                            ▼
                                              LLM SFT 训练（待实现）
                                                            │
                                                            ▼
                              SIDTrie + ConstrainedLogitsProcessor
                                                            │
                                                            ▼
                                                    推荐结果（item_id）
```

## 物品 Tokenization

### 为什么需要 SID？

生成式推荐的核心思想是让 LLM 直接"说出"推荐结果。但 LLM 的词表中没有商品 ID，因此需要将每个商品映射为一组离散 token —— 语义 ID（Semantic ID, SID）。

SID 具备两个关键特性：

1. **唯一性** — 每个商品对应唯一的 SID，可无歧义地反查
2. **层次性** — 相似商品共享 SID 前缀，使 LLM 能利用语义结构

### 工作流程

```
Embedding (384-dim float)
    │
    ▼  MLP Encoder（仅 RQ-VAE）
Latent (64-dim float)
    │
    ▼  残差量化（4 层 × 256 码本）
Raw Codes [42, 103, 7, 255]
    │
    ▼  碰撞消解
Unique Codes [42, 103, 7, 255] 或 [42, 103, 7, 255, 0]
    │
    ▼  Token 格式化
SID String "<s_a_42><s_b_103><s_c_7><s_d_255>"
```

### RQ-VAE Tokenizer

RQ-VAE（Residual Quantization VAE）使用可学习的编码器将 embedding 投影到潜空间后逐层量化：

1. **编码器**: `input_dim → hidden_dim → latent_dim` MLP
2. **残差量化**: 每层从码本中找到最近向量，减去后传入下一层
3. **解码器**: 从量化后的潜表示重建原始 embedding
4. **损失函数**: 重建损失（MSE）+ 量化损失（commitment loss）

**防坍缩机制**（码本坍缩是 VQ-VAE 的常见问题）：

- **数据初始化**: 首个 batch 时从编码器输出 K-Means 初始化码本
- **EMA 更新**: 码本向量通过指数移动平均更新，而非梯度下降
- **死码替换**: 使用次数过低的码本条目从当前 batch 重新采样

### RQ-KMeans Tokenizer

RQ-KMeans 不使用神经网络，直接在 embedding 空间逐层聚类：

1. 对原始 embedding 执行均衡约束 KMeans（保证码本利用率）
2. 计算残差 = embedding - 聚类中心
3. 对残差重复步骤 1，直到达到指定层数

基于 FAISS 和 `k-means-constrained`，可在 CPU 上运行。

### 碰撞消解

量化后可能出现多个商品映射到相同 SID 的碰撞。当前支持的策略：

**append_level**: 对碰撞组中的每个商品追加一个消歧索引作为额外层级。SID 长度可变（碰撞商品比无碰撞商品多一层）。

## SFT 数据构建

### 任务设计

SFT 数据将推荐问题转化为 LLM 可理解的自然语言指令：

#### SeqRec（序列推荐）

```
Instruction: 根据用户的购物历史，推荐下一个可能感兴趣的商品。
Input: 用户浏览历史: <s_a_42><s_b_103>... <s_a_7><s_b_55>...
Output: <s_a_12><s_b_88><s_c_3><s_d_201>
```

数据来源：Stage 2 滑动窗口增强后的 `train/` 数据。每条增强样本直接提供 `history_item_ids` → `target_item_id`，充分利用数据增强带来的多样性。

#### Item2Index（物品 → SID）

```
Instruction: 给出以下商品的语义编码。
Input: "Revlon ColorStay Liquid Eye Pen"
Output: <s_a_42><s_b_103><s_c_7><s_d_255>
```

让 LLM 学会将商品文本描述映射到 SID 空间。

#### Index2Item（SID → 物品）

```
Instruction: 以下语义编码对应什么商品？
Input: <s_a_42><s_b_103><s_c_7><s_d_255>
Output: "Revlon ColorStay Liquid Eye Pen"
```

Item2Index 的反向任务，帮助 LLM 建立 SID 与商品之间的双向映射。

### Prompt 模板

模板存储在 `configs/templates/sft_prompts.yaml`，每种任务至少 5 个模板。构建时随机采样模板以增加指令多样性。新增模板只需编辑 YAML 文件，无需修改代码。

### 数据规模示例

以 Amazon Beauty 数据集（K-core=5）为例：

| 任务 | 样本数 | 说明 |
|------|--------|------|
| SeqRec | ~131K | 来自 22K 用户的滑动窗口增强 |
| Item2Index | ~12K | 全部商品 |
| Index2Item | ~12K | 全部商品 |
| **总计** | **~155K** | |

## 约束解码

### 问题

LLM 生成时可能输出无效的 SID token 组合（不对应任何商品）。

### 解决方案

**SIDTrie**: 将所有有效 SID 构建为前缀树。生成每一步时，查询当前 SID 前缀对应的合法下一步 token。

**SIDConstrainedLogitsProcessor**: HuggingFace `LogitsProcessor` 子类，将非法 token 的 logits 设为 `-inf`，确保 LLM 只能生成有效的 SID 序列。

```python
# 使用示例
from saegenrec.modeling.decoding.trie import SIDTrie
from saegenrec.modeling.decoding.constrained import SIDConstrainedLogitsProcessor

trie = SIDTrie.from_sid_map(sid_map, tokenizer)
processor = SIDConstrainedLogitsProcessor(trie, sid_begin_id, sid_end_id)

outputs = model.generate(input_ids, logits_processor=[processor])
```

## GenRecModel 接口

`GenRecModel` 定义了生成式推荐模型的标准接口，遵循 HuggingFace 设计哲学：

```python
class GenRecModel(ABC):
    def train(self, dataset, training_args) -> dict: ...
    def generate(self, input_text, **kwargs) -> list[str]: ...
    def evaluate(self, dataset, metrics) -> dict[str, float]: ...
    def save_pretrained(self, path) -> None: ...
    def from_pretrained(cls, path, **kwargs) -> GenRecModel: ...
```

当前为抽象接口，具体 LLM 训练实现将在后续迭代中完成。
