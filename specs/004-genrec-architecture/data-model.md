# Data Model: 生成式推荐架构

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02

## Entities

### ItemTokenizer（物品 Tokenizer）

物品 embedding → 层次化离散码（SID）的转换器。位于 `saegenrec/modeling/tokenizers/`。

| Field | Type | Description |
|-------|------|-------------|
| name | str | 注册表名称（如 `"rqvae"`, `"rqkmeans"`） |
| num_codebooks | int | 码本层数（默认 4） |
| codebook_size | int | 每层码本大小（默认 256） |
| embedding_dim | int | 输入 embedding 维度（由数据推断） |
| collision_strategy | str | 碰撞消解策略：`"append_level"` \| `"sinkhorn"` |

**状态转换**: `未训练` → `train()` → `已训练` → `encode()` → 输出 codes → `resolve_collisions()` → 唯一 SID

**关系**: 消费 `data/interim/.../item_semantic_embeddings/`（和/或协同 embedding），产出 `data/processed/.../item_sid_map/`

### SID Map（语义 ID 映射表）

每个物品的层次化离散编码。存储在 `data/processed/{dataset}/{category}/item_sid_map/`。

| Field | Type | Description |
|-------|------|-------------|
| item_id | int32 | 映射后的物品 ID |
| codes | Sequence(int32) | 各层码本索引序列（长度 = num_codebooks 或 num_codebooks+1） |
| sid_tokens | str | SID token 字符串（如 `"<s_a_42><s_b_103><s_c_7><s_d_255>"`） |

**HF Dataset Features**:

```python
SID_MAP_FEATURES = Features({
    "item_id": Value("int32"),
    "codes": Sequence(Value("int32")),
    "sid_tokens": Value("string"),
})
```

**唯一性约束**: codes 序列在全部物品中 MUST 唯一（碰撞消解后）

### RQ-VAE Model（RQ-VAE 模型）

PyTorch Lightning Module，位于 `saegenrec/modeling/tokenizers/models/rqvae_model.py`。

| Field | Type | Description |
|-------|------|-------------|
| encoder | nn.Sequential | MLP 编码器（embedding_dim → hidden_dim → latent_dim） |
| decoder | nn.Sequential | MLP 解码器（latent_dim → hidden_dim → embedding_dim） |
| quantizer | ResidualVectorQuantizer | 残差向量量化器 |
| codebooks | nn.ParameterList | 各层码本（codebook_size × latent_dim / num_codebooks） |

**损失函数**:
- 重建损失: MSE(input, reconstructed)
- 量化损失: codebook_loss + β × commitment_loss
- 总损失: recon_loss + α × quant_loss

### SFTTaskBuilder（SFT 任务构建器）

将推荐数据转换为特定 SFT 任务指令数据的构建器。位于 `saegenrec/modeling/sft/`。

| Field | Type | Description |
|-------|------|-------------|
| task_type | str | 任务类型名称（如 `"seqrec"`, `"item2index"`, `"index2item"`） |
| templates | list[dict] | 从 YAML 加载的 prompt 模板列表 |
| seed | int | 随机种子（模板采样用） |

**关系**: 消费 `data/processed/.../item_sid_map/`、`data/interim/.../train_sequences/`（SeqRec）或 `item_metadata/`（Item2Index/Index2Item），产出 `data/processed/.../sft_data/`

### SFT Record（SFT 指令记录）

Alpaca 格式的 LLM 微调数据记录。存储在 `data/processed/{dataset}/{category}/sft_data/`。

| Field | Type | Description |
|-------|------|-------------|
| task_type | str | 任务类型（seqrec / item2index / index2item） |
| instruction | str | 指令文本 |
| input | str | 输入上下文 |
| output | str | 期望输出 |

**HF Dataset Features**:

```python
SFT_FEATURES = Features({
    "task_type": Value("string"),
    "instruction": Value("string"),
    "input": Value("string"),
    "output": Value("string"),
})
```

### GenRecModel（生成式推荐模型）

LLM 模型本体的抽象接口。位于 `saegenrec/modeling/genrec/`。遵循 HuggingFace 设计哲学。

| Field | Type | Description |
|-------|------|-------------|
| base_model_name | str | 基础 LLM 名称（如 `"Qwen/Qwen2.5-0.5B"`） |
| lora_config | dict | LoRA 配置参数 |
| training_strategy | str | `"sft"` \| `"rl"`（预留） |
| sid_token_list | list[str] | SID 特殊 token 列表（用于词表扩展） |

**方法签名**:
- `train(dataset, training_args) -> TrainOutput`
- `generate(input_ids, **kwargs) -> list[str]`
- `evaluate(dataset, metrics) -> dict[str, float]`
- `save_pretrained(path)` / `from_pretrained(path)`

### SIDTrie（SID 前缀树）

约束解码用的前缀树。位于 `saegenrec/modeling/decoding/trie.py`。

| Field | Type | Description |
|-------|------|-------------|
| root | TrieNode | 前缀树根节点 |
| sid_token_ids | dict[str, int] | SID token → tokenizer ID 映射 |

**用途**: 在 LLM 生成时，通过前缀树约束生成的 token 序列必须是有效 SID。确保推荐结果可映射回真实物品。

**关系**: 从 `item_sid_map` 构建，被 `GenRecModel.generate()` 使用

## Entity Relationships

```text
data/interim/                             data/processed/
─────────────                             ───────────────
item_semantic_embeddings ──┐
                           ├──→ ItemTokenizer ──→ item_sid_map ──┐
item_collaborative_embeddings ┘   (modeling/      (processed/)   │
                                   tokenizers/)                  │
                                                                 │
train_sequences ────────────────────────┐                        │
item_metadata ──────────────────────────┤                        │
                                        ▼                        ▼
                                 SFTTaskBuilder ←──── item_sid_map
                                 (modeling/sft/)
                                        │
                                        ▼
                                    sft_data ──────→ GenRecModel.train()
                                    (processed/)     (modeling/genrec/)
                                                          │
                                 item_sid_map ──→ SIDTrie  │
                                                     │     │
                                                     ▼     ▼
                                              GenRecModel.generate()
                                              + constrained decoding
                                              (modeling/decoding/)
```

## Configuration Extensions

### ItemTokenizerConfig (新增)

```python
@dataclass
class ItemTokenizerConfig:
    enabled: bool = False
    name: str = "rqvae"                    # "rqvae" | "rqkmeans"
    num_codebooks: int = 4                 # 码本层数
    codebook_size: int = 256               # 每层码本大小
    collision_strategy: str = "append_level"  # "append_level" | "sinkhorn"
    sid_token_format: str = "<s_{level}_{code}>"
    sid_begin_token: str = "<|sid_begin|>"
    sid_end_token: str = "<|sid_end|>"
    params: dict = field(default_factory=dict)
```

### SFTBuilderConfig (新增)

```python
@dataclass
class SFTBuilderConfig:
    enabled: bool = False
    tasks: list[str] = field(default_factory=lambda: ["seqrec", "item2index", "index2item"])
    task_weights: dict[str, float] = field(default_factory=dict)
    template_file: str = "configs/templates/sft_prompts.yaml"
    max_history_len: int = 20
    seed: int = 42
```

### PipelineConfig 扩展

```python
@dataclass
class PipelineConfig:
    # ... 已有字段 ...
    item_tokenizer: ItemTokenizerConfig = field(default_factory=ItemTokenizerConfig)
    sft_builder: SFTBuilderConfig = field(default_factory=SFTBuilderConfig)
```

## Output Paths

| 产物 | 路径 | 说明 |
|------|------|------|
| item_sid_map | `data/processed/{dataset}/{category}/item_sid_map/` | 物品 SID 映射表 |
| tokenizer_model | `data/processed/{dataset}/{category}/tokenizer_model/` | 训练好的 tokenizer 权重 |
| sft_data | `data/processed/{dataset}/{category}/sft_data/` | SFT 指令数据集 |

## Validation Rules

1. `item_sid_map` 中的 `codes` 序列 MUST 全局唯一
2. `codes` 中每个值 MUST 在 `[0, codebook_size)` 范围内
3. SFT 数据的 `instruction`、`input`、`output` MUST 非空
4. SFT `task_type` MUST 是已注册的任务类型之一
5. `item_sid_map` 中的 `item_id` MUST 与 `item_id_map` 中的 `mapped_id` 一一对应
6. SIDTrie 构建后，`trie.search(valid_sid_prefix)` MUST 返回非空候选集
