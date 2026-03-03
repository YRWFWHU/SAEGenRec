# CLI & Config Contract: SAE Item Tokenizer

## CLI 命令（无新增）

SAE tokenizer 复用已有的 `tokenize` CLI 命令，无需新增命令：

```bash
# 与 RQ-VAE/RQ-KMeans 使用完全相同的命令
python -m saegenrec.dataset tokenize configs/default.yaml

# 或通过 Makefile
make data-tokenize
```

## 配置变更

### `configs/default.yaml` — `item_tokenizer` 段

将 `name` 从 `"rqvae"` 切换为 `"sae"` 即可启用 SAE tokenizer：

```yaml
item_tokenizer:
  enabled: true
  name: "sae"                          # 切换为 SAE tokenizer
  num_codebooks: 8                     # → maps to top_k (SID code 数量)
  codebook_size: 8192                  # → maps to d_sae (概念词表大小)
  collision_strategy: "append_level"   # 碰撞处理策略（与 RQ 方法共用）
  sid_token_format: "<s_{level}_{code}>"
  sid_begin_token: "<|sid_begin|>"
  sid_end_token: "<|sid_end|>"
  params:
    # SAE 核心参数（可选，覆盖 num_codebooks/codebook_size）
    # d_sae: 8192                      # 显式指定 SAE 隐藏维度
    # top_k: 8                         # 显式指定 top_k 选取数

    # 训练超参数
    learning_rate: 0.001
    epochs: 50
    batch_size: 256
    device: "cuda"

    # JumpReLU 特有参数
    l0_coefficient: 0.001              # L0 稀疏性损失系数
    jumprelu_bandwidth: 0.05           # STE 梯度 bandwidth
    jumprelu_init_threshold: 0.01      # 阈值初始值
```

### 参数优先级

当 `params` 中显式指定 `d_sae` 或 `top_k` 时，它们优先于 `codebook_size` 和 `num_codebooks`：

| 来源 | `codebook_size` | `params.d_sae` | 实际 d_sae |
|------|----------------|----------------|------------|
| 仅 codebook_size | 8192 | - | 8192 |
| 仅 params.d_sae | 256 (default) | 4096 | 4096 |
| 两者都有 | 8192 | 4096 | 4096 (params 优先) |

`num_codebooks` / `params.top_k` 的优先级逻辑相同。

## 注册表集成

SAE tokenizer 通过 `@register_item_tokenizer("sae")` 自动注册到已有的 tokenizer 注册表。Pipeline 中的调用方式无需变更：

```python
# saegenrec/data/pipeline.py — 已有代码，无需修改
tokenizer = get_item_tokenizer(
    tok_cfg.name,                    # "sae"
    num_codebooks=tok_cfg.num_codebooks,  # → top_k
    codebook_size=tok_cfg.codebook_size,  # → d_sae
    **tok_cfg.params,                # lr, epochs, l0_coefficient, ...
)
sid_map = tokenizer.generate(semantic_dir, collab_dir, modeling_dir, gen_config)
```

## `__init__.py` 导出变更

`saegenrec/modeling/tokenizers/__init__.py` 新增导出：

```python
from saegenrec.modeling.tokenizers.sae import SAETokenizer

__all__ = [
    "ItemTokenizer",
    "RQKMeansTokenizer",
    "RQVAETokenizer",
    "SAETokenizer",         # 新增
    "get_item_tokenizer",
    "register_item_tokenizer",
]
```
