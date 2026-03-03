# Research: SAE Item Tokenizer

**Feature Branch**: `006-item-tokenizer-sae`  
**Date**: 2026-03-01

## 1. SAE 模型实现策略

**Decision**: 在项目内部独立实现 JumpReLU SAE，不依赖 SAELens 包

**Rationale**:
- SAELens 是为 LLM 内部激活分析设计的完整框架，包含大量与本项目无关的功能（hooking、activation caching、wandb 集成等）
- 本项目仅需 JumpReLU SAE 的核心前向/反向传播逻辑（约 200 行代码），引入整个 SAELens 违反 Constitution VII 简洁性原则
- 项目内实现可精确控制接口，与 `ItemTokenizer` 抽象无缝对接
- 参考 SAELens 的 `JumpReLUSAE` + `JumpReLUTrainingSAE` 作为架构蓝本

**Alternatives Considered**:
- 直接依赖 SAELens 包：引入大量无关依赖（transformer_lens, wandb 等），增加安装复杂度
- 使用 TopK SAE 替代 JumpReLU：TopK 直接约束激活数量，但 JumpReLU 的可学习阈值在训练中提供更好的特征学习质量（参见 DeepMind JumpReLU 论文）

## 2. 训练循环策略

**Decision**: 使用简单 PyTorch 训练循环（for epoch → for batch → loss.backward()）

**Rationale**:
- SAE 训练是标准的重构任务（MSE + 稀疏性损失），不需要 LightningModule 或 HF Trainer 的复杂编排
- 已有的 RQ-VAE tokenizer 使用 `pytorch-lightning.Trainer`，但 SAE 训练逻辑更简单（无 codebook EMA、无 dead code replacement 需要 Lightning hooks）
- 简单循环更透明、更易调试，且便于未来扩展（如添加 dead feature 检测）
- Constitution VII 要求最小必要原则

**Alternatives Considered**:
- PyTorch Lightning：可行但对 SAE 训练而言过度封装，增加不必要的抽象层
- 复制 SAELens 的 `SAETrainer`：太复杂（wandb、activation scaling、gradient scaler 等），大部分功能用不到

## 3. 损失函数设计

**Decision**: MSE 重构损失 + L0 稀疏性损失（Step 函数 STE 梯度），与 SAELens JumpReLU 一致

**Rationale**:
- `total_loss = MSE(x, x_hat) + l0_coefficient * L0(hidden_pre, threshold)`
- MSE：`||x - decode(encode(x))||²`，标准重构目标
- L0 稀疏性：使用 `Step.apply(hidden_pre, threshold, bandwidth)` 近似可微 L0，通过 rectangle 函数提供 STE 梯度
- `l0_coefficient` 控制稀疏性强度，通过配置文件调节
- 这是 Google DeepMind 的标准 JumpReLU 训练方案

**Alternatives Considered**:
- Anthropic 的 tanh 稀疏性损失：SAELens 也支持，但 step 模式更直接、更标准
- 额外的 pre-activation loss（Anthropic dead feature 辅助）：初始版本不需要，可作为后续优化

## 4. 接口映射：ItemTokenizer 属性与 SAE 概念

**Decision**: `num_codebooks` → `top_k`，`codebook_size` → `d_sae`

**Rationale**:
- `ItemTokenizer` 接口定义了 `num_codebooks`（SID 中的 code 数量）和 `codebook_size`（每个位置的词表大小）
- 对于 RQ 方法：`num_codebooks` = 残差量化层数，`codebook_size` = 每层的 cluster 数
- 对于 SAE：`num_codebooks` = top_k（选取的激活特征数），`codebook_size` = d_sae（总概念词表大小）
- 这种映射保持了语义一致性：SID 由 `num_codebooks` 个 code 组成，每个 code 的取值范围是 `[0, codebook_size)`
- 用户可通过 `item_tokenizer.num_codebooks` 和 `item_tokenizer.codebook_size` 配置，也可通过 `item_tokenizer.params.top_k` 和 `item_tokenizer.params.d_sae` 显式覆盖

**Alternatives Considered**:
- 只通过 `params` 传递 `d_sae`/`top_k`：可行但不利用已有的 config 字段
- 新增 config 字段：违反简洁性，且破坏已有配置结构

## 5. SID Code 排序策略

**Decision**: 按激活值降序排列（最强激活在前）

**Rationale**:
- `_build_sid_map` 为每个位置分配 level label（a, b, c...），位置具有隐含的重要性排序
- 对 RQ 方法，位置 0 是最粗粒度（最重要）；对 SAE，位置 0 应是最强激活（最显著概念）
- 降序排列使 collision resolution 的 `append_level` 策略更合理：追加的是次要概念
- 有利于下游 LLM 学习：最重要的概念出现在前面

**Alternatives Considered**:
- 按特征索引升序：不保留重要性信息
- 不排序（按原始激活顺序）：不确定性大，不利于一致性

## 6. 模型持久化格式

**Decision**: safetensors 保存权重 + JSON 保存超参数

**Rationale**:
- `safetensors` 是 HuggingFace 生态的标准格式，安全且加载快
- JSON 保存 `d_in`, `d_sae`, `top_k` 等超参数，`load()` 时可重建模型
- 与项目已有模式一致（RQ-VAE 使用 `torch.save` + `json`，但 safetensors 更安全）

**Alternatives Considered**:
- `torch.save`：可行但 safetensors 更安全（无 pickle 风险）
- HuggingFace `save_pretrained`：SAE 不是 PreTrainedModel，强行适配不自然

## 7. 新增依赖分析

**Decision**: 新增 `safetensors` 依赖到 `pyproject.toml`

**Rationale**:
- `safetensors` 用于模型权重保存/加载，是 HuggingFace 生态标准
- 其他所有依赖（torch, datasets, loguru, numpy）均已存在
- SAE 实现不需要 pytorch-lightning（与 RQ-VAE 不同），训练循环用纯 PyTorch

**Alternatives Considered**:
- 不引入 safetensors，使用 torch.save：可行但不符合 HuggingFace 最佳实践
