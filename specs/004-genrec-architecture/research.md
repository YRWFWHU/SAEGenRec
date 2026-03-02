# Research: 生成式推荐架构

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02

## R1: RQ-VAE 架构选型

**Decision**: MLP Encoder → Residual Vector Quantization → MLP Decoder，使用 PyTorch Lightning 训练

**Rationale**:
- 参考 Align3GR 和 MiniOneRec 的 RQ-VAE 实现，MLP 编码器足以处理预计算的 embedding 输入（无需卷积或 Transformer）
- RQ（残差量化）逐层量化残差，每层使用独立码本，层数和码本大小可配置
- PyTorch Lightning 提供标准训练循环、checkpoint、logging，符合 Constitution III（测试驱动）和 II（可复现性）
- 损失函数：重建损失（MSE）+ 量化损失（codebook loss + commitment loss）

**Alternatives considered**:
- VQ-VAE（非残差）：码本利用率低，无法生成层次化 SID
- Transformer Encoder：输入是固定长度 embedding，MLP 足够；Transformer 增加不必要的复杂度（Constitution VII）

## R2: RQ-KMeans 架构选型

**Decision**: 使用 FAISS KMeans 做逐层残差聚类，可选 KMeansConstrained 做均衡约束

**Rationale**:
- FAISS KMeans 在大规模向量上性能优异，GPU 加速
- 逐层残差：对每层做 KMeans，计算残差后传入下一层（参考 OpenOneRec ResKmeans）
- 均衡约束：使用 `k-means-constrained` 库的 `KMeansConstrained`，通过 `size_min`/`size_max` 保证码本利用率（参考 MiniOneRec rqkmeans_constrained）
- 无需 GPU 训练，CPU 即可运行

**Alternatives considered**:
- FAISS ResidualQuantizer：一步到位但不支持自定义均衡约束
- scikit-learn KMeans：不支持 GPU 加速，大规模物品集性能不足
- RQ-KMeans+ (GPR)：需要额外 MLP warm-start，复杂度高，留作后续迭代

## R3: 碰撞消解策略

**Decision**: 实现两种可配置策略：Sinkhorn 重分配（SID 层数固定）和 Append-Level 去重（SID 层数可变）

**Rationale**:
- **Sinkhorn 重分配**：在量化时使用 Sinkhorn 算法做软分配，迫使码本利用均匀化。参考 Align3GR 的 `sinkhorn_algorithm`，对距离矩阵做行列归一化迭代。适用于 RQ-VAE 训练时（可微分）和 RQ-KMeans 后处理
- **Append-Level 去重**：对碰撞的 SID 组追加一个序号层级。参考 MiniOneRec 的 `deal_with_deduplicate`，使用 rank-over-group 追加序号。实现简单，保证唯一性，但 SID 长度不固定
- 默认策略：Append-Level（简单可靠），高级用户可选 Sinkhorn

**Alternatives considered**:
- 迭代重推理（Align3GR）：对碰撞物品反复用 Sinkhorn 重推理直到无碰撞。不确定收敛，最多 20 次迭代。不作为独立策略，Sinkhorn 策略内部包含此逻辑

## R4: SFT 数据构建架构

**Decision**: SFTTaskBuilder ABC + 注册表，每种任务类型一个实现，由 SFTDatasetBuilder 编排混合

**Rationale**:
- 与 ItemTokenizer、DatasetLoader 保持一致的 ABC + 注册表模式
- 每个 TaskBuilder 负责单一任务类型的数据生成：输入为原始数据 + SID 映射，输出为 Alpaca 格式记录
- SFTDatasetBuilder 作为编排器，根据配置启用/禁用各任务类型，控制采样比例，合并为最终数据集
- Prompt 模板存储在外部 YAML 文件中，运行时加载，随机采样（FR-013）

**Alternatives considered**:
- 单一 Builder 类处理所有任务：违反模块化原则，新增任务需修改核心代码
- 独立脚本：不利于管道集成和测试

## R5: SID Token 格式

**Decision**: 使用 `<s_{level}_{code}>` 格式，带 `<|sid_begin|>` 和 `<|sid_end|>` 界定符

**Rationale**:
- 参考 OpenOneRec 的 SID token 格式，每层使用字母前缀区分层级
- 具体格式：`<|sid_begin|><s_a_42><s_b_103><s_c_7><s_d_255><|sid_end|>`
- 前缀使用 `a`-`z` 字母（最多支持 26 层，远超实际需要的 3-5 层）
- 格式可通过配置定制（token 前缀、界定符等），满足 FR-010

**Alternatives considered**:
- 纯数字格式 `<42_103_7_255>`：层级不明确，解析困难
- 参考 Align3GR 的 `<a_{}>` 格式：与我们的选择本质相同

## R6: ItemTokenizer ABC 重新设计

**Decision**: 重写 spec-001 的 ItemTokenizer 契约，扩展为支持训练和批量推理的完整接口

**Rationale**:
- spec-001 的 `ItemTokenizer` 仅定义了 `tokenize(item_id) -> list[int]` 的简单接口，适用于 PassthroughTokenizer
- 新的 RQ-VAE/RQ-KMeans tokenizer 需要：`train(embeddings)` 训练、`encode(embeddings) -> codes` 批量编码、碰撞消解、SID token 生成
- 新接口需要同时接收语义和协同 embedding 路径（FR-007），具体使用由实现决定
- 保留与旧 `tokenize(item_id)` 的兼容性：训练后 encode 所有物品生成 SID map，tokenize 仅做查表

**Alternatives considered**:
- 保留旧接口并扩展：旧接口的 `tokenize(item_id)` 语义与新需求不匹配（新需求是 embedding → codes）

## R7: 约束解码策略

**Decision**: 基于 SID Prefix Trie 的约束 beam search，定义接口并实现 Trie 数据结构

**Rationale**:
- 生成式推荐的 LLM 在推理时生成 SID token 序列，必须约束为有效 SID（存在于 `item_sid_map` 中）
- Prefix Trie 从 `item_sid_map` 的所有 SID 构建，在每步解码时限制候选 token 集合
- 参考 FM-Index 和 Constrained Beam Search 的通用实现，Trie 方案简单且与 HuggingFace `generate()` 的 `LogitsProcessor` 接口兼容
- 本期实现 Trie 数据结构 + `SIDConstrainedLogitsProcessor`（兼容 `transformers.LogitsProcessor`），LLM 生成调用留作后续

**Alternatives considered**:
- 无约束生成 + 后验最近邻匹配：生成质量差，无法保证输出有效 SID
- FM-Index：复杂度高，当前 SID 规模（~100K）用 Trie 足够

## R8: 模块位置与代码组织

**Decision**: ItemTokenizer、SFT、GenRec、Decoding 四个子系统统一放在 `saegenrec/modeling/` 下

**Rationale**:
- 这四个模块共同构成生成式推荐的核心模型栈，不是数据预处理步骤
- `saegenrec/data/` 专注于原始数据到 embedding 的管道（loaders → processors → embeddings）
- `saegenrec/modeling/` 消费 embedding 产出建模数据和模型，符合 CCDS 的职责分离
- 输出路径使用 `data/processed/`（CCDS 的 "最终建模用数据集"），而非 `data/interim/`

**Alternatives considered**:
- 放在 `saegenrec/data/tokenizers/` 和 `saegenrec/data/sft/`：将模型级组件混入数据管道，职责不清

## R9: 新增依赖评估

**Decision**: 新增 `faiss-cpu`（或 `faiss-gpu`）和 `k-means-constrained`

**Rationale**:
- **faiss-cpu/faiss-gpu**: RQ-KMeans 的 KMeans 聚类核心依赖。FAISS 是 Meta 开发的向量相似度搜索库，成熟稳定。GPU 版本可显著加速大规模聚类
- **k-means-constrained** (必需): 均衡 KMeans 约束，保证码本利用率（FR-009 MUST）
- 不引入 POT (Python Optimal Transport)：Sinkhorn 算法实现简单（~20 行），直接内联实现避免额外依赖

**Alternatives considered**:
- 使用 scikit-learn KMeans：性能不足，不支持 GPU
- 使用 POT 库做 Sinkhorn：仅用一个函数，不值得引入整个库
