# Contract: SFTTaskBuilder 抽象接口

**Branch**: `004-genrec-architecture` | **Date**: 2026-03-02

## Interface Definition

```python
from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Dataset


class SFTTaskBuilder(ABC):
    """SFT 任务构建器抽象接口。

    将推荐数据转换为特定 SFT 任务的 Alpaca 格式指令数据。
    每个子类负责一种任务类型。
    """

    @abstractmethod
    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        config: dict,
    ) -> Dataset:
        """构建该任务类型的 SFT 数据。

        Args:
            stage1_dir: Stage 1 interim 目录（包含 item_metadata/、item_id_map/ 等）。
            stage2_dir: Stage 2 interim 目录（包含 train_sequences/ 等）。
            sid_map: item_sid_map HuggingFace Dataset（已从 data/processed/ 加载）。
            config: 任务配置字典。

        Returns:
            HuggingFace Dataset，每行包含 task_type、instruction、input、output。
        """
        ...

    @property
    @abstractmethod
    def task_type(self) -> str:
        """任务类型标识符。"""
        ...

    def load_templates(self, template_file: Path) -> list[dict]:
        """从 YAML 文件加载 prompt 模板。

        提供默认实现，子类可覆写。

        Args:
            template_file: YAML 模板文件路径。

        Returns:
            模板列表，每个模板包含 instruction 和可选 input_template。
        """
        import yaml
        with open(template_file) as f:
            all_templates = yaml.safe_load(f)
        return all_templates.get(self.task_type, [])
```

## Registry

```python
SFT_TASK_REGISTRY: dict[str, type[SFTTaskBuilder]] = {}


def register_sft_task(name: str):
    """装饰器：注册 SFTTaskBuilder 实现。"""
    def decorator(cls: type[SFTTaskBuilder]):
        SFT_TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_sft_task_builder(name: str, **kwargs) -> SFTTaskBuilder:
    """根据名称获取 SFTTaskBuilder 实例。"""
    if name not in SFT_TASK_REGISTRY:
        raise ValueError(
            f"Unknown SFT task: '{name}'. "
            f"Available: {list(SFT_TASK_REGISTRY.keys())}"
        )
    return SFT_TASK_REGISTRY[name](**kwargs)
```

## Implementations

### SeqRecTaskBuilder

```python
@register_sft_task("seqrec")
class SeqRecTaskBuilder(SFTTaskBuilder):
    """序列推荐任务：根据用户历史交互 SID 序列预测下一个物品 SID。

    数据来源：仅训练集交互（train_sequences）。
    """

    @property
    def task_type(self) -> str:
        return "seqrec"

    def build(self, stage1_dir, stage2_dir, sid_map, config) -> Dataset:
        """构建 SeqRec SFT 数据。

        从 stage2_dir/train_sequences/ 加载训练序列，
        对每个用户的训练序列：
        1. 取历史物品 SID 序列（截断到 max_history_len）
        2. 取目标物品 SID
        3. 随机选一个 prompt 模板填充
        """
        ...
```

### Item2IndexTaskBuilder

```python
@register_sft_task("item2index")
class Item2IndexTaskBuilder(SFTTaskBuilder):
    """物品→SID 任务：根据物品标题/描述预测物品 SID。

    数据来源：全部物品（K-core 过滤后）。
    """

    @property
    def task_type(self) -> str:
        return "item2index"

    def build(self, stage1_dir, stage2_dir, sid_map, config) -> Dataset:
        """从 stage1_dir/item_metadata/ 加载物品信息，取标题/描述作为 input，SID 作为 output。"""
        ...
```

### Index2ItemTaskBuilder

```python
@register_sft_task("index2item")
class Index2ItemTaskBuilder(SFTTaskBuilder):
    """SID→物品 任务：根据 SID 预测物品标题/描述。

    数据来源：全部物品（K-core 过滤后）。
    """

    @property
    def task_type(self) -> str:
        return "index2item"

    def build(self, stage1_dir, stage2_dir, sid_map, config) -> Dataset:
        """从 stage1_dir/item_metadata/ 加载物品信息，取 SID 作为 input，标题/描述作为 output。"""
        ...
```

## Orchestrator: SFTDatasetBuilder

```python
class SFTDatasetBuilder:
    """多任务 SFT 数据编排器。

    根据配置启用/禁用各任务类型，控制采样比例，合并为最终数据集。
    """

    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        output_dir: Path,
        config: dict,
    ) -> Dataset:
        """构建混合 SFT 数据集。

        1. 遍历配置中启用的任务类型
        2. 对每个任务调用对应 TaskBuilder.build(stage1_dir, stage2_dir, sid_map, config)
        3. 按 task_weights 采样（如果配置了权重）
        4. 合并、shuffle、保存到 output_dir/sft_data/

        Returns:
            合并后的 SFT HuggingFace Dataset。
        """
        ...
```

## Prompt Template Format (YAML)

```yaml
# configs/templates/sft_prompts.yaml

seqrec:
  - instruction: "Based on the user's interaction history, predict the next item the user might interact with."
    input_template: "User interaction history (in chronological order): {history_sids}"
    output_template: "{target_sid}"
  - instruction: "The user has sequentially interacted with the following items. What item will they engage with next?"
    input_template: "Interaction sequence: {history_sids}"
    output_template: "{target_sid}"
  # ... at least 5 templates per task type

item2index:
  - instruction: "Given the following item information, predict its semantic ID."
    input_template: "Item title: {title}"
    output_template: "{sid_tokens}"
  # ...

index2item:
  - instruction: "Given the following semantic ID, predict the item title."
    input_template: "Semantic ID: {sid_tokens}"
    output_template: "{title}"
  # ...
```

## Output Schema

```python
SFT_FEATURES = Features({
    "task_type": Value("string"),
    "instruction": Value("string"),
    "input": Value("string"),
    "output": Value("string"),
})
```

## Invariants

1. 每条 SFT 记录的 `instruction`、`input`、`output` 均非空字符串
2. `task_type` 值 MUST 属于已注册的任务类型集合
3. SeqRec 任务的历史 SID 序列长度 ≤ `max_history_len`
4. 每种任务类型 MUST 有 ≥5 个 prompt 模板（SC-004）
5. 输出数据集可被 HuggingFace `datasets.load_from_disk` 正常加载
