"""SFT data building and training subsystem."""

from .base import SFTTaskBuilder, get_sft_task_builder, register_sft_task
from .builder import SFTDatasetBuilder
from .callbacks import RecMetricsCallback
from .collator import convert_to_conversational
from .config import LoRAConfig, SFTTrainingConfig
from .dataset import add_sid_tokens_to_tokenizer, load_sft_dataset
from .evaluator import RecEvalResult, RecEvaluator
from .index2item import Index2ItemTaskBuilder
from .item2index import Item2IndexTaskBuilder
from .seqrec import SeqRecTaskBuilder
from .trainer import SFTRecTrainer

__all__ = [
    "SFTTaskBuilder",
    "SFTDatasetBuilder",
    "SeqRecTaskBuilder",
    "Item2IndexTaskBuilder",
    "Index2ItemTaskBuilder",
    "get_sft_task_builder",
    "register_sft_task",
    "SFTTrainingConfig",
    "LoRAConfig",
    "SFTRecTrainer",
    "RecEvaluator",
    "RecEvalResult",
    "RecMetricsCallback",
    "add_sid_tokens_to_tokenizer",
    "load_sft_dataset",
    "convert_to_conversational",
]
