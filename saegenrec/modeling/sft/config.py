"""SFT training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]
    )
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])


@dataclass
class SFTTrainingConfig:
    """Aggregates all SFT training parameters.

    Note: seed is intentionally omitted — the global ``processing.seed``
    from PipelineConfig is used instead to ensure consistency across modules.
    """

    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    sft_data_dir: str | None = None
    sid_map_path: str | None = None
    output_dir: str = "models/sft"

    # Tasks
    tasks: list[str] = field(default_factory=lambda: ["seqrec", "item2index", "index2item"])
    task_weights: dict[str, float] = field(default_factory=dict)

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Evaluation
    eval_steps: int = 100
    rec_eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    eval_top_k: list[int] = field(default_factory=lambda: [1, 5, 10])
    max_new_tokens: int = 32
    constrained_decoding: bool = True
    do_test: bool = True

    # Logging
    report_to: str = "tensorboard"

    def __post_init__(self):
        if self.rec_eval_steps < self.eval_steps:
            raise ValueError(
                f"rec_eval_steps ({self.rec_eval_steps}) must be >= eval_steps ({self.eval_steps})"
            )
        if any(k <= 0 for k in self.eval_top_k):
            raise ValueError(f"All eval_top_k values must be > 0, got {self.eval_top_k}")

    @classmethod
    def from_dict(cls, raw: dict) -> SFTTrainingConfig:
        """Create from a nested dict (YAML ``sft_training`` section)."""
        flat: dict = {}

        # Top-level simple keys
        for key in (
            "model_name_or_path",
            "sft_data_dir",
            "sid_map_path",
            "output_dir",
            "tasks",
            "task_weights",
            "report_to",
        ):
            if key in raw:
                flat[key] = raw[key]

        # LoRA sub-section
        if "lora" in raw:
            flat["lora"] = LoRAConfig(**raw["lora"])

        # Training sub-section
        training = raw.get("training", {})
        _training_keys = {
            "num_epochs": "num_train_epochs",
            "per_device_train_batch_size": "per_device_train_batch_size",
            "per_device_eval_batch_size": "per_device_eval_batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "lr_scheduler_type": "lr_scheduler_type",
            "warmup_ratio": "warmup_ratio",
            "weight_decay": "weight_decay",
            "max_seq_length": "max_seq_length",
            "fp16": "fp16",
            "bf16": "bf16",
            "gradient_checkpointing": "gradient_checkpointing",
        }
        for yaml_key, attr_key in _training_keys.items():
            if yaml_key in training:
                flat[attr_key] = training[yaml_key]

        # Evaluation sub-section
        evaluation = raw.get("evaluation", {})
        _eval_keys = {
            "eval_steps": "eval_steps",
            "rec_eval_steps": "rec_eval_steps",
            "save_steps": "save_steps",
            "save_total_limit": "save_total_limit",
            "eval_top_k": "eval_top_k",
            "max_new_tokens": "max_new_tokens",
            "constrained_decoding": "constrained_decoding",
            "do_test": "do_test",
        }
        for yaml_key, attr_key in _eval_keys.items():
            if yaml_key in evaluation:
                flat[attr_key] = evaluation[yaml_key]

        # Logging sub-section
        logging_cfg = raw.get("logging", {})
        if "report_to" in logging_cfg:
            flat["report_to"] = logging_cfg["report_to"]

        return cls(**flat)
