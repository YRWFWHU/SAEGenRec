"""Configuration dataclass for generative recommendation models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenRecConfig:
    """Configuration for generative recommendation models."""

    base_model_name: str = "Qwen/Qwen2.5-0.5B"
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] | None = None
    training_strategy: str = "sft"
    sid_tokens_path: str | None = None
