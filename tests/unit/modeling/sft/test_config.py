"""Tests for SFTTrainingConfig dataclass."""

from __future__ import annotations

import pytest

from saegenrec.modeling.sft.config import LoRAConfig, SFTTrainingConfig


class TestSFTTrainingConfig:
    def test_defaults(self):
        cfg = SFTTrainingConfig()
        assert cfg.model_name_or_path == "Qwen/Qwen2.5-0.5B"
        assert cfg.num_train_epochs == 3
        assert cfg.lora.enabled is True
        assert cfg.lora.r == 8
        assert cfg.eval_top_k == [1, 5, 10]
        assert cfg.report_to == "tensorboard"

    def test_rec_eval_steps_validation(self):
        with pytest.raises(ValueError, match="rec_eval_steps"):
            SFTTrainingConfig(eval_steps=500, rec_eval_steps=100)

    def test_eval_top_k_validation(self):
        with pytest.raises(ValueError, match="eval_top_k"):
            SFTTrainingConfig(eval_top_k=[1, 0, 5])

    def test_from_dict_full(self):
        raw = {
            "model_name_or_path": "meta-llama/Llama-3-8B",
            "sft_data_dir": "/data/sft",
            "sid_map_path": "/data/sid_map",
            "output_dir": "/output",
            "tasks": ["seqrec"],
            "task_weights": {"seqrec": 1.0},
            "lora": {
                "enabled": False,
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj"],
                "modules_to_save": ["embed_tokens"],
            },
            "training": {
                "num_epochs": 5,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-5,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0.05,
                "weight_decay": 0.0,
                "max_seq_length": 1024,
                "fp16": True,
                "bf16": False,
                "gradient_checkpointing": False,
            },
            "evaluation": {
                "eval_steps": 50,
                "rec_eval_steps": 200,
                "save_steps": 200,
                "save_total_limit": 5,
                "eval_top_k": [5, 10, 20],
                "max_new_tokens": 64,
                "constrained_decoding": False,
                "do_test": False,
            },
            "logging": {"report_to": "wandb"},
        }
        cfg = SFTTrainingConfig.from_dict(raw)
        assert cfg.model_name_or_path == "meta-llama/Llama-3-8B"
        assert cfg.num_train_epochs == 5
        assert cfg.lora.enabled is False
        assert cfg.lora.r == 16
        assert cfg.learning_rate == 2e-5
        assert cfg.eval_steps == 50
        assert cfg.rec_eval_steps == 200
        assert cfg.eval_top_k == [5, 10, 20]
        assert cfg.report_to == "wandb"
        assert cfg.fp16 is True
        assert cfg.bf16 is False
        assert cfg.do_test is False

    def test_from_dict_minimal(self):
        cfg = SFTTrainingConfig.from_dict({})
        assert cfg.model_name_or_path == "Qwen/Qwen2.5-0.5B"
        assert cfg.lora.enabled is True

    def test_from_dict_partial_training_section(self):
        raw = {"training": {"num_epochs": 10}}
        cfg = SFTTrainingConfig.from_dict(raw)
        assert cfg.num_train_epochs == 10
        assert cfg.learning_rate == 1e-4  # default preserved


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.enabled is True
        assert cfg.r == 8
        assert "q_proj" in cfg.target_modules
        assert "embed_tokens" in cfg.modules_to_save
