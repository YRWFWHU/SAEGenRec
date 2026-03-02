"""Tests for SFTRecTrainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from saegenrec.data.schemas import SFT_FEATURES
from saegenrec.modeling.sft.config import SFTTrainingConfig


@pytest.fixture
def sft_data_dir(tmp_path: Path, mock_sid_map) -> Path:
    """Create mock SFT data directory with train/valid/test splits."""
    data_dir = tmp_path / "sft_data"
    for split in ("train", "valid", "test"):
        ds = Dataset.from_dict(
            {
                "task_type": ["seqrec"] * 4,
                "instruction": ["Predict next."] * 4,
                "input": [f"History: <s_a_0> item{i}" for i in range(4)],
                "output": ["<s_a_0><s_b_1>"] * 4,
            },
            features=SFT_FEATURES,
        )
        ds.save_to_disk(str(data_dir / split))
    return data_dir


@pytest.fixture
def sid_map_dir(tmp_path: Path, mock_sid_map) -> Path:
    """Save mock SID map to disk."""
    path = tmp_path / "sid_map"
    mock_sid_map.save_to_disk(str(path))
    return path


@pytest.fixture
def training_config(sft_data_dir: Path, sid_map_dir: Path, tmp_path: Path) -> SFTTrainingConfig:
    return SFTTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        sft_data_dir=str(sft_data_dir),
        sid_map_path=str(sid_map_dir),
        output_dir=str(tmp_path / "output"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        eval_steps=2,
        rec_eval_steps=4,
        save_steps=4,
        max_seq_length=128,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        report_to="none",
        lora=SFTTrainingConfig.__dataclass_fields__["lora"].default_factory(),
    )


class TestSFTRecTrainerSetup:
    def test_setup_loads_model_and_tokenizer(self, training_config):
        from saegenrec.modeling.sft.trainer import SFTRecTrainer

        trainer = SFTRecTrainer(training_config, seed=42)
        trainer.setup()

        assert trainer.tokenizer is not None
        assert trainer.model is not None
        assert trainer.sid_map is not None

    def test_setup_adds_sid_tokens(self, training_config, mock_sid_map):
        from saegenrec.modeling.sft.trainer import SFTRecTrainer

        trainer = SFTRecTrainer(training_config, seed=42)
        trainer.setup()

        for row in mock_sid_map:
            for tok in row["sid_tokens"].split("><"):
                tok = tok.strip()
                if not tok.startswith("<"):
                    tok = "<" + tok
                if not tok.endswith(">"):
                    tok = tok + ">"
                token_id = trainer.tokenizer.convert_tokens_to_ids(tok)
                assert token_id != trainer.tokenizer.unk_token_id

    def test_setup_applies_lora(self, training_config):
        from saegenrec.modeling.sft.trainer import SFTRecTrainer

        training_config.lora.enabled = True
        trainer = SFTRecTrainer(training_config, seed=42)
        trainer.setup()

        assert hasattr(trainer.model, "peft_config")

    def test_setup_no_lora(self, training_config):
        from saegenrec.modeling.sft.trainer import SFTRecTrainer

        training_config.lora.enabled = False
        trainer = SFTRecTrainer(training_config, seed=42)
        trainer.setup()

        assert not hasattr(trainer.model, "peft_config")
