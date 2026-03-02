"""Tests for GenRecModel ABC and registry (T034)."""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset

from saegenrec.modeling.genrec.base import (
    GENREC_MODEL_REGISTRY,
    GenRecModel,
    get_genrec_model,
    register_genrec_model,
)
from saegenrec.modeling.genrec.config import GenRecConfig


class TestGenRecModelABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            GenRecModel()

    def test_subclass_must_implement_all_methods(self):
        class IncompleteModel(GenRecModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_concrete_subclass_can_instantiate(self):
        class ConcreteModel(GenRecModel):
            def train(self, dataset: Dataset, training_args: dict) -> dict:
                return {}

            def generate(self, input_text: str | list[str], **kwargs) -> list[str]:
                return []

            def evaluate(
                self, dataset: Dataset, metrics: list[str] | None = None
            ) -> dict[str, float]:
                return {}

            def save_pretrained(self, path: Path) -> None:
                pass

            @classmethod
            def from_pretrained(cls, path: Path, **kwargs) -> "ConcreteModel":
                return cls()

        model = ConcreteModel()
        assert isinstance(model, GenRecModel)


class TestRegistry:
    def setup_method(self):
        self._original_registry = dict(GENREC_MODEL_REGISTRY)

    def teardown_method(self):
        GENREC_MODEL_REGISTRY.clear()
        GENREC_MODEL_REGISTRY.update(self._original_registry)

    def test_register_genrec_model_adds_to_registry(self):
        @register_genrec_model("test_model")
        class TestModel(GenRecModel):
            def train(self, dataset, training_args):
                return {}

            def generate(self, input_text, **kwargs):
                return []

            def evaluate(self, dataset, metrics=None):
                return {}

            def save_pretrained(self, path):
                pass

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return cls()

        assert "test_model" in GENREC_MODEL_REGISTRY
        assert GENREC_MODEL_REGISTRY["test_model"] is TestModel

    def test_get_genrec_model_returns_instance(self):
        @register_genrec_model("retrievable_model")
        class RetrievableModel(GenRecModel):
            def train(self, dataset, training_args):
                return {}

            def generate(self, input_text, **kwargs):
                return []

            def evaluate(self, dataset, metrics=None):
                return {}

            def save_pretrained(self, path):
                pass

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return cls()

        model = get_genrec_model("retrievable_model")
        assert isinstance(model, RetrievableModel)

    def test_get_genrec_model_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown GenRec model"):
            get_genrec_model("nonexistent_model_xyz")


class TestGenRecConfig:
    def test_default_values(self):
        config = GenRecConfig()
        assert config.base_model_name == "Qwen/Qwen2.5-0.5B"
        assert config.lora_enabled is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_target_modules is None
        assert config.training_strategy == "sft"
        assert config.sid_tokens_path is None

    def test_custom_values(self):
        config = GenRecConfig(
            base_model_name="meta-llama/Llama-2-7b",
            lora_enabled=False,
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=["q_proj", "v_proj"],
            training_strategy="dpo",
            sid_tokens_path="/data/sid_tokens.json",
        )
        assert config.base_model_name == "meta-llama/Llama-2-7b"
        assert config.lora_enabled is False
        assert config.lora_r == 8
        assert config.lora_target_modules == ["q_proj", "v_proj"]
        assert config.training_strategy == "dpo"
        assert config.sid_tokens_path == "/data/sid_tokens.json"
