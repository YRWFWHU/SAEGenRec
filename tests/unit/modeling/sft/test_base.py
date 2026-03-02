"""Tests for SFT task builder base class and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from datasets import Dataset

from saegenrec.modeling.sft.base import (
    SFT_TASK_REGISTRY,
    SFTTaskBuilder,
    get_sft_task_builder,
    register_sft_task,
)


class TestSFTTaskBuilderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SFTTaskBuilder()

    def test_subclass_must_implement_abstract_methods(self):
        class Incomplete(SFTTaskBuilder):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class Concrete(SFTTaskBuilder):
            @property
            def task_type(self) -> str:
                return "test"

            def build(
                self,
                stage1_dir: Path,
                stage2_dir: Path,
                sid_map: Dataset,
                config: dict[str, Any],
            ) -> Dataset:
                return Dataset.from_dict(
                    {"task_type": [], "instruction": [], "input": [], "output": []}
                )

        builder = Concrete()
        assert builder.task_type == "test"


class TestRegistry:
    def test_register_and_get(self):
        @register_sft_task("_test_dummy")
        class DummyBuilder(SFTTaskBuilder):
            @property
            def task_type(self) -> str:
                return "_test_dummy"

            def build(self, stage1_dir, stage2_dir, sid_map, config):
                return Dataset.from_dict(
                    {"task_type": [], "instruction": [], "input": [], "output": []}
                )

        assert "_test_dummy" in SFT_TASK_REGISTRY
        instance = get_sft_task_builder("_test_dummy")
        assert isinstance(instance, DummyBuilder)

        del SFT_TASK_REGISTRY["_test_dummy"]

    def test_get_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown SFT task"):
            get_sft_task_builder("__nonexistent__")

    def test_builtin_tasks_registered(self):
        assert "seqrec" in SFT_TASK_REGISTRY
        assert "item2index" in SFT_TASK_REGISTRY
        assert "index2item" in SFT_TASK_REGISTRY


class TestLoadTemplates:
    def test_load_templates_from_yaml(self, tmp_path: Path):
        templates = {
            "mytask": [
                {
                    "instruction": "Do something",
                    "input_template": "{x}",
                    "output_template": "{y}",
                }
            ]
        }
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text(yaml.dump(templates))

        class MyBuilder(SFTTaskBuilder):
            @property
            def task_type(self) -> str:
                return "mytask"

            def build(self, stage1_dir, stage2_dir, sid_map, config):
                pass

        builder = MyBuilder()
        loaded = builder.load_templates(yaml_file)
        assert len(loaded) == 1
        assert loaded[0]["instruction"] == "Do something"

    def test_load_templates_missing_task_type_raises(self, tmp_path: Path):
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text(yaml.dump({"other_task": [{"instruction": "x"}]}))

        class MyBuilder(SFTTaskBuilder):
            @property
            def task_type(self) -> str:
                return "missing_task"

            def build(self, stage1_dir, stage2_dir, sid_map, config):
                pass

        builder = MyBuilder()
        with pytest.raises(ValueError, match="No templates found"):
            builder.load_templates(yaml_file)
