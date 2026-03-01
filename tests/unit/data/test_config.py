"""Tests for pipeline configuration loading and validation."""

import pytest
import yaml

from saegenrec.data.config import (
    DatasetConfig,
    EmbeddingConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
    TokenizerConfig,
    load_config,
)


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.name == "amazon2015"
        assert cfg.category == "Baby"
        assert cfg.raw_dir == "data/raw"

    def test_data_path_amazon2015(self):
        cfg = DatasetConfig(name="amazon2015", category="Baby")
        assert str(cfg.data_path) == "data/raw/Amazon2015/Baby"

    def test_data_path_amazon2023(self):
        cfg = DatasetConfig(name="amazon2023", category="All_Beauty")
        assert str(cfg.data_path) == "data/raw/Amazon2023/All_Beauty"

    def test_unknown_dataset_name(self):
        cfg = DatasetConfig(name="unknown")
        with pytest.raises(ValueError, match="Unknown dataset name"):
            _ = cfg.data_path


class TestProcessingConfig:
    def test_defaults(self):
        cfg = ProcessingConfig()
        assert cfg.kcore_threshold == 5
        assert cfg.split_strategy == "loo"
        assert cfg.split_ratio == [0.8, 0.1, 0.1]
        assert cfg.max_seq_len == 20

    def test_invalid_split_strategy(self):
        with pytest.raises(ValueError, match="split_strategy must be"):
            ProcessingConfig(split_strategy="invalid")

    def test_invalid_split_ratio_sum(self):
        with pytest.raises(ValueError, match="split_ratio must sum to 1.0"):
            ProcessingConfig(split_strategy="to", split_ratio=[0.5, 0.2, 0.1])

    def test_valid_to_split(self):
        cfg = ProcessingConfig(split_strategy="to", split_ratio=[0.7, 0.15, 0.15])
        assert cfg.split_strategy == "to"

    def test_invalid_kcore_threshold(self):
        with pytest.raises(ValueError, match="kcore_threshold must be >= 1"):
            ProcessingConfig(kcore_threshold=0)

    def test_invalid_max_seq_len(self):
        with pytest.raises(ValueError, match="max_seq_len must be >= 1"):
            ProcessingConfig(max_seq_len=0)


class TestTokenizerConfig:
    def test_defaults(self):
        cfg = TokenizerConfig()
        assert cfg.name == "passthrough"
        assert cfg.params == {}


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.enabled is False
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.text_fields == ["title", "brand", "categories"]
        assert cfg.batch_size == 256
        assert cfg.device == "cpu"


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.interim_dir == "data/interim"
        assert cfg.processed_dir == "data/processed"

    def test_interim_path(self):
        cfg = OutputConfig()
        path = cfg.interim_path("amazon2015", "Baby")
        assert str(path) == "data/interim/amazon2015/Baby"

    def test_processed_path(self):
        cfg = OutputConfig()
        path = cfg.processed_path("amazon2015", "Baby", "loo")
        assert str(path) == "data/processed/amazon2015/Baby/loo"


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.tokenizer, TokenizerConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.output, OutputConfig)


class TestLoadConfig:
    def test_load_full_config(self, tmp_path):
        config_data = {
            "dataset": {"name": "amazon2023", "category": "All_Beauty"},
            "processing": {"kcore_threshold": 10, "split_strategy": "to"},
            "tokenizer": {"name": "custom", "params": {"dim": 64}},
            "embedding": {"enabled": True, "model_name": "bge-base-en-v1.5"},
            "output": {"interim_dir": "/tmp/interim"},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_file)
        assert cfg.dataset.name == "amazon2023"
        assert cfg.dataset.category == "All_Beauty"
        assert cfg.processing.kcore_threshold == 10
        assert cfg.processing.split_strategy == "to"
        assert cfg.tokenizer.name == "custom"
        assert cfg.tokenizer.params == {"dim": 64}
        assert cfg.embedding.enabled is True
        assert cfg.output.interim_dir == "/tmp/interim"

    def test_load_partial_config(self, tmp_path):
        config_data = {"dataset": {"name": "amazon2015"}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_file)
        assert cfg.dataset.name == "amazon2015"
        assert cfg.processing.kcore_threshold == 5
        assert cfg.tokenizer.name == "passthrough"

    def test_load_empty_config(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        cfg = load_config(config_file)
        assert cfg.dataset.name == "amazon2015"
        assert cfg.processing.split_strategy == "loo"

    def test_load_config_from_fixture(self, sample_config_path):
        cfg = load_config(sample_config_path)
        assert cfg.processing.kcore_threshold == 3
        assert cfg.processing.max_seq_len == 10
