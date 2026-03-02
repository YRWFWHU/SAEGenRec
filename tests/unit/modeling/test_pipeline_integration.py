"""Integration test: tokenize + build-sft on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, load_from_disk

from saegenrec.data.schemas import (
    ID_MAP_FEATURES,
    ITEM_METADATA_FEATURES,
    SEMANTIC_EMBEDDING_FEATURES,
    SFT_FEATURES,
    SID_MAP_FEATURES,
    USER_SEQUENCES_FEATURES,
)


@pytest.fixture
def synthetic_pipeline_dirs(tmp_path: Path) -> dict[str, Path]:
    """Set up synthetic stage1, stage2, and modeling directories with required artifacts."""
    stage1_dir = tmp_path / "data" / "interim" / "test_ds" / "TestCat"
    stage2_dir = stage1_dir / "loo"
    modeling_dir = tmp_path / "data" / "processed" / "test_ds" / "TestCat"

    stage1_dir.mkdir(parents=True)
    stage2_dir.mkdir(parents=True)
    modeling_dir.mkdir(parents=True)

    num_items = 20
    embed_dim = 32
    rng = np.random.default_rng(42)

    embeddings_ds = Dataset.from_dict(
        {
            "item_id": list(range(num_items)),
            "embedding": rng.standard_normal((num_items, embed_dim)).tolist(),
        },
        features=SEMANTIC_EMBEDDING_FEATURES,
    )
    embeddings_ds.save_to_disk(str(stage1_dir / "item_semantic_embeddings"))

    item_metadata = Dataset.from_dict(
        {
            "item_id": [f"orig_{i}" for i in range(num_items)],
            "title": [f"Product {i}" for i in range(num_items)],
            "brand": [f"Brand{i % 3}" for i in range(num_items)],
            "categories": [["cat1"] for _ in range(num_items)],
            "description": [f"Description {i}" for i in range(num_items)],
            "price": [float(i * 10 + 5) for i in range(num_items)],
            "image_url": [f"http://img/{i}.jpg" for i in range(num_items)],
        },
        features=ITEM_METADATA_FEATURES,
    )
    item_metadata.save_to_disk(str(stage1_dir / "item_metadata"))

    item_id_map = Dataset.from_dict(
        {
            "original_id": [f"orig_{i}" for i in range(num_items)],
            "mapped_id": list(range(num_items)),
        },
        features=ID_MAP_FEATURES,
    )
    item_id_map.save_to_disk(str(stage1_dir / "item_id_map"))

    train_sequences = Dataset.from_dict(
        {
            "user_id": [0, 1, 2],
            "item_ids": [
                list(range(0, 8)),
                list(range(3, 10)),
                list(range(5, 15)),
            ],
            "timestamps": [
                list(range(100, 900, 100)),
                list(range(100, 800, 100)),
                list(range(100, 1100, 100)),
            ],
            "ratings": [
                [4.0] * 8,
                [3.0] * 7,
                [5.0] * 10,
            ],
            "review_texts": [
                [f"r{j}" for j in range(8)],
                [f"r{j}" for j in range(7)],
                [f"r{j}" for j in range(10)],
            ],
            "review_summaries": [
                [f"s{j}" for j in range(8)],
                [f"s{j}" for j in range(7)],
                [f"s{j}" for j in range(10)],
            ],
        },
        features=USER_SEQUENCES_FEATURES,
    )
    train_sequences.save_to_disk(str(stage2_dir / "train_sequences"))

    return {
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
        "modeling_dir": modeling_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def sft_template_file(tmp_path: Path) -> Path:
    """Create a temporary SFT template file."""
    import yaml

    templates = {
        "seqrec": [
            {
                "instruction": "Predict next item.",
                "input_template": "History: {history_sids}",
                "output_template": "{target_sid}",
            },
        ],
        "item2index": [
            {
                "instruction": "Get SID for item.",
                "input_template": "Title: {title}",
                "output_template": "{sid_tokens}",
            },
        ],
        "index2item": [
            {
                "instruction": "Get item for SID.",
                "input_template": "SID: {sid_tokens}",
                "output_template": "{title}",
            },
        ],
    }
    template_path = tmp_path / "sft_prompts.yaml"
    with open(template_path, "w") as f:
        yaml.dump(templates, f)
    return template_path


class TestPipelineIntegration:
    def test_tokenize_step(self, synthetic_pipeline_dirs):
        """Tokenize step produces valid item_sid_map."""
        from saegenrec.data.config import ItemTokenizerConfig, OutputConfig, PipelineConfig
        from saegenrec.data.pipeline import run_pipeline

        dirs = synthetic_pipeline_dirs
        cfg = PipelineConfig()
        cfg.dataset.name = "test_ds"
        cfg.dataset.category = "TestCat"
        cfg.output = OutputConfig(
            interim_dir=str(dirs["tmp_path"] / "data" / "interim"),
            processed_dir=str(dirs["tmp_path"] / "data" / "processed"),
        )
        cfg.item_tokenizer = ItemTokenizerConfig(
            enabled=True,
            name="rqkmeans",
            num_codebooks=2,
            codebook_size=8,
            collision_strategy="append_level",
            params={"kmeans_niter": 5, "use_gpu": False},
        )

        stats = run_pipeline(cfg, steps=["tokenize"], force=True)

        assert "tokenize_items" in stats
        assert stats["tokenize_items"] == 20

        sid_map = load_from_disk(str(dirs["modeling_dir"] / "item_sid_map"))
        assert len(sid_map) == 20
        assert set(sid_map.column_names) == {"item_id", "codes", "sid_tokens"}

        codes_set = set(tuple(c) for c in sid_map["codes"])
        assert len(codes_set) == 20, "SID uniqueness constraint violated"

    def test_build_sft_step(self, synthetic_pipeline_dirs, sft_template_file):
        """Build-sft step produces valid SFT dataset after tokenize."""
        from saegenrec.data.config import (
            ItemTokenizerConfig,
            OutputConfig,
            PipelineConfig,
            SFTBuilderConfig,
        )
        from saegenrec.data.pipeline import run_pipeline

        dirs = synthetic_pipeline_dirs
        cfg = PipelineConfig()
        cfg.dataset.name = "test_ds"
        cfg.dataset.category = "TestCat"
        cfg.output = OutputConfig(
            interim_dir=str(dirs["tmp_path"] / "data" / "interim"),
            processed_dir=str(dirs["tmp_path"] / "data" / "processed"),
        )
        cfg.item_tokenizer = ItemTokenizerConfig(
            enabled=True,
            name="rqkmeans",
            num_codebooks=2,
            codebook_size=8,
            collision_strategy="append_level",
            params={"kmeans_niter": 5, "use_gpu": False},
        )
        cfg.sft_builder = SFTBuilderConfig(
            enabled=True,
            tasks=["seqrec", "item2index", "index2item"],
            template_file=str(sft_template_file),
            max_history_len=10,
            seed=42,
        )

        run_pipeline(cfg, steps=["tokenize"], force=True)
        stats = run_pipeline(cfg, steps=["build-sft"], force=True)

        assert "sft_samples" in stats
        assert stats["sft_samples"] > 0

        sft_ds = load_from_disk(str(dirs["modeling_dir"] / "sft_data"))
        assert len(sft_ds) > 0
        assert set(sft_ds.column_names) == {"task_type", "instruction", "input", "output"}

        task_types = set(sft_ds["task_type"])
        assert "seqrec" in task_types
        assert "item2index" in task_types
        assert "index2item" in task_types

        for row in sft_ds:
            assert row["instruction"]
            assert row["input"]
            assert row["output"]

    def test_tokenize_then_build_sft(self, synthetic_pipeline_dirs, sft_template_file):
        """End-to-end: tokenize + build-sft in sequence."""
        from saegenrec.data.config import (
            ItemTokenizerConfig,
            OutputConfig,
            PipelineConfig,
            SFTBuilderConfig,
        )
        from saegenrec.data.pipeline import run_pipeline

        dirs = synthetic_pipeline_dirs
        cfg = PipelineConfig()
        cfg.dataset.name = "test_ds"
        cfg.dataset.category = "TestCat"
        cfg.output = OutputConfig(
            interim_dir=str(dirs["tmp_path"] / "data" / "interim"),
            processed_dir=str(dirs["tmp_path"] / "data" / "processed"),
        )
        cfg.item_tokenizer = ItemTokenizerConfig(
            enabled=True,
            name="rqkmeans",
            num_codebooks=2,
            codebook_size=8,
            collision_strategy="append_level",
            params={"kmeans_niter": 5, "use_gpu": False},
        )
        cfg.sft_builder = SFTBuilderConfig(
            enabled=True,
            tasks=["seqrec", "item2index", "index2item"],
            template_file=str(sft_template_file),
            max_history_len=10,
            seed=42,
        )

        stats = run_pipeline(cfg, steps=["tokenize", "build-sft"], force=True)

        assert "tokenize_items" in stats
        assert "sft_samples" in stats

        sid_map = load_from_disk(str(dirs["modeling_dir"] / "item_sid_map"))
        sft_ds = load_from_disk(str(dirs["modeling_dir"] / "sft_data"))

        assert len(sid_map) == 20
        assert len(sft_ds) > 0
