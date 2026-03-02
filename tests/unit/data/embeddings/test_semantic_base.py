"""Tests for SemanticEmbedder ABC and registry."""

from pathlib import Path

import pytest
from datasets import Dataset

from saegenrec.data.embeddings.semantic.base import (
    SEMANTIC_EMBEDDER_REGISTRY,
    SemanticEmbedder,
    get_semantic_embedder,
    register_semantic_embedder,
)


@register_semantic_embedder("_test-dummy")
class _DummySemanticEmbedder(SemanticEmbedder):
    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        return Dataset.from_dict({"item_id": [1], "embedding": [[0.1, 0.2]]})


class TestSemanticEmbedderRegistry:
    def test_register_and_lookup(self):
        assert "_test-dummy" in SEMANTIC_EMBEDDER_REGISTRY
        assert SEMANTIC_EMBEDDER_REGISTRY["_test-dummy"] is _DummySemanticEmbedder

    def test_get_semantic_embedder_valid(self):
        embedder = get_semantic_embedder("_test-dummy")
        assert isinstance(embedder, _DummySemanticEmbedder)

    def test_get_semantic_embedder_invalid(self):
        with pytest.raises(ValueError, match="Unknown semantic embedder: 'nonexistent'"):
            get_semantic_embedder("nonexistent")

    def test_error_lists_available_names(self):
        with pytest.raises(ValueError, match="_test-dummy"):
            get_semantic_embedder("nonexistent")

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SemanticEmbedder()
