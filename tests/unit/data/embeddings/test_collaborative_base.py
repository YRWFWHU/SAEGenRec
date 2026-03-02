"""Tests for CollaborativeEmbedder ABC and registry."""

from pathlib import Path

import pytest
from datasets import Dataset

from saegenrec.data.embeddings.collaborative.base import (
    COLLABORATIVE_EMBEDDER_REGISTRY,
    CollaborativeEmbedder,
    get_collaborative_embedder,
    register_collaborative_embedder,
)


@register_collaborative_embedder("_test-dummy")
class _DummyCollaborativeEmbedder(CollaborativeEmbedder):
    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        return Dataset.from_dict({"item_id": [1], "embedding": [[0.1, 0.2]]})


class TestCollaborativeEmbedderRegistry:
    def test_register_and_lookup(self):
        assert "_test-dummy" in COLLABORATIVE_EMBEDDER_REGISTRY
        assert COLLABORATIVE_EMBEDDER_REGISTRY["_test-dummy"] is _DummyCollaborativeEmbedder

    def test_get_collaborative_embedder_valid(self):
        embedder = get_collaborative_embedder("_test-dummy")
        assert isinstance(embedder, _DummyCollaborativeEmbedder)

    def test_get_collaborative_embedder_invalid(self):
        with pytest.raises(ValueError, match="Unknown collaborative embedder: 'nonexistent'"):
            get_collaborative_embedder("nonexistent")

    def test_error_lists_available_names(self):
        with pytest.raises(ValueError, match="_test-dummy"):
            get_collaborative_embedder("nonexistent")

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CollaborativeEmbedder()
