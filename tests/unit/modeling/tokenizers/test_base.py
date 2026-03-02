"""Tests for ItemTokenizer ABC, registry, and helpers (T007)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from saegenrec.data.schemas import SEMANTIC_EMBEDDING_FEATURES
from saegenrec.modeling.tokenizers.base import (
    ITEM_TOKENIZER_REGISTRY,
    ItemTokenizer,
    _build_sid_map,
    get_item_tokenizer,
    register_item_tokenizer,
)


class _DummyTokenizer(ItemTokenizer):
    """Minimal concrete tokenizer for testing."""

    def train(self, semantic_embeddings_dir, collaborative_embeddings_dir, config):
        return {"loss": 0.0}

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(embeddings), 2, dtype=torch.long)

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(self, path):
        pass

    @property
    def num_codebooks(self) -> int:
        return 2

    @property
    def codebook_size(self) -> int:
        return 8


class TestItemTokenizerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ItemTokenizer()

    def test_incomplete_subclass_cannot_instantiate(self):
        class Incomplete(ItemTokenizer):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestRegistry:
    def test_register_and_get(self):
        name = "_test_reg_dummy"
        register_item_tokenizer(name)(_DummyTokenizer)
        try:
            tok = get_item_tokenizer(name)
            assert isinstance(tok, _DummyTokenizer)
        finally:
            ITEM_TOKENIZER_REGISTRY.pop(name, None)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown item tokenizer"):
            get_item_tokenizer("nonexistent_xyz_999")


class TestBuildSidMap:
    def test_basic(self):
        item_ids = [0, 1, 2]
        codes = [[0, 1], [2, 3], [1, 0]]
        ds = _build_sid_map(item_ids, codes)
        assert len(ds) == 3
        assert ds[0]["item_id"] == 0
        assert ds[0]["codes"] == [0, 1]
        assert ds[0]["sid_tokens"] == "<s_a_0><s_b_1>"
        assert ds[2]["sid_tokens"] == "<s_a_1><s_b_0>"

    def test_custom_format(self):
        item_ids = [10]
        codes = [[5, 3, 7]]
        ds = _build_sid_map(
            item_ids,
            codes,
            token_format="[{level}:{code}]",
            begin_token="<BOS>",
            end_token="<EOS>",
        )
        assert ds[0]["sid_tokens"] == "<BOS>[a:5][b:3][c:7]<EOS>"

    def test_variable_length_codes(self):
        item_ids = [0, 1]
        codes = [[0, 1], [2, 3, 4]]
        ds = _build_sid_map(item_ids, codes)
        assert ds[0]["codes"] == [0, 1]
        assert ds[1]["codes"] == [2, 3, 4]
        assert ds[0]["sid_tokens"] == "<s_a_0><s_b_1>"
        assert ds[1]["sid_tokens"] == "<s_a_2><s_b_3><s_c_4>"


class TestGenerate:
    def test_orchestration(self, tmp_path, synthetic_embeddings, synthetic_item_ids):
        emb_dir = tmp_path / "semantic_embeddings"
        ds = Dataset.from_dict(
            {
                "item_id": synthetic_item_ids,
                "embedding": synthetic_embeddings.tolist(),
            },
            features=SEMANTIC_EMBEDDING_FEATURES,
        )
        ds.save_to_disk(str(emb_dir))

        name = "_test_gen_dummy"
        register_item_tokenizer(name)(_DummyTokenizer)
        try:
            tok = _DummyTokenizer()
            output_dir = tmp_path / "output"
            config = {"collision_strategy": "append_level"}

            sid_map = tok.generate(
                semantic_embeddings_dir=emb_dir,
                collaborative_embeddings_dir=None,
                output_dir=output_dir,
                config=config,
            )
            assert len(sid_map) == 20
            assert "item_id" in sid_map.column_names
            assert "codes" in sid_map.column_names
            assert "sid_tokens" in sid_map.column_names
            assert (output_dir / "item_sid_map").exists()
            assert (output_dir / "tokenizer_model").exists()
        finally:
            ITEM_TOKENIZER_REGISTRY.pop(name, None)
