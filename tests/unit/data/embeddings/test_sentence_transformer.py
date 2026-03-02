"""Tests for SentenceTransformerEmbedder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from saegenrec.data.embeddings.semantic.base import SEMANTIC_EMBEDDER_REGISTRY
from saegenrec.data.embeddings.semantic.sentence_transformer import (
    SentenceTransformerEmbedder,
)


EMBED_DIM = 384


def _make_stage1_data(tmp_path: Path, items: list[dict], id_map: list[dict]):
    """Create mock Stage 1 data on disk."""
    meta_ds = Dataset.from_dict(
        {k: [row[k] for row in items] for k in items[0].keys()}
    )
    meta_ds.save_to_disk(str(tmp_path / "item_metadata"))

    map_ds = Dataset.from_dict(
        {k: [row[k] for row in id_map] for k in id_map[0].keys()}
    )
    map_ds.save_to_disk(str(tmp_path / "item_id_map"))


def _mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = EMBED_DIM
    model.encode.side_effect = lambda texts, **kw: np.random.randn(
        len(texts), EMBED_DIM
    ).astype(np.float32)
    return model


class TestSentenceTransformerRegistration:
    def test_registered(self):
        assert "sentence-transformer" in SEMANTIC_EMBEDDER_REGISTRY


class TestSentenceTransformerEmbedder:
    @patch("sentence_transformers.SentenceTransformer")
    def test_normal_generation(self, mock_st_cls, tmp_path):
        mock_st_cls.return_value = _mock_model()

        items = [
            {"item_id": "A", "title": "Baby Toy", "brand": "BrandX",
             "description": "A nice toy", "price": 9.99,
             "categories": ["Toys"], "image_url": ""},
            {"item_id": "B", "title": "Blanket", "brand": "BrandY",
             "description": "Warm blanket", "price": 19.99,
             "categories": ["Bedding"], "image_url": ""},
        ]
        id_map = [
            {"original_id": "A", "mapped_id": 1},
            {"original_id": "B", "mapped_id": 2},
        ]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        config = {"model_name": "test-model", "device": "cpu"}
        ds = embedder.generate(tmp_path, tmp_path, config)

        assert len(ds) == 2
        assert set(ds.column_names) == {"item_id", "embedding"}
        assert len(ds[0]["embedding"]) == EMBED_DIM

    @patch("sentence_transformers.SentenceTransformer")
    def test_missing_item_in_metadata(self, mock_st_cls, tmp_path):
        mock_st_cls.return_value = _mock_model()

        items = [
            {"item_id": "A", "title": "Baby Toy", "brand": "", "description": "",
             "price": 0.0, "categories": [], "image_url": ""},
        ]
        id_map = [
            {"original_id": "A", "mapped_id": 1},
            {"original_id": "C", "mapped_id": 3},
        ]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        ds = embedder.generate(tmp_path, tmp_path, {})

        assert len(ds) == 1
        assert ds[0]["item_id"] == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_all_empty_text_zero_vector(self, mock_st_cls, tmp_path):
        mock_st_cls.return_value = _mock_model()

        items = [
            {"item_id": "A", "title": "", "brand": "", "description": "",
             "price": None, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        ds = embedder.generate(tmp_path, tmp_path, {})

        assert len(ds) == 1
        assert all(v == 0.0 for v in ds[0]["embedding"])

    @patch("sentence_transformers.SentenceTransformer")
    def test_price_numeric_to_text(self, mock_st_cls, tmp_path):
        mock_model = _mock_model()
        mock_st_cls.return_value = mock_model

        items = [
            {"item_id": "A", "title": "Toy", "brand": "", "description": "",
             "price": 29.99, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        ds = embedder.generate(
            tmp_path, tmp_path, {"text_fields": ["title", "price"]}
        )

        call_args = mock_model.encode.call_args
        encoded_texts = call_args[0][0]
        assert "29.99" in encoded_texts[0] or "29.9" in encoded_texts[0]

    @patch("sentence_transformers.SentenceTransformer")
    def test_normalize_on(self, mock_st_cls, tmp_path):
        mock_model = _mock_model()
        mock_st_cls.return_value = mock_model

        items = [
            {"item_id": "A", "title": "Toy", "brand": "", "description": "",
             "price": 0.0, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        embedder.generate(tmp_path, tmp_path, {"normalize": True})

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is True

    @patch("sentence_transformers.SentenceTransformer")
    def test_normalize_off_default(self, mock_st_cls, tmp_path):
        mock_model = _mock_model()
        mock_st_cls.return_value = mock_model

        items = [
            {"item_id": "A", "title": "Toy", "brand": "", "description": "",
             "price": 0.0, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        embedder.generate(tmp_path, tmp_path, {})

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is False

    @patch("sentence_transformers.SentenceTransformer")
    def test_skip_if_exists(self, mock_st_cls, tmp_path):
        mock_st_cls.return_value = _mock_model()

        items = [
            {"item_id": "A", "title": "Toy", "brand": "", "description": "",
             "price": 0.0, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        ds1 = embedder.generate(tmp_path, tmp_path, {})
        assert len(ds1) == 1

        mock_st_cls.reset_mock()
        ds2 = embedder.generate(tmp_path, tmp_path, {})
        mock_st_cls.assert_not_called()
        assert len(ds2) == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_force_regeneration(self, mock_st_cls, tmp_path):
        mock_st_cls.return_value = _mock_model()

        items = [
            {"item_id": "A", "title": "Toy", "brand": "", "description": "",
             "price": 0.0, "categories": [], "image_url": ""},
        ]
        id_map = [{"original_id": "A", "mapped_id": 1}]
        _make_stage1_data(tmp_path, items, id_map)

        embedder = SentenceTransformerEmbedder()
        embedder.generate(tmp_path, tmp_path, {})

        mock_st_cls.reset_mock()
        mock_st_cls.return_value = _mock_model()
        embedder.generate(tmp_path, tmp_path, {"force": True})
        mock_st_cls.assert_called_once()
