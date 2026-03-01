"""Tests for text embedding generation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from saegenrec.data.schemas import ID_MAP_FEATURES, ITEM_METADATA_FEATURES


class TestTextEmbeddingGenerator:
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Create item_metadata and item_id_map datasets on disk."""
        metadata = Dataset.from_dict(
            {
                "item_id": ["i1", "i2", "i3"],
                "title": ["Item One", "Item Two", "Item Three"],
                "brand": ["BrandA", "BrandB", ""],
                "categories": [["cat1"], ["cat2"], []],
                "description": ["Desc 1", "Desc 2", "Desc 3"],
                "price": [9.99, 19.99, None],
                "image_url": ["url1", "url2", ""],
            },
            features=ITEM_METADATA_FEATURES,
        )
        metadata.save_to_disk(str(tmp_path / "item_metadata"))

        id_map = Dataset.from_dict(
            {"original_id": ["i1", "i2", "i3"], "mapped_id": [0, 1, 2]},
            features=ID_MAP_FEATURES,
        )
        id_map.save_to_disk(str(tmp_path / "item_id_map"))

        return tmp_path

    def test_field_concatenation(self, setup_dirs):
        """Text fields should be concatenated for encoding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            import saegenrec.data.embeddings.text as emb_module

            with patch.object(emb_module, "SentenceTransformer", create=True) as mock_st:
                mock_st.return_value = mock_model

                # Re-import to get the patched version
                from importlib import reload
                reload(emb_module)

        # Instead, mock at the sentence_transformers module level
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from importlib import reload

            import saegenrec.data.embeddings.text as emb_mod

            reload(emb_mod)

            ds = emb_mod.generate_text_embeddings(
                item_metadata_dir=setup_dirs / "item_metadata",
                item_id_map_dir=setup_dirs / "item_id_map",
                output_dir=setup_dirs,
                text_fields=["title", "brand"],
            )

            call_args = mock_model.encode.call_args
            texts = call_args[0][0]
            assert "Item One" in texts[0]
            assert "BrandA" in texts[0]

    def test_output_shape(self, setup_dirs):
        """Output should have one embedding per mapped item."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from importlib import reload

            import saegenrec.data.embeddings.text as emb_mod

            reload(emb_mod)

            ds = emb_mod.generate_text_embeddings(
                item_metadata_dir=setup_dirs / "item_metadata",
                item_id_map_dir=setup_dirs / "item_id_map",
                output_dir=setup_dirs,
            )
            assert len(ds) == 3
            assert len(ds[0]["embedding"]) == 384

    def test_batch_processing(self, setup_dirs):
        """Items should be processed in batches."""
        mock_model = MagicMock()

        def encode_side_effect(texts, **kwargs):
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_model.encode.side_effect = encode_side_effect
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from importlib import reload

            import saegenrec.data.embeddings.text as emb_mod

            reload(emb_mod)

            ds = emb_mod.generate_text_embeddings(
                item_metadata_dir=setup_dirs / "item_metadata",
                item_id_map_dir=setup_dirs / "item_id_map",
                output_dir=setup_dirs,
                batch_size=2,
            )
            assert mock_model.encode.call_count == 2
            assert len(ds) == 3
