"""Tests for RQ-VAE tokenizer (T009)."""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from saegenrec.data.schemas import SEMANTIC_EMBEDDING_FEATURES
from saegenrec.modeling.tokenizers.rqvae import RQVAETokenizer


class TestRQVAETokenizer:
    @pytest.fixture
    def tokenizer(self):
        return RQVAETokenizer(
            num_codebooks=2, codebook_size=8, hidden_dim=32, latent_dim=16
        )

    @staticmethod
    def _save_embeddings(path, embeddings, item_ids=None):
        if item_ids is None:
            item_ids = list(range(len(embeddings)))
        ds = Dataset.from_dict(
            {"item_id": item_ids, "embedding": embeddings.tolist()},
            features=SEMANTIC_EMBEDDING_FEATURES,
        )
        ds.save_to_disk(str(path))

    def test_train_returns_stats(
        self, tokenizer, synthetic_embeddings, tmp_path
    ):
        emb_dir = tmp_path / "embeddings"
        self._save_embeddings(emb_dir, synthetic_embeddings)
        stats = tokenizer.train(emb_dir, None, {"max_epochs": 2, "batch_size": 8})
        assert isinstance(stats, dict)
        assert "train_loss" in stats

    def test_encode_shape_and_range(
        self, tokenizer, synthetic_embeddings, tmp_path
    ):
        emb_dir = tmp_path / "embeddings"
        self._save_embeddings(emb_dir, synthetic_embeddings)
        tokenizer.train(emb_dir, None, {"max_epochs": 2, "batch_size": 8})
        codes = tokenizer.encode(synthetic_embeddings)
        assert codes.shape == (20, 2)
        assert codes.min() >= 0
        assert codes.max() < 8

    def test_save_load_roundtrip(
        self, tokenizer, synthetic_embeddings, tmp_path
    ):
        emb_dir = tmp_path / "embeddings"
        self._save_embeddings(emb_dir, synthetic_embeddings)
        tokenizer.train(emb_dir, None, {"max_epochs": 2, "batch_size": 8})
        codes_before = tokenizer.encode(synthetic_embeddings)

        model_dir = tmp_path / "model"
        tokenizer.save(model_dir)

        tok2 = RQVAETokenizer(
            num_codebooks=2, codebook_size=8, hidden_dim=32, latent_dim=16
        )
        tok2.load(model_dir)
        codes_after = tok2.encode(synthetic_embeddings)

        assert torch.equal(codes_before, codes_after)

    def test_codebook_utilization(
        self, tokenizer, synthetic_embeddings, tmp_path
    ):
        emb_dir = tmp_path / "embeddings"
        self._save_embeddings(emb_dir, synthetic_embeddings)
        tokenizer.train(emb_dir, None, {"max_epochs": 2, "batch_size": 8})
        codes = tokenizer.encode(synthetic_embeddings)
        for cb in range(tokenizer.num_codebooks):
            unique = codes[:, cb].unique().numel()
            assert unique > 0, f"Codebook {cb} has zero utilization"
