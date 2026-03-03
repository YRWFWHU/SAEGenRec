"""Integration tests for SAETokenizer (T011, T014, T017)."""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from saegenrec.data.schemas import SEMANTIC_EMBEDDING_FEATURES
from saegenrec.modeling.tokenizers.base import get_item_tokenizer
from saegenrec.modeling.tokenizers.sae import SAETokenizer

D_SAE = 128
TOP_K = 4


@pytest.fixture
def sae_tokenizer():
    return SAETokenizer(num_codebooks=TOP_K, codebook_size=D_SAE)


def _save_embeddings(path, embeddings, item_ids=None):
    if item_ids is None:
        item_ids = list(range(len(embeddings)))
    ds = Dataset.from_dict(
        {"item_id": item_ids, "embedding": embeddings.tolist()},
        features=SEMANTIC_EMBEDDING_FEATURES,
    )
    ds.save_to_disk(str(path))


class TestSAETokenizerTrainEncode:
    """T011: train + encode integration tests."""

    def test_train_returns_metrics(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        stats = sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})

        assert isinstance(stats, dict)
        for key in [
            "final_mse_loss",
            "final_l0_loss",
            "final_total_loss",
            "mean_l0",
            "vocab_utilization",
            "num_dead_features",
        ]:
            assert key in stats, f"Missing metric: {key}"

    def test_encode_shape(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})
        codes = sae_tokenizer.encode(synthetic_embeddings)
        assert codes.shape == (20, TOP_K)

    def test_encode_code_range(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})
        codes = sae_tokenizer.encode(synthetic_embeddings)
        assert codes.min() >= 0
        assert codes.max() < D_SAE

    def test_encode_dtype(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})
        codes = sae_tokenizer.encode(synthetic_embeddings)
        assert codes.dtype == torch.long

    def test_registry_lookup(self):
        tok = get_item_tokenizer("sae", num_codebooks=4, codebook_size=128)
        assert isinstance(tok, SAETokenizer)
        assert tok.num_codebooks == 4
        assert tok.codebook_size == 128

    def test_encode_without_train_raises(self, sae_tokenizer, synthetic_embeddings):
        with pytest.raises(RuntimeError, match="not trained or loaded"):
            sae_tokenizer.encode(synthetic_embeddings)

    def test_encode_dimension_mismatch_raises(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 2, "batch_size": 8})
        wrong_dim = torch.randn(5, 32)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            sae_tokenizer.encode(wrong_dim)


class TestSAETokenizerParamOverride:
    """T014: parameter priority tests."""

    def test_params_d_sae_overrides_codebook_size(self):
        tok = SAETokenizer(num_codebooks=8, codebook_size=256, d_sae=4096)
        assert tok.codebook_size == 4096

    def test_params_top_k_overrides_num_codebooks(self):
        tok = SAETokenizer(num_codebooks=8, codebook_size=256, top_k=16)
        assert tok.num_codebooks == 16

    def test_defaults_used_when_no_override(self):
        tok = SAETokenizer(num_codebooks=8, codebook_size=256)
        assert tok.num_codebooks == 8
        assert tok.codebook_size == 256

    def test_d_sae_less_than_top_k_raises(self):
        with pytest.raises(ValueError, match="must be greater than"):
            SAETokenizer(num_codebooks=8, codebook_size=4)


class TestSAETokenizerSaveLoad:
    """T017: save/load round-trip tests."""

    def test_save_load_roundtrip(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})
        codes_before = sae_tokenizer.encode(synthetic_embeddings)

        model_dir = tmp_path / "model"
        sae_tokenizer.save(model_dir)

        tok2 = SAETokenizer(num_codebooks=TOP_K, codebook_size=D_SAE)
        tok2.load(model_dir)
        codes_after = tok2.encode(synthetic_embeddings)

        assert torch.equal(codes_before, codes_after)

    def test_load_restores_properties(self, sae_tokenizer, synthetic_embeddings, tmp_path):
        emb_dir = tmp_path / "embeddings"
        _save_embeddings(emb_dir, synthetic_embeddings)
        sae_tokenizer.train(emb_dir, None, {"epochs": 3, "batch_size": 8})

        model_dir = tmp_path / "model"
        sae_tokenizer.save(model_dir)

        tok2 = SAETokenizer(num_codebooks=1, codebook_size=1024)
        tok2.load(model_dir)
        assert tok2.num_codebooks == TOP_K
        assert tok2.codebook_size == D_SAE

    def test_save_without_train_raises(self, sae_tokenizer, tmp_path):
        with pytest.raises(RuntimeError, match="No model to save"):
            sae_tokenizer.save(tmp_path / "model")
