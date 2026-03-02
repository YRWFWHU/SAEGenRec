"""Tests for SASRecEmbedder and SASRec model."""

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from saegenrec.data.embeddings.collaborative.base import COLLABORATIVE_EMBEDDER_REGISTRY
from saegenrec.data.embeddings.collaborative.models.sasrec_model import SASRec
from saegenrec.data.embeddings.collaborative.sasrec import SASRecEmbedder


NUM_ITEMS = 50
HIDDEN_SIZE = 16
MAX_SEQ_LEN = 10
BATCH_SIZE = 4


class TestSASRecModel:
    def setup_method(self):
        self.model = SASRec(
            num_items=NUM_ITEMS,
            hidden_size=HIDDEN_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            num_layers=1,
            num_heads=1,
            dropout=0.0,
        )

    def test_forward_shape(self):
        seq = torch.randint(0, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        output = self.model(seq)
        assert output.shape == (BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE)

    def test_bpr_loss_scalar(self):
        seq = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        pos = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        neg = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        loss = self.model.bpr_loss(seq, pos, neg)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_ce_loss_scalar(self):
        seq = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        pos = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        loss = self.model.ce_loss(seq, pos)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_ce_loss_with_padding(self):
        seq = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long)
        pos = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long)
        seq[:, -3:] = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, 3))
        pos[:, -3:] = torch.randint(1, NUM_ITEMS + 1, (BATCH_SIZE, 3))
        loss = self.model.ce_loss(seq, pos)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_predict_shape(self):
        seq = torch.randint(0, NUM_ITEMS + 1, (BATCH_SIZE, MAX_SEQ_LEN))
        scores = self.model.predict(seq)
        assert scores.shape == (BATCH_SIZE, NUM_ITEMS)

    def test_extract_embeddings_shape(self):
        embeddings = self.model.extract_item_embeddings()
        assert embeddings.shape == (NUM_ITEMS, HIDDEN_SIZE)

    def test_padding_embedding_is_zero(self):
        pad_emb = self.model.item_embedding.weight[0]
        assert torch.all(pad_emb == 0)


class TestSASRecEmbedderRegistration:
    def test_registered(self):
        assert "sasrec" in COLLABORATIVE_EMBEDDER_REGISTRY


def _make_stage2_data(
    tmp_path: Path,
    num_users: int = 20,
    num_items: int = NUM_ITEMS,
    loo: bool = False,
):
    """Create minimal Stage 1 + Stage 2 data for testing.

    Args:
        loo: If True, simulate LOO split (valid/test have 1 item per user).
    """
    import random
    random.seed(42)

    id_map = Dataset.from_dict({
        "original_id": [f"item_{i}" for i in range(num_items)],
        "mapped_id": list(range(num_items)),
    })
    id_map.save_to_disk(str(tmp_path / "item_id_map"))

    if loo:
        _make_fields = lambda uids, seqs: {
            "user_id": uids,
            "item_ids": seqs,
            "timestamps": [[0] * len(s) for s in seqs],
            "ratings": [[1.0] * len(s) for s in seqs],
            "review_texts": [[""] * len(s) for s in seqs],
            "review_summaries": [[""] * len(s) for s in seqs],
        }
        train_seqs, valid_seqs, test_seqs = [], [], []
        user_ids = list(range(num_users))
        for uid in user_ids:
            full_len = random.randint(5, 10)
            full_seq = [random.randint(0, num_items - 1) for _ in range(full_len)]
            train_seqs.append(full_seq[:-2])
            valid_seqs.append(full_seq[-2:-1])
            test_seqs.append(full_seq[-1:])
        Dataset.from_dict(_make_fields(user_ids, train_seqs)).save_to_disk(
            str(tmp_path / "train_sequences")
        )
        Dataset.from_dict(_make_fields(user_ids, valid_seqs)).save_to_disk(
            str(tmp_path / "valid_sequences")
        )
        Dataset.from_dict(_make_fields(user_ids, test_seqs)).save_to_disk(
            str(tmp_path / "test_sequences")
        )
    else:
        for split_name in ("train_sequences", "valid_sequences", "test_sequences"):
            seqs = []
            for _ in range(num_users):
                seq_len = random.randint(3, 8)
                seq = [random.randint(0, num_items - 1) for _ in range(seq_len)]
                seqs.append(seq)
            ds = Dataset.from_dict({
                "user_id": list(range(num_users)),
                "item_ids": seqs,
                "timestamps": [[0] * len(s) for s in seqs],
                "ratings": [[1.0] * len(s) for s in seqs],
                "review_texts": [[""] * len(s) for s in seqs],
                "review_summaries": [[""] * len(s) for s in seqs],
            })
            ds.save_to_disk(str(tmp_path / split_name))


class TestBuildEvalSequences:
    """Test that eval sequences are correctly reconstructed from split data."""

    def test_valid_combines_train_and_target(self):
        train = {0: [1, 2, 3], 1: [4, 5]}
        valid = {0: [6], 1: [7]}
        result = SASRecEmbedder._build_eval_sequences(train, valid)
        assert len(result) == 2
        assert [1, 2, 3, 6] in result
        assert [4, 5, 7] in result

    def test_test_includes_valid_augmentation(self):
        train = {0: [1, 2, 3]}
        valid = {0: [4]}
        test = {0: [5]}
        result = SASRecEmbedder._build_eval_sequences(train, test, augment_seqs=valid)
        assert result == [[1, 2, 3, 4, 5]]

    def test_skips_user_with_too_short_sequence(self):
        train = {0: []}
        valid = {0: [1]}
        result = SASRecEmbedder._build_eval_sequences(train, valid)
        assert result == []

    def test_user_missing_from_train(self):
        train = {}
        valid = {0: [1, 2, 3]}
        result = SASRecEmbedder._build_eval_sequences(train, valid)
        assert result == [[1, 2, 3]]


class TestSASRecEmbedderGenerate:
    @pytest.mark.parametrize("loss_type", ["CE", "BPR"])
    def test_end_to_end_small(self, tmp_path, loss_type):
        _make_stage2_data(tmp_path)

        embedder = SASRecEmbedder()
        ds = embedder.generate(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 1,
                "num_heads": 1,
                "max_seq_len": MAX_SEQ_LEN,
                "dropout": 0.0,
                "batch_size": 8,
                "num_epochs": 2,
                "eval_top_k": [5, 10],
                "device": "cpu",
                "seed": 42,
                "loss_type": loss_type,
            },
        )

        assert len(ds) == NUM_ITEMS
        assert set(ds.column_names) == {"item_id", "embedding"}
        assert len(ds[0]["embedding"]) == HIDDEN_SIZE

    def test_end_to_end_loo_split(self, tmp_path):
        """LOO split: valid/test have 1 item each, eval should still work."""
        _make_stage2_data(tmp_path, loo=True)

        embedder = SASRecEmbedder()
        ds = embedder.generate(
            data_dir=tmp_path,
            output_dir=tmp_path,
            config={
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 1,
                "num_heads": 1,
                "max_seq_len": MAX_SEQ_LEN,
                "dropout": 0.0,
                "batch_size": 8,
                "num_epochs": 2,
                "eval_top_k": [5, 10],
                "device": "cpu",
                "seed": 42,
            },
        )

        assert len(ds) == NUM_ITEMS
        assert set(ds.column_names) == {"item_id", "embedding"}
        assert len(ds[0]["embedding"]) == HIDDEN_SIZE

    def test_skip_if_exists(self, tmp_path):
        _make_stage2_data(tmp_path)

        embedder = SASRecEmbedder()
        config = {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": 1,
            "num_heads": 1,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": 8,
            "num_epochs": 1,
            "device": "cpu",
            "seed": 42,
        }

        ds1 = embedder.generate(tmp_path, tmp_path, config)
        ds2 = embedder.generate(tmp_path, tmp_path, config)
        assert len(ds1) == len(ds2)
