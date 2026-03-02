"""SASRec-based collaborative embedder implementation."""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from saegenrec.data.embeddings.collaborative.base import (
    CollaborativeEmbedder,
    register_collaborative_embedder,
)
from saegenrec.data.embeddings.collaborative.models.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
)
from saegenrec.data.embeddings.collaborative.models.sasrec_model import SASRec
from saegenrec.data.schemas import COLLABORATIVE_EMBEDDING_FEATURES


class SequenceDataset(TorchDataset):
    """PyTorch Dataset wrapping user interaction sequences for SASRec training."""

    def __init__(
        self,
        sequences: list[list[int]],
        max_seq_len: int,
        num_items: int,
        mode: str = "train",
        loss_type: str = "CE",
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.mode = mode
        self.loss_type = loss_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        if self.mode == "train":
            input_ids = seq[:-1]
            pos_ids = seq[1:]
        else:
            input_ids = seq[:-1]
            pos_ids = seq[1:]

        input_ids = input_ids[-self.max_seq_len :]
        pos_ids = pos_ids[-self.max_seq_len :]

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        pos_ids = [0] * pad_len + pos_ids

        if self.mode == "train":
            if self.loss_type == "BPR":
                neg_ids = []
                for p in pos_ids:
                    if p == 0:
                        neg_ids.append(0)
                    else:
                        neg = random.randint(1, self.num_items)
                        while neg == p:
                            neg = random.randint(1, self.num_items)
                        neg_ids.append(neg)
                return (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(pos_ids, dtype=torch.long),
                    torch.tensor(neg_ids, dtype=torch.long),
                )
            else:
                return (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(pos_ids, dtype=torch.long),
                )
        else:
            target = seq[-1]
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
            )


class SASRecLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for SASRec model."""

    def __init__(
        self,
        model: SASRec,
        learning_rate: float = 0.001,
        eval_top_k: list[int] | None = None,
        loss_type: str = "CE",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.eval_top_k = eval_top_k or [10, 20]
        self.loss_type = loss_type

    def training_step(self, batch, batch_idx):
        if self.loss_type == "BPR":
            seq, pos, neg = batch
            loss = self.model.bpr_loss(seq, pos, neg)
        else:
            seq, pos = batch
            loss = self.model.ce_loss(seq, pos)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, target = batch
        scores = self.model.predict(seq)

        for k in self.eval_top_k:
            hr = hit_rate_at_k(scores, target - 1, k)
            ndcg = ndcg_at_k(scores, target - 1, k)
            self.log(f"val_HR@{k}", hr, prog_bar=True, sync_dist=True)
            self.log(f"val_NDCG@{k}", ndcg, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        seq, target = batch
        scores = self.model.predict(seq)

        for k in self.eval_top_k:
            hr = hit_rate_at_k(scores, target - 1, k)
            ndcg = ndcg_at_k(scores, target - 1, k)
            self.log(f"test_HR@{k}", hr, sync_dist=True)
            self.log(f"test_NDCG@{k}", ndcg, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


@register_collaborative_embedder("sasrec")
class SASRecEmbedder(CollaborativeEmbedder):
    """Train SASRec on user sequences and extract item collaborative embeddings."""

    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        hidden_size: int = config.get("hidden_size", 64)
        num_layers: int = config.get("num_layers", 2)
        num_heads: int = config.get("num_heads", 1)
        max_seq_len: int = config.get("max_seq_len", 50)
        dropout: float = config.get("dropout", 0.2)
        learning_rate: float = config.get("learning_rate", 0.001)
        batch_size: int = config.get("batch_size", 256)
        num_epochs: int = config.get("num_epochs", 200)
        eval_top_k: list[int] = config.get("eval_top_k", [10, 20])
        device_str: str = config.get("device", "auto")
        seed: int = config.get("seed", 42)
        force: bool = config.get("force", False)
        loss_type: str = config.get("loss_type", "CE")

        output_path = Path(output_dir) / "item_collaborative_embeddings"

        if output_path.exists() and not force:
            logger.info(
                f"Collaborative embeddings already exist at {output_path}, skipping. "
                "Use --force to regenerate."
            )
            return load_from_disk(str(output_path))

        if output_path.exists() and force:
            shutil.rmtree(output_path)

        data_dir = Path(data_dir)
        stage1_dir = self._find_stage1_dir(data_dir)

        train_seq_dir = data_dir / "train_sequences"
        valid_seq_dir = data_dir / "valid_sequences"
        test_seq_dir = data_dir / "test_sequences"
        item_id_map_dir = stage1_dir / "item_id_map"

        for required, name in [
            (train_seq_dir, "train_sequences"),
            (valid_seq_dir, "valid_sequences"),
            (test_seq_dir, "test_sequences"),
            (item_id_map_dir, "item_id_map"),
        ]:
            if not required.exists():
                raise FileNotFoundError(
                    f"{name} not found at {required}. "
                    "Run Stage 2 (split step) first."
                )

        pl.seed_everything(seed, workers=True)

        t0 = time.time()

        item_id_map = load_from_disk(str(item_id_map_dir))
        num_items = len(item_id_map)

        train_by_user = self._load_sequences_by_user(train_seq_dir)
        valid_by_user = self._load_sequences_by_user(valid_seq_dir)
        test_by_user = self._load_sequences_by_user(test_seq_dir)

        train_seqs = list(train_by_user.values())

        short_seq_count = sum(1 for s in train_seqs if len(s) <= 1)
        if short_seq_count == len(train_seqs):
            logger.warning(
                "All training sequences have length <= 1. "
                "Model training may not be effective."
            )

        valid_eval_seqs = self._build_eval_sequences(
            train_by_user, valid_by_user,
        )
        test_eval_seqs = self._build_eval_sequences(
            train_by_user, test_by_user, augment_seqs=valid_by_user,
        )

        logger.info(
            f"Eval sequences: valid={len(valid_eval_seqs)}, test={len(test_eval_seqs)} "
            f"(reconstructed from train history)"
        )

        train_dataset = SequenceDataset(
            train_seqs, max_seq_len, num_items, mode="train", loss_type=loss_type,
        )
        valid_dataset = SequenceDataset(valid_eval_seqs, max_seq_len, num_items, mode="eval")
        test_dataset = SequenceDataset(test_eval_seqs, max_seq_len, num_items, mode="eval")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, persistent_workers=False,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, persistent_workers=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, persistent_workers=False,
        )

        model = SASRec(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        lightning_module = SASRecLightningModule(
            model=model,
            learning_rate=learning_rate,
            eval_top_k=eval_top_k,
            loss_type=loss_type,
        )

        accelerator, devices = self._resolve_device(device_str)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            gradient_clip_val=5.0,
        )

        logger.info(
            f"Training SASRec (hidden={hidden_size}, layers={num_layers}, "
            f"loss={loss_type}, epochs={num_epochs}) on {accelerator}"
        )

        trainer.fit(lightning_module, train_loader, valid_loader)

        test_results = trainer.test(lightning_module, test_loader)
        if test_results:
            for k, v in test_results[0].items():
                logger.info(f"  {k}: {v:.4f}")

        embeddings = model.extract_item_embeddings().cpu().numpy()

        item_ids = list(range(0, num_items))
        ds = Dataset.from_dict(
            {
                "item_id": item_ids,
                "embedding": embeddings.tolist(),
            },
            features=COLLABORATIVE_EMBEDDING_FEATURES,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_path))

        elapsed = time.time() - t0
        logger.info(
            f"Saved {len(ds)} collaborative embeddings to {output_path}"
        )
        logger.info(
            f"Stats: items={len(ds)}, dim={hidden_size}, elapsed={elapsed:.1f}s"
        )

        return ds

    @staticmethod
    def _find_stage1_dir(data_dir: Path) -> Path:
        """Navigate from Stage 2 split directory up to Stage 1 directory.

        Stage 2 path: data/interim/{dataset}/{category}/{split_strategy}/
        Stage 1 path: data/interim/{dataset}/{category}/
        """
        if (data_dir / "item_id_map").exists():
            return data_dir
        parent = data_dir.parent
        if (parent / "item_id_map").exists():
            return parent
        raise FileNotFoundError(
            f"Cannot find item_id_map relative to {data_dir}. "
            "Ensure Stage 1 data exists in the parent directory."
        )

    @staticmethod
    def _load_sequences_by_user(seq_dir: Path) -> dict[int, list[int]]:
        """Load sequences keyed by user_id, with +1 ID shift for padding reservation."""
        ds = load_from_disk(str(seq_dir))
        return {
            row["user_id"]: [item_id + 1 for item_id in row["item_ids"]]
            for row in ds
        }

    @staticmethod
    def _build_eval_sequences(
        train_seqs: dict[int, list[int]],
        eval_seqs: dict[int, list[int]],
        augment_seqs: dict[int, list[int]] | None = None,
    ) -> list[list[int]]:
        """Reconstruct full eval sequences by prepending train history.

        RecBole-style evaluation requires the full interaction history as input:
        - Valid: seq = train_items + [valid_target]
        - Test:  seq = train_items + [valid_item] + [test_target]

        The SequenceDataset eval mode then splits seq into input=seq[:-1], target=seq[-1].
        """
        sequences = []
        for uid, eval_items in eval_seqs.items():
            seq = list(train_seqs.get(uid, []))
            if augment_seqs is not None:
                seq.extend(augment_seqs.get(uid, []))
            seq.extend(eval_items)
            if len(seq) >= 2:
                sequences.append(seq)
        return sequences

    @staticmethod
    def _resolve_device(device_str: str) -> tuple[str, int | str]:
        """Resolve device string to Lightning accelerator/devices config."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return "gpu", 1
            else:
                logger.warning(
                    "GPU not available, falling back to CPU. "
                    "Training will be significantly slower."
                )
                return "cpu", "auto"
        elif device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    f"Requested {device_str} but GPU not available. "
                    "Falling back to CPU."
                )
                return "cpu", "auto"
            return "gpu", 1
        else:
            return "cpu", "auto"
