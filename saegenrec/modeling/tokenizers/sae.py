"""SAE-based ItemTokenizer using JumpReLU Sparse Autoencoder."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk
from loguru import logger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from saegenrec.modeling.tokenizers.base import ItemTokenizer, register_item_tokenizer
from saegenrec.modeling.tokenizers.models.jumprelu_sae import (
    JumpReLUSAE,
    compute_loss,
)


class _EmbeddingDataset(TorchDataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


@register_item_tokenizer("sae")
class SAETokenizer(ItemTokenizer):
    """ItemTokenizer that uses a JumpReLU SAE for concept-based SID generation.

    Maps ``num_codebooks`` → ``top_k`` and ``codebook_size`` → ``d_sae``.
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_size: int = 8192,
        **kwargs,
    ):
        self._top_k = kwargs.pop("top_k", None) or num_codebooks
        self._d_sae = kwargs.pop("d_sae", None) or codebook_size
        self._kwargs = kwargs
        self._model: JumpReLUSAE | None = None

        if self._d_sae <= self._top_k:
            raise ValueError(
                f"d_sae ({self._d_sae}) must be greater than top_k ({self._top_k})"
            )

    @property
    def num_codebooks(self) -> int:
        return self._top_k

    @property
    def codebook_size(self) -> int:
        return self._d_sae

    def train(
        self,
        semantic_embeddings_dir: Path | str,
        collaborative_embeddings_dir: Path | str | None,
        config: dict,
    ) -> dict:
        ds = load_from_disk(str(semantic_embeddings_dir))
        embeddings = torch.tensor(ds["embedding"], dtype=torch.float32)
        d_in = embeddings.shape[1]

        seed = int(config.get("seed", self._kwargs.get("seed", 42)))
        torch.manual_seed(seed)

        lr = config.get("lr", config.get("learning_rate", self._kwargs.get("learning_rate", 1e-3)))
        epochs = int(config.get("epochs", self._kwargs.get("epochs", 50)))
        batch_size = int(config.get("batch_size", self._kwargs.get("batch_size", 256)))
        l0_coefficient = float(
            config.get("l0_coefficient", self._kwargs.get("l0_coefficient", 1e-3))
        )
        jumprelu_bandwidth = float(
            config.get("jumprelu_bandwidth", self._kwargs.get("jumprelu_bandwidth", 0.05))
        )
        jumprelu_init_threshold = float(
            config.get(
                "jumprelu_init_threshold", self._kwargs.get("jumprelu_init_threshold", 0.01)
            )
        )
        device_str = config.get("device", self._kwargs.get("device", "cpu"))
        device = torch.device(device_str)

        model = JumpReLUSAE(
            d_in=d_in,
            d_sae=self._d_sae,
            jumprelu_init_threshold=jumprelu_init_threshold,
            jumprelu_bandwidth=jumprelu_bandwidth,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        loader = DataLoader(
            _EmbeddingDataset(embeddings), batch_size=batch_size, shuffle=True
        )

        model.train()
        final_mse = 0.0
        final_l0 = 0.0
        final_total = 0.0

        for epoch in range(epochs):
            epoch_mse = 0.0
            epoch_l0 = 0.0
            epoch_total = 0.0
            n_batches = 0

            for batch in loader:
                batch = batch.to(device)
                sae_out, feature_acts, hidden_pre = model(batch)
                total_loss, mse_loss, l0_loss = compute_loss(
                    batch, sae_out, hidden_pre,
                    model.threshold, model.bandwidth, l0_coefficient,
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_mse += mse_loss.item()
                epoch_l0 += l0_loss.item()
                epoch_total += total_loss.item()
                n_batches += 1

            final_mse = epoch_mse / n_batches
            final_l0 = epoch_l0 / n_batches
            final_total = epoch_total / n_batches

            logger.info(
                f"Epoch {epoch + 1}/{epochs} — "
                f"MSE: {final_mse:.6f}, L0: {final_l0:.2f}, Total: {final_total:.6f}"
            )

        self._model = model
        self._model.eval()

        with torch.no_grad():
            all_acts = model.encode(embeddings.to(device))
            active_per_sample = (all_acts > 0).float().sum(dim=-1)
            mean_l0 = active_per_sample.mean().item()
            ever_active = (all_acts > 0).any(dim=0)
            num_alive = ever_active.sum().item()
            num_dead = self._d_sae - num_alive
            vocab_utilization = num_alive / self._d_sae

        logger.info(
            f"Training complete — vocab utilization: {vocab_utilization:.1%} "
            f"({num_alive}/{self._d_sae}), mean L0: {mean_l0:.1f}, "
            f"dead features: {num_dead}"
        )

        return {
            "final_mse_loss": final_mse,
            "final_l0_loss": final_l0,
            "final_total_loss": final_total,
            "mean_l0": mean_l0,
            "vocab_utilization": vocab_utilization,
            "num_dead_features": num_dead,
        }

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError(
                "SAE model not trained or loaded. Call train() or load() first."
            )
        if embeddings.shape[-1] != self._model.d_in:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._model.d_in}, "
                f"got {embeddings.shape[-1]}"
            )
        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            feature_acts = self._model.encode(embeddings.to(device))
            _, top_indices = torch.topk(feature_acts, k=self._top_k, dim=-1)
        return top_indices.cpu()

    def save(self, path: Path | str) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._model is None:
            raise RuntimeError("No model to save. Call train() first.")

        tensors = {
            "W_enc": self._model.W_enc.data,
            "b_enc": self._model.b_enc.data,
            "W_dec": self._model.W_dec.data,
            "b_dec": self._model.b_dec.data,
            "log_threshold": self._model.log_threshold.data,
        }
        save_file(tensors, path / "sae_weights.safetensors")

        hparams = {
            "d_in": self._model.d_in,
            "d_sae": self._model.d_sae,
            "top_k": self._top_k,
            "jumprelu_bandwidth": self._model.bandwidth,
            "jumprelu_init_threshold": float(
                torch.exp(self._model.log_threshold[0]).item()
            ),
        }
        with open(path / "hparams.json", "w") as f:
            json.dump(hparams, f, indent=2)

    def load(self, path: Path | str) -> None:
        from safetensors.torch import load_file

        path = Path(path)

        with open(path / "hparams.json") as f:
            hparams = json.load(f)

        model = JumpReLUSAE(
            d_in=hparams["d_in"],
            d_sae=hparams["d_sae"],
            jumprelu_init_threshold=hparams.get("jumprelu_init_threshold", 0.01),
            jumprelu_bandwidth=hparams.get("jumprelu_bandwidth", 0.05),
        )

        weights = load_file(path / "sae_weights.safetensors")
        model.W_enc.data.copy_(weights["W_enc"])
        model.b_enc.data.copy_(weights["b_enc"])
        model.W_dec.data.copy_(weights["W_dec"])
        model.b_dec.data.copy_(weights["b_dec"])
        model.log_threshold.data.copy_(weights["log_threshold"])

        model.eval()
        self._model = model
        self._top_k = hparams.get("top_k", self._top_k)
        self._d_sae = hparams["d_sae"]
