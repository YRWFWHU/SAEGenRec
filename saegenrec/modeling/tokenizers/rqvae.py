"""RQ-VAE ItemTokenizer implementation (T014)."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk
from loguru import logger
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from saegenrec.modeling.tokenizers.base import ItemTokenizer, register_item_tokenizer
from saegenrec.modeling.tokenizers.models.rqvae_model import RQVAEModel


class EmbeddingDataset(TorchDataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


@register_item_tokenizer("rqvae")
class RQVAETokenizer(ItemTokenizer):
    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        **kwargs,
    ):
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._model: RQVAEModel | None = None

    def train(
        self,
        semantic_embeddings_dir: Path | str,
        collaborative_embeddings_dir: Path | str | None,
        config: dict,
    ) -> dict:
        ds = load_from_disk(str(semantic_embeddings_dir))
        embeddings = torch.tensor(ds["embedding"], dtype=torch.float32)
        embedding_dim = embeddings.shape[1]

        learning_rate = config.get("lr", config.get("learning_rate", 1e-3))
        model = RQVAEModel(
            embedding_dim=embedding_dim,
            hidden_dim=self._hidden_dim,
            latent_dim=self._latent_dim,
            num_codebooks=self._num_codebooks,
            codebook_size=self._codebook_size,
            learning_rate=learning_rate,
            commitment_cost=config.get("commitment_cost", 0.25),
            ema_decay=config.get("ema_decay", 0.99),
            dead_code_threshold=config.get("dead_code_threshold", 2),
        )

        batch_size = config.get("batch_size", 256)
        max_epochs = config.get("epochs", config.get("max_epochs", 50))
        train_loader = DataLoader(
            EmbeddingDataset(embeddings), batch_size=batch_size, shuffle=True
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="auto",
        )
        trainer.fit(model, train_loader)

        self._model = model
        self._model.eval()

        with torch.no_grad():
            codes = self.encode(embeddings)
        for i in range(self._num_codebooks):
            unique = codes[:, i].unique().numel()
            logger.info(
                f"Codebook {i} utilization: "
                f"{unique}/{self._codebook_size} ({unique / self._codebook_size:.1%})"
            )

        metrics = {}
        for k, v in trainer.callback_metrics.items():
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        return metrics

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "Model not trained or loaded"
        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            z = self._model.encoder(embeddings.to(device))
            _, codes, _ = self._model.residual_quantize(z)
        return codes.cpu()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "rqvae_state.pt")
        with open(path / "hparams.json", "w") as f:
            json.dump(dict(self._model.hparams), f)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with open(path / "hparams.json") as f:
            hparams = json.load(f)
        self._model = RQVAEModel(**hparams)
        self._model.load_state_dict(torch.load(path / "rqvae_state.pt", weights_only=True))
        self._model.eval()

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size
