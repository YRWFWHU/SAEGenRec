"""RQ-VAE PyTorch Lightning module with codebook collapse prevention."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAEModel(pl.LightningModule):
    """Residual-Quantised Variational Auto-Encoder for item tokenisation.

    Anti-collapse mechanisms:
    - EMA codebook updates (more stable than pure gradient)
    - Dead code replacement (re-init unused entries from batch)
    - Codebook initialization from data (first batch)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        learning_rate: float = 1e-3,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for _ in range(num_codebooks)]
        )
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        for i in range(num_codebooks):
            self.register_buffer(f"ema_count_{i}", torch.zeros(codebook_size))
            self.register_buffer(f"ema_weight_{i}", torch.zeros(codebook_size, latent_dim))

        self.register_buffer("_initialized", torch.tensor(False))

    def _init_codebooks_from_data(self, z: torch.Tensor) -> None:
        """Initialize codebook entries from k-means-like data sampling."""
        n = z.shape[0]
        for i, cb in enumerate(self.codebooks):
            k = min(self.hparams.codebook_size, n)
            perm = torch.randperm(n, device=z.device)[:k]
            noise = torch.randn_like(z[perm]) * 0.01
            cb.weight.data[:k] = z[perm] + noise
            if k < self.hparams.codebook_size:
                extra = torch.randint(0, n, (self.hparams.codebook_size - k,), device=z.device)
                cb.weight.data[k:] = z[extra] + torch.randn_like(z[extra]) * 0.1
            getattr(self, f"ema_weight_{i}").data.copy_(cb.weight.data)
            getattr(self, f"ema_count_{i}").data.fill_(1.0)

    def _replace_dead_codes(self, codebook_idx: int, z: torch.Tensor) -> None:
        """Replace codebook entries that haven't been used."""
        ema_count = getattr(self, f"ema_count_{codebook_idx}")
        dead_mask = ema_count < self.hparams.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return
        n = z.shape[0]
        replace_idx = torch.randint(0, n, (num_dead,), device=z.device)
        self.codebooks[codebook_idx].weight.data[dead_mask] = (
            z[replace_idx].detach() + torch.randn(num_dead, z.shape[1], device=z.device) * 0.01
        )
        ema_count[dead_mask] = 1.0
        ema_weight = getattr(self, f"ema_weight_{codebook_idx}")
        ema_weight[dead_mask] = self.codebooks[codebook_idx].weight.data[dead_mask]

    def _ema_update(self, codebook_idx: int, indices: torch.Tensor, encoded: torch.Tensor) -> None:
        """EMA update for codebook entries."""
        decay = self.hparams.ema_decay
        ema_count = getattr(self, f"ema_count_{codebook_idx}")
        ema_weight = getattr(self, f"ema_weight_{codebook_idx}")

        one_hot = F.one_hot(indices, self.hparams.codebook_size).float()
        new_count = one_hot.sum(dim=0)
        new_weight = one_hot.t() @ encoded.detach()

        ema_count.data.mul_(decay).add_(new_count, alpha=1 - decay)
        ema_weight.data.mul_(decay).add_(new_weight, alpha=1 - decay)

        n = ema_count.clamp(min=1e-5)
        self.codebooks[codebook_idx].weight.data.copy_(ema_weight / n.unsqueeze(1))

    def residual_quantize(
        self, z: torch.Tensor, update_codebook: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes: list[torch.Tensor] = []
        residual = z.clone()
        quantized_sum = torch.zeros_like(z)
        total_commitment = torch.tensor(0.0, device=z.device)

        for i, codebook in enumerate(self.codebooks):
            dists = torch.cdist(residual, codebook.weight)
            indices = dists.argmin(dim=-1)
            quantized = codebook(indices)

            total_commitment = total_commitment + F.mse_loss(residual, quantized.detach())

            if update_codebook and self.training:
                self._ema_update(i, indices, residual)
                if self.global_step > 0 and self.global_step % 100 == 0:
                    self._replace_dead_codes(i, residual)

            quantized_st = residual + (quantized - residual).detach()
            quantized_sum = quantized_sum + quantized_st
            residual = residual - quantized.detach()
            codes.append(indices)

        return quantized_sum, torch.stack(codes, dim=-1), total_commitment

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)

        if not self._initialized and self.training:
            self._init_codebooks_from_data(z.detach())
            self._initialized.fill_(True)

        quantized, codes, commitment_loss = self.residual_quantize(
            z, update_codebook=self.training
        )
        x_recon = self.decoder(quantized)
        return x_recon, codes, commitment_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_recon, _, commitment_loss = self(batch)
        recon_loss = F.mse_loss(x_recon, batch)
        loss = recon_loss + self.hparams.commitment_cost * commitment_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_recon, codes, commitment_loss = self(batch)
        recon_loss = F.mse_loss(x_recon, batch)
        loss = recon_loss + self.hparams.commitment_cost * commitment_loss
        self.log("val_loss", loss, prog_bar=True)

        for i in range(self.hparams.num_codebooks):
            unique = codes[:, i].unique().numel()
            utilization = unique / self.hparams.codebook_size
            self.log(f"codebook_{i}_utilization", utilization)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
