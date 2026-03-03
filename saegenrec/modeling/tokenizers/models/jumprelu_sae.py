"""JumpReLU Sparse Autoencoder model for item tokenization.

Architecture reference: SAELens JumpReLU implementation
(references/SAELens/sae_lens/saes/jumprelu_sae.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def rectangle(x: torch.Tensor) -> torch.Tensor:
    """Rectangle function used as STE gradient approximation."""
    return ((x > -0.5) & (x < 0.5)).to(x)


class JumpReLU(torch.autograd.Function):
    """JumpReLU activation: x * (x > threshold) with STE gradient for threshold."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class Step(torch.autograd.Function):
    """Step function: (x > threshold) with STE gradient for threshold."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder for learning sparse concept representations.

    Parameters
    ----------
    d_in : int
        Input embedding dimension.
    d_sae : int
        SAE hidden dimension (number of concept features / vocabulary size).
    jumprelu_init_threshold : float
        Initial value for the JumpReLU threshold (before log transform).
    jumprelu_bandwidth : float
        Bandwidth parameter for the STE gradient approximation.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        jumprelu_init_threshold: float = 0.01,
        jumprelu_bandwidth: float = 0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = jumprelu_bandwidth

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(
            torch.ones(d_sae) * np.log(jumprelu_init_threshold)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    @property
    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations via JumpReLU."""
        hidden_pre = x @ self.W_enc + self.b_enc
        return JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse feature activations back to input space."""
        return feature_acts @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        sae_out : torch.Tensor
            Reconstructed input, shape ``(batch, d_in)``.
        feature_acts : torch.Tensor
            Sparse feature activations after JumpReLU, shape ``(batch, d_sae)``.
        hidden_pre : torch.Tensor
            Pre-activation hidden values (before JumpReLU), shape ``(batch, d_sae)``.
        """
        hidden_pre = x @ self.W_enc + self.b_enc
        feature_acts = JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)
        sae_out = self.decode(feature_acts)
        return sae_out, feature_acts, hidden_pre


def compute_loss(
    x: torch.Tensor,
    sae_out: torch.Tensor,
    hidden_pre: torch.Tensor,
    threshold: torch.Tensor,
    bandwidth: float,
    l0_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute JumpReLU SAE training loss.

    Returns
    -------
    total_loss : torch.Tensor
        Combined MSE + L0 sparsity loss.
    mse_loss : torch.Tensor
        Mean squared error reconstruction loss.
    l0_loss : torch.Tensor
        L0 sparsity penalty via Step function with STE gradient.
    """
    mse_loss = torch.nn.functional.mse_loss(sae_out, x)
    l0_per_sample = Step.apply(hidden_pre, threshold, bandwidth).sum(dim=-1)
    l0_loss = l0_per_sample.mean()
    total_loss = mse_loss + l0_coefficient * l0_loss
    return total_loss, mse_loss, l0_loss
