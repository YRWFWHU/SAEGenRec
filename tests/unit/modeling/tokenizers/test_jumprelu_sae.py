"""Unit tests for JumpReLU SAE model (T006)."""

from __future__ import annotations

import torch

from saegenrec.modeling.tokenizers.models.jumprelu_sae import (
    JumpReLUSAE,
    compute_loss,
)

D_IN = 64
D_SAE = 128
BATCH = 16


def _make_model() -> JumpReLUSAE:
    return JumpReLUSAE(d_in=D_IN, d_sae=D_SAE)


def test_forward_output_shapes():
    model = _make_model()
    x = torch.randn(BATCH, D_IN)
    sae_out, feature_acts, hidden_pre = model(x)
    assert sae_out.shape == (BATCH, D_IN)
    assert feature_acts.shape == (BATCH, D_SAE)
    assert hidden_pre.shape == (BATCH, D_SAE)


def test_encode_output_is_sparse():
    model = _make_model()
    x = torch.randn(BATCH, D_IN)
    feature_acts = model.encode(x)
    zero_fraction = (feature_acts == 0).float().mean().item()
    assert zero_fraction > 0.1, "JumpReLU should produce some zero activations"


def test_decode_restores_dimension():
    model = _make_model()
    feature_acts = torch.randn(BATCH, D_SAE)
    decoded = model.decode(feature_acts)
    assert decoded.shape == (BATCH, D_IN)


def test_gradients_flow():
    model = _make_model()
    x = torch.randn(BATCH, D_IN, requires_grad=True)
    sae_out, feature_acts, hidden_pre = model(x)
    total_loss, _, _ = compute_loss(
        x, sae_out, hidden_pre, model.threshold, model.bandwidth, l0_coefficient=1e-3
    )
    total_loss.backward()

    assert x.grad is not None
    assert model.W_enc.grad is not None
    assert model.W_dec.grad is not None
    assert model.log_threshold.grad is not None


def test_compute_loss_values():
    model = _make_model()
    x = torch.randn(BATCH, D_IN)
    sae_out, feature_acts, hidden_pre = model(x)
    total_loss, mse_loss, l0_loss = compute_loss(
        x, sae_out, hidden_pre, model.threshold, model.bandwidth, l0_coefficient=1e-3
    )
    assert total_loss.ndim == 0
    assert mse_loss.ndim == 0
    assert l0_loss.ndim == 0
    assert total_loss.item() >= mse_loss.item()
    assert l0_loss.item() >= 0


def test_threshold_property():
    model = _make_model()
    threshold = model.threshold
    assert threshold.shape == (D_SAE,)
    assert (threshold > 0).all(), "threshold = exp(log_threshold) must be positive"


def test_model_parameters_count():
    model = _make_model()
    param_names = {name for name, _ in model.named_parameters()}
    expected = {"W_enc", "b_enc", "W_dec", "b_dec", "log_threshold"}
    assert param_names == expected
