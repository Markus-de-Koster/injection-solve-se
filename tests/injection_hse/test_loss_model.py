"""Loss + DNN model: shapes, gradient flow, weight modes, and metrics."""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import DNNConfig, HarmonicWeightConfig, LossConfig
from injection_hse.encoding import TrigEncoding
from injection_hse.io import build_model_input
from injection_hse.loss import StateEstimationLoss
from injection_hse.masking import build_masker
from injection_hse.metrics import state_errors
from injection_hse.models import build_model
from injection_hse.normalization import HarmonicNormalizer

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _stub_graph(n: int):
    """A minimal no-edge PyG ``Data`` (``x [N,8]``, empty edges) for graph-agnostic models."""
    from torch_geometric.data import Data

    d = Data(
        x=torch.zeros(n, 8),
        edge_index=torch.zeros(2, 0, dtype=torch.int64),
        edge_attr=torch.zeros(0, 5),
    )
    d.num_nodes = n
    return d


def _setup(B=8, N=12, encoding=None):
    enc = encoding or TrigEncoding()
    torch.manual_seed(0)
    v = (torch.randn(B, 7, N) + 1j * torch.randn(B, 7, N)).to(torch.complex64)
    v = v * torch.tensor([1.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.003]).view(1, 7, 1)
    feats = enc.encode(v).transpose(1, 2)  # [B, N, H, C]
    norm = HarmonicNormalizer().fit(feats)
    return enc, norm, v, norm.transform(feats)


def test_dnn_forward_backward_shapes():
    enc, norm, v, tgt = _setup()
    B, N, H, C = tgt.shape
    model = build_model(DNNConfig(hidden_dims=[64, 64]), n_features=H * C, n_nodes=N)
    mask = build_masker("random", observed_fraction=0.5).mask(B, N)
    masked = (tgt.reshape(B, N, H * C) * mask.unsqueeze(-1)).clone()
    mi = build_model_input(_stub_graph(N), masked, mask)
    pred = model(mi).reshape(B, N, H, C)  # squeeze T=1
    assert pred.shape == tgt.shape

    loss_fn = StateEstimationLoss(
        LossConfig(), encoding=enc, normalizer=norm, orders=ORDERS
    )
    loss = loss_fn(pred, tgt)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert float(grad) > 0


@pytest.mark.parametrize("mode", ["physical", "uniform", "custom"])
def test_weight_modes(mode):
    enc, norm, v, tgt = _setup()
    values = [1.0] * len(ORDERS) if mode == "custom" else None
    cfg = LossConfig(harmonic_weight=HarmonicWeightConfig(mode=mode, values=values))
    loss_fn = StateEstimationLoss(cfg, encoding=enc, normalizer=norm, orders=ORDERS)
    w = loss_fn._weights
    assert w.shape == (len(ORDERS),)
    if mode == "physical":
        # the fundamental is weighted far above the high orders
        assert w[0] > w[-1] * 10
    else:
        assert torch.allclose(w, torch.ones_like(w))


def test_magnitude_term_increases_loss():
    enc, norm, v, tgt = _setup()
    B, N, H, C = tgt.shape
    pred = tgt + 0.1 * torch.randn_like(tgt)
    base = StateEstimationLoss(
        LossConfig(magnitude_term_weight=0.0),
        encoding=enc,
        normalizer=norm,
        orders=ORDERS,
    )(pred, tgt)
    withmag = StateEstimationLoss(
        LossConfig(magnitude_term_weight=1.0),
        encoding=enc,
        normalizer=norm,
        orders=ORDERS,
    )(pred, tgt)
    assert float(withmag) > float(base)


def test_loss_mask_restricts_supervision():
    enc, norm, v, tgt = _setup()
    B, N, H, C = tgt.shape
    pred = tgt.clone()
    # introduce error only on node 0
    pred[:, 0] = pred[:, 0] + 1.0
    loss_fn = StateEstimationLoss(
        LossConfig(), encoding=enc, normalizer=norm, orders=ORDERS
    )
    mask_excl0 = torch.ones(B, N, dtype=torch.bool)
    mask_excl0[:, 0] = False  # supervise everything BUT node 0
    assert float(loss_fn(pred, tgt, mask=mask_excl0)) == pytest.approx(0.0, abs=1e-6)
    assert float(loss_fn(pred, tgt, mask=None)) > 0


def test_state_errors_split_and_finite():
    enc, norm, v, tgt = _setup()
    B, N, H, C = tgt.shape
    pred_v = enc.decode(norm.inverse_transform(tgt)).transpose(1, 2)  # perfect
    mask = build_masker("random", observed_fraction=0.5).mask(B, N)
    errs = state_errors(pred_v, v, mask=mask, orders=ORDERS)
    for key in ("mae_obs", "rmse_obs", "mae_unobs", "rmse_unobs"):
        assert key in errs
        assert errs[key] == pytest.approx(0.0, abs=1e-4)
    for order in ORDERS:
        assert {"mae_obs", "rmse_obs", "mae_unobs", "rmse_unobs"} <= set(errs[order])
