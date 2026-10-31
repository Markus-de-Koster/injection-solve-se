"""SEModule per-step handling of a T>1 sequence batch (and the T=1 snapshot baseline).

Temporal MIXING is a later task; for now a [B,T,H,N] sequence is estimated per step (the mask
broadcasts over T). These checks exercise the SEModule pipeline directly (no Lightning loop).
"""

from __future__ import annotations

import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    MaskStageConfig,
)
from injection_hse.data import base_voltage_per_row, build_graph
from injection_hse.encoding import TrigEncoding
from injection_hse.normalization import HarmonicNormalizer
from injection_hse.train import SEModule

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _module(cigre_grid, v):
    cfg = ExperimentConfig(
        name="seq",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="x", batch_size=4),
        model=DNNConfig(hidden_dims=[32]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
    )
    g = build_graph(cigre_grid)
    feats = TrigEncoding().encode(v).transpose(-3, -2)  # [*lead, N, H, C]
    norm = HarmonicNormalizer().fit(feats)
    return SEModule(
        cfg,
        n_nodes=g.num_nodes,
        normalizer=norm,
        graph=g,
        base_v=base_voltage_per_row(cigre_grid),
    )


def test_sequence_step_is_per_step(cigre_grid):
    n = build_graph(cigre_grid).num_nodes
    b, t, h = 6, 4, len(ORDERS)
    v = (torch.randn(b, t, h, n) + 1j * torch.randn(b, t, h, n)).to(torch.complex64)
    mod = _module(cigre_grid, v)
    loss, mask, pred_v, target_v, inj_loss, phys_loss, _ = mod._step(
        {"v": v}, train=True
    )
    assert torch.isfinite(loss) and loss.requires_grad
    assert mask.shape == (b, t, n)  # a fresh per-(scenario, step) mask
    assert pred_v.shape == (b, t, h, n)  # T preserved in the metric layout
    assert target_v.shape == (b, t, h, n)
    assert inj_loss is None and phys_loss is None  # both off by default


def test_snapshot_step_unchanged(cigre_grid):
    n = build_graph(cigre_grid).num_nodes
    b, h = 6, len(ORDERS)
    v = (torch.randn(b, h, n) + 1j * torch.randn(b, h, n)).to(torch.complex64)
    mod = _module(cigre_grid, v)
    loss, mask, pred_v, target_v, inj_loss, phys_loss, _ = mod._step(
        {"v": v}, train=True
    )
    assert mask.shape == (b, n)
    assert pred_v.shape == (b, h, n)
