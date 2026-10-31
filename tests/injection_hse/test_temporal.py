"""Temporal estimator: forward shape, CAUSALITY, gradient flow, and a coherent curriculum.

The :class:`injection_hse.models.TemporalEstimator` wraps a spatial model with a causal per-node GRU
memory over the ``T`` axis. Causality is the load-bearing property for the streaming use case:
the estimate at step ``t`` must NOT depend on observations at steps ``> t``.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    GNNConfig,
    GraphormerConfig,
    LossConfig,
    MaskStageConfig,
    OptimConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import build_graph
from injection_hse.io import build_model_input
from injection_hse.models import TemporalEstimator, build_model

ORDERS = [1, 3, 5, 7, 9, 11, 13]
F = len(ORDERS) * 3  # trig channels


def _spatials():
    return [
        DNNConfig(hidden_dims=[64]),
        GNNConfig(hidden_dim=48, num_layers=2, conv="gine"),
        GraphormerConfig(hidden_dim=32, num_layers=1, num_heads=4),
    ]


def test_build_model_wraps_temporal(cigre_grid):
    g = build_graph(cigre_grid, orders=ORDERS)
    cfg = TemporalConfig(spatial=GNNConfig(conv="sage"), temporal="gru")
    m = build_model(cfg, n_features=F, n_nodes=g.num_nodes)
    assert isinstance(m, TemporalEstimator)
    assert type(m.spatial).__name__ == "GNN"


def test_causal_attention_seam_raises(cigre_grid):
    g = build_graph(cigre_grid, orders=ORDERS)
    cfg = TemporalConfig(spatial=DNNConfig(), temporal="causal_attention")
    with pytest.raises(NotImplementedError):
        build_model(cfg, n_features=F, n_nodes=g.num_nodes)


@pytest.mark.parametrize("spatial_idx", [0, 1, 2])
def test_temporal_forward_backward(cigre_grid, spatial_idx):
    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    b, t = 4, 5
    cfg = TemporalConfig(
        spatial=_spatials()[spatial_idx], temporal="gru", hidden_dim=32
    )
    m = build_model(cfg, n_features=F, n_nodes=n)
    obs = torch.randn(b, t, n, F, requires_grad=True)
    mask = torch.rand(b, t, n) > 0.5
    out = m(build_model_input(g, obs, mask, has_time=True))
    assert out.shape == (b, t, n, F)  # [*B, T, N, Fnd]
    out.pow(2).mean().backward()
    grad = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert float(grad) > 0  # both the GRU memory and the spatial model train
    assert obs.grad is not None and float(obs.grad.abs().sum()) > 0


@pytest.mark.parametrize("spatial_idx", [0, 1])
def test_temporal_is_causal(cigre_grid, spatial_idx):
    """Perturbing observations at steps > t0 must NOT change the output at steps <= t0."""
    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    b, t, t0 = 3, 6, 2
    cfg = TemporalConfig(
        spatial=_spatials()[spatial_idx], temporal="gru", hidden_dim=32
    )
    m = build_model(cfg, n_features=F, n_nodes=n).eval()
    torch.manual_seed(0)
    obs = torch.randn(b, t, n, F)
    mask = torch.rand(b, t, n) > 0.5
    with torch.no_grad():
        out1 = m(build_model_input(g, obs, mask.clone(), has_time=True))
        # perturb BOTH the observations and the mask at every step strictly after t0
        obs2 = obs.clone()
        obs2[:, t0 + 1 :] += 10.0 * torch.randn_like(obs2[:, t0 + 1 :])
        mask2 = mask.clone()
        mask2[:, t0 + 1 :] = ~mask2[:, t0 + 1 :]
        out2 = m(build_model_input(g, obs2, mask2, has_time=True))
    # steps <= t0 are byte-identical (causal); steps > t0 genuinely change
    assert torch.equal(out1[:, : t0 + 1], out2[:, : t0 + 1])
    assert (out1[:, t0 + 1 :] - out2[:, t0 + 1 :]).abs().max() > 0


def test_temporal_curriculum_smoke(
    cigre_grid, se_sequence_dataset, tmp_path, monkeypatch
):
    """The temporal model trains on the real coherent [B,T,H,N] dataset (val loss drops)."""
    from injection_hse import train as train_mod

    history: list[float] = []

    class Recorder(L.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_loss")
            if v is not None:
                history.append(float(v))

    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        cbs = list(kwargs.get("callbacks") or [])
        cbs.append(Recorder())
        kwargs["callbacks"] = cbs
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    cfg = ExperimentConfig(
        name="temporal-smoke",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_sequence_dataset),
            batch_size=16,
            val_fraction=0.25,
            num_workers=0,
        ),
        model=TemporalConfig(
            spatial=GNNConfig(hidden_dim=48, num_layers=2, conv="gine"),
            temporal="gru",
            hidden_dim=48,
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=15,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=2e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    module = train_mod.train_experiment(cfg, grid=cigre_grid)
    assert isinstance(module.model, TemporalEstimator)
    assert len(history) >= 3, history
    assert all(torch.isfinite(torch.tensor(history)))
    assert history[-1] < history[0] * 0.9, history


def test_temporal_composes_with_injection_head(cigre_grid, se_dataset):
    """The wrapper composes with the per-device localization head (snapshot T=1)."""
    from injection_hse.data import ParquetDataset, base_voltage_per_row
    from injection_hse.encoding import TrigEncoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.data import StateCollate
    from injection_hse.train import SEModule

    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2, seed=0)
    batch = StateCollate()([ds.train_dataset()[i] for i in range(16)])
    v = batch["v"]
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(v).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="t",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=TemporalConfig(
            spatial=DNNConfig(hidden_dims=[64]), temporal="gru", hidden_dim=32
        ),
        loss=LossConfig(predict_injection=True),
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
    mod = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        injection_spec=ds.injection_spec,
    )
    assert isinstance(mod.model, TemporalEstimator)
    assert mod.injection_head is not None
    loss, mask, pred_v, _, inj, _, _ = mod._step(batch, train=True)
    assert torch.isfinite(loss) and inj is not None
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in mod.model.parameters() if p.grad is not None)
    assert float(grad) > 0  # gradient reaches the temporal model (GRU + spatial)


def test_temporal_localization_on_coherent(cigre_grid, se_sequence_dataset):
    """The localization head works on the COHERENT [B,T,H,N] data (time axis preserved)."""
    from injection_hse.data import ParquetDataset, base_voltage_per_row
    from injection_hse.encoding import TrigEncoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.data import StateCollate
    from injection_hse.train import SEModule

    ds = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, val_fraction=0.25, seed=0
    )
    assert not ds.injection_spec.is_empty
    batch = StateCollate()([ds.train_dataset()[i] for i in range(8)])
    # the per-step targets keep the time axis through the collate
    assert batch["v"].dim() == 4  # [B, T, H, N]
    assert batch["injection_target"].dim() == 5  # [B, T, D, O, 1]

    v = batch["v"]
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(v).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="t",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=TemporalConfig(
            spatial=GNNConfig(hidden_dim=32, num_layers=1, conv="gine"),
            temporal="gru",
            hidden_dim=32,
        ),
        loss=LossConfig(predict_injection=True),
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
    mod = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        injection_spec=ds.injection_spec,
    )
    assert mod.injection_head is not None
    loss, mask, pred_v, _, inj, _, _ = mod._step(batch, train=True)
    assert pred_v.shape == v.shape  # per-step [B, T, H, N]
    assert torch.isfinite(loss) and inj is not None
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in mod.model.parameters() if p.grad is not None)
    assert float(grad) > 0
