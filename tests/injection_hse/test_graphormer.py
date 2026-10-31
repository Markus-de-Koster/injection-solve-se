"""Graphormer: structural encodings, forward/grad/device, and a curriculum smoke run.

The Graphormer is pure torch (no torch-geometric), so unlike the GNN these run on any
supported Python.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GraphormerConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    TrainConfig,
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def test_structural_hop_distance_and_degree():
    """Hand-checkable path graph 0-1-2-3: hop distances + degrees are exact."""
    from injection_hse.models.graphormer import Graphormer

    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]], dtype=torch.int64
    )  # undirected path
    dist, degree = Graphormer._compute_structural(edge_index, n=4)
    assert dist[0, 0] == 0 and dist[1, 1] == 0
    assert dist[0, 1] == 1 and dist[1, 2] == 1 and dist[2, 3] == 1
    assert dist[0, 2] == 2 and dist[1, 3] == 2
    assert dist[0, 3] == 3
    assert torch.equal(dist, dist.T)  # symmetric
    assert degree.tolist() == [1, 2, 2, 1]  # endpoints degree 1, interior degree 2


def test_unreachable_gets_far_bucket():
    """A disconnected node sits in the far/unreachable bucket (NUM_SPATIAL-1)."""
    from injection_hse.models.graphormer import Graphormer

    edge_index = torch.tensor(
        [[0], [1]], dtype=torch.int64
    )  # 0-1 edge; node 2 isolated
    dist, _ = Graphormer._compute_structural(edge_index, n=3)
    assert dist[0, 2] == Graphormer.NUM_SPATIAL - 1
    assert dist[2, 2] == 0  # self still 0


def test_graphormer_forward_backward(cigre_grid):
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    f = len(ORDERS) * 3  # H * trig channels
    b = 8
    model = build_model(
        GraphormerConfig(hidden_dim=32, num_layers=2, num_heads=4),
        n_features=f,
        n_nodes=n,
    )
    feats = torch.randn(b, n, f, requires_grad=True)
    mask = torch.rand(b, n) > 0.5
    out = model(build_model_input(g, feats, mask))
    assert out.shape == (b, 1, n, f)  # T=1 lifted
    out.pow(2).mean().backward()
    grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert float(grad) > 0
    assert feats.grad is not None and float(feats.grad.abs().sum()) > 0


def test_graphormer_learns_spectral_lappe(cigre_grid):
    from injection_hse.config import StructuralEncodingConfig
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    k = 6
    g = build_graph(
        cigre_grid,
        orders=ORDERS,
        encoding=StructuralEncodingConfig(kind="lappe", lap_pe_dim=k),
    )
    n, f = g.num_nodes, len(ORDERS) * 3
    model = build_model(
        GraphormerConfig(
            hidden_dim=32, num_layers=1, num_heads=4, spectral_pe_dim=k
        ),
        n_features=f,
        n_nodes=n,
        n_static_features=int(g.x.shape[1]),
    )
    feats = torch.randn(2, n, f, requires_grad=True)
    mask = torch.rand(2, n) > 0.5
    out = model(build_model_input(g, feats, mask))
    assert out.shape == (2, 1, n, f)
    out.square().mean().backward()
    spectral_grad = sum(
        p.grad.abs().sum()
        for p in model.spectral_pe.parameters()
        if p.grad is not None
    )
    assert float(spectral_grad) > 0.0


def test_graphormer_hidden_not_divisible_raises():
    from injection_hse.models import build_model

    with pytest.raises(ValueError):
        build_model(
            GraphormerConfig(hidden_dim=30, num_layers=1, num_heads=4),  # 30 % 4 != 0
            n_features=21,
            n_nodes=132,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_graphormer_device_honoring(cigre_grid):
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)  # built on CPU
    n = g.num_nodes
    f = len(ORDERS) * 3
    model = build_model(
        GraphormerConfig(hidden_dim=32, num_layers=2, num_heads=4),
        n_features=f,
        n_nodes=n,
    ).cuda()
    feats = torch.randn(4, n, f, device="cuda")
    mask = torch.ones(4, n, dtype=torch.bool, device="cuda")
    # graph + cached structural indices moved to CUDA inside forward
    out = model(build_model_input(g, feats, mask))
    assert out.device.type == "cuda"
    assert out.shape == (4, 1, n, f)


def test_graphormer_train_smoke(cigre_grid, se_dataset, tmp_path, monkeypatch):
    """A short masked Graphormer curriculum stage learns (val loss drops)."""
    import lightning as L

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
        name="graphormer-smoke",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=GraphormerConfig(hidden_dim=64, num_layers=2, num_heads=4, dropout=0.0),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,  # unreachable -> runs the full epoch budget
                    max_epochs=25,
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
    train_mod.train_experiment(cfg, grid=cigre_grid)

    assert len(history) >= 3, history
    assert all(torch.isfinite(torch.tensor(history)))
    assert history[-1] < history[0] * 0.9, history
    assert (tmp_path / "ckpt" / "stage_masked.ckpt").is_file()
