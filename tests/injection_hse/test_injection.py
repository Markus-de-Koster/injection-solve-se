"""Per-device injection / localization head: target assembly, head forward, byte-identity.

The injection target/input is per-device, per-injected-order MAGNITUDE ``[*B, T, D, O, C]``
(C=1). It is assembled from the dataset's ``samples`` (the sampled per-device h_mag spectra),
testable on the existing snapshot dataset. The head is OFF by default — node-V SE must be
byte-identical when disabled.
"""

from __future__ import annotations

import lightning as L
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    LossConfig,
    MaskStageConfig,
    OptimConfig,
    TrainConfig,
)
from injection_hse.data import (
    InjectionSpec,
    ParquetDataset,
    base_voltage_per_row,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.models.injection import InjectionHead
from injection_hse.normalization import HarmonicNormalizer
from injection_hse.train import SEModule

ORDERS = [1, 3, 5, 7, 9, 11, 13]


# -- target assembly ---------------------------------------------------------
def test_injection_spec_devices_and_orders(cigre_grid, se_dataset):
    from injection_hse.data.scenarios import read_dataset

    loaded = read_dataset(str(se_dataset))
    spec = InjectionSpec(cigre_grid, loaded.config)
    # 15 loads + 8 PV = 23 injecting devices, all emitting at the benchmark's orders
    assert spec.n_devices == 23
    assert spec.orders == [3, 5, 7, 9, 11, 13]
    assert not spec.is_empty
    # device→rows: each 3-phase device has 3 valid node-phase rows
    assert spec.device_rows.shape == (23, 3)
    assert spec.device_rows_mask.shape == (23, 3)
    assert spec.device_rows_mask.all()  # all devices are 3-phase here


def test_injection_target_known_values(cigre_grid, se_dataset):
    from injection_hse.data import _reconstruct_sampled
    from injection_hse.data.scenarios import read_dataset

    loaded = read_dataset(str(se_dataset))
    spec = InjectionSpec(cigre_grid, loaded.config)
    tgt = spec.assemble_target()  # [B, D, O, 1] from the REALIZED harmonic injection
    assert tgt.shape == (256, 23, 6, 1)
    assert tgt.dtype == torch.float32

    # the target equals the realized per-device per-order injection MAGNITUDE (the physical
    # injected current the head should recover) — NOT the raw U(0,1) spectrum fraction.
    hi = _reconstruct_sampled(cigre_grid, loaded.config).harmonic_injection
    # every device (15 loads then 8 PV inverters) matches its realized magnitude exactly
    for j in range(spec.n_devices):
        dev = spec.device_ids[j]
        for k, o in enumerate(spec.orders):
            assert torch.allclose(
                tgt[:, j, k, 0], hi[dev][o][0].to(torch.float32), atol=1e-5
            )
    # the PV inverters carry their OWN per-order shape (h5-dominant), which is what
    # separates them from a co-located load whose fundamental cancels against theirs
    pv = tgt[:, 15:, :, 0]
    assert float(pv.amax()) > 0.0
    assert float(pv[:, :, 1].mean()) > float(pv[:, :, 0].mean())  # h5 above h3


def test_injection_target_on_coherent_sequence(cigre_grid, se_sequence_dataset):
    """The localization target works on a COHERENT [B,T,H,N] dataset (T preserved)."""
    from injection_hse.data.scenarios import read_dataset

    loaded = read_dataset(str(se_sequence_dataset))
    b, t = loaded.v.shape[0], loaded.v.shape[1]
    spec = InjectionSpec(cigre_grid, loaded.config)
    assert not spec.is_empty
    tgt = spec.assemble_target()  # [B, T, D, O, 1]
    assert tgt.shape == (b, t, spec.n_devices, spec.n_orders, 1)
    assert tgt.dtype == torch.float32
    # the dataset ships a per-(scenario, step) target keeping the time axis
    ds = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, val_fraction=0.25, seed=0
    )
    item = ds.train_dataset()[0]
    assert item["injection_target"].shape == (t, spec.n_devices, spec.n_orders, 1)


def test_dataset_exposes_injection_target(cigre_grid, se_dataset):
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2, seed=0)
    assert ds.injection_spec is not None and not ds.injection_spec.is_empty
    item = ds.train_dataset()[0]
    assert "injection_target" in item
    assert item["injection_target"].shape == (23, 6, 1)


# -- head forward / grad -----------------------------------------------------
def test_injection_head_forward_backward(cigre_grid, se_dataset):
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2, seed=0)
    spec = ds.injection_spec
    n, f, b = ds.n_nodes, len(ORDERS) * 3, 8
    head = InjectionHead(
        node_feature_dim=f, n_orders=spec.n_orders, hidden_dim=64, n_channels=1
    )
    node_feats = torch.randn(b, 1, n, f, requires_grad=True)  # [B, T=1, N, F]
    out = head(node_feats, spec.device_rows, spec.device_rows_mask)
    assert out.shape == (b, 1, spec.n_devices, spec.n_orders, 1)
    assert (out >= 0).all()  # softplus -> non-negative magnitudes
    out.pow(2).mean().backward()
    grad = sum(p.grad.abs().sum() for p in head.parameters() if p.grad is not None)
    assert float(grad) > 0
    assert node_feats.grad is not None and float(node_feats.grad.abs().sum()) > 0


# -- byte-identity when disabled ---------------------------------------------
def _module(cigre_grid, ds, *, predict: bool):
    v = torch.stack([ds.train_dataset()[i]["v"] for i in range(64)])
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(v).transpose(-3, -2))
    L.seed_everything(0, workers=True)
    cfg = ExperimentConfig(
        name="x",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[64, 64]),
        loss=LossConfig(predict_injection=predict),
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
    return SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        injection_spec=ds.injection_spec,
    )


def test_state_estimation_byte_identical_when_head_off(cigre_grid, se_dataset):
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2, seed=0)
    off = _module(cigre_grid, ds, predict=False)
    on = _module(cigre_grid, ds, predict=True)
    assert off.injection_head is None and on.injection_head is not None
    # the state model parameters are byte-identical (the head is built after the model)
    assert all(
        torch.equal(a, b) for a, b in zip(off.model.parameters(), on.model.parameters())
    )
    # the node-V SE output is exactly identical (an aligned batch via the dataset + collate)
    from injection_hse.data import StateCollate

    batch = StateCollate()([ds.train_dataset()[i] for i in range(64)])
    off.eval()
    on.eval()
    with torch.no_grad():
        _, _, pv_off, _, il_off, _, _ = off._step(batch, train=False)
        _, _, pv_on, _, il_on, _, _ = on._step(batch, train=False)
    assert il_off is None and il_on is not None
    assert torch.equal(pv_off, pv_on)


# -- train smoke with the head ENABLED --------------------------------------
def test_train_with_injection_head_loss_decreases(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    from injection_hse import train as train_mod

    history: list[float] = []

    class Recorder(L.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_injection_loss")
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
        name="injection-smoke",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[128, 128]),
        loss=LossConfig(
            predict_injection=True, injection_loss_weight=1.0, injection_hidden_dim=64
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
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    module = train_mod.train_experiment(cfg, grid=cigre_grid)
    assert module.injection_head is not None
    assert len(history) >= 3, history
    assert all(torch.isfinite(torch.tensor(history)))
    assert history[-1] < history[0] * 0.9, history
