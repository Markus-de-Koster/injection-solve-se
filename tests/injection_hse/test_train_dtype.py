"""Generation-vs-training precision split (``TrainConfig.dtype``).

Data generation defaults to ``complex128`` (a feeder's harmonic Y-bus is ill-conditioned, so
single precision loses the small harmonic voltages the labels carry), but the model is trained
at ``TrainConfig.dtype`` (``complex64`` by default). These two precisions are DECOUPLED, so:

- ``complex64`` data trained at ``complex64`` is BYTE-IDENTICAL to before (no cast occurs);
- ``complex128`` data trained at the default ``complex64`` (float32 model) trains without the
  old crash (float64 features hitting float32 GRU parameters);
- ``complex128`` data trained at ``complex128`` runs a full float64 model.

The physics decode always solves in ``complex128`` internally; only its inputs/outputs follow
the training dtype.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    PhysicsConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import ParquetDataset, StateCollate, base_voltage_per_row
from injection_hse.encoding import TrigEncoding
from injection_hse.io import build_model_input
from injection_hse.normalization import HarmonicNormalizer
from injection_hse.train import SEModule, _flatten_features, training_dtypes

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _seq_batch(dataset_path, grid, *, n=8, dtype=torch.complex64):
    """A collated SEQUENCE batch ``v [B, T, H, N]`` cast to ``dtype`` + a fitted normalizer.

    Casting the (stored-``complex64``) sequence batch to ``complex128`` SIMULATES a
    complex128-generated dataset without a heavy double-precision harmonic solve — the crash
    reproduces from the batch dtype alone (float64 features into a float32 GRU).
    """
    ds = ParquetDataset(str(dataset_path), grid=grid, val_fraction=0.25, seed=0)
    batch = StateCollate()([ds.train_dataset()[i] for i in range(n)])
    batch["v"] = batch["v"].to(dtype)
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(batch["v"]).transpose(-3, -2))
    return ds, batch, norm


def _temporal_cfg(train_dtype: str, *, decode: str = "voltage") -> ExperimentConfig:
    return ExperimentConfig(
        name="dtype",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=TemporalConfig(
            spatial=GNNConfig(hidden_dim=32, num_layers=1, conv="gine"),
            temporal="gru",
            hidden_dim=32,
        ),
        physics=PhysicsConfig(decode=decode),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
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
        train=TrainConfig(dtype=train_dtype),
    )


# =============================================================================
# Backward compatibility: complex64 data trained at complex64 is byte-identical
# =============================================================================
def test_complex64_default_no_cast_byte_identical(cigre_grid, se_sequence_dataset):
    """The default (complex64) path takes NO cast branch and keeps a float32 model.

    Every dtype-machinery statement is guarded on the training dtype DIFFERING from the
    complex64/float32 default; this asserts each guard is false, so the complex64 pipeline
    runs the identical instructions as before the split existed.
    """
    ds, batch, norm = _seq_batch(se_sequence_dataset, cigre_grid, dtype=torch.complex64)
    cfg = _temporal_cfg("complex64")
    assert cfg.train.dtype == "complex64"  # the shipped default
    mod = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    # the derived dtypes are the float32/complex64 default
    assert mod._real_dtype is torch.float32 and mod._complex_dtype is torch.complex64
    assert training_dtypes("complex64") == (torch.float32, torch.complex64)
    # model parameters are untouched (float32) — the cast guard was not taken
    assert all(p.dtype is torch.float32 for p in mod.model.parameters())
    # the boundary cast is a genuine no-op: ``.to`` returns the SAME tensor (no copy, no node)
    assert batch["v"].to(mod._complex_dtype) is batch["v"]

    loss, _, pred_v, _, _, _, _ = mod._step(batch, train=True)
    assert pred_v.dtype is torch.complex64 and loss.dtype is torch.float32
    assert torch.isfinite(loss)

    # same outputs: an explicitly complex64-configured module built with the same seed is
    # bitwise identical (params AND the forward), so the default reproduces "today" exactly.
    L.seed_everything(0)
    mod_a = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    L.seed_everything(0)
    mod_b = SEModule(
        _temporal_cfg("complex64"),
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    for pa, pb in zip(mod_a.model.parameters(), mod_b.model.parameters()):
        assert torch.equal(pa, pb)
    with torch.no_grad():
        oa = mod_a.infer(batch)["pred_v"]
        ob = mod_b.infer(batch)["pred_v"]
    assert torch.equal(oa, ob)


# =============================================================================
# The old crash case: complex128 data + a temporal (GRU) model + default training
# =============================================================================
def test_complex128_data_default_training_no_crash(cigre_grid, se_sequence_dataset):
    """complex128 labels + a GRU model + the default complex64 training now train.

    Before the split the float64 encoded features hit the float32 GRU parameters and raised
    ``input must have the type torch.float32, got type torch.float64``. The boundary cast
    (loaded state -> the training complex dtype) resolves it: the features become float32 and
    the float32 model consumes them.
    """
    ds, batch, norm = _seq_batch(
        se_sequence_dataset, cigre_grid, dtype=torch.complex128
    )
    assert batch["v"].dtype is torch.complex128  # a complex128 "dataset"
    cfg = _temporal_cfg("complex64")  # default: float32 model
    mod = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    assert all(p.dtype is torch.float32 for p in mod.model.parameters())

    # the raw (uncast) float64 features are exactly what used to crash the float32 GRU
    feats64 = _flatten_features(mod._encode_normalize(batch["v"]))
    assert feats64.dtype is torch.float64
    mask = torch.ones(*feats64.shape[:-1], dtype=torch.bool)
    # the GRU rejects float64 input against its float32 weights (the historical crash); the
    # backend surfaces it as a ValueError/RuntimeError naming the dtype mismatch.
    with pytest.raises((RuntimeError, ValueError), match="float"):
        mod.model(build_model_input(mod._graph, feats64, mask, has_time=True))

    # the full step casts at the boundary, so training runs and back-propagates
    loss, _, pred_v, _, _, _, _ = mod._step(batch, train=True)
    assert pred_v.dtype is torch.complex64  # decoded state follows the training dtype
    assert torch.isfinite(loss)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in mod.model.parameters() if p.grad is not None)
    assert float(grad) > 0


# =============================================================================
# Full-precision study: complex128 data + complex128 training (float64 params)
# =============================================================================
@pytest.mark.parametrize("decode", ["voltage", "injection"])
def test_complex128_training_float64_params(cigre_grid, se_sequence_dataset, decode):
    """complex128 training runs a float64 model; the injection decode stays consistent."""
    ds, batch, norm = _seq_batch(
        se_sequence_dataset, cigre_grid, dtype=torch.complex128
    )
    cfg = _temporal_cfg("complex128", decode=decode)
    assert cfg.train.dtype == "complex128"

    solver = inj_norm = None
    if decode == "injection":
        from injection_hse.physics import NetworkSolver, fit_injection_normalizer

        _, cdt = training_dtypes(cfg.train.dtype)
        solver = NetworkSolver.from_grid(cigre_grid, ORDERS, dtype=cdt)
        inj_norm = fit_injection_normalizer(solver, batch["v"], TrigEncoding())

    mod = SEModule(
        cfg,
        n_nodes=ds.n_nodes,
        normalizer=norm,
        graph=ds.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        solver=solver,
        inj_normalizer=inj_norm,
    )
    assert mod._real_dtype is torch.float64 and mod._complex_dtype is torch.complex128
    assert all(p.dtype is torch.float64 for p in mod.model.parameters())
    if decode == "injection":
        # the solver's DECODE dtype follows the training precision (internal solve is complex128)
        assert mod.solver._dtype is torch.complex128

    loss, _, pred_v, _, _, _, _ = mod._step(batch, train=True)
    assert pred_v.dtype is torch.complex128 and loss.dtype is torch.float64
    assert torch.isfinite(loss)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in mod.model.parameters() if p.grad is not None)
    assert float(grad) > 0


# =============================================================================
# End-to-end fit smoke through Lightning (double-precision plugin for complex128)
# =============================================================================
@pytest.fixture(scope="module")
def se_sequence_dataset_c128(cigre_grid, tmp_path_factory):
    """A tiny complex128 SEQUENCE dataset (the generation-at-double-precision case)."""
    from pgml.scenarios import (
        Selector,
        run_scenarios,
        write_dataset,
    )
    from injection_hse.data.scenarios import CoherentSpectrumConfig

    out = tmp_path_factory.mktemp("se_seq_c128")
    cfg = CoherentSpectrumConfig(
        name="coherent128",
        selector=Selector(component="load"),
        orders=[3, 5, 7],
        n_steps=4,
        n_scenarios=12,
        n_modes=2,
        seed=0,
    )
    result = run_scenarios(
        cigre_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


@pytest.mark.parametrize("train_dtype", ["complex64", "complex128"])
def test_complex128_dataset_temporal_fits(
    cigre_grid, se_sequence_dataset_c128, tmp_path, monkeypatch, train_dtype
):
    """A temporal (GRU) model fits a real complex128 dataset at BOTH training precisions.

    ``complex64`` is the headline case (double-precision labels, single-precision training);
    ``complex128`` exercises the Lightning double-precision plugin end to end (float64 params).
    Both must run a real ``StageManager`` fit without a dtype crash and leave a finite loss.
    """
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
        name=f"c128-{train_dtype}",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_sequence_dataset_c128),
            batch_size=16,
            val_fraction=0.25,
            num_workers=0,
        ),
        model=TemporalConfig(
            spatial=GNNConfig(hidden_dim=32, num_layers=1, conv="gine"),
            temporal="gru",
            hidden_dim=32,
        ),
        physics=PhysicsConfig(decode="voltage"),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,
                    max_epochs=3,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=2e-3, scheduler="none"),
        train=TrainConfig(
            dtype=train_dtype,
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    module = train_mod.train_experiment(cfg, grid=cigre_grid)

    real, complex_dtype = training_dtypes(train_dtype)
    assert all(p.dtype is real for p in module.model.parameters())
    assert len(history) >= 2, history
    assert all(torch.isfinite(torch.tensor(history))), history

    # the checkpoint round-trips (its config carries the training dtype -> a reload rebuilds
    # the model at the same precision before loading the weights)
    ckpt = tmp_path / "ckpt" / "stage_masked.ckpt"
    assert ckpt.is_file()
    reloaded = SEModule.from_checkpoint(ckpt, grid=cigre_grid)
    assert reloaded.cfg.train.dtype == train_dtype
    assert all(p.dtype is real for p in reloaded.model.parameters())
