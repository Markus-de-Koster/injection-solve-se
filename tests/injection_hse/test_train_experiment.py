"""End-to-end self-check: a masked DNN stage learns (val loss drops).

Builds a minimal :class:`ExperimentConfig` (one ``random`` masking stage + ``DNNConfig`` with
a tight target so it does not early-stop), runs ``train_experiment`` for a handful of epochs
on CPU over a tiny generated dataset, and asserts the validation loss DECREASES and the
per-harmonic state errors are finite. This is the vertical-slice gate (data -> encode ->
normalize -> mask -> DNN -> loss -> Lightning). A masked stage is used (not full
observability) because under full observability the unobserved set is empty, so the unobserved
pu-MAE early-stop fires immediately.
"""

from __future__ import annotations

import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    TrainConfig,
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _experiment_config(
    dataset_path: str, ckpt_dir: str, mlruns: str
) -> ExperimentConfig:
    return ExperimentConfig(
        name="vertical-slice",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=dataset_path,
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[128, 128], dropout=0.0),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,  # unreachable -> runs the full epoch budget
                    max_epochs=12,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=mlruns,
            checkpoint_dir=ckpt_dir,
        ),
    )


def test_train_experiment_val_loss_decreases(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    import lightning as L

    from injection_hse import train as train_mod

    history: list[float] = []

    class Recorder(L.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_loss")
            if v is not None:
                history.append(float(v))

    # inject the recorder into every Trainer the StageManager builds
    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        cbs = list(kwargs.get("callbacks") or [])
        cbs.append(Recorder())
        kwargs["callbacks"] = cbs
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    cfg = _experiment_config(
        str(se_dataset),
        ckpt_dir=str(tmp_path / "ckpt"),
        mlruns=f"file:{tmp_path / 'mlruns'}",
    )
    module = train_mod.train_experiment(cfg, grid=cigre_grid)

    # validation loss recorded across epochs and trends DOWN
    assert len(history) >= 3, history
    assert all(torch.isfinite(torch.tensor(history)))
    # the last epoch is clearly below the first (the surrogate learns)
    assert history[-1] < history[0] * 0.9, history

    # a forced stage-boundary checkpoint exists and carries the normalizer + config
    ckpt = tmp_path / "ckpt" / "stage_masked.ckpt"
    assert ckpt.is_file()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "normalizer" in payload and "pgl_config" in payload

    # per-harmonic pu + physical state errors are finite on the validation set
    from injection_hse.metrics import state_errors

    from injection_hse.io import build_model_input

    from injection_hse.data import build_data_source

    val_ds = build_data_source(cfg.data, grid=cigre_grid).val_dataset()
    v = torch.stack([val_ds[i]["v"] for i in range(len(val_ds))])  # [B, H, N]
    module.eval()
    with torch.no_grad():
        feats = module._encode_normalize(v)
        mask = torch.ones(v.shape[0], module.n_nodes, dtype=torch.bool)
        flat = train_mod._flatten_features(feats) * mask.unsqueeze(-1)
        pred = module.model(build_model_input(module._graph, flat, mask))[:, 0]
        pred_feats = train_mod._unflatten_features(
            pred, module.n_harmonics, module.n_channels
        )
        pred_v = module.encoding.decode(
            module.normalizer.inverse_transform(pred_feats)
        ).transpose(1, 2)
    errs = state_errors(pred_v, v, mask=mask, orders=ORDERS, base_v=module.base_v)
    for key in ("mae_obs", "rmse_obs", "mae_obs_phys"):
        assert torch.isfinite(torch.tensor(errs[key]))
    for order in ORDERS:
        assert torch.isfinite(torch.tensor(errs[order]["mae_obs"]))


def test_training_step_skips_nonfinite_loss(cigre_grid, monkeypatch):
    """A batch whose loss is NaN is skipped (no optimizer step, weights survive);
    a persistent stream of non-finite batches aborts instead of training on."""
    import pytest

    from injection_hse import train as train_mod
    from injection_hse.data import base_voltage_per_row, build_graph
    from injection_hse.encoding import TrigEncoding
    from injection_hse.normalization import HarmonicNormalizer

    cfg = _experiment_config("unused", ckpt_dir="unused", mlruns="file:unused")
    g = build_graph(cigre_grid)
    v = (
        torch.randn(2, len(ORDERS), g.num_nodes)
        + 1j * torch.randn(2, len(ORDERS), g.num_nodes)
    ).to(torch.complex64)
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(v).transpose(-3, -2))
    module = train_mod.SEModule(
        cfg,
        n_nodes=g.num_nodes,
        normalizer=norm,
        graph=g,
        base_v=base_voltage_per_row(cigre_grid),
    )

    nan_step = (torch.tensor(float("nan")), None, None, None, None, None, None)
    monkeypatch.setattr(module, "_step", lambda batch, train=True: nan_step)
    batch = {"v": v}
    with pytest.warns(UserWarning, match="non-finite training loss"):
        surrogate = module.training_step(batch, 0)
    # DDP-safe skip: a zero-valued, graph-connected scalar (never None — Lightning
    # forbids skipping via None under distributed training)
    assert surrogate is not None and surrogate.requires_grad
    assert float(surrogate) == 0.0
    surrogate.backward()
    assert all(p.grad is None or bool((p.grad == 0).all()) for p in module.parameters())
    assert module._nonfinite_batches == 1

    module._nonfinite_batches = 50
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="persistent"):
        module.training_step(batch, 1)


def test_on_after_backward_zeroes_nonfinite_grads(cigre_grid):
    """A NaN gradient (finite loss, broken backward) is zeroed before the
    optimizer step; the shared skip counter increments and eventually aborts."""
    import pytest

    from injection_hse import train as train_mod
    from injection_hse.data import base_voltage_per_row, build_graph
    from injection_hse.encoding import TrigEncoding
    from injection_hse.normalization import HarmonicNormalizer

    cfg = _experiment_config("unused", ckpt_dir="unused", mlruns="file:unused")
    g = build_graph(cigre_grid)
    v = (
        torch.randn(2, len(ORDERS), g.num_nodes)
        + 1j * torch.randn(2, len(ORDERS), g.num_nodes)
    ).to(torch.complex64)
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(v).transpose(-3, -2))
    module = train_mod.SEModule(
        cfg,
        n_nodes=g.num_nodes,
        normalizer=norm,
        graph=g,
        base_v=base_voltage_per_row(cigre_grid),
    )
    params = [p for p in module.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.zeros_like(p)
    params[0].grad[...] = float("nan")
    with pytest.warns(UserWarning, match="non-finite gradients"):
        module.on_after_backward()
    assert module._nonfinite_batches == 1
    assert all(bool((p.grad == 0).all()) for p in params)

    params[0].grad[...] = float("inf")
    module._nonfinite_batches = 50
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="persistent"):
        module.on_after_backward()
