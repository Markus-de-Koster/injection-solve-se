"""StageManager adopts its own config onto the module, and the module records the epoch-mean
training loss.

Covers the two-task provenance fix (a sequence ``StageManager`` run must make the reused module
TRAIN with and CHECKPOINT its own config, not the one the module was built with) and the
epoch-mean training loss (the running per-epoch mean, not the last minibatch). Runs a tiny DNN
curriculum on the shared CIGRE LV fixtures.
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
    LossConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    TrainConfig,
)
from injection_hse.train import SEModule, StageManager, _to_nhc, make_datamodule

ORDERS = [1, 3, 5, 7, 9, 11, 13]


class _EpochMeanProbe(L.Callback):
    """Records, per epoch, the sample-weighted mean of the per-batch losses and compares it to
    the module's ``train_epoch_loss`` at validation time."""

    def __init__(self) -> None:
        self.batches: list[tuple[float, int]] = []
        self.checks: list[
            tuple[float, float, float]
        ] = []  # (manual_mean, property, last)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.batches = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.batches.append((float(loss), int(batch["v"].shape[0])))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False) or not self.batches:
            return
        num = sum(loss * bs for loss, bs in self.batches)
        den = sum(bs for _, bs in self.batches)
        self.checks.append(
            (num / den, float(pl_module.train_epoch_loss), self.batches[-1][0])
        )


def _cfg(name, dataset, *, val_fraction, injection_weight, ckpt_dir, mlruns):
    return ExperimentConfig(
        name=name,
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset),
            batch_size=64,
            val_fraction=val_fraction,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[64]),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        loss=LossConfig(injection_loss_weight=injection_weight),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,  # unreachable -> runs the epoch budget
                    max_epochs=2,
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


def test_stage_manager_adopts_cfg_and_records_epoch_mean(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse import train as train_mod

    # the module is BUILT with cfg_a (snapshot-like); a StageManager RUNS cfg_b (sequence-like).
    # cfg_b differs in name, val_fraction, and a runtime loss weight.
    ckpt_dir = str(tmp_path / "ckpt")
    mlruns = f"file:{tmp_path / 'mlruns'}"
    cfg_a = _cfg(
        "run-A",
        se_dataset,
        val_fraction=0.1,
        injection_weight=1.0,
        ckpt_dir=ckpt_dir,
        mlruns=mlruns,
    )
    cfg_b = _cfg(
        "run-B",
        se_dataset,
        val_fraction=0.3,
        injection_weight=5.0,
        ckpt_dir=ckpt_dir,
        mlruns=mlruns,
    )

    # deterministic small run (skip Lightning's sanity validation)
    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    source_a = build_data_source(cfg_a.data, grid=cigre_grid)
    enc = build_encoding(cfg_a.encoding.kind)
    tv = torch.stack([source_a.train_dataset()[i]["v"] for i in range(64)])
    from injection_hse.normalization import HarmonicNormalizer

    norm = HarmonicNormalizer().fit(_to_nhc(enc.encode(tv)))
    module = SEModule(
        cfg_a,
        n_nodes=source_a.n_nodes,
        normalizer=norm,
        graph=source_a.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
    )
    assert module.cfg is cfg_a

    source_b = build_data_source(cfg_b.data, grid=cigre_grid)
    dm_b = make_datamodule(source_b, batch_size=cfg_b.data.batch_size, num_workers=0)
    probe = _EpochMeanProbe()
    ckpts = StageManager(cfg_b).run(module, dm_b, logger=False, extra_callbacks=[probe])

    # -- task 5: the module TRAINS with and CHECKPOINTS the manager's cfg (cfg_b) --
    assert module.cfg is cfg_b
    payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert payload["pgl_config"]["name"] == "run-B"
    assert payload["pgl_config"]["data"]["val_fraction"] == pytest.approx(0.3)
    assert payload["pgl_config"]["loss"]["injection_loss_weight"] == pytest.approx(5.0)

    # -- task 3: the checkpoint carries the grid fingerprint --
    from pgml.topology import layout_fingerprint

    assert payload["grid_fingerprint"] == layout_fingerprint(cigre_grid)

    # -- task 6: the module's epoch loss is the sample-weighted MEAN, not the last minibatch --
    assert probe.checks
    for manual_mean, prop, last_minibatch in probe.checks:
        assert prop == pytest.approx(manual_mean, rel=1e-4, abs=1e-6)
    # with several batches per epoch the mean is genuinely NOT the last minibatch value
    assert any(
        abs(manual_mean - last_minibatch) > 1e-6
        for manual_mean, _, last_minibatch in probe.checks
    )
