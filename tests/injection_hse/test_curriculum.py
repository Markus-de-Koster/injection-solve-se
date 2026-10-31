"""Multi-stage curriculum: StageManager early-stops on target + writes boundary checkpoints.

Uses the DNN baseline (no PyG dependency) so the curriculum mechanics are tested
independently of the GNN's PyG-conv availability.
"""

from __future__ import annotations

import pytest
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
    cigre_lv_curriculum,
    cigre_lv_measurement,
)
from injection_hse.train import TargetLossEarlyStop, _BestWeightKeeper


class _SingleDeviceStrategy:
    """Mirror of Lightning's single-device ``Strategy.reduce_boolean_decision`` no-op."""

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        return decision


ORDERS = [1, 3, 5, 7, 9, 11, 13]


def test_target_loss_early_stop_callback():
    # the default monitor is the all-rows pu-MAE (well-defined for a fully-observed stage,
    # unlike the unobserved-only metric which is vacuously 0 there)
    assert TargetLossEarlyStop(1e-2).monitor == "val_pu_mae"

    cb = TargetLossEarlyStop(1e-2, monitor="val_pu_mae_unobs")

    class _Trainer:
        def __init__(self, value):
            self.callback_metrics = {"val_pu_mae_unobs": torch.tensor(value)}
            self.should_stop = False
            self.strategy = _SingleDeviceStrategy()

    tr = _Trainer(0.05)
    cb.on_validation_epoch_end(tr, None)
    assert not tr.should_stop and not cb.stopped_on_target  # above target

    tr = _Trainer(0.005)
    cb.on_validation_epoch_end(tr, None)
    assert tr.should_stop and cb.stopped_on_target  # reached target
    assert cb.best == pytest.approx(0.005)


def test_best_weight_keeper_tracks_best_and_halts_on_nonfinite():
    """The keeper snapshots the lowest-monitor weights and stops the stage on a NaN loss."""

    class _Module(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            with torch.no_grad():
                self.lin.weight.fill_(w)

    class _Trainer:
        def __init__(self, metrics):
            self.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
            self.should_stop = False
            self.sanity_checking = False
            self.strategy = _SingleDeviceStrategy()

    keeper = _BestWeightKeeper()  # monitor="val_pu_mae"
    m = _Module(1.0)

    # epoch 1: pu-MAE 0.05 -> recorded as best
    keeper.on_validation_epoch_end(_Trainer({"val_pu_mae": 0.05, "val_loss": 0.1}), m)
    assert keeper.best == pytest.approx(0.05)
    snap_a = keeper.best_state["lin.weight"].clone()

    # epoch 2: BETTER pu-MAE 0.02 with NEW weights -> snapshot updates
    with torch.no_grad():
        m.lin.weight.fill_(2.0)
    keeper.on_validation_epoch_end(_Trainer({"val_pu_mae": 0.02, "val_loss": 0.05}), m)
    assert keeper.best == pytest.approx(0.02)
    assert torch.allclose(keeper.best_state["lin.weight"], torch.full((2, 2), 2.0))
    assert not torch.allclose(keeper.best_state["lin.weight"], snap_a)

    # epoch 3: WORSE pu-MAE -> best is unchanged (still the epoch-2 weights)
    keeper.on_validation_epoch_end(_Trainer({"val_pu_mae": 0.5, "val_loss": 0.5}), m)
    assert keeper.best == pytest.approx(0.02)
    assert torch.allclose(keeper.best_state["lin.weight"], torch.full((2, 2), 2.0))

    # epoch 4: NaN loss -> halt the stage; the best snapshot is NOT overwritten by NaN weights
    tr = _Trainer({"val_pu_mae": float("nan"), "val_loss": float("nan")})
    keeper.on_validation_epoch_end(tr, m)
    assert tr.should_stop and keeper.halted_nonfinite
    assert torch.allclose(keeper.best_state["lin.weight"], torch.full((2, 2), 2.0))


def test_stage_optim_overrides_flow_into_the_optimizer():
    """Per-stage MaskStageConfig lr/scheduler overrides reach configure_optimizers."""
    import types

    import yaml

    from injection_hse.data import build_graph
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule
    from tests.injection_hse.tiny_grids import single_phase_chain

    orders = [1, 3]
    grid = single_phase_chain()
    graph = build_graph(grid, orders=orders)
    cfg = ExperimentConfig(
        name="stage-optim",
        seed=0,
        orders=orders,
        data=DataConfig(source="parquet", dataset_path="unused"),
        model=DNNConfig(hidden_dims=[8]),
        measurement=MeasurementConfig(voltage_nodes=[], slack_known=True),
        optim=OptimConfig(name="adamw", lr=2e-4, scheduler="none"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="full", kind="full", target_loss=1.0, max_epochs=1
                ),
                MaskStageConfig(
                    name="placement",
                    kind="placement",
                    target_loss=1.0,
                    max_epochs=7,
                    lr=5e-5,
                    scheduler="cosine",
                ),
            ]
        ),
    )
    feats = torch.rand(8, graph.num_nodes, len(orders), 3)
    module = SEModule(
        cfg,
        n_nodes=graph.num_nodes,
        normalizer=HarmonicNormalizer().fit(feats),
        graph=graph,
    )

    # no override installed -> the global OptimConfig applies
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4)

    # the placement stage's overrides (as the StageManager installs them)
    stage = cfg.curriculum.stages[1]
    module.set_stage_optim(lr=stage.lr, scheduler=stage.scheduler)
    module._trainer = types.SimpleNamespace(max_epochs=stage.max_epochs)
    out = module.configure_optimizers()
    assert out["optimizer"].param_groups[0]["lr"] == pytest.approx(5e-5)
    sched = out["lr_scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 7  # anneals once over the STAGE's epoch budget

    # clearing (None values) restores inheritance for the next stage
    module.set_stage_optim(lr=None, scheduler=None)
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4)

    # the per-stage optimization is part of the serialized, reproducible config
    restored = ExperimentConfig.model_validate(
        yaml.safe_load(yaml.safe_dump(cfg.model_dump()))
    )
    assert restored.curriculum.stages[1].lr == pytest.approx(5e-5)
    assert restored.curriculum.stages[1].scheduler == "cosine"


def test_default_cigre_curriculum_ladder():
    cur = cigre_lv_curriculum()
    names = [s.name for s in cur.stages]
    assert names == ["full", "frac_50", "frac_25", "frac_10", "placement"]
    assert cur.stages[0].kind == "full"
    assert cur.stages[-1].kind == "placement"
    fracs = [s.observed_fraction for s in cur.stages]
    assert fracs == [None, 0.5, 0.25, 0.1, None]
    # targets ramp up monotonically (tighter under full obs, looser at placement)
    targets = [s.target_loss for s in cur.stages]
    assert targets == sorted(targets)
    assert cigre_lv_measurement().voltage_nodes == ["Bus R4", "Bus C5"]


def test_two_stage_curriculum_runs_and_checkpoints(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    import lightning as L

    from injection_hse import train as train_mod

    # a loose 2-stage curriculum: full (trivially hits target) -> random 0.5 (loose target)
    cfg = ExperimentConfig(
        name="two-stage",
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
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="full", kind="full", target_loss=1e-3, max_epochs=8
                ),
                MaskStageConfig(
                    name="frac_50",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1.0,  # very loose -> early-stops in 1 val epoch
                    max_epochs=8,
                ),
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

    # silence sanity val + force deterministic small run
    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    train_mod.train_experiment(cfg, grid=cigre_grid)

    # both stage-boundary checkpoints written, carrying normalizer + config
    for name in ("full", "frac_50"):
        ckpt = tmp_path / "ckpt" / f"stage_{name}.ckpt"
        assert ckpt.is_file(), name
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "normalizer" in payload and "pgl_config" in payload


def test_full_cigre_curriculum_smoke(cigre_grid, se_dataset, tmp_path, monkeypatch):
    """The full 5-stage ladder runs (relaxed targets) and writes the pre-placement ckpt."""
    import lightning as L

    from injection_hse import train as train_mod

    cfg = ExperimentConfig(
        name="full-ladder",
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
        measurement=cigre_lv_measurement(),
        curriculum=cigre_lv_curriculum(
            target_full=5e-3, target_placement=5e-2, max_epochs=3
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

    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    train_mod.train_experiment(cfg, grid=cigre_grid)

    ckpts = sorted(p.name for p in (tmp_path / "ckpt").glob("stage_*.ckpt"))
    assert ckpts == [
        "stage_frac_10.ckpt",
        "stage_frac_25.ckpt",
        "stage_frac_50.ckpt",
        "stage_full.ckpt",
        "stage_placement.ckpt",
    ]
    # the pre-placement checkpoint (frac_10) must exist (the finetune-safety guarantee)
    assert (tmp_path / "ckpt" / "stage_frac_10.ckpt").is_file()


def test_probe_step_memory_is_reproducible_and_readonly(cigre_grid, se_dataset):
    """probe_step_memory runs a step without consuming a mask draw or moving the weights."""
    import lightning as L

    from injection_hse.config import DataConfig, DNNConfig
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import TrigEncoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule, make_datamodule, probe_step_memory

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    orders = list(source.orders)
    tv = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    norm = HarmonicNormalizer().fit(TrigEncoding().encode(tv).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="probe",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="f",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    mod = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=norm,
        graph=source.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    dm = make_datamodule(source, batch_size=16)
    rng_before = mod._mask_gen.get_state().clone()
    w_before = [p.detach().clone() for p in mod.model.parameters()]

    peak = probe_step_memory(mod, dm, device="cpu")

    assert peak is None  # CPU reports no CUDA peak
    assert torch.equal(mod._mask_gen.get_state(), rng_before)  # no mask draw consumed
    assert all(
        torch.equal(a, b) for a, b in zip(w_before, mod.model.parameters())
    )  # the probe does not step the optimizer
