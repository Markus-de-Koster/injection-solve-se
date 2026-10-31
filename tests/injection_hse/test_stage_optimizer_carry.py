"""Stage warmup and the optimizer state carried across stages (``carry_optimizer_state``)."""

from __future__ import annotations

import types

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
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _tiny_module(stages):
    from injection_hse.data import build_graph
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule
    from tests.injection_hse.tiny_grids import single_phase_chain

    orders = [1, 3]
    grid = single_phase_chain()
    graph = build_graph(grid, orders=orders)
    cfg = ExperimentConfig(
        name="warmup",
        seed=0,
        orders=orders,
        data=DataConfig(source="parquet", dataset_path="unused"),
        model=DNNConfig(hidden_dims=[8]),
        measurement=MeasurementConfig(voltage_nodes=[], slack_known=True),
        optim=OptimConfig(name="adamw", lr=2e-4, scheduler="none"),
        curriculum=CurriculumConfig(stages=stages),
    )
    feats = torch.rand(8, graph.num_nodes, len(orders), 3)
    return SEModule(
        cfg,
        n_nodes=graph.num_nodes,
        normalizer=HarmonicNormalizer().fit(feats),
        graph=graph,
    )


def _lrs(out, epochs):
    opt, sched = out["optimizer"], out["lr_scheduler"]
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_warmup_ramps_linearly_then_hands_over_to_the_stage_schedule():
    stage = MaskStageConfig(
        name="p",
        kind="placement",
        target_loss=1.0,
        max_epochs=10,
        lr=4e-4,
        scheduler="cosine",
        warmup_epochs=3,
    )
    module = _tiny_module([stage])
    module.set_stage_optim(lr=stage.lr, scheduler=stage.scheduler, warmup_epochs=3)
    module._trainer = types.SimpleNamespace(max_epochs=stage.max_epochs)
    lrs = _lrs(module.configure_optimizers(), 10)
    # lr/4, lr/2, 3lr/4, then the cosine from lr over the remaining 7 epochs
    assert lrs[:4] == pytest.approx([1e-4, 2e-4, 3e-4, 4e-4])
    assert all(a > b for a, b in zip(lrs[3:], lrs[4:]))
    assert lrs[-1] < 0.1 * 4e-4

    # constant rate after the warmup without a scheduler
    module.set_stage_optim(lr=4e-4, scheduler="none", warmup_epochs=2)
    lrs = _lrs(module.configure_optimizers(), 6)
    assert lrs == pytest.approx([4e-4 / 3, 8e-4 / 3, 4e-4, 4e-4, 4e-4, 4e-4])

    # no warmup: the previous behaviour (the bare optimizer)
    module.set_stage_optim(lr=4e-4, scheduler="none")
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)


def test_warmup_is_validated():
    with pytest.raises(ValueError, match="plateau"):
        MaskStageConfig(
            name="p",
            target_loss=1.0,
            max_epochs=5,
            scheduler="plateau",
            warmup_epochs=1,
        )
    with pytest.raises(ValueError, match="below max_epochs"):
        MaskStageConfig(name="p", target_loss=1.0, max_epochs=3, warmup_epochs=3)


def test_optimizer_state_loader_keeps_the_new_rate():
    from injection_hse.train.stages import _OptimizerStateLoader

    w = torch.nn.Parameter(torch.ones(3))
    old = torch.optim.AdamW([w], lr=1e-3)
    w.grad = torch.ones(3)
    old.step()
    state = old.state_dict()

    w2 = torch.nn.Parameter(torch.ones(3))
    new = torch.optim.AdamW([w2], lr=5e-5)
    new.param_groups[0]["initial_lr"] = 2e-4
    trainer = types.SimpleNamespace(optimizers=[new])
    _OptimizerStateLoader(state).on_train_start(trainer, None)
    assert new.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert new.param_groups[0]["initial_lr"] == pytest.approx(2e-4)
    assert torch.equal(new.state[w2]["exp_avg"], old.state[w]["exp_avg"])
    assert int(new.state[w2]["step"]) == 1


def test_stage_manager_carries_the_optimizer_state(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    import lightning as L

    from injection_hse import train as train_mod

    def cfg_with(carry):
        return ExperimentConfig(
            name="carry",
            seed=0,
            orders=ORDERS,
            data=DataConfig(
                source="parquet",
                dataset_path=str(se_dataset),
                batch_size=64,
                val_fraction=0.2,
                num_workers=0,
            ),
            model=DNNConfig(hidden_dims=[32]),
            measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="a", kind="full", target_loss=1e-12, max_epochs=1
                    ),
                    MaskStageConfig(
                        name="b",
                        kind="full",
                        target_loss=1e-12,
                        max_epochs=1,
                        warmup_epochs=0,
                    ),
                ]
            ),
            optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
            train=TrainConfig(
                precision="32",
                accelerator="cpu",
                devices=1,
                carry_optimizer_state=carry,
                mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
                checkpoint_dir=str(tmp_path / f"ckpt{int(carry)}"),
            ),
        )

    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)

    steps = {}
    for carry in (False, True):
        cfg = cfg_with(carry)
        train_mod.train_experiment(cfg, grid=cigre_grid)
        payload = torch.load(
            tmp_path / f"ckpt{int(carry)}" / "stage_b.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        state = payload["optimizer_states"][0]["state"]
        steps[carry] = int(next(iter(state.values()))["step"])
    # the second stage continues the first stage's step count only when carried
    assert steps[True] == 2 * steps[False]
