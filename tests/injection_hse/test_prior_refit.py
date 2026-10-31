"""The linear injection prior follows the training pass that decodes with it.

A two-pass run (snapshots, then sequences) with the prior-residual decode fits the prior on
the snapshot pass's train split at build time and refits it on the sequence pass's train
split before that pass trains; every stage checkpoint carries the prior its pass used, and
:meth:`injection_hse.train.SEModule.from_checkpoint` restores it. A run without a prior fits none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.workflow import TrainingPass, TrainingRunConfig

ORDERS = [1, 3, 5, 7, 9, 11, 13]
SENSORS = ["Bus R4", "Bus C5"]


def _experiment(name, dataset, stage, tmp_path, *, residual, fit_states=4096):
    return ExperimentConfig(
        name=name,
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset),
            batch_size=32,
            val_fraction=0.25,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[32], dropout=0.0),
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        physics=PhysicsConfig(
            decode="injection",
            injection_prior_residual=residual,
            injection_prior_fit_states=fit_states,
        ),
        curriculum=CurriculumConfig(stages=[stage]),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )


def _stage(name, kind):
    # a target above any achievable loss ends the stage after its first validation epoch
    return MaskStageConfig(
        name=name,
        kind=kind,
        observed_fraction=0.5 if kind == "random" else None,
        target_loss=1e9,
        max_epochs=1,
    )


def _two_pass(tmp_path, snapshots, sequences, *, residual, seq_fit_states=4096):
    return TrainingRunConfig(
        name="prior-refit",
        passes=[
            TrainingPass(
                name="taskA",
                experiment=_experiment(
                    "prior-refit-A",
                    snapshots,
                    _stage("random_50", "random"),
                    tmp_path,
                    residual=residual,
                ),
            ),
            TrainingPass(
                name="taskB",
                experiment=_experiment(
                    "prior-refit-B",
                    sequences,
                    _stage("placement", "placement"),
                    tmp_path,
                    residual=residual,
                    fit_states=seq_fit_states,
                ),
            ),
        ],
    )


def test_prior_fit_states_flattens_sequence_steps(cigre_grid, se_sequence_dataset):
    """The fit sample is the first ``n`` train STATES, whole scenarios read in order."""
    from injection_hse.data import build_data_source
    from injection_hse.train import prior_fit_states

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_sequence_dataset)),
        grid=cigre_grid,
    )
    ds = source.train_dataset()
    steps = int(ds[0]["v"].shape[0])
    states = prior_fit_states(source, 2 * steps + 1)  # reaches into a third scenario
    assert states.shape == (2 * steps + 1, *ds[0]["v"].shape[-2:])
    assert torch.equal(states[:steps], ds[0]["v"])
    assert torch.equal(states[2 * steps], ds[2]["v"][0])
    everything = prior_fit_states(source, 10**6)
    assert everything.shape[0] == len(ds) * steps


def test_the_sequence_pass_refits_the_prior(
    tmp_path, cigre_grid, se_dataset, se_sequence_dataset
):
    from injection_hse.analysis import fit_linear_injection_prior
    from injection_hse.data import MeasurementModel
    from injection_hse.train import SEModule, prior_fit_states
    from injection_hse.workflow import build_training_setup, run_training

    cfg = _two_pass(
        tmp_path, se_dataset, se_sequence_dataset, residual=True, seq_fit_states=20
    )
    setup = build_training_setup(cfg, grid=cigre_grid)
    snap_prior = setup.module.injection_prior.prior
    assert setup.prior_pass == 0
    assert snap_prior.fit_dataset == f"{se_dataset}:train"
    n_snap_train = len(setup.sources[0].train_dataset())
    assert snap_prior.n_train == min(4096, n_snap_train)

    result = run_training(cfg, setup=setup, logger=False)
    seq_prior = result.module.injection_prior.prior
    assert seq_prior.fit_dataset == f"{se_sequence_dataset}:train"
    assert seq_prior.n_train == 20
    # the same map layout (placement, rows, orders), fitted on other data
    assert seq_prior.coef.shape == snap_prior.coef.shape
    np.testing.assert_array_equal(seq_prior.observed_rows, snap_prior.observed_rows)
    assert not np.allclose(seq_prior.coef, snap_prior.coef)

    # it IS the fit on the sequence pass's first 20 train states at the placement
    rows = MeasurementModel.resolve(
        cigre_grid, SENSORS, slack_known=True
    ).observed_rows(cigre_grid)
    states = prior_fit_states(result.setup.sources[1], 20)
    mask = torch.zeros((states.shape[0], states.shape[-1]), dtype=torch.bool)
    mask[:, rows] = True
    direct = fit_linear_injection_prior(result.module, states, mask)
    np.testing.assert_allclose(seq_prior.coef, direct.coef, rtol=1e-9, atol=1e-12)

    # every stage checkpoint carries the prior of its own pass; a reload restores it
    ckpts = {Path(c).name: Path(c) for c in result.checkpoints}
    snap_ckpt = torch.load(
        ckpts["stage_random_50.ckpt"], map_location="cpu", weights_only=False
    )
    assert snap_ckpt["injection_prior"]["fit_dataset"] == f"{se_dataset}:train"
    reloaded = SEModule.from_checkpoint(
        ckpts["stage_placement.ckpt"], grid=cigre_grid, device="cpu"
    )
    restored = reloaded.injection_prior.prior
    assert restored.fit_dataset == f"{se_sequence_dataset}:train"
    np.testing.assert_array_equal(restored.coef, seq_prior.coef)


def test_a_run_without_a_prior_fits_none(
    tmp_path, cigre_grid, se_dataset, se_sequence_dataset
):
    from injection_hse.workflow import build_training_setup, run_training

    cfg = _two_pass(tmp_path, se_dataset, se_sequence_dataset, residual=False)
    setup = build_training_setup(cfg, grid=cigre_grid)
    assert setup.prior_pass is None and setup.module.injection_prior is None
    result = run_training(cfg, setup=setup, logger=False)
    assert result.module.injection_prior is None
    for path in result.checkpoints:
        assert "injection_prior" not in torch.load(
            path, map_location="cpu", weights_only=False
        )


def test_the_fit_size_is_a_config_field():
    from injection_hse.config import MULTIGRID_IGNORED

    assert PhysicsConfig().injection_prior_fit_states == 4096
    assert "physics.injection_prior_fit_states" in MULTIGRID_IGNORED
    with pytest.raises(ValueError):
        PhysicsConfig(injection_prior_fit_states=0)
