"""Fixtures for the injection_hse.evaluation (model-evaluation plot) tests.

A tiny three-phase CIGRE LV grid + a small parquet dataset are built ONCE per session, and a
short two-stage curriculum is trained to produce a real stage checkpoint — so the inference
round-trip (``load_se_module`` / ``predict_dataset``) and the plot functions run end-to-end on
genuine artifacts without re-solving per test. The matplotlib Agg backend is forced (no display).
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

ORDERS = [1, 3, 5]
SENSORS = ["Bus R4", "Bus C5", "Bus I2"]


@pytest.fixture(scope="session")
def eval_grid():
    """Full 3-phase CIGRE LV grid with Carson geometry + PV (the SE benchmark grid)."""
    from pgml.grids import add_pv_systems
    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.oracles import cigre_lv_full_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    warnings.filterwarnings("ignore")
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    synthesize_grid_geometry(grid)
    add_pv_systems(grid, fraction=0.5)
    return grid


@pytest.fixture(scope="session")
def eval_dataset(eval_grid, tmp_path_factory):
    """A tiny randomized parquet SE dataset (snapshot ``[B, H, N]``) -> directory path."""
    from injection_hse.data.scenarios import se_benchmark_scenario_config
    from pgml.scenarios import (
        run_scenarios,
        write_dataset,
    )

    out = tmp_path_factory.mktemp("eval_ds")
    cfg = se_benchmark_scenario_config(eval_grid, n_samples=24, seed=0)
    result = run_scenarios(
        eval_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


@pytest.fixture(scope="session")
def trained_run(eval_grid, eval_dataset, tmp_path_factory):
    """Train a 2-stage curriculum -> (checkpoint_path, dataset_path) for the round-trip tests."""
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        ExperimentConfig,
        GNNConfig,
        MaskStageConfig,
        MeasurementConfig,
        OptimConfig,
        TemporalConfig,
        TrainConfig,
    )
    from injection_hse.train import train_experiment

    out = tmp_path_factory.mktemp("eval_run")
    cfg = ExperimentConfig(
        name="eval-test",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(eval_dataset),
            batch_size=8,
            val_fraction=0.3,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=32, num_layers=2)),
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random_50",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                ),
                MaskStageConfig(
                    name="placement", kind="placement", target_loss=1e-9, max_epochs=1
                ),
            ]
        ),
        optim=OptimConfig(name="adamw", lr=2e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{out / 'mlruns'}",
            checkpoint_dir=str(out / "checkpoints"),
        ),
    )
    train_experiment(cfg, grid=eval_grid)
    ckpt = out / "checkpoints" / "stage_placement.ckpt"
    return str(ckpt), str(eval_dataset)


@pytest.fixture
def synthetic_prediction(eval_grid):
    """A small synthetic :class:`Prediction` (random pred/target/mask) for the pure-plot tests."""
    from pgml.assembly import node_phase_index

    from injection_hse.data import MeasurementModel, base_voltage_per_row
    from injection_hse.evaluation.inference import Prediction

    idx = node_phase_index(eval_grid)
    n = int(idx.size)
    h = len(ORDERS)
    s = 6
    rng = np.random.default_rng(0)
    target = (
        rng.standard_normal((s, h, n)) + 1j * rng.standard_normal((s, h, n))
    ).astype(np.complex64)
    pred = target + 0.1 * (
        rng.standard_normal((s, h, n)) + 1j * rng.standard_normal((s, h, n))
    ).astype(np.complex64)
    observed_rows = (
        MeasurementModel(SENSORS).observed_rows(eval_grid).numpy().astype(np.int64)
    )
    mask = np.zeros((s, n), dtype=bool)
    mask[:, observed_rows] = True
    return Prediction(
        target_v=target,
        pred_v=pred,
        mask=mask,
        orders=list(ORDERS),
        base_v=base_voltage_per_row(eval_grid).numpy().astype(np.float64),
        node_ids=idx.node_ids.numpy().astype(np.int64),
        phase_codes=idx.phase_codes.numpy().astype(np.int64),
        observed_rows=observed_rows,
        sensor_names=list(SENSORS),
        grid=eval_grid,
    )
