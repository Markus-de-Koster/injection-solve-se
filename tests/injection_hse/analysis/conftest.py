"""Fixtures for the injection_hse.analysis (model-interpretability) tests.

The same tiny-but-real pattern as the injection_hse.evaluation tests: the full 3-phase
CIGRE LV benchmark grid is built once per session, and a controlled synthetic
:class:`~injection_hse.evaluation.Prediction` with known THD / unbalance / deviation
feeds the physical-feature tests without any training. The matplotlib Agg
backend is forced (no display).
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

ORDERS = [1, 3, 5]
SENSORS = ["Bus R4", "Bus C5", "Bus I2"]

# Known relative harmonic magnitudes of the controlled state (fractions of the
# fundamental): THD = sqrt(0.04^2 + 0.03^2) = 5 % on every energized row.
H3_FRACTION = 0.04
H5_FRACTION = 0.03


@pytest.fixture(scope="session")
def analysis_grid():
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
def analysis_dataset(analysis_grid, tmp_path_factory):
    """A tiny randomized parquet SE dataset (snapshot ``[B, H, N]``) -> directory path."""
    import torch
    from injection_hse.data.scenarios import se_benchmark_scenario_config
    from pgml.scenarios import (
        run_scenarios,
        write_dataset,
    )

    out = tmp_path_factory.mktemp("analysis_ds")
    cfg = se_benchmark_scenario_config(analysis_grid, n_samples=24, seed=0)
    result = run_scenarios(
        analysis_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


@pytest.fixture(scope="session")
def trained_run(analysis_grid, analysis_dataset, tmp_path_factory):
    """Train a tiny 2-stage curriculum -> (checkpoint_path, dataset_path, run_dir)."""
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

    out = tmp_path_factory.mktemp("analysis_run")
    cfg = ExperimentConfig(
        name="analysis-test",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(analysis_dataset),
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
    train_experiment(cfg, grid=analysis_grid)
    ckpt = out / "checkpoints" / "stage_placement.ckpt"
    return str(ckpt), str(analysis_dataset), str(out)


@pytest.fixture(scope="session")
def trained_current_run(analysis_grid, analysis_dataset, tmp_path_factory):
    """A tiny current-measuring, measurement-anchored INJECTION-decode run.

    Covers the measured-current input seam (edge features in attribution /
    occlusion) and the anchored decode path of the estimator-level analyses.
    """
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
    from injection_hse.train import train_experiment

    out = tmp_path_factory.mktemp("analysis_current_run")
    cfg = ExperimentConfig(
        name="analysis-current-test",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(analysis_dataset),
            batch_size=8,
            val_fraction=0.3,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=32, num_layers=2)),
        measurement=MeasurementConfig(
            voltage_nodes=SENSORS, slack_known=True, measure_current=True
        ),
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, anchor_current_weight=1.0
        ),
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
    train_experiment(cfg, grid=analysis_grid)
    ckpt = out / "checkpoints" / "stage_placement.ckpt"
    return str(ckpt), str(analysis_dataset), str(out)


@pytest.fixture(scope="session")
def trained_current_module(trained_current_run, analysis_grid):
    """The current-measuring anchored module rebuilt from its checkpoint."""
    from injection_hse.evaluation import load_se_module

    ckpt, dataset, _ = trained_current_run
    return load_se_module(ckpt, grid=analysis_grid, dataset_path=dataset)


def _write_run_dir(trained, grid):
    """Turn a ``(checkpoint, dataset, out_dir)`` triple into a FULL run directory.

    Writes the ``manifest.json`` + ``grid.json`` every ``injection_hse.analysis`` run-dir consumer
    reads: the two-stage curriculum of the smoke runs, the sensor placement, the orders and
    both dataset keys pointed at the one smoke dataset.
    """
    import json
    from pathlib import Path

    ckpt, dataset, out = trained
    out = Path(out)
    manifest = {
        "stages": {
            "0": {
                "task": "A",
                "stage_name": "random_50",
                "kind": "random",
                "observed_fraction": 0.5,
            },
            "1": {
                "task": "B",
                "stage_name": "placement",
                "kind": "placement",
                "observed_fraction": None,
            },
        },
        "switch_stage_index": 1,
        "sensor_names": SENSORS,
        "orders": ORDERS,
        "datasets": {"dataset_random": dataset, "dataset_coherent": dataset},
        "final_checkpoint": ckpt,
        "grid": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "grid.json").write_text(grid.model_dump_json())
    return out


@pytest.fixture(scope="session")
def run_dir(trained_run, analysis_grid):
    """The smoke run as a FULL run directory (manifest.json + grid.json added)."""
    out = _write_run_dir(trained_run, analysis_grid)
    import polars as pl

    pl.DataFrame(
        {
            "global_epoch": [0, 1],
            "stage_index": [0, 1],
            "train_loss": [1.0, 0.5],
            "val_loss": [1.1, 0.6],
            "val_pu_mae": [0.4, 0.2],
            "val_pu_mae_unobs": [0.5, 0.3],
        }
    ).write_parquet(out / "training_history.parquet")
    return out


@pytest.fixture(scope="session")
def current_run_dir(trained_current_run, analysis_grid):
    """The current-measuring INJECTION-decode run as a full run directory."""
    return _write_run_dir(trained_current_run, analysis_grid)


@pytest.fixture(scope="session")
def linear_prior_inputs(trained_current_run, analysis_grid):
    """The ``(v, mask)`` TRAIN pair the linear injection prior is fitted on."""
    from injection_hse.analysis import attribution_batch
    from injection_hse.config import MeasurementConfig

    _, dataset, _ = trained_current_run
    return attribution_batch(
        dataset,
        grid=analysis_grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        split="train",
        n_samples=16,
        seed=3,
    )


@pytest.fixture(scope="session")
def trained_module(trained_run, analysis_grid):
    """The trained module rebuilt from the smoke checkpoint (CPU, eval mode)."""
    from injection_hse.evaluation import load_se_module

    ckpt, dataset, _ = trained_run
    return load_se_module(ckpt, grid=analysis_grid, dataset_path=dataset)


@pytest.fixture(scope="session")
def attribution_inputs(trained_run, analysis_grid):
    """The standard ``(v, mask)`` attribution pair on the smoke dataset (8 samples)."""
    from injection_hse.analysis import attribution_batch
    from injection_hse.config import MeasurementConfig

    _, dataset, _ = trained_run
    return attribution_batch(
        dataset,
        grid=analysis_grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS),
        split="train",
        n_samples=8,
        seed=0,
    )


@pytest.fixture(scope="session")
def balanced_prediction(analysis_grid):
    """A controlled :class:`Prediction` with a perfectly balanced ground truth.

    The target state is an exact positive-sequence fundamental at the per-row
    base magnitude (phase angles 0 / −120° / +120°), plus 3rd and 5th harmonics
    at fixed fractions of the fundamental — so THD is exactly 5 %, the
    voltage-unbalance factor is 0, and the fundamental voltage deviation is 0.
    Neutral rows are de-energized. The estimate adds seeded noise so error
    statistics are non-trivial.
    """
    from pgml.assembly import node_phase_index

    from injection_hse.data import MeasurementModel, base_voltage_per_row
    from injection_hse.evaluation import Prediction

    idx = node_phase_index(analysis_grid)
    n = int(idx.size)
    h = len(ORDERS)
    s = 24
    base_v = base_voltage_per_row(analysis_grid).numpy().astype(np.float64)
    phase_codes = idx.phase_codes.numpy().astype(np.int64)

    angle = np.zeros(n)
    angle[phase_codes == 1] = -2.0 * np.pi / 3.0
    angle[phase_codes == 2] = 2.0 * np.pi / 3.0
    fund = (base_v * np.exp(1j * angle)).astype(np.complex128)
    fund[phase_codes == 3] = 0.0

    target = np.zeros((s, h, n), dtype=np.complex128)
    target[:, ORDERS.index(1), :] = fund
    target[:, ORDERS.index(3), :] = H3_FRACTION * fund
    target[:, ORDERS.index(5), :] = H5_FRACTION * fund

    rng = np.random.default_rng(0)
    noise = rng.standard_normal((s, h, n)) + 1j * rng.standard_normal((s, h, n))
    pred = target + 0.01 * base_v[None, None, :] * noise

    observed_rows = (
        MeasurementModel(SENSORS).observed_rows(analysis_grid).numpy().astype(np.int64)
    )
    mask = np.zeros((s, n), dtype=bool)
    mask[:, observed_rows] = True
    return Prediction(
        target_v=target,
        pred_v=pred,
        mask=mask,
        orders=list(ORDERS),
        base_v=base_v,
        node_ids=idx.node_ids.numpy().astype(np.int64),
        phase_codes=phase_codes,
        observed_rows=observed_rows,
        sensor_names=list(SENSORS),
        grid=analysis_grid,
    )


@pytest.fixture(scope="session")
def analysis_sequence_dataset(analysis_grid, tmp_path_factory):
    """A tiny coherent SEQUENCE dataset (``[B, T, H, N]``, 16 scenarios x 4 steps)."""
    import torch
    from pgml.scenarios import Selector, write_dataset

    from injection_hse.data.scenarios import CoherentSpectrumConfig, run_scenarios

    out = tmp_path_factory.mktemp("analysis_seq_ds")
    cfg = CoherentSpectrumConfig(
        name="coherent",
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=4,
        n_scenarios=16,
        n_modes=2,
        seed=0,
    )
    result = run_scenarios(
        analysis_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


@pytest.fixture(scope="session")
def two_task_run_dir(
    analysis_grid, analysis_dataset, analysis_sequence_dataset, tmp_path_factory
):
    """A tiny TWO-TASK prior-residual run (snapshots, then sequences) as a run directory.

    Both passes carve a TEST split (``test_fraction=0.25``); the final model is the
    sequence pass's placement stage, measurement-anchored, with the prior-residual decode
    (so its checkpoint carries the sequence pass's linear prior).
    """
    import json

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
    from injection_hse.workflow import TrainingPass, TrainingRunConfig, run_training

    out = tmp_path_factory.mktemp("analysis_two_task_run")

    def experiment(name, dataset, stage):
        return ExperimentConfig(
            name=name,
            seed=0,
            orders=ORDERS,
            data=DataConfig(
                source="parquet",
                dataset_path=str(dataset),
                batch_size=8,
                val_fraction=0.25,
                test_fraction=0.25,
                num_workers=0,
            ),
            model=TemporalConfig(spatial=GNNConfig(hidden_dim=32, num_layers=2)),
            measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
            physics=PhysicsConfig(
                decode="injection",
                anchor_voltage_weight=1.0,
                injection_prior_residual=True,
            ),
            curriculum=CurriculumConfig(stages=[stage]),
            optim=OptimConfig(name="adamw", lr=2e-3, scheduler="none"),
            train=TrainConfig(
                precision="32",
                accelerator="cpu",
                devices=1,
                mlflow_tracking_uri=f"file:{out / 'mlruns'}",
                checkpoint_dir=str(out / "checkpoints"),
            ),
        )

    stages = {
        "random_50": MaskStageConfig(
            name="random_50",
            kind="random",
            observed_fraction=0.5,
            target_loss=1e-9,
            max_epochs=1,
        ),
        "placement": MaskStageConfig(
            name="placement", kind="placement", target_loss=1e-9, max_epochs=1
        ),
    }
    cfg = TrainingRunConfig(
        name="analysis-two-task",
        passes=[
            TrainingPass(
                name="taskA",
                experiment=experiment(
                    "two-task-A", analysis_dataset, stages["random_50"]
                ),
            ),
            TrainingPass(
                name="taskB",
                experiment=experiment(
                    "two-task-B", analysis_sequence_dataset, stages["placement"]
                ),
            ),
        ],
    )
    result = run_training(cfg, grid=analysis_grid, logger=False)
    manifest = {
        "stages": {
            "0": {
                "task": "A",
                "stage_name": "random_50",
                "kind": "random",
                "observed_fraction": 0.5,
            },
            "1": {
                "task": "B",
                "stage_name": "placement",
                "kind": "placement",
                "observed_fraction": None,
            },
        },
        "switch_stage_index": 1,
        "sensor_names": SENSORS,
        "orders": ORDERS,
        "datasets": {
            "dataset_random": str(analysis_dataset),
            "dataset_coherent": str(analysis_sequence_dataset),
        },
        "final_checkpoint": str(result.checkpoints[-1]),
        "grid": {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "grid.json").write_text(analysis_grid.model_dump_json())
    return out
