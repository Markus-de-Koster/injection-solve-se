"""The training-run contract: the curriculum ladders and the executable run config.

Two guarantees are tested here.

1. **The promoted ladder builders reproduce the curriculum the drivers ran.** The expected
   stage lists are the ones the single-grid workflow and the multi-grid driver produced
   before the ladder moved into :mod:`injection_hse.curriculum` — masking fractions, the geometric
   target ramp across the COMBINED ladder, the per-stage epoch budgets and optimizer
   overrides, and the anchoring schedule stamped on top.
2. **A dumped :class:`injection_hse.workflow.TrainingRunConfig` is executable.** The run directory's
   ``training_run.yaml`` round-trips through YAML and trains again through
   ``python -m injection_hse.train --config`` — "one config + seed reproduces a run" for training.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

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
from injection_hse.curriculum import (
    anneal_stage,
    apply_anchor_schedule,
    parse_anchor_schedule,
    parse_orders,
    snapshot_stages,
    sequence_stages,
    profiled_sequence_stages,
)
from injection_hse.workflow import TrainingPass, TrainingRunConfig

ORDERS = [1, 3, 5, 7, 9, 11, 13]

# The multi-grid driver's ladder: the same shape under its own stage names.
MG_SNAPSHOT_LADDER = (
    ("full", "full", None),
    ("frac_50", "random", 0.5),
    ("frac_25", "random", 0.25),
)
MG_SEQUENCE_LADDER = (
    ("frac_10", "random", 0.10),
    ("placement", "placement", None),
)


def _shape(stages):
    """The fields a curriculum ladder is defined by (the rest of the stage is defaulted)."""
    return [
        (
            s.name,
            s.kind,
            s.observed_fraction,
            s.fundamental_fraction,
            pytest.approx(s.target_loss),
            s.max_epochs,
            s.lr,
            s.scheduler,
        )
        for s in stages
    ]


def test_single_grid_ladder_matches_the_recorded_curriculum():
    """The snapshot/B/C ladders + the anneal stage, for a fixed set of workflow arguments."""
    common = dict(target_full=1e-3, target_placement=1e-2)
    stages_a = snapshot_stages(epochs=7, fundamental_fraction=0.25, **common)
    stages_b = sequence_stages(
        epochs=7,
        fundamental_fraction=0.25,
        placement_lr=3e-4,
        placement_scheduler="cosine",
        **common,
    )
    stages_c = profiled_sequence_stages(
        epochs=7, target_loss=1e-2, lr=3e-4, scheduler="cosine"
    )
    anneal = anneal_stage(epochs=4, lr=3e-4)

    assert _shape(stages_a) == [
        ("full", "full", None, None, 1e-3, 7, None, None),
        ("random_50", "random", 0.5, 0.25, 0.0017782794100389228, 7, None, None),
        ("random_25", "random", 0.25, 0.25, 0.0031622776601683794, 7, None, None),
    ]
    assert _shape(stages_b) == [
        ("random_10", "random", 0.10, 0.25, 0.005623413251903491, 7, None, None),
        ("placement", "placement", None, None, 1e-2, 7, 3e-4, "cosine"),
    ]
    assert _shape(stages_c) == [
        ("profile_placement", "placement", None, None, 1e-2, 7, 3e-4, "cosine")
    ]
    # the anneal runs its full budget: a negligible target, cosine-annealed once
    assert _shape([anneal]) == [
        ("placement_anneal", "placement", None, None, 1e-9, 4, 3e-4, "cosine")
    ]
    # the ramp is one geometric interpolation across BOTH passes' rungs
    targets = [s.target_loss for s in (*stages_a, *stages_b)]
    ratios = [b / a for a, b in zip(targets, targets[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_multigrid_ladder_matches_the_recorded_curriculum():
    """The multi-grid arm: its own stage names, per-pass epochs, and the anchor schedule."""
    common = dict(
        target_full=1e-3,
        target_placement=1e-2,
        ladder_length=len(MG_SNAPSHOT_LADDER) + len(MG_SEQUENCE_LADDER),
    )
    stages = [
        *snapshot_stages(epochs=3, ladder=MG_SNAPSHOT_LADDER, **common),
        *sequence_stages(
            epochs=9,
            ladder=MG_SEQUENCE_LADDER,
            offset=len(MG_SNAPSHOT_LADDER),
            **common,
        ),
    ]
    assert _shape(stages) == [
        ("full", "full", None, None, 1e-3, 3, None, None),
        ("frac_50", "random", 0.5, None, 0.0017782794100389228, 3, None, None),
        ("frac_25", "random", 0.25, None, 0.0031622776601683794, 3, None, None),
        ("frac_10", "random", 0.10, None, 0.005623413251903491, 9, None, None),
        ("placement", "placement", None, None, 1e-2, 9, None, None),
    ]

    schedule = parse_anchor_schedule(
        "full:0,frac_50:0,frac_25:0~2,frac_10:0~10,placement:0~20"
    )
    anchored, unmatched = apply_anchor_schedule(stages, schedule)
    assert not unmatched
    weights = [(s.anchor_voltage_weight, s.anchor_voltage_range) for s in anchored]
    assert weights == [
        (0.0, None),
        (0.0, None),
        (1.0, (0.0, 2.0)),
        (5.0, (0.0, 10.0)),
        (10.0, (0.0, 20.0)),
    ]


def test_parse_orders_rejects_a_spectrum_without_harmonics():
    assert parse_orders("1,3,5,3") == [1, 3, 5]
    assert parse_orders(None) == list(range(1, 21))
    with pytest.raises(SystemExit, match="fundamental"):
        parse_orders("3,5")
    with pytest.raises(SystemExit, match="above 1"):
        parse_orders("1")


# =============================================================================
# the run config
# =============================================================================
def _experiment(name, dataset, stages, ckpt_dir, mlruns, **kwargs):
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
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
        curriculum=CurriculumConfig(stages=stages),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=str(mlruns),
            checkpoint_dir=str(ckpt_dir),
        ),
        **kwargs,
    )


def _stage(name, kind="random", *, fraction=0.5, target=1e9):
    # a target above any achievable loss ends the stage after its first validation epoch
    return MaskStageConfig(
        name=name,
        kind=kind,
        observed_fraction=fraction if kind == "random" else None,
        target_loss=target,
        max_epochs=1,
    )


def _two_pass_config(tmp_path, se_dataset, se_sequence_dataset, grid):
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(grid.model_dump_json())
    ckpt, mlruns = tmp_path / "checkpoints", f"file:{tmp_path / 'mlruns'}"
    return TrainingRunConfig(
        name="round-trip",
        grid_path=str(grid_path),
        passes=[
            TrainingPass(
                name="taskA",
                experiment=_experiment(
                    "round-trip-A", se_dataset, [_stage("random_50")], ckpt, mlruns
                ),
            ),
            TrainingPass(
                name="taskB",
                experiment=_experiment(
                    "round-trip-B",
                    se_sequence_dataset,
                    [_stage("placement", "placement")],
                    ckpt,
                    mlruns,
                ),
            ),
        ],
    )


def test_run_config_rejects_passes_that_cannot_share_one_module():
    exp = _experiment("a", "d", [_stage("s")], "c", "file:m")
    other = exp.model_copy(update={"seed": 1})
    with pytest.raises(ValueError, match="different seeds"):
        TrainingRunConfig(
            passes=[
                TrainingPass(name="A", experiment=exp),
                TrainingPass(name="B", experiment=other),
            ]
        )
    with pytest.raises(ValueError, match="different harmonic orders"):
        TrainingRunConfig(
            passes=[
                TrainingPass(name="A", experiment=exp),
                TrainingPass(
                    name="B", experiment=exp.model_copy(update={"orders": [1, 3]})
                ),
            ]
        )
    with pytest.raises(ValueError, match="unique"):
        TrainingRunConfig(
            passes=[
                TrainingPass(name="A", experiment=exp),
                TrainingPass(name="A", experiment=exp),
            ]
        )
    # the resume pointer must agree with the (already trimmed) ladders
    run = TrainingRunConfig(passes=[TrainingPass(name="A", experiment=exp)])
    with pytest.raises(ValueError, match="not the first stage"):
        run.model_copy(update={"start_stage": "elsewhere"}).model_validate(
            run.model_dump() | {"start_stage": "elsewhere"}
        )
    assert (
        TrainingRunConfig.model_validate(
            run.model_dump() | {"start_stage": "s"}
        ).start_stage
        == "s"
    )


def test_run_config_round_trips_and_wraps_a_bare_experiment(tmp_path):
    exp = _experiment("solo", "d", [_stage("s")], "c", "file:m")
    run = TrainingRunConfig.from_experiment(exp)
    assert [p.name for p in run.passes] == ["train"]
    assert run.seed == exp.seed and run.orders == exp.orders

    path = run.write(tmp_path / "training_run.yaml")
    assert TrainingRunConfig.load(path) == run

    bare = tmp_path / "experiment.yaml"
    bare.write_text(yaml.safe_dump(exp.model_dump(mode="json")))
    wrapped = TrainingRunConfig.load(bare)
    assert wrapped.passes[0].experiment == exp

    moved = run.with_output_dir(tmp_path / "elsewhere")
    train = moved.passes[0].experiment.train
    assert train.checkpoint_dir == str(tmp_path / "elsewhere" / "checkpoints")
    assert train.mlflow_tracking_uri.endswith("elsewhere/mlruns")
    assert "solo" in moved.identity() and "round" not in moved.identity()


def test_dumped_run_config_trains_through_the_train_cli(
    tmp_path, cigre_grid, se_dataset, se_sequence_dataset
):
    """dump -> ``python -m injection_hse.train --config`` -> both passes train into the run dir."""
    from injection_hse.train import main

    cfg = _two_pass_config(tmp_path, se_dataset, se_sequence_dataset, cigre_grid)
    path = cfg.write(tmp_path / "training_run.yaml")

    out = tmp_path / "rerun"
    main(["--config", str(path), "--out", str(out), "--device", "cpu"])

    # both passes ran, in order, into the relocated run directory
    assert sorted(p.name for p in (out / "checkpoints").glob("*.ckpt")) == [
        "stage_placement.ckpt",
        "stage_random_50.ckpt",
    ]
    # the resolved config is dumped beside them: the run dir stays re-runnable
    replayed = TrainingRunConfig.load(out / "training_run.yaml")
    assert [p.name for p in replayed.passes] == ["taskA", "taskB"]
    assert replayed.passes[0].experiment.train.checkpoint_dir == str(
        out / "checkpoints"
    )
    assert replayed.grid_path == cfg.grid_path


def test_train_module_cli_prints_the_contract():
    """``python -m injection_hse.train`` is wired: the schema and an example config are printable."""
    example = subprocess.run(
        [sys.executable, "-m", "injection_hse.train", "--example"],
        capture_output=True,
        text=True,
        check=True,
    )
    run = TrainingRunConfig.model_validate(yaml.safe_load(example.stdout))
    assert len(run.passes) == 2
    schema = subprocess.run(
        [sys.executable, "-m", "injection_hse.train", "--json-schema"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"passes"' in schema.stdout
