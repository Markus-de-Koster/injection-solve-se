"""End-to-end CPU smoke runs of the two-task workflow driver with its generation options.

Each test runs ``run/examples/injection_hse/train_se_full_workflow.py`` at the smallest sizes that
exercise every stage (generation, validation, both passes, the checkpoint) in a subprocess,
then checks what the run directory records and that the trained module reloads with the
settings the run was configured with.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
_DRIVER = _REPO / "run" / "examples" / "injection_hse" / "train_se_full_workflow.py"

#: The smallest run that still trains both passes on CIGRE LV (about a minute on CPU).
SMOKE = [
    "--orders",
    "1,3,5",
    "--n-random",
    "12",
    "--n-scenarios",
    "6",
    "--n-steps",
    "2",
    "--epochs",
    "1",
    "--hidden-dim",
    "8",
    "--gnn-layers",
    "1",
    "--device",
    "cpu",
    "--skip-coverage",
    "--no-progress",
]


def _run(out_dir: Path, *flags: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(_DRIVER), str(out_dir), *SMOKE, *flags],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


def test_load_shunt_flags_set_data_and_decode_and_are_recorded(tmp_path):
    from pgml.schemas.grid_schema import Grid

    from injection_hse.evaluation import load_se_module

    out = tmp_path / "run"
    log = _run(
        out,
        "--load-shunt",
        "opendss",
        "--load-shunt-basis",
        "nameplate",
        "--skip-eval",
    )
    assert (
        "harmonic load shunt: data opendss/nameplate, decode opendss/nameplate" in log
    )
    assert "differ by the per-scenario operating-point" not in log

    expected = {
        "data": {"model": "opendss", "basis": "nameplate"},
        "decode": {"model": "opendss", "basis": "nameplate"},
    }
    for name in ("dataset_random", "dataset_coherent"):
        meta = json.loads((out / name / "meta.json").read_text())
        assert meta["extra_provenance"]["harmonic_load_shunt"] == expected
    assert yaml.safe_load((out / "data_generation.yaml").read_text()) == {
        "harmonic_load_shunt": expected
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["data_generation"]["harmonic_load_shunt"] == expected
    for name in ("experiment_snapshots.yaml", "experiment_sequences.yaml"):
        cfg = yaml.safe_load((out / name).read_text())
        assert cfg["physics"]["load_shunt"] == "opendss"

    # the trained decode reloads with the configured shunt
    grid = Grid.model_validate_json((out / "grid.json").read_text())
    module = load_se_module(
        manifest["final_checkpoint"],
        grid=grid,
        dataset_path=str(out / "dataset_coherent"),
    )
    assert module.solver.load_shunt == "opendss"
    assert torch.isfinite(module.solver.y_harmonic).all()


def test_the_default_run_records_no_generation_settings(tmp_path):
    out = tmp_path / "run"
    log = _run(out, "--skip-eval")
    assert "harmonic load shunt" not in log
    assert not (out / "data_generation.yaml").exists()
    assert "data_generation" not in json.loads((out / "manifest.json").read_text())
    meta = json.loads((out / "dataset_random" / "meta.json").read_text())
    assert meta["extra_provenance"] == {}
    cfg = yaml.safe_load((out / "experiment_snapshots.yaml").read_text())
    assert cfg["physics"]["load_shunt"] is None


def _perturbation_yaml(path: Path, n_copies: int, n_test_copies: int) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "grid_perturbation": {
                    "n_copies": n_copies,
                    "n_test_copies": n_test_copies,
                    "seed": 3,
                    "line_r": {"shared": 0.05, "element": 0.1},
                    "line_length": {"element": 0.05},
                    "source_impedance": {"shared": 0.3},
                }
            }
        )
    )
    return path


def test_grid_perturbation_trains_on_copies_and_scores_held_out_ones(tmp_path):
    """Regime (ii): an ensemble of perturbed copies in training, unseen copies in test."""
    import numpy as np
    from pgml.schemas.grid_schema import Grid

    from injection_hse.data.scenarios import GRID_COPY_RECORD, read_dataset
    from injection_hse.evaluation import load_se_module
    from injection_hse.physics import NetworkSolver

    out = tmp_path / "run"
    config = _perturbation_yaml(tmp_path / "perturbation.yaml", 2, 2)
    log = _run(
        out,
        "--grid-perturbation",
        str(config),
        "--perturb-train",
        "perturbed",
        "--perturb-test",
        "heldout",
        "--perturb-test-scenarios",
        "4",
        "--test-fraction",
        "0.34",
    )
    assert "grid perturbation (train data, lognormal): 2 copies" in log
    assert "PERTURBED TEST dataset (coherent, 4 scenarios on 2 test-pool copies)" in log

    # every training sample records its copy, the factors are in meta.json
    for name, n in (("dataset_random", 12), ("dataset_coherent", 6)):
        loaded = read_dataset(out / name)
        assert loaded.samples[GRID_COPY_RECORD].tolist() == [0] * (n // 2) + [1] * (
            n // 2
        )
        record = json.loads((out / name / "meta.json").read_text())["extra_provenance"]
        copies = record["grid_perturbation"]["copies"]
        assert [c["pool"] for c in copies] == ["train", "train"]
        assert set(copies[0]["factors"]) == {
            "line_r",
            "line_length",
            "source_impedance",
        }
    test_dir = out / "dataset_coherent_perturbed_test"
    test_record = json.loads((test_dir / "meta.json").read_text())["extra_provenance"]
    test_factors = test_record["grid_perturbation"]["copies"]
    assert [c["pool"] for c in test_factors] == ["test", "test"]
    assert test_factors[0]["factors"] != copies[0]["factors"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["datasets"]["dataset_perturbed_test"] == str(test_dir.resolve())
    generation = yaml.safe_load((out / "data_generation.yaml").read_text())
    assert generation["grid_perturbation"]["train"] == "perturbed"
    assert len(generation["grid_perturbation"]["test_copies"]) == 2

    # the estimator's grid and decode stay nominal
    grid = Grid.model_validate_json((out / "grid.json").read_text())
    module = load_se_module(
        manifest["final_checkpoint"], grid=grid, dataset_path=str(test_dir)
    )
    nominal = NetworkSolver.from_grid(grid, [1, 3, 5])
    assert torch.equal(module.solver.y_harmonic.cpu(), nominal.y_harmonic)

    # per-copy metrics: both copies and the pooled block, per order and overall
    for name in ("metrics_perturbed_test_per_copy.csv", "metrics_test_per_copy.csv"):
        lines = (out / "evaluation" / name).read_text().strip().splitlines()
        header = lines[0].split(",")
        assert header[:2] == ["pool", "copy"] and "pu_mae_unobs" in header
        rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
        assert {r["order"] for r in rows} == {"all", "1", "3", "5"}
        assert all(np.isfinite(float(r["pu_mae_unobs"])) for r in rows)
    held_out = (out / "evaluation" / "metrics_perturbed_test_per_copy.csv").read_text()
    assert held_out.count("\ntest,0,") == 4 and held_out.count("\nall,all,") == 4


def test_a_systematic_perturbation_is_shared_by_training_and_test(tmp_path):
    """Regime (iii): one copy (K = 1) for training and the perturbed test set."""
    out = tmp_path / "run"
    config = _perturbation_yaml(tmp_path / "perturbation.yaml", 1, 1)
    _run(
        out,
        "--grid-perturbation",
        str(config),
        "--perturb-train",
        "perturbed",
        "--perturb-test",
        "train",
        "--perturb-test-scenarios",
        "2",
        "--skip-eval",
    )
    train = json.loads((out / "dataset_coherent" / "meta.json").read_text())
    test = json.loads(
        (out / "dataset_coherent_perturbed_test" / "meta.json").read_text()
    )
    train_copy = train["extra_provenance"]["grid_perturbation"]["copies"]
    test_copy = test["extra_provenance"]["grid_perturbation"]["copies"]
    assert len(train_copy) == len(test_copy) == 1
    assert test_copy[0]["factors"] == train_copy[0]["factors"]
    assert test_copy[0]["pool"] == "train"


def test_perturbation_flags_without_a_regime_are_rejected(tmp_path):
    config = _perturbation_yaml(tmp_path / "perturbation.yaml", 2, 2)
    for flags, message in (
        (["--grid-perturbation", str(config)], "perturbs nothing"),
        (["--perturb-test", "heldout"], "need --grid-perturbation"),
        (
            ["--grid-perturbation", str(config), "--perturb-test", "train"],
            "needs --perturb-train perturbed",
        ),
    ):
        proc = subprocess.run(
            [sys.executable, str(_DRIVER), str(tmp_path / "r"), *SMOKE, *flags],
            capture_output=True,
            text=True,
            cwd=_REPO,
        )
        assert proc.returncode != 0 and message in proc.stderr, proc.stderr[-2000:]


@pytest.mark.parametrize("basis", ["nameplate", "operating_point"])
def test_recipe_v4_runs_on_both_load_shunt_bases(tmp_path, basis):
    """Recipe v4: device capacitance, TB 575 source and the v4 scenario file, both arms."""
    import hashlib

    from pgml.schemas.grid_schema import Grid, ShuntAppliance

    from injection_hse.evaluation import load_se_module
    from injection_hse.physics import NetworkSolver

    v4 = _REPO / "run" / "configs" / "pgl_se_scenario_v4.yaml"
    out = tmp_path / "run"
    log = _run(
        out,
        "--scenario-config",
        str(v4),
        "--device-capacitance",
        str(_REPO / "run" / "configs" / "pgl_device_capacitance.yaml"),
        "--source-impedance",
        "cigre-tb575",
        "--load-shunt",
        "opendss",
        "--load-shunt-basis",
        basis,
        "--skip-eval",
    )
    assert "device capacitance: 23 shunts" in log
    assert ("differ by the per-scenario operating-point" in log) == (
        basis == "operating_point"
    )
    grid = Grid.model_validate_json((out / "grid.json").read_text())
    assert sum(isinstance(a, ShuntAppliance) for a in grid.appliances) == 23
    sha = hashlib.sha256(v4.read_bytes()).hexdigest()
    for name in ("dataset_random", "dataset_coherent"):
        meta = json.loads((out / name / "meta.json").read_text())["extra_provenance"]
        assert meta["scenario_config_file"]["sha256"] == sha
        assert meta["harmonic_load_shunt"]["data"]["basis"] == basis
        model = meta["grid_model"]
        assert model["source_impedance"]["preset"] == "cigre-tb575"
        assert len(model["device_capacitance"]["shunts"]) == 23
    generation = yaml.safe_load((out / "data_generation.yaml").read_text())
    assert set(generation) == {
        "harmonic_load_shunt",
        "grid_model",
        "scenario_config_file",
    }

    # the estimator decodes with the capacitance-carrying grid
    manifest = json.loads((out / "manifest.json").read_text())
    module = load_se_module(
        manifest["final_checkpoint"],
        grid=grid,
        dataset_path=str(out / "dataset_coherent"),
    )
    expected = NetworkSolver.from_grid(grid, [1, 3, 5], load_shunt="opendss")
    assert torch.equal(module.solver.y_harmonic.cpu(), expected.y_harmonic)
    bare = grid.model_copy(
        update={
            "appliances": [
                a for a in grid.appliances if not isinstance(a, ShuntAppliance)
            ]
        }
    )
    without = NetworkSolver.from_grid(bare, [1, 3, 5], load_shunt="opendss")
    assert not torch.allclose(expected.y_harmonic, without.y_harmonic)


def test_class_a_noise_trains_and_reloads_with_the_instrument_model(tmp_path):
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import InstrumentErrorConfig
    from injection_hse.data import InstrumentNoise
    from injection_hse.evaluation import load_se_module

    out = tmp_path / "run"
    _run(
        out,
        "--noise-class",
        "A",
        "--measure-currents",
        "--current-channels",
        "upstream",
        "--skip-eval",
    )
    for name in ("experiment_snapshots.yaml", "experiment_sequences.yaml"):
        cfg = yaml.safe_load((out / name).read_text())
        assert cfg["noise"]["enabled"] is True
        assert (
            InstrumentErrorConfig.model_validate(cfg["noise"]["instrument"])
            == InstrumentErrorConfig.class_a()
        )
    manifest = json.loads((out / "manifest.json").read_text())
    grid = Grid.model_validate_json((out / "grid.json").read_text())
    module = load_se_module(
        manifest["final_checkpoint"],
        grid=grid,
        dataset_path=str(out / "dataset_coherent"),
    )
    # every voltage row and CT channel is read by the class-A instrument
    assert isinstance(module._noise, InstrumentNoise)
    assert module._noise.quantity == "voltage"
    assert isinstance(module._edge_noise, InstrumentNoise)
    assert module._edge_noise.quantity == "current"


def test_carried_optimizer_state_and_stage_warmup_reach_every_stage(tmp_path):
    out = tmp_path / "run"
    _run(
        out,
        "--no-curriculum",
        "--anneal-epochs",
        "2",
        "--epochs",
        "2",
        "--carry-optimizer-state",
        "--stage-warmup",
        "1",
    )
    snap = yaml.safe_load((out / "experiment_snapshots.yaml").read_text())
    seq = yaml.safe_load((out / "experiment_sequences.yaml").read_text())
    for cfg in (snap, seq):
        assert cfg["train"]["carry_optimizer_state"] is True
    # the run's first stage starts plain, every later stage warms up
    assert snap["curriculum"]["stages"][0]["warmup_epochs"] == 0
    assert [s["warmup_epochs"] for s in seq["curriculum"]["stages"]] == [1, 1]
    # the optimizer step count continues through the stage borders: the anneal stage's
    # optimizer has taken more steps than the anneal stage itself
    payload = torch.load(
        out / "checkpoints" / "stage_placement_anneal.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    state = payload["optimizer_states"][0]["state"]
    assert int(next(iter(state.values()))["step"]) > int(payload["global_step"])
