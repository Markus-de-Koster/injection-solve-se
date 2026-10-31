"""evaluate_run writes the full plot set from a manifest + checkpoint; StageManager plumbing."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")


def test_evaluate_run_writes_full_set(trained_run, eval_grid, tmp_path):
    ckpt, dataset = trained_run
    # a manifest pointing at the trained checkpoint + the dataset (the random split here)
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
        "sensor_names": ["Bus R4", "Bus C5", "Bus I2"],
        "orders": [1, 3, 5],
        "datasets": {"dataset_coherent": dataset, "dataset_random": dataset},
        "final_checkpoint": ckpt,
        "grid": {"n_nodes": len(eval_grid.nodes), "n_rows": 0},
    }
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(manifest))
    # the run persists the grid so evaluation is self-contained (no `grid=` needed)
    (out / "grid.json").write_text(eval_grid.model_dump_json())

    from injection_hse.evaluation import evaluate_run

    # grid omitted on purpose: it must be auto-loaded from grid.json
    eval_dir = evaluate_run(out, dataset="coherent", device="cpu")
    names = {p.name for p in eval_dir.iterdir()}
    # the four insight plots + per-phase heatmaps + at least one state plot
    for expect in (
        "error_by_order.svg",
        "error_cdf.svg",
        "error_vs_distance.svg",
        "parity.svg",
        "error_heatmap_mean_a.svg",
        "error_heatmap_mean_a.html",
        "error_heatmap_max_a.svg",
        "state_h1.svg",
        "state_3d.html",
    ):
        assert expect in names, f"missing {expect} in {names}"


def test_evaluate_run_reports_test_split(eval_grid, eval_dataset, tmp_path):
    """evaluate_run additionally writes the held-out TEST split metrics when the run carved one.

    A run trained with ``DataConfig.test_fraction > 0`` gets a ``metrics_test.json`` +
    ``metrics_test_per_order.csv`` alongside the validation ``metrics.json`` — clearly separated,
    never touching the test split during training. A run without a test split writes only the
    validation metrics (no ``metrics_test.json``).
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
        TemporalConfig,
        TrainConfig,
    )
    from injection_hse.evaluation import evaluate_run
    from injection_hse.train import train_experiment

    def _run(out, *, test_fraction):
        cfg = ExperimentConfig(
            name="split-report",
            seed=0,
            orders=[1, 3, 5],
            data=DataConfig(
                source="parquet",
                dataset_path=str(eval_dataset),
                batch_size=8,
                val_fraction=0.3,
                test_fraction=test_fraction,
                num_workers=0,
            ),
            model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
            measurement=MeasurementConfig(
                voltage_nodes=["Bus R4", "Bus C5", "Bus I2"], slack_known=True
            ),
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="placement",
                        kind="placement",
                        target_loss=1e-9,
                        max_epochs=1,
                    )
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
        manifest = {
            "stages": {
                "0": {
                    "task": "B",
                    "stage_name": "placement",
                    "kind": "placement",
                    "observed_fraction": None,
                }
            },
            "switch_stage_index": 0,
            "sensor_names": ["Bus R4", "Bus C5", "Bus I2"],
            "orders": [1, 3, 5],
            "datasets": {
                "dataset_coherent": str(eval_dataset),
                "dataset_random": str(eval_dataset),
            },
            "final_checkpoint": str(out / "checkpoints" / "stage_placement.ckpt"),
            "grid": {"n_nodes": len(eval_grid.nodes), "n_rows": 0},
        }
        (out / "manifest.json").write_text(json.dumps(manifest))
        (out / "grid.json").write_text(eval_grid.model_dump_json())
        return evaluate_run(out, dataset="coherent", device="cpu")

    # WITH a held-out test split: both summaries written, clearly labeled + structurally complete
    with_test = tmp_path / "with_test"
    with_test.mkdir()
    ed = _run(with_test, test_fraction=0.25)
    names = {p.name for p in ed.iterdir()}
    assert {"metrics.json", "metrics_test.json"} <= names
    assert {"metrics_per_order.csv", "metrics_test_per_order.csv"} <= names
    test_summary = json.loads((ed / "metrics_test.json").read_text())
    assert "overall" in test_summary and "per_order" in test_summary
    assert "pu_mae_unobs" in test_summary["overall"]
    assert set(test_summary["per_order"]) == {"1", "3", "5"}

    # WITHOUT a test split: only the validation metrics (no test file)
    no_test = tmp_path / "no_test"
    no_test.mkdir()
    ed0 = _run(no_test, test_fraction=0.0)
    names0 = {p.name for p in ed0.iterdir()}
    assert "metrics.json" in names0
    assert "metrics_test.json" not in names0


def test_stage_manager_extra_callbacks(eval_grid, eval_dataset, tmp_path):
    """A shared callback attached via extra_callbacks fires across the stages of a run."""
    import lightning as L

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
    from injection_hse.data import build_data_source, base_voltage_per_row
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule, StageManager, make_datamodule

    class Counter(L.Callback):
        def __init__(self):
            self.fits = 0

        def on_fit_start(self, trainer, pl_module):
            self.fits += 1

    cfg = ExperimentConfig(
        name="t",
        seed=0,
        orders=[1, 3, 5],
        data=DataConfig(
            source="parquet",
            dataset_path=str(eval_dataset),
            batch_size=8,
            val_fraction=0.3,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="a",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                ),
                MaskStageConfig(
                    name="b",
                    kind="random",
                    observed_fraction=0.25,
                    target_loss=1e-9,
                    max_epochs=1,
                ),
            ]
        ),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path / "ckpts"),
        ),
    )
    src = build_data_source(cfg.data, grid=eval_grid)
    enc = build_encoding("trig")
    td = src.train_dataset()
    tv = __import__("torch").stack([td[i]["v"] for i in range(len(td))])
    norm = HarmonicNormalizer().fit(enc.encode(tv).transpose(-3, -2))
    module = SEModule(
        cfg,
        n_nodes=src.n_nodes,
        normalizer=norm,
        graph=src.graph(),
        base_v=base_voltage_per_row(eval_grid),
    )
    dm = make_datamodule(src, batch_size=8)
    counter = Counter()
    StageManager(cfg).run(module, dm, logger=False, extra_callbacks=[counter])
    assert counter.fits == 2  # fired once per stage


def test_stage_manager_plateau_early_stop(eval_grid, eval_dataset, tmp_path):
    """``train.early_stop_patience`` ends a stage on a plateau, well before ``max_epochs``.

    With an unreachable ``min_delta`` no epoch ever counts as improvement, so the plateau
    guard trips after ``patience`` validation epochs — far short of the 12-epoch cap.
    """
    import lightning as L

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
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule, StageManager, make_datamodule

    class ValEpochCounter(L.Callback):
        def __init__(self):
            self.val_epochs = 0

        def on_validation_epoch_end(self, trainer, pl_module):
            if not getattr(trainer, "sanity_checking", False):
                self.val_epochs += 1

    max_epochs = 12
    cfg = ExperimentConfig(
        name="t",
        seed=0,
        orders=[1, 3, 5],
        data=DataConfig(
            source="parquet",
            dataset_path=str(eval_dataset),
            batch_size=8,
            val_fraction=0.3,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="a",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,  # unreachable -> only the plateau/epoch guards end it
                    max_epochs=max_epochs,
                ),
            ]
        ),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path / "ckpts"),
            early_stop_patience=2,
            early_stop_min_delta=1e9,  # nothing ever improves by this much
        ),
    )
    src = build_data_source(cfg.data, grid=eval_grid)
    enc = build_encoding("trig")
    td = src.train_dataset()
    tv = __import__("torch").stack([td[i]["v"] for i in range(len(td))])
    norm = HarmonicNormalizer().fit(enc.encode(tv).transpose(-3, -2))
    module = SEModule(
        cfg,
        n_nodes=src.n_nodes,
        normalizer=norm,
        graph=src.graph(),
        base_v=base_voltage_per_row(eval_grid),
    )
    dm = make_datamodule(src, batch_size=8)
    counter = ValEpochCounter()
    StageManager(cfg).run(module, dm, logger=False, extra_callbacks=[counter])
    # patience=2 -> stops after a handful of validation epochs, nowhere near the 12-epoch cap
    assert 1 <= counter.val_epochs < max_epochs
