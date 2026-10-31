"""analyze_run writes the full PDF+SVG figure set from a smoke run directory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import polars as pl

from injection_hse.analysis import analyze_run, plot_gradient_dynamics
from injection_hse.analysis.report import _random_stage_checkpoint


def test_random_stage_checkpoint_picks_random_over_placement(run_dir):
    """Node importance sources the deepest RANDOM stage, never the placement finetune."""
    import json

    manifest = json.loads((run_dir / "manifest.json").read_text())
    ckpt, frac = _random_stage_checkpoint(manifest)
    assert ckpt is not None and ckpt.name == "stage_random_50.ckpt"
    assert frac == 0.5
    # a curriculum with no random stage -> (None, None) so the caller falls back to final
    only_placement = dict(manifest)
    only_placement["stages"] = {"0": manifest["stages"]["1"]}  # the placement stage
    assert _random_stage_checkpoint(only_placement) == (None, None)


def test_analyze_run_writes_figure_set(run_dir):
    analysis_dir = analyze_run(
        run_dir, dataset="random", ig_steps=2, n_attribution_samples=4, bins=3
    )
    for stem in (
        "attr_order_importance",
        "occlusion_orders",
        "attr_harmonic_coupling",
        "attr_harmonic_coupling_marginal",
        "attr_node_importance",
        "occlusion_sensors",
        "training_weight_deltas",
        "error_vs_scenario",
        "error_vs_load_scale",
    ):
        for ext in ("pdf", "svg"):
            path = analysis_dir / f"{stem}.{ext}"
            assert path.exists() and path.stat().st_size > 0, path
    assert (analysis_dir / "occlusion_sensors.csv").exists()
    # not instrumented -> no gradient-dynamics figure
    assert not (analysis_dir / "training_grad_dynamics.pdf").exists()
    plt.close("all")


def test_plot_gradient_dynamics_standalone():
    frame = pl.DataFrame(
        {
            "stage_index": [0] * 6 + [1] * 6,
            "step": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "group": ["encoder", "head"] * 6,
            "grad_norm": [0.5, 0.1] * 6,
            "param_norm": [1.0, 1.0] * 6,
        }
    )
    fig = plot_gradient_dynamics(frame, switch_stage_index=1, smooth=1)
    assert fig is not None
    plt.close(fig)


def test_banded_layer_groups_aggregate_numbered_layers():
    """Numbered conv/block/mlp groups band together; everything else passes through."""
    from injection_hse.analysis.report import _banded_layer_groups

    groups = [
        "temporal.memory",
        "spatial.encoder",
        *(f"spatial.conv_{k}" for k in range(6)),
        "spatial.head",
    ]
    banded = _banded_layer_groups(groups, band=3)
    assert banded["spatial.conv_0-2"] == [f"spatial.conv_{k}" for k in range(3)]
    assert banded["spatial.conv_3-5"] == [f"spatial.conv_{k}" for k in range(3, 6)]
    assert banded["temporal.memory"] == ["temporal.memory"]
    assert banded["spatial.head"] == ["spatial.head"]
    # band=1 disables aggregation (identity grouping)
    assert list(_banded_layer_groups(groups, band=1)) == groups


def test_plot_harmonic_coupling_marginal_variant():
    """row_mass=True renders the absolute row-mass marginal next to each panel."""
    import numpy as np

    from injection_hse.analysis import plot_harmonic_coupling

    rng = np.random.default_rng(0)
    orders = [1, 3, 5]
    fig = plot_harmonic_coupling(
        orders, rng.random((3, 3)), rng.random((3, 3)), row_mass=True
    )
    assert len(fig.axes) >= 4  # 2 heatmaps + 2 marginal strips (+ colorbars)
    plt.close(fig)


def test_analyze_run_records_the_linear_injection_prior(current_run_dir):
    """An injection-decode run's band ablation carries the linear-prior row and the fitted
    map travels with the run."""
    from injection_hse.analysis import LinearInjectionPrior

    analysis_dir = analyze_run(
        current_run_dir,
        dataset="random",
        ig_steps=2,
        n_attribution_samples=4,
        n_prior_samples=8,
        bins=3,
    )
    prior_path = analysis_dir / "linear_prior.npz"
    assert prior_path.exists()
    prior = LinearInjectionPrior.load(prior_path)
    assert prior.n_train == 8 and prior.n_features > 0

    frame = pl.read_csv(analysis_dir / "estimator_band_ablation.csv")
    variants = frame["variant"].to_list()
    assert "linear_prior" in variants and "anchors_only" in variants
    plt.close("all")


def test_analyze_run_band_ablation_scores_the_sequence_test_split(two_task_run_dir):
    """A two-task run's band ablation scores the SEQUENCE test split with the prior its
    checkpoint carries; the snapshot-validation reading is kept beside it, named."""
    import json

    import numpy as np
    from pgml.schemas.grid_schema import Grid

    from injection_hse.analysis import estimator_band_ablation, evaluation_batch
    from injection_hse.config import MeasurementConfig
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.evaluation import load_se_module

    from .conftest import SENSORS

    analysis_dir = analyze_run(
        two_task_run_dir,
        dataset="coherent",
        ig_steps=2,
        n_attribution_samples=4,
        n_prior_samples=8,
        n_band_samples=3,
        bins=3,
    )
    manifest = json.loads((two_task_run_dir / "manifest.json").read_text())
    seq_path = manifest["datasets"]["dataset_coherent"]
    record = json.loads((analysis_dir / "estimator_band_ablation.json").read_text())
    main = record["estimator_band_ablation.csv"]
    assert (main["dataset"], main["split"], main["dataset_path"]) == (
        "coherent",
        "test",
        seq_path,
    )
    assert main["n_steps"] == 4 and main["n_states"] == 4 * main["n_samples"]
    assert main["linear_prior"]["origin"] == "checkpoint"
    assert main["linear_prior"]["fit_dataset"] == f"{seq_path}:train"
    snap = record["estimator_band_ablation_snapshot_val.csv"]
    assert (snap["dataset"], snap["split"], snap["n_steps"]) == ("random", "val", 1)
    # the carried prior serves both tables; no second bundle is written
    assert snap["linear_prior"]["file"] == "linear_prior.npz"
    assert not (analysis_dir / "linear_prior_snapshot_train.npz").exists()

    # the table IS the band ablation of the final model on the sequence test split
    grid = Grid.model_validate_json((two_task_run_dir / "grid.json").read_text())
    module = load_se_module(
        manifest["final_checkpoint"], grid=grid, dataset_path=seq_path
    )
    source = build_data_source(
        module.cfg.data.model_copy(update={"dataset_path": seq_path}), grid=grid
    )
    v, mask, time_unix_s = evaluation_batch(
        source,
        grid=grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        split="test",
        n_samples=3,
    )
    assert v.shape[0] == main["n_samples"]
    expected = estimator_band_ablation(
        module,
        v,
        mask,
        base_v=base_voltage_per_row(grid),
        linear_prior=module.injection_prior.prior,
        time_unix_s=time_unix_s,
    )
    written = pl.read_csv(analysis_dir / "estimator_band_ablation.csv")
    assert written["variant"].to_list() == expected["variant"].to_list()
    assert "linear_prior" in written["variant"].to_list()
    cols = [c for c in written.columns if c.startswith("mae_h")]
    np.testing.assert_allclose(
        written.select(cols).to_numpy(), expected.select(cols).to_numpy(), rtol=1e-6
    )
    snapshot = pl.read_csv(analysis_dir / "estimator_band_ablation_snapshot_val.csv")
    assert not np.allclose(
        snapshot.select(cols).to_numpy(), written.select(cols).to_numpy()
    )
    plt.close("all")


def _overlapping_xticklabels(fig) -> int:
    """Number of adjacent x tick-label pairs whose drawn extents overlap, over all axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    count = 0
    for ax in fig.axes:
        boxes = [
            t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()
        ]
        boxes.sort(key=lambda b: b.x0)
        count += sum(1 for a, b in zip(boxes, boxes[1:]) if a.x1 > b.x0)
    return count


@pytest.mark.parametrize(
    "orders", [[1, 3, 5, 7, 9], list(range(1, 20, 2)), list(range(1, 26))]
)
def test_order_axes_keep_their_labels_apart(orders):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from injection_hse.analysis.report import (
        plot_estimator_coupling,
        plot_harmonic_coupling,
        plot_order_importance,
    )

    rng = np.random.default_rng(0)
    h = len(orders)
    figs = [
        plot_order_importance(orders, rng.random(h) + 0.1),
        plot_harmonic_coupling(
            orders, rng.random((h, h)) + 0.1, rng.random((h, h)) + 0.1
        ),
        plot_harmonic_coupling(
            orders, rng.random((h, h)) + 0.1, rng.random((h, h)) + 0.1, row_mass=True
        ),
        plot_estimator_coupling(
            orders, rng.random((h, h)) + 0.1, rng.random((h, h)) - 0.5
        ),
    ]
    for fig in figs:
        assert _overlapping_xticklabels(fig) == 0
        plt.close(fig)
