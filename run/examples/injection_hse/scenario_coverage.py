"""Scenario-coverage report: what operating conditions does a dataset actually span?

A trained state estimator is only as good as the operating points it saw. This reports, over
the samples of a generated dataset, the full DISTRIBUTIONS of the power-quality quantities the
scenarios are supposed to cover — total harmonic distortion, voltage unbalance, fundamental
voltage deviation — plus the load-scale structure (per-sample mean / min / max), so a
too-narrow range or an un-covered corner (e.g. no correlated system-wide peak) is visible at a
glance. Reference lines mark the EN 50160 limits.

Written by DEFAULT next to every training run (``scenario_coverage.{pdf,svg}`` +
``scenario_coverage.csv``) so a run records the conditions it trained on; re-run standalone on
any dataset directory:

    pixi run -e cpu python run/examples/injection_hse/scenario_coverage.py <dataset_dir> --grid <grid.json>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def coverage_features(dataset_path, grid, *, device="cpu"):
    """Per-sample coverage features of a dataset's GROUND-TRUTH state -> ``polars.DataFrame``.

    Reuses :func:`injection_hse.analysis.scenario_features` for the physical quantities (THD, VUF, relative
    loading, voltage deviation) and adds the sampled ``load_scale`` structure when present
    (per-sample mean / min / max over all loads × phases — the diagnostic for whether the
    aggregate feeder ever reaches an all-peak or all-min condition).

    Handles both dataset shapes. A snapshot dataset is ``[B, H, N]``; each scenario is one
    coverage sample. A coherent SEQUENCE dataset is ``[B, T, H, N]`` — the ``T`` axis is
    flattened into the batch so every ``(scenario, step)`` becomes its own coverage sample
    (``[B*T, H, N]``), the honest picture of what the model sees step by step. The
    fundamental operating point (``load_scale``) is drawn once per scenario and held constant
    across its ``T`` steps, so its per-scenario draw is broadcast across those steps to align
    with the flattened rows.
    """
    import polars as pl
    from pgml.assembly import base_voltage_per_row, node_phase_index
    from injection_hse.data.scenarios import read_dataset

    from injection_hse.analysis import scenario_features
    from injection_hse.evaluation import Prediction

    loaded = read_dataset(dataset_path)
    v = np.asarray(loaded.v)
    if v.ndim == 4:
        # coherent SEQUENCE [B, T, H, N]: each (scenario, step) is one coverage sample.
        n_scenarios, n_steps = int(v.shape[0]), int(v.shape[1])
        v = v.reshape(n_scenarios * n_steps, v.shape[2], v.shape[3])
    elif v.ndim == 3:
        # snapshot [B, H, N]: one coverage sample per scenario.
        n_scenarios, n_steps = int(v.shape[0]), 1
    else:
        raise SystemExit(
            f"scenario coverage expects a snapshot [B, H, N] or coherent sequence "
            f"[B, T, H, N] dataset; got {v.shape}."
        )
    f0 = float(grid.base_frequency_hz)
    orders = (
        [int(round(float(f) / f0)) for f in loaded.frequencies_hz]
        if loaded.frequencies_hz is not None
        else list(range(1, v.shape[1] + 1))
    )
    idx = node_phase_index(grid)
    prediction = Prediction(
        target_v=v,
        pred_v=v,
        mask=np.ones((v.shape[0], v.shape[-1]), dtype=bool),
        orders=orders,
        base_v=base_voltage_per_row(grid).numpy().astype(np.float64),
        node_ids=idx.node_ids.numpy().astype(np.int64),
        phase_codes=idx.phase_codes.numpy().astype(np.int64),
        observed_rows=np.zeros(0, dtype=np.int64),
        sensor_names=[],
        grid=grid,
    )
    feats = scenario_features(prediction, device=device)

    # add the sampled load-scale structure (raw draws live in the dataset's samples table).
    # The draw is per-scenario ([B, loads*phases]); a coherent sequence holds it constant
    # across its T steps, so broadcast it to the flattened [B*T, loads*phases] rows.
    ls = loaded.samples.get("load_scale")
    if ls is not None:
        arr = np.asarray(ls).reshape(n_scenarios, -1)  # [B, loads*phases]
        if n_steps != 1:
            arr = np.repeat(arr, n_steps, axis=0)  # [B*T, loads*phases]
        feats = feats.with_columns(
            pl.Series("load_scale_mean", arr.mean(axis=1)),
            pl.Series("load_scale_min", arr.min(axis=1)),
            pl.Series("load_scale_max", arr.max(axis=1)),
        )
    return feats


# EN 50160 reference envelopes (percent) for the marker lines.
_EN50160 = {"thd_max_pct": 8.0, "vuf_max_pct": 2.0, "vdev_max_pct": 10.0}


def plot_coverage(features, out_dir, stem="scenario_coverage"):
    """Four-panel histogram of the coverage features -> ``(pdf_path, svg_path)``.

    THD / voltage deviation / VUF each against their EN 50160 envelope, and the load-scale
    mean vs per-sample min/max (which reveals whether the feeder is ever all-peak or all-min).
    ``stem`` names the written figure files (``<stem>.pdf`` + ``<stem>.svg``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from injection_hse.analysis import SERIES_COLORS, figure_size, paper_style, save_paper_figure

    def col(name):
        return features[name].to_numpy() if name in features.columns else None

    with paper_style(columns=2):
        fig, ax = plt.subplots(2, 2, figsize=figure_size(columns=2, aspect=0.75))
        panels = [
            (ax[0, 0], "thd_max_pct", "max THD [%]", "Harmonic distortion"),
            (ax[0, 1], "vdev_max_pct", "max |V| deviation [%]", "Voltage deviation"),
            (ax[1, 0], "vuf_max_pct", "max VUF [%]", "Voltage unbalance"),
        ]
        for a, key, xlabel, title in panels:
            x = col(key)
            if x is None or not np.isfinite(x).any():
                a.text(0.5, 0.5, "n/a", ha="center", va="center", transform=a.transAxes)
                a.set_xlabel(xlabel)
                continue
            a.hist(x[np.isfinite(x)], bins=50, color=SERIES_COLORS[0])
            lim = _EN50160.get(key)
            if lim is not None:
                a.axvline(
                    lim, color=SERIES_COLORS[1], ls="--", label=f"EN 50160 {lim:g}%"
                )
                a.legend(fontsize=6)
            a.set_xlabel(xlabel)
            a.set_title(title, fontsize=7, loc="left")

        a = ax[1, 1]
        if col("load_scale_mean") is not None:
            a.hist(
                col("load_scale_mean"),
                bins=50,
                color=SERIES_COLORS[0],
                label="per-sample MEAN",
            )
            a.hist(
                col("load_scale_min"),
                bins=50,
                color=SERIES_COLORS[2],
                alpha=0.7,
                label="MIN load",
            )
            a.hist(
                col("load_scale_max"),
                bins=50,
                color=SERIES_COLORS[3],
                alpha=0.7,
                label="MAX load",
            )
            a.set_xlabel("load scale (nameplate fraction)")
            a.set_title("Load coverage (aggregate vs extremes)", fontsize=7, loc="left")
            a.legend(fontsize=6)
        else:
            a.axis("off")
        return save_paper_figure(fig, out_dir, stem)


def coverage_report(
    dataset_path, grid, out_dir, *, device="cpu", stem="scenario_coverage"
):
    """Write ``<stem>.{pdf,svg}`` + ``<stem>.csv`` for a dataset.

    ``<stem>.csv`` holds the per-feature summary (min / p1 / median / p99 / max) so the
    coverage is machine-readable alongside the figure. ``stem`` defaults to
    ``"scenario_coverage"``; pass e.g. ``"scenario_coverage_taskb"`` to write the coherent
    (sequence) report alongside the snapshot one. Returns the figure PDF path.
    """
    import polars as pl

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = coverage_features(dataset_path, grid, device=device)

    # per-feature summary table
    rows = []
    for c in feats.columns:
        if c == "sample":
            continue
        x = feats[c].to_numpy().astype(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        q = np.percentile(x, [0, 1, 50, 99, 100])
        rows.append(
            {
                "feature": c,
                "min": q[0],
                "p1": q[1],
                "median": q[2],
                "p99": q[3],
                "max": q[4],
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(out / f"{stem}.csv")
    pdf, _svg = plot_coverage(feats, out, stem=stem)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "dataset_dir", help="A written dataset directory (meta.json inside)."
    )
    ap.add_argument(
        "--grid",
        default=None,
        help="grid.json the dataset was generated on (default: <dataset_dir>/../grid.json).",
    )
    ap.add_argument(
        "--out", default=None, help="Output dir (default: the dataset dir)."
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from pgml.schemas.grid_schema import Grid

    ds = Path(args.dataset_dir)
    gp = Path(args.grid) if args.grid else ds.parent / "grid.json"
    grid = Grid.model_validate_json(Path(gp).read_text())
    out = Path(args.out) if args.out else ds
    pdf = coverage_report(ds, grid, out, device=args.device)
    print(f"scenario coverage -> {pdf.resolve()} (+ .svg, scenario_coverage.csv)")


if __name__ == "__main__":
    main()
