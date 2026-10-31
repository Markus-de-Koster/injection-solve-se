"""The paper's data figures and its results table, from the evaluation outputs.

- :func:`parity_figure` — estimated against true per-unit ``|V_h|`` of a network at h1, h5
  and h9 on the test sequences at the deployed placement (``parity.pdf``), one row of three
  square panels; the plotted points are kept as ``<name>_points.parquet`` so that
  :func:`parity_replot` restyles the figure without the checkpoint;
- :func:`meter_sweep_figure` — the error against the number of metered LV buses
  (``meter_sweep.pdf``) from the ``placement_sweep_*.csv`` of :mod:`se_ablation.evaluate`:
  Physics, LMMSE [ours] and GNN-P-R [ours] as lines (median, range as band), GNN-P [ours]
  only at the deployed placement (star), since it corrects a linear map fitted there;
- :func:`coupling_figure` — the order coupling through network and solve beside the order
  occlusion (``attr_harmonic_coupling_solve.pdf``) from the CSV of
  :mod:`se_ablation.attribution`;
- :func:`results_table` — the paper's table (``table2.csv``, ``table2.tex``): pu-MAE at the
  deployed and the alternative placement (mean and 95 % confidence half-width over the test
  sequences), on the held-out perturbed grid copies (pooled), and the relative error per
  order at the deployed placement, with the bold rule of the paper.

Conventions: "harmonics" = the mean over h3–h19 of the per-order errors. Every figure is
written as PDF and SVG with the plotted values as CSV beside it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .names import MACK_KEYS, NAMES, PLACEMENTS, TABLE_ROWS, label  # noqa: E402

INK, MUTED = "#3d3d3a", "#8a8a85"
RC = {
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
}
COLORS = {
    "physics": "#52514e",
    "linear": "#2a78d6",
    "lmmse": "#1f9e89",
    "gnn_p": "#e08a1e",
    "gnn_p_r": "#8e44ad",
}


def _save(fig, df, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    df.to_csv(out / f"{name}.csv", index=False)
    plt.close(fig)


def _read(paths) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in paths]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- parity
PARITY_ORDERS = (1, 5, 9)


def parity_points(
    checkpoint: Path, data: Path, *, orders=PARITY_ORDERS, sensors=None, split: str = "test"
) -> pd.DataFrame:
    """Per-unit ``|V_h|``, true and estimated, of every (state, node-phase row) of a split
    of ``<data>/dataset_coherent`` at ``sensors`` (default: the deployed placement plus the
    supply, with the checkpoint's current channels and reading model). Columns ``order,
    state, row, observed, true_pu, pred_pu``."""
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import MeasurementConfig
    from injection_hse.evaluation import load_se_module, predict_dataset

    data = Path(data)
    grid = Grid.model_validate_json((data / "grid.json").read_text())
    module = load_se_module(str(checkpoint), grid=grid, dataset_path=str(data / "dataset_coherent"))
    meas = MeasurementConfig(
        voltage_nodes=list(sensors or PLACEMENTS["deployed"]),
        slack_known=True,
        measure_current=module.cfg.measurement.measure_current,
        current_channels=module.cfg.measurement.current_channels,
    )
    pred = predict_dataset(
        module,
        source_or_path=str(data / "dataset_coherent"),
        grid=grid,
        measurement=meas,
        split=split,
    )
    base = pred.base_v[None, None, :]
    true = np.abs(pred.target_v) / base
    est = np.abs(pred.pred_v) / base
    s, n = true.shape[0], true.shape[2]
    parts = []
    for o in orders:
        k = list(pred.orders).index(o)
        parts.append(
            pd.DataFrame(
                {
                    "order": o,
                    "state": np.repeat(np.arange(s), n),
                    "row": np.tile(np.arange(n), s),
                    "observed": pred.mask.reshape(-1).astype(bool),
                    "true_pu": true[:, k, :].reshape(-1),
                    "pred_pu": est[:, k, :].reshape(-1),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def parity_plot(points: pd.DataFrame, out: Path, name: str = "parity") -> Path:
    """One row of square parity panels (3.4 in wide; h1 in pu, harmonics in % of the base
    voltage; dashed y = x; points rasterized). Writes ``<name>.pdf`` / ``.svg`` / ``.png``
    (300 dpi)."""
    from matplotlib.ticker import MaxNLocator

    orders = sorted(points.order.unique())
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 6.5, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, len(orders), figsize=(3.4, 1.3), constrained_layout=True)
        for ax, o in zip(np.atleast_1d(axes), orders):
            d = points[points.order == o]
            lo = float(min(d.true_pu.min(), d.pred_pu.min()))
            hi = float(max(d.true_pu.max(), d.pred_pu.max()))
            scale = 1.0 if o == 1 else 100.0
            ax.scatter(
                d.true_pu * scale,
                d.pred_pu * scale,
                s=0.6,
                alpha=0.25,
                lw=0,
                color=COLORS["linear"],
                rasterized=True,
            )
            ax.plot([lo * scale, hi * scale], [lo * scale, hi * scale], "--", color=INK, lw=0.6)
            pad = 0.03 * (hi - lo)
            ax.set_xlim((lo - pad) * scale, (hi + pad) * scale)
            ax.set_ylim((lo - pad) * scale, (hi + pad) * scale)
            ax.set_box_aspect(1)
            ax.set_title(f"h{o}", fontsize=7, pad=2)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.set_xlabel("true " + ("[pu]" if o == 1 else "[%]"), fontsize=6, labelpad=1)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel("estimated |V| [pu, %]", fontsize=6, labelpad=1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    return out / f"{name}.pdf"


def parity_figure(
    checkpoint: Path, data: Path, out: Path, *, orders=PARITY_ORDERS, name: str = "parity"
) -> Path:
    """:func:`parity_points` of the test split, kept as ``<name>_points.parquet``, and
    :func:`parity_plot`."""
    points = parity_points(checkpoint, data, orders=orders)
    out.mkdir(parents=True, exist_ok=True)
    points.to_parquet(out / f"{name}_points.parquet", index=False)
    return parity_plot(points, out, name)


def parity_replot(points_parquet: Path, out: Path, name: str = "parity") -> Path:
    """The parity figure from kept points (``<name>_points.parquet``)."""
    return parity_plot(pd.read_parquet(points_parquet), out, name)


# ----------------------------------------------------------------------------- sweep
#: (estimator in the sweep CSV, name key, marker, deployed star only)
SWEEP_SERIES = [
    ("physics solve", "physics", "o", False),
    ("statistics-based", "lmmse", "D", False),
    ("gnn_p_r, as deployed", "gnn_p_r", "s", False),
    ("gnn_p, as deployed", "gnn_p", "*", True),
]


def meter_sweep_figure(sweep_csvs, out: Path, *, name: str = "meter_sweep") -> Path:
    """pu-MAE against the number of metered LV buses, h1 (left) and the mean over h3–h19
    (right): median (line) and range (band) over the random placements of each count,
    stars at the deployed placement; GNN-P [ours] only as its star."""
    sw = _read(sweep_csvs)
    sw["part"] = sw.order.map(lambda h: "h1" if h == 1 else "harmonics")
    per = sw.groupby(["placement", "k", "estimator", "part"]).pu_mae.mean().reset_index()
    series = [
        (est, COLORS[key], mk, label(key), star)
        for est, key, mk, star in SWEEP_SERIES
        if est in set(per.estimator)
    ]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
        recs = []
        for ax, part in zip(axes, ["h1", "harmonics"]):
            p = per[per.part == part]
            for est, col, mk, lab, star_only in series:
                dep = p[(p.estimator == est) & (p.placement == "deployed")]
                if len(dep):
                    ax.scatter(
                        [3],
                        dep.pu_mae.values,
                        marker="*",
                        s=70,
                        color=col,
                        edgecolors=INK,
                        linewidths=0.4,
                        zorder=4,
                    )
                if star_only:
                    if len(dep):
                        recs.append(
                            {
                                "part": part,
                                "estimator": lab,
                                "k": 3,
                                "deployed": float(dep.pu_mae.values[0]),
                            }
                        )
                    continue
                q = p[(p.estimator == est) & (p.placement != "deployed")]
                if not len(q):
                    continue
                g = q.groupby("k").pu_mae
                med, lo, hi = g.median(), g.min(), g.max()
                ax.plot(med.index, med.values, color=col, marker=mk, ms=3.5, lw=1.1)
                ax.fill_between(med.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
                for k in med.index:
                    recs.append(
                        {
                            "part": part,
                            "estimator": lab,
                            "k": k,
                            "median": med[k],
                            "min": lo[k],
                            "max": hi[k],
                            "deployed": float(dep.pu_mae.values[0])
                            if (len(dep) and k == 3)
                            else np.nan,
                        }
                    )
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(range(1, 7))
            ax.set_xlabel("metered LV buses (plus the supply)")
            ax.set_title(part, color=INK, fontsize=8)
        axes[0].set_ylabel("pu-MAE, unobserved buses")
        handles = [
            Line2D(
                [],
                [],
                color=c,
                marker=m,
                ms=7 if so else 3.5,
                lw=0 if so else 1.1,
                markeredgecolor=INK if so else c,
                markeredgewidth=0.4 if so else 1,
            )
            for _, c, m, _, so in series
        ]
        fig.legend(
            handles,
            [s[3] for s in series],
            frameon=False,
            fontsize=6.5,
            ncol=len(series),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
        )
        _save(fig, pd.DataFrame(recs), out, name)
    return out / f"{name}.pdf"


# ----------------------------------------------------------------------------- coupling
def coupling_figure(
    coupling_csv: Path, out: Path, *, name: str = "attr_harmonic_coupling_solve"
) -> Path:
    """The two-panel order coupling from ``attr_harmonic_coupling_solve.csv``."""
    from injection_hse.analysis.style import save_paper_figure

    from .attribution import coupling_plot

    d = pd.read_csv(coupling_csv)
    orders = sorted(int(o) for o in d.output_order.unique())

    def matrix(col):
        return d.pivot(index="output_order", columns="input_order", values=col).loc[orders, orders]

    fig = coupling_plot(
        orders, matrix("ig_coupling_solve").to_numpy(), matrix("order_occlusion_solve").to_numpy()
    )
    out.mkdir(parents=True, exist_ok=True)
    pdf, _ = save_paper_figure(fig, out, name)
    plt.close(fig)
    d.to_csv(out / f"{name}.csv", index=False)
    return pdf


# ----------------------------------------------------------------------------- table
#: the table's error columns: (column, placement or "perturbed", part)
TABLE_COLUMNS = [
    ("deployed_h1", "deployed", "h1"),
    ("deployed_harm", "deployed", "harm"),
    ("alternative_h1", "alternative", "h1"),
    ("alternative_harm", "alternative", "harm"),
    ("perturbed_h1", "perturbed", "h1"),
    ("perturbed_harm", "perturbed", "harm"),
]
#: within this relative distance of the best value of a column, a value is bold as well
BOLD_WITHIN = 0.05


def _pooled(df: pd.DataFrame) -> pd.DataFrame:
    if "copy" in df.columns:
        df = df[df["copy"].isna() | (df["copy"].astype(str) == "all")]
    return df


def _parts(per_order: pd.DataFrame) -> dict:
    """``{"h1": ..., "harm": mean over h3–h19, order: ...}`` of one estimator's rows."""
    s = per_order.set_index("order").sort_index()
    return {"h1": float(s.pu_mae.loc[1]), "harm": float(s.pu_mae[s.index > 1].mean())}


def results_table(
    evaluation_csvs,
    out: Path,
    *,
    perturbed_csvs=(),
    mack_csv: Path | None = None,
    mack_perturbed_csv: Path | None = None,
    name: str = "table2",
) -> tuple[Path, Path]:
    """The paper's table from the evaluation outputs.

    ``evaluation_csvs``: the ``baseline_table_*.csv`` and ``ci_*.csv`` of
    ``evaluate --experiment baselines`` on the nominal test data (placements ``deployed``
    and ``alternative``; the networks named ``gnn_p`` and ``gnn_p_r``); ``perturbed_csvs``:
    the ``baseline_table_*.csv`` of the same evaluation on the held-out perturbed grid copies
    (``--test-dataset dataset_coherent_heldout``, deployed, pooled over the copies);
    ``mack_csv`` / ``mack_perturbed_csv``: the ``per_order.csv`` of ``mack table`` on the
    nominal and on the perturbed test data.

    Writes ``<name>.csv`` (pu-MAE in 1e-4 pu: ``<column>`` and, where the test sequences
    give one, ``<column>_ci95``; relative errors per order in % as ``rel_h<order>``; the
    bold flags as ``bold_<column>``) and ``<name>.tex`` (the table's body rows: three
    significant digits, bold = best per column and every value within 5 % of it, ``*``
    on Linear-P at the alternative placement, which is refitted there).
    """
    ev = _read([p for p in evaluation_csvs if not Path(p).name.startswith("ci_")])
    ci = _read([p for p in evaluation_csvs if Path(p).name.startswith("ci_")])
    ev = _pooled(ev)
    pert = _pooled(_read(perturbed_csvs)) if perturbed_csvs else pd.DataFrame()
    mack = _read([mack_csv]) if mack_csv else pd.DataFrame()
    mack_p = _read([mack_perturbed_csv]) if mack_perturbed_csv else pd.DataFrame()

    def rows_of(df, run, est, placement="deployed"):
        if not len(df):
            return df
        return df[(df.placement == placement) & (df.run == run) & (df.estimator == est)]

    recs = []
    for key, run, est in TABLE_ROWS:
        rec = {"key": key, "estimator": NAMES[key]}
        if est is None:  # Mack et al.: deployed placement only
            mk = {v: k for k, v in MACK_KEYS.items()}[key]
            for src, place in ((mack, "deployed"), (mack_p, "perturbed")):
                d = src[src.run == mk] if len(src) else src
                if len(d):
                    for part, v in _parts(d).items():
                        rec[f"{place}_{part}"] = 1e4 * v
                    if place == "deployed":
                        for r in d.itertuples():
                            rec[f"rel_h{int(r.order)}"] = 100 * float(r.rel)
            recs.append(rec)
            continue
        run = run or key
        for place in ("deployed", "alternative"):
            d = rows_of(ev, run, est, place)
            if not len(d):
                continue
            for part, v in _parts(d).items():
                rec[f"{place}_{part}"] = 1e4 * v
            c = (
                ci[(ci.placement == place) & (ci.run == run) & (ci.estimator == est)]
                if len(ci)
                else ci
            )
            for r in c.itertuples():
                rec[f"{place}_{r.part}_ci95"] = 1e4 * float(r.ci95_half_width)
            if place == "deployed":
                for r in d.itertuples():
                    rec[f"rel_h{int(r.order)}"] = 100 * float(r.rel)
        d = rows_of(pert, run, est)
        if len(d):
            for part, v in _parts(d).items():
                rec[f"perturbed_{part}"] = 1e4 * v
        recs.append(rec)
    df = pd.DataFrame(recs)
    rel_cols = sorted((c for c in df.columns if c.startswith("rel_h")), key=lambda c: int(c[5:]))
    value_cols = [c for c, *_ in TABLE_COLUMNS if c in df.columns] + rel_cols
    for col in value_cols:
        best = df[col].min()
        df[f"bold_{col}"] = df[col] <= (1 + BOLD_WITHIN) * best
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)

    lines = []
    all_cols = [c for c, *_ in TABLE_COLUMNS] + rel_cols
    for r in df.itertuples(index=False):
        r = r._asdict()
        cells = [r["estimator"]]
        for col in all_cols:
            v = r.get(col, np.nan)
            if v is None or pd.isna(v):
                cells.append("--")
                continue
            txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:#.3g}".rstrip(".")
            if r["key"] == "linear" and col.startswith("alternative"):
                txt += "$^{*}$"
            cells.append(f"\\textbf{{{txt}}}" if r.get(f"bold_{col}") else txt)
        lines.append(" & ".join(cells) + r" \\")
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[["estimator"] + value_cols].round(3).to_string(index=False))
    return out / f"{name}.csv", out / f"{name}.tex"
