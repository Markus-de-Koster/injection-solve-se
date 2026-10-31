"""One-call interpretability analysis of a finished run -> the paper figure set.

:func:`analyze_run` mirrors :func:`injection_hse.evaluation.evaluate_run`: it reads the
run's ``manifest.json`` + ``grid.json`` + final checkpoint and writes every
analysis figure into ``out_dir/analysis/`` as a PDF + SVG pair in the IEEE
double-column style (:mod:`injection_hse.analysis.style`). The attribution,
order-occlusion and physical error-binning figures always use the RANDOMIZED
snapshot dataset (clean single-step attribution, and the only dataset whose
fundamental operating point varies); the sensor occlusion uses the selected
evaluation dataset (default the coherent sequences); the gradient-dynamics figure appears when the run was instrumented
(``gradient_norms.parquet``), and the weight-delta figure works from the stage
checkpoints of any run.

The plot functions are also public, each consuming plain data (numpy / polars)
and returning a matplotlib figure styled for the paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .attribution import (
    attribution_batch,
    evaluation_batch,
    harmonic_coupling,
    integrated_gradients,
    node_importance,
    order_importance,
    saliency,
)
from .baselines import fit_linear_injection_prior
from .binning import binned_errors, sample_features, scenario_features
from .gradients import checkpoint_weight_deltas, read_gradient_norms
from .occlusion import (
    estimator_band_ablation,
    estimator_order_coupling,
    order_occlusion,
    order_occlusion_error,
    sensor_occlusion,
)
from .style import (
    SERIES_COLORS,
    figure_size,
    paper_style,
    save_paper_figure,
    series_style,
)

if TYPE_CHECKING:
    import polars as pl

    from pgml.schemas.grid_schema import Grid

_MANIFEST = "manifest.json"
_GRADIENT_NORMS = "gradient_norms.parquet"


def _order_ticks(ax, orders, *, x: bool = True, y: bool = False) -> None:
    """Label an axis of harmonic orders so the labels never run into each other.

    Up to six orders, or single-digit orders only, stay horizontal. A longer axis with
    two-digit orders gets vertical labels whose font fits the category pitch (the axis
    width per order, measured on the figure as laid out when this is called; a colorbar
    added later narrows it, hence the margin); below a 5 pt font every second label is
    left blank instead of shrinking further. The y axis keeps every label.
    """
    ticks = range(len(orders))
    labels = [str(o) for o in orders]
    if y:
        ax.set_yticks(ticks, labels)
    if not x:
        return
    crowded = len(orders) > 6 and max(int(o) for o in orders) >= 10
    if not crowded:
        ax.set_xticks(ticks, labels)
        return
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    pitch = 0.7 * width_pt / len(orders)  # margin for a colorbar taken from the axis
    size = min(float(_xtick_fontsize(ax)), pitch / 1.2)
    if size < 5.0:
        size = min(float(_xtick_fontsize(ax)), 2.0 * pitch / 1.2)
        labels = [lab if i % 2 == 0 else "" for i, lab in enumerate(labels)]
    ax.set_xticks(ticks, labels, rotation=90, fontsize=size)


def _xtick_fontsize(ax) -> float:
    """The x tick-label font size in points the axis currently uses."""
    from matplotlib.font_manager import FontProperties
    import matplotlib

    return float(
        FontProperties(size=matplotlib.rcParams["xtick.labelsize"]).get_size_in_points()
    )


# =============================================================================
# Plot functions (plain data in, one paper-styled figure out)
# =============================================================================
def plot_order_importance(orders: list[int], importance: np.ndarray):
    """Bar chart of attribution mass per INPUT harmonic order (log scale).

    Degenerate attribution (all-zero or non-finite mass — e.g. a diverged model
    whose forward pass is NaN) renders on a linear axis with a warning in the
    title instead of crashing the figure set: the analysis of a broken run must
    still produce its report so the breakage is visible.
    """
    import matplotlib.pyplot as plt

    with paper_style(columns=1):
        fig, ax = plt.subplots()
        share = np.asarray(importance, dtype=np.float64)
        finite = np.isfinite(share).all()
        share = np.where(np.isfinite(share), share, 0.0)
        total = share.sum()
        if total > 0:
            share = share / total
        ax.bar(range(len(orders)), share, color=SERIES_COLORS[0], width=0.7)
        _order_ticks(ax, orders)
        if finite and (share > 0).any():
            ax.set_yscale("log")
        else:
            ax.set_title(
                "degenerate attribution (all-zero or non-finite) — linear axis",
                loc="left",
                fontsize=7,
            )
        ax.set_xlabel("input harmonic order")
        ax.set_ylabel("attribution share")
        ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_order_occlusion(frame: "pl.DataFrame"):
    """Leave-one-frequency-out: estimation-error increase per removed input order.

    Bars of ``delta_mae`` — how much the unobserved per-unit MAE RISES when the
    estimator is denied each harmonic order at the sensors (from
    :func:`~injection_hse.analysis.order_occlusion_error`). A tall bar is a frequency the
    model genuinely relies on; a bar near zero means that order carries little
    information for the estimate. The unoccluded baseline error is stated in the
    subtitle. This is the leave-one-out counterpart to the gradient-attribution
    ``attr_order_importance`` figure.
    """
    import matplotlib.pyplot as plt

    rows = frame.filter(frame["order"] != "none")
    baseline = float(frame.filter(frame["order"] == "none")["mae"][0])
    orders = rows["order"].to_list()
    delta = rows["delta_mae"].to_numpy()

    with paper_style(columns=1):
        fig, ax = plt.subplots()
        ax.bar(range(len(orders)), delta, color=SERIES_COLORS[0], width=0.7)
        ax.axhline(0.0, color="0.4", linewidth=0.6)
        _order_ticks(ax, [str(o) for o in orders])
        ax.set_xlabel("input harmonic order removed")
        ax.set_ylabel("Δ unobserved pu-MAE")
        ax.set_title(
            f"Frequency importance (leave-one-out); baseline pu-MAE {baseline:.2e}",
            loc="left",
            fontsize=7,
        )
        ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_harmonic_coupling(
    orders: list[int],
    gradient: np.ndarray,
    occlusion: np.ndarray,
    *,
    row_mass: bool = False,
    columns: Optional[int] = None,
):
    """Two-panel output-order x input-order coupling: gradient and occlusion.

    Each panel is row-normalized (every output order's input attribution sums
    to one) so the two methods are visually comparable; a log colour scale
    keeps the off-diagonal structure visible next to the dominant entries.

    ``columns`` sizes the figure for a 1- or 2-column paper slot (default: 1
    without the marginal, 2 with it). At ``columns=1`` the fonts render at
    full paper size when the figure is included at column width — the small
    order-count heatmaps stay legible side by side in a single column.

    Row normalization deliberately equalizes the output orders: it shows the input
    MIX each output order relies on, but hides that some output orders receive far
    more absolute attribution than others. ``row_mass=True`` attaches a marginal
    column to each panel — the ABSOLUTE row mass (each output order's total
    attribution over all inputs, normalized to the largest row, log scale) — so both
    readings are available at once: the heatmap says "of what this output uses, how
    is it distributed over input orders"; the marginal says "how much this output
    uses at all". A row with a bright heatmap but a dark marginal contributes little
    to the model's behaviour despite its visible structure.
    """
    import matplotlib.pyplot as plt
    from pgml.evaluation.style import positive_log_norm

    def _row_normalized(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(m, dtype=np.float64)
        sums = m.sum(axis=1, keepdims=True)
        return (
            np.divide(m, sums, out=np.zeros_like(m), where=sums > 0),
            sums.reshape(-1),
        )

    if columns is None:
        columns = 2 if row_mass else 1
    with paper_style(columns=columns):
        if row_mass:
            fig, axes = plt.subplots(
                1,
                4,
                figsize=figure_size(columns=columns, aspect=0.45),
                width_ratios=[1.0, 0.08, 1.0, 0.08],
            )
            panel_axes = [(axes[0], axes[1]), (axes[2], axes[3])]
        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=figure_size(
                    columns=columns, aspect=0.52 if columns == 1 else 0.45
                ),
            )
            panel_axes = [(axes[0], None), (axes[1], None)]
        for (ax, ax_mass), matrix, title in zip(
            panel_axes,
            (gradient, occlusion),
            ("gradient attribution", "order occlusion"),
        ):
            m, mass = _row_normalized(matrix)
            im = ax.imshow(
                m,
                origin="upper",
                aspect="auto",
                cmap="magma",
                norm=positive_log_norm(m),
            )
            _order_ticks(ax, orders, y=True)
            ax.set_xlabel("input harmonic order")
            ax.set_title(title)
            # with a marginal, hang the colorbar right of BOTH axes so the mass strip
            # stays directly adjacent to its heatmap
            cbar_ax = [ax] if ax_mass is None else [ax, ax_mass]
            fig.colorbar(im, ax=cbar_ax, fraction=0.046, pad=0.03)
            if ax_mass is not None:
                rel = mass / mass.max() if mass.max() > 0 else mass
                ax_mass.imshow(
                    rel.reshape(-1, 1),
                    origin="upper",
                    aspect="auto",
                    cmap="magma",
                    norm=positive_log_norm(rel.reshape(-1, 1)),
                )
                ax_mass.set_xticks([0], ["Σ"])
                ax_mass.set_yticks([])
                ax_mass.set_title("row\nmass", fontsize=6)
        panel_axes[0][0].set_ylabel("output harmonic order")
    return fig


def plot_estimator_coupling(
    orders: list[int],
    shift: np.ndarray,
    delta_mae: np.ndarray,
    *,
    columns: int = 2,
):
    """Two-panel estimator-level leave-one-order-out coupling.

    Left: the row-normalized state shift (log colour scale) — of the state
    change output order ``i`` sees, how it distributes over the removed input
    orders. Right: the signed change of the unobserved pu-MAE (diverging scale
    around zero) — what removing each order's measurements actually costs per
    output order. Both from :func:`~injection_hse.analysis.occlusion.estimator_order_coupling`.
    """
    import matplotlib.pyplot as plt
    from pgml.evaluation.style import positive_log_norm

    with paper_style(columns=columns):
        fig, axes = plt.subplots(
            1, 2, figsize=figure_size(columns=columns, aspect=0.45)
        )
        sums = shift.sum(axis=1, keepdims=True)
        m = np.divide(shift, sums, out=np.zeros_like(shift), where=sums > 0)
        im = axes[0].imshow(
            m, origin="upper", aspect="auto", cmap="magma", norm=positive_log_norm(m)
        )
        axes[0].set_title("state shift (row-normalized)")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)

        lim = float(np.abs(delta_mae).max()) or 1.0
        im = axes[1].imshow(
            delta_mae, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim
        )
        axes[1].set_title("Δ unobserved pu-MAE")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

        for ax in axes:
            _order_ticks(ax, orders, y=True)
            ax.set_xlabel("removed input harmonic order")
        axes[0].set_ylabel("output harmonic order")
    return fig


def plot_node_importance(
    grid: "Grid",
    node_values: dict[int, float],
    *,
    sensor_names: Sequence = (),
    value_label: str = "attribution mass",
    positions: Optional[dict] = None,
):
    """The grid graph coloured by per-node attribution mass, sensors outlined.

    Wraps :func:`pgml.evaluation.plot_grid_graph`; the sensor nodes are marked with an
    outline ring (resolved by node name or id — an unresolvable name warns rather than
    silently dropping the marker). ``positions`` places the nodes at explicit
    (geographic) coordinates instead of the automatic layout — any mapping
    :func:`pgml.evaluation.graph_layout` accepts, e.g. loaded via
    :func:`pgml.evaluation.load_node_positions` from a JSON keyed by zero-based node
    numbers (``run/configs/cigre_lv_geo.json`` for the CIGRE LV benchmark). The
    node drawing and the sensor overlay share ONE resolved position set, so the markers
    stay on the nodes for any layout source.
    """
    import warnings

    import matplotlib.pyplot as plt
    from pgml.evaluation import graph_layout, plot_grid_graph

    pos = graph_layout(grid, positions=positions)
    with paper_style(columns=1):
        fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.9))
        plot_grid_graph(
            grid,
            node_values=node_values,
            value_label=value_label,
            ax=ax,
            node_size=40,
            title="",
            positions=pos,
        )
        if sensor_names:
            by_name = {n.name: n.id for n in grid.nodes}
            sensor_ids = []
            for s in sensor_names:
                sid = by_name.get(s, s if isinstance(s, int) else None)
                if sid is None or sid not in pos:
                    warnings.warn(
                        f"sensor {s!r} matches no grid node — marker skipped.",
                        stacklevel=2,
                    )
                    continue
                sensor_ids.append(sid)
            xy = np.array([pos[s] for s in sensor_ids])
            if xy.size:
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=110,
                    facecolors="none",
                    edgecolors="tab:red",
                    linewidths=1.2,
                    zorder=5,
                    label="sensor",
                )
                ax.legend(loc="lower right")
    return fig


def plot_sensor_occlusion(frame: "pl.DataFrame"):
    """Error increase over the full placement when sensors are dropped.

    Grouped bars of the unobserved pu-MAE DELTA vs the ``dropped == "none"``
    baseline, fundamental and harmonic band side by side, one group per
    dropped sensor / sensor pair.
    """
    import matplotlib.pyplot as plt

    base = frame.filter(frame["dropped"] == "none")
    rest = frame.filter(frame["dropped"] != "none")
    base_fund = float(base["mae_fund"][0])
    base_harm = float(base["mae_harm"][0])
    labels = rest["dropped"].to_list()
    d_fund = rest["mae_fund"].to_numpy() - base_fund
    d_harm = rest["mae_harm"].to_numpy() - base_harm

    with paper_style(columns=1):
        fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.7))
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, d_fund, width, label="fundamental", color="0.25")
        ax.bar(x + width / 2, d_harm, width, label="harmonics", color="0.65")
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.axhline(0.0, color="0.2", linewidth=0.6)
        ax.set_ylabel("Δ unobserved pu-MAE")
        ax.set_xlabel("dropped sensor(s)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    return fig


def _banded_layer_groups(groups: list[str], *, band: int = 3) -> dict[str, list[str]]:
    """Aggregate numbered per-layer groups (``conv_<k>``/``block_<k>``/``mlp_<k>``)
    into bands of ``band`` consecutive layers.

    Keeps the figure legible (and every curve visually distinct — the categorical
    style cycle repeats after 8 series) while preserving the early/mid/late depth
    reading: e.g. six ``spatial.conv_0..5`` become ``spatial.conv_0-2`` and
    ``spatial.conv_3-5``; a ``band`` at least the layer count merges them all into
    one ``spatial.conv_0-5`` group. Bands are labeled by the actual layer range
    they contain. Non-numbered groups map to themselves. Returns
    ``{display_group: [source groups]}`` in first-encounter order.
    """
    import re

    pattern = re.compile(r"^(?P<stem>.*(?:conv|block|mlp)_)(?P<k>\d+)$")
    by_slot: dict[tuple[str, int], list[tuple[int, str]]] = {}
    order: list = []  # display slots in first-encounter order
    for g in groups:
        m = pattern.match(g)
        if m is None:
            if ("plain", g) not in by_slot:
                order.append(("plain", g))
            by_slot.setdefault(("plain", g), []).append((0, g))
            continue
        k = int(m.group("k"))
        slot = (m.group("stem"), (k // band) * band)
        if slot not in by_slot:
            order.append(slot)
        by_slot.setdefault(slot, []).append((k, g))
    banded: dict[str, list[str]] = {}
    for slot in order:
        members = sorted(by_slot[slot])
        if slot[0] == "plain":
            banded[slot[1]] = [g for _, g in members]
            continue
        lo, hi = members[0][0], members[-1][0]
        label = members[0][1] if lo == hi else f"{slot[0]}{lo}-{hi}"
        banded[label] = [g for _, g in members]
    return banded


def plot_gradient_dynamics(
    frame: "pl.DataFrame",
    *,
    switch_stage_index: Optional[int] = None,
    smooth: int = 9,
    layer_band: int = 3,
    band_reduce: str = "rss",
):
    """Per-layer-group gradient norms over the training steps.

    One line per layer group (log scale, light moving-average smoothing over
    ``smooth`` samples), a vertical line at every curriculum stage boundary,
    and a highlighted line at the snapshots→sequences switch when
    ``switch_stage_index`` is given.

    Numbered per-layer groups are aggregated into bands of ``layer_band``
    consecutive layers, so a deep model stays legible and no two curves share
    an identical color+linestyle pair; ``layer_band=1`` disables aggregation,
    a band at least the layer count merges them all into one curve.
    ``band_reduce`` combines the member norms per step: ``"rss"``
    (root-sum-square — the L2 norm of the union of the band's parameters) or
    ``"mean"`` (the average per-layer norm, the typical layer's magnitude).
    """
    import matplotlib.pyplot as plt

    if band_reduce not in ("rss", "mean"):
        raise ValueError(f"band_reduce must be 'rss'|'mean', got {band_reduce!r}.")
    raw_groups = list(dict.fromkeys(frame["group"].to_list()))
    banded = _banded_layer_groups(raw_groups, band=max(1, layer_band))
    groups = list(banded)
    import polars as pl

    with paper_style(columns=2):
        fig, ax = plt.subplots(figsize=figure_size(columns=2, aspect=0.42))
        for i, group in enumerate(groups):
            members = banded[group]
            # combine member norms per step: RSS = the L2 norm of the union of the
            # band's parameters; mean = the typical member layer's norm.
            reducer = (
                pl.col("grad_norm").pow(2).sum().sqrt()
                if band_reduce == "rss"
                else pl.col("grad_norm").mean()
            )
            sub = (
                frame.filter(frame["group"].is_in(members))
                .group_by("step")
                .agg(reducer.alias("grad_norm"))
                .sort("step")
            )
            steps = sub["step"].to_numpy()
            norms = sub["grad_norm"].to_numpy().astype(np.float64)
            if smooth > 1 and norms.size > smooth:
                kernel = np.ones(smooth) / smooth
                norms = np.convolve(norms, kernel, mode="valid")
                steps = steps[smooth - 1 :]
            color, ls = series_style(i)
            ax.plot(
                steps,
                np.maximum(norms, 1e-12),
                label=group,
                color=color,
                linestyle=ls,
                linewidth=1.0,
            )
        for si in sorted(set(frame["stage_index"].to_list())):
            first = int(frame.filter(frame["stage_index"] == si)["step"].min())
            is_switch = switch_stage_index is not None and si == switch_stage_index
            ax.axvline(
                first,
                color="tab:red" if is_switch else "0.5",
                linewidth=1.0 if is_switch else 0.6,
                linestyle="-" if is_switch else ":",
            )
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("gradient L2 norm")
        ax.grid(True, alpha=0.3)
        # legend outside the axes — the curves fill the full step range
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.005, 0.5))
    return fig


def plot_weight_deltas(frame: "pl.DataFrame"):
    """Heatmap of relative weight motion: layer group x curriculum stage."""
    import matplotlib.pyplot as plt
    from pgml.evaluation.style import positive_log_norm

    groups = list(dict.fromkeys(frame["group"].to_list()))
    stages = list(dict.fromkeys(frame["stage"].to_list()))
    matrix = np.zeros((len(groups), len(stages)))
    for row in frame.iter_rows(named=True):
        matrix[groups.index(row["group"]), stages.index(row["stage"])] = row[
            "rel_delta"
        ]

    with paper_style(columns=1):
        fig, ax = plt.subplots(
            figsize=figure_size(columns=1, aspect=0.16 + 0.07 * len(groups))
        )
        im = ax.imshow(
            matrix, aspect="auto", cmap="magma", norm=positive_log_norm(matrix)
        )
        ax.set_xticks(range(len(stages)), stages, rotation=30, ha="right")
        ax.set_yticks(range(len(groups)), groups)
        ax.set_xlabel("curriculum stage")
        fig.colorbar(im, ax=ax, label="relative weight change", fraction=0.05)
    return fig


def plot_binned_errors(panels: list[tuple[str, Optional["pl.DataFrame"]]]):
    """Binned unobserved pu-MAE vs physical scenario features (one panel each).

    ``panels`` is ``[(xlabel, table)]`` with tables from :func:`binned_errors`
    (``None`` marks a feature that could not be binned, e.g. constant across
    the dataset — the panel says so). Fundamental and harmonic-band errors are
    drawn separately (log scale). Bins are equal-population (quantile) bins, so
    the per-bin sample count is stated once in each panel's title rather than
    repeated at every marker (as a range when duplicate-edge collapse makes the
    populations differ).
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    with paper_style(columns=2 if n > 1 else 1):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figure_size(columns=2 if n > 1 else 1, aspect=0.42 * nrows),
            squeeze=False,
        )
        for k, (xlabel, table) in enumerate(panels):
            ax = axes[k // ncols][k % ncols]
            if table is None or table.height == 0:
                ax.text(
                    0.5,
                    0.5,
                    "feature constant\n(no bins)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.4",
                )
                ax.set_xlabel(xlabel)
                continue
            centers = table["center"].to_numpy()
            ax.plot(
                centers,
                table["mae_fund_mean"].to_numpy(),
                marker="o",
                label="fundamental",
                color="0.25",
            )
            harm = table["mae_harm_mean"].to_numpy()
            if np.isfinite(harm).any():
                ax.plot(centers, harm, marker="s", label="harmonics", color="0.65")
            ns = table["n"].to_list()
            n_note = (
                f"n={ns[0]}/bin" if len(set(ns)) == 1 else f"n={min(ns)}–{max(ns)}/bin"
            )
            ax.set_title(n_note, loc="right", fontsize=6, color="0.4")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("unobserved pu-MAE")
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.legend()
        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
    return fig


# =============================================================================
# One-call analysis of a finished run
# =============================================================================
def _load_grid(out_dir: Path, grid):
    """Return ``grid`` if given, else the serialized grid the run wrote (``grid.json``)."""
    if grid is not None:
        return grid
    gp = out_dir / "grid.json"
    if gp.exists():
        from pgml.schemas.grid_schema import Grid

        return Grid.model_validate_json(gp.read_text())
    raise FileNotFoundError(
        f"analyze_run needs the grid the run trained on: pass grid=... or place a "
        f"serialized grid at {gp} (the experiment writes one)."
    )


def _stage_checkpoints(manifest: dict) -> list[Path]:
    """The run's stage checkpoints in curriculum order (existing files only).

    A resumed run records its source checkpoint as ``init_checkpoint`` (copied into
    the run's checkpoint dir) — prepended here so the first TRAINED stage still has a
    weight-delta baseline.
    """
    ckpt_dir = Path(manifest["final_checkpoint"]).parent
    names = [
        manifest["stages"][key]["stage_name"]
        for key in sorted(manifest["stages"], key=int)
    ]
    paths = [ckpt_dir / f"stage_{name}.ckpt" for name in names]
    init_name = manifest.get("init_checkpoint")
    if init_name and (ckpt_dir / init_name) not in paths:
        paths.insert(0, ckpt_dir / init_name)
    return [p for p in paths if p.exists()]


def _random_stage_checkpoint(
    manifest: dict,
) -> tuple["Optional[Path]", "Optional[float]"]:
    """The DEEPEST random-masking stage's checkpoint + its observed fraction.

    Node importance is read from the model trained to reconstruct from ARBITRARY sensor
    subsets (the last ``random`` curriculum stage), not the placement-finetuned final model:
    once finetuned to one fixed placement the nodes ADJACENT to those sensors dominate the
    saliency by construction, conflating "informative node" with "near the chosen sensors".
    The random-masking model has no such bias. Returns ``(None, None)`` when the curriculum
    has no random stage or its checkpoint is missing (the caller then falls back to the final
    model + the fixed placement).
    """
    ckpt_dir = Path(manifest["final_checkpoint"]).parent
    chosen = None
    for key in sorted(manifest.get("stages", {}), key=int):
        st = manifest["stages"][key]
        if st.get("kind") == "random":
            chosen = st
    if chosen is None:
        return None, None
    path = ckpt_dir / f"stage_{chosen['stage_name']}.ckpt"
    if not path.exists():
        return None, None
    return path, chosen.get("observed_fraction")


def _prior_fit_batch(
    source,
    *,
    grid: "Grid",
    measurement,
    n_states: int,
    device,
):
    """``n_states`` TRAIN states of ``source`` at the fixed placement: ``(v [S, H, N], mask)``.

    Scenarios are drawn seeded (the :func:`evaluation_batch` draw); a sequence contributes
    every step, so ``ceil(n_states / T)`` sequences are drawn and the states truncated.
    """
    import math

    steps = max(1, int(source.n_steps))
    v, mask, _ = evaluation_batch(
        source,
        grid=grid,
        measurement=measurement,
        split="train",
        n_samples=math.ceil(n_states / steps),
        device=device,
    )
    if v.dim() == 4:
        mask = mask.unsqueeze(1).expand(-1, v.shape[1], -1).reshape(-1, mask.shape[-1])
        v = v.reshape(-1, *v.shape[-2:])
    return v[:n_states], mask[:n_states]


def _write_band_ablations(
    module,
    analysis_dir: Path,
    *,
    grid: "Grid",
    measurement,
    base_v,
    eval_source,
    eval_dataset: tuple[str, str],
    random_source,
    random_dataset: str,
    snapshot_batch,
    n_band_samples: int,
    n_prior_samples: int,
    device,
) -> None:
    """The band ablation on the EVALUATION data, plus its snapshot-validation counterpart.

    Writes ``estimator_band_ablation.csv`` (the evaluation dataset's TEST split — its
    validation split when the run carved no test split), ``linear_prior.npz`` (the prior
    its ``linear_prior`` row decodes), ``estimator_band_ablation_snapshot_val.csv`` (the
    randomized snapshots' validation split, the attribution batch) and
    ``estimator_band_ablation.json`` naming the data of each file. The linear prior is the
    module's own when it carries one (a distilling / prior-residual run: the map it was
    trained against), otherwise a fit on ``n_prior_samples`` TRAIN states of the dataset
    the table scores — never on the scored samples.
    """
    teacher = getattr(module, "injection_prior", None)
    carried = None if teacher is None else teacher.prior  # one bundle for both tables
    fitted: dict[str, object] = {}

    def prior_for(source, path: str):
        if carried is not None:
            return carried
        if path not in fitted:
            v_fit, mask_fit = _prior_fit_batch(
                source,
                grid=grid,
                measurement=measurement,
                n_states=n_prior_samples,
                device=device,
            )
            fitted[path] = fit_linear_injection_prior(
                module, v_fit, mask_fit, device=device
            ).with_fit_dataset(f"{path}:train")
        return fitted[path]

    kind, eval_path = eval_dataset
    split = "test" if len(eval_source.test_dataset()) else "val"
    v_e, mask_e, time_e = evaluation_batch(
        eval_source,
        grid=grid,
        measurement=measurement,
        split=split,
        n_samples=n_band_samples,
        device=device,
    )
    prior = prior_for(eval_source, str(eval_path))
    prior.save(analysis_dir / "linear_prior.npz")
    estimator_band_ablation(
        module,
        v_e,
        mask_e,
        base_v=base_v,
        device=device,
        linear_prior=prior,
        time_unix_s=time_e,
    ).write_csv(analysis_dir / "estimator_band_ablation.csv")

    v_s, mask_s = snapshot_batch
    snap_prior = prior_for(random_source, str(random_dataset))
    if snap_prior is not prior:
        snap_prior.save(analysis_dir / "linear_prior_snapshot_train.npz")
    estimator_band_ablation(
        module, v_s, mask_s, base_v=base_v, device=device, linear_prior=snap_prior
    ).write_csv(analysis_dir / "estimator_band_ablation_snapshot_val.csv")

    def scored(dataset_kind, path, split_name, v_batch):
        steps = int(v_batch.shape[1]) if v_batch.dim() == 4 else 1
        return {
            "dataset": dataset_kind,
            "dataset_path": str(path),
            "split": split_name,
            "n_samples": int(v_batch.shape[0]),
            "n_steps": steps,
            "n_states": int(v_batch.shape[0]) * steps,
        }

    def prior_record(p, file_name):
        return {
            "file": file_name,
            "origin": "checkpoint" if teacher is not None else "fitted",
            "fit_dataset": p.fit_dataset,
            "n_train": int(p.n_train),
            "noisy": bool(p.noisy),
        }

    record = {
        "estimator_band_ablation.csv": {
            **scored(kind, eval_path, split, v_e),
            "linear_prior": prior_record(prior, "linear_prior.npz"),
        },
        "estimator_band_ablation_snapshot_val.csv": {
            **scored("random", random_dataset, "val", v_s),
            "linear_prior": prior_record(
                snap_prior,
                "linear_prior.npz"
                if snap_prior is prior
                else "linear_prior_snapshot_train.npz",
            ),
        },
    }
    (analysis_dir / "estimator_band_ablation.json").write_text(
        json.dumps(record, indent=2)
    )


def analyze_run(
    out_dir,
    *,
    grid: "Optional[Grid]" = None,
    dataset: str = "coherent",
    device: Union[str, "torch.device"] = "cpu",  # noqa: F821 - torch imported lazily
    ig_steps: int = 32,
    n_attribution_samples: int = 64,
    n_prior_samples: int = 2000,
    n_band_samples: int = 64,
    bins: int = 5,
    geo_positions: Optional[Union[str, Path, dict]] = None,
) -> Path:
    """Run the full interpretability analysis; write PDF+SVG figures to ``out_dir/analysis/``.

    Figures (each ``<stem>.pdf`` + ``<stem>.svg``):

    - ``attr_order_importance`` — integrated-gradients mass per input order;
    - ``occlusion_orders`` — leave-one-frequency-out: the estimation-error
      increase when each input order is removed from the sensors
      (+ ``occlusion_orders.csv``) — the direct "which frequency does the model
      need" reading, complementing the gradient attribution above;
    - ``attr_harmonic_coupling`` — output-order x input-order coupling of the
      NETWORK alone (gradient + occlusion panels, row-normalized; for a
      current-measuring module both probe the model with its measured-current
      inputs present and occlude voltage and current jointly), plus
      ``attr_harmonic_coupling_marginal`` — the same panels with an absolute
      row-mass marginal column (how much attribution each output order receives
      at all, which row normalization alone hides);
    - ``estimator_order_coupling`` — the same leave-one-frequency-out through
      the DEPLOYED decode (the order removed from the model inputs AND the
      anchor measurements): state shift + signed Δ pu-MAE per output order;
      ``estimator_band_ablation.csv`` (injection decode) additionally splits
      the information sources — baseline / fundamental-only / harmonic-only
      sensors, anchors-only (zero injections), model-only (anchors off), and
      ``linear_prior``: the same anchored decode fed the closed-form
      least-squares estimate of the injections from the SAME measurements
      (:func:`~injection_hse.analysis.fit_linear_injection_prior`, fitted on
      ``n_prior_samples`` TRAIN states of the scored dataset and recorded as
      ``linear_prior.npz``) — the reference the deployed row has to beat to have
      learned anything a regression on its own inputs could not. It is scored on
      the EVALUATION data: ``n_band_samples`` scenarios of the ``dataset``'s
      TEST split (its validation split when the run carved no test split), so
      for a two-task run on the sequences the final model is deployed on, read
      as sequences by the network (see
      :func:`~injection_hse.analysis.estimator_band_ablation`).
      ``estimator_band_ablation_snapshot_val.csv`` repeats it on the
      attribution batch (randomized snapshots, validation split), and
      ``estimator_band_ablation.json`` records the dataset, split and sample
      counts of each file and where its prior came from. A run that carries a
      fitted prior in its checkpoint (prior distillation, the prior-residual
      decode) is scored against that map instead of a second fit;
    - ``attr_node_importance`` — grid graph coloured by saliency mass per node
      (where a measurement WOULD help), sensors outlined. Read from the deepest
      random-masking model under random placements, so the map is not biased
      toward the final model's fixed sensor set. ``geo_positions`` (a path to a
      position JSON, or a mapping accepted by
      :func:`pgml.evaluation.graph_layout`) draws the grid at explicit
      geographic coordinates instead of the automatic layout;
    - ``occlusion_sensors`` — error increase per dropped sensor / pair
      (+ ``occlusion_sensors.csv``);
    - ``training_grad_dynamics`` — per-layer gradient norms (only when the run
      wrote ``gradient_norms.parquet`` via the workflow's
      ``--collect-gradients``);
    - ``training_weight_deltas`` — per-layer relative weight motion between
      stage checkpoints (any run with >= 2 stage checkpoints);
    - ``error_vs_scenario`` — binned unobserved error vs THD / VUF / relative
      loading / voltage deviation (+ ``binned_<feature>.csv``), plus each
      feature as its own single-column ``error_vs_<feature>`` figure;
    - ``error_vs_load_scale`` — binned error vs the sampled load scale of the
      randomized dataset (when sampled inputs are persisted).

    Attribution, order occlusion, the estimator order coupling and the physical
    binning always run on the RANDOMIZED snapshot dataset — the one whose
    scenarios span operating conditions (the coherent sequences keep the
    fundamental operating point nominal by design). ``dataset`` selects the
    EVALUATION data — the band ablation and the SENSOR occlusion
    (``"coherent"`` default = the deployment-like sequences, or ``"random"``).
    Every dataset consumer is built with the module's own ``cfg.data`` split
    settings, so the sample rows of the prediction, the attribution batch and
    the sampled inputs align. ``grid`` is reloaded from ``grid.json`` when
    omitted. Returns the
    ``out_dir/analysis`` directory.
    """
    from injection_hse.config import MeasurementConfig
    from injection_hse.data import build_data_source
    from injection_hse.evaluation import load_se_module, predict_dataset

    out_dir = Path(out_dir)
    grid = _load_grid(out_dir, grid)
    manifest = json.loads((out_dir / _MANIFEST).read_text())
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    key = "dataset_coherent" if dataset == "coherent" else "dataset_random"
    eval_dataset_path = manifest["datasets"][key]
    random_path = manifest["datasets"]["dataset_random"]
    sensor_names = manifest["sensor_names"]
    orders = list(manifest["orders"])
    measurement = MeasurementConfig(voltage_nodes=sensor_names, slack_known=True)

    module = load_se_module(
        manifest["final_checkpoint"],
        grid=grid,
        dataset_path=eval_dataset_path,
        device=device,
    )

    def _source_for(path):
        # the TRAINING run's split settings, so every consumer below (prediction,
        # attribution, sampled inputs) sees one identical train/val partition.
        cfg = module.cfg.data.model_copy(
            update={"source": "parquet", "dataset_path": str(path)}
        )
        return build_data_source(cfg, grid=grid)

    eval_source = _source_for(eval_dataset_path)
    random_source = (
        eval_source
        if str(random_path) == str(eval_dataset_path)
        else _source_for(random_path)
    )

    # -- attribution + order occlusion (randomized snapshots) ------------------
    v, mask = attribution_batch(
        random_source,
        grid=grid,
        measurement=measurement,
        split="val",
        n_samples=n_attribution_samples,
        device=device,
    )
    attr_ig = integrated_gradients(
        module, v, mask, grid=grid, steps=ig_steps, device=device
    )
    fig = plot_order_importance(orders, order_importance(attr_ig))
    save_paper_figure(fig, analysis_dir, "attr_order_importance")

    coupling = harmonic_coupling(module, v, mask, grid=grid, device=device)
    occl = order_occlusion(module, v, mask, device=device)
    fig = plot_harmonic_coupling(orders, coupling, occl)
    save_paper_figure(fig, analysis_dir, "attr_harmonic_coupling")
    fig = plot_harmonic_coupling(orders, coupling, occl, row_mass=True)
    save_paper_figure(fig, analysis_dir, "attr_harmonic_coupling_marginal")

    # leave-one-frequency-out: how much worse the ESTIMATE gets without each input order
    from injection_hse.data import base_voltage_per_row

    base_v = base_voltage_per_row(grid)
    occ_orders = order_occlusion_error(module, v, mask, base_v=base_v, device=device)
    occ_orders.write_csv(analysis_dir / "occlusion_orders.csv")
    fig = plot_order_occlusion(occ_orders)
    save_paper_figure(fig, analysis_dir, "occlusion_orders")

    # estimator-level coupling + band split (the deployed decode, anchors included)
    shift, delta_mae, _ = estimator_order_coupling(
        module, v, mask, base_v=base_v, device=device
    )
    fig = plot_estimator_coupling(orders, shift, delta_mae)
    save_paper_figure(fig, analysis_dir, "estimator_order_coupling")
    if getattr(module, "_predict_injection_state", False):
        _write_band_ablations(
            module,
            analysis_dir,
            grid=grid,
            measurement=measurement,
            base_v=base_v,
            eval_source=eval_source,
            eval_dataset=(dataset, eval_dataset_path),
            random_source=random_source,
            random_dataset=random_path,
            snapshot_batch=(v, mask),
            n_band_samples=n_band_samples,
            n_prior_samples=n_prior_samples,
            device=device,
        )

    # node importance ("where WOULD a measurement help") is read from the deepest
    # random-masking model probed with RANDOM placements — NOT the placement-finetuned
    # final model on its fixed sensors, which biases the saliency toward the chosen
    # sensors' neighbourhood. Fall back to the final model + fixed placement when the run
    # has no random stage.
    node_ckpt, node_fraction = _random_stage_checkpoint(manifest)
    if node_ckpt is not None:
        node_module = load_se_module(
            str(node_ckpt), grid=grid, dataset_path=random_path, device=device
        )
        v_node, mask_node = attribution_batch(
            random_source,
            grid=grid,
            measurement=measurement,
            split="val",
            n_samples=n_attribution_samples,
            observed_fraction=node_fraction,
            device=device,
        )
    else:
        node_module, v_node, mask_node = module, v, mask
    attr_sal = saliency(node_module, v_node, mask_node, grid=grid, device=device)
    if geo_positions is not None and not isinstance(geo_positions, dict):
        from pgml.evaluation import load_node_positions

        geo_positions = load_node_positions(geo_positions, grid)
    fig = plot_node_importance(
        grid,
        node_importance(attr_sal),
        sensor_names=sensor_names,
        positions=geo_positions,
    )
    save_paper_figure(fig, analysis_dir, "attr_node_importance")

    # -- sensor occlusion (evaluation dataset) ----------------------------------
    occl_frame = sensor_occlusion(
        module,
        eval_source,
        grid=grid,
        sensor_names=sensor_names,
        device=device,
    )
    occl_frame.write_csv(analysis_dir / "occlusion_sensors.csv")
    fig = plot_sensor_occlusion(occl_frame)
    save_paper_figure(fig, analysis_dir, "occlusion_sensors")

    # -- training dynamics ------------------------------------------------------
    gn_path = out_dir / _GRADIENT_NORMS
    if gn_path.exists():
        norms = read_gradient_norms(gn_path)
        fig = plot_gradient_dynamics(
            norms, switch_stage_index=manifest.get("switch_stage_index")
        )
        save_paper_figure(fig, analysis_dir, "training_grad_dynamics")
        # compact variant: ALL numbered layers of a stem merged into one curve
        # (mean member norm) — fewer lines for a space-constrained figure slot
        fig = plot_gradient_dynamics(
            norms,
            switch_stage_index=manifest.get("switch_stage_index"),
            layer_band=64,
            band_reduce="mean",
        )
        save_paper_figure(fig, analysis_dir, "training_grad_dynamics_compact")

    ckpts = _stage_checkpoints(manifest)
    if len(ckpts) >= 2:
        fig = plot_weight_deltas(checkpoint_weight_deltas(ckpts))
        save_paper_figure(fig, analysis_dir, "training_weight_deltas")

    # -- physical binning (randomized dataset — the one spanning operating
    # conditions: the coherent data keeps the fundamental P/Q nominal by design,
    # so VUF / loading / voltage deviation are constant there) -------------------
    prediction = predict_dataset(
        module,
        source_or_path=random_source,
        grid=grid,
        measurement=measurement,
        split="val",
        device=device,
    )
    features = scenario_features(prediction, device=device)
    min_count = 10 if prediction.n_samples >= bins * 10 else 1
    panels: list[tuple[str, Optional[object]]] = []
    for column, xlabel in (
        ("thd_max_pct", "max THD [%]"),
        ("vuf_max_pct", "max VUF [%]"),
        ("loading_rel_max", "relative line loading (p95-norm.)"),
        ("vdev_max_pct", "max |V| deviation [%]"),
    ):
        try:
            table = binned_errors(
                prediction, features, by=column, bins=bins, min_count=min_count
            )
            table.write_csv(analysis_dir / f"binned_{column}.csv")
        except ValueError:
            table = None
        panels.append((xlabel, table))
        if table is not None:
            # each feature also as its own single-column figure, so a paper can
            # include one panel (e.g. THD only) without cropping the 4-panel figure
            fig = plot_binned_errors([(xlabel, table)])
            save_paper_figure(fig, analysis_dir, f"error_vs_{column}")
    fig = plot_binned_errors(panels)
    save_paper_figure(fig, analysis_dir, "error_vs_scenario")

    # -- binned error vs the sampled inputs (same randomized prediction) ---------
    try:
        sf = sample_features(random_source, grid=grid, module=module)
        feature_cols = [c for c in sf.columns if c not in ("sample", "scenario")]
        load_cols = [c for c in feature_cols if "load" in c] or feature_cols
        if load_cols:
            column = load_cols[0]
            table = binned_errors(
                prediction, sf, by=column, bins=bins, min_count=min_count
            )
            table.write_csv(analysis_dir / f"binned_{column}.csv")
            fig = plot_binned_errors([(column.replace("_", " "), table)])
            save_paper_figure(fig, analysis_dir, "error_vs_load_scale")
    except ValueError:
        pass  # sequence dataset or constant feature — the figure is optional

    return analysis_dir


__all__ = [
    "plot_order_importance",
    "plot_order_occlusion",
    "plot_harmonic_coupling",
    "plot_node_importance",
    "plot_sensor_occlusion",
    "plot_gradient_dynamics",
    "plot_weight_deltas",
    "plot_binned_errors",
    "analyze_run",
]
