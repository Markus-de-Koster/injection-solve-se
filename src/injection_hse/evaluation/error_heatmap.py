"""Per-order x per-node-phase error heatmap (where, and at which harmonic, the SE struggles).

A heatmap of the per-unit ``|error|`` reduced over the evaluation samples: harmonic ORDER on
the x-axis, node-phase ROW on the y-axis. The fundamental error is typically orders of
magnitude larger than the harmonic errors, so a single shared colour scale renders the
harmonics almost flat (black). The plot therefore SPLITS the fundamental (``h1``) into its own
panel with its own colour scale + colour bar on the LEFT, a small gap, then the harmonics
(``h>1``) in a second panel on their own (log) scale + colour bar on the RIGHT. Sensor rows are
highlighted so the spatial error pattern reads against the placement. matplotlib SVG + a
plotly HTML version.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pgml.evaluation.style import positive_log_norm, save_figure, save_html
from pgml.schemas.grid_schema import Phase

from .inference import Prediction

_PHASE_CODE = {Phase.A: 0, Phase.B: 1, Phase.C: 2, Phase.N: 3}


def _reduce(err_shn: np.ndarray, statistic: str) -> np.ndarray:
    """Reduce ``[S, H, N]`` over samples to ``[H, N]`` by ``mean`` or ``max``."""
    if statistic == "mean":
        return err_shn.mean(axis=0)
    if statistic == "max":
        return err_shn.max(axis=0)
    raise ValueError(f"statistic must be 'mean' or 'max', got {statistic!r}.")


_PHASE_LETTER = ("a", "b", "c", "n")


def _row_selection(prediction: Prediction, phase: Optional[Phase]):
    """Row indices (and their labels) to display, optionally filtered to one phase.

    Labels are built from the prediction's OWN ``node_ids``/``phase_codes`` (matching
    :func:`pgml.evaluation.data.row_labels`), so the heatmap is correct whether the prediction
    covers the full grid or a subset of node-phase rows (e.g. only the metered rows of a field
    deployment). Nodes are shown by their zero-based display number
    (:func:`pgml.evaluation.node_numbering` — the source tool's bus numbering for a
    converted grid), not the internal node id."""
    from pgml.evaluation import node_numbering

    numbering = node_numbering(prediction.grid)
    labels = [
        f"{numbering.get(int(nid), int(nid))}·{_PHASE_LETTER[int(pc)]}"
        for nid, pc in zip(prediction.node_ids, prediction.phase_codes)
    ]
    if phase is None:
        rows = np.arange(prediction.n_rows)
    else:
        want = _PHASE_CODE[phase]
        rows = np.flatnonzero(prediction.phase_codes == want)
    return rows, [labels[int(r)] for r in rows]


def _mark_sensor_yticks(ax, sensor_mask: np.ndarray) -> None:
    """Colour the sensor rows' y-tick labels red + add a left marker bar (matplotlib)."""
    for i, is_sensor in enumerate(sensor_mask):
        if is_sensor and i < len(ax.get_yticklabels()):
            ax.get_yticklabels()[i].set_color("tab:red")
            ax.get_yticklabels()[i].set_fontweight("bold")
    if sensor_mask.any():
        ax.scatter(
            np.full(int(sensor_mask.sum()), -0.6),
            np.flatnonzero(sensor_mask),
            marker="<",
            color="tab:red",
            s=30,
            clip_on=False,
            zorder=5,
            label="sensor",
        )


def _harmonic_norm(harm: np.ndarray, harmonic_log: bool):
    """A ``LogNorm`` spanning the positive harmonic errors (``None`` = linear scale).

    Delegates to :func:`pgml.evaluation.style.positive_log_norm` — NaN entries
    (empty statistic buckets) are ignored, never propagated into the scale.
    """
    if not harmonic_log:
        return None
    return positive_log_norm(harm)


def plot_error_heatmap(
    prediction: Prediction,
    *,
    statistic: str = "mean",
    phase: Optional[Phase] = None,
    out_svg: Optional[str] = None,
    out_html: Optional[str] = None,
    fundamental_cmap: str = "viridis",
    harmonic_cmap: str = "magma",
    harmonic_log: bool = True,
    title: Optional[str] = None,
):
    """Heatmap of the per-unit ``|error|`` per harmonic order (x) and node-phase row (y).

    The fundamental (``h1``) is drawn in its OWN panel with its own colour scale and a colour
    bar on the LEFT; the harmonics (``h>1``) follow in a second panel (a small gap between) on
    their own scale (``harmonic_log`` → a log colour scale, the default) with a colour bar on
    the RIGHT — so the harmonic error pattern is visible instead of being swamped by the much
    larger fundamental error. The two colour bars sit in equal-width columns, so the narrow
    fundamental panel's bar is not drawn thinner. ``statistic`` reduces over the evaluation
    samples (``"mean"`` typical error / ``"max"`` worst case). ``phase`` (``None`` = all rows) filters the y-axis to
    one phase's rows. Sensor rows are highlighted (red tick + a marker). ``title`` (default
    ``None``) draws NO figure title, so the matplotlib figure is caption-ready for external
    typesetting; pass a string to add one (the per-panel ``fundamental`` / ``harmonics`` labels
    are always drawn). The matplotlib figure is written to ``out_svg`` (paper-ready) and a
    plotly HTML to ``out_html``; the matplotlib ``Figure`` is returned. (When the orders contain
    no fundamental or no harmonics, a single combined panel is drawn instead.)
    """
    import matplotlib.pyplot as plt

    err = _reduce(prediction.abs_error_pu(), statistic)  # [H, N]
    rows, labels = _row_selection(prediction, phase)
    mat = err[:, rows].T  # [n_rows_shown, H]
    orders = list(prediction.orders)
    obs = set(int(r) for r in prediction.observed_rows)
    sensor_mask = np.array([int(r) in obs for r in rows])

    fund_cols = [i for i, o in enumerate(orders) if o == 1]
    harm_cols = [i for i, o in enumerate(orders) if o != 1]
    split = bool(fund_cols) and bool(harm_cols)

    height = max(4.0, 0.16 * len(rows) + 1.5)

    if not split:  # nothing to separate — one combined panel
        fig, ax = plt.subplots(
            figsize=(max(6.0, 0.5 * len(orders) + 3.0), height),
            constrained_layout=True,
        )
        im = ax.imshow(mat, aspect="auto", cmap=harmonic_cmap, origin="lower")
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels([f"h{o}" for o in orders])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=6)
        _mark_sensor_yticks(ax, sensor_mask)
        if sensor_mask.any():
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel("Harmonic order")
        ax.set_ylabel("Node-phase row")
        if title is not None:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, label="|error| [pu]", pad=0.02)
        if out_svg is not None:
            save_figure(fig, out_svg)
        if out_html is not None:
            _plot_error_heatmap_plotly(
                prediction,
                mat,
                orders,
                fund_cols,
                harm_cols,
                rows,
                labels,
                sensor_mask,
                statistic,
                fundamental_cmap,
                harmonic_cmap,
                harmonic_log,
                out_html,
            )
        return fig

    fund = mat[:, fund_cols]  # [n_rows, 1]
    harm = mat[:, harm_cols]  # [n_rows, n_harm]

    width = max(8.0, 0.5 * len(harm_cols) + 5.0)
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    # DEDICATED, equal-width colour-bar columns flank the two panels. A colour bar auto-sized
    # to its parent axes inherits that axes' width, so the fundamental bar (its panel is one
    # column wide) would render far thinner than the harmonic bar in the SVG backend; explicit
    # equal-width cax columns make the two bars the same size.
    gs = fig.add_gridspec(
        1, 5, width_ratios=[0.12, 1.0, 0.18, max(len(harm_cols), 1), 0.12]
    )
    cax_f = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[0, 3], sharey=ax_f)
    cax_h = fig.add_subplot(gs[0, 4])

    im_f = ax_f.imshow(fund, aspect="auto", cmap=fundamental_cmap, origin="lower")
    norm = _harmonic_norm(harm, harmonic_log)
    im_h = ax_h.imshow(
        harm, aspect="auto", cmap=harmonic_cmap, origin="lower", norm=norm
    )

    # fundamental panel: the y-axis (row labels + sensor highlight) lives here
    ax_f.set_xticks([0])
    ax_f.set_xticklabels(["h1"])
    ax_f.set_yticks(range(len(rows)))
    ax_f.set_yticklabels(labels, fontsize=6)
    _mark_sensor_yticks(ax_f, sensor_mask)
    ax_f.set_ylabel("Node-phase row")
    ax_f.set_title("fundamental")

    # harmonic panel: its own x-ticks; share the y-axis (hide its labels)
    ax_h.set_xticks(range(len(harm_cols)))
    ax_h.set_xticklabels([f"h{orders[i]}" for i in harm_cols])
    plt.setp(ax_h.get_yticklabels(), visible=False)
    ax_h.set_title("harmonics" + (" (log scale)" if norm is not None else ""))

    fig.colorbar(im_f, cax=cax_f, label="h1 |error| [pu]")
    cax_f.yaxis.set_ticks_position("left")
    cax_f.yaxis.set_label_position("left")
    fig.colorbar(im_h, cax=cax_h, label="h>1 |error| [pu]")
    if sensor_mask.any():
        # the sensor key sits in the top-left corner — above the fundamental colour bar —
        # so it never overlaps the (narrow) fundamental panel itself.
        handles, handle_labels = ax_f.get_legend_handles_labels()
        fig.legend(
            handles,
            handle_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=7,
        )
    if title is not None:
        fig.suptitle(title)
    if out_svg is not None:
        save_figure(fig, out_svg)

    if out_html is not None:
        _plot_error_heatmap_plotly(
            prediction,
            mat,
            orders,
            fund_cols,
            harm_cols,
            rows,
            labels,
            sensor_mask,
            statistic,
            fundamental_cmap,
            harmonic_cmap,
            harmonic_log,
            out_html,
        )
    return fig


def _plot_error_heatmap_plotly(
    prediction,
    mat,
    orders,
    fund_cols,
    harm_cols,
    rows,
    labels,
    sensor_mask,
    statistic,
    fundamental_cmap,
    harmonic_cmap,
    harmonic_log,
    out_html,
) -> None:
    """The interactive split heatmap (plotly HTML): h1 panel + h>1 panel, separate colour bars."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    y_labels = [
        f"<b style='color:red'>{lab}</b>" if sensor_mask[i] else lab
        for i, lab in enumerate(labels)
    ]
    if not fund_cols or not harm_cols:  # single combined panel
        fig = go.Figure(
            go.Heatmap(
                z=mat,
                x=[f"h{o}" for o in orders],
                y=y_labels,
                colorscale=harmonic_cmap,
                colorbar=dict(title="|err| [pu]"),
            )
        )
        fig.update_layout(
            title=None,
            xaxis_title="Harmonic order",
            yaxis_title="Node-phase row",
            width=max(600, 40 * len(orders) + 300),
            height=max(500, 14 * len(rows) + 150),
        )
        save_html(fig, out_html)
        return

    fund = mat[:, fund_cols]
    harm = mat[:, harm_cols]
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[1.0, max(len(harm_cols), 1)],
        horizontal_spacing=0.08,
        shared_yaxes=True,
        subplot_titles=(
            "fundamental",
            "harmonics" + (" (log10)" if harmonic_log else ""),
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=fund,
            x=["h1"],
            y=y_labels,
            colorscale=fundamental_cmap,
            colorbar=dict(title="h1 |err| [pu]", x=-0.18, len=0.92),
        ),
        row=1,
        col=1,
    )
    # harmonics on a log10 colour scale (errors are >= 0; clip to the positive floor)
    if harmonic_log:
        pos = harm[np.isfinite(harm) & (harm > 0)]
        floor = float(pos.min()) if pos.size else 1e-12
        z_h = np.log10(np.clip(harm, floor, None))
        cb_title = "log10 h>1 |err|"
    else:
        z_h = harm
        cb_title = "h>1 |err| [pu]"
    fig.add_trace(
        go.Heatmap(
            z=z_h,
            x=[f"h{orders[i]}" for i in harm_cols],
            y=y_labels,
            colorscale=harmonic_cmap,
            colorbar=dict(title=cb_title, x=1.02, len=0.92),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=None,
        width=max(700, 40 * len(harm_cols) + 400),
        height=max(500, 14 * len(rows) + 160),
        margin=dict(l=140),
    )
    fig.update_yaxes(title_text="Node-phase row", row=1, col=1)
    save_html(fig, out_html)


__all__ = ["plot_error_heatmap"]
