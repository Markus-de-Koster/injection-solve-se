"""Diagnostic / insight plots over a :class:`~injection_hse.evaluation.inference.Prediction`.

Four lenses on an estimator's quality:

- :func:`plot_parity_per_harmonic` — predicted vs true ``|V_h|`` scatter per order (calibration);
- :func:`plot_error_by_order` — per-order pu-MAE, observed vs unobserved (where masking hurts);
- :func:`plot_error_vs_sensor_distance` — unobserved-node pu-error vs distance to the nearest
  sensor (does error grow away from a measurement?);
- :func:`plot_error_cdf` — the cumulative distribution of per-node pu-error (the tail), with
  :func:`worst_scenario` to pick the hardest sample for a state-vs-prediction overlay.

All consume the plain-numpy :class:`Prediction`; matplotlib SVG output (paper-ready).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from pgml.evaluation.style import save_figure
from pgml.evaluation.topology import grid_graph

from .inference import Prediction


def _sensor_node_ids(prediction: Prediction) -> list[int]:
    """The distinct node ids carrying a sensor (from the observed placement rows)."""
    obs = set(int(r) for r in prediction.observed_rows)
    nid_of_row = prediction.node_ids
    return sorted(set(int(nid_of_row[r]) for r in obs))


def _draw_parity(prediction: Prediction, show: list, max_points: int):
    """Draw the parity small-multiples for ``show`` orders; see :func:`plot_parity_per_harmonic`.

    Sized for print: the figure is rendered wide (one panel per order) and typically
    scaled to the full text width of a double-column page, so fonts are set LARGE in
    figure units to land at a legible size after downscaling, and each axis carries at
    most a few SHORT tick labels (scientific offset notation for the small harmonic
    magnitudes) so adjacent labels cannot collide on the narrow panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    all_orders = list(prediction.orders)
    idx_of = {o: i for i, o in enumerate(all_orders)}
    base = prediction.base_v[
        None, None, :
    ]  # [1, 1, N] per-row L-N base (pu denominator)
    true_mag = np.abs(prediction.target_v) / base  # [S, H, N] per-unit
    pred_mag = np.abs(prediction.pred_v) / base
    rng = np.random.default_rng(0)  # deterministic subsample

    n = len(show)
    # a single row keeps the per-order panels directly comparable; wrap only when the
    # order count would make the panels too narrow (e.g. the full 1..20 range).
    ncols = n if n <= 6 else 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for j, o in enumerate(show):
        ax = axes.flat[j]
        ax.set_visible(True)
        k = idx_of[o]
        x = true_mag[:, k, :].reshape(-1)
        y = pred_mag[:, k, :].reshape(-1)
        # axis limits from the FULL data — the ACTUAL range (min..max), not 0..max, so a tight
        # cloud away from zero (e.g. the fundamental ~1 pu) fills the panel instead of hugging a
        # corner. A small margin keeps the edge points off the frame.
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max(), lo + 1e-12))
        pad = 0.02 * (hi - lo)
        if x.size > max_points:  # down-sample for a legible, small figure
            sel = rng.choice(x.size, size=max_points, replace=False)
            x, y = x[sel], y[sel]
        ax.scatter(
            x, y, s=4, alpha=0.3, color="tab:blue", edgecolors="none", rasterized=True
        )
        ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"h{o}", fontsize=16)
        ax.set_xlabel("true |V| [pu]", fontsize=14)
        if j % ncols == 0:
            ax.set_ylabel("pred |V| [pu]", fontsize=14)
        # Few, short tick labels: side-by-side panels leave no horizontal room, so cap
        # the tick count and move the harmonic magnitudes' common 1e-k factor into a
        # single offset annotation instead of repeating long decimals on every tick.
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.ticklabel_format(
            style="sci", scilimits=(-2, 3), axis="both", useMathText=True
        )
        ax.xaxis.get_offset_text().set_fontsize(11)
        ax.yaxis.get_offset_text().set_fontsize(11)
        ax.grid(True, alpha=0.3)
    return fig


def plot_parity_per_harmonic(
    prediction: Prediction,
    *,
    orders: Optional[Sequence[int]] = None,
    out_svg: Optional[str] = None,
    max_points: int = 40000,
    out_png: Optional[str] = None,
    png_max_points: int = 1_000_000,
):
    """Predicted vs true PER-UNIT ``|V_h|`` scatter per harmonic order (a calibration check).

    One small-multiple panel per order (a single row up to six orders): each point is a
    (sample, node-phase row) pair, x = true ``|V_h|``, y = predicted ``|V_h|``, both in
    PER-UNIT (``|V|`` ÷ the per-row line-to-neutral base), with the ``y = x`` reference
    line. Per-unit is essential at the fundamental, where two voltage levels (e.g. MV vs
    LV nodes) otherwise split the cloud into disjoint volt bands that hide the
    calibration; in per-unit every node shares a ~1 pu scale. Tight scatter on the
    diagonal means a well-calibrated magnitude estimate at that order. ``orders``
    selects which orders to show (default all). Returns the matplotlib ``Figure``.

    A full dataset has ``samples * rows`` points PER ORDER (millions on a large run).
    The points are therefore RASTERIZED (so the vector SVG stays small regardless of
    count) and down-sampled to ``max_points`` per panel (a fixed-seed random subsample;
    the axis limits use the full data). ``out_png`` additionally writes a print-ready
    300-DPI raster whose per-panel budget is the much larger ``png_max_points`` (a PNG's
    size is bounded by its pixel count, not the point count), for the cases where the
    SVG subsample visibly thins dense structure — note that at the fundamental the
    remaining vertical banding is REAL data structure (each node-phase row spans only a
    narrow true-voltage band across scenarios), not a sampling artifact.
    """
    show = list(orders) if orders is not None else list(prediction.orders)
    fig = _draw_parity(prediction, show, max_points)
    if out_svg is not None:
        save_figure(fig, out_svg)
    if out_png is not None:
        import matplotlib.pyplot as plt

        fig_png = _draw_parity(prediction, show, png_max_points)
        save_figure(fig_png, out_png, dpi=300)
        plt.close(fig_png)
    return fig


def plot_error_by_order(prediction: Prediction, *, out_svg: Optional[str] = None):
    """Grouped bar of per-order pu-MAE: OBSERVED vs UNOBSERVED node-phase rows.

    For each harmonic order, the mean per-unit ``|error|`` over the observed rows and over the
    unobserved rows (split by the placement mask), side by side — quantifying how much the
    estimate degrades away from a sensor. Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    err = prediction.abs_error_pu()  # [S, H, N]
    obs = prediction.mask  # [S, N] True = observed
    obs_hsn = obs[:, None, :]  # broadcast over H
    # masked means over (sample, row) per order
    obs_w = np.broadcast_to(obs_hsn, err.shape)
    unobs_w = ~obs_w
    obs_mae = _masked_mean(err, obs_w)  # [H]
    unobs_mae = _masked_mean(err, unobs_w)  # [H]

    orders = prediction.orders
    x = np.arange(len(orders))
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.6 * len(orders) + 2), 4.2), constrained_layout=True
    )
    ax.bar(x - 0.2, obs_mae, width=0.4, color="tab:green", label="observed")
    ax.bar(x + 0.2, unobs_mae, width=0.4, color="tab:red", label="unobserved")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{o}" for o in orders])
    ax.set_xlabel("Harmonic order")
    ax.set_ylabel("pu-MAE")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(frameon=False)
    if out_svg is not None:
        save_figure(fig, out_svg)
    return fig


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over the True positions of ``mask``, reduced over all but H (axis 1).

    An EMPTY selection yields NaN (not plotted), never 0 — a zero bar would read
    as "perfect estimate" when there was nothing to measure (e.g. no unobserved
    rows in a fully-observed stage).
    """
    m = mask.astype(values.dtype)
    num = (values * m).sum(axis=(0, 2))  # [H]
    den = m.sum(axis=(0, 2))
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


def plot_error_vs_sensor_distance(
    prediction: Prediction,
    *,
    order: Optional[int] = None,
    out_svg: Optional[str] = None,
):
    """Per unobserved node: pu-error vs hop distance to the NEAREST sensor node.

    For every node-phase row that is unobserved (in at least one sample), the mean per-unit
    ``|error|`` (over samples where it is unobserved, summed/averaged over orders or one
    ``order``) is plotted against the electrical-graph HOP distance from that node to the
    nearest of the sensor nodes. A rising trend means the estimate degrades with distance from
    a measurement. Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    grid = prediction.grid
    g = grid_graph(grid)
    sensors = _sensor_node_ids(prediction)
    # hop distance (unweighted) from every node to the nearest sensor
    hops_to_nearest: dict[int, float] = {}
    per_sensor = [nx.single_source_shortest_path_length(g, s) for s in sensors]
    for node in grid.nodes:
        nid = int(node.id)
        dists = [d[nid] for d in per_sensor if nid in d]
        hops_to_nearest[nid] = float(min(dists)) if dists else float("nan")

    err = prediction.abs_error_pu()  # [S, H, N]
    if order is not None:
        k = list(prediction.orders).index(order)
        err = err[:, k : k + 1, :]
    # mean over orders -> [S, N]
    err_sn = err.mean(axis=1)
    unobs = ~prediction.mask  # [S, N] True = unobserved
    # per-row mean over the samples where the row was unobserved
    num = (err_sn * unobs).sum(axis=0)  # [N]
    den = unobs.sum(axis=0).clip(min=1)  # [N]
    row_err = num / den

    node_ids = prediction.node_ids
    xs, ys = [], []
    for r in range(prediction.n_rows):
        if not unobs[:, r].any():
            continue  # always observed -> not an SE target here
        d = hops_to_nearest.get(int(node_ids[r]), float("nan"))
        if np.isfinite(d):
            xs.append(d)
            ys.append(float(row_err[r]))
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.scatter(xs, ys, s=18, alpha=0.6, color="tab:purple", edgecolors="none")
    # mean error per integer hop distance (a trend overlay)
    if xs.size:
        for d in np.unique(xs):
            m = xs == d
            ax.scatter(d, ys[m].mean(), s=70, color="black", marker="_", zorder=5)
    ax.set_xlabel("Hops to nearest sensor node")
    ax.set_ylabel("pu-MAE (unobserved)")
    ax.grid(True, alpha=0.3)
    if out_svg is not None:
        save_figure(fig, out_svg)
    return fig


def plot_error_cdf(
    prediction: Prediction, *, xscale: str = "log", out_svg: Optional[str] = None
):
    """Cumulative distribution of per-(sample, order, row) pu-error, observed vs unobserved.

    The empirical CDF of the per-unit ``|error|`` over all entries — split into observed and
    unobserved entries by the placement mask. ``xscale="log"`` (the default) resolves the error
    TAIL (the worst few percent); ``xscale="linear"`` shows the bulk of the distribution on an
    even axis, including the exactly-zero quantiles a log axis must drop. Returns the
    matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    err = prediction.abs_error_pu()  # [S, H, N]
    obs = np.broadcast_to(prediction.mask[:, None, :], err.shape)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    # evaluate the empirical CDF at a fixed grid of quantiles, so the line has a bounded
    # vertex count (and a small SVG) no matter how many millions of entries there are.
    q = np.linspace(0.0, 1.0, 2001)
    for values, mask, color, label in (
        (err, obs, "tab:green", "observed"),
        (err, ~obs, "tab:red", "unobserved"),
    ):
        e = values[mask].reshape(-1)
        if e.size == 0:
            continue
        xe = np.quantile(e, q)
        if xscale == "log":
            keep = xe > 0  # a log x-axis cannot show the zero-error quantiles
            xe, qe = xe[keep], q[keep]
        else:
            qe = q
        ax.plot(xe, qe, color=color, label=label, linewidth=1.8)
    ax.set_xscale(xscale)
    ax.set_xlabel("|error| [pu]")
    ax.set_ylabel("CDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="best")
    if out_svg is not None:
        save_figure(fig, out_svg)
    return fig


def worst_scenario(prediction: Prediction) -> int:
    """Index of the sample with the largest mean per-unit error (the hardest case).

    Reduces the per-unit ``|error|`` over orders and rows to one scalar per sample and returns
    the ``argmax`` — feed it to :func:`~injection_hse.evaluation.state_plots.plot_state_vs_prediction`
    (``scenario=...``) to inspect where the estimator struggles most.
    """
    per_sample = prediction.abs_error_pu().mean(axis=(1, 2))  # [S]
    return int(np.argmax(per_sample))


__all__ = [
    "plot_parity_per_harmonic",
    "plot_error_by_order",
    "plot_error_vs_sensor_distance",
    "plot_error_cdf",
    "worst_scenario",
]
