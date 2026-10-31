"""Actual-vs-predicted harmonic state plots, reusing the :mod:`pgml.evaluation` figures.

The estimator outputs a raw complex state ``V`` ``[H, N]`` (per sample), not a
:class:`~pgml.solver.HarmonicFlowResult`, so :func:`harmonic_profile_from_voltages` adapts a
raw array into the SAME :class:`pgml.evaluation.data.HarmonicProfile` container the existing
plot functions consume — picking the ``H``-slice for an order, converting to per-unit on the
node line-to-neutral base, and pairing each node with its distance from the slack.
:func:`plot_state_vs_prediction` then overlays the ground-truth and the estimate on the
reusable 2D / interactive / 3D harmonic figures, with the measured sensor nodes marked.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from pgml.assembly import node_phase_index
from pgml.evaluation import (
    plot_harmonic_profile,
    plot_harmonic_profile_3d,
    plot_harmonic_profile_interactive,
    save_figure,
    save_html,
)
from pgml.evaluation.data import HarmonicProfile, harmonic_profile_from_array
from pgml.evaluation.topology import distance_from_slack
from pgml.schemas.grid_schema import Phase


from .inference import Prediction

_TRUTH_LABEL = "pgml (truth)"
_ESTIMATE_LABEL = "state estimate"


# =============================================================================
# raw [H, N] complex state -> HarmonicProfile (the pgml.evaluation container)
# =============================================================================
def harmonic_profile_from_voltages(
    v_hn: np.ndarray,
    grid,
    order: int,
    *,
    label: str,
    phase: Phase = Phase.A,
    unit: str = "pu",
    slack: Optional[int] = None,
    frequencies_hz: Optional[Sequence[float]] = None,
) -> HarmonicProfile:
    """Build a :class:`~pgml.evaluation.data.HarmonicProfile` from a raw complex ``[H, N]``.

    A thin wrapper over :func:`pgml.evaluation.data.harmonic_profile_from_array`
    (the shared row-building primitive — same per-unit base and node/phase/order
    alignment as the solver-result path). ``frequencies_hz`` (one entry per
    ``H``-row) is REQUIRED: a bare array cannot reveal its own row-to-order
    layout, and guessing "rows are the consecutive orders 1..H" silently plots
    the WRONG order on any sparse order set (e.g. ``[1, 3, 5, 7]``).
    """
    if frequencies_hz is None:
        raise ValueError(
            "frequencies_hz is required: a raw [H, N] array cannot reveal which "
            "row carries which harmonic order (pass e.g. [o * f0 for o in orders])."
        )
    return harmonic_profile_from_array(
        np.asarray(v_hn),
        grid,
        order,
        frequencies_hz=frequencies_hz,
        label=label,
        phase=phase,
        slack=slack,
        unit=unit,
    )


# =============================================================================
# sensor overlay helpers
# =============================================================================
def _sensor_distances(prediction: Prediction, phase: Phase) -> dict[int, float]:
    """Map ``sensor node_id -> distance from slack`` for the sensors observed at ``phase``."""
    grid = prediction.grid
    index = node_phase_index(grid)
    dist = distance_from_slack(grid)
    obs = set(int(r) for r in prediction.observed_rows)
    out: dict[int, float] = {}
    for node in grid.nodes:
        if phase not in node.phases:
            continue
        if index.row(int(node.id), phase) in obs:
            out[int(node.id)] = dist[int(node.id)]
    return out


def _mark_sensors_mpl(
    ax, prediction: Prediction, profile: HarmonicProfile, phase: Phase
) -> None:
    """Mark the measured sensor NODES with a distinct marker on the magnitude profile.

    Plots an open red star at each sensor node's ``(distance-from-slack, magnitude)`` point of
    ``profile`` (the truth curve). This pins the sensor to its exact node — a vertical line at
    the sensor's distance is ambiguous when several nodes sit at (or near) the same distance.
    """
    sensors = _sensor_distances(prediction, phase)
    mag_by_node = {
        int(n): float(m) for n, m in zip(profile.node_ids, profile.magnitude)
    }
    pts = [(d, mag_by_node[nid]) for nid, d in sensors.items() if nid in mag_by_node]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.scatter(
        xs,
        ys,
        marker="*",
        s=240,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.6,
        zorder=6,
        label="sensor",
    )
    ax.legend(frameon=False, loc="best")


def _sensor_scatter_2d(prediction: Prediction, profile: HarmonicProfile, phase: Phase):
    """A plotly scatter trace marking the sensor nodes on the 2D harmonic figure."""
    import plotly.graph_objects as go

    sensors = _sensor_distances(prediction, phase)
    mag_by_node = {
        int(n): float(m) for n, m in zip(profile.node_ids, profile.magnitude)
    }
    xs = [d for nid, d in sensors.items() if nid in mag_by_node]
    ys = [mag_by_node[nid] for nid in sensors if nid in mag_by_node]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=15, color="black", symbol="x", line=dict(width=2)),
        name="sensor",
        hovertemplate="sensor<br>dist=%{x:.3f} km<extra></extra>",
    )


# =============================================================================
# the full actual-vs-predicted plot set
# =============================================================================
def plot_state_vs_prediction(
    prediction: Prediction,
    *,
    order: int,
    phase: Phase = Phase.A,
    scenario: int = 0,
    out_svg: Optional[str] = None,
    out_html: Optional[str] = None,
    out_html_3d: Optional[str] = None,
    orders_3d: Optional[Sequence[int]] = None,
):
    """Ground-truth vs estimated harmonic state for one sample, sensors marked.

    Builds two :class:`~pgml.evaluation.data.HarmonicProfile`\\ s for ``scenario`` — the
    ``pgml (truth)`` profile from the dataset target and the ``state estimate`` profile from
    the prediction — and renders them on the reusable :mod:`pgml.evaluation` figures:

    - ``out_svg``      a paper-ready 2D magnitude-vs-distance figure
      (:func:`pgml.evaluation.plot_harmonic_profile`), sensor nodes marked by a star;
    - ``out_html``     the interactive 2D figure
      (:func:`pgml.evaluation.plot_harmonic_profile_interactive`) + a sensor scatter trace;
    - ``out_html_3d``  the interactive 3D figure over ``orders_3d``
      (:func:`pgml.evaluation.plot_harmonic_profile_3d`, estimate dashed) + a sensor trace.

    None of the three carries an in-figure title (caption-ready): the default title the
    :mod:`pgml.evaluation` functions draw is cleared after each call.

    Returns ``(fig_mpl, ax_mpl)`` (the 2D matplotlib figure); the interactive figures are
    written to the given paths. ``orders_3d`` defaults to every order in the prediction.
    """
    grid = prediction.grid
    f0 = float(grid.base_frequency_hz)
    # map an order to the correct harmonic-axis slice via the prediction's (possibly sparse)
    # order list, not the 1..H fallback (orders may skip even harmonics)
    freqs = [o * f0 for o in prediction.orders]
    truth = harmonic_profile_from_voltages(
        prediction.target_v[scenario],
        grid,
        order,
        label=_TRUTH_LABEL,
        phase=phase,
        frequencies_hz=freqs,
    )
    est = harmonic_profile_from_voltages(
        prediction.pred_v[scenario],
        grid,
        order,
        label=_ESTIMATE_LABEL,
        phase=phase,
        frequencies_hz=freqs,
    )

    # -- 2D matplotlib (paper-ready) --
    fig, ax = plot_harmonic_profile([truth, est], grid=grid)
    ax.set_title("")  # no in-figure title — the caption is written externally
    _mark_sensors_mpl(ax, prediction, truth, phase)
    if out_svg is not None:
        save_figure(fig, out_svg)

    # -- 2D interactive --
    if out_html is not None:
        ifig = plot_harmonic_profile_interactive(
            [truth, est],
            grid=grid,
            out_html=None,
        )
        ifig.update_layout(
            title=None
        )  # no in-figure title — caption written externally
        ifig.add_trace(_sensor_scatter_2d(prediction, truth, phase))

        save_html(ifig, out_html)

    # -- 3D interactive (several orders, estimate dashed) --
    if out_html_3d is not None:
        orders = list(orders_3d) if orders_3d is not None else list(prediction.orders)
        profiles = []
        for o in orders:
            profiles.append(
                harmonic_profile_from_voltages(
                    prediction.target_v[scenario],
                    grid,
                    o,
                    label=_TRUTH_LABEL,
                    phase=phase,
                    frequencies_hz=freqs,
                )
            )
            profiles.append(
                harmonic_profile_from_voltages(
                    prediction.pred_v[scenario],
                    grid,
                    o,
                    label=_ESTIMATE_LABEL,
                    phase=phase,
                    frequencies_hz=freqs,
                )
            )
        fig3d = plot_harmonic_profile_3d(
            profiles,
            grid=grid,
            reference_labels=[_ESTIMATE_LABEL],  # estimate drawn dashed
            out_html=None,
        )
        fig3d.update_layout(
            title=None
        )  # no in-figure title — caption written externally
        _add_sensor_trace_3d(fig3d, prediction, profiles[0], phase)

        save_html(fig3d, out_html_3d)

    return fig, ax


def _add_sensor_trace_3d(
    fig, prediction: Prediction, profile: HarmonicProfile, phase: Phase
):
    """A 3D scatter trace marking the sensor nodes (at the first order's magnitude/angle)."""
    import plotly.graph_objects as go

    sensors = _sensor_distances(prediction, phase)
    by_node = {
        int(n): (float(m), float(a))
        for n, m, a in zip(profile.node_ids, profile.magnitude, profile.angle_deg)
    }
    xs, ys, zs = [], [], []
    for nid, d in sensors.items():
        if nid in by_node:
            m, a = by_node[nid]
            xs.append(d)
            ys.append(m)
            zs.append(a)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=6, color="black", symbol="x"),
            name="sensor",
        )
    )


__all__ = ["harmonic_profile_from_voltages", "plot_state_vs_prediction"]
