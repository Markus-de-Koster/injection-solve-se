"""Each plot function returns a figure and writes its file (Agg backend, synthetic data)."""

from __future__ import annotations

import matplotlib
import pytest
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from pgml.schemas.grid_schema import Phase

from injection_hse.evaluation.error_heatmap import plot_error_heatmap
from injection_hse.evaluation.insight_plots import (
    plot_error_by_order,
    plot_error_cdf,
    plot_error_vs_sensor_distance,
    plot_parity_per_harmonic,
    worst_scenario,
)
from injection_hse.evaluation.state_plots import (
    harmonic_profile_from_voltages,
    plot_state_vs_prediction,
)


def test_harmonic_profile_adapter(synthetic_prediction):
    p = synthetic_prediction
    f0 = float(p.grid.base_frequency_hz)
    prof = harmonic_profile_from_voltages(
        p.target_v[0],
        p.grid,
        order=p.orders[0],
        label="truth",
        phase=Phase.A,
        frequencies_hz=[o * f0 for o in p.orders],
    )
    assert prof.order == p.orders[0]
    assert prof.magnitude.shape == prof.distances_km.shape
    assert prof.node_ids is not None and prof.magnitude.size > 0


def test_harmonic_profile_adapter_requires_frequencies(synthetic_prediction):
    """Without frequencies_hz the row-to-order layout is unknowable — raise,
    never guess (a sparse order set would silently plot the wrong order)."""
    p = synthetic_prediction
    with pytest.raises(ValueError, match="frequencies_hz"):
        harmonic_profile_from_voltages(
            p.target_v[0], p.grid, order=p.orders[0], label="truth", phase=Phase.A
        )


def test_plot_state_vs_prediction_writes(synthetic_prediction, tmp_path):
    p = synthetic_prediction
    svg = tmp_path / "state.svg"
    html = tmp_path / "state.html"
    html3d = tmp_path / "state3d.html"
    fig, ax = plot_state_vs_prediction(
        p,
        order=p.orders[0],
        phase=Phase.A,
        scenario=0,
        out_svg=str(svg),
        out_html=str(html),
        out_html_3d=str(html3d),
        orders_3d=p.orders,
    )
    assert svg.exists() and svg.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0
    assert html3d.exists() and html3d.stat().st_size > 0
    plt.close(fig)


def test_plot_error_heatmap_writes(synthetic_prediction, tmp_path):
    p = synthetic_prediction
    svg = tmp_path / "heat.svg"
    html = tmp_path / "heat.html"
    fig = plot_error_heatmap(p, statistic="mean", out_svg=str(svg), out_html=str(html))
    assert svg.exists() and html.exists()
    plt.close(fig)
    # phase filter reduces the number of y rows
    fig2 = plot_error_heatmap(p, statistic="max", phase=Phase.A)
    assert len(fig2.axes[0].get_yticklabels()) < p.n_rows
    plt.close(fig2)


def test_insight_plots_write(synthetic_prediction, tmp_path):
    p = synthetic_prediction
    for fn, name in (
        (plot_parity_per_harmonic, "parity.svg"),
        (plot_error_by_order, "by_order.svg"),
        (plot_error_vs_sensor_distance, "by_dist.svg"),
        (plot_error_cdf, "cdf.svg"),
    ):
        out = tmp_path / name
        fig = fn(p, out_svg=str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)


def test_error_cdf_linear_xscale(synthetic_prediction, tmp_path):
    """The CDF has a log default and a linear variant; the linear axis keeps zero quantiles."""
    p = synthetic_prediction
    fig_log = plot_error_cdf(p)
    assert fig_log.axes[0].get_xscale() == "log"
    plt.close(fig_log)

    out = tmp_path / "cdf_linear.svg"
    fig_lin = plot_error_cdf(p, xscale="linear", out_svg=str(out))
    assert fig_lin.axes[0].get_xscale() == "linear"
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig_lin)


def test_worst_scenario_in_range(synthetic_prediction):
    w = worst_scenario(synthetic_prediction)
    assert 0 <= w < synthetic_prediction.n_samples


def test_error_by_order_split_consistent(synthetic_prediction):
    # the unobserved bars are computed over the unobserved rows only; sanity-check the
    # figure has one observed + one unobserved bar group per order
    fig = plot_error_by_order(synthetic_prediction)
    bars = [c for c in fig.axes[0].containers]
    assert len(bars) == 2  # observed + unobserved
    plt.close(fig)
