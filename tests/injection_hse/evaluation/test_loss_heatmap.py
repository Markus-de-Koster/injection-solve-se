"""The loss heatmap: which supervised term, which band, which injection node.

Where consumption and generation cancel — on one bus, or on neighbouring unmetered buses —
the voltages admit a whole family of injections, so a LOW voltage loss sits beside a HIGH
injection loss. These tests pin the arithmetic that makes that readable: the two terms are
reported on ONE scale (the blend's), the total is exactly the blend of the maps, and the
figure lays the fundamental beside the harmonics over the injection rows of one phase.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pgml.schemas.grid_schema import Phase

SPLIT = 0.5


@pytest.fixture(scope="module")
def injection_run(cigre_grid, se_dataset):
    """An injection-decode module + its data source (the decode the heatmap is about)."""
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        LossConfig,
        MaskStageConfig,
        MeasurementConfig,
        PhysicsConfig,
    )
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    orders = list(source.orders)
    enc = build_encoding("trig")
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    normalizer = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    solver = NetworkSolver.from_grid(cigre_grid, orders)
    inj_norm = fit_injection_normalizer(solver, train_v, enc)
    cfg = ExperimentConfig(
        name="loss-heatmap",
        seed=0,
        orders=orders,
        data=DataConfig(source="parquet", dataset_path=str(se_dataset), batch_size=8),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode="injection"),
        loss=LossConfig(injection_state_loss_split=SPLIT),
        measurement=MeasurementConfig(slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,
                    max_epochs=1,
                )
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=normalizer,
        graph=source.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        solver=solver,
        inj_normalizer=inj_norm,
        grid=cigre_grid,
    )
    return module, source, cfg.measurement


@pytest.fixture(scope="module")
def maps(injection_run, cigre_grid):
    from injection_hse.evaluation import loss_term_maps

    module, source, measurement = injection_run
    return loss_term_maps(
        module,
        source_or_path=source,
        grid=cigre_grid,
        measurement=measurement,
        split="val",
        device="cpu",
        max_batches=2,
    )


def test_injection_rows_are_the_load_and_generator_buses(cigre_grid):
    """The y-axis is the buses that can host a cancelling pair, derived from the grid."""
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import InjectionAppliance

    from injection_hse.evaluation.loss_heatmap import _injection_rows

    index = node_phase_index(cigre_grid)
    nodes = {
        int(a.node)
        for a in cigre_grid.appliances
        if isinstance(a, InjectionAppliance) and a.in_service
    }
    expected = sorted({r for n in nodes for r in index.rows(n)})
    assert _injection_rows(cigre_grid).tolist() == expected
    assert expected, "the CIGRE fixture is expected to carry injection appliances"


def test_maps_carry_both_terms_per_row_and_order(maps, injection_run):
    """One ``[N, H]`` map per term, in the loss's own units."""
    module, _, _ = injection_run
    n_rows, n_orders = module.n_nodes, len(module.orders)
    assert maps.has_injection_term
    for name in ("voltage", "injection", "total"):
        mat = getattr(maps, name)
        assert mat.shape == (n_rows, n_orders)
        assert np.isfinite(mat).all()
        assert (mat >= 0.0).all()
    assert maps.orders == list(module.orders)


def test_total_is_exactly_the_blend_of_the_two_maps(maps):
    """``total = (1-s)·voltage + s·injection`` cell for cell — the map IS the objective.

    Only true because the injection map carries the blend's own scaling; it is what lets a
    voltage cell and an injection cell be compared (and summed) at all.
    """
    expected = (1.0 - SPLIT) * maps.voltage + SPLIT * maps.injection
    assert np.allclose(maps.total, expected, rtol=1e-6)


def test_the_two_terms_land_on_a_comparable_scale(maps):
    """The two maps carry the same total mass — which is what makes the columns comparable.

    The blend's scaling equalizes the quantity the loss REDUCES (the mean over rows and
    orders), so neither column is systematically brighter for a reason unrelated to any
    particular row. Unscaled the terms differ by 82x-45000x and one column would be uniformly
    dark. Their per-row SPREAD around that common mean is left untouched — that spread is the
    signal the heatmap is for.
    """
    assert maps.injection.mean() == pytest.approx(maps.voltage.mean(), rel=0.25)
    # and the rows genuinely disagree: some carry far more injection loss than voltage loss
    per_row = maps.injection.mean(axis=1) / np.maximum(maps.voltage.mean(axis=1), 1e-30)
    assert per_row.max() > 10.0 * np.median(per_row)


def test_plot_lays_the_bands_side_by_side_over_one_phase(maps, tmp_path):
    """Two panels (fundamental | harmonics), three term columns each, phase-A rows only."""
    from injection_hse.evaluation import plot_loss_heatmap

    svg = tmp_path / "loss_heatmap.svg"
    fig = plot_loss_heatmap(maps, phase=Phase.A, out_svg=str(svg))
    assert svg.exists() and svg.stat().st_size > 0

    panels = [ax for ax in fig.axes if ax.get_images()]
    assert len(panels) == 2
    # the panels already carry the band, so the columns name the TERMS only — and the
    # injection term is a current residual, never labelled as a power
    terms = ["$L_{volt}$", "$L_{inj}$", "$L_{tot}$"]
    assert [t.get_text() for t in panels[0].get_xticklabels()] == terms
    assert [t.get_text() for t in panels[1].get_xticklabels()] == terms
    # every plotted row is a phase-A injection row
    n_shown = panels[0].get_images()[0].get_array().shape[0]
    phase_a = [r for r in maps.injection_rows if int(maps.phase_codes[int(r)]) == 0]
    assert n_shown == len(phase_a)
    plt.close(fig)


def test_a_voltage_only_run_drops_the_injection_column(injection_run, cigre_grid):
    """Without the direct term there is no injection map — two columns, and total = voltage."""
    from injection_hse.evaluation import loss_term_maps, plot_loss_heatmap

    module, source, measurement = injection_run
    module.cfg = module.cfg.model_copy(
        update={
            "loss": module.cfg.loss.model_copy(
                update={"injection_state_loss_split": 0.0}
            )
        }
    )
    voltage_only = loss_term_maps(
        module,
        source_or_path=source,
        grid=cigre_grid,
        measurement=measurement,
        split="val",
        device="cpu",
        max_batches=1,
    )
    assert not voltage_only.has_injection_term
    assert np.allclose(voltage_only.total, voltage_only.voltage)
    fig = plot_loss_heatmap(voltage_only, phase=Phase.A)
    panels = [ax for ax in fig.axes if ax.get_images()]
    assert [t.get_text() for t in panels[0].get_xticklabels()] == [
        "$L_{volt}$",
        "$L_{tot}$",
    ]
    plt.close(fig)
