"""Switch-tied rows are not part of the injection space (``PhysicsConfig.exclude_tied_rows``).

A node tied to the slack by a closed switch is the slack for every electrical purpose: an
injection there leaves through the tie and moves no voltage, while its ``Y·V`` entry is the
whole feeder's current. These tests pin that the solver finds exactly those rows (and not
an open switch, nor one a switchable topology may open), that the head's output there is
pinned to the zero injection without changing the decoded state, that the normalizer and
the linear prior skip them, and that the escape hatch restores the previous space.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    EncodingConfig,
    ExperimentConfig,
    MaskStageConfig,
    MeasurementConfig,
    PhysicsConfig,
)
from injection_hse.encoding import TrigEncoding

SENSOR_NODES = ("Bus R4", "Bus C5", "Bus I2")
TIED_BUSES = ("Bus R0", "Bus I0", "Bus C0")


def _tied_rows(grid):
    from pgml.assembly import node_phase_index

    idx = node_phase_index(grid)
    return sorted(
        int(idx.row(int(n.id), p))
        for n in grid.nodes
        if n.name in TIED_BUSES
        for p in n.phases
    )


@pytest.fixture(scope="module")
def setup(cigre_grid, se_dataset):
    from injection_hse.data import MeasurementModel, StateCollate, build_data_source
    from injection_hse.normalization import HarmonicNormalizer

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    enc = TrigEncoding()
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(64)])
    norm = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    mm = MeasurementModel.resolve(cigre_grid, list(SENSOR_NODES), slack_known=True)
    return {
        "source": source,
        "train_v": train_v,
        "normalizer": norm,
        "batch": batch,
        "observed_rows": mm.observed_rows(cigre_grid),
    }


def _module(setup, grid, *, exclude: bool, residual: bool = False):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.encoding import build_injection_encoding
    from injection_hse.masking import build_masker
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule, attach_injection_prior

    source = setup["source"]
    cfg = ExperimentConfig(
        name="tied-rows",
        seed=0,
        orders=list(source.orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        encoding=EncodingConfig(injection_kind="cartesian"),
        physics=PhysicsConfig(
            decode="injection",
            anchor_voltage_weight=1.0,
            exclude_tied_rows=exclude,
            injection_prior_residual=residual,
        ),
        measurement=MeasurementConfig(
            voltage_nodes=list(SENSOR_NODES), slack_known=True
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="placement", kind="placement", target_loss=1e-12, max_epochs=1
                )
            ]
        ),
    )
    solver = NetworkSolver.from_grid(grid, list(source.orders), anchor_v_weight=1.0)
    inj_norm = fit_injection_normalizer(
        solver,
        setup["train_v"],
        build_injection_encoding(cfg.encoding),
        exclude_tied=exclude,
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=setup["normalizer"],
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        solver=solver,
        inj_normalizer=inj_norm,
        grid=grid,
    )
    attach_injection_prior(module, cfg, setup["train_v"], setup["observed_rows"])
    module.set_masker(build_masker("placement", observed_rows=setup["observed_rows"]))
    module.set_stage_anchor(v_weight=1.0, i_weight=0.0)
    return module


def _captured_injection(module, batch):
    got = {}
    orig = module._decode_state

    def patched(i_pred, **kw):
        got["i"] = i_pred.detach()
        return orig(i_pred, **kw)

    module._decode_state = patched
    try:
        out = module.infer(batch)
    finally:
        module._decode_state = orig
    return got["i"], out["pred_v"]


# =============================================================================
# The solver finds the ties
# =============================================================================
def test_from_grid_finds_the_switch_ties_and_only_those(cigre_grid):
    from pgml.schemas.grid_schema import Switch

    from injection_hse.physics import NetworkSolver, tied_rows_of

    solver = NetworkSolver.from_grid(cigre_grid, [1, 3])
    assert solver.tied_rows.tolist() == _tied_rows(cigre_grid)
    assert len(solver.tied_rows) == 9
    # the tied rows are not slack rows, and never a device or sensor row
    assert not set(solver.tied_rows.tolist()) & set(solver.slack_rows.tolist())
    # an OPEN tie is no tie
    opened = cigre_grid.model_copy(deep=True)
    switches = [b for b in opened.branches if isinstance(b, Switch)]
    switches[0].default_closed = False
    tied = tied_rows_of(opened, solver.slack_rows).tolist()
    assert len(tied) == 6 and not set(tied) & set(_tied_rows(cigre_grid)[:3])
    # a switch a switchable topology may open is not a static tie either
    states = {switches[1].id: 1.0}
    assert len(tied_rows_of(cigre_grid, solver.slack_rows, states)) == 6
    # the zeroing helper is the identity without ties, out of place with them
    i = torch.ones((2, 2, int(solver.slack_rows.max()) + 200), dtype=torch.complex64)
    z = solver.zero_tied(i)
    assert torch.all(z[..., solver.tied_rows] == 0) and torch.all(i == 1)


# =============================================================================
# The estimator
# =============================================================================
def test_tied_rows_carry_no_injection_and_the_state_is_unchanged(cigre_grid, setup):
    on = _module(setup, cigre_grid, exclude=True)
    off = _module(setup, cigre_grid, exclude=False)
    off.load_state_dict(on.state_dict())
    # one injection frame for both, so the mask is the only difference
    off.inj_normalizer.load_state_dict(on.inj_normalizer.state_dict())
    tied = torch.as_tensor(_tied_rows(cigre_grid))
    assert on._tied_rows.tolist() == tied.tolist() and off._tied_rows.numel() == 0
    batch = setup["batch"]
    i_on, v_on = _captured_injection(on, batch)
    i_off, v_off = _captured_injection(off, batch)
    assert float(i_on[..., tied].abs().max()) == 0.0
    assert float(i_off[..., tied].abs().max()) > 0.0
    # the excluded directions are exactly flat: the decoded state does not depend on them
    assert torch.allclose(v_on, v_off, rtol=1e-6, atol=1e-3)
    # ... and the targets of the injection-space terms are zero there too
    v = batch["v"].to(on._complex_dtype)
    assert float(on._injection_targets(v, None)[..., tied].abs().max()) == 0.0
    assert float(off._injection_targets(v, None)[..., tied].abs().max()) > 0.0


def test_the_normalizer_and_the_prior_skip_the_tied_rows(cigre_grid, setup):
    from injection_hse.encoding import build_injection_encoding
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer

    solver = NetworkSolver.from_grid(cigre_grid, list(setup["source"].orders))
    enc = build_injection_encoding(EncodingConfig(injection_kind="cartesian"))
    with_ties = fit_injection_normalizer(
        solver, setup["train_v"], enc, exclude_tied=False
    )
    without = fit_injection_normalizer(solver, setup["train_v"], enc, exclude_tied=True)
    # the feeder totals at the tied rows inflated the fundamental's scale
    assert float(without.std[0].max()) < float(with_ties.std[0].max())
    # the prior's own scale never saw them (device rows only) — unchanged
    assert torch.allclose(without.aux_scale, with_ties.aux_scale)
    mod = _module(setup, cigre_grid, exclude=True, residual=True)
    prior = mod.injection_prior.prior
    assert not set(prior.rows.tolist()) & set(_tied_rows(cigre_grid))
    assert len(prior.rows) == int(mod.n_nodes) - len(mod.solver.slack_rows) - 9
    # the residual decode is pinned there as well
    i, _ = _captured_injection(mod, setup["batch"])
    assert float(i[..., torch.as_tensor(_tied_rows(cigre_grid))].abs().max()) == 0.0


def test_the_multigrid_ledger_classifies_the_knob():
    from injection_hse.config import MULTIGRID_IGNORED

    assert "physics.exclude_tied_rows" in MULTIGRID_IGNORED
