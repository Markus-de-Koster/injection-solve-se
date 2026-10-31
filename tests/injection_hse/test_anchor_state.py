"""Anchoring lives in ONE place: the solver's weights own their derived state.

The measurement-anchored decode used to keep a second copy of "am I anchored" beside the two
weights, re-synced by hand from four modules — one missed sync silently makes or destroys the
anchored solve, and no metric would show it. The weights are now properties whose derived
tensors (the per-order operator scale, the ``V -> branch current`` map) are materialised on
demand, so the only way to change the anchoring is to set a weight.

Checked here: the round trip through the public surface (up, down, and back to a
never-anchored solve BIT-FOR-BIT), the loud failure when a current weight has no map to use,
that the deployed weights survive into the evaluation output, and that the multi-grid bank
dials its members through the same surface.
"""

from __future__ import annotations

import json

import pytest
import torch

DT = torch.complex128
ORDERS = [1, 3, 5]


def _setup(cigre_grid, **anchor):
    """``(states, slack rows, edge operator, solver)`` on the CIGRE LV benchmark."""
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver

    v = solve_harmonic_flow(cigre_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    vb = torch.stack([v, 1.02 * v, 0.98 * v])  # snapshot batch [B, H, N]
    slack = MeasurementModel(["Bus R4"], slack_known=True).slack_rows(cigre_grid)
    op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=DT)
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT, edge_op=op, **anchor)
    return vb, slack, op, solver


def _solve_args(solver, vb, op, slack):
    """The anchored-solve inputs (fully observed) for a batch of states.

    The predicted injections are deliberately WRONG (half the true ones), so an active anchor
    visibly pulls the state toward the measurements — with consistent inputs the anchored and
    the slack-only solve agree by construction and would hide a dead anchor.
    """
    b, n = vb.shape[0], vb.shape[-1]
    return dict(
        i_pred=0.5 * solver.injection_targets(vb),
        v_slack=vb.select(-2, ORDERS.index(1)).index_select(-1, slack),
        v_meas=vb,
        v_mask=torch.ones(b, n, dtype=torch.bool),
        i_meas=op(vb),
        edge_mask=torch.ones(b, op.n_edges, dtype=torch.bool),
    )


def _solve(solver, args):
    kwargs = dict(args)
    return solver.solve(kwargs.pop("i_pred"), kwargs.pop("v_slack"), **kwargs)


# =============================================================================
# The weights own the anchoring
# =============================================================================
def test_the_anchoring_flag_is_derived_not_stored(cigre_grid):
    _, _, _, solver = _setup(cigre_grid)
    assert solver.anchoring is False
    solver.anchor_v_weight = 2.0
    assert solver.anchoring is True and solver.anchor_v_weight == 2.0
    solver.anchor_v_weight = 0.0
    assert solver.anchoring is False
    # read-only: there is no second copy to write, by construction
    with pytest.raises(AttributeError):
        solver.anchoring = True


def test_dialling_up_then_down_returns_the_unanchored_solve(cigre_grid):
    """weights > 0 then zeroed must solve EXACTLY like a solver that never anchored."""
    vb, slack, op, solver = _setup(cigre_grid)
    args = _solve_args(solver, vb, op, slack)
    plain = _solve(solver, args)

    solver.set_anchor_weights(v_weight=5.0, i_weight=0.5)
    anchored = _solve(solver, args)
    assert not torch.allclose(anchored, plain)

    solver.set_anchor_weights(v_weight=0.0, i_weight=0.0)
    assert torch.equal(_solve(solver, args), plain)


def test_a_solver_built_unanchored_can_be_dialled_up(cigre_grid):
    """The derived tensors are materialised on demand, so the build weight is not a ceiling."""
    vb, slack, op, unanchored = _setup(cigre_grid)
    _, _, _, born_anchored = _setup(
        cigre_grid, anchor_v_weight=3.0, anchor_i_weight=0.5
    )

    unanchored.set_anchor_weights(v_weight=3.0, i_weight=0.5)
    args = _solve_args(unanchored, vb, op, slack)
    assert torch.equal(unanchored.operator_scale(), born_anchored.operator_scale())
    assert torch.allclose(
        _solve(unanchored, args), _solve(born_anchored, args), atol=1e-9
    )


def test_a_current_weight_without_a_map_fails_loudly(cigre_grid):
    """No branch-current operator, no current anchoring — at the moment it is requested."""
    from injection_hse.data import MeasurementModel
    from injection_hse.physics import NetworkSolver

    slack = MeasurementModel(["Bus R4"], slack_known=True).slack_rows(cigre_grid)
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT)
    assert solver.can_anchor_current is False
    with pytest.raises(ValueError, match="BranchCurrentOperator"):
        solver.anchor_i_weight = 1.0

    _, _, _, with_op = _setup(cigre_grid)
    assert with_op.can_anchor_current is True
    with_op.anchor_i_weight = 1.0  # the map is built on demand
    assert with_op.can_anchor_current is True


def test_the_module_stage_schedule_goes_through_the_same_surface(cigre_grid):
    """``set_stage_anchor`` dials the solver; nothing else has to be kept in sync."""
    from injection_hse.train import SEModule

    vb, slack, op, solver = _setup(cigre_grid)
    module = SEModule.__new__(SEModule)
    module.solver = solver
    module._bank = None
    module._anchor_gen = torch.Generator().manual_seed(0)
    module._anchor_draw = "uniform"
    module._topology = None

    module.set_stage_anchor(v_weight=4.0, i_weight=0.25)
    assert (solver.anchor_v_weight, solver.anchor_i_weight) == (4.0, 0.25)
    assert solver.anchoring
    module.set_stage_anchor(v_weight=0.0, i_weight=0.0)
    assert not solver.anchoring
    args = _solve_args(solver, vb, op, slack)
    assert torch.equal(
        _solve(solver, args), solver.solve(args["i_pred"], args["v_slack"])
    )


# =============================================================================
# The deployed weights are recorded with the numbers they produced
# =============================================================================
def test_the_evaluation_metrics_record_the_deployed_anchor(cigre_grid, tmp_path):
    import numpy as np

    from pgml.assembly import node_phase_index

    from injection_hse.data import base_voltage_per_row
    from injection_hse.evaluation.report import _write_metrics_summary
    from injection_hse.evaluation.inference import Prediction

    index = node_phase_index(cigre_grid)
    n = int(index.size)
    target = np.ones((2, len(ORDERS), n), dtype=np.complex128)
    prediction = Prediction(
        target_v=target,
        pred_v=target * 1.01,
        mask=np.zeros((2, n), dtype=bool),
        orders=list(ORDERS),
        base_v=base_voltage_per_row(cigre_grid).numpy(),
        node_ids=index.node_ids.numpy(),
        phase_codes=index.phase_codes.numpy(),
        observed_rows=np.zeros(0, dtype=np.int64),
        sensor_names=[],
        grid=cigre_grid,
    )
    _write_metrics_summary(prediction, tmp_path, anchor_weights=(2.0, 0.5))
    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert summary["anchor"] == {"voltage_weight": 2.0, "current_weight": 0.5}

    # a voltage decode has no solve to anchor: the field is present and explicitly empty
    _write_metrics_summary(prediction, tmp_path, stem="metrics_plain")
    plain = json.loads((tmp_path / "metrics_plain.json").read_text())
    assert plain["anchor"] is None
