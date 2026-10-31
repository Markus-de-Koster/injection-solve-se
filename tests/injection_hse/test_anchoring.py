"""Measurement-anchored (over-determined) injection decode.

The plain injection decode reconstructs the state from the predicted nodal injections with only
the slack held fixed. With anchoring, the OBSERVED voltages and branch currents become soft
least-squares constraints, so the reconstructed state is pulled toward every measurement — the
network law is traded off against the observations rather than satisfied exactly. These tests
check the two limits (off = the plain solve; consistent measurements = exact recovery), that a
strong anchor drives the observed rows onto the measurements, and that gradients still flow to
the predicted injections, the measurements and the grid.

A fully observed stage anchored at ``anchor_voltage_weight >= 1`` drives the supervised
VOLTAGE loss to round-off: the anchored least squares reproduces the measured state almost
whatever injection was predicted. With ``injection_state_loss_split > 0`` the TOTAL loss is
then dominated by the injection term and numerically insensitive to the anchor weight, so a
test of anchor-dependent behaviour uses a sparse mask or asserts on
``SEModule._loss_components["voltage"]``, never on the total.

The stage / batch anchor tests build the module with ``SEModule.__new__`` and set only the
attributes the method under test reads (``solver``, ``_bank``, ``_anchor_gen``,
``_anchor_draw``; the solver stand-in needs ``anchor_*_weight`` and
``can_anchor_current``). Any new attribute read by ``set_stage_anchor`` /
``_apply_batch_anchor`` must be added to those stubs here and in
``test_injection_state_loss.py``. The solver's anchoring state is not settable directly:
``anchor_v_weight`` / ``anchor_i_weight`` route through ``set_anchor_weights`` and
``anchoring`` is derived from them.
"""

from __future__ import annotations

import torch

DT = torch.complex128
ORDERS = [1, 3, 5]


def _setup(cigre_grid, **anchor):
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver

    v = solve_harmonic_flow(cigre_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    vb = torch.stack([v, 1.02 * v, 0.98 * v])  # snapshot batch [B, H, N]
    slack = MeasurementModel(["Bus R4"], slack_known=True).known_boundary_rows(
        cigre_grid
    )
    op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=DT)
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT, edge_op=op, **anchor)
    return vb, slack, op, solver


def test_anchor_off_dispatches_to_plain(cigre_grid):
    """Zero weights keep the slack-only solve — even when measurements are supplied."""
    vb, slack, op, plain = _setup(cigre_grid)
    assert plain.anchoring is False
    fund = ORDERS.index(1)
    i_true = plain.injection_targets(vb)
    v_slack = vb.select(-2, fund).index_select(-1, slack)
    b, n = vb.shape[0], vb.shape[-1]
    out = plain.solve(
        i_true,
        v_slack,
        v_meas=vb,
        v_mask=torch.ones(b, n, dtype=torch.bool),
        i_meas=op(vb),
        edge_mask=torch.ones(b, op.n_edges, dtype=torch.bool),
    )
    assert torch.equal(out, plain.solve(i_true, v_slack))


def test_anchor_recovers_truth_when_consistent(cigre_grid):
    """Consistent injections + measurements have a unique minimiser at the true state."""
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1.0, anchor_i_weight=1.0)
    assert solver.anchoring
    fund = ORDERS.index(1)
    i_true = solver.injection_targets(vb)
    i_edges = op(vb)
    v_slack = vb.select(-2, fund).index_select(-1, slack)
    b, n, e = vb.shape[0], vb.shape[-1], op.n_edges
    vmask = torch.ones(b, n, dtype=torch.bool)
    emask = torch.ones(b, e, dtype=torch.bool)
    out = solver.solve(
        i_true, v_slack, v_meas=vb, v_mask=vmask, i_meas=i_edges, edge_mask=emask
    )
    assert torch.allclose(out, vb, atol=1e-5)

    # gradients reach every differentiable input (injections, measurements, slack)
    ip = i_true.clone().requires_grad_(True)
    vm = vb.clone().requires_grad_(True)
    im = i_edges.clone().requires_grad_(True)
    vs = v_slack.clone().requires_grad_(True)
    g = torch.autograd.grad(
        solver.solve(ip, vs, v_meas=vm, v_mask=vmask, i_meas=im, edge_mask=emask)
        .abs()
        .sum(),
        [ip, vm, im, vs],
    )
    assert all(torch.isfinite(x).all() for x in g)


def test_strong_anchor_pulls_observed_rows(cigre_grid):
    """A strong voltage anchor overrides wrong (zero) injections at the observed rows."""
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1e8)
    fund = ORDERS.index(1)
    v_slack = vb.select(-2, fund).index_select(-1, slack)
    b, n = vb.shape[0], vb.shape[-1]
    obs_rows = torch.arange(6, 12)
    vmask = torch.zeros(b, n, dtype=torch.bool)
    vmask[:, obs_rows] = True
    vmask[:, solver.slack_rows] = True
    out = solver.solve(
        vb.new_zeros(b, len(ORDERS), n), v_slack, v_meas=vb, v_mask=vmask
    )
    # the observed HARMONIC rows track the measurement despite the zeroed injection prior
    assert torch.allclose(out[:, 1:, obs_rows], vb[:, 1:, obs_rows], atol=1e-4)


def test_anchor_build_weights_takes_max_over_stages():
    """The solver must be BUILT for the largest weight any stage requests (per-stage schedule)."""
    from injection_hse.config import CurriculumConfig, MaskStageConfig, PhysicsConfig
    from injection_hse.train import anchor_build_weights

    physics = PhysicsConfig(
        decode="injection", anchor_voltage_weight=1.0, anchor_current_weight=0.0
    )
    curriculum = CurriculumConfig(
        stages=[
            MaskStageConfig(name="full", kind="full", target_loss=1e-9, max_epochs=1),
            MaskStageConfig(
                name="placement",
                kind="placement",
                target_loss=1e-9,
                max_epochs=1,
                anchor_voltage_weight=1e4,
                anchor_current_weight=5.0,
            ),
        ]
    )
    # global v=1 vs stage v=1e4 -> 1e4; global i=0 vs stage i=5 -> 5
    assert anchor_build_weights(physics, [curriculum]) == (1e4, 5.0)


def test_current_anchor_requires_edge_op(cigre_grid):
    """Current anchoring needs the branch-current operator to build its V->current map."""
    from injection_hse.data import MeasurementModel
    from injection_hse.physics import NetworkSolver

    slack = MeasurementModel(["Bus R4"], slack_known=True).known_boundary_rows(
        cigre_grid
    )
    try:
        NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT, anchor_i_weight=1.0)
    except ValueError:
        return
    raise AssertionError("current anchoring without an edge_op must raise")


def test_per_batch_anchor_randomization_draws_within_range():
    """set_stage_anchor with ranges: training batches draw varying weights inside
    the range (one shared sample preserves the v/i trust ratio); eval batches
    restore the fixed stage weights."""
    import torch

    from injection_hse.train import SEModule

    class _Solver:  # minimal stand-in: the runtime scalars + the current capability
        anchor_v_weight = 0.0
        anchor_i_weight = 0.0
        can_anchor_current = True

    mod = SEModule.__new__(SEModule)
    mod.solver = _Solver()
    mod._bank = None  # single-grid run: no multi-grid operator bank to dial
    mod._anchor_gen = torch.Generator().manual_seed(0)
    mod._anchor_draw = "uniform"  # normally resolved from PhysicsConfig at construction
    mod.set_stage_anchor(
        v_weight=1.0, i_weight=0.1, v_range=(0.5, 2.0), i_range=(0.05, 0.2)
    )
    draws = []
    for _ in range(16):
        mod._apply_batch_anchor(train=True)
        wv, wi = mod.solver.anchor_v_weight, mod.solver.anchor_i_weight
        assert 0.5 <= wv <= 2.0 and 0.05 <= wi <= 0.2
        # one shared uniform sample -> the trust ratio is preserved
        assert abs((wv - 0.5) / 1.5 - (wi - 0.05) / 0.15) < 1e-9
        draws.append(wv)
    assert len(set(round(d, 6) for d in draws)) > 4  # genuinely varying
    mod._apply_batch_anchor(train=False)
    assert mod.solver.anchor_v_weight == 1.0
    assert mod.solver.anchor_i_weight == 0.1
