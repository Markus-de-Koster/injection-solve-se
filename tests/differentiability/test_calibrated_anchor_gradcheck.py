"""float64 gradcheck: the NOISE-CALIBRATED measurement-anchored injection decode.

Under ``PhysicsConfig.anchor_weighting="noise_calibrated"`` the anchored solve multiplies its
runtime anchor scalars by per-channel inverse-variance weights installed once on the solver
(:meth:`injection_hse.physics.NetworkSolver.set_anchor_calibration`). Those weights are constants off
the tape; the solve must stay exactly differentiable in the predicted injections, exactly as
the manually weighted one is (``test_multigrid_decode_gradcheck``).
"""

from __future__ import annotations

import torch

DT = torch.complex128
ORDERS = [1, 3]


def _calibrated_solver():
    """A current-anchored solver on a small feeder with a clean-sensor calibration installed."""
    from pgml.grids import synthetic_feeder

    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver, clean_sensor_sigma

    grid = synthetic_feeder(4, n_feeders=2)
    op = BranchCurrentOperator(grid, ORDERS, dtype=DT)
    solver = NetworkSolver.from_grid(
        grid, ORDERS, dtype=DT, anchor_v_weight=1.0, anchor_i_weight=1.0, edge_op=op
    )
    n = int(solver.y_fundamental.shape[-1])
    gen = torch.Generator().manual_seed(0)
    v_meas = torch.complex(
        torch.rand((len(ORDERS), n), generator=gen, dtype=torch.float64) + 0.5,
        0.1 * torch.rand((len(ORDERS), n), generator=gen, dtype=torch.float64),
    )
    i_meas = op(v_meas.unsqueeze(0)).squeeze(0)  # [H, E]
    typ_v = v_meas.abs().pow(2).mean(dim=-1).sqrt()
    typ_i = i_meas.abs().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    solver.set_anchor_calibration(
        sigma_injection=torch.full((len(ORDERS),), 1e-2, dtype=torch.float64),
        sigma_voltage=clean_sensor_sigma(typ_v, 1e-2),
        sigma_current=clean_sensor_sigma(typ_i, 1e-2),
    )
    assert solver.anchor_calibrated
    return solver, op, v_meas, i_meas


def test_gradcheck_calibrated_anchored_solve_wrt_the_predicted_injections():
    solver, op, v_meas, i_meas = _calibrated_solver()
    n, e = v_meas.shape[-1], op.n_edges
    vmask = torch.zeros(n, dtype=torch.bool)
    vmask[::2] = True
    vmask[solver.slack_rows] = True
    emask = torch.ones(e, dtype=torch.bool)
    v_slack = v_meas[0].index_select(-1, solver.slack_rows)
    i_pred = torch.randn((len(ORDERS), n), dtype=DT, requires_grad=True)

    def fn(i):
        # scaled to the amperes a per-unit injection maps to, so the solved voltages stay
        # in a range where the finite differences are meaningful
        return solver.solve(
            i * 1e-2,
            v_slack,
            v_meas=v_meas,
            v_mask=vmask,
            i_meas=i_meas,
            edge_mask=emask,
        )

    assert torch.autograd.gradcheck(fn, (i_pred,), eps=1e-6, atol=1e-6)


def test_gradcheck_calibrated_anchored_solve_wrt_the_measurements():
    """The anchor targets are MEASURED inputs; their gradient path must be exact too."""
    solver, op, v_meas, i_meas = _calibrated_solver()
    n, e = v_meas.shape[-1], op.n_edges
    vmask = torch.ones(n, dtype=torch.bool)
    emask = torch.ones(e, dtype=torch.bool)
    v_slack = v_meas[0].index_select(-1, solver.slack_rows)
    i_pred = torch.randn((len(ORDERS), n), dtype=DT) * 1e-2
    vm = v_meas.clone().requires_grad_(True)
    im = i_meas.clone().requires_grad_(True)

    def fn(v, i):
        return solver.solve(
            i_pred, v_slack, v_meas=v, v_mask=vmask, i_meas=i, edge_mask=emask
        )

    assert torch.autograd.gradcheck(fn, (vm, im), eps=1e-6, atol=1e-6)
