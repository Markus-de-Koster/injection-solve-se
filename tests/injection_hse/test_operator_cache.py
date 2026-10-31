"""The factor-once operator cache reproduces the direct network solve exactly.

``physics.operator_cache_mb`` swaps the per-call factorization of the (grid-constant)
solve for cached images — the plain decode's LU factors, the anchored decode's
:class:`pgml.solver.AnchoredSystem`. Results and gradients must match the direct paths
to machine rounding, the budget must degrade to the direct solve when spent, and a
``Y`` on the autograd tape (the learned-calibration path) must never be cached.
"""

from __future__ import annotations

import torch

from injection_hse.physics import NetworkSolver, OperatorCacheBudget

from .tiny_grids import single_phase_chain

ORDERS = [1, 3, 5]
torch.manual_seed(0)


def _solver(anchor_v: float = 0.0) -> NetworkSolver:
    return NetworkSolver.from_grid(
        single_phase_chain(), ORDERS, anchor_v_weight=anchor_v
    )


def _inputs(n: int, batch: int = 3):
    i_pred = torch.randn(batch, len(ORDERS), n, dtype=torch.complex128)
    v_slack = torch.full((batch, 1), 230.0, dtype=torch.complex128)
    v_meas = 230.0 * (
        1.0 + 0.01 * torch.randn(batch, len(ORDERS), n, dtype=torch.complex128)
    )
    v_mask = torch.zeros(batch, n)
    v_mask[:, -1] = 1.0
    v_mask[0, 1] = 1.0  # heterogeneous anchor pattern across the batch
    return i_pred, v_slack, v_meas, v_mask


def test_plain_solve_matches_direct():
    solver = _solver()
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, _, _ = _inputs(n)
    ref = solver.solve(i_pred, v_slack)
    solver.enable_operator_cache(OperatorCacheBudget(64 * 2**20))
    out = solver.solve(i_pred, v_slack)
    assert solver._plain_factors  # the cache actually engaged
    assert torch.allclose(out, ref, rtol=1e-10, atol=1e-9)
    # second call reuses the factors and stays put
    again = solver.solve(i_pred, v_slack)
    assert torch.equal(out, again)


def test_anchored_solve_matches_direct_with_gradients():
    solver = _solver(anchor_v=1.0)
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, v_meas, v_mask = _inputs(n)

    def run():
        i = i_pred.clone().requires_grad_(True)
        v = solver.solve(i, v_slack, v_meas=v_meas, v_mask=v_mask)
        v.abs().sum().backward()
        return v.detach(), i.grad

    ref, ref_grad = run()
    solver.enable_operator_cache(OperatorCacheBudget(64 * 2**20))
    out, out_grad = run()
    assert solver._anchored_systems  # the cache actually engaged
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-8)
    assert torch.allclose(out_grad, ref_grad, rtol=1e-7, atol=1e-8)


def test_spent_budget_falls_back_to_direct():
    solver = _solver(anchor_v=1.0)
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, v_meas, v_mask = _inputs(n)
    ref = solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    solver.enable_operator_cache(OperatorCacheBudget(8))  # nothing fits
    out = solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    assert not solver._anchored_systems and not solver._plain_factors
    assert torch.equal(out, ref)  # the direct path, bit for bit


def test_operators_on_tape_are_never_cached():
    solver = _solver(anchor_v=1.0)
    solver.y_harmonic.requires_grad_(True)
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, v_meas, v_mask = _inputs(n)
    solver.enable_operator_cache(OperatorCacheBudget(64 * 2**20))
    v = solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    assert not solver._anchored_systems and not solver._plain_factors
    assert v.abs().sum().requires_grad  # Y stayed on the tape through the solve


def test_release_refunds_and_returns_to_direct_solve():
    """Disabling the cache frees it, refunds the budget, and later solves run direct."""
    solver = _solver(anchor_v=1.0)
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, v_meas, v_mask = _inputs(n)
    budget = OperatorCacheBudget(64 * 2**20)
    solver.enable_operator_cache(budget)
    solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    solver.solve(i_pred, v_slack)
    assert solver._anchored_systems and solver._plain_factors
    assert budget.remaining < 64 * 2**20
    solver.enable_operator_cache(None)
    assert not solver._anchored_systems and not solver._plain_factors
    assert budget.remaining == 64 * 2**20
    ref = NetworkSolver.from_grid(
        single_phase_chain(), ORDERS, anchor_v_weight=1.0
    ).solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    out = solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    assert not solver._anchored_systems  # nothing was rebuilt
    assert torch.equal(out, ref)  # the direct path, bit for bit


def test_reassembly_and_device_move_invalidate_and_refund():
    solver = _solver(anchor_v=1.0)
    n = solver.y_harmonic.shape[-1]
    i_pred, v_slack, v_meas, v_mask = _inputs(n)
    budget = OperatorCacheBudget(64 * 2**20)
    solver.enable_operator_cache(budget)
    solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    solver.solve(i_pred, v_slack)
    assert budget.remaining < 64 * 2**20
    solver.assemble()
    assert not solver._anchored_systems and not solver._plain_factors
    assert budget.remaining == 64 * 2**20
    # rebuilds transparently after the invalidation
    ref = NetworkSolver.from_grid(
        single_phase_chain(), ORDERS, anchor_v_weight=1.0
    ).solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    out = solver.solve(i_pred, v_slack, v_meas=v_meas, v_mask=v_mask)
    assert torch.allclose(out, ref, rtol=1e-9, atol=1e-8)
