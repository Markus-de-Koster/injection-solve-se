"""The differentiable physics decoder: injection currents -> voltages, exact + grad-clean.

Validates the foundation of the injection-estimation state estimator. The estimator predicts
the nodal INJECTION currents and reconstructs the full state with the differentiable network
solve (no ground-truth injection on the RHS, so no measurement leakage):

- the pgml ``Y``-only helper (``assemble_harmonic_ybus``) matches the full harmonic assembly;
- :class:`injection_hse.physics.NetworkSolver` reconstructs the true state from the true nodal injections
  (fundamental via the ideal-slack reduced solve at the measured slack voltage, harmonics via
  the Norton system) for snapshot AND coherent leading shapes, with gradients flowing to the
  predicted injections.
"""

from __future__ import annotations

import torch

DT = torch.complex128
ORDERS = [1, 2, 5, 7]


def test_assemble_harmonic_ybus_matches_full_system(cigre_grid):
    from pgml.solver import (
        assemble_harmonic_system,
        assemble_harmonic_ybus,
        solve_power_flow,
    )

    harm = [h for h in ORDERS if h > 1]
    v1 = solve_power_flow(cigre_grid, slack="ideal", dtype=DT).v
    y_only, _ = assemble_harmonic_ybus(cigre_grid, harm, dtype=DT)
    y_full, _, _ = assemble_harmonic_system(cigre_grid, harm, v1, dtype=DT)
    assert torch.allclose(y_only, y_full, atol=1e-9)


def test_assemble_harmonic_ybus_rejects_fundamental(cigre_grid):
    from pgml.errors import InputError
    from pgml.solver import assemble_harmonic_ybus

    try:
        assemble_harmonic_ybus(cigre_grid, [1, 5], dtype=DT)
    except InputError:
        return
    raise AssertionError("order 1 must be rejected (it is the fundamental)")


def test_network_solver_round_trip_and_grad(cigre_grid):
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel
    from injection_hse.physics import NetworkSolver

    v = solve_harmonic_flow(cigre_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    vb = torch.stack([v, 1.01 * v, 0.99 * v])  # snapshot batch [B, H, N]
    slack = MeasurementModel(["Bus R4"], slack_known=True).known_boundary_rows(
        cigre_grid
    )
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT)

    fund = ORDERS.index(1)
    i_true = solver.injection_targets(vb)  # [B, H, N]
    v_slack = vb.select(-2, fund).index_select(-1, slack)  # measured slack boundary
    assert torch.allclose(solver.solve(i_true, v_slack), vb, atol=1e-6)

    # coherent [B, T, H, N] leading shape round-trips identically
    vt = vb.unsqueeze(1).expand(3, 4, *vb.shape[1:]).contiguous()
    vs_t = vt.select(-2, fund).index_select(-1, slack)
    assert torch.allclose(
        solver.solve(solver.injection_targets(vt), vs_t), vt, atol=1e-6
    )

    # gradients flow to the predicted injections (the model's output)
    i_pred = solver.injection_targets(vb).detach().requires_grad_(True)
    solver.solve(i_pred, v_slack).abs().sum().backward()
    assert i_pred.grad is not None
    assert torch.isfinite(i_pred.grad).all()
    assert i_pred.grad.abs().sum() > 0


def test_network_solver_harmonics_only(cigre_grid):
    """A run without the fundamental (only h>1) round-trips (Norton solve, no slack used)."""
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel
    from injection_hse.physics import NetworkSolver

    orders = [2, 5, 7]
    v = solve_harmonic_flow(cigre_grid, [1, *orders], slack="ideal", dtype=DT).v[1:]
    slack = MeasurementModel(["Bus R4"], slack_known=True).known_boundary_rows(
        cigre_grid
    )
    solver = NetworkSolver(cigre_grid, orders, slack, dtype=DT)
    i_true = solver.injection_targets(v)
    unused_slack = v.new_zeros(
        (*v.shape[:-2], slack.numel())
    )  # no fundamental -> unused
    assert torch.allclose(solver.solve(i_true, unused_slack), v, atol=1e-6)


def test_injection_decode_seModule_trains(cigre_grid, se_dataset):
    """The injection-decode SEModule predicts injections, SOLVES for V, and trains (grad flows).

    The decoded ``pred_v`` satisfies the network law by construction; the supervised loss is on
    the solved voltages, so it is comparable to the direct-voltage estimator's.
    """
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        LossConfig,
        MaskStageConfig,
        PhysicsConfig,
    )
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.data import StateCollate
    from injection_hse.train import SEModule

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    orders = list(source.orders)
    enc = build_encoding("trig")
    tv = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    norm = HarmonicNormalizer().fit(enc.encode(tv).transpose(-3, -2))
    solver = NetworkSolver.from_grid(cigre_grid, orders)
    inj_norm = fit_injection_normalizer(solver, tv, enc)

    cfg = ExperimentConfig(
        name="inj",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[64]),
        physics=PhysicsConfig(decode="injection"),
        loss=LossConfig(),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    mod = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=norm,
        graph=source.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        solver=solver,
        inj_normalizer=inj_norm,
    )
    assert mod._predict_injection_state
    batch = StateCollate()([source.train_dataset()[i] for i in range(16)])
    loss, mask, pred_v, target_v, _, phys, _ = mod._step(batch, train=True)
    assert torch.isfinite(loss)
    assert pred_v.shape == target_v.shape  # solved full state [B, H, N]
    assert phys is None  # physics is STRUCTURAL (the solve), not a soft residual
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in mod.model.parameters() if p.grad is not None)
    assert (
        float(grad) > 0
    )  # gradient reaches the model THROUGH the differentiable solve


def test_network_solver_c64_matches_c128_reference(cigre_grid):
    """The complex64-facing decode must match the complex128 truth.

    A physical feeder's Y reaches condition numbers of 1e6-1e9, so any complex64
    representation of the OPERATOR (assembly or solve) loses up to all of its
    digits (kappa * eps ~ O(1)) — and CPU/CUDA then disagree at the percent
    level. NetworkSolver therefore assembles and solves in complex128 internally
    and only casts the outputs; this guards that precision contract.
    """
    from injection_hse.physics import NetworkSolver

    orders = [1, 5, 7]
    s64 = NetworkSolver.from_grid(cigre_grid, orders)  # complex64 training dtype
    s128 = NetworkSolver.from_grid(cigre_grid, orders, dtype=torch.complex128)
    n = s64.y_harmonic.shape[-1]
    torch.manual_seed(0)
    i = torch.randn(4, len(orders), n, dtype=torch.complex64)
    vs = torch.randn(4, len(s64.slack_rows), dtype=torch.complex64)
    v64 = s64.solve(i, vs)
    v128 = s128.solve(i.to(torch.complex128), vs.to(torch.complex128))
    assert v64.dtype == torch.complex64
    rel = ((v64.to(torch.complex128) - v128).abs() / v128.abs().clamp_min(1e-12)).max()
    assert float(rel) < 1e-5  # only the final output rounding remains
