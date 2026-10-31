"""Stage 1 grid calibration: a twin-grid recovers a known parameter error via the solve.

The TRUE grid has a line resistance off by +30% from the recorded (nominal) value. Given the
true nodal injections, the differentiable network solve + a learned correction ``Δ`` on that
line's resistance is optimised to fit the measured sensor voltages. The test asserts that ``Δ``
recovers the +30% error and that the FULL-state voltage error then drops far below the
nominal-``Y`` baseline — i.e. calibration beats trusting the imperfect grid description.
"""

from __future__ import annotations

import numpy as np
import torch

DT = torch.complex128
ORDERS = [1, 5]


def _line_with_r(grid):
    return next(
        b
        for b in grid.branches
        if getattr(b, "series_resistance_ohm_per_m", None) is not None
        and getattr(b, "conductor_geometry", None) is None
    )


def _cigre_no_geometry():
    """The 3-phase CIGRE LV grid WITHOUT synthesized geometry (explicit R/L lines)."""
    import warnings

    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.oracles import cigre_lv_full_grid

    warnings.filterwarnings("ignore")
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    return grid


def test_twin_grid_recovers_resistance_error():
    from pgml.solver import solve_harmonic_flow

    from injection_hse.calibration import CalibrationKnob, GridCalibration
    from injection_hse.data import MeasurementModel
    from injection_hse.physics import NetworkSolver

    true_grid = _cigre_no_geometry()
    line_id = int(_line_with_r(true_grid).id)
    # the TRUE grid: this line's resistance is 30% higher than recorded
    ln = _line_with_r(true_grid)
    r_nominal = torch.as_tensor(
        np.asarray(ln.series_resistance_ohm_per_m), dtype=torch.float64
    )
    ln.series_resistance_ohm_per_m = r_nominal * 1.3

    slack = MeasurementModel(["Bus R4"], slack_known=True).known_boundary_rows(
        true_grid
    )
    true_solver = NetworkSolver(true_grid, ORDERS, slack, dtype=DT)
    v_true = solve_harmonic_flow(true_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    i_true = true_solver.injection_targets(v_true)  # the true nodal injections
    v_slack = v_true.select(-2, ORDERS.index(1)).index_select(-1, slack)

    # sensors observe a handful of nodes; the rest are unobserved (the SE target)
    sensor_rows = MeasurementModel(
        ["Bus R4", "Bus C5", "Bus I2"], slack_known=True
    ).observed_rows(true_grid)
    meas = v_true.index_select(-1, sensor_rows)  # [H, K] sensor measurements

    # the NOMINAL grid the estimator is told about: resistance at the RECORDED value
    nominal_grid = _cigre_no_geometry()
    solver = NetworkSolver(nominal_grid, ORDERS, slack, dtype=DT)

    # baseline: trust the (wrong) nominal grid -> full-state error with the true injections
    with torch.no_grad():
        v_nom = solver.solve(i_true, v_slack)
        baseline_err = (v_nom - v_true).abs().mean().item()

    # calibrate: a single knob on the suspect line's resistance (±50% confidence)
    calib = GridCalibration(
        nominal_grid,
        [CalibrationKnob("line", line_id, "series_resistance_ohm_per_m", 0.5)],
    )
    opt = torch.optim.Adam(calib.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        calib.apply_to_grid()
        solver.assemble()  # Y(Δ) on the live tape
        v_pred = solver.solve(i_true, v_slack)
        resid = (v_pred.index_select(-1, sensor_rows) - meas).abs().pow(2).mean()
        loss = resid + 1e-6 * calib.regularizer()
        loss.backward()
        opt.step()

    recovered = calib.corrections()[f"line:{line_id}:series_resistance_ohm_per_m"]
    with torch.no_grad():
        calib.apply_to_grid()
        solver.assemble()
        calibrated_err = (solver.solve(i_true, v_slack) - v_true).abs().mean().item()

    # Δ recovers the +30% error, and the full-state error collapses far below the nominal cap
    assert abs(recovered - 0.30) < 0.05, recovered
    assert calibrated_err < 0.1 * baseline_err, (calibrated_err, baseline_err)
