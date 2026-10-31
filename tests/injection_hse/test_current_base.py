"""The per-unit CURRENT base — ``NetworkSolver.i_base`` and its reference voltage.

A per-unit system fixes ONE power base per grid and derives every row's current base from it,
``I_base = S_base/V_base``: the same power at 50× the voltage is 1/50 of the current, so the
rows of a higher voltage class must be divided by a SMALLER base. ``injection_hse`` builds that power
base from the network itself, ``S_base(h) = s(h)·V_ref²`` with ``s(h)`` the median singular
value of the solved operator and ``V_ref`` the grid's median rated row voltage, so
``i_base[h, n] = s(h)·V_ref² / V_base[n]``. The invariants under test:

- a SINGLE-VOLTAGE-CLASS grid gets ``s(h)·V_base`` back BIT-FOR-BIT (the reference is then the
  one class), including a grid with unrated rows, which keep base ``1``;
- on a two-class grid the base scales INVERSELY with the class, so the per-unit currents of the
  two classes are more commensurate than the amperes, not less;
- the zero-injection prior the noise-calibrated anchor weighting installs is consequently
  TIGHTEST on the passive rows of the higher class — the most certain rows in the grid —
  instead of loosest;
- the multi-grid bank carries no second formula: it delegates, per member, to this one.
"""

from __future__ import annotations

import math

import pytest
import torch

DT = torch.complex128
ORDERS = [1, 3, 5]


# =============================================================================
# Fixtures: the two-voltage-class benchmark (120 LV rows + 12 MV rows)
# =============================================================================
@pytest.fixture(scope="module")
def two_class(cigre_grid):
    """``dict`` of the operators, states and row classes of the CIGRE LV benchmark.

    The grid spans two voltage classes — the LV feeder rows and the MV rows above the
    substation transformer — which is what makes the class dependence of the base visible.
    """
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel, base_voltage_per_row
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver

    v = solve_harmonic_flow(cigre_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    # the benchmark grid carries no harmonic source: give every harmonic order a finite state
    # so its currents have a scale to compare classes with
    harm = torch.stack(
        [
            v[0] * 0.02 * pos * torch.exp(1j * torch.tensor(pos, dtype=torch.float64))
            for pos in range(1, len(ORDERS))
        ]
    )
    states = torch.stack([v, 1.02 * v, 0.98 * v])
    states = torch.cat([states[:, :1], harm.to(DT).expand(3, -1, -1)], dim=1)
    slack = MeasurementModel(["Bus R4"], slack_known=True).slack_rows(cigre_grid)
    edge_op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=DT)
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT, edge_op=edge_op)
    base_v = base_voltage_per_row(cigre_grid)
    lv, mv = float(base_v.min()), float(base_v.max())
    return {
        "grid": cigre_grid,
        "solver": solver,
        "edge_op": edge_op,
        "states": states,
        "base_v": base_v,
        "slack": slack,
        "lv": lv,
        "mv": mv,
        "is_mv": base_v == mv,
    }


@pytest.fixture(scope="module")
def injection_spec(cigre_grid):
    """The benchmark's :class:`~injection_hse.data.InjectionSpec` — which rows carry a device."""
    from injection_hse.data.scenarios import se_benchmark_scenario_config

    from injection_hse.data.physics import InjectionSpec

    return InjectionSpec(
        cigre_grid, se_benchmark_scenario_config(cigre_grid, n_samples=8, seed=0)
    )


def _class_ratio(magnitude: torch.Tensor, is_mv: torch.Tensor) -> torch.Tensor:
    """``[H]`` RMS over the LOW-class columns divided by the RMS over the HIGH-class ones."""
    rms = magnitude.pow(2).mean(dim=0).sqrt()  # [H, K]
    low = rms[:, ~is_mv].pow(2).mean(dim=-1).sqrt()
    high = rms[:, is_mv].pow(2).mean(dim=-1).sqrt()
    return low / high


# =============================================================================
# (1) the reference voltage
# =============================================================================
def test_the_reference_is_the_median_rated_row_voltage():
    from injection_hse.physics import reference_base_voltage

    v_base = torch.tensor([400.0, 230.0, 230.0, 20000.0], dtype=torch.float64)
    assert float(reference_base_voltage(v_base)) == 230.0
    # unrated rows do not enter the median; a grid with no rated row at all falls back to 1
    with_unrated = torch.tensor([0.0, 0.0, 0.0, 400.0, 400.0], dtype=torch.float64)
    assert float(reference_base_voltage(with_unrated)) == 400.0
    assert float(reference_base_voltage(torch.zeros(4, dtype=torch.float64))) == 1.0


def test_the_reference_is_the_class_of_a_single_class_grid(two_class):
    from injection_hse.physics import reference_base_voltage

    base_v = two_class["base_v"]
    assert float(reference_base_voltage(base_v)) == two_class["lv"]  # 120 LV vs 12 MV
    for value in (two_class["lv"], two_class["mv"]):
        uniform = torch.full_like(base_v, value)
        assert float(reference_base_voltage(uniform)) == value


# =============================================================================
# (2) a single voltage class is bit-for-bit the operator scale times the base voltage
# =============================================================================
@pytest.mark.parametrize("which", ["lv", "mv"])
def test_a_single_voltage_class_grid_keeps_the_plain_operator_base(two_class, which):
    """One class ``->`` ``V_ref ≡ V_base``, so ``s(h)·V_ref²/V_base`` collapses to
    ``s(h)·V_base`` — BIT-FOR-BIT, not within a rounding step: the convention is backward
    compatible on every grid that has only one voltage class."""
    solver, base_v = two_class["solver"], two_class["base_v"]
    uniform = torch.full_like(base_v, two_class[which])
    plain = solver.operator_scale().unsqueeze(-1) * uniform.to(torch.float64)
    assert torch.equal(solver.i_base(uniform), plain)


def test_unrated_rows_keep_base_one_and_do_not_move_the_others(two_class):
    """A row with no rated voltage is left unscaled (base ``1``) and is excluded from the
    reference, so the rated rows of a single-class grid are still bit-for-bit unchanged."""
    solver, base_v = two_class["solver"], two_class["base_v"]
    uniform = torch.full_like(base_v, two_class["lv"])
    uniform[:5] = 0.0
    plain = solver.operator_scale().unsqueeze(-1) * torch.where(
        uniform > 0, uniform, torch.ones_like(uniform)
    ).to(torch.float64)
    i_base = solver.i_base(uniform)
    assert torch.equal(i_base, plain)
    assert torch.equal(
        i_base[:, :5], solver.operator_scale().unsqueeze(-1).expand(-1, 5)
    )


# =============================================================================
# (3) two classes: the base scales inversely, so the per-unit currents come together
# =============================================================================
def test_the_base_is_the_power_base_over_the_row_voltage(two_class):
    solver, base_v = two_class["solver"], two_class["base_v"]
    from injection_hse.physics import reference_base_voltage

    v_ref = reference_base_voltage(base_v)
    expected = solver.operator_scale()[:, None] * v_ref**2 / base_v.to(torch.float64)
    torch.testing.assert_close(solver.i_base(base_v), expected, rtol=1e-15, atol=0.0)
    # the MV rows are divided by the SMALLER base, in exactly the class ratio
    i_base = solver.i_base(base_v)
    is_mv = two_class["is_mv"]
    ratio = i_base[:, is_mv][:, 0] / i_base[:, ~is_mv][:, 0]
    assert float(ratio[0]) == pytest.approx(
        two_class["lv"] / two_class["mv"], rel=1e-12
    )


def test_per_unit_currents_bring_the_two_classes_together(two_class):
    """The half-edge currents — the channels a CT reads, a third of them MV-sourced on this
    grid — are CLOSER between the classes in the per-unit frame than in amperes. A base
    proportional to the row voltage would instead have driven them apart by the squared class
    ratio, which is the disparity the frame exists to remove."""
    solver, edge_op = two_class["solver"], two_class["edge_op"]
    base_v, states, is_mv = two_class["base_v"], two_class["states"], two_class["is_mv"]
    class_ratio = two_class["mv"] / two_class["lv"]

    src = edge_op.edge_src_rows
    edge_is_mv = is_mv.index_select(0, src)  # the CT's own terminal sets its class
    assert int(edge_is_mv.sum()) and int((~edge_is_mv).sum())
    magnitude = edge_op(states).abs()
    base = solver.i_base(base_v).index_select(-1, src)
    proportional = (
        solver.operator_scale().unsqueeze(-1) * base_v.to(torch.float64)
    ).index_select(-1, src)

    amperes = float(_class_ratio(magnitude, edge_is_mv)[0])
    per_unit = float(_class_ratio(magnitude / base, edge_is_mv)[0])
    widened = float(_class_ratio(magnitude / proportional, edge_is_mv)[0])
    # the two conventions differ by exactly the squared class ratio
    assert widened == pytest.approx(per_unit * class_ratio**2, rel=1e-9)
    # and the per-unit frame is the one that brings the classes together
    assert abs(math.log(per_unit)) < abs(math.log(amperes)) < abs(math.log(widened))
    assert 0.1 < per_unit < 10.0
    assert widened > 100.0


# =============================================================================
# (4) the zero-injection prior across the classes
# =============================================================================
def test_the_prior_sigma_is_tightest_on_the_passive_high_class_rows(
    two_class, injection_spec
):
    """``σ_I[h, r] = ρ·aux_scale(h)·i_base[h, r]``: with a power base per row voltage the
    prior installed on the MV rows — passive junctions whose true injection is zero to solver
    roundoff — is the SMALLEST in the grid, in the inverse class ratio. A base proportional to
    the row voltage put the LOOSEST prior of the grid on exactly those rows, under-weighting
    their KCL equation by the squared class ratio."""
    from injection_hse.encoding import build_encoding
    from injection_hse.physics import fit_injection_normalizer

    solver, base_v, states = (
        two_class["solver"],
        two_class["base_v"],
        two_class["states"],
    )
    is_mv, class_ratio = two_class["is_mv"], two_class["mv"] / two_class["lv"]
    i_base = solver.i_base(base_v)
    proportional = solver.operator_scale().unsqueeze(-1) * base_v.to(torch.float64)

    normalizer = fit_injection_normalizer(
        solver,
        states,
        build_encoding("trig"),
        base=i_base,
        injection_spec=injection_spec,
    )
    n = int(base_v.shape[0])
    device_rows = torch.zeros(n, dtype=torch.bool)
    device_rows[injection_spec.device_rows[injection_spec.device_rows_mask]] = True
    non_slack = torch.ones(n, dtype=torch.bool)
    non_slack[solver.slack_rows] = False
    tied = torch.zeros(n, dtype=torch.bool)
    tied[solver.tied_rows] = True
    passive_mv = is_mv & non_slack & ~tied & ~device_rows
    lv_devices = device_rows & ~is_mv
    assert int(passive_mv.sum()) and int(lv_devices.sum())

    truth = solver.injection_targets(states).abs().pow(2).mean(dim=0).sqrt()  # [H, N]
    rms = lambda x, m: float(x[0][m].pow(2).mean().sqrt())  # noqa: E731
    # the passive MV rows really are the most certain rows in the grid
    assert rms(truth, passive_mv) < 1e-6 * rms(truth, lv_devices)

    rho = 1.0
    sigma = rho * normalizer.aux_scale[:, None] * i_base
    widened = rho * normalizer.aux_scale[:, None] * proportional
    assert rms(sigma, passive_mv) / rms(sigma, lv_devices) == pytest.approx(
        1.0 / class_ratio, rel=1e-9
    )
    assert rms(widened, passive_mv) / rms(widened, lv_devices) == pytest.approx(
        class_ratio, rel=1e-9
    )
    assert rms(widened, passive_mv) / rms(sigma, passive_mv) == pytest.approx(
        class_ratio**2, rel=1e-9
    )


# =============================================================================
# (5) the multi-grid bank delegates
# =============================================================================
def test_the_bank_and_the_solver_share_one_current_base(two_class):
    """The bank keeps no second formula: a two-class member's base is bit-for-bit the one the
    single-grid solver builds from the same operator and base voltages."""
    from types import SimpleNamespace

    from injection_hse.physics import MultiGridPhysicsBank

    grid, base_v = two_class["grid"], two_class["base_v"]
    source = SimpleNamespace(
        orders=ORDERS,
        grids=[SimpleNamespace(key="two-class")],
        bundle=lambda i: SimpleNamespace(
            grid=grid, base_voltage=base_v, key="two-class"
        ),
    )
    bank = MultiGridPhysicsBank(source, dtype=DT)
    assert torch.equal(bank.i_base(0), bank.solver(0).i_base(base_v))
    from injection_hse.physics import reference_base_voltage

    v_ref = reference_base_voltage(base_v)
    expected = (
        bank.solver(0).operator_scale()[:, None] * v_ref**2 / base_v.to(torch.float64)
    )
    torch.testing.assert_close(bank.i_base(0), expected, rtol=1e-15, atol=0.0)
