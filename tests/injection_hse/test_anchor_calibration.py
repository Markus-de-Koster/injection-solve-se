"""Noise-calibrated anchor weights of the measurement-anchored injection decode.

The anchored solve ``min ‖Y·V − Î‖² + Σ w_r|V_r − y_r|² + Σ w_k|(C·V)_k − c_k|²`` is the
Gaussian MAP estimate whose correct weights are inverse variances: with the physics block at
weight 1, ``w = σ_I² / σ_meas²`` per channel. ``PhysicsConfig.anchor_weighting =
"noise_calibrated"`` installs exactly those weights (``NetworkSolver.set_anchor_calibration``)
from the fitted scales and the sensor-noise model, and turns the anchor weights into
dimensionless multipliers on them. Checked here: ``manual`` is byte-identical to the previous
construction; a calibrated clean sensor is a (near-)hard anchor; the prior sigma moves the
solution monotonically between the prediction and the measurements; the CT and voltage
weights share one dimension (the formula is asserted directly); the module wiring installs
the same calibration at construction and after a checkpoint reload; the multi-grid bank
converts the per-unit calibration per member, keeping the RATIO of that member's bases (the
plain operator scale on a single-voltage-class member). The prior scale itself is the RMS
injection of the rows that CARRY A DEVICE. The float64 gradcheck of the calibrated solve
lives in ``tests/differentiability``.
"""

from __future__ import annotations

import math

import pytest
import torch

DT = torch.complex128
ORDERS = [1, 3, 5]


# =============================================================================
# Solver-level fixtures
# =============================================================================
def _setup(cigre_grid, **anchor):
    """``(states [B,H,N], slack rows, edge operator, solver)`` on the CIGRE LV benchmark."""
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import MeasurementModel
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver

    v = solve_harmonic_flow(cigre_grid, ORDERS, slack="ideal", dtype=DT).v  # [H, N]
    # give the harmonic orders a non-zero state (the benchmark grid carries no harmonic
    # sources), so every order has a finite RMS injection to calibrate against
    harm = torch.stack(
        [
            v[0] * 0.02 * pos * torch.exp(1j * torch.tensor(pos, dtype=torch.float64))
            for pos in range(1, len(ORDERS))
        ]
    )
    v = torch.cat([v[:1], harm.to(DT)])
    vb = torch.stack([v, 1.02 * v, 0.98 * v])  # [B, H, N]
    slack = MeasurementModel(["Bus R4"], slack_known=True).slack_rows(cigre_grid)
    op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=DT)
    solver = NetworkSolver(cigre_grid, ORDERS, slack, dtype=DT, edge_op=op, **anchor)
    return vb, slack, op, solver


def _scales(solver, op, vb):
    """The RMS ``|·|`` per order of the state / injections / branch currents ``[H]``."""
    i_true = solver.injection_targets(vb)
    i_edges = op(vb)
    rms = lambda x: x.abs().pow(2).mean(dim=(0, -1)).sqrt().to(torch.float64)  # noqa: E731
    return rms(vb), rms(i_true), rms(i_edges)


def _clean_calibration(solver, op, vb, *, rho: float, clean: float):
    """Install the noise-free calibration (every sensor at ``clean`` relative sigma)."""
    from injection_hse.physics import clean_sensor_sigma

    v_typ, i_typ, e_typ = _scales(solver, op, vb)
    solver.set_anchor_calibration(
        sigma_injection=rho * i_typ,
        sigma_voltage=clean_sensor_sigma(v_typ, clean),
        sigma_current=clean_sensor_sigma(e_typ, clean),
    )
    return solver


def _partial_mask(b: int, n: int, slack_rows: torch.Tensor) -> torch.Tensor:
    """Every other row observed, the slack always (the decode's boundary)."""
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, ::2] = True
    mask[:, slack_rows] = True
    return mask


# =============================================================================
# (1) manual = byte-identical to the direct pgml construction
# =============================================================================
def test_manual_weighting_matches_the_direct_solve_anchored_construction(cigre_grid):
    """A solver without a calibration weights exactly as ``w_v = a·s(h)²`` / ``w_i = a_i``."""
    from pgml.solver import solve_anchored

    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=2.0, anchor_i_weight=0.5)
    assert solver.anchor_calibrated is False
    b, n, e = vb.shape[0], vb.shape[-1], op.n_edges
    i_pred = 0.5 * solver.injection_targets(vb)  # deliberately wrong prior
    i_edges = op(vb)
    v_slack = vb.select(-2, ORDERS.index(1)).index_select(-1, slack)
    vmask = _partial_mask(b, n, solver.slack_rows)
    emask = torch.zeros(b, e, dtype=torch.bool)
    emask[:, ::3] = True
    out = solver.solve(
        i_pred, v_slack, v_meas=vb, v_mask=vmask, i_meas=i_edges, edge_mask=emask
    )

    scale = solver.operator_scale()  # [H]
    expected = []
    for pos, h in enumerate(ORDERS):
        y = (solver.y_fundamental[0] if h == 1 else solver.y_harmonic[pos - 1]).to(DT)
        fixed = solver.slack_rows if h == 1 else None
        expected.append(
            solve_anchored(
                y,
                i_pred[:, pos],
                row_weight=vmask.to(torch.float64) * (2.0 * scale[pos] ** 2),
                row_target=vb[:, pos],
                op=solver._c_map[pos].to(DT),
                op_weight=emask.to(torch.float64) * 0.5,
                op_target=i_edges[:, pos],
                fixed_rows=fixed,
                v_fixed=v_slack if fixed is not None else None,
            )
        )
    assert torch.equal(out, torch.stack(expected, dim=1))


def test_clearing_the_calibration_restores_the_manual_solve_bitwise(cigre_grid):
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=2.0, anchor_i_weight=0.5)
    b, n, e = vb.shape[0], vb.shape[-1], op.n_edges
    args = dict(
        v_meas=vb,
        v_mask=_partial_mask(b, n, solver.slack_rows),
        i_meas=op(vb),
        edge_mask=torch.ones(b, e, dtype=torch.bool),
    )
    i_pred = 0.5 * solver.injection_targets(vb)
    v_slack = vb.select(-2, ORDERS.index(1)).index_select(-1, slack)
    manual = solver.solve(i_pred, v_slack, **args)
    _clean_calibration(solver, op, vb, rho=1.0, clean=1e-3)
    assert solver.anchor_calibrated
    calibrated = solver.solve(i_pred, v_slack, **args)
    assert not torch.equal(calibrated, manual)
    solver.clear_anchor_calibration()
    assert solver.anchor_calibrated is False
    assert torch.equal(solver.solve(i_pred, v_slack, **args), manual)


# =============================================================================
# (2) clean sensors = a hard anchor; the prior sigma moves the solution monotonically
# =============================================================================
def test_clean_calibrated_anchor_reproduces_the_observed_rows(cigre_grid):
    """Noise off + a tiny clean-sensor sigma: the observed rows reproduce the measurements
    to within the implied tolerance, whatever the injection prior says."""
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1.0)
    clean = 1e-6
    _clean_calibration(solver, op, vb, rho=1.0, clean=clean)
    b, n = vb.shape[0], vb.shape[-1]
    vmask = _partial_mask(b, n, solver.slack_rows)
    v_slack = vb.select(-2, ORDERS.index(1)).index_select(-1, slack)
    # a zero-injection prior: the anchors alone must carry the observed rows
    out = solver.solve(
        vb.new_zeros(b, len(ORDERS), n), v_slack, v_meas=vb, v_mask=vmask
    )
    v_typ = _scales(solver, op, vb)[0]  # [H]
    err = (out - vb).abs()[vmask.unsqueeze(1).expand_as(out)]
    tol = (
        v_typ.max() * 1e-3
    )  # the hard-anchor limit, far inside 1e-6 · σ-ratio² pull-back
    assert float(err.max()) < float(tol)


def test_the_prior_sigma_moves_the_solution_monotonically(cigre_grid):
    """A larger ``ρ`` (a prior that trusts the injections less) pulls the observed rows closer
    to the measurements, a smaller one leaves them nearer the prior's own solve."""
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1.0)
    b, n = vb.shape[0], vb.shape[-1]
    vmask = _partial_mask(b, n, solver.slack_rows)
    v_slack = vb.select(-2, ORDERS.index(1)).index_select(-1, slack)
    i_pred = 0.5 * solver.injection_targets(vb)
    sel = vmask.unsqueeze(1).expand(b, len(ORDERS), n)
    distances = []
    for rho in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        _clean_calibration(solver, op, vb, rho=rho, clean=1e-2)
        out = solver.solve(i_pred, v_slack, v_meas=vb, v_mask=vmask)
        distances.append(float((out - vb).abs()[sel].mean()))
    assert all(a > b_ for a, b_ in zip(distances, distances[1:])), distances


# =============================================================================
# (3) the CT weight shares the voltage weight's dimension
# =============================================================================
def test_calibrated_ct_and_voltage_weights_share_one_dimension():
    """A 0.1 % CT and a 0.1 % voltage sensor (no floor) at typical magnitudes ``I_typ`` /
    ``V_typ``: ``w_v / w_i = (I_typ / V_typ)²`` — the ratio of the typical magnitudes
    squared, i.e. the admittance² that puts the voltage term into the current term's units
    (both are then ``σ_I²`` over the SAME relative error)."""
    from injection_hse.physics import anchor_measurement_sigma

    rel = 1e-3
    v_typ = torch.tensor([230.0, 2.3, 0.7], dtype=torch.float64)
    i_typ = torch.tensor([40.0, 1.2, 0.3], dtype=torch.float64)
    sigma_inj = torch.tensor([50.0, 2.0, 0.5], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    sigma_v = anchor_measurement_sigma(
        v_typ, mag_sigma=torch.full((1,), rel), phase_sigma_rad=zero
    )  # [H, 1]
    sigma_i = anchor_measurement_sigma(
        i_typ, mag_sigma=torch.full((1,), rel), phase_sigma_rad=zero
    )
    w_v = (sigma_inj[:, None] / sigma_v) ** 2
    w_i = (sigma_inj[:, None] / sigma_i) ** 2
    torch.testing.assert_close(w_v / w_i, ((i_typ / v_typ) ** 2)[:, None])
    # and the per-component sigma is the isotropic half of the reading error
    torch.testing.assert_close(sigma_v[:, 0], math.sqrt(0.5) * rel * v_typ)


def test_measurement_sigma_adds_the_order_scaled_floor():
    """``σ² = ½(σm² + σp²)·typ² + (floor·order_floor)²`` per channel and order."""
    from injection_hse.physics import anchor_measurement_sigma

    typ = torch.tensor([230.0, 2.3], dtype=torch.float64)
    mag = torch.tensor([1e-3, 2e-3], dtype=torch.float64)
    phase = torch.tensor([1e-3, 0.0], dtype=torch.float64)
    floor = torch.tensor([0.1, 0.2], dtype=torch.float64)
    order_floor = torch.tensor([1.0, 0.1], dtype=torch.float64)
    out = anchor_measurement_sigma(
        typ,
        mag_sigma=mag,
        phase_sigma_rad=phase,
        floor_sigma=floor,
        order_floor_scale=order_floor,
    )
    expected = torch.sqrt(
        0.5 * (mag**2 + phase**2)[None, :] * typ[:, None] ** 2
        + (floor[None, :] * order_floor[:, None]) ** 2
    )
    torch.testing.assert_close(out, expected)


def test_the_solver_installs_and_validates_the_calibration(cigre_grid):
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1.0, anchor_i_weight=1.0)
    v_typ, i_typ, e_typ = _scales(solver, op, vb)
    n, e = vb.shape[-1], op.n_edges
    solver.set_anchor_calibration(
        sigma_injection=i_typ,
        sigma_voltage=(1e-3 * v_typ)[:, None].expand(-1, n),
        sigma_current=(1e-3 * e_typ)[:, None].expand(-1, e),
    )
    assert solver.anchor_voltage_calibration.shape == (len(ORDERS), n)
    assert solver.anchor_current_calibration.shape == (len(ORDERS), e)
    assert solver.anchor_voltage_calibration.dtype == torch.float64
    torch.testing.assert_close(
        solver.anchor_voltage_calibration[:, 0], (i_typ / (1e-3 * v_typ)) ** 2
    )
    with pytest.raises(ValueError, match="sigma_voltage"):
        solver.set_anchor_calibration(
            sigma_injection=i_typ, sigma_voltage=torch.ones(len(ORDERS), n + 1)
        )
    with pytest.raises(ValueError, match="positive"):
        solver.set_anchor_calibration(
            sigma_injection=i_typ, sigma_voltage=torch.zeros(len(ORDERS), 1)
        )
    # a calibrated solve asked to anchor currents without a current sigma fails loudly
    solver.set_anchor_calibration(
        sigma_injection=i_typ, sigma_voltage=1e-3 * v_typ[:, None]
    )
    b = vb.shape[0]
    with pytest.raises(ValueError, match="current sigma"):
        solver.solve(
            solver.injection_targets(vb),
            vb.select(-2, 0).index_select(-1, slack),
            v_meas=vb,
            v_mask=torch.ones(b, n, dtype=torch.bool),
            i_meas=op(vb),
            edge_mask=torch.ones(b, e, dtype=torch.bool),
        )


def test_calibrated_weights_move_with_the_solver(cigre_grid):
    vb, slack, op, solver = _setup(cigre_grid, anchor_v_weight=1.0, anchor_i_weight=1.0)
    _clean_calibration(solver, op, vb, rho=1.0, clean=1e-3)
    solver.to("cpu")
    assert solver.anchor_voltage_calibration.device.type == "cpu"
    assert solver.anchor_current_calibration.device.type == "cpu"


# =============================================================================
# The module wiring: installed at construction, re-derived on a checkpoint reload
# =============================================================================
@pytest.fixture(scope="module")
def cal_setup(cigre_grid, se_dataset):
    """Source + fitted normalizers + branch-current operator (shared by the module tests)."""
    from injection_hse.config import DataConfig
    from injection_hse.data import StateCollate, build_data_source
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    orders = list(source.orders)
    enc = build_encoding("trig")
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    normalizer = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    edge_op = BranchCurrentOperator(cigre_grid, orders, dtype=torch.complex64)
    i_train = edge_op(train_v.to(torch.complex64))
    edge_normalizer = HarmonicNormalizer().fit(enc.encode(i_train).transpose(-3, -2))
    batch = StateCollate()([source.train_dataset()[i] for i in range(4)])
    return {
        "source": source,
        "orders": orders,
        "encoding": enc,
        "train_v": train_v,
        "normalizer": normalizer,
        "edge_op": edge_op,
        "edge_normalizer": edge_normalizer,
        "batch": batch,
    }


def _module(
    setup,
    grid,
    *,
    weighting: str,
    noise: bool,
    rho: float = 1.0,
    normalizer=None,
    instrument=None,
):
    """An injection-decode, current-anchored :class:`~injection_hse.train.SEModule`."""
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        MaskStageConfig,
        MeasurementConfig,
        NoiseConfig,
        PhysicsConfig,
    )
    from injection_hse.data import base_voltage_per_row
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    orders = setup["orders"]
    solver = NetworkSolver.from_grid(
        grid, orders, anchor_v_weight=1.0, anchor_i_weight=1.0, edge_op=setup["edge_op"]
    )
    inj_norm = fit_injection_normalizer(solver, setup["train_v"], setup["encoding"])
    cfg = ExperimentConfig(
        name="anchor-calibration",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(
            decode="injection",
            anchor_voltage_weight=1.0,
            anchor_current_weight=1.0,
            anchor_weighting=weighting,
            prior_injection_rel_sigma=rho,
        ),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], measure_current=True),
        noise=NoiseConfig(enabled=noise, seed=3, instrument=instrument),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=setup["source"].n_nodes,
        normalizer=normalizer if normalizer is not None else setup["normalizer"],
        graph=setup["source"].graph(),
        base_v=base_voltage_per_row(grid),
        solver=solver,
        inj_normalizer=inj_norm,
        grid=grid,
        edge_op=setup["edge_op"],
        edge_normalizer=setup["edge_normalizer"],
    )
    module.eval()
    return module


def test_manual_weighting_leaves_the_solver_uncalibrated(cigre_grid, cal_setup):
    module = _module(cal_setup, cigre_grid, weighting="manual", noise=True)
    assert module.solver.anchor_calibrated is False


def test_the_module_installs_the_noise_models_own_sigmas(cigre_grid, cal_setup):
    """Calibrated + noise on: the installed weights are ``σ_I² / σ_meas²`` with ``σ_meas``
    built from the SAME per-row / per-edge sigmas the noise model draws from, each at its
    own channel's typical magnitude."""
    from injection_hse.physics import anchor_measurement_sigma
    from injection_hse.train.anchoring import channel_scale

    rho = 0.3
    module = _module(
        cal_setup, cigre_grid, weighting="noise_calibrated", noise=True, rho=rho
    )
    solver = module.solver
    assert solver.anchor_calibrated
    n, e = cal_setup["source"].n_nodes, cal_setup["edge_op"].n_edges
    h = len(cal_setup["orders"])
    assert solver.anchor_voltage_calibration.shape == (h, n)
    assert solver.anchor_current_calibration.shape == (h, e)

    s_inj = rho * module.inj_normalizer.aux_scale.to(torch.float64)
    sigma_v = anchor_measurement_sigma(
        channel_scale(module.normalizer),
        mag_sigma=module._noise.mag_sigma,
        phase_sigma_rad=module._noise.phase_sigma_rad,
        floor_sigma=module._noise.floor_sigma,
        order_floor_scale=module._noise.order_floor_scale,
    )
    sigma_i = anchor_measurement_sigma(
        channel_scale(module.edge_normalizer),
        mag_sigma=module._edge_noise.mag_sigma,
        phase_sigma_rad=module._edge_noise.phase_sigma_rad,
        floor_sigma=module._edge_noise.floor_sigma,
        order_floor_scale=module._edge_noise.order_floor_scale,
    )
    torch.testing.assert_close(
        solver.anchor_voltage_calibration, (s_inj[:, None] / sigma_v) ** 2
    )
    torch.testing.assert_close(
        solver.anchor_current_calibration, (s_inj[:, None] / sigma_i) ** 2
    )
    assert sigma_v.shape == (h, n) and sigma_i.shape == (h, e)
    # the range floor is order-scaled on the voltage rows (harmonic_range) and the reading
    # error shrinks with the per-order typical magnitude, so the weights differ per order
    assert (
        len({round(float(w), 6) for w in solver.anchor_voltage_calibration[:, 0]}) == h
    )


def test_noise_off_uses_the_clean_sensor_sigma(cigre_grid, cal_setup):
    from injection_hse.train.anchoring import channel_scale

    module = _module(cal_setup, cigre_grid, weighting="noise_calibrated", noise=False)
    solver = module.solver
    clean = module.cfg.physics.clean_sensor_rel_sigma
    s_inj = module.inj_normalizer.aux_scale.to(torch.float64)
    v_typ = channel_scale(module.normalizer)  # [H, N]: each row's own magnitude
    torch.testing.assert_close(
        solver.anchor_voltage_calibration, (s_inj[:, None] / (clean * v_typ)) ** 2
    )


def _pooled_normalizer(normalizer):
    """A copy of ``normalizer`` as a checkpoint written before the per-row scale existed."""
    from injection_hse.normalization import HarmonicNormalizer

    state = normalizer.state_dict()
    state.pop("row_scale")
    legacy = HarmonicNormalizer()
    legacy.load_state_dict(state)
    assert legacy.row_scale is None
    return legacy


def test_a_normalizer_without_row_scale_keeps_the_pooled_calibration(
    cigre_grid, cal_setup
):
    """Old checkpoints reload with the calibration they were trained with."""
    module = _module(
        cal_setup,
        cigre_grid,
        weighting="noise_calibrated",
        noise=False,
        normalizer=_pooled_normalizer(cal_setup["normalizer"]),
    )
    clean = module.cfg.physics.clean_sensor_rel_sigma
    s_inj = module.inj_normalizer.aux_scale.to(torch.float64)
    v_typ = module.normalizer.cartesian_scale.to(torch.float64)
    assert module.solver.anchor_voltage_calibration.shape[-1] == 1
    torch.testing.assert_close(
        module.solver.anchor_voltage_calibration[:, 0], (s_inj / (clean * v_typ)) ** 2
    )


def test_row_scale_is_each_rows_rms_magnitude(cal_setup):
    """``row_scale [H, N]`` is the RMS ``|V|`` of every row; the pooled scale mixes them."""
    v = cal_setup["train_v"].to(torch.complex128)
    expected = v.abs().pow(2).mean(dim=0).sqrt()  # [H, N]
    row_scale = cal_setup["normalizer"].row_scale.to(torch.float64)
    torch.testing.assert_close(row_scale, expected, rtol=1e-5, atol=1e-8)
    # rows of two voltage classes: the fundamental's pooled scale fits neither
    fund = row_scale[0]
    assert float(fund.max() / fund.min()) > 10.0


def test_a_per_order_prior_sigma_scales_each_order(cigre_grid, cal_setup):
    """``prior_injection_rel_sigma`` as a list: ``σ_I(h) = ρ(h)·s_I(h)``."""
    h = len(cal_setup["orders"])
    rho = [0.05 * (k + 1) for k in range(h)]
    module = _module(
        cal_setup, cigre_grid, weighting="noise_calibrated", noise=True, rho=rho
    )
    one = _module(cal_setup, cigre_grid, weighting="noise_calibrated", noise=True)
    ratio = module.solver.anchor_voltage_calibration / one.solver.anchor_voltage_calibration
    expected = torch.tensor(rho, dtype=torch.float64).pow(2)[:, None].expand_as(ratio)
    torch.testing.assert_close(ratio, expected)


def test_a_per_order_prior_sigma_needs_one_value_per_order():
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        MaskStageConfig,
        PhysicsConfig,
    )

    stages = [MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)]
    with pytest.raises(ValueError, match="one per"):
        ExperimentConfig(
            name="x",
            orders=[1, 3, 5],
            data=DataConfig(dataset_path="x"),
            model=DNNConfig(hidden_dims=[8]),
            physics=PhysicsConfig(prior_injection_rel_sigma=[0.1, 0.2]),
            curriculum=CurriculumConfig(stages=stages),
        )
    with pytest.raises(ValueError, match="positive"):
        PhysicsConfig(prior_injection_rel_sigma=[0.1, 0.0, 0.2])


def test_the_calibration_changes_the_decode_and_survives_a_reload(
    cigre_grid, cal_setup, tmp_path
):
    """The calibrated module decodes differently from the manual one at the same weights, and
    a checkpoint reload rebuilds the identical calibration (config + normalizers travel)."""
    from injection_hse.masking import FullMasker
    from injection_hse.train import SEModule

    manual = _module(cal_setup, cigre_grid, weighting="manual", noise=False)
    calibrated = _module(
        cal_setup, cigre_grid, weighting="noise_calibrated", noise=False
    )
    calibrated.load_state_dict(
        manual.state_dict()
    )  # same weights, only the decode differs
    for m in (manual, calibrated):
        m.set_masker(FullMasker())
        m.set_stage_anchor(v_weight=1.0, i_weight=1.0)
    batch = cal_setup["batch"]
    with torch.no_grad():
        pred_manual = manual.infer(batch)["pred_v"]
        pred_cal = calibrated.infer(batch)["pred_v"]
    assert not torch.equal(pred_manual, pred_cal)

    ckpt = {"state_dict": calibrated.state_dict()}
    calibrated.on_save_checkpoint(ckpt)
    path = tmp_path / "calibrated.ckpt"
    torch.save(ckpt, path)
    reloaded = SEModule.from_checkpoint(
        path, grid=cigre_grid, source=cal_setup["source"]
    )
    assert reloaded.solver.anchor_calibrated
    torch.testing.assert_close(
        reloaded.solver.anchor_voltage_calibration,
        calibrated.solver.anchor_voltage_calibration,
    )
    torch.testing.assert_close(
        reloaded.solver.anchor_current_calibration,
        calibrated.solver.anchor_current_calibration,
    )
    reloaded.set_masker(FullMasker())
    with torch.no_grad():
        pred_reloaded = reloaded.infer(batch)["pred_v"]
    torch.testing.assert_close(pred_reloaded, pred_cal)


def test_the_calibration_fields_are_behaviour_fields():
    """A config swap between passes may not change the installed calibration silently."""
    from injection_hse.train import CFG_BEHAVIOUR_FIELDS

    for name in (
        "physics.anchor_weighting",
        "physics.prior_injection_rel_sigma",
        "physics.clean_sensor_rel_sigma",
    ):
        assert name in CFG_BEHAVIOUR_FIELDS


# =============================================================================
# The multi-grid bank: per-member conversion, the weight keeps the base RATIO
# =============================================================================
class _StubSource:
    """The attributes :class:`~injection_hse.physics.MultiGridPhysicsBank` reads of a data source."""

    def __init__(self, grids, base_voltages) -> None:
        from types import SimpleNamespace

        self._grids = list(grids)
        self.orders = [1, 3]
        self._bundles = [
            SimpleNamespace(grid=g, base_voltage=b, key=f"g{i}")
            for i, (g, b) in enumerate(zip(self._grids, base_voltages))
        ]

    @property
    def grids(self):
        return self._grids

    def bundle(self, index: int):
        return self._bundles[int(index)]


def test_multigrid_config_accepts_the_calibrated_weighting():
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        ExperimentConfig,
        GNNConfig,
        MaskStageConfig,
        MeasurementConfig,
        MultiGridConfig,
        PhysicsConfig,
    )

    cfg = ExperimentConfig(
        name="mg-cal",
        orders=[1, 3],
        data=DataConfig(
            dataset_path="unused", multigrid=MultiGridConfig(task="random")
        ),
        model=GNNConfig(hidden_dim=8, num_layers=1),
        physics=PhysicsConfig(
            decode="injection",
            anchor_voltage_weight=1.0,
            anchor_weighting="noise_calibrated",
            prior_injection_rel_sigma=0.2,
            clean_sensor_rel_sigma=1e-3,
        ),
        measurement=MeasurementConfig(slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="full", kind="full", target_loss=1e-9, max_epochs=1
                )
            ]
        ),
    )
    assert cfg.physics.anchor_weighting == "noise_calibrated"


def test_the_bank_converts_the_per_unit_calibration_per_member():
    """Per member ``σ_I = σ_I_pu · i_base`` and ``σ_y = σ_y_pu · V_base``: the voltage weight
    keeps the RATIO of the two bases, ``s(h)·V_ref²/V_base[r]²`` — the member's own operator
    scale, and its own voltage classes the way a per-unit weighting must — and members built
    AFTER the call inherit the same calibration."""
    from pgml.grids import synthetic_feeder

    from injection_hse.physics import MultiGridPhysicsBank, reference_base_voltage

    grids = [synthetic_feeder(4, n_feeders=2), synthetic_feeder(5, n_feeders=2)]
    bases = []
    for g in grids:
        from pgml.assembly import node_phase_index

        n = node_phase_index(g).size
        bases.append(torch.linspace(230.0, 400.0, n, dtype=torch.float64))
    bank = MultiGridPhysicsBank(
        _StubSource(grids, bases), dtype=torch.complex128, anchor_v_weight=1.0
    )
    bank.solver(0)  # member 0 built before the calibration, member 1 after
    sigma_inj = torch.tensor([0.5, 0.05], dtype=torch.float64)
    sigma_v = torch.tensor([1e-3, 1e-4], dtype=torch.float64)
    bank.set_anchor_calibration(
        sigma_injection=sigma_inj, sigma_voltage=sigma_v, voltage_per_unit=True
    )
    for i in range(2):
        solver = bank.solver(i)
        assert solver.anchor_calibrated
        cal = solver.anchor_voltage_calibration  # [H, N]
        v_base = bases[i]
        v_ref = reference_base_voltage(v_base)
        ratio = solver.operator_scale()[:, None] * (v_ref / v_base) ** 2  # [H, N]
        torch.testing.assert_close(
            cal, ((sigma_inj[:, None] * ratio) / sigma_v[:, None]) ** 2
        )
        # the base ratio IS the operator scale on a single-voltage-class member
        assert not torch.allclose(cal, cal[:, :1].expand_as(cal))
    bank.clear_anchor_calibration()
    assert all(not bank.solver(i).anchor_calibrated for i in range(2))


def test_a_single_class_member_keeps_row_uniform_calibrated_weights():
    """One voltage class per member ``->`` the base ratio is exactly ``s(h)`` and the weight is
    the base-free ``(σ_I_pu·s(h)/σ_y_pu)²`` at every row: the frame leaves the anchoring of a
    single-voltage-class grid untouched, which is what makes the convention backward
    compatible."""
    from pgml.assembly import node_phase_index
    from pgml.grids import synthetic_feeder

    from injection_hse.physics import MultiGridPhysicsBank

    grids = [synthetic_feeder(4, n_feeders=2)]
    n = node_phase_index(grids[0]).size
    bases = [torch.full((n,), 230.0, dtype=torch.float64)]
    bank = MultiGridPhysicsBank(
        _StubSource(grids, bases), dtype=torch.complex128, anchor_v_weight=1.0
    )
    sigma_inj = torch.tensor([0.5, 0.05], dtype=torch.float64)
    sigma_v = torch.tensor([1e-3, 1e-4], dtype=torch.float64)
    bank.set_anchor_calibration(
        sigma_injection=sigma_inj, sigma_voltage=sigma_v, voltage_per_unit=True
    )
    solver = bank.solver(0)
    cal = solver.anchor_voltage_calibration  # [H, N]
    expected = ((sigma_inj * solver.operator_scale()) / sigma_v) ** 2  # [H], base-free
    torch.testing.assert_close(cal, expected[:, None].expand_as(cal))


@pytest.fixture(scope="module")
def tiny_solver_states(cigre_grid):
    vb, _slack, _op, solver = _setup(cigre_grid)
    return solver, vb


class TestDeviceRowPriorScale:
    """The prior sigma reads the RMS injection of the rows that carry a DEVICE.

    The slack row's ``Y·V`` entry is the whole feeder current and a passive junction's is zero
    to solver roundoff, so pooling either into the statistic misstates how large a device
    injection is. Without an injection spec — a dataset with no per-device injection — the
    statistic falls back to every non-slack row.
    """

    def test_fit_records_aux_scale_below_pooled(self, tiny_solver_states):
        from injection_hse.encoding import build_encoding
        from injection_hse.physics import fit_injection_normalizer

        solver, states = tiny_solver_states
        normalizer = fit_injection_normalizer(solver, states, build_encoding("trig"))
        assert normalizer.aux_scale is not None
        assert normalizer.aux_scale.shape == normalizer.cartesian_scale.shape
        # without an injection spec: the RMS injection magnitude over the NON-SLACK rows
        i_true = solver.injection_targets(states)
        keep = torch.ones(i_true.shape[-1], dtype=torch.bool)
        keep[solver.slack_rows.cpu()] = False
        expected = (
            i_true[..., keep].abs().pow(2).mean(dim=(0, 2)).sqrt().to(torch.float64)
        )
        torch.testing.assert_close(normalizer.aux_scale, expected.cpu())
        assert not torch.equal(
            normalizer.aux_scale, normalizer.cartesian_scale.to(torch.float64)
        )

    def test_an_injection_spec_restricts_the_scale_to_the_device_rows(
        self, cigre_grid, tiny_solver_states
    ):
        """With the spec the statistic is exactly the device-row RMS. At the fundamental every
        estimable row without a device is an exact zero, so dropping them raises the scale by
        ``sqrt(n_estimable / n_device)`` — the dilution one scalar suffers over a population
        that is mostly zeros. Switch-tied rows are outside the injection space."""
        from injection_hse.data.scenarios import se_benchmark_scenario_config

        from injection_hse.data.physics import InjectionSpec
        from injection_hse.encoding import build_encoding
        from injection_hse.physics import fit_injection_normalizer

        solver, states = tiny_solver_states
        spec = InjectionSpec(
            cigre_grid, se_benchmark_scenario_config(cigre_grid, n_samples=8, seed=0)
        )
        encoding = build_encoding("trig")
        pooled = fit_injection_normalizer(solver, states, encoding)
        fitted = fit_injection_normalizer(solver, states, encoding, injection_spec=spec)

        i_true = solver.injection_targets(states)
        n = i_true.shape[-1]
        device_rows = torch.zeros(n, dtype=torch.bool)
        device_rows[spec.device_rows[spec.device_rows_mask]] = True
        device_rows[solver.slack_rows.cpu()] = False
        expected = (
            i_true[..., device_rows]
            .abs()
            .pow(2)
            .mean(dim=(0, 2))
            .sqrt()
            .to(torch.float64)
        )
        torch.testing.assert_close(fitted.aux_scale, expected.cpu())
        # the FEATURE statistics still describe every row, device or not
        torch.testing.assert_close(fitted.cartesian_scale, pooled.cartesian_scale)

        n_estimable = n - int(solver.slack_rows.numel()) - int(solver.tied_rows.numel())
        n_device = int(device_rows.sum())
        assert 0 < n_device < n_estimable
        assert float(fitted.aux_scale[0] / pooled.aux_scale[0]) == pytest.approx(
            math.sqrt(n_estimable / n_device), rel=1e-9
        )

    def test_state_dict_roundtrip_and_legacy_fallback(self, tiny_solver_states):
        import torch

        from injection_hse.encoding import build_encoding
        from injection_hse.normalization import HarmonicNormalizer
        from injection_hse.physics import fit_injection_normalizer

        solver, states = tiny_solver_states
        normalizer = fit_injection_normalizer(solver, states, build_encoding("trig"))
        state = normalizer.state_dict()
        clone = HarmonicNormalizer()
        clone.load_state_dict(state)
        assert torch.equal(clone.aux_scale, normalizer.aux_scale)
        # a checkpoint written before the masked scale existed has no such key
        legacy = {k: v for k, v in state.items() if k != "aux_scale"}
        old = HarmonicNormalizer()
        old.load_state_dict(legacy)
        assert old.aux_scale is None


def test_the_module_installs_the_instrument_sigmas(cigre_grid, cal_setup, tmp_path):
    """With the datasheet instrument model the weights use its per-reading sigma, at each
    channel's typical magnitude; a checkpoint reload rebuilds the same model and weights."""
    from injection_hse.config import InstrumentErrorConfig
    from injection_hse.data import InstrumentNoise
    from injection_hse.masking import FullMasker
    from injection_hse.train import SEModule
    from injection_hse.train.anchoring import channel_scale

    rho = 0.3
    module = _module(
        cal_setup,
        cigre_grid,
        weighting="noise_calibrated",
        noise=True,
        rho=rho,
        instrument=InstrumentErrorConfig(),
    )
    assert isinstance(module._noise, InstrumentNoise)
    assert isinstance(module._edge_noise, InstrumentNoise)
    assert module._edge_noise.quantity == "current"
    solver = module.solver
    s_inj = rho * module.inj_normalizer.aux_scale.to(torch.float64)
    sigma_v = module._noise.reading_sigma(channel_scale(module.normalizer))
    sigma_i = module._edge_noise.reading_sigma(channel_scale(module.edge_normalizer))
    torch.testing.assert_close(
        solver.anchor_voltage_calibration, (s_inj[:, None] / sigma_v) ** 2
    )
    torch.testing.assert_close(
        solver.anchor_current_calibration, (s_inj[:, None] / sigma_i) ** 2
    )

    module.set_masker(FullMasker())
    module.set_stage_anchor(v_weight=1.0, i_weight=1.0)
    with torch.no_grad():
        out = module.infer(cal_setup["batch"])
    assert torch.isfinite(out["pred_v"].abs()).all()

    ckpt = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(ckpt)
    path = tmp_path / "instrument.ckpt"
    torch.save(ckpt, path)
    reloaded = SEModule.from_checkpoint(
        path, grid=cigre_grid, source=cal_setup["source"]
    )
    assert isinstance(reloaded._noise, InstrumentNoise)
    torch.testing.assert_close(
        reloaded.solver.anchor_voltage_calibration, solver.anchor_voltage_calibration
    )
