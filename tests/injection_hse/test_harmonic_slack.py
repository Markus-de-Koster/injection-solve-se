"""The metered slack as the harmonic boundary (``PhysicsConfig.harmonic_slack="measured"``).

A feeder whose harmonic voltage is dominated by its supply is mis-decoded by a Norton
harmonic slack: the source's near-ideal shunt pins the supply node to zero at every order
above the fundamental, and the supply's own distortion can only be reproduced by injections.
These tests pin that under ``"measured"`` the slack rows hold their measured harmonic
voltage exactly (plain, anchored and factor-once paths), that a distorted supply reaches
every node with zero injections, that the fundamental and the default mode are untouched,
and that the module hands the solver every order's slack voltage.
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

ORDERS = [1, 3, 5]
SENSOR_NODES = ("Bus R4", "Bus C5", "Bus I2")


def _slack_voltage(solver, b, distort=(0.02, 0.01), u1=230.0):
    """``[B, H, K]``: a balanced supply at the fundamental, distorted at h3 / h5.

    The slack rows are the source's phases in order (a, b, c); order ``h`` carries the
    sequence a three-phase system gives it (``-h·120°`` on b, ``+h·120°`` on c), so h3 is
    a zero-sequence set and h5 a negative-sequence one — which is what a Dyn transformer
    then blocks or passes.
    """
    k = int(solver.slack_rows.numel())
    v = torch.zeros(b, len(ORDERS), k, dtype=torch.complex128)
    phase = torch.tensor([0.0, -2.0943951, 2.0943951])[:k]
    for j, o in enumerate(ORDERS):
        level = u1 if o == 1 else u1 * distort[j - 1]
        v[:, j] = level * torch.exp(1j * (o * phase + 0.3 * j))
    return v


def test_measured_slack_holds_the_supply_voltage_and_reaches_every_node(cigre_grid):
    from injection_hse.physics import NetworkSolver

    b = 3
    norton = NetworkSolver.from_grid(cigre_grid, ORDERS, dtype=torch.complex128)
    measured = NetworkSolver.from_grid(
        cigre_grid, ORDERS, dtype=torch.complex128, harmonic_slack="measured"
    )
    n = int(norton.y_fundamental.shape[-1])
    i_zero = torch.zeros(b, len(ORDERS), n, dtype=torch.complex128)
    vsl = _slack_voltage(measured, b)

    v_m = measured.solve(i_zero, vsl)
    v_n = norton.solve(i_zero, vsl)  # the harmonic part of a per-order slack is ignored
    v_n2 = norton.solve(i_zero, vsl[:, 0])
    assert torch.equal(v_n, v_n2), "Norton: per-order and fundamental-only slack agree"
    # the fundamental is identical in both modes
    assert torch.allclose(v_m[:, 0], v_n[:, 0])
    # measured: the slack rows carry their harmonic voltage exactly ...
    sr = measured.slack_rows
    for j in (1, 2):
        assert torch.allclose(v_m[:, j, sr], vsl[:, j], atol=1e-9)
    # ... and the supply's distortion reaches the feeder through the transformers with the
    # sequence physics intact: the negative-sequence h5 appears on every LV row, the
    # zero-sequence h3 is blocked by the Dyn transformers (the LV side sees none of it)
    from pgml.assembly import base_voltage_per_row

    lv = base_voltage_per_row(cigre_grid) < 1000.0
    lv[sr] = False
    assert (v_m[:, 2, lv].abs() > 1e-3 * vsl[:, 2].abs().mean()).all()
    assert v_m[:, 1, lv].abs().max() < 1e-6 * vsl[:, 1].abs().mean()
    # Norton: an undistorted supply -> no harmonic voltage without injections
    assert v_n[:, 1:].abs().max() < 1e-9

    # a fundamental-only slack is refused under the measured mode (the boundary is missing)
    with pytest.raises(ValueError, match="per-order slack voltage"):
        measured.solve(i_zero, vsl[:, 0])


def test_measured_slack_survives_anchoring_and_the_operator_cache(cigre_grid):
    from injection_hse.physics import NetworkSolver

    b = 2
    solver = NetworkSolver.from_grid(
        cigre_grid,
        ORDERS,
        dtype=torch.complex128,
        anchor_v_weight=1.0,
        harmonic_slack="measured",
    )
    n = int(solver.y_fundamental.shape[-1])
    gen = torch.Generator().manual_seed(3)
    i_pred = 0.1 * torch.randn(b, len(ORDERS), n, generator=gen, dtype=torch.complex128)
    vsl = _slack_voltage(solver, b)
    v_meas = torch.zeros(b, len(ORDERS), n, dtype=torch.complex128)
    mask = torch.zeros(b, n, dtype=torch.bool)
    rows = torch.tensor([5, 17, 40])
    mask[:, rows] = True
    v_meas[:, :, rows] = (
        230.0 * torch.tensor([1.0, 0.03, 0.02], dtype=torch.complex128)[None, :, None]
    )
    mask[:, solver.slack_rows] = True
    v_meas[:, :, solver.slack_rows] = vsl

    direct = solver.solve(i_pred, vsl, v_meas=v_meas, v_mask=mask)
    plain = solver.solve(i_pred, vsl)
    for j in range(len(ORDERS)):
        assert torch.allclose(direct[:, j, solver.slack_rows], vsl[:, j], atol=1e-8)
        assert torch.allclose(plain[:, j, solver.slack_rows], vsl[:, j], atol=1e-8)
    assert not torch.allclose(direct, plain), "the anchors move the free rows"

    # the factor-once caches reproduce the direct solves
    from injection_hse.physics import OperatorCacheBudget

    solver.enable_operator_cache(OperatorCacheBudget(64 << 20))
    cached = solver.solve(i_pred, vsl, v_meas=v_meas, v_mask=mask)
    cached_plain = solver.solve(i_pred, vsl)
    assert torch.allclose(cached, direct, atol=1e-8, rtol=1e-8)
    assert torch.allclose(cached_plain, plain, atol=1e-8, rtol=1e-8)


@pytest.fixture(scope="module")
def setup(cigre_grid, se_dataset):
    from injection_hse.data import MeasurementModel, StateCollate, build_data_source
    from injection_hse.encoding import TrigEncoding
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


def _module(setup, grid, harmonic_slack):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.encoding import build_injection_encoding
    from injection_hse.masking import build_masker
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    source = setup["source"]
    cfg = ExperimentConfig(
        name="harmonic-slack",
        seed=0,
        orders=list(source.orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        encoding=EncodingConfig(injection_kind="cartesian"),
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, harmonic_slack=harmonic_slack
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
    solver = NetworkSolver.from_grid(
        grid, list(source.orders), anchor_v_weight=1.0, harmonic_slack=harmonic_slack
    )
    inj_norm = fit_injection_normalizer(
        solver, setup["train_v"], build_injection_encoding(cfg.encoding)
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
    module.set_masker(build_masker("placement", observed_rows=setup["observed_rows"]))
    module.set_stage_anchor(v_weight=1.0, i_weight=0.0)
    return module


def test_module_hands_the_solver_every_orders_slack_voltage(setup, cigre_grid):
    v = setup["batch"]["v"]
    m_meas = _module(setup, cigre_grid, "measured")
    m_nort = _module(setup, cigre_grid, "norton")
    k = int(m_meas._slack_rows.numel())
    assert tuple(m_meas._slack_voltage(v).shape) == (v.shape[0], v.shape[1], k)
    assert tuple(m_nort._slack_voltage(v).shape) == (v.shape[0], k)
    assert m_meas.solver.harmonic_slack == "measured"
    # a training step runs and the decoded state holds the slack rows at the measured
    # harmonic voltage (the fundamental already did)
    out_m = m_meas.training_step(setup["batch"], 0)
    assert torch.isfinite(out_m)
    with torch.no_grad():
        state = (
            m_meas._decode_state(m_meas(m_meas._input_from_batch(setup["batch"])))
            if hasattr(m_meas, "_input_from_batch")
            else None
        )
    if state is not None:
        sr = m_meas._slack_rows
        assert torch.allclose(state[:, :, sr], v[:, :, sr].to(state.dtype), atol=1e-6)


def test_injection_normalizer_ignores_the_slack_rows_current(cigre_grid):
    """A harmonic voltage at the slack must not reach the injection normalizer's scales.

    The slack rows' ``Y·V`` is the source's current (``V_h / R_source`` at a harmonic the
    slack carries — huge for a stiff source); the head never predicts those rows, so the
    fitted feature scales, which set the head's output range, must be the same with and
    without a distorted slack.
    """
    from injection_hse.encoding import build_injection_encoding
    from injection_hse.config import EncodingConfig
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer

    solver = NetworkSolver.from_grid(
        cigre_grid, ORDERS, dtype=torch.complex128, harmonic_slack="measured"
    )
    n = int(solver.y_fundamental.shape[-1])
    gen = torch.Generator().manual_seed(11)
    i_dev = 0.5 * torch.randn(16, len(ORDERS), n, generator=gen, dtype=torch.complex128)
    i_dev = solver.zero_slack(solver.zero_tied(i_dev))
    clean = solver.solve(i_dev, _slack_voltage(solver, 16, distort=(0.0, 0.0)))
    distorted = solver.solve(i_dev, _slack_voltage(solver, 16, distort=(0.02, 0.01)))
    enc = build_injection_encoding(EncodingConfig(injection_kind="cartesian"))
    a = fit_injection_normalizer(solver, clean, enc)
    b = fit_injection_normalizer(solver, distorted, enc)
    assert torch.allclose(a.std, b.std, rtol=1e-6, atol=1e-9), (a.std, b.std)
    assert torch.allclose(a.mean, b.mean, rtol=1e-6, atol=1e-9)
    assert torch.allclose(a.aux_scale, b.aux_scale, rtol=1e-6)
    # and the slack rows' raw targets DO differ, so the test is not vacuous
    raw_c = solver.injection_targets(clean)[:, 1:, solver.slack_rows]
    raw_d = solver.injection_targets(distorted)[:, 1:, solver.slack_rows]
    assert not torch.allclose(raw_c, raw_d, atol=1e-6)
