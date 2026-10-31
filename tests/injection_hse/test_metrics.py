"""Per-unit / physical metric correctness (a known residual -> a known value)."""

from __future__ import annotations

import pytest
import torch

from injection_hse.metrics import state_errors

ORDERS = [1, 3, 5]


def test_pu_metric_known_residual():
    # residual = 3 + 4j -> |.| = 5 V everywhere; base 10 V -> pu residual = 0.5
    B, H, N = 2, 3, 4
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    pred = target + (3.0 + 4.0j)
    base = torch.full((N,), 10.0, dtype=torch.float64)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 0] = True

    e = state_errors(pred, target, mask=mask, orders=ORDERS, base_v=base)
    assert e["mae_unobs"] == pytest.approx(0.5)
    assert e["rmse_unobs"] == pytest.approx(0.5)
    assert e["mae_obs"] == pytest.approx(0.5)
    # residual is uniform, so the all-rows aggregate equals the obs/unobs values
    assert e["mae_all"] == pytest.approx(0.5)
    assert e["rmse_all"] == pytest.approx(0.5)
    assert e["mae_all_phys"] == pytest.approx(5.0)
    assert e["mae_unobs_phys"] == pytest.approx(5.0)
    assert e["mae_obs_phys"] == pytest.approx(5.0)
    for o in ORDERS:
        assert e[o]["mae_unobs"] == pytest.approx(0.5)
        assert e[o]["mae_unobs_phys"] == pytest.approx(5.0)


def test_pu_metric_per_row_base():
    # different base per row -> different pu per row
    B, H, N = 1, 1, 2
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    pred = target + (0.0 + 2.0j)  # |.| = 2 V on both rows
    base = torch.tensor([4.0, 10.0], dtype=torch.float64)  # pu 0.5 and 0.2
    mask = torch.zeros(B, N, dtype=torch.bool)  # all unobserved
    e = state_errors(pred, target, mask=mask, orders=[1], base_v=base)
    # mean pu over the two rows = (0.5 + 0.2) / 2 = 0.35
    assert e["mae_unobs"] == pytest.approx(0.35)
    assert e["mae_unobs_phys"] == pytest.approx(2.0)


def test_no_base_v_is_physical():
    B, H, N = 1, 1, 3
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    pred = target + (3.0 + 4.0j)
    mask = torch.zeros(B, N, dtype=torch.bool)
    e = state_errors(pred, target, mask=mask, orders=[1])
    assert e["mae_unobs"] == pytest.approx(5.0)
    assert e["mae_unobs"] == e["mae_unobs_phys"]


def test_per_harmonic_relative_error_known():
    # order signal scale (RMS of |V_h|) vs a known residual -> a known relative error.
    B, H, N = 1, 2, 4
    orders = [1, 3]
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    target[:, 0, :] = 10.0  # fundamental |V_1| = 10 on every row
    target[:, 1, :] = 2.0  # 3rd harmonic |V_3| = 2 on every row
    # residual: 1 V everywhere at both orders (real)
    pred = target + 1.0
    mask = torch.zeros(B, N, dtype=torch.bool)  # all unobserved
    e = state_errors(pred, target, mask=mask, orders=orders)
    # rel = mean|residual| / RMS(|V_h|): order 1 -> 1/10, order 3 -> 1/2
    assert e[1]["rel_unobs"] == pytest.approx(0.1)
    assert e[3]["rel_unobs"] == pytest.approx(0.5)


def test_relative_error_guards_silent_order():
    # an order with zero true signal -> relative error is nan (no divide-by-zero).
    B, H, N = 1, 2, 3
    orders = [1, 5]
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    target[:, 0, :] = 10.0  # fundamental present
    # order 5 stays exactly zero everywhere (silent); the prediction is off by 1 V there
    pred = target.clone()
    pred[:, 1, :] = 1.0
    mask = torch.zeros(B, N, dtype=torch.bool)
    e = state_errors(pred, target, mask=mask, orders=orders)
    import math

    assert math.isnan(e[5]["rel_unobs"])


def test_thd_absolute_error_known():
    # true THD = sqrt(0.04^2 + 0.03^2) = 5 %; predicted harmonics zero -> THD error = 5 %.
    B, H, N = 1, 3, 2
    orders = [1, 3, 5]
    target = torch.zeros(B, H, N, dtype=torch.complex64)
    target[:, 0, :] = 100.0  # |V_1| = 100
    target[:, 1, :] = 4.0  # |V_3| = 4  -> 4 %
    target[:, 2, :] = 3.0  # |V_5| = 3  -> 3 %
    pred = torch.zeros_like(target)
    pred[:, 0, :] = 100.0  # fundamental correct, harmonics predicted as zero
    mask = torch.zeros(B, N, dtype=torch.bool)  # all unobserved
    e = state_errors(pred, target, mask=mask, orders=orders)
    assert e["thd_mae_unobs"] == pytest.approx(0.05, abs=1e-6)
    # a perfect prediction has zero THD error
    e0 = state_errors(target, target, mask=mask, orders=orders)
    assert e0["thd_mae_unobs"] == pytest.approx(0.0, abs=1e-9)


def test_thd_error_nan_without_fundamental():
    B, H, N = 1, 2, 2
    orders = [3, 5]  # no fundamental -> THD undefined
    target = torch.ones(B, H, N, dtype=torch.complex64)
    mask = torch.zeros(B, N, dtype=torch.bool)
    e = state_errors(target, target, mask=mask, orders=orders)
    import math

    assert math.isnan(e["thd_mae_unobs"])


def test_base_voltage_per_row_convention(cigre_grid):
    import math

    from injection_hse.data import base_voltage_per_row
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre_grid)
    base = base_voltage_per_row(cigre_grid)
    assert base.shape == (idx.size,)
    # 3-phase nodes: base = u_rated / sqrt(3) (u_rated is line-to-line for >= 3 phases)
    for node in cigre_grid.nodes:
        expected = (
            float(node.u_rated_v) / math.sqrt(3.0)
            if len(node.phases) >= 3
            else float(node.u_rated_v)
        )
        for row in idx.rows(node.id):
            assert float(base[row]) == pytest.approx(expected)
