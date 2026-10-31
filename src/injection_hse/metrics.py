"""State-estimation evaluation metrics.

Reported in PER-UNIT (the primary numbers) and physical volts, split by harmonic order and
by observed vs. UNOBSERVED nodes (plus the ALL-rows aggregate). The unobserved error is the
headline estimation quality (observed rows are near-trivial once placement-finetuned); the
all-rows aggregate is the curriculum early-stop monitor because it is well-defined for EVERY
stage, including a fully-observed one where the unobserved set is empty. Used by the
validation loop, the stage early-stop target, and cross-model comparison (DNN vs GNN vs
Graphormer).

The error magnitude is the complex residual ``|V_pred - V_target|`` (Cartesian distance in
the complex plane), so it degrades gracefully for near-zero harmonics (no angle wrapping).
Per-unit divides that residual by the per-node-phase-row line-to-neutral base voltage
(:func:`injection_hse.data.base_voltage_per_row`) BEFORE averaging, so every order/feeder voltage level
is comparable and the curriculum's pu target losses (≈1e-6 → ≈1e-2 MAE) apply directly.

Because the pu base is the FUNDAMENTAL line-to-neutral voltage, the pu error of a small
harmonic order is tiny even when the order is estimated poorly relative to its own magnitude.
Two complementary numbers therefore accompany the pu MAE so small orders are not hidden:

- **per-harmonic RELATIVE error** — the residual magnitude of order ``h`` normalized by that
  order's own signal scale (the RMS of ``|V_h|`` over the evaluated rows), so a 50 %-off 3rd
  harmonic reads ``0.5`` regardless of how small the 3rd harmonic is in pu terms. Orders with
  no signal on the evaluated rows are guarded (``nan``, not a divide-by-zero);
- **THD error** — the absolute difference between the predicted and the true voltage total
  harmonic distortion per energized node-phase row, aggregated as an MAE.

All ops are vectorized over the batch; honors the input device/dtype.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _split_stats(err: Tensor, sel: Tensor) -> tuple[float, float]:
    """``(MAE, RMSE)`` of ``err`` ``[B, N]`` over the boolean selection ``sel`` ``[B, N]``."""
    m = sel.to(err.dtype)
    count = m.sum().clamp_min(1.0)
    mae = (err * m).sum() / count
    rmse = ((err.pow(2) * m).sum() / count).sqrt()
    return float(mae), float(rmse)


def _rel_error(err_phys: Tensor, signal: Tensor, sel: Tensor) -> float:
    """Relative error over ``sel``: mean ``|residual|`` divided by the RMS signal scale.

    ``signal`` ``[B, N]`` is the ground-truth magnitude ``|V_h|`` of the same order; the scale
    is its RMS over the SAME selection. Returns ``nan`` when the selection carries no signal
    (guarding the divide-by-zero of a silent harmonic order on the evaluated rows).
    """
    m = sel.to(err_phys.dtype)
    count = m.sum().clamp_min(1.0)
    mae = (err_phys * m).sum() / count
    rms = ((signal.pow(2) * m).sum() / count).sqrt()
    if float(rms) <= 0.0:
        return float("nan")
    return float(mae / rms)


def _masked_mae(err: Tensor, sel: Tensor) -> float:
    """Mean of ``err`` ``[B, N]`` over ``sel``; ``nan`` when the selection is empty."""
    m = sel.to(err.dtype)
    count = m.sum()
    if float(count) <= 0.0:
        return float("nan")
    return float((err * m).sum() / count)


def state_errors(
    pred_v: Tensor,
    target_v: Tensor,
    *,
    mask: Tensor,
    orders: list[int],
    base_v: Optional[Tensor] = None,
    order_mask: Optional[Tensor] = None,
) -> dict:
    """Per-harmonic MAE/RMSE of complex ``V`` ``[B, H, N]``, split observed/unobserved.

    Parameters
    ----------
    pred_v, target_v:
        Complex node voltages ``[B, H, N]`` (same shape), in PHYSICAL VOLTS line-to-neutral
        (the unit ``pgml.scenarios`` stores the state in); per-unit numbers come only from
        ``base_v`` below. The training path always passes ``base_v``, so every
        ``val_pu_*`` metric and the curriculum target losses are per-unit.
    mask:
        ``[B, N]`` bool, ``True`` = observed; defines the split (broadcast over harmonics).
    orders:
        The harmonic orders ``[H]`` (the keys of the per-order sub-dicts).
    order_mask:
        Optional per-ORDER observability ``[B, N, H]`` bool of a heterogeneous sensor
        fleet: order ``h`` of row ``i`` counts as observed only where both the location
        mask AND its order column are ``True`` — so a fundamental-only smart meter's row
        is observed at ``h=1`` and UNOBSERVED (estimated) at every harmonic order. The
        THD split keeps the location-level ``mask`` (THD is a property of the whole
        signal, not of one order). ``None`` = homogeneous (every observed location covers
        every order; identical to before).
    base_v:
        Optional per-row line-to-neutral base voltage ``[N]`` (from
        :func:`injection_hse.data.base_voltage_per_row`). When given, the PRIMARY keys are PER-UNIT
        (residual divided by ``base_v`` per row) and the physical-volt numbers are added under
        ``*_phys`` keys. When ``None``, the primary keys are physical volts (and equal the
        ``*_phys`` keys).

    Returns
    -------
    dict
        ``{order: {"mae_obs", "rmse_obs", "mae_unobs", "rmse_unobs", "mae_all",
        "rmse_all", "mae_obs_phys", ..., "rel_obs", "rel_unobs", "rel_all"}}`` plus the same
        aggregate keys at the top level (over all orders), plus the top-level THD errors
        ``"thd_mae_obs"`` / ``"thd_mae_unobs"`` / ``"thd_mae_all"``. The ``rel_*`` keys are the
        residual normalized by the order's own signal scale (``nan`` for a silent order); the
        ``thd_*`` keys are the absolute voltage-THD error (dimensionless fraction) over the
        ENERGIZED rows. The aggregate ``"mae_all"`` (pu when ``base_v`` is given; over ALL
        node-phase rows) is the curriculum early-stop monitor; ``"mae_unobs"`` is the headline
        estimation quality reported at test time.
    """
    if pred_v.shape != target_v.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred_v.shape)} vs {tuple(target_v.shape)}."
        )
    err_phys = (pred_v - target_v).abs()  # [B, H, N] real, complex distance, volts
    sig_phys = target_v.abs()  # [B, H, N] the ground-truth magnitude per order (scale)
    if base_v is not None:
        base = base_v.to(err_phys.device, err_phys.dtype).clamp_min(1e-12)
        err_pu = err_phys / base.view(1, 1, -1)  # [B, H, N] per-unit
    else:
        err_pu = err_phys

    if order_mask is None:
        obs = mask.unsqueeze(1).expand_as(err_phys)  # [B, H, N]
    else:
        obs = order_mask.permute(0, 2, 1) & mask.unsqueeze(1)  # [B, H, N]
    unobs = ~obs

    def _entry(e_pu: Tensor, e_phys: Tensor, sig: Tensor, o: Tensor, u: Tensor) -> dict:
        a = o | u  # every row (observed OR unobserved) — the all-rows aggregate
        mae_o, rmse_o = _split_stats(e_pu, o)
        mae_u, rmse_u = _split_stats(e_pu, u)
        mae_a, rmse_a = _split_stats(e_pu, a)
        mae_op, rmse_op = _split_stats(e_phys, o)
        mae_up, rmse_up = _split_stats(e_phys, u)
        mae_ap, rmse_ap = _split_stats(e_phys, a)
        return {
            "mae_obs": mae_o,
            "rmse_obs": rmse_o,
            "mae_unobs": mae_u,
            "rmse_unobs": rmse_u,
            "mae_all": mae_a,
            "rmse_all": rmse_a,
            "mae_obs_phys": mae_op,
            "rmse_obs_phys": rmse_op,
            "mae_unobs_phys": mae_up,
            "rmse_unobs_phys": rmse_up,
            "mae_all_phys": mae_ap,
            "rmse_all_phys": rmse_ap,
            # relative to the order's OWN signal scale (RMS of |V_h| over the same rows)
            "rel_obs": _rel_error(e_phys, sig, o),
            "rel_unobs": _rel_error(e_phys, sig, u),
            "rel_all": _rel_error(e_phys, sig, a),
        }

    out: dict = {}
    for hi, order in enumerate(orders):
        out[int(order)] = _entry(
            err_pu[:, hi, :],
            err_phys[:, hi, :],
            sig_phys[:, hi, :],
            obs[:, hi, :],
            unobs[:, hi, :],
        )

    b = err_phys.shape[0]
    out.update(
        _entry(
            err_pu.reshape(b, -1),
            err_phys.reshape(b, -1),
            sig_phys.reshape(b, -1),
            obs.reshape(b, -1),
            unobs.reshape(b, -1),
        )
    )
    out.update(_thd_errors(pred_v, target_v, mask=mask, orders=orders))
    return out


def _thd_errors(
    pred_v: Tensor, target_v: Tensor, *, mask: Tensor, orders: list[int]
) -> dict:
    """Absolute voltage-THD error per energized node-phase row, split obs/unobs/all.

    THD is ``sqrt(sum_{h>1} |V_h|^2) / |V_1|`` per ``(sample, row)`` — a self-consistent
    property of each signal (predicted THD uses the predicted fundamental). Rows whose TRUE
    fundamental is near zero (de-energized / neutral rows) are excluded so a ``0/0`` never
    enters the mean; the returned MAE is a dimensionless fraction. ``nan`` when the orders
    carry no fundamental (THD is undefined).
    """
    keys = ("thd_mae_obs", "thd_mae_unobs", "thd_mae_all")
    if 1 not in orders:
        return dict.fromkeys(keys, float("nan"))
    h1 = orders.index(1)
    harm = [i for i in range(len(orders)) if i != h1]

    v1_true = target_v[:, h1, :].abs()  # [B, N]
    v1_pred = pred_v[:, h1, :].abs()
    # exclude near-zero true fundamentals (neutral / de-energized rows): a row is energized
    # if its fundamental is above a small fraction of the batch peak.
    floor = (1e-6 * v1_true.amax()).clamp_min(1e-12)
    energized = v1_true > floor

    if harm:
        harm_true = target_v[:, harm, :].abs().pow(2).sum(dim=1).sqrt()  # [B, N]
        harm_pred = pred_v[:, harm, :].abs().pow(2).sum(dim=1).sqrt()
    else:  # no harmonic orders present -> THD is exactly 0 everywhere
        harm_true = torch.zeros_like(v1_true)
        harm_pred = torch.zeros_like(v1_pred)
    thd_true = harm_true / v1_true.clamp_min(floor)
    thd_pred = harm_pred / v1_pred.clamp_min(floor)
    thd_err = (thd_pred - thd_true).abs()  # [B, N]

    obs = mask & energized
    unobs = (~mask) & energized
    return {
        "thd_mae_obs": _masked_mae(thd_err, obs),
        "thd_mae_unobs": _masked_mae(thd_err, unobs),
        "thd_mae_all": _masked_mae(thd_err, energized),
    }


__all__ = ["state_errors"]
