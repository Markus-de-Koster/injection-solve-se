"""The voltage error of the Mack et al. estimators, per order, at the unobserved rows.

numpy only (it runs in the estimators' own environment). The estimators predict the
voltages of every row directly, the slack bus included. Two per-state errors, per order, at
the unobserved rows (in per unit of each row's base):

- ``slack_ref``: ``|V̂ - V - G_s·(V̂_s - V_s)|_r / V_base,r`` with ``V̂_s`` the estimator's
  own prediction at the slack rows. For an estimate ``V̂ = G·Î + G_s·V̂_s`` this is the
  voltage error ``|G·(Î - I)|_r`` its injection error causes.
- ``full``: ``|V̂ - V|_r / V_base,r``, the plain phasor error, the metric of the paper's
  evaluation (the adapter stores no slack response, ``G_s = 0``, so both are equal on its
  arrays).
"""

from __future__ import annotations

import numpy as np

METRICS = ("slack_ref", "full")


def to_rows(v_bus: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
    """``[Q, 3, B, H]`` per-phase bus voltages -> ``[Q, H, N]`` frame rows (rows outside 0)."""
    q, _, _, h = v_bus.shape
    out = np.zeros((q, h, n_rows), dtype=np.complex128)
    out[:, :, rows.T.reshape(-1)] = np.transpose(v_bus, (0, 3, 1, 2)).reshape(q, h, -1)
    return out


def interval_errors(v_hat: np.ndarray, v_true: np.ndarray, g_s: np.ndarray, slack_rows: np.ndarray,
                    unobs: np.ndarray, base: np.ndarray) -> dict:
    """``{metric: float [Q, H]}`` mean error over the unobserved rows per interval and order.

    ``v_hat``, ``v_true`` complex ``[Q, H, N]`` volts (each row in its own physical units),
    ``g_s`` complex ``[H, N, K]`` the frame's slack response, ``slack_rows`` int ``[K]``,
    ``unobs`` int ``[U]``, ``base`` float ``[N]`` volts.
    """
    e = v_hat - v_true
    e_ref = e - np.einsum("hnk,qhk->qhn", g_s, e[:, :, slack_rows])
    b = base[unobs][None, None, :]
    return {"slack_ref": (np.abs(e_ref[:, :, unobs]) / b).mean(-1),
            "full": (np.abs(e[:, :, unobs]) / b).mean(-1)}


def site_rows(err: dict, site: np.ndarray, site_names: list, orders, **key) -> list:
    """Per-site sums ``[{**key, site, metric, order, sum, n}]`` of the per-interval errors."""
    out = []
    for k, name in enumerate(site_names):
        sel = site == k
        if not sel.any():
            continue
        for metric, e in err.items():
            s = e[sel].sum(0)
            for j, o in enumerate(orders):
                out.append(dict(**key, site=name, metric=metric, order=int(o), sum=float(s[j]),
                                n=int(sel.sum())))
    return out
