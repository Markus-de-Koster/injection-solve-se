"""Physical scenario features and error binning for a trained-estimator evaluation.

Derives, per evaluated sample, the power-quality quantities a grid operator bins
operating states by — total harmonic distortion, voltage unbalance, relative
line loading, and fundamental voltage deviation — from the ground-truth state in
a :class:`~injection_hse.evaluation.Prediction`, and aggregates the estimator's unobserved
per-unit error over quantile bins of any such feature. This answers WHERE (in
operating-condition space) the estimator degrades, complementing the per-order
and per-node diagnostics in :mod:`injection_hse.evaluation`.

Everything here runs off the autograd tape on plain numpy/polars data; only the
branch-current computation round-trips through torch
(:func:`pgml.assembly.branch_currents`, chunked over samples).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:  # heavy / optional imports kept off the import path
    import polars as pl

    from pgml.schemas.grid_schema import Grid

    from injection_hse.data import DataSource
    from injection_hse.evaluation import Prediction
    from injection_hse.train import SEModule

# Phase codes in the node-phase row layout (pgml.assembly.node_phase_index).
_NEUTRAL_CODE = 3
# Rows whose fundamental magnitude falls below this fraction of the per-row base
# voltage are excluded from ratio features (THD, VUF, deviation) — a de-energized
# or neutral row would otherwise produce an unbounded ratio.
_FUNDAMENTAL_FLOOR = 1e-6


def _fundamental_index(orders: list[int]) -> int:
    """Index of the fundamental (order 1) in the harmonic axis, or raise."""
    try:
        return orders.index(1)
    except ValueError:
        raise ValueError(
            f"scenario features need the fundamental among the orders, got {orders}."
        ) from None


def scenario_features(
    prediction: "Prediction",
    *,
    device: Union[str, torch.device] = "cpu",
    chunk_size: int = 256,
) -> "pl.DataFrame":
    """Per-sample power-quality features of the GROUND-TRUTH state.

    One row per sample of ``prediction`` (same order as its sample axis), with
    the columns:

    - ``thd_max_pct`` / ``thd_mean_pct`` — voltage THD (``100 ·
      sqrt(Σ_{h>1}|V_h|²)/|V_1|``) per energized non-neutral row, reduced over
      rows by max / mean;
    - ``vuf_max_pct`` — voltage-unbalance factor ``100 · |V⁻|/|V⁺|`` (Fortescue
      negative over positive sequence at the fundamental), max over the nodes
      carrying all three phases a/b/c;
    - ``loading_rel_max`` — RELATIVE line loading proxy: per two-terminal
      branch, the phase-RSS fundamental terminal current ``sqrt(Σ_p|I_p|²)``
      normalized by that branch's 95th percentile over the evaluated samples
      (the grid model carries no populated ampacity ratings, so loading is
      relative to the dataset, not to a physical limit), max over branches;
    - ``vdev_max_pct`` — ``100 · | |V_1|/V_base − 1 |``, max over energized
      non-neutral rows.

    Ratio features skip rows whose fundamental magnitude is below a small
    fraction of the per-row base voltage (de-energized/neutral rows). The
    branch currents are computed with :func:`pgml.assembly.branch_currents` in
    chunks of ``chunk_size`` samples on ``device``.
    """
    import polars as pl

    v = np.asarray(prediction.target_v)  # complex [S, H, N]
    s, h, n = v.shape
    fund = _fundamental_index(prediction.orders)
    base_v = np.asarray(prediction.base_v, dtype=np.float64)  # [N]
    phase_codes = np.asarray(prediction.phase_codes)

    v_fund = np.abs(v[:, fund, :])  # [S, N]
    energized = (v_fund > _FUNDAMENTAL_FLOOR * base_v[None, :]) & (
        phase_codes[None, :] != _NEUTRAL_CODE
    )  # [S, N]

    # -- THD ------------------------------------------------------------------
    harm = np.ones(h, dtype=bool)
    harm[fund] = False
    if harm.any():
        rss = np.sqrt((np.abs(v[:, harm, :]) ** 2).sum(axis=1))  # [S, N]
        thd = np.where(
            energized, 100.0 * rss / np.where(energized, v_fund, 1.0), np.nan
        )
        thd_max = np.nanmax(thd, axis=1)
        thd_mean = np.nanmean(thd, axis=1)
    else:
        thd_max = np.zeros(s)
        thd_mean = np.zeros(s)

    # -- VUF (Fortescue at the fundamental) ------------------------------------
    rows_abc = _three_phase_rows(prediction)
    if rows_abc is not None:
        va = v[:, fund, rows_abc[0]]  # [S, M]
        vb = v[:, fund, rows_abc[1]]
        vc = v[:, fund, rows_abc[2]]
        alpha = np.exp(2j * np.pi / 3.0)
        v_pos = (va + alpha * vb + alpha**2 * vc) / 3.0
        v_neg = (va + alpha**2 * vb + alpha * vc) / 3.0
        pos_ok = np.abs(v_pos) > _FUNDAMENTAL_FLOOR * base_v[rows_abc[0]][None, :]
        vuf = np.where(
            pos_ok, 100.0 * np.abs(v_neg) / np.where(pos_ok, np.abs(v_pos), 1.0), np.nan
        )
        vuf_max = np.nanmax(vuf, axis=1)
    else:
        vuf_max = np.full(s, np.nan)

    # -- relative line loading (p95-normalized proxy) --------------------------
    loading_rel_max = _relative_loading(
        prediction, fund, device=device, chunk_size=chunk_size
    )

    # -- fundamental voltage deviation -----------------------------------------
    dev = np.where(energized, 100.0 * np.abs(v_fund / base_v[None, :] - 1.0), np.nan)
    vdev_max = np.nanmax(dev, axis=1)

    return pl.DataFrame(
        {
            "sample": np.arange(s, dtype=np.int64),
            "thd_max_pct": thd_max,
            "thd_mean_pct": thd_mean,
            "vuf_max_pct": vuf_max,
            "loading_rel_max": loading_rel_max,
            "vdev_max_pct": vdev_max,
        }
    )


def _three_phase_rows(prediction: "Prediction") -> Optional[list[np.ndarray]]:
    """Row indices ``[rows_a, rows_b, rows_c]`` of every node carrying a, b AND c.

    Returns ``None`` when no node carries all three phases (single-phase grid).
    Row groups are aligned: entry ``m`` of each array belongs to the same node.
    """
    node_ids = np.asarray(prediction.node_ids)
    phase_codes = np.asarray(prediction.phase_codes)
    per_node: dict[int, dict[int, int]] = {}
    for row, (nid, pc) in enumerate(zip(node_ids.tolist(), phase_codes.tolist())):
        per_node.setdefault(nid, {})[pc] = row
    rows_a, rows_b, rows_c = [], [], []
    for nid in per_node:
        phases = per_node[nid]
        if all(code in phases for code in (0, 1, 2)):
            rows_a.append(phases[0])
            rows_b.append(phases[1])
            rows_c.append(phases[2])
    if not rows_a:
        return None
    return [np.asarray(rows_a), np.asarray(rows_b), np.asarray(rows_c)]


def _relative_loading(
    prediction: "Prediction",
    fund: int,
    *,
    device: Union[str, torch.device],
    chunk_size: int,
) -> np.ndarray:
    """Per-sample max over branches of the p95-normalized fundamental RSS current.

    Uses the ground-truth state; two-terminal branches only (shunts carry no
    through-flow). Branches whose p95 current is numerically zero (e.g. open
    switches) are excluded from the normalization.
    """
    from pgml.assembly import branch_currents, node_phase_index
    from injection_hse._switches import materialize_estimator_grid

    grid = materialize_estimator_grid(prediction.grid)
    idx = node_phase_index(grid)
    freqs = torch.tensor(
        [o * grid.base_frequency_hz for o in prediction.orders], dtype=torch.float64
    )
    dev = torch.device(device)
    v_all = torch.as_tensor(np.asarray(prediction.target_v)).to(torch.complex128)

    s = v_all.shape[0]
    per_branch: list[list[np.ndarray]] = []
    for start in range(0, s, chunk_size):
        v_chunk = v_all[start : start + chunk_size].to(dev)
        currents = branch_currents(grid, v_chunk, freqs, idx, device=dev)
        chunk_mags = [
            torch.sqrt((bc.i_from[..., fund, :].abs() ** 2).sum(dim=-1)).cpu().numpy()
            for bc in currents
            if bc.to_node is not None
        ]
        if not per_branch:
            per_branch = [[m] for m in chunk_mags]
        else:
            for slot, m in zip(per_branch, chunk_mags):
                slot.append(m)
    if not per_branch:
        return np.full(s, np.nan)

    mags = np.stack([np.concatenate(slot) for slot in per_branch], axis=1)  # [S, K]
    p95 = np.percentile(mags, 95.0, axis=0)  # [K]
    carrying = p95 > 1e-9
    if not carrying.any():
        return np.full(s, np.nan)
    rel = mags[:, carrying] / p95[None, carrying]
    return rel.max(axis=1)


def sample_features(
    source_or_path: Union[str, Path, "DataSource"],
    *,
    grid: "Grid",
    module: "SEModule",
    split: str = "val",
) -> "pl.DataFrame":
    """The SAMPLED scenario inputs of a dataset split, one row per sample.

    Loads the same split (same ``DataConfig`` split settings as ``module``'s
    training run, so rows align with a :func:`injection_hse.evaluation.predict_dataset`
    over the same split) and exposes every persisted ``sample_*`` column as a
    scalar feature: scalar samples pass through, per-component/per-phase
    samples are reduced to their mean (column suffixed ``_mean``). Snapshot
    datasets only — a sequence dataset folds time into the sample axis, which
    has no per-step sampled inputs to align with.

    The returned frame carries ``sample`` (running row index, matching the
    prediction's sample axis) and ``scenario`` (the dataset scenario id).
    """
    import polars as pl

    from injection_hse.data import build_data_source

    if isinstance(source_or_path, (str, Path)):
        data_cfg = module.cfg.data.model_copy(
            update={"source": "parquet", "dataset_path": str(source_or_path)}
        )
        source = build_data_source(data_cfg, grid=grid)
    else:
        source = source_or_path
    if source.n_steps > 1:
        raise ValueError(
            "sample_features supports snapshot datasets only (n_steps == 1); "
            "a coherent sequence dataset has no per-step sampled inputs."
        )

    if split == "train":
        ds = source.train_dataset()
    elif split == "val":
        ds = source.val_dataset()
    elif split == "test":
        ds = source.test_dataset()
    else:
        raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}.")

    columns: dict[str, list[float]] = {}
    scenarios: list[int] = []
    for i in range(len(ds)):
        item = ds[i]
        scenarios.append(int(item["index"]))
        for key, value in item.items():
            if not key.startswith("sample_"):
                continue
            t = torch.as_tensor(value)
            name = key.removeprefix("sample_")
            if t.numel() > 1:
                name = f"{name}_mean"
            columns.setdefault(name, []).append(float(t.to(torch.float64).mean()))

    frame = {"sample": np.arange(len(ds), dtype=np.int64), "scenario": scenarios}
    frame.update(columns)
    return pl.DataFrame(frame)


def binned_errors(
    prediction: "Prediction",
    features: "pl.DataFrame",
    *,
    by: str,
    bins: int = 5,
    min_count: int = 10,
) -> "pl.DataFrame":
    """Unobserved per-unit error statistics over quantile bins of a feature.

    Bins the per-sample feature ``features[by]`` into up to ``bins``
    equal-population (quantile) bins and reduces the estimator's UNOBSERVED
    per-unit absolute error per bin. One output row per kept bin (bins with
    fewer than ``min_count`` samples are dropped):

    - ``bin``, ``n``, ``lo``/``hi`` (bin edges), ``center`` (median feature);
    - ``mae_mean`` / ``mae_p50`` / ``mae_p90`` — per-sample unobserved pu-MAE
      over ALL orders, reduced over the bin;
    - the same triplet for the fundamental only (``mae_fund_*``) and for the
      harmonic band ``h>1`` (``mae_harm_*``).

    ``features`` must align row-for-row with ``prediction``'s sample axis
    (as produced by :func:`scenario_features` / :func:`sample_features` over
    the same split).
    """
    import polars as pl

    x = features[by].to_numpy().astype(np.float64)
    if x.shape[0] != prediction.n_samples:
        raise ValueError(
            f"features ({x.shape[0]} rows) do not align with the prediction "
            f"({prediction.n_samples} samples)."
        )

    err = prediction.abs_error_pu()  # [S, H, N]
    unobs = ~prediction.mask  # [S, N]
    fund = _fundamental_index(prediction.orders)
    harm = np.ones(err.shape[1], dtype=bool)
    harm[fund] = False

    def sample_mae(order_sel: np.ndarray) -> np.ndarray:
        sel = err[:, order_sel, :]  # [S, Hs, N]
        w = np.broadcast_to(unobs[:, None, :], sel.shape).astype(np.float64)
        denom = w.sum(axis=(1, 2))
        return np.where(
            denom > 0, (sel * w).sum(axis=(1, 2)) / np.maximum(denom, 1), np.nan
        )

    mae_all = sample_mae(np.ones(err.shape[1], dtype=bool))
    mae_fund = sample_mae(~harm)
    mae_harm = sample_mae(harm) if harm.any() else np.full(err.shape[0], np.nan)

    finite = np.isfinite(x)
    edges = np.unique(np.quantile(x[finite], np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        raise ValueError(f"feature {by!r} is constant; cannot bin.")
    assignment = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    assignment[~finite] = -1

    records: list[dict] = []
    for b in range(edges.size - 1):
        in_bin = assignment == b
        n = int(in_bin.sum())
        if n < min_count:
            continue
        rec = {
            "bin": b,
            "n": n,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "center": float(np.median(x[in_bin])),
        }
        for label, values in (
            ("mae", mae_all),
            ("mae_fund", mae_fund),
            ("mae_harm", mae_harm),
        ):
            v = values[in_bin]
            v = v[np.isfinite(v)]
            rec[f"{label}_mean"] = float(v.mean()) if v.size else float("nan")
            rec[f"{label}_p50"] = float(np.median(v)) if v.size else float("nan")
            rec[f"{label}_p90"] = (
                float(np.percentile(v, 90.0)) if v.size else float("nan")
            )
        records.append(rec)
    return pl.DataFrame(records)


__all__ = [
    "scenario_features",
    "sample_features",
    "binned_errors",
]
