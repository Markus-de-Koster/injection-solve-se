"""Occlusion analysis: what breaks when a sensor or an input order is removed.

The perturbation-based complement of :mod:`injection_hse.analysis.attribution` — instead
of differentiating the model, an input component is REMOVED (a sensor dropped
from the placement, or one harmonic order zeroed out of the observed spectra)
and the resulting degradation is measured. Gradients say what the model is
locally sensitive to; occlusion says what it actually cannot do without.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from torch import Tensor

from injection_hse.physics import solve_injection

from .attribution import (
    _decode_state,
    _edge_features,
    _masked_features,
    _model_output,
    _prior_features,
    _warm_up,
)
from .baselines import linear_prior_injections

if TYPE_CHECKING:  # heavy / optional imports kept off the import path
    import polars as pl

    from pgml.schemas.grid_schema import Grid

    from injection_hse.data import DataSource
    from injection_hse.train import SEModule

    from .baselines import LinearInjectionPrior


def sensor_occlusion(
    module: "SEModule",
    source_or_path: Union[str, Path, "DataSource"],
    *,
    grid: "Grid",
    sensor_names: list,
    split: str = "val",
    device: Union[str, torch.device] = "cpu",
    include_pairs: bool = True,
    max_samples: Optional[int] = None,
) -> "pl.DataFrame":
    """Estimation error with each sensor (and optionally each pair) removed.

    Re-runs :func:`injection_hse.evaluation.predict_dataset` once per placement variant:
    the full placement (row ``dropped="none"``, the baseline), every single
    sensor removed, and — with ``include_pairs`` and at least three sensors —
    every pair removed. All variants are scored on the SAME fixed row set (the
    rows unobserved under the FULL placement), so a difference to the baseline
    row isolates the information lost with the sensor, not a change of the
    metric's support. The metered slack is not a sensor of the placement and is
    never dropped (``slack_known`` is carried over from ``module.cfg``): the
    injection decode cannot run without its boundary.

    Returns one row per variant: ``dropped`` (sensor name(s), ``"+"``-joined),
    ``n_sensors``, the aggregate unobserved pu-MAE ``mae``, its fundamental /
    harmonic-band splits ``mae_fund`` / ``mae_harm`` (mean of the per-order
    values over ``h>1``), and the per-order values ``mae_h<order>``.
    """
    import polars as pl

    from injection_hse.config import DataConfig, MeasurementConfig
    from injection_hse.data import build_data_source
    from injection_hse.evaluation import predict_dataset
    from injection_hse.metrics import state_errors

    if isinstance(source_or_path, (str, Path)):
        source = build_data_source(
            DataConfig(source="parquet", dataset_path=str(source_or_path)), grid=grid
        )
    else:
        source = source_or_path

    names = list(sensor_names)
    variants: list[tuple[str, list]] = [("none", names)]
    variants += [
        (str(gone), [s for s in names if s != gone]) for gone in names if len(names) > 1
    ]
    if include_pairs and len(names) > 2:
        variants += [
            ("+".join(str(g) for g in pair), [s for s in names if s not in pair])
            for pair in combinations(names, 2)
        ]

    slack_known = module.cfg.measurement.slack_known
    eval_mask: Optional[Tensor] = None
    records: list[dict] = []
    for dropped, kept in variants:
        pred = predict_dataset(
            module,
            source_or_path=source,
            grid=grid,
            measurement=MeasurementConfig(voltage_nodes=kept, slack_known=slack_known),
            split=split,
            device=device,
            max_samples=max_samples,
        )
        if eval_mask is None:  # the FULL placement defines the fixed evaluation rows
            eval_mask = torch.as_tensor(pred.mask)
        errors = state_errors(
            torch.as_tensor(pred.pred_v),
            torch.as_tensor(pred.target_v),
            mask=eval_mask,
            orders=pred.orders,
            base_v=torch.as_tensor(pred.base_v),
        )
        harm_orders = [o for o in pred.orders if o != 1]
        rec = {
            "dropped": dropped,
            "n_sensors": len(kept),
            "mae": float(errors["mae_unobs"]),
            "mae_fund": float(errors[1]["mae_unobs"])
            if 1 in pred.orders
            else float("nan"),
            "mae_harm": (
                float(np.mean([errors[o]["mae_unobs"] for o in harm_orders]))
                if harm_orders
                else float("nan")
            ),
        }
        for o in pred.orders:
            rec[f"mae_h{o}"] = float(errors[o]["mae_unobs"])
        records.append(rec)
    return pl.DataFrame(records)


def order_occlusion(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    occlude: tuple = ("voltage", "current"),
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Output shift per OUTPUT order when one INPUT order is zeroed: ``[H_out, H_in]``.

    Entry ``(i, j)``: zero the H-major feature slice of input order
    ``orders[j]`` at the observed inputs, re-run the model, and take the mean
    absolute change of the output features of order ``orders[i]`` at the
    UNOBSERVED rows — the perturbation analogue of
    :func:`~injection_hse.analysis.attribution.harmonic_coupling`, in the same normalized
    output feature space (voltage features for the voltage decode, injection
    features for the injection decode). For a current-measuring module the
    measured-current edge inputs are part of the probe; ``occlude`` selects
    which blocks the zeroing hits — the default removes the order from BOTH the
    voltage and the current inputs (the leave-one-frequency-out reading), while
    ``("voltage",)`` / ``("current",)`` isolate one block with the other intact.
    Runs entirely without gradients.
    """
    if not set(occlude) <= {"voltage", "current"}:
        raise ValueError(
            f"occlude must be a subset of ('voltage', 'current'), got {occlude!r}."
        )
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask = v.to(dev), mask.to(dev)
    _warm_up(module, v, mask)

    h = module.n_harmonics
    c = module.n_channels  # INPUT channels per order (the occluded slice)
    c_out = module.n_out_channels  # OUTPUT channels per order (the measured response)
    total = np.zeros((h, h), dtype=np.float64)
    weight = 0.0
    with torch.no_grad():
        for start in range(0, v.shape[0], batch_size):
            vb = v[start : start + batch_size]
            mb = mask[start : start + batch_size]
            x = _masked_features(module, vb, mb)  # [b, N, F]
            ex, em, i_edges = _edge_features(module, vb, mb)  # [b, E, Fe] | None
            ec = int(ex.shape[-1]) // h if ex is not None else 0
            pr = _prior_features(module, vb, mb, em, i_edges)
            base = _model_output(module, x, mb, ex, em, pr)  # [b, N, Fout]
            unobs = (~mb).to(base.dtype).unsqueeze(-1)  # [b, N, 1]
            n_unobs = float(unobs.sum())
            for h_in in range(h):
                x_occ, ex_occ = x, ex
                if "voltage" in occlude:
                    x_occ = x.clone()
                    x_occ[..., h_in * c : (h_in + 1) * c] = 0.0
                if "current" in occlude and ex is not None:
                    ex_occ = ex.clone()
                    ex_occ[..., h_in * ec : (h_in + 1) * ec] = 0.0
                # the priors read the measurements too, so the occluded order is dropped
                # from them exactly as from the anchors
                pr_occ = _prior_features(
                    module, vb, mb, em, i_edges, drop_orders=(h_in,)
                )
                out = _model_output(module, x_occ, mb, ex_occ, em, pr_occ)
                delta = ((out - base).abs() * unobs).reshape(*base.shape[:-1], h, c_out)
                total[:, h_in] += delta.sum(dim=(0, 1, 3)).cpu().numpy()
            weight += n_unobs * c_out
    if weight > 0:
        total /= weight
    return total


def order_occlusion_error(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    base_v: Tensor,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 8,
) -> "pl.DataFrame":
    """Estimation-error increase when each INPUT harmonic order is removed.

    The frequency analogue of :func:`sensor_occlusion` (which drops a whole
    sensor): for each harmonic order the model observes, remove that order from
    the sensors ENTIRELY — zeroed out of the voltage AND measured-current model
    inputs, and dropped from the anchor measurements when the deployed decode
    is measurement-anchored — decode the full state through the model's real
    decode path, and measure the resulting increase in the UNOBSERVED per-unit
    MAE against the ground truth: how much worse the estimate gets without that
    frequency at the sensors. This is the leave-one-frequency-out reading of
    "which frequency matters", complementary to the gradient attribution in
    :func:`~injection_hse.analysis.attribution.order_importance`.

    Returns one row per input order plus a ``"none"`` baseline row: ``order``,
    ``mae`` (aggregate unobserved pu-MAE with that order occluded),
    ``delta_mae`` (increase over the baseline), and the fundamental /
    harmonic-band splits ``mae_fund`` / ``mae_harm``.
    """
    import polars as pl

    dev = torch.device(device)
    module.to(dev).eval()
    v, mask, base_v = v.to(dev), mask.to(dev), base_v.to(dev)
    steps = int(v.shape[1]) if v.dim() == 4 else 1
    if v.dim() == 4 and mask.dim() == 2:
        mask = mask.unsqueeze(1).expand(-1, steps, -1)  # [B, T, N]
    _warm_up(module, v.reshape(-1, *v.shape[-2:]), mask.reshape(-1, mask.shape[-1]))

    orders = list(module.orders)
    h, c = module.n_harmonics, module.n_channels
    fund_idx = orders.index(1) if 1 in orders else None
    harm_mask = torch.ones(h, dtype=torch.bool, device=dev)
    if fund_idx is not None:
        harm_mask[fund_idx] = False

    # accumulators: per occlusion key ("none" + each order) -> summed pu abs error
    keys = ["none"] + [str(o) for o in orders]
    err_all = {k: 0.0 for k in keys}
    err_fund = {k: 0.0 for k in keys}
    err_harm = {k: 0.0 for k in keys}
    n_unobs_all = n_unobs_fund = n_unobs_harm = 0.0

    with torch.no_grad():
        for start in range(0, v.shape[0], batch_size):
            vb = v[start : start + batch_size]
            mb = mask[start : start + batch_size]
            unobs = (~mb).to(base_v.dtype)  # [b, N]
            n_u = float(unobs.sum())
            n_unobs_all += n_u * h
            n_unobs_fund += n_u * (1 if fund_idx is not None else 0)
            n_unobs_harm += n_u * int(harm_mask.sum())

            x = _masked_features(module, vb, mb)
            ex, em, i_edges = _edge_features(module, vb, mb)
            ec = int(ex.shape[-1]) // h if ex is not None else 0
            for key, h_in in [("none", None)] + [
                (str(o), i) for i, o in enumerate(orders)
            ]:
                xo, exo = x, ex
                drop = () if h_in is None else (h_in,)
                if h_in is not None:
                    xo = x.clone()
                    xo[..., h_in * c : (h_in + 1) * c] = 0.0
                    if ex is not None:
                        exo = ex.clone()
                        exo[..., h_in * ec : (h_in + 1) * ec] = 0.0
                pr = _prior_features(module, vb, mb, em, i_edges, drop_orders=drop)
                pred_v = _decode_state(
                    module,
                    _model_output(module, xo, mb, exo, em, pr),
                    vb,
                    mask=mb,
                    edge_mask=em,
                    i_edges=i_edges,
                    drop_orders=drop,
                )
                e = (pred_v - vb).abs() / base_v.view(1, 1, -1)  # [b, H, N]
                w = unobs.unsqueeze(1)
                err_all[key] += float((e * w).sum())
                if fund_idx is not None:
                    err_fund[key] += float((e[:, fund_idx : fund_idx + 1, :] * w).sum())
                err_harm[key] += float((e[:, harm_mask, :] * w).sum())

    def mean(store, n):
        return {k: (store[k] / n if n > 0 else float("nan")) for k in keys}

    mae = mean(err_all, n_unobs_all)
    maf = mean(err_fund, n_unobs_fund)
    mah = mean(err_harm, n_unobs_harm)
    base = mae["none"]
    rows = [
        {
            "order": k,
            "mae": mae[k],
            "delta_mae": mae[k] - base,
            "mae_fund": maf[k],
            "mae_harm": mah[k],
        }
        for k in keys
    ]
    return pl.DataFrame(rows)


def estimator_order_coupling(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    base_v: Tensor,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-order-out coupling of the DEPLOYED estimator: ``[H_out, H_in]``.

    The estimator-level counterpart of :func:`order_occlusion`: input order
    ``orders[j]`` is removed from the voltage and measured-current model inputs
    AND from the anchor measurements, the state is decoded through the model's
    real decode path (measurement-anchored when the deployed solver anchors),
    and the per-OUTPUT-order effect is measured at the unobserved rows in per
    unit of the row base voltage. Where :func:`order_occlusion` describes the
    network alone, this matrix describes the estimator the user runs — with the
    anchors carrying each order's measurements, it is near-diagonal whenever
    the learned cross-order coupling contributes little at deployment.

    Returns ``(shift, delta_mae, mae_base)``: ``shift[i, j]`` — mean pu ``|ΔV|``
    of output order ``i`` against the un-occluded decode; ``delta_mae[i, j]`` —
    change of the unobserved pu-MAE against the ground truth; ``mae_base[i]`` —
    the un-occluded per-order unobserved pu-MAE.
    """
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask, base_v = v.to(dev), mask.to(dev), base_v.to(dev)
    steps = int(v.shape[1]) if v.dim() == 4 else 1
    if v.dim() == 4 and mask.dim() == 2:
        mask = mask.unsqueeze(1).expand(-1, steps, -1)  # [B, T, N]
    _warm_up(module, v.reshape(-1, *v.shape[-2:]), mask.reshape(-1, mask.shape[-1]))

    h, c = module.n_harmonics, module.n_channels
    shift = np.zeros((h, h))
    err_occ = np.zeros((h, h))
    err_base = np.zeros(h)
    n_unobs = 0.0
    bpu = base_v.view(1, 1, -1)
    with torch.no_grad():
        for start in range(0, v.shape[0], batch_size):
            vb = v[start : start + batch_size]
            mb = mask[start : start + batch_size]
            x = _masked_features(module, vb, mb)
            ex, em, i_edges = _edge_features(module, vb, mb)
            ec = int(ex.shape[-1]) // h if ex is not None else 0
            pr0 = _prior_features(module, vb, mb, em, i_edges)
            ctx = dict(mask=mb, edge_mask=em, i_edges=i_edges)
            pred0 = _decode_state(
                module, _model_output(module, x, mb, ex, em, pr0), vb, **ctx
            )
            w = (~mb).to(base_v.dtype).unsqueeze(1)  # [b, 1, N]
            n_unobs += float(w.sum())
            err_base += (
                ((pred0 - vb.to(pred0.dtype)).abs() / bpu * w)
                .sum(dim=(0, 2))
                .double()
                .cpu()
                .numpy()
            )
            for j in range(h):
                xo = x.clone()
                xo[..., j * c : (j + 1) * c] = 0.0
                exo = ex
                if ex is not None:
                    exo = ex.clone()
                    exo[..., j * ec : (j + 1) * ec] = 0.0
                prj = _prior_features(module, vb, mb, em, i_edges, drop_orders=(j,))
                predj = _decode_state(
                    module,
                    _model_output(module, xo, mb, exo, em, prj),
                    vb,
                    drop_orders=(j,),
                    **ctx,
                )
                shift[:, j] += (
                    ((predj - pred0).abs() / bpu * w)
                    .sum(dim=(0, 2))
                    .double()
                    .cpu()
                    .numpy()
                )
                err_occ[:, j] += (
                    ((predj - vb.to(predj.dtype)).abs() / bpu * w)
                    .sum(dim=(0, 2))
                    .double()
                    .cpu()
                    .numpy()
                )
    shift /= max(n_unobs, 1e-30)
    err_occ /= max(n_unobs, 1e-30)
    err_base /= max(n_unobs, 1e-30)
    return shift, err_occ - err_base[:, None], err_base


def _band_inputs(module: "SEModule", vb: Tensor, mb: Tensor):
    """The (features, edge features, edge mask, CT currents) probe inputs for a batch."""
    x = _masked_features(module, vb, mb)
    ex, em, i_edges = _edge_features(module, vb, mb)
    return x, ex, em, i_edges


def _band_error(pred: Tensor, target: Tensor, bpu: Tensor, w: Tensor) -> np.ndarray:
    """Per-order sum of the unobserved pu residual of one decoded batch ``[H]``."""
    return (
        ((pred - target.to(pred.dtype)).abs() / bpu * w)
        .sum(dim=(0, 2))
        .double()
        .cpu()
        .numpy()
    )


def estimator_band_ablation(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    base_v: Tensor,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 8,
    linear_prior: Optional["LinearInjectionPrior"] = None,
    time_unix_s: Optional[Tensor] = None,
) -> "pl.DataFrame":
    """Band-level information split of the deployed estimator (per-order pu-MAE).

    Scores five variants of the injection-decode estimator on the same
    unobserved rows, separating what the anchored solve knows from what the
    network adds:

    - ``baseline`` — full sensor inputs, deployed anchors;
    - ``no_harmonics`` — every order above the fundamental removed from the
      model inputs and the anchors (fundamental-only sensors);
    - ``no_fundamental`` — the fundamental removed from the model inputs and
      the anchors (harmonic-only sensors; the hard slack boundary remains — it
      is structural);
    - ``anchors_only`` — the model's injections replaced by ZERO injections
      (prior-free), deployed anchors: what the over-determined solve achieves
      without the network;
    - ``model_only`` — full model inputs, anchoring off (slack-only decode):
      what the network achieves without the measurement anchors.

    A fitted ``linear_prior``
    (:func:`~injection_hse.analysis.fit_linear_injection_prior`) adds one further row —
    ``linear_prior``, or ``linear_prior_noisy`` when the prior was fitted on
    noisy features — in which the network's injections are replaced by the
    closed-form least-squares estimate of the injections from the SAME
    measurements, decoded through the same anchored solve. It is the direct
    upgrade of ``anchors_only``: the same decode under a fitted linear injection
    prior instead of a zero one, and therefore the reference the deployed
    ``baseline`` row has to beat to have learned anything a regression on its own
    inputs could not. Fit it on the TRAIN split, never on the scored samples.

    When the module carries a sensor-noise model (``cfg.noise.enabled``), three
    NOISY counterparts are appended — ``baseline_noisy``, ``anchors_only_noisy``
    and ``no_harmonics_noisy`` — with the module's own noise applied to the
    observed phasors, the derived CT currents AND the anchor targets (the
    training/eval input path), still scored against the CLEAN truth. The clean
    rows answer the perfect-sensor (laboratory) question; the noisy rows the
    deployed one — with a zero-injection prior the boundary-current inversion
    amplifies CT noise at the fundamental, which only the noisy rows reveal.

    ``v`` is the true state of the scored samples, snapshots ``[B, H, N]`` or sequences
    ``[B, T, H, N]`` (e.g. :func:`~injection_hse.analysis.evaluation_batch`), and ``mask`` their
    placement, ``[B, N]`` (the same at every step) or ``[B, T, N]``. A sequence is scored
    the way it is deployed: the network reads it as a sequence (its temporal memory runs
    over the ``T`` steps, with the per-step time features of ``time_unix_s`` — ``[T]`` or
    ``[B, T]`` — for a module trained with them), while everything per-step — the inputs
    and their noise draw, the prior features, the injection priors, the anchored solve and
    the score — runs on the ``B·T`` steps as independent state rows, which is how the
    training step decodes a sequence too (the network solve has no time coupling).
    ``batch_size`` counts samples (sequences), so a batch decodes ``batch_size·T`` states.
    ``base_v`` ``[N]`` is the per-row per-unit base in volts.

    Returns one row per variant: ``variant`` plus ``mae_h<order>`` columns
    (unobserved pu-MAE per order, averaged over every unobserved row of every scored
    state). Requires the injection decode.
    """
    import polars as pl

    if not getattr(module, "_predict_injection_state", False):
        raise ValueError("estimator_band_ablation requires the injection decode.")
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask, base_v = v.to(dev), mask.to(dev), base_v.to(dev)
    steps = int(v.shape[1]) if v.dim() == 4 else 1
    if v.dim() == 4 and mask.dim() == 2:
        mask = mask.unsqueeze(1).expand(-1, steps, -1)  # [B, T, N]
    _warm_up(module, v.reshape(-1, *v.shape[-2:]), mask.reshape(-1, mask.shape[-1]))

    orders = list(module.orders)
    h, c = module.n_harmonics, module.n_channels
    harm = tuple(i for i, o in enumerate(orders) if o != 1)
    fund = tuple(i for i, o in enumerate(orders) if o == 1)
    variants: dict[str, dict] = {
        "baseline": dict(zero=()),
        "no_harmonics": dict(zero=harm),
        "no_fundamental": dict(zero=fund),
        "anchors_only": dict(zero=(), zero_injections=True),
        "model_only": dict(zero=(), anchors=False),
    }
    noise = getattr(module, "_noise", None)
    if noise is not None:
        noise.reseed(int(module.cfg.noise.seed) + 1000)
        if module._edge_noise is not None:
            module._edge_noise.reseed(int(module.cfg.noise.seed) + 1007)
        variants["baseline_noisy"] = dict(zero=(), noisy=True)
        variants["anchors_only_noisy"] = dict(zero=(), zero_injections=True, noisy=True)
        variants["no_harmonics_noisy"] = dict(zero=harm, noisy=True)
    if linear_prior is not None:
        if linear_prior.noisy and noise is None:
            raise ValueError(
                "the linear prior was fitted on NOISY features; scoring it needs a module "
                "with the same enabled sensor-noise model."
            )
        name = "linear_prior_noisy" if linear_prior.noisy else "linear_prior"
        variants[name] = dict(
            zero=(), prior=linear_prior, noisy=bool(linear_prior.noisy)
        )
    err = {k: np.zeros(h) for k in variants}
    n_unobs = 0.0
    bpu = base_v.view(1, 1, -1)
    solver = module.solver
    saved = (solver.anchor_v_weight, solver.anchor_i_weight)
    with torch.no_grad():
        for start in range(0, v.shape[0], batch_size):
            node_time = None
            if steps > 1:
                tb = time_unix_s
                if tb is not None and tb.dim() == 2:
                    tb = tb[start : start + batch_size]
                n_seq = int(v[start : start + batch_size].shape[0])
                node_time = module._build_node_time(
                    {} if tb is None else {"time_unix_s": tb}, (n_seq, steps), dev
                )
            # every step is one state row from here on; only the model regroups them
            vb = v[start : start + batch_size].reshape(-1, *v.shape[-2:])
            mb = mask[start : start + batch_size].reshape(-1, mask.shape[-1])
            inputs = {False: (vb, *_band_inputs(module, vb, mb))}
            if noise is not None:
                v_n = noise.apply(vb.to(module._complex_dtype))
                # the CTs read the TRUE currents with their own error; the model's edge
                # features and the anchors then see that one noisy reading
                i_n = inputs[False][4]
                if i_n is not None and module._edge_noise is not None:
                    i_n = module._edge_noise.apply(i_n)
                x_n, (ex_n, em_n, i_n) = (
                    _masked_features(module, v_n, mb),
                    _edge_features(module, v_n, mb, i_edges=i_n),
                )
                inputs[True] = (v_n, x_n, ex_n, em_n, i_n)
            ec = (
                int(inputs[False][2].shape[-1]) // h
                if inputs[False][2] is not None
                else 0
            )
            w = (~mb).to(base_v.dtype).unsqueeze(1)
            n_unobs += float(w.sum())
            for key, spec in variants.items():
                v_src, x, ex, em, i_edges = inputs[bool(spec.get("noisy", False))]
                b, n = mb.shape
                # A FIXED injection prior replaces the network outright: the decode is
                # affine in the injection, so nothing of the model output is read and the
                # forward pass is skipped. Zeroing the model FEATURES would not do — they
                # decode to the inverse-normalized zero point, not to zero current.
                i_prior = None
                if spec.get("zero_injections", False):
                    i_prior = torch.zeros(
                        (b, h, n), dtype=module._complex_dtype, device=mb.device
                    )
                elif spec.get("prior") is not None:
                    i_prior = linear_prior_injections(
                        module,
                        spec["prior"],
                        v_src,
                        mb,
                        i_edges=i_edges,
                        edge_mask=em,
                    )
                if i_prior is not None:
                    pred = solve_injection(
                        module, i_prior, v_src, mask=mb, edge_mask=em, i_edges=i_edges
                    )
                    err[key] += _band_error(pred, vb, bpu, w)
                    continue

                xo, exo = x, ex
                for j in spec.get("zero", ()):
                    if xo is x:
                        xo = x.clone()
                        exo = ex.clone() if ex is not None else None
                    xo[..., j * c : (j + 1) * c] = 0.0
                    if exo is not None:
                        exo[..., j * ec : (j + 1) * ec] = 0.0
                pr = _prior_features(
                    module,
                    v_src,
                    mb,
                    em,
                    i_edges,
                    drop_orders=tuple(spec.get("zero", ())),
                )
                out = _model_output(
                    module, xo, mb, exo, em, pr, steps=steps, node_time=node_time
                )
                if not spec.get("anchors", True):
                    solver.set_anchor_weights(v_weight=0.0, i_weight=0.0)
                    try:
                        pred = _decode_state(module, out, v_src, mask=mb)
                    finally:
                        solver.set_anchor_weights(v_weight=saved[0], i_weight=saved[1])
                else:
                    pred = _decode_state(
                        module,
                        out,
                        v_src,
                        mask=mb,
                        edge_mask=em,
                        i_edges=i_edges,
                        drop_orders=spec.get("zero", ()),
                    )
                err[key] += _band_error(pred, vb, bpu, w)
    rows = [
        {
            "variant": k,
            **{
                f"mae_h{o}": float(e[i] / max(n_unobs, 1e-30))
                for i, o in enumerate(orders)
            },
        }
        for k, e in err.items()
    ]
    return pl.DataFrame(rows)


__all__ = [
    "sensor_occlusion",
    "order_occlusion",
    "order_occlusion_error",
    "estimator_order_coupling",
    "estimator_band_ablation",
]
