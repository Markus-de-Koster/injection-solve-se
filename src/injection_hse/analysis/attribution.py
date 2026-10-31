"""Gradient-based input attribution for a trained state estimator.

Answers WHICH observed inputs — which sensor node-phase rows, and which harmonic
orders of their spectra — the trained model actually uses: plain input-gradient
saliency, integrated gradients along the path from the all-zero (fully masked)
input, and the harmonic coupling matrix (sensitivity of every OUTPUT order to
every INPUT order).

The attribution seam reproduces the training-time input pipeline exactly:
``encode → normalize → flatten (H-major) → zero unobserved rows``, with the
resulting feature tensor made a gradient leaf and fed through
:func:`injection_hse.io.build_model_input` into the model. When the module measures
branch currents, the measured-current (and apparent-power) edge features are
derived from the same state and fed alongside the node features — probing the
model WITHOUT its current inputs would attribute a degraded, off-training
regime — and the edge features are gradient leaves too, so every input
attribution carries a voltage and a current share. Two consequences keep the
interpretation clean:

- the ZERO input IS the model's encoding of "unmeasured" (masked rows are
  zeroed in training), so the integrated-gradients baseline is semantically the
  no-measurement state — the sensor placement (the mask channel) is held fixed
  along the path;
- attribution explains the model's raw NORMALIZED output features: the voltage
  features for ``physics.decode="voltage"``, the nodal injection-current
  features for ``decode="injection"`` (the network solve after the model is a
  fixed linear map of those currents).

Attribution runs on SNAPSHOT data (one time step): a sequence input would mix
temporal credit through the recurrent memory. Draw the standard input pair from
the randomized snapshot dataset with :func:`attribution_batch`.

This module also owns the DECODE seam every analysis in the subpackage shares:
the model output pushed through the module's network solve with the deployed
measurement anchoring (the solve of a given nodal injection itself is
``injection_hse.physics.solve_injection``, shared with the training preconditioner).
Keeping the seam here means the occlusion probes and the non-learned injection
priors take one path to a decoded state, and one file has to follow
``SEModule._step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from torch import Tensor

from injection_hse.io import build_model_input
from injection_hse.physics import solve_injection

if TYPE_CHECKING:  # heavy / optional imports kept off the import path
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import MeasurementConfig
    from injection_hse.data import DataSource
    from injection_hse.train import SEModule


@dataclass(frozen=True)
class Attribution:
    """Per-sample input attribution over node-phase rows and harmonic orders.

    ``values[s, n, h]`` is the channel-summed absolute attribution of input row
    ``n`` at input order ``orders[h]`` for sample ``s`` — non-negative
    "attribution mass". For integrated gradients the mass is exactly zero at
    unobserved rows (their input is the baseline); for saliency it is the local
    sensitivity and is non-zero everywhere.

    ``edge_values``/``edge_observed`` carry the attribution of the measured-
    current (edge) inputs when the module has a current channel (``None`` for a
    voltage-only model): ``edge_values[s, e, h]`` is the channel-summed absolute
    attribution of half-edge ``e`` at input order ``orders[h]``, and
    ``edge_observed[s, e]`` marks the metered half-edges.
    """

    values: np.ndarray  # float [S, N, H]
    orders: list[int]
    node_ids: np.ndarray  # int64 [N]
    phase_codes: np.ndarray  # int64 [N]
    mask: np.ndarray  # bool [S, N] (True = the input row was observed)
    method: str  # "saliency" | "integrated_gradients"
    target: str  # e.g. "unobs_all_orders" | "unobs_order_5"
    edge_values: Optional[np.ndarray] = None  # float [S, E, H]
    edge_observed: Optional[np.ndarray] = None  # bool [S, E]


# =============================================================================
# The attribution seam (training-time input pipeline, gradient-enabled)
# =============================================================================
def _masked_features(module: "SEModule", v: Tensor, mask: Tensor) -> Tensor:
    """Complex ``[B, H, N]`` (VOLTS) + mask ``[B, N]`` -> masked model input ``[B, N, H·C]``.

    The exact training-time encoding, through the module's own seam
    (:meth:`injection_hse.train.SEModule._encode_normalize`): the run's feature frame, trig/cartesian
    encode, per-harmonic normalization, H-major flatten, unobserved rows zeroed — including
    the training-precision boundary cast (a complex128 dataset otherwise yields float64
    features that a float32 model rejects at the GRU).
    """
    if v.dtype != module._complex_dtype:
        v = v.to(module._complex_dtype)
    feats = module._encode_normalize(v)  # [B, N, H, C]
    b, n, h, c = feats.shape
    flat = feats.reshape(b, n, h * c)
    return flat * mask.unsqueeze(-1).to(flat.dtype)


def _edge_features(
    module: "SEModule", v: Tensor, mask: Tensor, *, i_edges: Optional[Tensor] = None
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Measured-current edge inputs ``(edge_obs [B, E, F], edge_mask [B, E], i_edges [B, H, E])``.

    The exact training-time derivation for a current-measuring module (noise-free:
    analysis probes the clean pipeline): half-edge currents from the Y-bus-exact
    state->current operator, trig/cartesian encode, edge normalizer, optional
    apparent-power channels, zeroed outside the metered half-edges (a half-edge is
    observed iff its source row is observed). ``(None, None, None)`` for a module
    without a current channel — the node-only model input is unchanged.

    ``i_edges`` (amperes, ``[B, H, E]``) supplies already-measured currents — e.g. the true
    currents with the CT error drawn — in place of the noise-free derivation from ``v``;
    ``v`` then only feeds the apparent-power channels.
    """
    if getattr(module, "edge_op", None) is None or not getattr(
        module.cfg.measurement, "measure_current", False
    ):
        return None, None, None
    if i_edges is None:
        if module.edge_op.device != v.device:
            module.edge_op.to(v.device)
        # derived at the state's own precision, then cast: a branch current between two
        # 20 kV terminals is below what complex64 resolves in the voltage difference
        i_edges = module.edge_op(v).to(module._complex_dtype)  # [B, H, E], amperes
    v = v.to(module._complex_dtype)
    # the model's channels live in the run's feature frame (per-unit over the half-edge
    # current base when ``normalization.per_unit``), while ``i_edges`` stays in amperes for
    # the anchored solve — the same split the training step makes
    e_base = module._edge_current_base()
    i_feat = (
        i_edges
        if e_base is None
        else i_edges / e_base.to(device=i_edges.device, dtype=i_edges.real.dtype)
    )
    feats = module.edge_normalizer.transform(
        module.encoding.encode(i_feat).transpose(-3, -2)  # [B, E, H, C]
    )
    if module.cfg.measurement.measure_power:
        v_src = module._to_feature_frame(v).index_select(-1, module._edge_src_rows)
        s_edges = v_src * i_feat.conj()
        feats = torch.cat(
            [
                feats,
                module.power_normalizer.transform(
                    module.encoding.encode(s_edges).transpose(-3, -2)
                ),
            ],
            dim=-1,
        )  # [B, E, H, 2C]
    edge_mask = mask.index_select(-1, module._edge_src_rows)  # [B, E]
    upstream = getattr(module, "_edge_upstream", None)
    if upstream is not None:
        # one-CT budget (measurement.current_channels="upstream"): the probe must
        # honor the trained channel set — feeding the full incident channels puts
        # the model off its training distribution and the anchors beyond the
        # deployed sensor budget.
        edge_mask = edge_mask & upstream
    feats = feats * edge_mask.unsqueeze(-1).unsqueeze(-1).to(feats.dtype)
    b, e, h, c = feats.shape
    return feats.reshape(b, e, h * c), edge_mask, i_edges


def _prior_features(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    edge_mask: Optional[Tensor] = None,
    i_edges: Optional[Tensor] = None,
    *,
    drop_orders: tuple = (),
) -> Optional[Tensor]:
    """The probe's physics-derived ``node_prior`` block ``[B, N, Fp]`` (``None`` = no priors).

    The exact training-time derivation (:meth:`injection_hse.train.SEModule.input_priors`) on the
    probe's state ``v [B, H, N]`` and mask, with the measured-current context of a
    current-measuring module. ``drop_orders`` removes those order indices from the
    measurements the priors read (via a per-order observability mask) — the occlusion
    probes' leave-one-frequency-out contract: an order zeroed at the model input must not
    re-enter through the priors, exactly as it is dropped from the anchor measurements.
    """
    if not getattr(module, "_input_priors", []):
        return None
    v = v.to(module._complex_dtype)
    order_mask = None
    edge_order_mask = None
    if drop_orders:
        b, n = mask.shape
        h = module.n_harmonics
        order_mask = mask.unsqueeze(-1).expand(b, n, h).clone()
        for j in drop_orders:
            order_mask[..., j] = False
        if edge_mask is not None:
            e = int(edge_mask.shape[-1])
            edge_order_mask = edge_mask.unsqueeze(-1).expand(b, e, h).clone()
            for j in drop_orders:
                edge_order_mask[..., j] = False
    return module.input_priors(
        v, mask, order_mask, i_edges, edge_mask, edge_order_mask, None
    )


def _model_output(
    module: "SEModule",
    node_obs: Tensor,
    mask: Tensor,
    edge_obs: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    node_prior: Optional[Tensor] = None,
    *,
    steps: int = 1,
    node_time: Optional[Tensor] = None,
) -> Tensor:
    """Run the model on (possibly gradient-leaf) inputs -> ``[B, N, F]``.

    Every input is in the STEP-ROW layout ``[B, ...]`` the probes work in. With ``steps``
    ``T > 1`` the rows are ``B = S·T`` consecutive steps of ``S`` sequences (scenario-major):
    they are regrouped to ``[S, T, ...]`` so the model runs as on a training sequence batch
    (the temporal memory sees the whole sequence, ``node_time [S, T, Ftime]`` the per-step
    time features), and the output returns to the row layout. ``steps=1`` is the snapshot
    probe.
    """
    rows = node_obs.shape[0]
    lead = (rows,) if steps == 1 else (rows // steps, steps)

    def _lift(t: Optional[Tensor]) -> Optional[Tensor]:
        return None if t is None else t.reshape(*lead, *t.shape[1:])

    # a switchable-topology module expects its topology channels; a probe carries no
    # per-sample configuration, so it runs on the default one (see SEModule.topology_channels)
    node_topology, edge_topology = module.topology_channels(rows, node_obs.device)
    mi = build_model_input(
        module._graph,
        _lift(node_obs),
        _lift(mask),
        has_time=steps > 1,
        edge_obs=_lift(edge_obs),
        edge_mask=_lift(edge_mask),
        node_time=node_time,
        node_topology=_lift(node_topology),
        edge_topology=_lift(edge_topology),
        node_prior=_lift(node_prior),
    )
    out = module.model(mi)  # [B, 1, N, F] or [S, T, N, F]
    return out.reshape(rows, node_obs.shape[1], -1)


# =============================================================================
# The decode seam (the module's own output pipeline, shared by every analysis)
# =============================================================================
def _decode_state(
    module: "SEModule",
    out_flat: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    i_edges: Optional[Tensor] = None,
    drop_orders: tuple = (),
) -> Tensor:
    """Decode the model's flat output ``[B, N, Fout]`` to the complex state ``[B, H, N]``.

    Mirrors :meth:`injection_hse.train.SEModule._step` exactly: the voltage decode
    inverse-normalizes and un-encodes the node features; the injection decode
    un-encodes the raw output in the INJECTION encoding (``module.inj_encoding``,
    ``module.n_out_channels`` channels per order) and reconstructs the voltages
    from the predicted nodal currents with ``injection_hse.physics.solve_injection`` — the
    differentiable network solve at the measured slack, anchored to the
    measurements exactly as the deployed estimator anchors. A PRIOR-RESIDUAL decode
    (:attr:`~injection_hse.config.PhysicsConfig.injection_prior_residual`) adds the linear
    prior's injections to the raw output in the head's feature space before
    decoding, as the training step does — so a probe that zeroes part of the raw
    output zeroes the NETWORK's contribution, prior kept. ``v`` is the measured
    state the model read (the prior reads its placement rows from it, restricted
    to ``mask`` when given).
    """
    b, n, _ = out_flat.shape
    h = module.n_harmonics
    feats = out_flat.reshape(b, n, h, module.n_out_channels)
    if getattr(module, "_predict_injection_state", False):
        module._sync_solver_device(feats.device)
        if getattr(module, "_prior_residual", False):
            feats = feats + module._injection_features(
                module._prior_residual_injection(
                    v.to(module._complex_dtype), i_edges, mask, edge_mask
                )
            ).to(feats.dtype)
        feats = module._pin_tied_features(feats)
        i_pred = module._zero_tied(
            module._from_injection_frame(
                module.inj_encoding.decode(
                    module.inj_normalizer.inverse_transform(feats)
                ).transpose(-2, -1)
            )
        )  # [B, H, N], amperes (exactly zero at the switch-tied rows)
        if getattr(module, "grid_calibration", None) is not None:
            module.grid_calibration.apply_to_grid()
            module.solver.assemble()
        return solve_injection(
            module,
            i_pred,
            v,
            mask=mask,
            edge_mask=edge_mask,
            i_edges=i_edges,
            drop_orders=drop_orders,
        )
    return module._from_feature_frame(
        module.encoding.decode(module.normalizer.inverse_transform(feats)).transpose(
            -2, -1
        )
    )


def _rnn_grad_safe():
    """Context that lets attribution gradients flow through a GRU in EVAL mode on CUDA.

    cuDNN's fused RNN backward can only run in TRAIN mode, but attribution runs the model in
    eval (no dropout, deterministic). Disabling cuDNN falls back to the native RNN, whose
    backward works in eval mode — so gradient saliency / integrated gradients work on the
    temporal (GRU) model on a GPU. A no-op on CPU (cuDNN is not used there).
    """
    return torch.backends.cudnn.flags(enabled=False)


def _target_scalar(
    module: "SEModule", out: Tensor, mask: Tensor, order_index: Optional[int]
) -> Tensor:
    """Sum of the model's output features at UNOBSERVED rows (one order or all).

    The scalar the gradients are taken of: the estimate at the rows the model
    must actually reconstruct. ``order_index`` selects one output order's
    H-major channel slice — ``module.n_out_channels`` wide, the OUTPUT channel
    count (the injection decode may emit its own encoding's channels).
    """
    sel = out * (~mask).unsqueeze(-1).to(out.dtype)
    if order_index is not None:
        c = module.n_out_channels
        sel = sel[..., order_index * c : (order_index + 1) * c]
    return sel.sum()


def _order_index(module: "SEModule", output_order: Optional[int]) -> Optional[int]:
    if output_order is None:
        return None
    if output_order not in module.orders:
        raise ValueError(
            f"output order {output_order} not among the module orders {module.orders}."
        )
    return module.orders.index(output_order)


def _target_name(output_order: Optional[int]) -> str:
    return "unobs_all_orders" if output_order is None else f"unobs_order_{output_order}"


def _warm_up(module: "SEModule", v: Tensor, mask: Tensor) -> None:
    """One no-grad forward (fills structural caches, e.g. Graphormer hop distances)."""
    with torch.no_grad():
        x = _masked_features(module, v[:1], mask[:1])
        ex, em, i_edges = _edge_features(module, v[:1], mask[:1])
        pr = _prior_features(module, v[:1], mask[:1], em, i_edges)
        _model_output(module, x, mask[:1], ex, em, pr)


def _per_order_mass(raw: Tensor, n_harmonics: int) -> np.ndarray:
    """Channel-summed |raw| ``[B, X, F]`` -> per-order mass ``[B, X, H]`` (numpy)."""
    b, x, f = raw.shape
    return (
        raw.abs().reshape(b, x, n_harmonics, f // n_harmonics).sum(dim=-1).cpu().numpy()
    )


def _to_attribution(
    module: "SEModule",
    raw: Tensor,
    mask: Tensor,
    *,
    grid: "Grid",
    method: str,
    output_order: Optional[int],
    raw_edge: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
) -> Attribution:
    """Channel-summed |raw| ``[B, N, F]`` (+ edge raw) -> :class:`Attribution` (numpy)."""
    from pgml.assembly import node_phase_index

    n = raw.shape[1]
    values = _per_order_mass(raw, module.n_harmonics)
    idx = node_phase_index(grid)
    if int(idx.size) != n:
        raise ValueError(
            f"grid row layout ({int(idx.size)} rows) does not match the model ({n})."
        )
    return Attribution(
        values=values,
        orders=list(module.orders),
        node_ids=idx.node_ids.cpu().numpy().astype(np.int64),
        phase_codes=idx.phase_codes.cpu().numpy().astype(np.int64),
        mask=mask.cpu().numpy().astype(bool),
        method=method,
        target=_target_name(output_order),
        edge_values=(
            _per_order_mass(raw_edge, module.n_harmonics)
            if raw_edge is not None
            else None
        ),
        edge_observed=(
            edge_mask.cpu().numpy().astype(bool) if edge_mask is not None else None
        ),
    )


# =============================================================================
# Input pair
# =============================================================================
def _as_source(source_or_path, grid: "Grid") -> "DataSource":
    """A :class:`~injection_hse.data.DataSource` from a source or a parquet dataset path."""
    if isinstance(source_or_path, (str, Path)):
        from injection_hse.config import DataConfig
        from injection_hse.data import build_data_source

        return build_data_source(
            DataConfig(source="parquet", dataset_path=str(source_or_path)), grid=grid
        )
    return source_or_path


def _split_dataset(source: "DataSource", split: str):
    """The ``"train"`` / ``"val"`` / ``"test"`` dataset of ``source``."""
    if split == "val":
        return source.val_dataset()
    if split == "test":
        return source.test_dataset()
    if split == "train":
        return source.train_dataset()
    raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}.")


def _pick_items(ds, n_samples: int, seed: int) -> list[dict]:
    """``min(n_samples, len(ds))`` items drawn seeded, without replacement."""
    n_take = min(n_samples, len(ds))
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(ds), generator=gen)[:n_take].tolist()
    return [ds[i] for i in pick]


def evaluation_batch(
    source_or_path: Union[str, Path, "DataSource"],
    *,
    grid: "Grid",
    measurement: "MeasurementConfig",
    split: str = "test",
    n_samples: int = 64,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Draw ``(v, mask, time_unix_s)`` from a snapshot OR sequence dataset split.

    The scored batch of the estimator-level probes
    (:func:`~injection_hse.analysis.estimator_band_ablation`) on the data the estimator is evaluated
    on, which for a two-task run is the sequence dataset:

    - ``v`` complex ``[B, H, N]`` (snapshots) or ``[B, T, H, N]`` (sequences): the
      ground-truth state of ``n_samples`` scenarios drawn seeded, without replacement,
      from the split (a sequence is drawn whole, every step kept), in the dataset dtype;
    - ``mask`` bool ``[B, N]``: the FIXED placement of ``measurement`` (the deployed
      reading), the same at every step;
    - ``time_unix_s``: the absolute step times, ``[T]`` (one window shared by the
      dataset) or ``[B, T]`` (a start per scenario), for a time-stamped sequence dataset;
      ``None`` otherwise.

    Same draw as :func:`attribution_batch` on a snapshot dataset (same ``seed`` -> same
    samples). ``v`` and ``mask`` are moved to ``device``; the time axis stays on the CPU.
    """
    from injection_hse.data import MeasurementModel, StateCollate

    source = _as_source(source_or_path, grid)
    items = _pick_items(_split_dataset(source, split), n_samples, seed)
    if not items:
        raise ValueError(f"the {split!r} split of this dataset is empty.")
    batch = StateCollate()(items)
    v = batch["v"]  # [B, (T,) H, N]
    rows = MeasurementModel.resolve(
        grid,
        measurement.voltage_nodes,
        fundamental_nodes=measurement.fundamental_nodes,
        slack_known=measurement.slack_known,
    ).observed_rows(grid)
    mask = torch.zeros((int(v.shape[0]), int(v.shape[-1])), dtype=torch.bool)
    mask[:, rows] = True
    dev = torch.device(device)
    return v.to(dev), mask.to(dev), batch.get("time_unix_s")


def attribution_batch(
    source_or_path: Union[str, Path, "DataSource"],
    *,
    grid: "Grid",
    measurement: "MeasurementConfig",
    split: str = "val",
    n_samples: int = 64,
    seed: int = 0,
    observed_fraction: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
) -> tuple[Tensor, Tensor]:
    """Draw ``(v [B, H, N], mask [B, N])`` from a snapshot dataset split.

    ``v`` is the complex ground-truth state of ``n_samples`` samples drawn
    (seeded, without replacement) from the split. The ``mask`` is the input
    pair's sensor placement for :func:`saliency` / :func:`integrated_gradients`:

    - by default the FIXED placement of ``measurement`` (the same observed rows
      for every sample) — the reading for the DEPLOYED estimator;
    - when ``observed_fraction`` is given, a fresh per-sample RANDOM subset of
      that fraction of rows (:class:`~injection_hse.masking.RandomMasker`) — the reading
      for a placement-agnostic model (the random-masking curriculum stage),
      so node importance is not pinned to one deployment's sensor neighbourhood.

    Sequence datasets are rejected (attribution is snapshot-only).
    """
    from injection_hse.data import MeasurementModel

    source = _as_source(source_or_path, grid)
    if source.n_steps > 1:
        raise ValueError(
            "attribution_batch needs a snapshot dataset (n_steps == 1); sequence "
            "attribution through the temporal memory is out of scope."
        )

    items = _pick_items(_split_dataset(source, split), n_samples, seed)
    n_take = len(items)
    v = torch.stack([torch.as_tensor(it["v"]) for it in items])  # [B, H, N]
    n_nodes = v.shape[-1]

    mm = MeasurementModel.resolve(
        grid,
        measurement.voltage_nodes,
        fundamental_nodes=measurement.fundamental_nodes,
        slack_known=measurement.slack_known,
    )
    if observed_fraction is not None:
        from injection_hse.masking import RandomMasker

        # per-sample random placements (the masker's own generator, seeded off ``seed`` so
        # the draw is reproducible yet distinct from the sample pick) -> placement-agnostic.
        # The metered slack rows are forced observed after the draw, matching the
        # training-time mask contract (the substation meter never disappears).
        mgen = torch.Generator().manual_seed(seed + 1)
        mask = RandomMasker(observed_fraction).mask(n_take, n_nodes, generator=mgen)
        slack = mm.slack_rows(grid)
        if slack.numel():
            mask[:, slack] = True
    else:
        rows = mm.observed_rows(grid)
        mask = torch.zeros((n_take, n_nodes), dtype=torch.bool)
        mask[:, rows] = True
    dev = torch.device(device)
    return v.to(dev), mask.to(dev)


# =============================================================================
# Saliency + integrated gradients
# =============================================================================
def saliency(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    grid: "Grid",
    output_order: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 16,
) -> Attribution:
    """Input-gradient saliency ``|∂ target / ∂ node_obs|`` of the masked encoding.

    ``target`` is the summed model output at unobserved rows (optionally one
    ``output_order``). Unlike integrated gradients, saliency is non-zero at
    UNOBSERVED input rows too — it maps where additional measurements WOULD
    move the estimate, which is the sensor-placement reading of the result.
    """
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask = v.to(dev), mask.to(dev)
    order_index = _order_index(module, output_order)
    _warm_up(module, v, mask)

    grads: list[Tensor] = []
    edge_grads: list[Tensor] = []
    edge_masks: list[Tensor] = []
    with _rnn_grad_safe():
        for start in range(0, v.shape[0], batch_size):
            vb = v[start : start + batch_size]
            mb = mask[start : start + batch_size]
            x = _masked_features(module, vb, mb).detach().requires_grad_(True)
            ex, em, i_edges = _edge_features(module, vb, mb)
            # the priors are a fixed function of the same measurements — a constant extra
            # input of the probe, not a gradient leaf
            pr = _prior_features(module, vb, mb, em, i_edges)
            leaves = [x]
            if ex is not None:
                ex = ex.detach().requires_grad_(True)
                leaves.append(ex)
            out = _model_output(module, x, mb, ex, em, pr)
            target = _target_scalar(module, out, mb, order_index)
            grad = torch.autograd.grad(target, leaves)
            grads.append(grad[0].detach())
            if ex is not None:
                edge_grads.append(grad[1].detach())
                edge_masks.append(em)
    raw = torch.cat(grads, dim=0)
    return _to_attribution(
        module,
        raw,
        mask,
        grid=grid,
        method="saliency",
        output_order=output_order,
        raw_edge=torch.cat(edge_grads, dim=0) if edge_grads else None,
        edge_mask=torch.cat(edge_masks, dim=0) if edge_masks else None,
    )


def _ig_raw(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    steps: int,
    order_index: Optional[int],
    batch_size: int,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], float, float]:
    """Signed integrated-gradients attribution + completeness pair.

    Midpoint-rule path integral from the zero baseline: ``attr = x ⊙ (1/K) Σ_k
    ∇F(α_k x)`` with ``α_k = (k + ½)/K``, the node AND (when present) edge
    features scaled jointly so the baseline is the fully unmeasured state.
    Returns ``(attr [B, N, F], edge_attr [B, E, Fe] | None, edge_mask | None,
    F(x), F(x0))`` summed over the batch, so completeness reads
    ``attr.sum() + edge_attr.sum() ≈ F(x) − F(x0)``.
    """
    raws: list[Tensor] = []
    edge_raws: list[Tensor] = []
    edge_masks: list[Tensor] = []
    f_x_total = 0.0
    f_x0_total = 0.0
    for start in range(0, v.shape[0], batch_size):
        vb = v[start : start + batch_size]
        mb = mask[start : start + batch_size]
        x = _masked_features(module, vb, mb).detach()
        ex, em, i_edges = _edge_features(module, vb, mb)
        ex = ex.detach() if ex is not None else None
        # held FIXED along the path: the priors are a constant input given the probe's
        # measurements, so completeness over the node + edge leaves is unaffected
        pr = _prior_features(module, vb, mb, em, i_edges)
        grad_sum = torch.zeros_like(x)
        egrad_sum = torch.zeros_like(ex) if ex is not None else None
        with _rnn_grad_safe():
            for k in range(steps):
                alpha = (k + 0.5) / steps
                xa = (alpha * x).detach().requires_grad_(True)
                leaves = [xa]
                exa = None
                if ex is not None:
                    exa = (alpha * ex).detach().requires_grad_(True)
                    leaves.append(exa)
                out = _model_output(module, xa, mb, exa, em, pr)
                target = _target_scalar(module, out, mb, order_index)
                grad = torch.autograd.grad(target, leaves)
                grad_sum += grad[0].detach()
                if egrad_sum is not None:
                    egrad_sum += grad[1].detach()
        raws.append(x * grad_sum / steps)
        if ex is not None:
            edge_raws.append(ex * egrad_sum / steps)
            edge_masks.append(em)
        with torch.no_grad():
            f_x_total += float(
                _target_scalar(
                    module, _model_output(module, x, mb, ex, em, pr), mb, order_index
                )
            )
            f_x0_total += float(
                _target_scalar(
                    module,
                    _model_output(
                        module,
                        torch.zeros_like(x),
                        mb,
                        torch.zeros_like(ex) if ex is not None else None,
                        em,
                        pr,
                    ),
                    mb,
                    order_index,
                )
            )
    return (
        torch.cat(raws, dim=0),
        torch.cat(edge_raws, dim=0) if edge_raws else None,
        torch.cat(edge_masks, dim=0) if edge_masks else None,
        f_x_total,
        f_x0_total,
    )


def integrated_gradients(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    grid: "Grid",
    steps: int = 32,
    output_order: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 4,
) -> Attribution:
    """Integrated gradients from the all-masked (zero-feature) baseline.

    The path scales the OBSERVED feature rows from zero (the training-time
    encoding of "unmeasured") to their actual values while the placement mask
    channel stays fixed, so the attribution decomposes ``F(x) − F(baseline)``
    over the observed inputs exactly (completeness, midpoint Riemann rule with
    ``steps`` points). Unobserved rows get exactly zero mass.
    """
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask = v.to(dev), mask.to(dev)
    order_index = _order_index(module, output_order)
    _warm_up(module, v, mask)
    raw, raw_edge, edge_mask, _, _ = _ig_raw(
        module, v, mask, steps=steps, order_index=order_index, batch_size=batch_size
    )
    return _to_attribution(
        module,
        raw,
        mask,
        grid=grid,
        method="integrated_gradients",
        output_order=output_order,
        raw_edge=raw_edge,
        edge_mask=edge_mask,
    )


def harmonic_coupling(
    module: "SEModule",
    v: Tensor,
    mask: Tensor,
    *,
    grid: "Grid",
    method: str = "saliency",
    inputs: str = "all",
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 16,
    ig_steps: int = 8,
) -> np.ndarray:
    """Learned order-to-order sensitivity matrix ``[H_out, H_in]``.

    Row ``i`` attributes OUTPUT order ``orders[i]`` (at the unobserved rows) to
    every INPUT order of the observed sensor inputs: gradient saliency
    (``method="saliency"``) or integrated gradients
    (``method="integrated_gradients"``, ``ig_steps`` path points), aggregated
    as total attribution mass over samples and observed rows. For a
    current-measuring module the measured-current edge inputs are part of the
    probe; ``inputs`` selects which input block the columns aggregate —
    ``"all"`` (default), ``"voltage"`` (node features only), or ``"current"``
    (edge features only, zeros for a voltage-only model). In a linear network
    the harmonics decouple (a near-diagonal matrix, plus every order's
    dependence on the fundamental through the operating point); off-diagonal
    mass shows the coupling the model has learned.
    """
    if method not in ("saliency", "integrated_gradients"):
        raise ValueError(
            f"method must be 'saliency'|'integrated_gradients', got {method!r}."
        )
    if inputs not in ("all", "voltage", "current"):
        raise ValueError(f"inputs must be 'all'|'voltage'|'current', got {inputs!r}.")
    rows = []
    for order in module.orders:
        if method == "saliency":
            attr = saliency(
                module,
                v,
                mask,
                grid=grid,
                output_order=order,
                device=device,
                batch_size=batch_size,
            )
        else:
            attr = integrated_gradients(
                module,
                v,
                mask,
                grid=grid,
                steps=ig_steps,
                output_order=order,
                device=device,
                batch_size=max(1, batch_size // 4),
            )
        rows.append(order_importance(attr, observed_only=True, inputs=inputs))
    return np.stack(rows, axis=0)


# =============================================================================
# Aggregations
# =============================================================================
def order_importance(
    attr: Attribution, *, observed_only: bool = True, inputs: str = "all"
) -> np.ndarray:
    """Total attribution mass per INPUT harmonic order ``[H]``.

    ``observed_only`` restricts to observed input rows / metered half-edges
    (recommended: it reads as "how much of each measured order the model
    consumes"; saliency carries non-zero mass at unobserved rows too).
    ``inputs`` selects the input block — ``"all"`` (voltage + measured-current
    mass, the default), ``"voltage"``, or ``"current"`` (zeros for a
    voltage-only model).
    """
    if inputs not in ("all", "voltage", "current"):
        raise ValueError(f"inputs must be 'all'|'voltage'|'current', got {inputs!r}.")
    total = np.zeros(len(attr.orders))
    if inputs in ("all", "voltage"):
        values = attr.values
        if observed_only:
            values = values * attr.mask[:, :, None]
        total = total + values.sum(axis=(0, 1))
    if inputs in ("all", "current") and attr.edge_values is not None:
        ev = attr.edge_values
        if observed_only:
            ev = ev * attr.edge_observed[:, :, None]
        total = total + ev.sum(axis=(0, 1))
    return total


def node_importance(attr: Attribution) -> dict[int, float]:
    """Attribution mass per node (summed over phases, orders, samples).

    With integrated gradients the mass concentrates on the sensor nodes (the
    only non-baseline inputs) — their relative weight ranks the sensors. With
    saliency every node carries its local sensitivity — the map reads as where
    a measurement WOULD matter (a placement signal).
    """
    per_row = attr.values.sum(axis=(0, 2))  # [N]
    out: dict[int, float] = {}
    for nid, val in zip(attr.node_ids.tolist(), per_row.tolist()):
        out[nid] = out.get(nid, 0.0) + float(val)
    return out


__all__ = [
    "Attribution",
    "attribution_batch",
    "evaluation_batch",
    "saliency",
    "integrated_gradients",
    "harmonic_coupling",
    "order_importance",
    "node_importance",
]
