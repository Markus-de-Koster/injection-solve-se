"""State-estimation loss: per-harmonic-weighted MSE on the standardized Cartesian state.

Combines the three decisions of ``docs/pgl/index.md`` §3–§5:

1. **Cartesian space** — reconstruct ``(re, im)`` from the (encoded) prediction and target
   (:meth:`injection_hse.encoding.StateEncoding.to_cartesian`), so a near-zero harmonic's meaningless
   angle is not penalized (both components → 0).
2. **Per-harmonic standardization** — scale the residual per harmonic
   (:meth:`injection_hse.normalization.HarmonicNormalizer.loss_standardize`) so high orders are
   comparable / learnable, not drowned by the fundamental.
3. **Per-harmonic weight** — multiply by ``w_h`` (priority): default ``physical`` down-
   weights high orders but keeps them learnable; ``uniform`` / ``custom`` available.

An optional explicit ``|V|`` term (``magnitude_term_weight``) sharpens drops/swells.

The inputs ``pred`` / ``target`` are encoded+NORMALIZED features ``[B, N, H, C]`` (the
model's space). The loss inverse-normalizes them back to encoded features, reconstructs the
Cartesian state, per-harmonic standardizes the residual, and applies the per-harmonic weight.
Normalization (visibility) and the weight (priority) compose and do not cancel — the residual
is standardized by the per-harmonic Cartesian SCALE, then re-weighted by ``w_h``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import LossConfig
from .encoding import StateEncoding
from .normalization import HarmonicNormalizer


class StateEstimationLoss:
    """Weighted Cartesian SE loss over the encoded/normalized state.

    ``pred`` / ``target`` are encoded+normalized features ``[B, N, H, C]`` (the model's
    space). The per-harmonic weight vector ``w_h`` is resolved from ``cfg.harmonic_weight``
    against ``orders`` (``physical`` uses the normalizer's per-harmonic magnitude scale).
    """

    def __init__(
        self,
        cfg: LossConfig,
        *,
        encoding: StateEncoding,
        normalizer: HarmonicNormalizer,
        orders: list[int],
    ) -> None:
        self.cfg = cfg
        self.encoding = encoding
        self.normalizer = normalizer
        self.orders = orders
        self._weights = self._resolve_weights()  # [H] float, on CPU until __call__

    def _resolve_weights(self) -> Tensor:
        """Per-harmonic loss weight ``w_h`` ``[H]`` from ``cfg.harmonic_weight``."""
        wc = self.cfg.harmonic_weight
        h = len(self.orders)
        if wc.mode == "uniform":
            return torch.ones(h, dtype=torch.float32)
        if wc.mode == "custom":
            if wc.values is None or len(wc.values) != h:
                raise ValueError(
                    f"custom harmonic weights must have {h} values (one per order)."
                )
            w = torch.tensor(wc.values, dtype=torch.float32)
            return w / w.mean().clamp_min(1e-12)
        # physical: w_h proportional to the order's typical magnitude (Cartesian scale),
        # normalized to mean 1 so the overall loss scale is encoding/weight-mode invariant.
        # NET EFFECT with the standardization in __call__: loss_standardize divides the
        # residual by s_h (squared -> 1/s_h²) and this weight multiplies by s_h, so a
        # constant per-order RELATIVE error contributes ∝ s_h — the fundamental
        # deliberately dominates, but LINEARLY in scale, not quadratically as raw
        # physical MSE would. Weight ("priority") and standardization ("visibility")
        # share the statistic by design here; "uniform" keeps them fully independent.
        scale = self.normalizer.cartesian_scale
        if scale is None:
            raise RuntimeError(
                "normalizer must be fit before resolving physical weights."
            )
        w = scale.detach().to(torch.float32).cpu()
        return w / w.mean().clamp_min(1e-12)

    def __call__(
        self, pred: Tensor, target: Tensor, *, mask: Tensor | None = None
    ) -> Tensor:
        """Scalar loss. ``mask`` ``[B, N]`` optionally restricts the row mean: boolean
        (``True`` = supervise) or non-negative float row weights (a weighted mean).
        ``None`` averages over every row."""
        weighted, mag_weighted = self._terms(pred, target)
        per_node = weighted.mean(dim=-1)  # [B, N] mean over harmonics

        loss = self._reduce(per_node, mask)

        if mag_weighted is not None:
            loss = loss + self.cfg.magnitude_term_weight * self._reduce(
                mag_weighted.mean(dim=-1), mask
            )
        return loss

    def per_node_order(self, pred: Tensor, target: Tensor) -> Tensor:
        """The loss BEFORE its means: per ``(sample, row, order)`` contribution ``[B, N, H]``.

        The same weighted, per-harmonic-standardized residual :meth:`__call__` averages —
        exposed so a diagnostic can ask WHERE and at WHICH order the loss sits, in the loss's
        own units. Averaging this over the harmonic orders and reducing over the rows
        reproduces :meth:`__call__` exactly.
        """
        weighted, mag_weighted = self._terms(pred, target)
        if mag_weighted is not None:
            weighted = weighted + self.cfg.magnitude_term_weight * mag_weighted
        return weighted

    def _terms(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor | None]:
        """``(weighted [B, N, H], magnitude-term [B, N, H] | None)`` — the shared residual."""
        # inverse-normalize back to encoded features, then reconstruct Cartesian (re, im)
        pred_enc = self.normalizer.inverse_transform(pred)
        tgt_enc = self.normalizer.inverse_transform(target)
        pred_xy = self.encoding.to_cartesian(pred_enc)  # [B, N, H, 2]
        tgt_xy = self.encoding.to_cartesian(tgt_enc)

        # per-harmonic standardized Cartesian residual
        resid = self.normalizer.loss_standardize(pred_xy - tgt_xy)  # [B, N, H, 2]
        w = self._weights.to(resid.device, resid.dtype)  # [H]
        per_hc = resid.pow(2).mean(dim=-1)  # [B, N, H] mean over (re, im)
        weighted = per_hc * w.view(*([1] * (per_hc.ndim - 1)), -1)  # [B, N, H]

        mag_weighted = None
        if self.cfg.magnitude_term_weight > 0.0:
            pm = pred_xy.norm(dim=-1)  # |V| [B, N, H]
            tm = tgt_xy.norm(dim=-1)
            mscale = self.normalizer.cartesian_scale.to(resid.device, resid.dtype)
            mresid = ((pm - tm) / mscale.view(*([1] * (pm.ndim - 1)), -1)).pow(2)
            mag_weighted = mresid * w.view(*([1] * (mresid.ndim - 1)), -1)
        return weighted, mag_weighted

    @staticmethod
    def _reduce(per_node: Tensor, mask: Tensor | None) -> Tensor:
        """Masked (or row-weighted) mean over ``[B, N]`` (all rows if ``mask`` is ``None``)."""
        if mask is None:
            return per_node.mean()
        m = mask.to(per_node.dtype)
        return (per_node * m).sum() / m.sum().clamp_min(1.0)


__all__ = ["StateEstimationLoss"]
