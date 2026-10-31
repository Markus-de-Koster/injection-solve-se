"""Observation masks for the state-estimation curriculum.

A mask is a boolean ``[B, N]`` over node-phase rows, ``True`` = observed. Masking is
PER-SAMPLE (each row of the batch draws its own mask): per-batch masking would bias every
gradient step toward a single placement and produce high-variance "sawtooth" updates;
per-sample averages over placements within a batch for ``[B, N]`` cost only. Rationale:
``docs/pgl/index.md`` §8.

The curriculum (``injection_hse.config.CurriculumConfig``) advances ``full`` -> ``random`` (rising mask
ratio) -> ``placement`` (the real sensor set); each stage trains to a target loss. The
maskers here are deterministic given a ``torch.Generator`` so a run reproduces exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Masker(ABC):
    """Draw an observation mask ``[B, N]`` (``True`` = observed)."""

    #: fixed observed half-edge indices of a device-based placement (``None`` = the
    #: all-incident default: edge observability derives from the node mask by gathering
    #: at each half-edge's source row). Only a placement carries an explicit set.
    observed_edge_rows: Optional[Tensor] = None

    @abstractmethod
    def mask(
        self,
        batch_size: int,
        n_nodes: int,
        *,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Boolean ``[B, N]``; per-sample (rows independent) unless the kind is fixed."""

    def order_mask(
        self,
        mask: Tensor,
        n_orders: int,
        *,
        fundamental_pos: Optional[int],
        generator: Optional[torch.Generator] = None,
    ) -> Optional[Tensor]:
        """Per-ORDER observability ``[B, N, H]`` of a heterogeneous fleet (``None`` = full).

        ``mask`` is the drawn location mask ``[B, N]``; ``fundamental_pos`` the column of
        order 1 in the solved-order axis (``None`` if the run solves no fundamental). The
        default (homogeneous sensors) returns ``None``: every observed location covers
        every order and the model input stays byte-identical.
        """
        return None


class FullMasker(Masker):
    """All nodes observed (Stage 0 — the full-observability surrogate)."""

    def mask(self, batch_size, n_nodes, *, device=None, generator=None) -> Tensor:
        return torch.ones((batch_size, n_nodes), dtype=torch.bool, device=device)


class RandomMasker(Masker):
    """Per-sample random subset of ``round(observed_fraction * N)`` observed nodes.

    ``fundamental_fraction`` (heterogeneous curriculum) demotes each observed row to a
    FUNDAMENTAL-ONLY monitor with that probability (an independent Bernoulli per observed
    location and sample), so the model trains on mixed fleets before a heterogeneous
    placement finetune. ``0`` keeps every observed row full-harmonic (homogeneous).
    """

    def __init__(
        self, observed_fraction: float, *, fundamental_fraction: float = 0.0
    ) -> None:
        if not 0.0 <= observed_fraction <= 1.0:
            raise ValueError("observed_fraction must be in [0, 1].")
        if not 0.0 <= fundamental_fraction <= 1.0:
            raise ValueError("fundamental_fraction must be in [0, 1].")
        self.observed_fraction = observed_fraction
        self.fundamental_fraction = fundamental_fraction

    def mask(self, batch_size, n_nodes, *, device=None, generator=None) -> Tensor:
        k = max(1, round(self.observed_fraction * n_nodes))
        # Rank random scores per row and observe the top-k -> a uniform random k-subset
        # per sample, fully vectorized (no Python loop over the batch).
        scores = torch.rand((batch_size, n_nodes), device=device, generator=generator)
        idx = scores.topk(k, dim=-1).indices  # [B, k]
        out = torch.zeros((batch_size, n_nodes), dtype=torch.bool, device=device)
        return out.scatter(-1, idx, True)

    def order_mask(
        self, mask, n_orders, *, fundamental_pos, generator=None
    ) -> Optional[Tensor]:
        if not self.fundamental_fraction:
            return None
        scores = torch.rand(mask.shape, device=mask.device, generator=generator)
        demoted = mask & (scores < self.fundamental_fraction)  # [B, N]
        out = mask.unsqueeze(-1).expand(*mask.shape, n_orders).clone()
        if fundamental_pos is None:  # no fundamental solved: a demoted row sees nothing
            out[demoted] = False
        else:
            only_fund = torch.zeros(n_orders, dtype=torch.bool, device=mask.device)
            only_fund[fundamental_pos] = True
            out[demoted] = only_fund
        return out


class PlacementMasker(Masker):
    """Fixed sensor placement: the same observed node rows for every sample (finetune).

    ``order_capability`` (``[N, H]`` bool, from
    :meth:`injection_hse.data.MeasurementModel.order_capability`) carries the per-row order coverage
    of a heterogeneous fleet; ``observed_edge_rows`` the explicit metered half-edges of a
    device-based placement (``None`` = every incident branch of an observed row).
    """

    def __init__(
        self,
        observed_rows: Tensor,
        *,
        order_capability: Optional[Tensor] = None,
        observed_edge_rows: Optional[Tensor] = None,
    ) -> None:
        self.observed_rows = observed_rows.to(torch.int64)
        self.order_capability = order_capability
        self.observed_edge_rows = (
            None if observed_edge_rows is None else observed_edge_rows.to(torch.int64)
        )

    def mask(self, batch_size, n_nodes, *, device=None, generator=None) -> Tensor:
        out = torch.zeros((batch_size, n_nodes), dtype=torch.bool, device=device)
        rows = self.observed_rows.to(device=device)
        out[:, rows] = True
        return out

    def order_mask(
        self, mask, n_orders, *, fundamental_pos, generator=None
    ) -> Optional[Tensor]:
        if self.order_capability is None:
            return None
        cap = self.order_capability.to(device=mask.device)
        if int(cap.shape[-1]) != int(n_orders):
            raise ValueError(
                f"order_capability covers {int(cap.shape[-1])} orders, the run solves "
                f"{n_orders}."
            )
        return mask.unsqueeze(-1) & cap.unsqueeze(0)


def build_masker(
    kind: str,
    *,
    observed_fraction: Optional[float] = None,
    observed_rows: Optional[Tensor] = None,
    fundamental_fraction: Optional[float] = None,
    order_capability: Optional[Tensor] = None,
    observed_edge_rows: Optional[Tensor] = None,
) -> Masker:
    """Construct a :class:`Masker` for a :class:`~injection_hse.config.MaskStageConfig`.

    ``"placement"`` requires ``observed_rows`` (the node-phase rows of the sensor set,
    resolved from the measurement model against the ``NodePhaseIndex``) and optionally
    carries the heterogeneous ``order_capability`` / device-based ``observed_edge_rows``;
    ``"random"`` requires ``observed_fraction`` and optionally a ``fundamental_fraction``
    (the heterogeneous curriculum).
    """
    if kind == "full":
        return FullMasker()
    if kind == "random":
        if observed_fraction is None:
            raise ValueError("random masking requires observed_fraction.")
        return RandomMasker(
            observed_fraction, fundamental_fraction=float(fundamental_fraction or 0.0)
        )
    if kind == "placement":
        if observed_rows is None:
            raise ValueError("placement masking requires observed_rows.")
        return PlacementMasker(
            observed_rows,
            order_capability=order_capability,
            observed_edge_rows=observed_edge_rows,
        )
    raise ValueError(f"unknown mask kind {kind!r} (use 'full'/'random'/'placement').")


__all__ = ["Masker", "FullMasker", "RandomMasker", "PlacementMasker", "build_masker"]
