"""Feature- and row-layout helpers shared by the single-grid and multi-grid steps.

The estimator pipeline moves between three layouts: the solver's ``[*lead, H, N]`` complex
state, the ``[*lead, N, H, C]`` encoded features the normalizer and the loss read, and the
``[*lead, N, H*C]`` flat per-node feature the models consume. A multi-grid batch adds one
more move: gathering ONE member grid's rows out of the concatenated node / half-edge axis
and scattering the per-member results back. Every helper here is differentiable,
out-of-place, and preserves device and dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _to_nhc(feats: Tensor) -> Tensor:
    """``[*lead, H, N, C]`` -> ``[*lead, N, H, C]`` (encode acts on the trailing complex axis).

    Leading-dim-agnostic (``*lead`` = ``[B]`` snapshot or ``[B, T]`` sequence): swaps the
    harmonic (``-3``) and node (``-2``) axes so the layout matches the normalizer/loss
    ``[*batch, N, H, C]`` convention.
    """
    return feats.transpose(-3, -2)


def _flatten_features(feats_nhc: Tensor) -> Tensor:
    """``[*lead, N, H, C]`` -> ``[*lead, N, H*C]`` (the model's flat per-node feature)."""
    *lead, n, h, c = feats_nhc.shape
    return feats_nhc.reshape(*lead, n, h * c)


def _unflatten_features(feats_nf: Tensor, h: int, c: int) -> Tensor:
    """``[*lead, N, H*C]`` -> ``[*lead, N, H, C]``."""
    *lead, n, _ = feats_nf.shape
    return feats_nf.reshape(*lead, n, h, c)


def _gather_segments(t: Tensor, rows: Tensor) -> Tensor:
    """``[..., N*]`` -> ``[k, ..., n]``: one member's slices of a batched node / half-edge axis.

    ``rows`` is int64 ``[k, n]`` — the GLOBAL indices the ``k`` items drawn from that member
    contribute (their node segment ``ptr[b]:ptr[b+1]``, their half-edge segment, or a row
    subset such as the member's slack rows). The gathered axis moves to the FRONT, so the
    result is a batch of ``k`` samples over the member's OWN layout: exactly what a per-member
    operator (:meth:`injection_hse.physics.NetworkSolver.solve`,
    :class:`~injection_hse.data.BranchCurrentOperator`) consumes as its leading batch dimension.
    Differentiable, device/dtype-preserving.
    """
    k, n = rows.shape
    seg = t.index_select(-1, rows.reshape(-1))
    seg = seg.reshape(*seg.shape[:-1], k, n)
    return seg.movedim(-2, 0)


def _scatter_segments(parts: list[Tensor], rows: list[Tensor], n_total: int) -> Tensor:
    """Inverse of :func:`_gather_segments` over every member: ``[k, ..., n]`` -> ``[..., N*]``.

    ``parts[j]`` is the per-member result for ``rows[j]``. Out-of-place — one concatenation
    plus one index gather, the :func:`injection_hse.data.merge_groups` construction — so gradients flow
    back to every member's part and no tracked tensor is written into.
    """
    flat: list[Tensor] = []
    for part in parts:
        moved = part.movedim(0, -2)  # [..., k, n]
        flat.append(moved.reshape(*moved.shape[:-2], -1))  # [..., k*n]
    merged = torch.cat(flat, dim=-1)
    if int(merged.shape[-1]) != int(n_total):
        raise ValueError(
            f"the member segments cover {int(merged.shape[-1])} columns, the batch has "
            f"{int(n_total)}; its ptr / grid_index bookkeeping disagrees."
        )
    if len(parts) == 1:
        # a single member owns every item, and its rows were built in item order — the
        # permutation below is the identity, so skip the gather entirely.
        return merged
    order = torch.cat([r.reshape(-1) for r in rows])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return merged.index_select(-1, inverse)


@dataclass(frozen=True)
class _BatchGraph:
    """The static graph tensors of ONE batch, in the layout :func:`build_model_input` reads.

    The single-grid path hands it the per-grid PyG ``Data``; a MULTI-GRID batch already carries
    the same three tensors concatenated over its member graphs
    (:class:`injection_hse.data.MultiGridCollate`), so this adapter presents them under the same attribute
    names without constructing a ``torch_geometric`` object.
    """

    x: Tensor  # [N, Fns] static node features
    edge_index: Tensor  # [2, E] int64
    edge_attr: Tensor  # [E, Fes] static edge features
