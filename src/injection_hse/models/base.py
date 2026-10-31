"""Common interface for the state-estimation models.

All models map a :class:`~injection_hse.io.ModelInput` (static topology + masked dynamic observations,
over nodes and edges, with a time axis) to the full encoded node state. The masking
curriculum and the encoding/normalization live outside the model (in the training step and
data layer); the model sees real feature tensors only.

Tensor contract:
- input ``model_input`` (:class:`~injection_hse.io.ModelInput`): ``node_static [N, Fns]``,
  ``edge_index [2, E]``, ``edge_static [E, Fes]`` (static); ``node_obs [*B, T, N, Fnd]``
  (encoded+normalized, unobserved rows zeroed), ``node_mask [*B, T, N]`` (``True`` = observed,
  also fed as a channel so the model distinguishes "observed 0" from "masked"), and optional
  ``edge_obs``/``edge_mask``/``injection_in``/``node_prior`` (the physics-derived input
  priors, appended to the node token when the model is built with ``n_prior_features``).
  ``Fnd = H * encoding.n_features``.
- returns ``[*B, T, N, Fout]`` — the predicted full encoded node state (decoded + de-normalized
  to complex by the caller for the loss / outputs). ``Fout = Fnd`` unless the model was built
  with ``n_out_features`` (the injection decode with an injection encoding of its own,
  :attr:`injection_hse.config.EncodingConfig.injection_kind`, emits ``H * C_out`` channels).

The snapshot case is ``T=1``, node-only, no edges: a model that consumes only
``node_obs``/``node_mask``/``node_static`` and folds ``T`` into the batch reproduces the prior
``forward(features, mask, graph)`` behavior exactly. Models declare which parts of the input
they consume; ignoring ``edge_*`` / ``T>1`` / ``injection_in`` is allowed (a documented
accept-and-ignore is fine), but the INTERFACE always carries them so edges / time / the
injection head slot in additively. ``T>1`` is handled per-step (temporal mixing is a later
task).

An optional :meth:`predict_injection` head (on the sequences / error localization) predicts the injected
spectrum per device under the same masking idea; default not implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from injection_hse.io import ModelInput


class StateEstimator(torch.nn.Module, ABC):
    """Estimate the full encoded node state from a :class:`~injection_hse.io.ModelInput`."""

    @abstractmethod
    def forward(self, model_input: ModelInput) -> Tensor:
        """:class:`~injection_hse.io.ModelInput` -> predicted node state ``[*B, T, N, Fout]``."""

    def predict_injection(self, model_input: ModelInput) -> Optional[Tensor]:
        """Optional auxiliary head: per-device injected spectrum (on the sequences). ``None`` if absent."""
        return None


def broadcast_global_features(
    values: Optional[Tensor],
    *,
    batch: int,
    n_nodes: int,
    n_features: int,
    device,
    dtype,
) -> Optional[Tensor]:
    """Broadcast a per-step GLOBAL vector ``[BT, F]`` to node channels ``[BT, N, F]``.

    A shared input-assembly step for every spatial model. Some inputs describe the whole
    network at a step rather than a node — the periodic time features
    (:mod:`injection_hse.time_features`) and the switch states of a switchable topology
    (:mod:`injection_hse.data.topology`) — so they are appended to every node's token exactly like the
    mask / static channels.

    Returns ``None`` when the block is disabled (``n_features == 0``) — the model then adds no
    channels and is byte-identical to a run without the feature. When ENABLED but ``values``
    is absent (a timeless batch, inference without timestamps, a single-topology probe) an
    all-zero block is synthesized so the input width stays fixed. Expects ``values`` already
    time-folded to ``[BT, F]``.
    """
    if n_features <= 0:
        return None
    if values is None:
        return torch.zeros(batch, n_nodes, n_features, device=device, dtype=dtype)
    v = values.to(device=device, dtype=dtype)  # [BT, F]
    return v.unsqueeze(-2).expand(batch, n_nodes, n_features)  # [BT, N, F]


def order_mask_channels(
    order_mask: Optional[Tensor],
    *,
    location_mask: Tensor,
    n_order_channels: int,
    dtype,
) -> Optional[Tensor]:
    """Per-order observability as input channels ``[BT, X, H]`` (nodes or edges).

    A heterogeneous sensor fleet (e.g. fundamental-only smart meters next to full-harmonic
    PQ meters) needs the model to know WHICH orders of a location are measured, so the
    per-order boolean mask is appended to the token exactly like the location mask channel.

    Returns ``None`` when the model was built without order-mask channels
    (``n_order_channels == 0``) — byte-identical to a homogeneous run. When ENABLED but the
    input carries no ``order_mask`` (a homogeneous batch, inference without capability
    metadata, attribution probes) the channels are SYNTHESIZED from the location mask —
    every observed location covers every order — so the input width stays fixed.
    """
    if n_order_channels <= 0:
        return None
    if order_mask is None:
        return (
            location_mask.to(dtype)
            .unsqueeze(-1)
            .expand(*location_mask.shape, n_order_channels)
        )
    return order_mask.to(dtype)


def prior_channels(
    node_prior: Optional[Tensor],
    *,
    batch: int,
    n_nodes: int,
    n_features: int,
    device,
    dtype,
) -> Optional[Tensor]:
    """Physics-derived input priors as per-node channels ``[BT, N, Fp]``.

    :attr:`injection_hse.config.PhysicsConfig.input_priors` supplies each node token with the
    injection its observed neighbourhood implies and / or the measurement-anchored
    zero-injection state (:meth:`injection_hse.train.SEModule.input_priors`), already encoded and
    normalized; every spatial model appends the block to its token exactly like the mask /
    static / time channels.

    Returns ``None`` when the model was built without priors (``n_features == 0``) —
    byte-identical to a run without them. When ENABLED but the input carries no
    ``node_prior`` (an attribution probe, a hand-built input) an all-zero block is
    synthesized so the input width stays fixed; a block of the wrong width raises, since a
    silently mis-sized prior would be read as other channels. Expects ``node_prior`` already
    time-folded to ``[BT, N, Fp]``.
    """
    if n_features <= 0:
        return None
    if node_prior is None:
        return torch.zeros(batch, n_nodes, n_features, device=device, dtype=dtype)
    if int(node_prior.shape[-1]) != int(n_features):
        raise ValueError(
            f"node_prior carries {int(node_prior.shape[-1])} channels, the model was built "
            f"for {int(n_features)} (physics.input_priors of the training config)."
        )
    return node_prior.to(device=device, dtype=dtype)


__all__ = [
    "StateEstimator",
    "broadcast_global_features",
    "order_mask_channels",
    "prior_channels",
]
