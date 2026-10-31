"""The ``ModelInput`` contract: the single structured input every ``injection_hse`` model consumes.

Separates **static** (topology + element parameters, time-invariant) from **dynamic**
(measurements + scenario state, time-varying) features, over **both nodes and edges**, with an
explicit **time** axis. The authoritative spec is ``docs/pgl/io-schema.md``; this module
is its in-code realization. The snapshot models are the ``T=1``, node-only, no-edge
special case — building a ``ModelInput`` from ``(node features, mask, graph)`` with ``T=1`` and
no edge dynamics reproduces their previous behavior exactly.

Layout (``*B`` = optional leading batch dims; ``T`` = time steps; ``N`` = node-phase rows;
``E`` = directed half-edges; ``D`` = devices):

- STATIC (computed once from the grid, broadcast over batch + time):
  ``node_static [N, Fns]``, ``edge_index [2, E]`` (int64, directed half-edge pairs),
  ``edge_static [E, Fes]``.
- DYNAMIC (per step, masked to observed):
  ``node_obs [*B, T, N, Fnd]``, ``node_mask [*B, T, N]`` (bool, True = observed),
  ``edge_obs [*B, T, E, Fed] | None``, ``edge_mask [*B, T, E] | None``,
  ``injection_in [*B, T, D, …] | None`` (known device injections; optional; the layout is
  pinned to the localization target's ``[*B, T, D, O, C]``: ``D`` injecting devices in fixed
  grid order, ``O`` injected orders ``h > 1``, ``C`` channels with ``1`` = magnitude),
  ``node_time [*B, T, Ftime] | None`` (per-step GLOBAL time features, node-agnostic;
  broadcast to every node by the model — :mod:`injection_hse.time_features`),
  ``node_topology [*B, T, S] | None`` (per-sample GLOBAL switch states of a switchable
  topology, ``1`` closed / ``0`` open — also node-agnostic, broadcast like the time
  features; :mod:`injection_hse.data.topology`),
  ``edge_topology [*B, T, E, 1] | None`` (the same topology, resolved per HALF-EDGE:
  ``1`` conducting / ``0`` open. Unlike ``edge_obs`` it describes the network, not a
  measurement, so it is NOT gated by the observation mask),
  ``node_order_mask [*B, T, N, H] | None`` / ``edge_order_mask [*B, T, E, H] | None``
  (bool, per-HARMONIC-ORDER observability of a heterogeneous sensor fleet — e.g. a
  fundamental-only smart meter observes ``h=1`` only; ``None`` = every observed
  location covers every order, the homogeneous default),
  ``node_prior [*B, T, N, Fp] | None`` (per-row PHYSICS-DERIVED input channels —
  :attr:`injection_hse.config.PhysicsConfig.input_priors`: the injection the observed neighbourhood
  implies through ``Y(h)`` and the measurement-anchored zero-injection state, already
  encoded and normalized like the block they mirror. Derived from the same measurements
  as ``node_obs``, so it is NOT gated by the mask again; ``None`` = no priors).
- MULTI-GRID INDEXING (``None`` on a single-grid input, which is the default and leaves
  everything above untouched):
  ``node_batch [N] | None`` (int64 — which ITEM of a heterogeneous batch each node row
  belongs to), ``node_ptr [B+1] | None`` (int64 — the item offsets of the node axis),
  ``edge_batch [E] | None`` (the same per half-edge).

A batch of DIFFERENT grids folds the sample axis into the node axis instead of stacking it
(the torch-geometric convention — one disjoint graph per batch; see
:class:`injection_hse.data.MultiGridCollate`): ``node_static [N, Fns]`` and ``node_obs [T, N, Fnd]``
then carry the CONCATENATED rows of every item with the leading ``*B`` empty, and
``node_batch`` / ``node_ptr`` say where one item's rows end and the next begin. Every
per-node and per-edge operation is unchanged; only per-GRAPH operations (a pooled readout, a
per-graph attention mask) need the index vectors.

A model declares which parts it consumes; ignoring ``edge_*`` / ``T>1`` / ``injection_in``
recovers a simpler model with no interface change. This is what keeps the library edge-ready
and temporal-ready: adding branch measurements or a temporal axis is additive, never a
breaking signature change.

Edge representation: an in-service branch joining ``from_phases[k]@from_node`` to
``to_phases[k]@to_node`` yields the directed pair ``from→to`` AND ``to→from`` (see
:func:`injection_hse.data.build_graph`). The ``from→to`` half-edge carries the FROM-terminal dynamic
quantity, ``to→from`` the TO-terminal — so a single edge channel transports terminal-anchored
measurements. ``build_graph`` emits the two directions ADJACENT (``[from→to, to→from, …]``).

All tensors honor the input device/dtype; the static tensors are detached topology metadata
(not on the autograd tape). Vectorized — no python loop over batch/time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelInput:
    """Structured network-state input for a ``injection_hse`` state-estimation model.

    See the module docstring for the full tensor layout. The dynamic tensors carry an
    explicit time axis ``T`` (``T=1`` for the snapshot models); ``node_obs`` is the
    encoded+normalized + masked node voltage, ``node_mask`` its observability gate.
    """

    # -- static (grid-constant) ----------------------------------------------
    node_static: Tensor  # [N, Fns]
    edge_index: Tensor  # [2, E] int64
    edge_static: Tensor  # [E, Fes]
    # -- dynamic (per step, masked) ------------------------------------------
    node_obs: Tensor  # [*B, T, N, Fnd]
    node_mask: Tensor  # [*B, T, N] bool
    edge_obs: Optional[Tensor] = None  # [*B, T, E, Fed]
    edge_mask: Optional[Tensor] = None  # [*B, T, E] bool
    injection_in: Optional[Tensor] = None  # [*B, T, D, ...]
    node_time: Optional[Tensor] = None  # [*B, T, Ftime] (per-step global time features)
    node_topology: Optional[Tensor] = None  # [*B, T, S] (global switch states)
    edge_topology: Optional[Tensor] = None  # [*B, T, E, 1] (per-half-edge conduction)
    node_order_mask: Optional[Tensor] = (
        None  # [*B, T, N, H] bool (heterogeneous orders)
    )
    edge_order_mask: Optional[Tensor] = (
        None  # [*B, T, E, H] bool (heterogeneous orders)
    )
    node_prior: Optional[Tensor] = None  # [*B, T, N, Fp] (physics-derived input priors)
    # -- multi-grid indexing (None = one graph; see the module docstring) ------
    node_batch: Optional[Tensor] = None  # [N] int64 (item of each node row)
    node_ptr: Optional[Tensor] = None  # [B+1] int64 (item offsets of the node axis)
    edge_batch: Optional[Tensor] = None  # [E] int64 (item of each half-edge)

    # -- shape accessors ------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.node_static.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_graphs(self) -> int:
        """Distinct GRAPHS in the input (``1`` unless it is a multi-grid batch)."""
        return 1 if self.node_ptr is None else int(self.node_ptr.numel()) - 1

    @property
    def n_steps(self) -> int:
        """The time-axis length ``T`` (``node_obs`` is ``[*B, T, N, Fnd]``)."""
        return int(self.node_obs.shape[-3])

    @property
    def node_feature_dim(self) -> int:
        """``Fnd`` — dynamic node-feature width (the encoded state width ``F = H*C``)."""
        return int(self.node_obs.shape[-1])

    @property
    def has_edge_obs(self) -> bool:
        return self.edge_obs is not None

    @property
    def device(self) -> torch.device:
        return self.node_obs.device

    # -- batch/time flattening (the per-step baseline) -----------------------
    def fold_time_into_batch(self) -> tuple["ModelInput", tuple[int, ...]]:
        """Collapse the leading ``[*B, T]`` dynamic dims into one ``BT`` batch axis.

        Returns a ``ModelInput`` whose dynamic tensors are ``[BT, N, Fnd]`` / ``[BT, N]`` /
        ``[BT, E, Fed]`` / ``[BT, E]`` / ``[BT, D, …]`` (static tensors unchanged) and the
        original leading shape ``[*B, T]`` so the caller can unfold the model output. This is
        the per-step baseline: temporal mixing is a later task; for now every step is
        estimated independently. No-op semantics for the snapshot ``T=1`` case.
        """
        lead = tuple(self.node_obs.shape[:-2])  # [*B, T]
        n = self.n_nodes
        fnd = self.node_feature_dim
        node_obs = self.node_obs.reshape(-1, n, fnd)
        node_mask = self.node_mask.reshape(-1, n)
        edge_obs = None
        edge_mask = None
        if self.edge_obs is not None:
            e = self.n_edges
            fed = int(self.edge_obs.shape[-1])
            edge_obs = self.edge_obs.reshape(-1, e, fed)
        if self.edge_mask is not None:
            edge_mask = self.edge_mask.reshape(-1, self.n_edges)
        injection_in = None
        if self.injection_in is not None:
            tail = tuple(self.injection_in.shape[len(lead) :])  # [D, ...]
            injection_in = self.injection_in.reshape(-1, *tail)
        node_time = None
        if self.node_time is not None:
            ftime = int(self.node_time.shape[-1])
            node_time = self.node_time.reshape(-1, ftime)  # [BT, Ftime]
        node_topology = None
        if self.node_topology is not None:
            fsw = int(self.node_topology.shape[-1])
            node_topology = self.node_topology.reshape(-1, fsw)  # [BT, S]
        edge_topology = None
        if self.edge_topology is not None:
            fsw = int(self.edge_topology.shape[-1])
            edge_topology = self.edge_topology.reshape(-1, self.n_edges, fsw)
        node_order_mask = None
        if self.node_order_mask is not None:
            h = int(self.node_order_mask.shape[-1])
            node_order_mask = self.node_order_mask.reshape(-1, n, h)  # [BT, N, H]
        edge_order_mask = None
        if self.edge_order_mask is not None:
            h = int(self.edge_order_mask.shape[-1])
            edge_order_mask = self.edge_order_mask.reshape(-1, self.n_edges, h)
        node_prior = None
        if self.node_prior is not None:
            fp = int(self.node_prior.shape[-1])
            node_prior = self.node_prior.reshape(-1, n, fp)  # [BT, N, Fp]
        folded = ModelInput(
            node_static=self.node_static,
            edge_index=self.edge_index,
            edge_static=self.edge_static,
            node_obs=node_obs,
            node_mask=node_mask,
            edge_obs=edge_obs,
            edge_mask=edge_mask,
            injection_in=injection_in,
            node_time=node_time,
            node_topology=node_topology,
            edge_topology=edge_topology,
            node_order_mask=node_order_mask,
            edge_order_mask=edge_order_mask,
            node_prior=node_prior,
            # index vectors describe the NODE/EDGE axes, which folding leaves untouched
            node_batch=self.node_batch,
            node_ptr=self.node_ptr,
            edge_batch=self.edge_batch,
        )
        return folded, lead

    def to(self, *, device=None, dtype=None) -> "ModelInput":
        """Move/cast tensors (static to ``dtype`` for real feats; masks/index stay typed)."""

        def _move(t: Optional[Tensor], cast: bool) -> Optional[Tensor]:
            if t is None:
                return None
            if device is not None:
                t = t.to(device)
            if cast and dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            return t

        return ModelInput(
            node_static=_move(self.node_static, True),
            edge_index=_move(self.edge_index, False),
            edge_static=_move(self.edge_static, True),
            node_obs=_move(self.node_obs, True),
            node_mask=_move(self.node_mask, False),
            edge_obs=_move(self.edge_obs, True),
            edge_mask=_move(self.edge_mask, False),
            injection_in=_move(self.injection_in, True),
            node_time=_move(self.node_time, True),
            node_topology=_move(self.node_topology, True),
            edge_topology=_move(self.edge_topology, True),
            node_order_mask=_move(self.node_order_mask, False),
            edge_order_mask=_move(self.edge_order_mask, False),
            node_prior=_move(self.node_prior, True),
            node_batch=_move(self.node_batch, False),
            node_ptr=_move(self.node_ptr, False),
            edge_batch=_move(self.edge_batch, False),
        )


def build_model_input(
    graph,
    node_obs: Tensor,
    node_mask: Tensor,
    *,
    has_time: bool = False,
    edge_obs: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    injection_in: Optional[Tensor] = None,
    node_time: Optional[Tensor] = None,
    node_topology: Optional[Tensor] = None,
    edge_topology: Optional[Tensor] = None,
    node_order_mask: Optional[Tensor] = None,
    edge_order_mask: Optional[Tensor] = None,
    node_prior: Optional[Tensor] = None,
    node_batch: Optional[Tensor] = None,
    node_ptr: Optional[Tensor] = None,
    edge_batch: Optional[Tensor] = None,
) -> ModelInput:
    """Assemble a :class:`ModelInput` from the per-grid graph + dynamic tensors.

    Parameters
    ----------
    graph:
        The PyG ``Data`` from :func:`injection_hse.data.build_graph` — supplies the static
        ``node_static`` (``graph.x``), ``edge_index``, and ``edge_static``
        (``graph.edge_attr``). Moved to ``node_obs``'s device.
    node_obs:
        Encoded+normalized + masked node voltage. When ``has_time`` is ``False`` (the
        snapshot path) it is ``[*B, N, Fnd]`` and a ``T=1`` axis is inserted; when
        ``True`` it is already ``[*B, T, N, Fnd]``.
    node_mask:
        Observability gate (bool), ``[*B, N]`` (snapshot) or ``[*B, T, N]`` (sequence).
    has_time:
        ``False`` (default): inputs are snapshots, lift to ``T=1``. ``True``: inputs already
        carry the time axis.
    edge_obs, edge_mask, injection_in:
        Optional edge dynamics / known injections (same ``T`` handling). Default ``None``
        (node-only model). ``edge_mask`` requires ``edge_obs``.
    node_time:
        Optional per-step GLOBAL time features ``[*B, Ftime]`` (snapshot) or
        ``[*B, T, Ftime]`` (sequence) — node-agnostic (the model broadcasts them to every
        node). Same ``T`` handling as ``node_obs``. Default ``None`` (no time features).
    node_topology:
        Optional per-sample GLOBAL switch states ``[*B, S]`` (snapshot) or ``[*B, T, S]``
        (sequence) of a switchable topology — node-agnostic, broadcast to every node like
        ``node_time``. Same ``T`` handling. Default ``None`` (one fixed topology).
    edge_topology:
        Optional per-HALF-EDGE conduction state ``[*B, E, 1]`` (snapshot) or
        ``[*B, T, E, 1]`` (sequence) of the same switchable topology. Independent of
        ``edge_obs`` (it is network structure, not a measurement), so it may be given alone.
    node_order_mask, edge_order_mask:
        Optional per-HARMONIC-ORDER observability (bool) of a heterogeneous sensor fleet:
        ``[*B, N, H]`` / ``[*B, E, H]`` (snapshot) or with a leading ``T`` axis (sequence).
        ``None`` (default) = every observed location covers every order (homogeneous).
        Same ``T`` handling as ``node_obs``; ``edge_order_mask`` requires ``edge_obs``.
    node_prior:
        Optional per-row PHYSICS-DERIVED input channels ``[*B, N, Fp]`` (snapshot) or
        ``[*B, T, N, Fp]`` (sequence) — :attr:`injection_hse.config.PhysicsConfig.input_priors`,
        built by :meth:`injection_hse.train.SEModule.input_priors` from the same measurements as
        ``node_obs`` (already encoded + normalized, not gated by the mask again). Same ``T``
        handling as ``node_obs``. Default ``None`` (no priors).
    node_batch, node_ptr, edge_batch:
        Optional MULTI-GRID index vectors (int64 ``[N]`` / ``[B+1]`` / ``[E]``) naming the
        item each concatenated node row / half-edge belongs to. ``None`` (default) = a
        single graph. Passed through unchanged (they index the node/edge axes, which the
        time handling does not touch).

    Returns
    -------
    ModelInput
        Static tensors on ``node_obs``'s device; the ``T`` axis is always present.
    """
    device = node_obs.device
    dtype = node_obs.dtype

    node_static = graph.x.to(device=device, dtype=dtype)
    edge_index = graph.edge_index.to(device=device)
    edge_static = graph.edge_attr.to(device=device, dtype=dtype)
    n, e = node_static.shape[0], edge_index.shape[1]
    if node_batch is not None and int(node_batch.numel()) != int(n):
        raise ValueError(
            f"node_batch must have one entry per node row (N={n}); got "
            f"{int(node_batch.numel())}."
        )
    if edge_batch is not None and int(edge_batch.numel()) != int(e):
        raise ValueError(
            f"edge_batch must have one entry per half-edge (E={e}); got "
            f"{int(edge_batch.numel())}."
        )

    if node_obs.shape[-2] != n:
        raise ValueError(
            f"node_obs trailing dims must be [..., N={n}, Fnd]; got {tuple(node_obs.shape)}."
        )
    if not has_time:  # lift snapshot [*B, N, Fnd] -> [*B, 1, N, Fnd]
        node_obs = node_obs.unsqueeze(-3)
        node_mask = node_mask.unsqueeze(-2)  # [*B, N] -> [*B, 1, N]
        if edge_obs is not None:
            edge_obs = edge_obs.unsqueeze(-3)
        if edge_mask is not None:
            edge_mask = edge_mask.unsqueeze(-2)
        if injection_in is not None:
            injection_in = injection_in.unsqueeze(-3)
        if node_time is not None:  # [*B, Ftime] -> [*B, 1, Ftime]
            node_time = node_time.unsqueeze(-2)
        if node_topology is not None:  # [*B, S] -> [*B, 1, S]
            node_topology = node_topology.unsqueeze(-2)
        if edge_topology is not None:  # [*B, E, 1] -> [*B, 1, E, 1]
            edge_topology = edge_topology.unsqueeze(-3)
        if node_order_mask is not None:  # [*B, N, H] -> [*B, 1, N, H]
            node_order_mask = node_order_mask.unsqueeze(-3)
        if edge_order_mask is not None:
            edge_order_mask = edge_order_mask.unsqueeze(-3)
        if node_prior is not None:  # [*B, N, Fp] -> [*B, 1, N, Fp]
            node_prior = node_prior.unsqueeze(-3)

    if edge_mask is not None and edge_obs is None:
        raise ValueError("edge_mask given without edge_obs.")
    if edge_order_mask is not None and edge_obs is None:
        raise ValueError("edge_order_mask given without edge_obs.")
    if edge_topology is not None and edge_topology.shape[-2] != e:
        raise ValueError(
            f"edge_topology trailing dims must be [..., E={e}, 1]; got "
            f"{tuple(edge_topology.shape)}."
        )
    if edge_obs is not None and edge_obs.shape[-2] != e:
        raise ValueError(
            f"edge_obs trailing dims must be [..., E={e}, Fed]; got {tuple(edge_obs.shape)}."
        )
    if node_prior is not None and node_prior.shape[-2] != n:
        raise ValueError(
            f"node_prior trailing dims must be [..., N={n}, Fp]; got "
            f"{tuple(node_prior.shape)}."
        )

    return ModelInput(
        node_static=node_static,
        edge_index=edge_index,
        edge_static=edge_static,
        node_obs=node_obs,
        node_mask=node_mask.to(torch.bool),
        edge_obs=edge_obs,
        edge_mask=None if edge_mask is None else edge_mask.to(torch.bool),
        injection_in=injection_in,
        node_time=node_time,
        node_topology=node_topology,
        edge_topology=edge_topology,
        node_order_mask=(
            None if node_order_mask is None else node_order_mask.to(torch.bool)
        ),
        edge_order_mask=(
            None if edge_order_mask is None else edge_order_mask.to(torch.bool)
        ),
        node_prior=node_prior,
        node_batch=None if node_batch is None else node_batch.to(torch.int64),
        node_ptr=None if node_ptr is None else node_ptr.to(torch.int64),
        edge_batch=None if edge_batch is None else edge_batch.to(torch.int64),
    )


__all__ = ["ModelInput", "build_model_input"]
