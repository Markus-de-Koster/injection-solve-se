"""GNN state estimator: PyG message passing over the node-phase graph (topology-general).

Propagates the masked observations across the topology to estimate the full state. The
message-passing layers are shared across nodes and carry no fixed-``N`` assumption, so the same
weights apply to a grid of different size/topology without a shape change (unlike the
fixed-input DNN baseline) — and to a batch that MIXES grids (:class:`injection_hse.data.MultiGridCollate`,
see below).

Tensor contract (see :class:`injection_hse.models.base.StateEstimator`):
``forward(model_input) -> [*B, T, N, Fout]`` with ``Fnd = H * encoding.n_features`` the input
width and ``Fout`` the output width (``n_out_features``; ``Fnd`` unless the injection decode
gives its head an encoding of its own — only the head projection follows it). From the
:class:`~injection_hse.io.ModelInput`, ``node_obs`` + ``node_mask`` (as a channel, so the net
distinguishes "observed zero" from "masked") + ``node_static`` (+ the optional per-row
``node_prior`` physics-derived block and the optional per-step ``node_time`` features,
broadcast to every node) form the per-node token, lifted
by an encoder to ``hidden_dim``; ``num_layers`` of message passing (``conv="gine"`` folds
``edge_static`` — and, when present, the dynamic ``edge_obs`` + ``edge_mask`` via a lazily-built
projection — into the edge message; ``conv="sage"`` ignores edges) refine with residual
connections + layer norm, and a head projects to ``Fout``. ``T`` is folded into the batch
and estimated per step. The static edge width follows the graph
(``n_edge_static_features``): the series block, or the wider one a phase-coupling graph
emits, whose flag columns let the message for a mutual-admittance edge differ from the one
for a series connection.

Batching (vectorized, NO python loop over the batch on the autograd tape)
------------------------------------------------------------------------
One ``ModelInput`` carries ONE node/edge set, shared across the leading batch/time axes.
Rather than a python loop over ``BT``, the ``BT`` copies are treated as one big DISJOINT graph
of ``BT*N`` nodes: node features are flattened ``[BT, N, ·] -> [BT*N, ·]`` and the
``edge_index`` ``[2, E]`` is tiled ``BT`` times with a per-copy row offset ``i*N`` (built with
``torch.arange``), giving ``[2, BT*E]``. Message passing then runs once over all copies in
parallel; the output is reshaped back to ``[*B, T, N, Fnd]``.

MULTI-GRID batches need no special case here, which is the point of an all-local architecture:
the collate already concatenates the member grids into one disjoint graph of ``N*`` rows with
the node offsets applied to ``edge_index``, so ``N`` above simply becomes ``N*`` and the copy
tiling replicates that whole disjoint graph per step. Every operation of this model is per node
or per edge — there is no pooled readout, no per-graph normalization and no ``N``-indexed
parameter — so a message can only travel along an edge, and the collate guarantees no edge
crosses a member boundary. A batched forward therefore equals running each member alone and
concatenating the outputs (verified in the tests).

Honors the input device/dtype; the (CPU-built) graph is moved to the feature device here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from injection_hse.config import GNNConfig
from injection_hse.io import ModelInput

from .base import (
    StateEstimator,
    broadcast_global_features,
    order_mask_channels,
    prior_channels,
)


class GNN(StateEstimator):
    """Message-passing estimator with GCN, SAGE, GIN, or edge-aware GINE layers.

    ``gat`` remains reserved in the config and raises ``NotImplementedError``.
    """

    #: static node-feature width of the BASE :func:`injection_hse.data.build_graph` block (the default
    #: when no structural encoding widens ``node_static``).
    NODE_FEATURE_DIM = 8
    #: static edge-feature width emitted by :func:`injection_hse.data.build_graph` for the SERIES
    #: edge set (the default; a phase-coupling graph is wider — ``n_edge_static_features``).
    EDGE_FEATURE_DIM = 5

    def __init__(
        self,
        cfg: GNNConfig,
        *,
        n_features: int,
        n_time_features: int = 0,
        n_topology_features: int = 0,
        n_order_mask_channels: int = 0,
        n_prior_features: int = 0,
        n_edge_features: int = 0,
        n_edge_topology_features: int = 0,
        n_static_features: int | None = None,
        n_edge_static_features: int | None = None,
        n_out_features: int | None = None,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv, GINConv, GINEConv, SAGEConv

        self.cfg = cfg
        self.n_features = n_features
        # per-node OUTPUT width (``None`` = the input width): only the head projection
        # follows it, so ``n_out_features == n_features`` is byte-identical.
        self.n_out_features = (
            n_features if n_out_features is None else int(n_out_features)
        )
        self.n_time_features = n_time_features
        self.n_topology_features = n_topology_features
        self.n_order_mask_channels = n_order_mask_channels
        self.n_prior_features = n_prior_features
        self.n_edge_features = n_edge_features
        self.n_edge_topology_features = n_edge_topology_features
        # width of ``node_static``: the base block, plus a structural encoding's ``K`` columns
        # when the graph carries one (:class:`~injection_hse.config.StructuralEncodingConfig`).
        self.n_static_features = (
            self.NODE_FEATURE_DIM
            if n_static_features is None
            else int(n_static_features)
        )
        # width of ``edge_static``: the series block, plus the flag columns a phase-coupling
        # graph appends (:class:`~injection_hse.config.PhaseCouplingConfig`).
        self.n_edge_static_features = (
            self.EDGE_FEATURE_DIM
            if n_edge_static_features is None
            else int(n_edge_static_features)
        )
        hidden = cfg.hidden_dim

        # per-node input: masked features + mask channel + optional per-order observability
        # channels (heterogeneous sensor fleets) + optional physics-derived prior channels +
        # static node features (+ optional per-step time channels, broadcast to every node)
        in_dim = (
            n_features
            + 1
            + n_order_mask_channels
            + n_prior_features
            + self.n_static_features
            + n_time_features
            + n_topology_features
        )
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())

        self._use_edges = cfg.conv == "gine"
        if self._use_edges:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.n_edge_static_features, hidden), nn.GELU()
            )
            # dynamic edge channel (edge_obs ⧺ edge_mask [⧺ edge order mask]
            # [⧺ per-half-edge topology state]), added to the static edge encoding. Built
            # EAGERLY when a dynamic edge width is declared (``n_edge_features`` — the
            # measured-current input path — or ``n_edge_topology_features`` — a switchable
            # topology), so its parameters are registered before
            # ``configure_optimizers``/checkpoint load; otherwise built lazily on first sight
            # (a runtime edge width unknown at construction).
            self.edge_obs_encoder: nn.Module | None = None
            dyn_edge_dim = (
                n_edge_features + 1 + n_order_mask_channels if n_edge_features else 0
            ) + n_edge_topology_features
            if dyn_edge_dim > 0:
                self.edge_obs_encoder = nn.Linear(dyn_edge_dim, hidden)
        self._hidden = hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            if cfg.conv == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
                self.convs.append(GINEConv(mlp, edge_dim=hidden, train_eps=True))
            elif cfg.conv == "sage":
                self.convs.append(SAGEConv(hidden, hidden))
            elif cfg.conv == "gcn":
                self.convs.append(GCNConv(hidden, hidden))
            elif cfg.conv == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            else:
                raise NotImplementedError(
                    f"GNN conv={cfg.conv!r} is not implemented yet."
                )
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        # long-range input skip (deep stacks): the head reads the last layer's output
        # CONCATENATED with the encoder embedding, so oversmoothed deep features cannot
        # wash out the pre-propagation signal. Off = the single-width head.
        self._input_skip = cfg.input_skip
        self.head = nn.Linear(
            hidden * (2 if cfg.input_skip else 1), self.n_out_features
        )

    def _batched_edges(self, edge_index: Tensor, n: int, b: int) -> Tensor:
        """Tile the shared ``edge_index`` ``[2, E]`` into ``[2, B*E]`` with per-copy offset.

        Copy ``i``'s nodes occupy rows ``[i*N, (i+1)*N)`` of the flattened ``[B*N]`` node
        set, so its edges are the shared edges shifted by ``i*N``. Built with ``arange`` /
        broadcasting — no python loop over the batch.
        """
        e = edge_index.shape[1]
        offsets = (torch.arange(b, device=edge_index.device) * n).view(
            b, 1, 1
        )  # [B,1,1]
        tiled = edge_index.unsqueeze(0) + offsets  # [B, 2, E]
        return tiled.permute(1, 0, 2).reshape(2, b * e)  # [2, B*E]

    def _encode_edges(self, mi: ModelInput, b: int, device, dtype) -> Tensor:
        """Per-(copy) edge embedding ``[B*E, hidden]`` from static (+ optional dynamic) feats.

        ``edge_static`` is grid-constant (same on the two half-edge directions); the dynamic
        part differs per copy and is added after a (eagerly or lazily built) projection. It
        holds the directed-half-edge SOURCE-terminal measurement (+ its mask), and — for a
        switchable topology — that copy's per-half-edge conduction state. The measurement is
        gated by the observation mask; the topology state is NOT (it describes the network,
        which is known everywhere, not a measurement).
        """
        es = self.edge_encoder(
            mi.edge_static.to(device=device, dtype=dtype)
        )  # [E, hidden]
        edge_emb = es.unsqueeze(0).expand(b, -1, -1)  # [B, E, hidden] (static, shared)
        if mi.edge_obs is not None or mi.edge_topology is not None:
            parts = []
            if mi.edge_obs is not None:
                eo = mi.edge_obs.to(device=device, dtype=dtype)  # [B, E, Fed]
                em_bool = (
                    mi.edge_mask
                    if mi.edge_mask is not None
                    else torch.ones(eo.shape[:-1], device=device, dtype=torch.bool)
                )
                em = em_bool.to(dtype).unsqueeze(-1)  # [B, E, 1]
                parts.extend([eo * em, em])
                order_ch = order_mask_channels(
                    mi.edge_order_mask,
                    location_mask=em_bool,
                    n_order_channels=self.n_order_mask_channels,
                    dtype=dtype,
                )
                if order_ch is not None:
                    parts.append(order_ch)  # [B, E, H]
            if mi.edge_topology is not None:
                parts.append(
                    mi.edge_topology.to(device=device, dtype=dtype)
                )  # [B, E, 1]
            dyn_in = torch.cat(parts, dim=-1)
            if self.edge_obs_encoder is None:  # lazy fallback (width unknown at build)
                self.edge_obs_encoder = nn.Linear(dyn_in.shape[-1], self._hidden).to(
                    device=device, dtype=dtype
                )
            edge_emb = edge_emb + self.edge_obs_encoder(dyn_in)  # [B, E, hidden]
        return edge_emb.reshape(-1, self._hidden)  # [B*E, hidden]

    def forward(self, model_input: ModelInput) -> Tensor:
        mi, lead = model_input.fold_time_into_batch()
        features = mi.node_obs  # [BT, N, Fnd]
        mask = mi.node_mask  # [BT, N]
        b, n, f = features.shape
        device, dtype = features.device, features.dtype

        x_static = mi.node_static.to(device=device, dtype=dtype)  # [N, Fns]
        edge_index = mi.edge_index.to(device=device)  # [2, E]

        mask_ch = mask.to(dtype).unsqueeze(-1)  # [BT, N, 1]
        parts = [features, mask_ch]
        order_ch = order_mask_channels(
            mi.node_order_mask,
            location_mask=mask,
            n_order_channels=self.n_order_mask_channels,
            dtype=dtype,
        )
        if order_ch is not None:
            parts.append(order_ch)  # [BT, N, H]
        prior_ch = prior_channels(
            mi.node_prior,
            batch=b,
            n_nodes=n,
            n_features=self.n_prior_features,
            device=device,
            dtype=dtype,
        )
        if prior_ch is not None:
            parts.append(prior_ch)  # [BT, N, Fp]
        parts.append(x_static.unsqueeze(0).expand(b, n, -1))
        time_ch = broadcast_global_features(
            mi.node_time,
            batch=b,
            n_nodes=n,
            n_features=self.n_time_features,
            device=device,
            dtype=dtype,
        )
        if time_ch is not None:
            parts.append(time_ch)  # [BT, N, Ftime]
        topology_ch = broadcast_global_features(
            mi.node_topology,
            batch=b,
            n_nodes=n,
            n_features=self.n_topology_features,
            device=device,
            dtype=dtype,
        )
        if topology_ch is not None:
            parts.append(topology_ch)  # [BT, N, S]
        node_in = torch.cat(parts, dim=-1)  # [BT, N, F+1(+Fp)+Fns(+Ftime)]
        h0 = self.encoder(node_in).reshape(b * n, -1)  # [BT*N, hidden]
        h = h0

        big_edge_index = self._batched_edges(edge_index, n, b)  # [2, BT*E]
        edge_attr = (
            self._encode_edges(mi, b, device, dtype) if self._use_edges else None
        )

        for conv, norm in zip(self.convs, self.norms):
            m = (
                conv(h, big_edge_index, edge_attr)
                if self._use_edges
                else conv(h, big_edge_index)
            )
            h = norm(h + self.dropout(m))  # residual + norm

        if self._input_skip:
            h = torch.cat([h, h0], dim=-1)  # [BT*N, 2*hidden]
        return self.head(h).reshape(*lead, n, self.n_out_features)  # [*B, T, N, Fout]


__all__ = ["GNN"]
