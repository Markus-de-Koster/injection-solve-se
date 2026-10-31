"""Graphormer state estimator: topology-aware Transformer (structural + spatial encodings).

A Transformer over the ``N`` node-phase tokens, made structure-aware Graphormer-style so
attention is informed by the grid topology rather than blind:

- **Degree (centrality) encoding** — a learned embedding indexed by each node-phase row's
  branch degree, added to its token embedding.
- **Attention bias** — a learned per-(distance bucket, head) scalar added to the attention
  logits. Keying the bias by DISTANCE BUCKET (not node identity) keeps it transferable across
  grids of differing size/topology. :attr:`~injection_hse.config.GraphormerConfig.attention_bias` selects
  the distance:

  - ``"spd"`` (default) — the unweighted shortest-path HOP count between node-phase rows on
    the :func:`injection_hse.data.build_graph` topology.
  - ``"electrical"`` — the pairwise EFFECTIVE RESISTANCE of the network, per-unit-normalized
    per grid and log-bucketed. A hop treats a 1 m switch and a 500 m cable alike; effective
    resistance does not, and it is the quantity a voltage drop responds to. Derived from the
    ``edge_static`` series-resistance column, so the bias needs nothing beyond the
    :class:`~injection_hse.io.ModelInput` and a reloaded checkpoint reproduces it.
  - ``"both"`` — the two learned biases are summed.

Edge features are not otherwise folded into the bias — the degree + distance encodings already
make attention topology-aware; an edge-feature bias on adjacent pairs is a documented future
extension. The Graphormer consumes ``node_obs`` + ``node_mask`` + ``node_static`` and
accepts-and-ignores ``edge_obs`` / ``injection_in`` (the interface carries them). Pure torch (no
PyG dependency), so unlike the GNN it constructs and runs on any supported Python. A dense
``[N, N]`` attention + bias is fine at the distribution-feeder scale (``N`` ~ hundreds).

Tensor contract (see :class:`injection_hse.models.base.StateEstimator`):
``forward(model_input) -> [*B, T, N, Fout]`` with ``Fnd = H * encoding.n_features`` the input
width and ``Fout`` the output width (``n_out_features``; ``Fnd`` unless the injection decode
gives its head an encoding of its own — only the head projection follows it). Per-node
token input = ``node_obs ⧺ node_mask channel ⧺ node_static[N, Fns]`` (⧺ the optional per-row
``node_prior`` physics-derived block ⧺ the optional per-step ``node_time`` features, broadcast
to every node); ``T`` is folded into the batch and estimated per step.

Batching (vectorized, NO python loop over the batch on the autograd tape)
------------------------------------------------------------------------
Standard ``[BT, heads, N, N]`` attention over the (time-folded) batch. The structural
encodings — the degree index ``[N]`` and the distance bucket matrices ``[N, N]`` — are
GRID-CONSTANT: computed once per graph and cached (by tensor identity first, then by the exact
edge set / edge features, so two different topologies can never share a stale bias), broadcast
over the batch as an additive ``[heads, N, N]`` bias.

MULTI-GRID batches (:class:`injection_hse.data.MultiGridCollate`) concatenate several member graphs onto
one node axis, and dense attention over that axis would let every grid read every other one. The
attention is therefore made BLOCK-DIAGONAL by construction: the concatenated tokens are
scattered into a padded per-graph layout ``[BT*B, maxN, hidden]``, attention runs there with a
key-padding bias, and the result is gathered back to ``[BT, N*, hidden]``. Each member carries
its OWN structural bundle (cached per member — a bias keyed to one grid's hop counts is
meaningless on another), so a batched forward equals running each member alone and concatenating
the outputs (verified in the tests). The scatter/gather are index ops on the autograd tape; the
only python loop is over the (few) member graphs while assembling their cached bias blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from injection_hse.config import GraphormerConfig
from injection_hse.io import ModelInput

from .base import (
    StateEstimator,
    broadcast_global_features,
    order_mask_channels,
    prior_channels,
)

#: Additive logit put on a PADDED (non-existent) key so softmax gives it zero weight. Large
#: enough to vanish in float32 and float64, finite so no ``-inf`` arithmetic can produce a NaN.
_PAD_LOGIT = -1.0e9

#: Cap on the number of distinct graphs whose structural tensors are held. Each entry is a pair
#: of ``[N, N]`` index matrices, so an unbounded cache would grow with the member count of a
#: large multi-grid dataset; the cache is dropped wholesale past the cap and re-derived on
#: demand.
_MAX_CACHED_GRAPHS = 256


class _SpectralPE(nn.Module):
    """SAN-style learned node encoding over Laplacian eigenpairs.

    Each node owns a length-``K`` sequence ``(lambda_i, phi_i(node))``.  A small Transformer
    attends across frequencies and sum-pools the sequence into one vector that is added to
    the Graphormer token.  Training draws one independent sign per graph and eigenvector,
    matching the sign augmentation used by Kreuzer et al. (NeurIPS 2021).
    """

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        width = min(32, hidden)
        heads = 4 if width % 4 == 0 else 1
        self.dim = int(dim)
        self.embed = nn.Linear(2, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(width, hidden)

    def forward(
        self, pairs: Tensor, *, node_batch: Tensor | None = None
    ) -> Tensor:
        # pairs [N, K, 2], with zero eigenvalues denoting fixed-width padding.
        values = pairs[..., 0]
        valid = values > 0.0
        work = pairs
        if self.training:
            n_graphs = 1 if node_batch is None else int(node_batch.max().item()) + 1
            signs = torch.randint(
                0, 2, (n_graphs, self.dim), device=pairs.device, dtype=torch.int64
            ).to(pairs.dtype)
            signs = signs.mul_(2.0).sub_(1.0)
            row_signs = signs[0] if node_batch is None else signs[node_batch]
            work = pairs.clone()
            work[..., 1] = work[..., 1] * row_signs

        # TransformerEncoder returns NaN when every position of a row is masked. Keep one
        # zero token alive for a degenerate/isolated graph, then discard it before pooling.
        padding = ~valid
        all_padding = padding.all(dim=1)
        if bool(all_padding.any()):
            padding = padding.clone()
            padding[all_padding, 0] = False
        encoded = self.encoder(self.embed(work), src_key_padding_mask=padding)
        encoded = encoded.masked_fill((~valid).unsqueeze(-1), 0.0)
        return self.out(encoded.sum(dim=1))


@dataclass(frozen=True)
class _StructuralBundle:
    """The grid-constant attention inputs of ONE graph (detached metadata, never on the tape).

    ``degree`` ``[N]`` and the bucket matrices ``[N, N]`` are int64 CPU tensors; ``edges`` /
    ``edge_static`` are the exact inputs the bundle was derived from and serve as its cache
    identity (an equality check, never a digest, so two topologies cannot silently share a
    bias).
    """

    n_nodes: int
    degree: Tensor  # [N] int64
    spatial_index: Tensor  # [N, N] int64 — hop-distance buckets
    electrical_index: Optional[Tensor]  # [N, N] int64 — effective-resistance buckets
    edges: Tensor  # [2, E] int64 (CPU)
    edge_static: Optional[
        Tensor
    ]  # [E, Fes] (CPU) — only when the electrical bias is on


class _GraphormerBlock(nn.Module):
    """Pre-norm multi-head self-attention (with an additive structural bias) + FFN."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(
                f"hidden_dim {hidden} must be divisible by num_heads {heads}."
            )
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_bias: Tensor) -> Tensor:
        """``x [B, N, hidden]`` + ``attn_bias [B|1, heads, N, N]`` -> ``[B, N, hidden]``."""
        b, n, hidden = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, heads, N, head_dim]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, heads, N, N]
        scores = scores + attn_bias  # broadcast over the token batch
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, hidden)  # [B, N, hidden]
        x = x + self.drop(self.proj(ctx))
        x = x + self.ffn(self.norm2(x))
        return x


class Graphormer(StateEstimator):
    """Structure-aware Transformer estimator (degree encoding + distance attention bias)."""

    #: static node-feature width of the BASE :func:`injection_hse.data.build_graph` block (the default
    #: when no structural encoding widens ``node_static``).
    NODE_FEATURE_DIM = 8
    #: degree buckets (node degree clamped into ``[0, NUM_DEGREE-1]``).
    NUM_DEGREE = 32
    #: hop-distance buckets: ``0`` (self) .. ``NUM_SPATIAL-2``, with ``NUM_SPATIAL-1`` =
    #: "far / unreachable".
    NUM_SPATIAL = 32
    #: effective-resistance buckets: ``0`` (zero distance) .. ``NUM_ELECTRICAL-2``
    #: (log-spaced over the per-grid-normalized resistance), ``NUM_ELECTRICAL-1`` = "no path"
    #: — the same convention as the hop buckets, so ``"both"`` sums two aligned tables.
    NUM_ELECTRICAL = 32
    #: decades of per-unit effective resistance the log-spaced buckets span (``1e-6 … 1``).
    ELECTRICAL_DECADES = 6.0

    def __init__(
        self,
        cfg: GraphormerConfig,
        *,
        n_features: int,
        n_time_features: int = 0,
        n_topology_features: int = 0,
        n_order_mask_channels: int = 0,
        n_prior_features: int = 0,
        n_static_features: int | None = None,
        n_out_features: int | None = None,
    ) -> None:
        super().__init__()
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
        # width of ``node_static``: the base block, plus a structural encoding's ``K`` columns
        # when the graph carries one (:class:`~injection_hse.config.StructuralEncodingConfig`).
        self.n_static_features = (
            self.NODE_FEATURE_DIM
            if n_static_features is None
            else int(n_static_features)
        )
        self.spectral_pe_dim = int(cfg.spectral_pe_dim)
        spectral_width = 2 * self.spectral_pe_dim
        if spectral_width > self.n_static_features:
            raise ValueError(
                f"spectral_pe_dim={self.spectral_pe_dim} needs {spectral_width} trailing "
                f"node-static columns, got {self.n_static_features}."
            )
        self.base_static_features = self.n_static_features - spectral_width
        hidden = cfg.hidden_dim
        # features + mask (+ optional per-order observability channels) (+ optional
        # physics-derived prior channels) + static node feats (+ optional per-step time
        # channels)
        in_dim = (
            n_features
            + 1
            + n_order_mask_channels
            + n_prior_features
            + self.base_static_features
            + n_time_features
            + n_topology_features
        )
        self.input_proj = nn.Linear(in_dim, hidden)
        self.spectral_pe = (
            _SpectralPE(self.spectral_pe_dim, hidden, cfg.dropout)
            if self.spectral_pe_dim
            else None
        )
        self.degree_emb = nn.Embedding(self.NUM_DEGREE, hidden)
        bias_kind = cfg.attention_bias
        self._use_spd = bias_kind in ("spd", "both")
        self._use_electrical = bias_kind in ("electrical", "both")
        # the hop-distance table is built for "spd"/"both" only, so an "electrical" model
        # carries no unused parameters (and vice versa).
        self.spatial_bias = (
            nn.Embedding(self.NUM_SPATIAL, cfg.num_heads) if self._use_spd else None
        )
        self.electrical_bias = (
            nn.Embedding(self.NUM_ELECTRICAL, cfg.num_heads)
            if self._use_electrical
            else None
        )
        self.blocks = nn.ModuleList(
            _GraphormerBlock(hidden, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.n_out_features)
        # grid-constant structural tensors. The identity slot short-circuits the common case
        # (one shared graph reused across steps) without ever touching the host; the keyed
        # cache serves a multi-grid batch, whose concatenated edge tensor is new every batch
        # while its MEMBERS repeat.
        self._identity_edges: Tensor | None = None  # last edge_index object seen
        self._identity_bundle: _StructuralBundle | None = None
        self._cache: dict[tuple[int, int], list[_StructuralBundle]] = {}
        self._cache_size = 0

    # -- structural tensors ---------------------------------------------------
    @classmethod
    def _compute_structural(cls, edge_index: Tensor, n: int) -> tuple[Tensor, Tensor]:
        """Hop-distance bucket matrix ``[N, N]`` + degree bucket ``[N]`` from ``edge_index``.

        Unweighted all-pairs shortest path via BFS layering done as a frontier matrix
        expansion over all sources at once (``N`` ~ hundreds, a handful of dense ``[N, N]``
        products — cheap). Distances are capped/bucketed into ``[0, NUM_SPATIAL-1]``
        (``NUM_SPATIAL-1`` = far/unreachable).
        """
        adj = torch.zeros((n, n), dtype=torch.float32)
        if edge_index.numel():
            adj[edge_index[0], edge_index[1]] = 1.0
            adj[edge_index[1], edge_index[0]] = 1.0  # ensure symmetric (undirected)
        adj.fill_diagonal_(0.0)
        degree = adj.sum(1).clamp(min=0, max=cls.NUM_DEGREE - 1).to(torch.int64)  # [N]

        far = cls.NUM_SPATIAL - 1
        dist = torch.full((n, n), far, dtype=torch.int64)
        idx = torch.arange(n)
        dist[idx, idx] = 0
        reached = torch.eye(n, dtype=torch.bool)
        frontier = torch.eye(n, dtype=torch.bool)
        for k in range(1, far):  # hops 1 .. NUM_SPATIAL-2
            nxt = (
                frontier.float() @ adj
            ) > 0  # [N, N]: neighbors of the current frontier
            newly = nxt & ~reached
            if not bool(newly.any()):
                break
            dist[newly] = k
            reached = reached | newly
            frontier = newly
        return dist, degree

    @classmethod
    def _compute_electrical(
        cls, edge_index: Tensor, edge_static: Tensor, n: int
    ) -> Tensor:
        """Effective-resistance bucket matrix ``[N, N]`` (int64) of one graph.

        The pairwise effective resistance ``L+_ii + L+_jj - 2 L+_ij`` of the weighted Laplacian
        (edge conductance ``1 / R``), normalized by the graph's own maximum so it is per-unit
        within the grid, then log-bucketed over :attr:`ELECTRICAL_DECADES` decades. The series
        resistances come from the ``edge_static`` column :func:`injection_hse.data.build_graph` fills with
        ``log1p(|R|)``, inverted here — so the model derives the bias from its own input and
        needs no grid object at inference. It therefore reads whatever edge set the graph
        carries: the series one (equal to :func:`injection_hse.data.pairwise_effective_resistance`
        bucket for bucket) or, with :class:`~injection_hse.config.PhaseCouplingConfig` on, the one that
        also carries the mutual-admittance edges. Pairs in different connected components (a
        grid's neutral rows, an islanded phase) get the "no path" bucket, matching the hop
        table's unreachable convention.
        """
        # ONE implementation of the effective-resistance algebra: the component split and the
        # grounded-Laplacian inverse live with the structural encodings.
        from injection_hse.data.encodings import (
            MIN_SERIES_OHM,
            _component_potential_matrix,
            _components,
        )

        out = torch.full((n, n), -1.0, dtype=torch.float64)
        # ``build_graph`` emits both directions of every connection; keep one so a parallel
        # branch counts once (an overall factor on the conductances cancels in the
        # normalization anyway).
        if edge_index.numel():
            keep = edge_index[0] < edge_index[1]
            src = edge_index[0][keep].tolist()
            dst = edge_index[1][keep].tolist()
            resistance = torch.expm1(edge_static[:, 0].to(torch.float64).clamp_min(0.0))
            weights = [
                1.0 / max(float(r), MIN_SERIES_OHM) for r in resistance[keep].tolist()
            ]
        else:
            src, dst, weights = [], [], []
        for rows in _components(src, dst, n):
            potential = _component_potential_matrix(src, dst, weights, rows)
            block = (
                torch.diagonal(potential).unsqueeze(1)
                + torch.diagonal(potential).unsqueeze(0)
                - 2.0 * potential
            )
            member = torch.tensor(rows, dtype=torch.int64)
            out[member.unsqueeze(1), member.unsqueeze(0)] = block
        out.fill_diagonal_(0.0)
        finite = out[out >= 0.0]
        peak = float(finite.max()) if finite.numel() else 0.0
        if peak > 0.0:
            out = torch.where(out >= 0.0, out / peak, out)

        far = cls.NUM_ELECTRICAL - 1
        span = cls.NUM_ELECTRICAL - 2  # buckets 1 .. NUM_ELECTRICAL-2
        decades = cls.ELECTRICAL_DECADES
        log_r = torch.log10(out.clamp_min(1e-30))
        bucket = 1 + torch.floor((log_r + decades) / decades * span).to(torch.int64)
        bucket = bucket.clamp(min=1, max=span)
        bucket = torch.where(out > 0.0, bucket, torch.full_like(bucket, far))
        return torch.where(out == 0.0, torch.zeros_like(bucket), bucket)

    def _build_bundle(
        self, edges: Tensor, edge_static: Optional[Tensor], n: int
    ) -> _StructuralBundle:
        """Derive one graph's structural tensors (CPU; called once per distinct graph)."""
        spatial, degree = self._compute_structural(edges, n)
        electrical = None
        if self._use_electrical:
            if edge_static is None:
                raise ValueError(
                    "the electrical attention bias needs edge_static (the series-impedance "
                    "columns of injection_hse.data.build_graph)."
                )
            electrical = self._compute_electrical(edges, edge_static, n)
        return _StructuralBundle(
            n_nodes=n,
            degree=degree,
            spatial_index=spatial,
            electrical_index=electrical,
            edges=edges,
            edge_static=edge_static if self._use_electrical else None,
        )

    def _cached_bundle(
        self, edges: Tensor, edge_static: Optional[Tensor], n: int
    ) -> _StructuralBundle:
        """The bundle for one graph, keyed by its EXACT edge set (and impedances)."""
        key = (n, int(edges.shape[1]))
        for cand in self._cache.get(key, ()):
            if not torch.equal(cand.edges, edges):
                continue
            if cand.edge_static is not None and not torch.equal(
                cand.edge_static, edge_static
            ):
                continue  # same topology, different impedances -> a different grid
            return cand
        bundle = self._build_bundle(edges, edge_static, n)
        if self._cache_size >= _MAX_CACHED_GRAPHS:
            self._cache.clear()
            self._cache_size = 0
        self._cache.setdefault(key, []).append(bundle)
        self._cache_size += 1
        return bundle

    def _batch_bundles(self, mi: ModelInput, n: int) -> list[_StructuralBundle]:
        """The structural bundle of every graph in this input (one entry per member).

        A single-graph input hits the identity slot and never leaves the device; a multi-grid
        input is split at the ``node_ptr`` boundaries and each member is looked up by its own
        (offset-removed) edge set.
        """
        n_graphs = mi.n_graphs
        if n_graphs <= 1:
            identity = self._identity_bundle
            if mi.edge_index is self._identity_edges and identity is not None:
                if identity.n_nodes == n:
                    return [identity]
            static = mi.edge_static.detach().cpu() if self._use_electrical else None
            bundle = self._cached_bundle(mi.edge_index.detach().cpu(), static, n)
            self._identity_edges = mi.edge_index
            self._identity_bundle = bundle
            return [bundle]

        edges = mi.edge_index.detach().cpu()
        static = mi.edge_static.detach().cpu() if self._use_electrical else None
        node_ptr = mi.node_ptr.detach().cpu().tolist()
        edge_batch = (
            mi.edge_batch
            if mi.edge_batch is not None
            else torch.zeros(edges.shape[1], dtype=torch.int64)
        )
        counts = torch.bincount(edge_batch.detach().cpu(), minlength=n_graphs)
        edge_ptr = [0, *torch.cumsum(counts, 0).tolist()]
        bundles = []
        for i in range(n_graphs):
            lo, hi = edge_ptr[i], edge_ptr[i + 1]
            local = edges[:, lo:hi] - node_ptr[i]
            local_static = None if static is None else static[lo:hi]
            bundles.append(
                self._cached_bundle(
                    local.contiguous(), local_static, node_ptr[i + 1] - node_ptr[i]
                )
            )
        return bundles

    # -- attention bias -------------------------------------------------------
    def _bias(self, spatial: Tensor, electrical: Optional[Tensor] = None) -> Tensor:
        """Learned additive bias ``[..., heads, N, N]`` from the distance bucket matrices."""
        bias = None
        if self.spatial_bias is not None:
            bias = self.spatial_bias(spatial)  # [..., N, N, heads]
        if self.electrical_bias is not None:
            electrical_bias = self.electrical_bias(electrical)
            bias = electrical_bias if bias is None else bias + electrical_bias
        if bias is None:
            return torch.zeros(
                (*spatial.shape[:-2], self.cfg.num_heads, *spatial.shape[-2:]),
                dtype=torch.float32,
                device=spatial.device,
            )
        return bias.movedim(-1, -3)  # [..., heads, N, N]

    def _padded_bias(
        self, bundles: list[_StructuralBundle], max_n: int, device
    ) -> Tensor:
        """Block-diagonal attention bias ``[B, heads, maxN, maxN]`` of a multi-grid batch.

        Each member's bucket matrix is written into its own ``[N_b, N_b]`` block of a padded
        index matrix (the padding takes the "unreachable" bucket), the learned tables turn that
        into per-head logits, and every key OUTSIDE a member's own rows is pushed to
        :data:`_PAD_LOGIT` — so a token can only attend within its own grid. A padded QUERY row
        keeps its diagonal so its softmax stays finite; its output is dropped by the gather.
        """
        g = len(bundles)
        spatial = torch.full((g, max_n, max_n), self.NUM_SPATIAL - 1, dtype=torch.int64)
        electrical = (
            torch.full((g, max_n, max_n), self.NUM_ELECTRICAL - 1, dtype=torch.int64)
            if self._use_electrical
            else None
        )
        for i, bundle in enumerate(bundles):  # few members; the blocks are cached
            nb = bundle.n_nodes
            spatial[i, :nb, :nb] = bundle.spatial_index
            if electrical is not None:
                electrical[i, :nb, :nb] = bundle.electrical_index
        bias = self._bias(
            spatial.to(device),
            None if electrical is None else electrical.to(device),
        )  # [B, heads, maxN, maxN]

        sizes = torch.tensor([b.n_nodes for b in bundles], device=device)
        positions = torch.arange(max_n, device=device)
        valid_key = positions.unsqueeze(0) < sizes.unsqueeze(1)  # [B, maxN]
        diagonal = torch.eye(max_n, dtype=torch.bool, device=device)
        blocked = ~valid_key[:, None, None, :] & ~diagonal[None, None, :, :]
        return bias.masked_fill(blocked, _PAD_LOGIT)

    # -- forward --------------------------------------------------------------
    def forward(self, model_input: ModelInput) -> Tensor:
        mi, lead = model_input.fold_time_into_batch()
        features = mi.node_obs  # [BT, N, Fnd]
        mask = mi.node_mask  # [BT, N]
        b, n, f = features.shape
        device, dtype = features.device, features.dtype

        bundles = self._batch_bundles(mi, n)
        degree = torch.cat([bundle.degree for bundle in bundles]).to(device)  # [N]
        x_static = mi.node_static.to(device=device, dtype=dtype)  # [N, Fns]
        spectral_pairs = None
        if self.spectral_pe is not None:
            spectral_width = 2 * self.spectral_pe_dim
            spectral_pairs = x_static[:, -spectral_width:].reshape(
                n, self.spectral_pe_dim, 2
            )
            x_static = x_static[:, :-spectral_width]

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
        h = self.input_proj(node_in) + self.degree_emb(degree).unsqueeze(
            0
        )  # [BT, N, hidden]
        if self.spectral_pe is not None and spectral_pairs is not None:
            node_batch = (
                None
                if mi.node_batch is None
                else mi.node_batch.to(device=device, dtype=torch.int64)
            )
            h = h + self.spectral_pe(spectral_pairs, node_batch=node_batch).unsqueeze(0)

        if len(bundles) == 1:
            bundle = bundles[0]
            bias = self._bias(
                bundle.spatial_index.to(device),
                None
                if bundle.electrical_index is None
                else bundle.electrical_index.to(device),
            ).unsqueeze(0)  # [1, heads, N, N]
            for blk in self.blocks:
                h = blk(h, bias)
        else:
            h = self._blocks_per_graph(h, mi, bundles)
        return self.head(self.norm_out(h)).reshape(
            *lead, n, self.n_out_features
        )  # [*B, T, N, Fout]

    def _blocks_per_graph(
        self, h: Tensor, mi: ModelInput, bundles: list[_StructuralBundle]
    ) -> Tensor:
        """Run the blocks BLOCK-DIAGONALLY over a multi-grid batch. ``[BT, N*, hidden]`` in/out.

        The concatenated rows are scattered into a padded ``[BT*B, maxN, hidden]`` layout (one
        slot per (step, member)), attended with the per-member bias, and gathered back — so no
        attention weight ever links two member grids.
        """
        bt, n, hidden = h.shape
        device = h.device
        g = len(bundles)
        sizes = [bundle.n_nodes for bundle in bundles]
        max_n = max(sizes)
        node_ptr = mi.node_ptr.to(device=device, dtype=torch.int64)
        node_batch = (
            mi.node_batch.to(device=device, dtype=torch.int64)
            if mi.node_batch is not None
            else torch.repeat_interleave(
                torch.arange(g, device=device), node_ptr[1:] - node_ptr[:-1]
            )
        )
        # position of each concatenated row in the padded [B, maxN] layout
        local = torch.arange(n, device=device) - node_ptr[node_batch]
        pad_index = node_batch * max_n + local  # [N*]

        padded = h.new_zeros(bt, g * max_n, hidden).index_copy(1, pad_index, h)
        padded = padded.reshape(bt * g, max_n, hidden)
        bias = self._padded_bias(bundles, max_n, device)  # [B, heads, maxN, maxN]
        if bt > 1:
            bias = (
                bias.unsqueeze(0)
                .expand(bt, *bias.shape)
                .reshape(bt * g, *bias.shape[1:])
            )
        for blk in self.blocks:
            padded = blk(padded, bias)
        return padded.reshape(bt, g * max_n, hidden).index_select(1, pad_index)


__all__ = ["Graphormer"]
