"""Network structure as a PyG graph (row-aligned to ``node_phase_index``).

Graph feature layout (frozen; also summarized in the model I/O schema)
----------------------------------------------------------------------
Nodes of the PyG ``Data`` are the ``N`` node-phase ROWS of :func:`node_phase_index`, in row
order, so ``data.x[i]`` aligns with state ``V[..., i]`` and mask column ``i``. Per-row node
feature vector ``x`` ``[N, 8]`` (real ``float32``):

    [0]    log1p(u_rated_v)             rated voltage (kV-scale, log-compressed)
    [1:5]  phase one-hot (a,b,c,n)      Phase ordinal of the row
    [5]    is_slack                     1.0 if the row's node is a slack/source node
    [6]    depth_from_slack             distance to the nearest slack (inf->-1); km, or the
                                        grid-NORMALIZED depth under the cross-grid base
    [7]    log1p(node_degree)           branch degree of the row's node (connectivity)

Two of the eight base features are grid-ABSOLUTE — the rated voltage and the slack distance
in km — which is exactly what a model must not memorize when it trains across grids of
different size and voltage class. :class:`~injection_hse.config.StructuralEncodingConfig` therefore
selects the base variant: ``base="absolute"`` (default) keeps column ``[6]`` in kilometres
(single-grid runs stay byte-identical), ``base="normalized"`` replaces it with the depth
divided by the grid's own maximum finite depth, so it reads as "how far down this feeder"
in ``[0, 1]`` regardless of the grid's physical extent. The same config appends the optional
STRUCTURAL ENCODING columns (:mod:`injection_hse.data.encodings`) after column ``[7]``, widening ``x``
to ``[N, 8 + K]``; ``K`` is recorded on the graph as ``node_encoding_dim``.

Edges are PHASE-RESOLVED branch connections: an in-service branch joining
``from_phases[k]`` of ``from_node`` to ``to_phases[k]`` of ``to_node`` produces an undirected
pair of directed edges between the two corresponding node-phase rows (open switches dropped,
via ``pgml.topology``). Per-edge feature vector ``edge_attr`` ``[E, 5]`` (``float32``):

    [0]    log1p(|series_resistance|)   per-phase series R magnitude (line: R0*len; tx/sw: R)
    [1]    log1p(|series_reactance@f0|) per-phase series X = 2*pi*f0*L magnitude
    [2]    log1p(length_m)              line length (0 for non-lines)
    [3:5]  branch-type one-hot          (line, transformer/other) — switches fold into line

That edge set connects phase ``k`` to phase ``k`` and nothing else, while the physics it
stands for does couple the phases: a line's series impedance is a per-phase MATRIX and a Dyn
transformer's winding mixes the phases (:mod:`injection_hse.data.coupling`). On a three-phase feeder
the default graph is therefore DISCONNECTED — one component per phase — so a cross-phase
dependency is unreachable at any message-passing depth.
:class:`~injection_hse.config.PhaseCouplingConfig` (``coupling=``, default off) adds the missing
edges: every structurally-nonzero off-diagonal entry of the branch primitive admittance
stamps that the series set does not already carry, APPENDED after all series half-edges (so
``edge_index[:, :E_series]`` and ``edge_attr[:E_series, :5]`` are unchanged) and carrying
their branch's provenance. Their ``[0]`` / ``[1]`` columns hold the equivalent COUPLING
impedance ``-1 / Y[p, q]`` in ohms — the same quantity and units as a series edge's ``R`` and
``X``, large where the coupling is weak — and ``[2]`` / ``[3:5]`` their branch's length and
type. Two columns are appended, for every edge, whenever the config is on:

    [5]    is_coupling                  1.0 on a mutual-admittance edge, 0.0 on a series one
    [6]    is_same_node                 1.0 when both rows belong to the same node

so ``edge_attr`` is ``[E, 5 + 2]``. The width follows the CONFIG, not the grid (a grid with
no coupling still yields the wider block, all-zero in the two columns), so every member of a
multi-grid corpus and every checkpoint of a run agree on the model's edge input width —
:func:`edge_feature_dim`. Node features are untouched by the coupling edges (the degree
column ``[7]`` stays the BRANCH degree), and so are the structural node encodings, which read
the series edge table.

All features are detached constants (topology metadata, not on the autograd tape): the graph
is a fixed model INPUT, not a differentiable quantity. Built on CPU; moved to the batch
device by the training step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from pgml.assembly import node_phase_index
from pgml.topology import distance_from_slack, slack_node_ids
from pgml.schemas.grid_schema import Line, Phase, Switch, Transformer

from .coupling import COUPLING_FEATURE_DIM, coupling_edges

if TYPE_CHECKING:  # heavy / optional imports kept out of the import path
    import torch_geometric.data as pyg

    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import PhaseCouplingConfig, StructuralEncodingConfig


_PHASE_CODE = {Phase.A: 0, Phase.B: 1, Phase.C: 2, Phase.N: 3}

#: Width of the base node-feature block (before any structural encoding columns).
BASE_NODE_FEATURE_DIM = 8
#: Width of the SERIES edge-feature block (before any phase-coupling columns).
EDGE_FEATURE_DIM = 5


def edge_feature_dim(coupling: Optional["PhaseCouplingConfig"] = None) -> int:
    """Width of :func:`build_graph`'s ``edge_attr`` under a phase-coupling config.

    ``None`` / disabled = :data:`EDGE_FEATURE_DIM`; enabled adds the
    :data:`~injection_hse.data.coupling.COUPLING_FEATURE_DIM` flag columns. Depends on the CONFIG
    alone, so a model can size its edge input before any grid is loaded (the counterpart of
    :func:`injection_hse.data.encoding_dim` on the node side).
    """
    enabled = coupling is not None and coupling.enabled
    return EDGE_FEATURE_DIM + (COUPLING_FEATURE_DIM if enabled else 0)


@dataclass(frozen=True)
class _PhaseEdges:
    """The UNDIRECTED phase-resolved branch connections of a grid (one entry per edge).

    The single source of the network's edge set: :func:`build_graph` emits each entry as the
    adjacent directed half-edge PAIR (``from->to`` then ``to->from``), and the electrical
    encodings (:mod:`injection_hse.data.encodings`) weight the same entries by their series impedance.
    All lists are aligned and ordered by (branch, phase) — the layout every half-edge index
    derives from. Values are plain python floats (detached topology metadata).
    """

    src: list[int]
    dst: list[int]
    resistance_ohm: list[float]
    reactance_ohm: list[float]
    length_m: list[float]
    is_line: list[bool]
    branch_id: list[int]
    #: branch degree per NODE id (parallel/phase-resolved branches counted once per branch)
    degree: dict[int, int]

    def __len__(self) -> int:
        return len(self.src)


def _per_phase_diag(matrix, k: int) -> float:
    """Diagonal entry ``[k, k]`` of a per-phase matrix, as a python float (0 if absent)."""
    if matrix is None:
        return 0.0
    val = matrix[k][k]
    if hasattr(val, "detach"):
        val = val.detach().reshape(-1)[0].item()
    return float(val)


def _phase_edges(grid: "Grid", idx) -> _PhaseEdges:
    """Collect the in-service, phase-resolved branch connections between node-phase rows.

    Open switches, out-of-service branches and self-loops are dropped (the
    ``pgml.topology`` drawable-branch rule); a branch phase absent at a terminal is skipped.
    """
    f0 = float(grid.base_frequency_hz)
    src: list[int] = []
    dst: list[int] = []
    r_ohm: list[float] = []
    x_ohm: list[float] = []
    length: list[float] = []
    is_line_flags: list[bool] = []
    branch_ids: list[int] = []
    degree: dict[int, int] = {int(node.id): 0 for node in grid.nodes}

    for b in grid.branches:
        if not b.in_service:
            continue
        if isinstance(b, Switch) and not b.default_closed:
            continue
        if int(b.from_node) == int(b.to_node):
            continue
        degree[int(b.from_node)] = degree.get(int(b.from_node), 0) + 1
        degree[int(b.to_node)] = degree.get(int(b.to_node), 0) + 1

        length_m = float(b.length_m) if isinstance(b, Line) else 0.0
        # a switch folds into the "line" type slot (a near-ideal series connection)
        is_line = isinstance(b, (Line, Switch))

        for k, (fph, tph) in enumerate(zip(b.from_phases, b.to_phases)):
            try:
                fr = idx.row(int(b.from_node), fph)
                to = idx.row(int(b.to_node), tph)
            except KeyError:
                continue  # branch phase absent at a terminal (defensive)
            r, xreac = _edge_series_rx(b, k, f0, length_m)
            src.append(fr)
            dst.append(to)
            r_ohm.append(r)
            x_ohm.append(xreac)
            length.append(length_m)
            is_line_flags.append(is_line)
            branch_ids.append(int(b.id))

    return _PhaseEdges(
        src=src,
        dst=dst,
        resistance_ohm=r_ohm,
        reactance_ohm=x_ohm,
        length_m=length,
        is_line=is_line_flags,
        branch_id=branch_ids,
        degree=degree,
    )


def build_graph(
    grid: "Grid",
    *,
    orders: Optional[list[int]] = None,
    encoding: Optional["StructuralEncodingConfig"] = None,
    coupling: Optional["PhaseCouplingConfig"] = None,
) -> "pyg.Data":
    """Build the PyG graph (node/edge features + ``edge_index``) from a ``pgml.Grid``.

    The graph's nodes are the ``N`` node-phase rows of
    ``pgml.assembly.node_phase_index(grid)``, IN ROW ORDER, so ``data.x[i]`` aligns with the
    state ``V[..., i]`` and the measurement mask column ``i``. Node features carry node
    parameters; edges are phase-resolved branch connections with branch (line/transformer)
    parameter features. See the module docstring for the exact feature layout.

    Parameters
    ----------
    grid:
        The grid the dataset was generated on (the topology + parameters).
    orders:
        Unused for the node/edge layout (kept for forward compatibility with a future
        per-order edge admittance feature); accepted so callers can pass the solved orders.
    encoding:
        Optional :class:`~injection_hse.config.StructuralEncodingConfig` selecting the base-feature
        variant (``base="normalized"`` makes the slack depth grid-relative) and the
        structural node encoding appended after the base block (``"rwse"`` / ``"lappe"`` /
        ``"electrical"``). ``None`` (default) = the absolute base, no encoding — the
        single-grid layout, byte-identical.
    coupling:
        Optional :class:`~injection_hse.config.PhaseCouplingConfig` adding the network's PHASE-TO-PHASE
        admittance couplings (:mod:`injection_hse.data.coupling`) as extra half-edges after the series
        ones, and two flag columns to ``edge_attr``. ``None`` (default) or ``enabled=False``
        = the series-only edge set, byte-identical.

    Returns
    -------
    torch_geometric.data.Data
        ``x`` ``[N, 8 + K]`` float32 (``K`` = the structural-encoding width, 0 by default),
        ``edge_index`` ``[2, E]`` int64, ``edge_attr`` ``[E, Fe]`` float32
        (``Fe = 5``, ``7`` with the phase coupling on — :func:`edge_feature_dim`),
        ``edge_branch_ids`` ``[E]`` int64 (the branch each half-edge came from),
        ``node_encoding_dim = K``, ``num_nodes = N``. All on CPU (detached topology
        metadata).
    """
    from torch_geometric.data import Data

    from injection_hse._switches import materialize_estimator_grid

    del orders  # layout is order-independent for now (see docstring)

    grid = materialize_estimator_grid(grid)

    idx = node_phase_index(grid)
    n = idx.size

    slack_ids = set(slack_node_ids(grid))
    dist = distance_from_slack(grid, slack_ids)
    edges = _phase_edges(grid, idx)

    # -- node features ---------------------------------------------------------
    normalized_depth = encoding is not None and encoding.base == "normalized"
    depth_scale = 1.0
    if normalized_depth:
        finite = [d for d in dist.values() if not math.isinf(d)]
        peak = max(finite) if finite else 0.0
        depth_scale = 1.0 / peak if peak > 0.0 else 1.0

    x = torch.zeros((n, BASE_NODE_FEATURE_DIM), dtype=torch.float32)
    for node in grid.nodes:
        node_rows = idx.rows(node.id)
        u = float(node.u_rated_v)
        d = dist.get(int(node.id), float("inf"))
        d_feat = -1.0 if math.isinf(d) else float(d) * depth_scale
        is_slack = 1.0 if int(node.id) in slack_ids else 0.0
        degree_feat = math.log1p(edges.degree.get(int(node.id), 0))
        for ph, row in zip(node.phases, node_rows):
            x[row, 0] = math.log1p(u)
            x[row, 1 + _PHASE_CODE[ph]] = 1.0
            x[row, 5] = is_slack
            x[row, 6] = d_feat
            x[row, 7] = degree_feat

    # -- edges: each undirected connection as an adjacent half-edge pair -------
    # The SERIES (phase-resolved branch) connections first, then — when configured — the
    # mutual-admittance ones, so the leading half-edges keep the series-only layout every
    # edge-indexed consumer (branch currents, metered half-edges) is built for.
    coupled = coupling_edges(grid, index=idx, series=edges, config=coupling)
    width = edge_feature_dim(coupling)
    rows = [
        _edge_feature_row(r, xr, lm, line, width, coupling=False, same_node=False)
        for r, xr, lm, line in zip(
            edges.resistance_ohm, edges.reactance_ohm, edges.length_m, edges.is_line
        )
    ]
    src = list(edges.src)
    dst = list(edges.dst)
    branch_ids = list(edges.branch_id)
    if coupled is not None:
        rows.extend(
            _edge_feature_row(r, xr, lm, line, width, coupling=True, same_node=node)
            for r, xr, lm, line, node in zip(
                coupled.resistance_ohm,
                coupled.reactance_ohm,
                coupled.length_m,
                coupled.is_line,
                coupled.same_node,
            )
        )
        src.extend(coupled.src)
        dst.extend(coupled.dst)
        branch_ids.extend(coupled.branch_id)

    if rows:
        pair_src = torch.tensor(src, dtype=torch.int64)
        pair_dst = torch.tensor(dst, dtype=torch.int64)
        edge_index = torch.stack(
            [
                torch.stack([pair_src, pair_dst], dim=-1).reshape(-1),
                torch.stack([pair_dst, pair_src], dim=-1).reshape(-1),
            ]
        )
        attr = torch.tensor(rows, dtype=torch.float32)
        edge_attr = attr.repeat_interleave(2, dim=0)  # identical on both directions
        edge_branch_ids = torch.tensor(branch_ids, dtype=torch.int64).repeat_interleave(
            2
        )
    else:  # no drawable branches (degenerate grid)
        edge_index = torch.zeros((2, 0), dtype=torch.int64)
        edge_attr = torch.zeros((0, width), dtype=torch.float32)
        edge_branch_ids = torch.zeros(0, dtype=torch.int64)

    # -- optional structural encoding, appended after the base block -----------
    from .encodings import structural_node_encoding

    extra = structural_node_encoding(grid, encoding, index=idx, edges=edges)
    if extra is not None:
        x = torch.cat([x, extra], dim=-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # provenance of every half-edge (which branch it came from) — not a model feature; the
    # switchable-topology layer maps per-variant switch states onto the half-edges with it.
    data.edge_branch_ids = edge_branch_ids
    data.node_encoding_dim = 0 if extra is None else int(extra.shape[-1])
    data.num_nodes = n
    return data


def _edge_feature_row(
    r: float,
    x: float,
    length_m: float,
    is_line: bool,
    width: int,
    *,
    coupling: bool,
    same_node: bool,
) -> list[float]:
    """One row of ``edge_attr`` (the module docstring's layout), padded to ``width``."""
    row = [
        math.log1p(abs(r)),
        math.log1p(abs(x)),
        math.log1p(length_m),
        *((1.0, 0.0) if is_line else (0.0, 1.0)),
    ]
    if width > EDGE_FEATURE_DIM:
        row.extend([1.0 if coupling else 0.0, 1.0 if same_node else 0.0])
    return row


def _edge_series_rx(b, k: int, f0: float, length_m: float) -> tuple[float, float]:
    """Per-phase series ``(R, X@f0)`` of a branch's phase ``k`` (line/transformer/switch)."""
    if isinstance(b, Line):
        r = _per_phase_diag(b.series_resistance_ohm_per_m, k) * length_m
        ll = _per_phase_diag(b.series_inductance_h_per_m, k) * length_m
        return r, 2.0 * math.pi * f0 * ll
    if isinstance(b, Transformer):
        r = (
            float(b.series_resistance_ohm)
            if b.series_resistance_ohm is not None
            else 0.0
        )
        ll = float(b.series_inductance_h) if b.series_inductance_h is not None else 0.0
        return r, 2.0 * math.pi * f0 * ll
    if isinstance(b, Switch):
        r = float(b.resistance_ohm)
        ll = float(b.inductance_h)
        return r, 2.0 * math.pi * f0 * ll
    return 0.0, 0.0


__all__ = [
    "build_graph",
    "edge_feature_dim",
    "BASE_NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
]
