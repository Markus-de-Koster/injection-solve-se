"""Structural node encodings: how a row is POSITIONED in its network.

A state estimator that trains on one grid can identify a row by its base features alone; one
that must generalize across grids cannot — the informative quantity is not "bus 12 of the
CIGRE feeder" but "a leaf three hops down a stiff branch". This module supplies that
positional signal as extra node-feature columns, appended to the base block of
:func:`injection_hse.data.build_graph` and selected by
:class:`~injection_hse.config.StructuralEncodingConfig`:

- **Random-walk structural encoding** (``kind="rwse"``) — the return probabilities
  ``diag((D^-1 A)^k)`` for ``k = 1 … K`` on the node-phase graph. Purely topological,
  sign-free and cheap:
  ``K`` sparse-times-dense products.
- **Laplacian positional encoding** (``kind="lappe"``) — the low-frequency non-trivial
  eigenpairs of the symmetric normalized Laplacian.  Each pair is stored as an eigenvalue
  and its row's eigenvector entry, matching the input representation of the Spectral
  Attention Network (SAN).  Eigenvector signs are canonicalized for reproducible graph
  metadata; the Graphormer's learned spectral encoder randomizes them during training.
- **Electrical encoding** (``kind="electrical"``) — the EFFECTIVE impedance from each row to
  the slack, split into ``(R_eff, X_eff)`` at the base frequency. Effective resistance is
  the electrical distance of a network: it equals the series path sum on a radial feeder,
  collapses parallel paths the way the network itself does, and is the quantity a voltage
  drop responds to. Per-unit-normalized per grid so it is comparable across voltage classes.

The pairwise counterpart, :func:`pairwise_effective_resistance`, returns the full
``[N, N]`` effective-resistance matrix of one grid — the electrical analogue of a
Graphormer's hop-distance attention bias (a topological hop treats a 1 m switch and a 500 m
cable alike; effective resistance does not).

Everything here is detached topology metadata computed ONCE per grid (never per sample and
never on the autograd tape): plain ``float32`` outputs on CPU, ``float64`` internally where
a linear solve is involved. Cost is stated per function; the dominant term for the
electrical features is one dense factorization of the grounded Laplacian, ``O(N^3)`` time and
``O(N^2)`` memory in the node-phase row count.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from pgml.assembly import node_phase_index
from pgml.topology import slack_node_ids

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import StructuralEncodingConfig

    from .graph import _PhaseEdges

#: Series impedance floor (ohm) used when a branch's R or X is zero (a closed switch is a
#: near-ideal conductor, not an infinite one). Keeps the weighted Laplacian finite and
#: well-conditioned without visibly changing any real branch.
MIN_SERIES_OHM = 1.0e-6

#: Value marking a row with no electrical path to a slack (mirrors the ``-1`` sentinel the
#: base features use for an unreachable slack distance).
UNREACHABLE = -1.0


def encoding_dim(config: Optional["StructuralEncodingConfig"]) -> int:
    """Width ``K`` of the structural encoding a config selects (``0`` for ``None``/``none``).

    The model input width is ``build_graph``'s base block plus this, so a model can be sized
    from the config alone (before any grid is loaded).
    """
    if config is None or config.kind == "none":
        return 0
    if config.kind == "rwse":
        return int(config.rwse_steps)
    if config.kind == "lappe":
        return 2 * int(config.lap_pe_dim)
    if config.kind == "electrical":
        return 2  # (R_eff, X_eff)
    raise ValueError(f"unknown structural encoding {config.kind!r}.")


def structural_node_encoding(
    grid: "Grid",
    config: Optional["StructuralEncodingConfig"],
    *,
    index=None,
    edges: Optional["_PhaseEdges"] = None,
) -> Optional[Tensor]:
    """The structural encoding columns ``[N, K]`` for a grid (``None`` when disabled).

    Parameters
    ----------
    grid:
        The grid whose node-phase rows are encoded.
    config:
        The selected encoding (``None`` or ``kind="none"`` -> ``None``, no columns).
    index, edges:
        Optional pre-computed ``NodePhaseIndex`` / phase-edge table (``build_graph`` passes
        the ones it already built; both are recomputed when omitted).

    Returns
    -------
    torch.Tensor | None
        ``float32`` ``[N, K]`` on CPU, aligned to ``node_phase_index(grid)`` row order.
    """
    if config is None or config.kind == "none":
        return None
    from .graph import _phase_edges

    idx = node_phase_index(grid) if index is None else index
    edge_table = _phase_edges(grid, idx) if edges is None else edges
    n = int(idx.size)

    if config.kind == "rwse":
        edge_index = _undirected_edge_index(edge_table)
        return random_walk_encoding(edge_index, n, steps=int(config.rwse_steps))
    if config.kind == "lappe":
        edge_index = _undirected_edge_index(edge_table)
        return laplacian_positional_encoding(edge_index, n, dim=int(config.lap_pe_dim))
    if config.kind == "electrical":
        return effective_impedance_to_slack(
            grid, index=idx, edges=edge_table, normalize=bool(config.normalize)
        )
    raise ValueError(f"unknown structural encoding {config.kind!r}.")


# =============================================================================
# Random-walk structural encoding
# =============================================================================
def random_walk_encoding(
    edge_index: Tensor, num_nodes: int, *, steps: int = 16
) -> Tensor:
    """``K``-step random-walk structural encoding ``[N, K]`` (``float32``).

    Column ``k`` (``k = 1 … steps``) holds ``diag(P^k)`` with ``P = D^-1 A`` — the
    probability that a ``k``-step uniform random walk started at a row returns to it. The
    profile over ``k`` characterizes a row's neighbourhood shape at growing radius (a leaf
    returns only on even steps and rarely, a hub returns often), so it separates
    structurally different rows without any node identity and transfers across grids.

    Parameters
    ----------
    edge_index:
        ``[2, E]`` int64 edge list. Both directions may be present (the adjacency is
        symmetrized here regardless); parallel entries are summed as multiplicity.
    num_nodes:
        ``N`` — the node count (rows of the result).
    steps:
        Walk length ``K`` (the encoding width).

    Returns
    -------
    torch.Tensor
        ``float32`` ``[N, steps]`` on the ``edge_index`` device. Isolated rows (degree 0)
        yield all-zero columns.

    Notes
    -----
    Computed as ``K`` sparse-times-dense products against the identity: ``O(K * E * N)``
    time, ``O(N^2)`` memory. Accumulated in ``float64`` and returned ``float32``.
    """
    if steps < 1:
        raise ValueError("random_walk_encoding needs steps >= 1.")
    device = edge_index.device
    n = int(num_nodes)
    out = torch.zeros((n, steps), dtype=torch.float32, device=device)
    if n == 0 or edge_index.numel() == 0:
        return out

    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])
    values = torch.ones(src.numel(), dtype=torch.float64, device=device)
    adjacency = torch.sparse_coo_tensor(
        torch.stack([src, dst]), values, (n, n)
    ).coalesce()

    degree = torch.zeros(n, dtype=torch.float64, device=device)
    degree.index_add_(0, adjacency.indices()[0], adjacency.values())
    inv_degree = torch.where(
        degree > 0, 1.0 / degree.clamp(min=1e-30), torch.zeros_like(degree)
    )
    transition = torch.sparse_coo_tensor(
        adjacency.indices(),
        adjacency.values() * inv_degree[adjacency.indices()[0]],
        (n, n),
    ).coalesce()

    walk = torch.eye(n, dtype=torch.float64, device=device)
    for k in range(steps):
        walk = torch.sparse.mm(transition, walk)
        out[:, k] = torch.diagonal(walk).to(torch.float32)
    return out


# =============================================================================
# Laplacian positional encoding
# =============================================================================
def laplacian_positional_encoding(
    edge_index: Tensor, num_nodes: int, *, dim: int = 16, tolerance: float = 1.0e-10
) -> Tensor:
    """Low-frequency normalized-Laplacian eigenpairs as ``[N, 2 * dim]`` features.

    For each of the first ``dim`` non-trivial eigenpairs of
    ``L_sym = I - D^-1/2 A D^-1/2``, adjacent columns hold the eigenvalue (broadcast to
    every row) and that row's L2-normalized eigenvector entry.  Disconnected-component
    constants (all eigenvalues ``<= tolerance``) are omitted; graphs with fewer modes are
    zero-padded to keep the model width fixed across a corpus.

    The sign of each eigenvector is fixed by making its largest-magnitude entry positive.
    This makes cached graph features reproducible.  It cannot remove rotations inside a
    repeated eigenspace; a learned consumer should therefore treat eigenpairs jointly and
    augment signs during training, as :class:`injection_hse.models.Graphormer` does when
    ``spectral_pe_dim > 0``.
    """
    if dim < 1:
        raise ValueError("laplacian_positional_encoding needs dim >= 1.")
    n = int(num_nodes)
    device = edge_index.device
    out = torch.zeros((n, 2 * dim), dtype=torch.float32, device=device)
    if n == 0 or edge_index.numel() == 0:
        return out

    adjacency = torch.zeros((n, n), dtype=torch.float64, device=device)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    adjacency[edge_index[1], edge_index[0]] = 1.0
    adjacency.fill_diagonal_(0.0)
    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.where(
        degree > 0.0, degree.rsqrt(), torch.zeros_like(degree)
    )
    laplacian = torch.diag((degree > 0.0).to(torch.float64)) - (
        inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    )
    values, vectors = torch.linalg.eigh(laplacian)
    keep = torch.nonzero(values > tolerance, as_tuple=False).flatten()[:dim]
    for column, mode in enumerate(keep.tolist()):
        vector = vectors[:, mode]
        pivot = int(torch.argmax(torch.abs(vector)))
        if float(vector[pivot]) < 0.0:
            vector = -vector
        out[:, 2 * column] = values[mode].to(torch.float32)
        out[:, 2 * column + 1] = vector.to(torch.float32)
    return out


# =============================================================================
# Effective impedance (electrical distance)
# =============================================================================
def effective_impedance_to_slack(
    grid: "Grid",
    *,
    index=None,
    edges: Optional["_PhaseEdges"] = None,
    normalize: bool = True,
    min_ohm: float = MIN_SERIES_OHM,
) -> Tensor:
    """Effective series impedance from every node-phase row to the slack, ``[N, 2]``.

    Column ``0`` is the effective RESISTANCE, column ``1`` the effective REACTANCE at the
    base frequency, each obtained from the corresponding weighted Laplacian of the
    in-service phase-resolved network (edge conductance ``1 / R`` resp. ``1 / X``). On a
    radial feeder this is exactly the series sum along the path to the slack; on a meshed or
    multi-terminal network it is the parallel combination the network actually presents,
    which is what a voltage drop responds to. Slack rows are ``0``; a row with no path to
    any slack is :data:`UNREACHABLE` (``-1``).

    Parameters
    ----------
    grid:
        The grid (its in-service branches define the network; open switches are excluded).
    index, edges:
        Optional pre-computed ``NodePhaseIndex`` / phase-edge table.
    normalize:
        Divide each column by the grid's own maximum finite value, so the feature is
        per-unit within the grid and comparable across voltage classes and grid sizes
        (the default and what the cross-grid encoding uses).
    min_ohm:
        Floor applied to a branch's ``|R|`` / ``|X|`` (a closed switch stores zero); see
        :data:`MIN_SERIES_OHM`.

    Returns
    -------
    torch.Tensor
        ``float32`` ``[N, 2]`` on CPU, row-aligned to ``node_phase_index(grid)``.

    Notes
    -----
    One dense Cholesky factorization + inverse of the slack-grounded Laplacian per column,
    ``O(N^3)`` time and ``O(N^2)`` memory in the node-phase row count; computed once per
    grid at graph-build time.
    """
    from .graph import _phase_edges

    idx = node_phase_index(grid) if index is None else index
    table = _phase_edges(grid, idx) if edges is None else edges
    n = int(idx.size)

    grounded: list[int] = []
    for node_id in slack_node_ids(grid):
        grounded.extend(int(r) for r in idx.rows(node_id))

    out = torch.full((n, 2), UNREACHABLE, dtype=torch.float32)
    if n == 0:
        return out
    for col, series in enumerate((table.resistance_ohm, table.reactance_ohm)):
        weights = [1.0 / max(abs(float(v)), min_ohm) for v in series]
        diag = _grounded_inverse_diagonal(
            table.src, table.dst, weights, n, grounded=grounded
        )
        if normalize:
            finite = diag[diag >= 0.0]
            peak = float(finite.max()) if finite.numel() else 0.0
            if peak > 0.0:
                diag = torch.where(diag >= 0.0, diag / peak, diag)
        out[:, col] = diag.to(torch.float32)
    return out


def pairwise_effective_resistance(
    grid: "Grid",
    *,
    quantity: str = "resistance",
    index=None,
    edges: Optional["_PhaseEdges"] = None,
    normalize: bool = True,
    min_ohm: float = MIN_SERIES_OHM,
) -> Tensor:
    """The full pairwise effective-resistance matrix ``[N, N]`` of one grid (``float32``).

    ``R_eff(i, j) = L+_ii + L+_jj - 2 L+_ij`` on the weighted Laplacian of the in-service
    phase-resolved network — symmetric, zero on the diagonal, and a metric. It is the
    ELECTRICAL analogue of the hop distance a topology-aware attention bias is usually built
    from: two rows a hop apart across a 1 m switch are electrically the same bus, two rows a
    hop apart across a 500 m cable are not, and only the effective resistance distinguishes
    them. Row pairs in different connected components are :data:`UNREACHABLE` (``-1``).

    Parameters
    ----------
    grid:
        The grid.
    quantity:
        ``"resistance"`` (default; edge weight ``1 / R``) or ``"reactance"`` (``1 / X`` at
        the base frequency).
    index, edges:
        Optional pre-computed ``NodePhaseIndex`` / phase-edge table.
    normalize:
        Divide by the grid's maximum finite entry (per-unit within the grid).
    min_ohm:
        Series impedance floor; see :data:`MIN_SERIES_OHM`.

    Returns
    -------
    torch.Tensor
        ``float32`` ``[N, N]`` on CPU, row/column-aligned to ``node_phase_index(grid)``.

    Notes
    -----
    ``O(N^3)`` time and ``O(N^2)`` memory (one dense inverse per connected component);
    intended as a per-grid, cached artifact, not a per-sample computation.
    """
    from .graph import _phase_edges

    idx = node_phase_index(grid) if index is None else index
    table = _phase_edges(grid, idx) if edges is None else edges
    n = int(idx.size)
    if quantity == "resistance":
        series = table.resistance_ohm
    elif quantity == "reactance":
        series = table.reactance_ohm
    else:
        raise ValueError(
            f"quantity must be 'resistance' or 'reactance', got {quantity!r}."
        )
    weights = [1.0 / max(abs(float(v)), min_ohm) for v in series]

    out = torch.full((n, n), UNREACHABLE, dtype=torch.float64)
    for rows in _components(table.src, table.dst, n):
        potential = _component_potential_matrix(table.src, table.dst, weights, rows)
        block = (
            torch.diagonal(potential).unsqueeze(1)
            + torch.diagonal(potential).unsqueeze(0)
            - 2.0 * potential
        )
        member = torch.tensor(rows, dtype=torch.int64)
        out[member.unsqueeze(1), member.unsqueeze(0)] = block
    out.fill_diagonal_(0.0)
    if normalize:
        finite = out[out >= 0.0]
        peak = float(finite.max()) if finite.numel() else 0.0
        if peak > 0.0:
            out = torch.where(out >= 0.0, out / peak, out)
    return out.to(torch.float32)


# =============================================================================
# Weighted-Laplacian internals (off the tape, float64)
# =============================================================================
def _undirected_edge_index(edges: "_PhaseEdges") -> Tensor:
    """``[2, E]`` int64 edge list of the undirected phase connections."""
    if not len(edges):
        return torch.zeros((2, 0), dtype=torch.int64)
    return torch.stack(
        [
            torch.tensor(edges.src, dtype=torch.int64),
            torch.tensor(edges.dst, dtype=torch.int64),
        ]
    )


def _adjacency_lists(src, dst, n: int) -> list[list[int]]:
    """Undirected adjacency lists over ``n`` rows."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(src, dst):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    return adjacency


def _components(src, dst, n: int) -> list[list[int]]:
    """Connected components of the phase-resolved network, as sorted row-index lists."""
    adjacency = _adjacency_lists(src, dst, n)
    seen = [False] * n
    out: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        members = [start]
        while queue:
            node = queue.popleft()
            for nb in adjacency[node]:
                if not seen[nb]:
                    seen[nb] = True
                    members.append(nb)
                    queue.append(nb)
        out.append(sorted(members))
    return out


def _laplacian(src, dst, weights, rows: list[int]) -> Tensor:
    """Dense weighted Laplacian ``[m, m]`` (float64) over the given row subset."""
    position = {row: i for i, row in enumerate(rows)}
    m = len(rows)
    laplacian = torch.zeros((m, m), dtype=torch.float64)
    for a, b, w in zip(src, dst, weights):
        i = position.get(int(a))
        j = position.get(int(b))
        if i is None or j is None or i == j:
            continue
        laplacian[i, i] += w
        laplacian[j, j] += w
        laplacian[i, j] -= w
        laplacian[j, i] -= w
    return laplacian


def _grounded_inverse_diagonal(
    src, dst, weights, n: int, *, grounded: list[int]
) -> Tensor:
    """Diagonal ``[N]`` (float64) of the inverse of the grounded weighted Laplacian.

    Grounding the slack rows turns the singular Laplacian into a symmetric positive-definite
    system whose inverse diagonal IS the effective resistance from each row to the grounded
    set (the standard one-port result). Grounded rows are ``0``; rows in a component holding
    no grounded row are :data:`UNREACHABLE`.
    """
    out = torch.full((n,), UNREACHABLE, dtype=torch.float64)
    ground = set(int(g) for g in grounded)
    for rows in _components(src, dst, n):
        free = [r for r in rows if r not in ground]
        if len(free) == len(rows):  # component holds no grounded row
            continue
        for row in rows:
            if row in ground:
                out[row] = 0.0
        if not free:
            continue
        laplacian = _laplacian(src, dst, weights, rows)
        position = {row: i for i, row in enumerate(rows)}
        keep = torch.tensor([position[r] for r in free], dtype=torch.int64)
        block = laplacian[keep.unsqueeze(1), keep.unsqueeze(0)]
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(block))
        out[torch.tensor(free, dtype=torch.int64)] = torch.diagonal(inverse)
    return out


def _component_potential_matrix(src, dst, weights, rows: list[int]) -> Tensor:
    """``[m, m]`` (float64) inverse-Laplacian potentials of one component, first row grounded.

    Grounding an arbitrary row of the component and padding its inverse with a zero
    row/column gives a matrix ``M`` for which ``M_ii + M_jj - 2 M_ij`` is the effective
    resistance between ``i`` and ``j`` — independent of which row was grounded.
    """
    m = len(rows)
    out = torch.zeros((m, m), dtype=torch.float64)
    if m < 2:
        return out
    laplacian = _laplacian(src, dst, weights, rows)
    block = laplacian[1:, 1:]
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(block))
    out[1:, 1:] = inverse
    return out


def _feeder_partition(grid: "Grid") -> "tuple[list[int], list[list[int]]]":
    """``(slack node ids, feeders)`` of an in-service grid.

    A FEEDER is a connected component of the branch graph with every slack node removed —
    the part of the network hanging off one substation departure. Used by
    :func:`injection_hse.data.central_placement` to spread a sensor budget over the feeders instead of
    clustering it in the best-connected one. Feeders are returned sorted by their lowest
    member id (deterministic), each member list sorted.
    """
    slack = [int(s) for s in slack_node_ids(grid)]
    slack_set = set(slack)
    adjacency = _feeder_adjacency(grid)

    feeders: list[list[int]] = []
    seen: set[int] = set(slack_set)
    for node in grid.nodes:
        start = int(node.id)
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        members = [start]
        while queue:
            current = queue.popleft()
            for nb in adjacency.get(current, ()):
                if nb not in seen and nb not in slack_set:
                    seen.add(nb)
                    members.append(nb)
                    queue.append(nb)
        feeders.append(sorted(members))
    feeders.sort(key=lambda members: members[0])
    return slack, feeders


def _hop_distances(members: list[int], adjacency: dict[int, list[int]]) -> dict:
    """All-pairs hop distances within one feeder (BFS per member; feeders are small)."""
    member_set = set(members)
    out: dict[int, dict[int, int]] = {}
    for start in members:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nb in adjacency.get(current, ()):
                if nb in member_set and nb not in dist:
                    dist[nb] = dist[current] + 1
                    queue.append(nb)
        out[start] = dist
    return out


def _eccentricities(members: list[int], adjacency: dict[int, list[int]]) -> dict:
    """``{node: eccentricity}`` (largest hop distance to another member) within a feeder."""
    distances = _hop_distances(members, adjacency)
    return {
        node: (max(dist.values()) if dist else 0) for node, dist in distances.items()
    }


def _feeder_adjacency(grid: "Grid") -> dict[int, list[int]]:
    """Undirected node adjacency over in-service branches (open switches dropped)."""
    from pgml.topology import branch_edges

    adjacency: dict[int, list[int]] = {int(n.id): [] for n in grid.nodes}
    for edge in branch_edges(grid):
        a, b = int(edge.a), int(edge.b)
        if a == b:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    return adjacency


def _select_central_nodes(
    members: list[int], adjacency: dict[int, list[int]], count: int
) -> list[int]:
    """``count`` spread-out central nodes of one feeder (min eccentricity, then k-center).

    The first pick is the feeder's center (minimum eccentricity — the node whose worst-case
    hop distance to the rest is smallest, i.e. the one that "sees" the whole feeder best);
    each further pick maximizes the minimum hop distance to the already chosen nodes, so a
    budget of several meters spreads over the feeder instead of clustering at its center.
    Ties break on the lower eccentricity, then the lower node id — deterministic.
    """
    if count <= 0 or not members:
        return []
    eccentricity = _eccentricities(members, adjacency)
    distances = _hop_distances(members, adjacency)
    ordered = sorted(members, key=lambda node: (eccentricity[node], node))
    chosen = [ordered[0]]
    unreachable = len(members) + 1
    while len(chosen) < min(count, len(members)):
        best = None
        best_key = None
        for node in ordered:
            if node in chosen:
                continue
            spread = min(distances[c].get(node, unreachable) for c in chosen)
            key = (-spread, eccentricity[node], node)
            if best_key is None or key < best_key:
                best, best_key = node, key
        if best is None:
            break
        chosen.append(best)
    return sorted(chosen)


__all__ = [
    "encoding_dim",
    "structural_node_encoding",
    "random_walk_encoding",
    "effective_impedance_to_slack",
    "pairwise_effective_resistance",
    "MIN_SERIES_OHM",
    "UNREACHABLE",
]
