"""Branch (edge) currents from the node-voltage state — the current-measurement input.

A measurement device at a bus meters, besides the bus voltage, the currents of its incident
branches (CT channels). The training datasets store only the solved node voltages, so the
per-edge currents the model consumes are DERIVED from the state: harmonic power flow is
linear per order, and every branch's terminal current is a fixed sparse linear map of the
node voltages (``I_term = Yprim @ V_term`` — the same primitive admittance block the Y-bus
stamp scatters, see :func:`pgml.assembly.branch_currents`).

:class:`BranchCurrentOperator` precomputes that map ONCE per (grid, orders) onto the model's
DIRECTED HALF-EDGE layout (:func:`injection_hse.data.build_graph`: the ``from→to`` half-edge carries
the FROM-terminal current, ``to→from`` the TO-terminal, adjacent pairs) and evaluates it per
step as a gather + einsum — differentiable in ``V`` (so it can also map a PREDICTED state to
predicted edge currents), device/dtype-honoring, vectorized over any leading batch/time axes.

Construction uses only the public ``pgml`` API: the per-edge weights are extracted from
:func:`pgml.assembly.branch_currents` responses to an identity voltage basis (chunked over
rows), which recovers each branch's primitive block exactly. Each half-edge's current depends
on at most the branch's own terminal rows, so the stored map is compact
(``cols [E, M]`` + ``weights [H, E, M]``, ``M`` ≤ 2 × phases).

Sign convention: positive current flows INTO the branch at the observing (source) terminal —
matching :class:`pgml.assembly.BranchCurrent` / ``BranchResult``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

from pgml.assembly import branch_currents, node_phase_index
from pgml.schemas.grid_schema import Switch

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid


def _half_edge_slots(
    grid: "Grid", idx, branch_states: Optional[dict] = None
) -> tuple[list[tuple[int, str, int]], Tensor]:
    """The model's directed half-edge list as ``(branch_id, terminal, phase_slot)`` triples.

    Mirrors the edge enumeration of :func:`injection_hse.data.build_graph` EXACTLY (in-service
    branches, open switches and self-loops dropped, per phase pair the ``from→to`` then
    ``to→from`` direction, adjacent), so half-edge ``e`` here is column ``e`` of the graph's
    ``edge_index``. Also returns the source node-phase row of every half-edge (``int64 [E]``,
    equal to ``edge_index[0]`` — asserted by the consumer against the actual graph).

    A branch listed in ``branch_states`` is KEPT whatever its own flags say (its state
    scales the admittance instead), matching the assembly's rule — that is what keeps the
    half-edge layout identical across the switch configurations of a topology set.
    """
    slots: list[tuple[int, str, int]] = []
    src_rows: list[int] = []
    for b in grid.branches:
        switched = branch_states is not None and int(b.id) in branch_states
        if not switched and not b.in_service:
            continue
        if not switched and isinstance(b, Switch) and not b.default_closed:
            continue
        if int(b.from_node) == int(b.to_node):
            continue
        for k, (fph, tph) in enumerate(zip(b.from_phases, b.to_phases)):
            try:
                fr = idx.row(int(b.from_node), fph)
                to = idx.row(int(b.to_node), tph)
            except KeyError:
                continue  # branch phase absent at a terminal (defensive, as in build_graph)
            slots.append((int(b.id), "from", k))
            slots.append((int(b.id), "to", k))
            src_rows.extend([fr, to])
    return slots, torch.tensor(src_rows, dtype=torch.int64)


class BranchCurrentOperator:
    """Sparse linear map: node voltages ``[*lead, H, N]`` -> half-edge currents ``[*lead, H, E]``.

    Parameters
    ----------
    grid:
        The (materialised) grid the state was solved on.
    orders:
        The solved harmonic orders (the state's ``H`` axis); frequencies are
        ``order * grid.base_frequency_hz``.
    dtype:
        Complex dtype of the STORED weights (the map is built at ``complex128`` internally
        and cast). The evaluation follows the INPUT voltage's dtype/device.
    chunk_size:
        Rows per identity-basis chunk during construction (memory guard for large grids).
    branch_states:
        Optional topology / switch-state mask ``{branch id: state}`` (see
        :mod:`injection_hse.data.topology`). A listed branch stays in the half-edge layout and has its
        admittance scaled by its state, so a variant with an open switch keeps the union
        layout and simply reads zero current there.

    Attributes
    ----------
    n_edges:
        ``E`` — the number of directed half-edges (== ``build_graph`` ``edge_index.shape[1]``).
    edge_src_rows:
        int64 ``[E]`` — each half-edge's source (observing-terminal) node-phase row; equal to
        the graph's ``edge_index[0]``, so an edge observability mask derives from a node mask
        by gathering at these rows.
    """

    def __init__(
        self,
        grid: "Grid",
        orders: Sequence[int],
        *,
        dtype: torch.dtype = torch.complex64,
        chunk_size: int = 256,
        branch_states: Optional[dict] = None,
    ) -> None:
        from injection_hse._switches import materialize_estimator_grid

        grid = materialize_estimator_grid(grid)
        self.orders = [int(o) for o in orders]
        idx = node_phase_index(grid)
        n = int(idx.size)
        f0 = float(grid.base_frequency_hz)
        freqs = [o * f0 for o in self.orders]
        h = len(freqs)

        slots, self.edge_src_rows = _half_edge_slots(grid, idx, branch_states)
        e = len(slots)
        self.n_edges = e

        # Extract each half-edge's response to every unit node voltage (chunked identity
        # basis). The response to e_r at (order, slot) is exactly the primitive-block entry
        # coupling row r into that terminal current, so nonzeros are confined to the branch's
        # own terminal rows (M ≤ 2 × phases) — collected sparsely below.
        cols_per_edge: list[list[int]] = [[] for _ in range(e)]
        weights_per_edge: list[list[Tensor]] = [[] for _ in range(e)]
        for start in range(0, n, int(chunk_size)):
            rows = torch.arange(start, min(start + int(chunk_size), n))
            basis = torch.zeros((rows.numel(), n), dtype=torch.complex128)
            basis[torch.arange(rows.numel()), rows] = 1.0
            # explicit [chunk, H, N] (expanded view, no copy) — a bare [chunk, N] basis
            # whose chunk count equals H would be misread as an H axis.
            basis = basis.unsqueeze(1).expand(-1, h, -1)
            by_id = {
                bc.branch_id: bc
                for bc in branch_currents(
                    grid,
                    basis,
                    freqs,
                    idx,
                    dtype=torch.complex128,
                    branch_states=branch_states,
                )
            }
            # responses on the half-edge axis: [chunk, H, E]
            resp = torch.stack(
                [
                    (by_id[bid].i_from if term == "from" else by_id[bid].i_to)[..., k]
                    for bid, term, k in slots
                ],
                dim=-1,
            )
            nz = resp.abs().amax(dim=1) > 0.0  # [chunk, E]
            for r_local, e_idx in torch.nonzero(nz, as_tuple=False).tolist():
                cols_per_edge[e_idx].append(int(rows[r_local]))
                weights_per_edge[e_idx].append(resp[r_local, :, e_idx])  # [H]

        m = max((len(c) for c in cols_per_edge), default=1) or 1
        cols = torch.zeros((e, m), dtype=torch.int64)
        weights = torch.zeros((h, e, m), dtype=torch.complex128)
        for e_idx in range(e):
            cnt = len(cols_per_edge[e_idx])
            if cnt:
                cols[e_idx, :cnt] = torch.tensor(
                    cols_per_edge[e_idx], dtype=torch.int64
                )
                weights[:, e_idx, :cnt] = torch.stack(
                    weights_per_edge[e_idx], dim=-1
                )  # [H, cnt]
        self._cols = cols  # int64 [E, M]
        self._weights = weights.to(dtype)  # [H, E, M]

    def to(self, device) -> "BranchCurrentOperator":
        """Move the stored map (weights / cols / src rows) to ``device``; returns self."""
        self._cols = self._cols.to(device)
        self._weights = self._weights.to(device)
        self.edge_src_rows = self.edge_src_rows.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._weights.device

    def __call__(self, v: Tensor) -> Tensor:
        """Half-edge currents ``[*lead, H, E]`` from the complex state ``[*lead, H, N]``.

        Differentiable in ``v`` (gather + einsum, nothing detached); any leading batch/time
        shape broadcasts. The weights follow ``v``'s device and, for a wider input dtype
        (``complex128`` state against ``complex64`` weights), are promoted so precision is
        governed by the input.
        """
        e, m = self._cols.shape
        cols = self._cols.to(v.device)
        w = self._weights.to(device=v.device)
        w = w.to(torch.promote_types(w.dtype, v.dtype))
        gathered = v.index_select(-1, cols.reshape(-1))  # [*lead, H, E*M]
        gathered = gathered.reshape(*v.shape[:-1], e, m)
        return torch.einsum("hem,...hem->...he", w, gathered)


def upstream_edge_mask(grid, edge_index: Tensor) -> Tensor:
    """Boolean ``[E]`` keeping, per node, only the half-edges TOWARD the slack.

    The one-CT-per-meter budget: of a metered location's incident half-edges, only the
    branch on the (BFS shortest) path toward the slack survives — one current channel
    per phase. The slack node(s) themselves keep every incident half-edge: the
    substation boundary meter reads its feeder heads. Nodes unreachable from a slack
    (de-energized islands) keep nothing.

    ``edge_index`` is the graph's directed half-edge layout ``[2, E]`` over node-phase
    ROWS (``build_graph`` / ``BranchCurrentOperator`` share it); the parent relation is
    computed on the NODE-level branch topology (in-service branches).
    """
    from collections import deque

    from pgml.topology import slack_node_ids

    adjacency: dict[int, set[int]] = {}
    for b in grid.branches:
        if not b.in_service:
            continue
        adjacency.setdefault(int(b.from_node), set()).add(int(b.to_node))
        adjacency.setdefault(int(b.to_node), set()).add(int(b.from_node))
    roots = [int(n) for n in slack_node_ids(grid)]
    parent: dict[int, int] = {r: r for r in roots}
    queue = deque(roots)
    while queue:
        u = queue.popleft()
        for w in adjacency.get(u, ()):
            if w not in parent:
                parent[w] = u
                queue.append(w)

    idx = node_phase_index(grid)
    node_of_row = idx.node_ids.to(torch.int64)
    src_nodes = node_of_row[edge_index[0].to(torch.int64)]
    dst_nodes = node_of_row[edge_index[1].to(torch.int64)]
    keep = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    root_set = set(roots)
    for e in range(int(edge_index.shape[1])):
        u = int(src_nodes[e])
        if u in root_set:
            keep[e] = True
        else:
            p = parent.get(u)
            keep[e] = p is not None and int(dst_nodes[e]) == p
    return keep


__all__ = ["BranchCurrentOperator", "upstream_edge_mask"]
