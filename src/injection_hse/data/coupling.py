"""Mutual-admittance edges: the phase coupling the per-phase-index edge set leaves out.

:func:`injection_hse.data.build_graph` connects a branch's phase ``k`` at one terminal to phase ``k``
at the other. The physics does not factor that way. A distribution line's series impedance is
a full per-phase MATRIX — the conductors share a right-of-way, so each phase's current
induces a voltage in its neighbours — and its inverse, the series admittance
``Ys = (R(f) + j·2πf·L(f))^-1``, is dense: the primitive stamp ``[[Ys, -Ys], [-Ys, Ys]]``
therefore writes an entry at every (phase p at one terminal, phase q at either terminal)
position, not only at ``p == q``. A transformer adds a second mechanism: its nodal primitive
is ``Nᵀ Y_winding N`` with the vector-group incidence ``N``, and a DELTA winding's
``N = [[1,-1,0],[0,1,-1],[-1,0,1]]`` mixes the phases both within the delta-side node and
across the winding — the ±30° clock shift of a Dyn transformer IS that mixing.

The consequence for a message-passing estimator is structural, not marginal: on a
three-phase feeder whose only phase-mixing devices are the network branches, the
per-phase-index edge set splits the node-phase rows into one connected component per phase,
so a dependency between phase ``a`` at one bus and phase ``b`` at another is unreachable at
ANY number of layers, however deep the stack. This module supplies the missing edges.

Where the edge set comes from
-----------------------------
Not from a heuristic and not re-derived here: from the SAME primitive admittance blocks the
Y-bus assembly scatters (``pgml.assembly.branch_stamp_blocks``, the registry
``pgml.assembly.assemble_network_ybus`` walks). For every branch the series edge table
already drew, the block's structurally-nonzero OFF-DIAGONAL entries name pairs of node-phase
rows that ``Y(h)`` couples; the pairs the series table already carries are removed and the
rest become coupling edges. The union of the two sets is therefore exactly the off-diagonal
sparsity pattern of the passive network admittance, branch by branch, with each edge keeping
the provenance (which branch stamped it) that the switchable-topology layer needs.

Two families come out of that rule, and they are physically distinct:

- **same-node phase-to-phase** — the ``Ys[p, q]`` (and shunt ``Y_sh[p, q]``) entries a branch
  writes into ONE terminal's diagonal block: the mutual coupling as seen at a bus, plus a
  delta winding's intra-node mixing. Flagged separately in the edge features.
- **cross-node cross-phase** — the ``-Ys[p, q]`` entries between the two terminals: the
  mutual impedance along the conductor bundle, and the phase mixing across a Dyn winding.

Both are per BRANCH, so two branches meeting at a bus contribute two parallel coupling edges
between the same rows — the same multiplicity convention the series edges already use for
parallel branches, and the one that matches the physics (the admittances add).

Scope: the branches the series edge table already draws, so every coupling edge inherits a
drawable branch's provenance and switch state. Two passive stamps therefore fall outside it —
a DELTA ``ShuntAppliance`` (an appliance, not a branch) and a single-terminal ``ShuntReactor``
(no series edge to inherit from). Neither appears in the grids this path is used on; a grid
that carries one keeps their phase-to-phase entries unrepresented in the graph.

Everything in this module is detached topology metadata computed ONCE per grid (never per
sample, never on the autograd tape): python floats and CPU int lists, evaluated at the base
frequency. The pattern is frequency-independent — ``X = 2πfL`` scales every entry of the
impedance matrix, so an entry that is structurally absent at ``f0`` is absent at every
harmonic — which keeps the graph layout independent of the solved order set.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pgml.assembly import branch_stamp_blocks

if TYPE_CHECKING:
    from pgml.assembly import NodePhaseIndex
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import PhaseCouplingConfig

    from .graph import _PhaseEdges

#: Number of ``edge_attr`` columns appended when a graph carries coupling edges
#: (``is_coupling``, ``is_same_node``).
COUPLING_FEATURE_DIM = 2


@dataclass(frozen=True)
class _CouplingEdges:
    """The UNDIRECTED mutual-admittance connections of a grid (one entry per edge).

    Mirrors :class:`injection_hse.data.graph._PhaseEdges` field for field so :func:`injection_hse.data.build_graph`
    can append these entries to the series ones and emit both through one code path; the
    extra ``same_node`` flag separates the two coupling families. All lists are aligned and
    ordered by (branch, block position). Values are plain python floats (detached topology
    metadata).
    """

    src: list[int]
    dst: list[int]
    #: equivalent coupling impedance ``Re(-1 / Y[p, q])``, magnitude in ohms
    resistance_ohm: list[float]
    #: equivalent coupling impedance ``Im(-1 / Y[p, q])`` at ``f0``, magnitude in ohms
    reactance_ohm: list[float]
    length_m: list[float]
    is_line: list[bool]
    branch_id: list[int]
    #: whether the two rows belong to the SAME node (a bus-local coupling)
    same_node: list[bool]

    def __len__(self) -> int:
        return len(self.src)


def coupling_edges(
    grid: "Grid",
    *,
    index: "NodePhaseIndex",
    series: "_PhaseEdges",
    config: Optional["PhaseCouplingConfig"],
) -> Optional[_CouplingEdges]:
    """The phase-coupling connections a grid's branch stamps imply, beyond ``series``.

    Parameters
    ----------
    grid:
        The materialised grid the graph is built on.
    index:
        Its compact ``pgml.assembly.node_phase_index`` — the row layout the returned
        endpoints refer to.
    series:
        The phase-resolved SERIES edge table (:func:`injection_hse.data.graph._phase_edges`). It
        selects which branches are drawable (in service, switches closed) and which row pairs
        are already connected, so a coupling edge is never a duplicate of a series one.
    config:
        The :class:`~injection_hse.config.PhaseCouplingConfig` selecting the behaviour. ``None`` or
        ``enabled=False`` returns ``None`` (no coupling edges).

    Returns
    -------
    _CouplingEdges | None
        ``None`` when disabled or when the grid's stamps couple no further rows (a grid whose
        impedance matrices are diagonal and whose transformers are wye-wye); otherwise the
        undirected coupling connections with their equivalent impedance, branch provenance
        and same-node flag.
    """
    if config is None or not config.enabled:
        return None
    from injection_hse._switches import materialize_estimator_grid

    grid = materialize_estimator_grid(grid)
    branch_ids = list(dict.fromkeys(int(b) for b in series.branch_id))
    if not branch_ids:
        return None

    drawn: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for s, d, b in zip(series.src, series.dst, series.branch_id):
        drawn[int(b)].add((min(int(s), int(d)), max(int(s), int(d))))
    # every entry of a branch is aligned, so the last write wins with the same value
    branch_kind: dict[int, tuple[bool, float]] = {
        int(b): (bool(line), float(lm))
        for b, line, lm in zip(series.branch_id, series.is_line, series.length_m)
    }

    tolerance = float(config.tolerance)
    node_of = index.node_ids.tolist()
    blocks = branch_stamp_blocks(
        grid, [float(grid.base_frequency_hz)], branch_ids, index
    )

    src: list[int] = []
    dst: list[int] = []
    r_ohm: list[float] = []
    x_ohm: list[float] = []
    length: list[float] = []
    is_line_flags: list[bool] = []
    branch_of_edge: list[int] = []
    same_node: list[bool] = []

    for block in blocks:
        bid = int(block.branch_id)
        if bid not in branch_kind:  # not a drawable branch (defensive)
            continue
        y = block.block[0]  # [M, M] complex primitive at f0
        rows = [int(r) for r in block.rows.tolist()]
        magnitude = y.abs()
        diagonal = magnitude.diagonal().tolist()
        seen: dict[tuple[int, int], complex] = {}
        for a in range(len(rows)):
            for c in range(a + 1, len(rows)):
                row_a, row_c = rows[a], rows[c]
                if row_a == row_c:
                    continue
                pair = (min(row_a, row_c), max(row_a, row_c))
                if pair in drawn[bid]:
                    continue  # already a series edge of this branch
                scale = (diagonal[a] * diagonal[c]) ** 0.5
                value = complex(y[a, c])
                if scale <= 0.0 or abs(value) <= tolerance * scale:
                    continue  # structurally absent (or numerical round-off)
                previous = seen.get(pair)
                if previous is None or abs(value) > abs(previous):
                    seen[pair] = value
        line, length_m = branch_kind[bid]
        for (row_a, row_c), value in seen.items():
            impedance = -1.0 / value
            src.append(row_a)
            dst.append(row_c)
            r_ohm.append(abs(impedance.real))
            x_ohm.append(abs(impedance.imag))
            length.append(length_m)
            is_line_flags.append(line)
            branch_of_edge.append(bid)
            same_node.append(node_of[row_a] == node_of[row_c])

    if not src:
        return None
    return _CouplingEdges(
        src=src,
        dst=dst,
        resistance_ohm=r_ohm,
        reactance_ohm=x_ohm,
        length_m=length,
        is_line=is_line_flags,
        branch_id=branch_of_edge,
        same_node=same_node,
    )


__all__ = ["COUPLING_FEATURE_DIM", "coupling_edges"]
