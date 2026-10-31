"""Switchable topology: one estimator across several switch configurations.

A grid whose switches can be operated is not one network but several. Everything the
estimator sees stays fixed across them — the node-phase row layout, the graph edge set, the
sensor placement, the metered half-edges — because the graph is built from the grid AS GIVEN,
with every switchable branch CLOSED (the union topology); a variant only opens some of them.
What differs per variant is the PHYSICS: the admittance the decode solves with, and the
branch currents the CTs read (an open switch carries none).

Two mechanisms express a variant, and each is used where it belongs:

- ``branch_states`` (``{branch id: 0.0 | 1.0}``) for everything built on the union layout —
  the assembly keeps a listed branch in the network and SCALES its admittance, so the branch
  list, and with it the half-edge layout, is preserved (:class:`~injection_hse.physics.NetworkSolver`,
  :class:`~injection_hse.data.currents.BranchCurrentOperator`);
- an open ``Switch.default_closed`` flag on a COPY of the grid for data generation, where the
  scenario runner takes a grid, not a state mask. Opening a switch never changes the row
  layout (rows come from nodes), so all variants' datasets stay row-compatible.

:class:`TopologySet` is the configuration (which switches, which variants);
:class:`TopologyBank` is the runtime bundle (per-variant solver / current operator plus the
model's topology channels). :func:`topology_groups` / :func:`merge_groups` split a batch by
variant and reassemble it, which is how a per-sample topology reaches per-variant operators
without a Python loop over samples.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Switch

from injection_hse._switches import materialize_estimator_grid

if TYPE_CHECKING:  # heavy / optional imports kept out of the import path
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import TopologyConfig


@dataclass(frozen=True)
class TopologySet:
    """The switch configurations one estimator is trained across.

    ``switch_ids`` fixes the switch ORDER (the channel order of every topology feature);
    ``closed`` is ``[K, S]`` — one row per variant, ``1.0`` where that switch is closed.
    """

    switch_ids: tuple[int, ...]
    names: tuple[str, ...]
    closed: Tensor  # [K, S] float32

    @classmethod
    def from_config(cls, cfg: "TopologyConfig") -> "TopologySet":
        """Build from the serialized :class:`~injection_hse.config.TopologyConfig`."""
        switch_ids = tuple(int(i) for i in cfg.switch_ids)
        closed = torch.ones((len(cfg.variants), len(switch_ids)), dtype=torch.float32)
        for k, variant in enumerate(cfg.variants):
            for sw in variant.open_switches:
                closed[k, switch_ids.index(int(sw))] = 0.0
        return cls(
            switch_ids=switch_ids,
            names=tuple(v.name for v in cfg.variants),
            closed=closed,
        )

    @property
    def n_variants(self) -> int:
        return len(self.names)

    @property
    def n_switches(self) -> int:
        return len(self.switch_ids)

    def branch_states(self, k: int) -> dict[int, float]:
        """Variant ``k`` as an assembly ``branch_states`` mask (layout-preserving)."""
        return {int(b): float(self.closed[k, s]) for s, b in enumerate(self.switch_ids)}

    def union_grid(self, grid: "Grid") -> "Grid":
        """A COPY of ``grid`` with EVERY switchable branch closed — the union topology.

        The layout every variant shares: it carries all branches, so the graph edge set, the
        half-edge layout and the metered CT channels are the same whichever switches a
        variant opens. Everything that DEFINES the model's layout (the graph, the
        branch-current operators, the network solvers) is built from this grid; the variants
        differ only by their ``branch_states``. A grid stored with a normally-open tie would
        otherwise drop that branch from the graph and the estimator could never see it close.
        """
        union = grid.model_copy(deep=True)
        ids = set(self.switch_ids)
        for b in union.branches:
            if isinstance(b, Switch) and int(b.id) in ids:
                b.default_closed = True
        return materialize_estimator_grid(union)

    def variant_grid(self, grid: "Grid", k: int) -> "Grid":
        """A COPY of ``grid`` with variant ``k``'s switch states written on its switches.

        For the scenario runner (data generation), which takes a grid rather than a
        branch-state mask. The node-phase row layout is unaffected, so every variant's
        dataset stays row-compatible with the others and with the union graph.
        """
        states = self.branch_states(k)
        variant = grid.model_copy(deep=True)
        for b in variant.branches:
            if isinstance(b, Switch) and int(b.id) in states:
                b.default_closed = bool(states[int(b.id)])
        return materialize_estimator_grid(variant)

    def edge_state(self, edge_branch_ids: Tensor) -> Tensor:
        """Per-variant half-edge conduction state ``[K, E]`` (``1.0`` = conducting).

        ``edge_branch_ids`` is the graph's per-half-edge branch id
        (:func:`~injection_hse.data.graph.build_graph`). Every non-switchable branch is 1.0 in every
        variant; a switchable one carries that variant's state — the channel the model
        conditions on.
        """
        state = torch.ones(
            (self.n_variants, int(edge_branch_ids.numel())), dtype=torch.float32
        )
        for s, branch in enumerate(self.switch_ids):
            hit = edge_branch_ids == int(branch)
            state[:, hit] = self.closed[:, s : s + 1]
        return state

    def dataset_path(self, root, k: int) -> Path:
        """Variant ``k``'s dataset directory under the run's dataset root."""
        return Path(root) / self.names[k]

    def describe(self, grid: Optional["Grid"] = None) -> str:
        """One human-readable line per variant (for run logs)."""
        labels = (
            {int(b.id): (b.name or f"branch {b.id}") for b in grid.branches}
            if grid
            else {}
        )
        lines = []
        for k, name in enumerate(self.names):
            states = ", ".join(
                f"{labels.get(b, f'branch {b}')}: {'closed' if self.closed[k, s] else 'open'}"
                for s, b in enumerate(self.switch_ids)
            )
            lines.append(f"{name} ({states})")
        return "; ".join(lines)


def enumerate_variants(
    grid: "Grid", switch_ids: Sequence[int]
) -> list[tuple[str, list[int]]]:
    """Every switch combination that leaves the grid connected: ``(name, open ids)``.

    Combinations that de-energize part of the grid are dropped — their harmonic system is
    singular, so they are not a topology an estimator can be trained on. The name spells the
    states in ``switch_ids`` order (``c`` closed, ``o`` open), e.g. ``"co"`` = first switch
    closed, second open.
    """
    from pgml.topology import connectivity_report

    ids = [int(i) for i in switch_ids]
    variants: list[tuple[str, list[int]]] = []
    for states in itertools.product([True, False], repeat=len(ids)):
        name = "".join("c" if s else "o" for s in states)
        open_ids = [i for i, s in zip(ids, states) if not s]
        probe = grid.model_copy(deep=True)
        for b in probe.branches:
            if isinstance(b, Switch) and int(b.id) in ids:
                b.default_closed = int(b.id) not in open_ids
        if connectivity_report(probe).connected:
            variants.append((name, open_ids))
    return variants


def switch_branch_ids(grid: "Grid") -> list[int]:
    """Ids of every in-service :class:`~pgml.schemas.grid_schema.Switch` in the grid."""
    return [int(b.id) for b in grid.branches if isinstance(b, Switch) and b.in_service]


# =============================================================================
# runtime bundle
# =============================================================================
@dataclass
class TopologyBank:
    """Per-variant physics operators + the model's per-variant topology channels.

    Built once per run (:meth:`build`) and held by
    :class:`~injection_hse.train.SEModule`, which selects each sample's entries by the batch's
    ``topology`` index. ``solvers`` is empty for the direct-voltage decode and ``edge_ops``
    empty when branch currents are not metered.
    """

    topology: TopologySet
    edge_state: Tensor  # [K, E] per-variant half-edge conduction state
    solvers: tuple = ()
    edge_ops: tuple = ()

    @classmethod
    def build(
        cls,
        topology: TopologySet,
        grid: "Grid",
        orders: Sequence[int],
        *,
        edge_branch_ids: Tensor,
        dtype: torch.dtype = torch.complex64,
        measure_current: bool = False,
        decode: str = "voltage",
        build_solvers: Optional[bool] = None,
        anchor_v_weight: float = 0.0,
        anchor_i_weight: float = 0.0,
        harmonic_slack: str = "norton",
        load_shunt: Optional[str] = None,
    ) -> "TopologyBank":
        """Build every variant's operators from ONE grid (the union topology).

        The variants differ only by their ``branch_states``, so all operators share the
        union half-edge layout and the row layout — which is what lets one model, one
        placement and one normalizer serve every topology. ``grid`` is normalised to the
        union topology (:meth:`TopologySet.union_grid`) first, so a grid stored with a
        normally-open switch builds the same operators as one stored with it closed.

        ``build_solvers`` says whether the per-variant network solvers are built: ``None``
        (default) builds them for ``decode="injection"`` only; ``True`` builds them whatever
        the decode — the physics-derived input priors
        (:attr:`injection_hse.config.PhysicsConfig.input_priors`, ``PhysicsConfig.needs_solver``)
        need each variant's operators under the voltage decode too. ``harmonic_slack`` and
        ``load_shunt`` are handed to every variant's :class:`~injection_hse.physics.NetworkSolver`.
        """
        from injection_hse.data.currents import BranchCurrentOperator
        from injection_hse.physics import NetworkSolver

        grid = topology.union_grid(grid)  # the layout every variant shares
        edge_ops: list = []
        if measure_current:
            edge_ops = [
                BranchCurrentOperator(
                    grid,
                    list(orders),
                    dtype=dtype,
                    branch_states=topology.branch_states(k),
                )
                for k in range(topology.n_variants)
            ]
        if build_solvers is None:
            build_solvers = decode == "injection"
        solvers: list = []
        if build_solvers:
            solvers = [
                NetworkSolver.from_grid(
                    grid,
                    list(orders),
                    dtype=dtype,
                    anchor_v_weight=anchor_v_weight,
                    anchor_i_weight=anchor_i_weight,
                    edge_op=edge_ops[k] if edge_ops else None,
                    branch_states=topology.branch_states(k),
                    harmonic_slack=harmonic_slack,
                    load_shunt=load_shunt,
                )
                for k in range(topology.n_variants)
            ]
        return cls(
            topology=topology,
            edge_state=topology.edge_state(edge_branch_ids),
            solvers=tuple(solvers),
            edge_ops=tuple(edge_ops),
        )

    @property
    def n_variants(self) -> int:
        return self.topology.n_variants

    def to(self, device) -> "TopologyBank":
        """Move the operators + the edge channel onto ``device``; returns self.

        The switch-state matrix stays where it is — it is read once per batch and moved with
        the batch (:meth:`injection_hse.train.SEModule._build_node_topology`).
        """
        self.edge_state = self.edge_state.to(device)
        for op in self.edge_ops:
            op.to(device)
        for solver in self.solvers:
            solver.to(device)
        return self


# =============================================================================
# batch dispatch
# =============================================================================
def topology_groups(topology: Tensor, n_variants: int) -> list[tuple[int, Tensor]]:
    """Split a per-sample variant index ``[B]`` into ``(variant, sample indices)`` groups.

    Empty variants are omitted, so a batch drawn from one topology dispatches to exactly
    one operator. The returned index tensors partition ``range(B)``.
    """
    flat = topology.reshape(-1).to(torch.int64)
    groups: list[tuple[int, Tensor]] = []
    for k in range(int(n_variants)):
        sel = torch.nonzero(flat == k, as_tuple=False).flatten()
        if sel.numel():
            groups.append((k, sel))
    return groups


def merge_groups(
    parts: Sequence[Tensor], groups: Sequence[tuple[int, Tensor]]
) -> Tensor:
    """Reassemble per-group results into the original sample order (differentiable).

    ``parts[i]`` is the result for ``groups[i]``'s samples, leading dim first. The parts are
    concatenated and permuted back with an index gather — no in-place write on a tracked
    tensor, so gradients flow to every group.
    """
    if len(parts) == 1:
        return parts[0]
    order = torch.cat([sel for _, sel in groups])
    merged = torch.cat(list(parts), dim=0)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return merged.index_select(0, inverse)


__all__ = [
    "TopologySet",
    "TopologyBank",
    "enumerate_variants",
    "switch_branch_ids",
    "topology_groups",
    "merge_groups",
]
