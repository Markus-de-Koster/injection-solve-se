"""Switch branches in the estimated-flow layout: half-edges, branch ids, open variants.

A CLOSED :class:`~pgml.schemas.grid_schema.Switch` is an ordinary series branch — it has a
primitive admittance and carries a current — so it must appear in the model's directed
half-edge layout exactly like a line, and a deployment reporting estimated branch flows must
be able to name it. The invariants pinned here:

- a closed switch contributes half-edges (both directions, per phase) and a nonzero current;
- an open switch has NO half-edge in a plain grid, while a topology variant that opens it
  (``branch_states``) KEEPS the union layout and reads exactly zero there;
- a branch is identified by its ID, not by its terminal pair: a switch bypassing the line it
  parallels shares ``(from_node, to_node)`` with it, and both must stay distinguishable.
"""

from __future__ import annotations

import pytest
import torch

ORDERS = [1, 3, 5]


@pytest.fixture(scope="module")
def switch_grid():
    """A small feeder whose tie switch is CLOSED — the switch is a live series branch."""
    from injection_hse.data import TopologySet, switch_branch_ids
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(7, n_feeders=2, tie_switches=1)
    topology = TopologySet(
        switch_ids=tuple(switch_branch_ids(grid)),
        names=("cc",),
        closed=torch.ones((1, 1)),
    )
    return topology.union_grid(grid)


@pytest.fixture(scope="module")
def parallel_grid():
    """A feeder with a switch WIRED IN PARALLEL to one of its lines (a bypass)."""
    from pgml.grids import synthetic_feeder
    from pgml.schemas.grid_schema import Grid, Switch

    grid = synthetic_feeder(7, n_feeders=2)
    line = grid.branches[0]
    bypass = Switch(
        id=31000,
        from_node=line.from_node,
        to_node=line.to_node,
        from_phases=line.from_phases,
        to_phases=line.to_phases,
        default_closed=True,
        resistance_ohm=1.0e-3,
    )
    return Grid(
        base_frequency_hz=grid.base_frequency_hz,
        nodes=grid.nodes,
        branches=[*grid.branches, bypass],
        appliances=grid.appliances,
    )


def _switch_id(grid) -> int:
    from pgml.schemas.grid_schema import Switch

    return next(int(b.id) for b in grid.branches if isinstance(b, Switch))


def _state(grid, *, seed: int = 0) -> torch.Tensor:
    """A random complex state ``[H, N]`` (float64) on the grid's node-phase rows."""
    from pgml.assembly import node_phase_index

    n = node_phase_index(grid).size
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn((len(ORDERS), n), generator=gen, dtype=torch.float64)
    im = torch.randn((len(ORDERS), n), generator=gen, dtype=torch.float64)
    return torch.complex(re, im)


def test_estimator_grid_only_regularizes_exact_ideal_switches(switch_grid):
    """The full-row view is copied, localized, and keeps tensor resistance gradients."""
    from injection_hse._switches import ESTIMATOR_IDEAL_SWITCH_OHM, materialize_estimator_grid
    from pgml.schemas.grid_schema import Switch

    original = switch_grid.model_copy(deep=True)
    switches = [b for b in original.branches if isinstance(b, Switch)]
    ideal = switches[0].model_copy(
        update={
            "resistance_ohm": torch.tensor(
                0.0, dtype=torch.float64, requires_grad=True
            ),
            "inductance_h": 0.0,
        }
    )
    explicit = ideal.model_copy(update={"id": ideal.id + 1, "resistance_ohm": 2.5e-3})
    original = original.model_copy(
        update={"branches": [ideal, explicit, *original.branches[1:]]}
    )

    effective = materialize_estimator_grid(original)
    assert effective is not original
    assert torch.equal(
        original.branches[0].resistance_ohm.detach(),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert torch.equal(
        effective.branches[0].resistance_ohm.detach(),
        torch.tensor(ESTIMATOR_IDEAL_SWITCH_OHM, dtype=torch.float64),
    )
    assert effective.branches[1].resistance_ohm == explicit.resistance_ohm
    assert effective.branches[2:] == original.branches[2:]

    effective.branches[0].resistance_ohm.backward()
    assert original.branches[0].resistance_ohm.grad.item() == 1.0


# =============================================================================
# a closed switch is a series branch like any other
# =============================================================================
def test_closed_switch_contributes_half_edges(switch_grid):
    """The switch's terminals AND its branch id are in the estimated-flow layout."""
    from injection_hse.data import build_graph
    from injection_hse.data.currents import BranchCurrentOperator
    from pgml.assembly import node_phase_index

    sw_id = _switch_id(switch_grid)
    switch = next(b for b in switch_grid.branches if int(b.id) == sw_id)
    graph = build_graph(switch_grid, orders=ORDERS)
    idx = node_phase_index(switch_grid)

    on_switch = graph.edge_branch_ids == sw_id
    # both directions of each phase pair
    assert int(on_switch.sum()) == 2 * len(switch.from_phases)
    terminals = {
        (int(idx.node_ids[s]), int(idx.node_ids[d]))
        for s, d in zip(*graph.edge_index[:, on_switch].tolist())
    }
    assert terminals == {
        (int(switch.from_node), int(switch.to_node)),
        (int(switch.to_node), int(switch.from_node)),
    }

    operator = BranchCurrentOperator(switch_grid, ORDERS, dtype=torch.complex128)
    assert operator.n_edges == int(graph.edge_index.shape[1])
    i = operator(_state(switch_grid))  # [H, E]
    assert torch.all(i[:, on_switch].abs() > 0.0)


def test_open_switch_has_no_half_edge(switch_grid):
    """Opened in the GRID, the switch leaves the layout — there is no flow to estimate."""
    from injection_hse.data import build_graph

    sw_id = _switch_id(switch_grid)
    opened = switch_grid.model_copy(
        update={
            "branches": [
                b.model_copy(update={"default_closed": False})
                if int(b.id) == sw_id
                else b
                for b in switch_grid.branches
            ]
        }
    )
    graph = build_graph(opened, orders=ORDERS)
    assert int((graph.edge_branch_ids == sw_id).sum()) == 0


def test_open_topology_variant_keeps_the_layout_and_reads_zero(switch_grid):
    """A variant that opens the switch keeps its half-edges and estimates zero current.

    This is how a switchable-topology model reports an open switch: the union layout is
    shared by every configuration (one placement, one normalizer, one edge set), and the
    variant's ``branch_states`` scales the switch's admittance to zero.
    """
    from injection_hse.data import build_graph
    from injection_hse.data.currents import BranchCurrentOperator

    sw_id = _switch_id(switch_grid)
    graph = build_graph(switch_grid, orders=ORDERS)
    closed_op = BranchCurrentOperator(switch_grid, ORDERS, dtype=torch.complex128)
    open_op = BranchCurrentOperator(
        switch_grid, ORDERS, dtype=torch.complex128, branch_states={sw_id: 0.0}
    )

    assert open_op.n_edges == closed_op.n_edges
    assert torch.equal(open_op.edge_src_rows, closed_op.edge_src_rows)

    v = _state(switch_grid, seed=1)
    on_switch = graph.edge_branch_ids == sw_id
    i_open = open_op(v)
    assert torch.all(i_open[:, on_switch] == 0)
    # every other branch still carries what the closed-switch operator says for it
    torch.testing.assert_close(
        i_open[:, ~on_switch], closed_op(v)[:, ~on_switch], rtol=1e-10, atol=1e-9
    )


# =============================================================================
# a terminal pair does not identify a branch
# =============================================================================
def test_parallel_switch_and_line_stay_distinguishable(parallel_grid):
    """A switch bypassing a line shares its node pair; only the branch id separates them."""
    from injection_hse.data import build_graph
    from pgml.assembly import node_phase_index

    sw_id = _switch_id(parallel_grid)
    line_id = int(parallel_grid.branches[0].id)
    graph = build_graph(parallel_grid, orders=ORDERS)
    idx = node_phase_index(parallel_grid)

    def pairs(branch_id):
        sel = graph.edge_branch_ids == branch_id
        return {
            (int(idx.node_ids[s]), int(idx.node_ids[d]))
            for s, d in zip(*graph.edge_index[:, sel].tolist())
        }

    assert pairs(sw_id) == pairs(line_id) != set()  # same terminals, different branches
    assert int((graph.edge_branch_ids == sw_id).sum()) == int(
        (graph.edge_branch_ids == line_id).sum()
    )


# =============================================================================
# the map stays differentiable with a switch in it
# =============================================================================
def test_gradcheck_through_a_switched_layout(switch_grid):
    """float64 gradcheck of the state -> half-edge-current map (switch half-edges included)."""
    from injection_hse.data.currents import BranchCurrentOperator

    operator = BranchCurrentOperator(switch_grid, ORDERS, dtype=torch.complex128)
    v = _state(switch_grid, seed=2).requires_grad_(True)
    assert torch.autograd.gradcheck(operator, (v,), eps=1e-6, atol=1e-6)
