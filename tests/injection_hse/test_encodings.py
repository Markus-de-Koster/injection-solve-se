"""Structural node encodings: random walk, effective impedance, and the graph hook."""

from __future__ import annotations

import math

import pytest
import torch

from injection_hse.config import StructuralEncodingConfig
from injection_hse.data import (
    build_graph,
    central_placement,
    effective_impedance_to_slack,
    encoding_dim,
    laplacian_positional_encoding,
    pairwise_effective_resistance,
    random_walk_encoding,
)

F0 = 50.0


def _chain_grid(resistances, inductances, *, u_rated_v=400.0):
    """A single-phase radial chain ``0 - 1 - ... - n`` with node 0 the slack.

    Segment ``k`` carries ``resistances[k]`` ohm and ``inductances[k]`` henry TOTAL (the
    per-metre parameters are scaled by a 1 m length so the series values are exact).
    """
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    phases = (Phase.A,)
    n = len(resistances) + 1
    nodes = [Node(id=i, u_rated_v=u_rated_v, phases=phases) for i in range(n)]
    branches = [
        Line(
            id=100 + k,
            from_node=k,
            to_node=k + 1,
            from_phases=phases,
            to_phases=phases,
            length_m=1.0,
            series_resistance_ohm_per_m=[[resistances[k]]],
            series_inductance_h_per_m=[[inductances[k]]],
            shunt_capacitance_f_per_m=[[0.0]],
        )
        for k in range(len(resistances))
    ]
    source = Source(
        id=1,
        node=0,
        phases=phases,
        u_ref_v=(u_rated_v,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[1e-3]],
        inductance_h=[[1e-6]],
    )
    return Grid(
        base_frequency_hz=F0,
        nodes=nodes,
        branches=branches,
        appliances=[source],
    )


# =============================================================================
# Random-walk structural encoding
# =============================================================================
def test_rwse_on_a_path_graph_matches_hand_computation():
    # path 0 - 1 - 2: P = D^-1 A has no odd-step returns (bipartite) and
    # diag(P^2) = [1/2, 1, 1/2] (the middle node always returns in two steps).
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    rwse = random_walk_encoding(edge_index, 3, steps=4)

    assert rwse.shape == (3, 4)
    assert rwse.dtype == torch.float32
    assert torch.allclose(rwse[:, 0], torch.zeros(3))  # k=1: no self-loops
    assert torch.allclose(rwse[:, 1], torch.tensor([0.5, 1.0, 0.5]))
    assert torch.allclose(rwse[:, 2], torch.zeros(3))  # k=3: bipartite
    assert torch.allclose(rwse[:, 3], torch.tensor([0.5, 1.0, 0.5]))


def test_rwse_is_direction_agnostic_and_handles_isolated_rows():
    both = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    one_way = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    assert torch.allclose(
        random_walk_encoding(both, 4, steps=3),
        random_walk_encoding(one_way, 4, steps=3),
    )
    # row 3 has no incident edge: a walk never returns to it
    assert torch.all(random_walk_encoding(one_way, 4, steps=3)[3] == 0.0)


def test_rwse_separates_a_leaf_from_a_hub():
    # star 0 - {1, 2, 3}: the hub returns with probability 1 in two steps, a leaf with 1/3
    star = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int64)
    rwse = random_walk_encoding(star, 4, steps=2)
    assert rwse[0, 1] == pytest.approx(1.0)
    assert rwse[1, 1] == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_lappe_on_path_pairs_nontrivial_eigenvalues_and_vectors():
    # The normalized Laplacian of path 0-1-2 has spectrum {0, 1, 2}.
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    lappe = laplacian_positional_encoding(edge_index, 3, dim=3)

    assert lappe.shape == (3, 6)
    assert lappe.dtype == torch.float32
    assert torch.allclose(lappe[:, 0], torch.ones(3), atol=1e-6)
    assert torch.allclose(lappe[:, 2], 2.0 * torch.ones(3), atol=1e-6)
    assert torch.all(lappe[:, 4:] == 0.0)  # only two non-trivial modes
    # Eigenvectors are L2-normalized and use a deterministic sign convention.
    assert torch.linalg.vector_norm(lappe[:, 1]) == pytest.approx(1.0, rel=1e-6)
    assert torch.linalg.vector_norm(lappe[:, 3]) == pytest.approx(1.0, rel=1e-6)
    for column in (1, 3):
        vector = lappe[:, column]
        pivot = int(torch.argmax(torch.abs(vector)))
        assert float(vector[pivot]) >= 0.0


# =============================================================================
# Effective impedance (electrical distance)
# =============================================================================
def test_effective_impedance_on_a_radial_chain_is_the_series_sum():
    r = [0.4, 0.25, 0.1]
    ell = [1.0e-3, 2.0e-3, 0.5e-3]
    grid = _chain_grid(r, ell)

    feats = effective_impedance_to_slack(grid, normalize=False)
    assert feats.shape == (4, 2)

    expected_r = [0.0, r[0], r[0] + r[1], r[0] + r[1] + r[2]]
    x = [2.0 * math.pi * F0 * value for value in ell]
    expected_x = [0.0, x[0], x[0] + x[1], x[0] + x[1] + x[2]]
    assert feats[:, 0].tolist() == pytest.approx(expected_r, rel=1e-6)
    assert feats[:, 1].tolist() == pytest.approx(expected_x, rel=1e-6)


def test_effective_impedance_normalizes_per_grid():
    grid = _chain_grid([0.4, 0.25], [1.0e-3, 2.0e-3])
    feats = effective_impedance_to_slack(grid, normalize=True)
    # per-unit within the grid: the deepest row is 1.0, the slack 0.0
    assert float(feats[:, 0].max()) == pytest.approx(1.0)
    assert float(feats[0, 0]) == pytest.approx(0.0)
    assert float(feats[1, 0]) == pytest.approx(0.4 / 0.65, rel=1e-6)


def test_effective_impedance_marks_rows_without_a_path_to_the_slack():
    from pgml.schemas.grid_schema import Node, Phase

    from injection_hse.data.encodings import UNREACHABLE

    grid = _chain_grid([0.4], [1.0e-3])
    grid = grid.model_copy(
        update={"nodes": [*grid.nodes, Node(id=99, u_rated_v=400.0, phases=(Phase.A,))]}
    )
    feats = effective_impedance_to_slack(grid, normalize=True)
    assert float(feats[-1, 0]) == UNREACHABLE
    assert float(feats[-1, 1]) == UNREACHABLE


# =============================================================================
# Pairwise effective resistance (the electrical attention-bias source)
# =============================================================================
def test_pairwise_effective_resistance_is_a_metric_on_a_chain():
    r = [0.4, 0.25]
    grid = _chain_grid(r, [1.0e-3, 2.0e-3])

    matrix = pairwise_effective_resistance(grid, normalize=False)
    assert matrix.shape == (3, 3)
    assert matrix.dtype == torch.float32
    assert torch.allclose(matrix, matrix.T, atol=1e-6)
    assert torch.allclose(matrix.diagonal(), torch.zeros(3), atol=1e-9)
    assert float(matrix[0, 1]) == pytest.approx(r[0], rel=1e-5)
    assert float(matrix[1, 2]) == pytest.approx(r[1], rel=1e-5)
    assert float(matrix[0, 2]) == pytest.approx(r[0] + r[1], rel=1e-5)


def test_pairwise_effective_resistance_normalizes_and_flags_other_components():
    from pgml.schemas.grid_schema import Node, Phase

    from injection_hse.data.encodings import UNREACHABLE

    grid = _chain_grid([0.4, 0.25], [1.0e-3, 2.0e-3])
    grid = grid.model_copy(
        update={"nodes": [*grid.nodes, Node(id=99, u_rated_v=400.0, phases=(Phase.A,))]}
    )
    matrix = pairwise_effective_resistance(grid)
    assert float(matrix.max()) == pytest.approx(1.0)  # per-unit within the grid
    assert float(matrix[0, 3]) == UNREACHABLE  # separate component
    assert float(matrix[3, 3]) == 0.0  # a row is always zero distance from itself


def test_pairwise_effective_resistance_collapses_a_parallel_path():
    """A meshed pair of equal branches is electrically half of one — a hop count is blind."""
    from pgml.schemas.grid_schema import Line, Phase

    phases = (Phase.A,)
    grid = _chain_grid([0.4], [1.0e-3])
    parallel = Line(
        id=200,
        from_node=0,
        to_node=1,
        from_phases=phases,
        to_phases=phases,
        length_m=1.0,
        series_resistance_ohm_per_m=[[0.4]],
        series_inductance_h_per_m=[[1.0e-3]],
        shunt_capacitance_f_per_m=[[0.0]],
    )
    meshed = grid.model_copy(update={"branches": [*grid.branches, parallel]})
    assert float(
        pairwise_effective_resistance(meshed, normalize=False)[0, 1]
    ) == pytest.approx(0.2, rel=1e-5)


# =============================================================================
# The graph hook (config-driven feature layout)
# =============================================================================
def test_encoding_dim_matches_the_appended_columns(cigre_grid):
    from injection_hse.data.graph import BASE_NODE_FEATURE_DIM

    base = build_graph(cigre_grid)
    assert base.x.shape[1] == BASE_NODE_FEATURE_DIM
    assert base.node_encoding_dim == 0
    assert encoding_dim(None) == 0

    for config in (
        StructuralEncodingConfig(kind="rwse", rwse_steps=5),
        StructuralEncodingConfig(kind="lappe", lap_pe_dim=5),
        StructuralEncodingConfig(kind="electrical"),
    ):
        graph = build_graph(cigre_grid, encoding=config)
        width = encoding_dim(config)
        assert graph.node_encoding_dim == width
        assert graph.x.shape[1] == BASE_NODE_FEATURE_DIM + width
        # the base block is untouched by the appended columns
        assert torch.equal(graph.x[:, :BASE_NODE_FEATURE_DIM], base.x)


def test_normalized_base_rescales_the_slack_depth(cigre_grid):
    base = build_graph(cigre_grid)
    normalized = build_graph(
        cigre_grid, encoding=StructuralEncodingConfig(base="normalized")
    )
    depth_abs, depth_rel = base.x[:, 6], normalized.x[:, 6]

    # the absolute depth is in kilometres (grid-absolute); the normalized one tops out at 1
    assert float(depth_abs.max()) != pytest.approx(1.0)
    assert float(depth_rel.max()) == pytest.approx(1.0)
    assert float(depth_rel.min()) >= 0.0
    # every other base column is unchanged; only the depth is rescaled
    keep = [0, 1, 2, 3, 4, 5, 7]
    assert torch.equal(base.x[:, keep], normalized.x[:, keep])
    assert torch.allclose(depth_rel, depth_abs / float(depth_abs.max()), atol=1e-6)


def test_structural_encoding_is_grid_size_agnostic():
    """The same config yields the same feature WIDTH on grids of different size."""
    from pgml.grids import synthetic_feeder

    config = StructuralEncodingConfig(kind="rwse", rwse_steps=4, base="normalized")
    widths = {
        build_graph(synthetic_feeder(n, n_feeders=2), encoding=config).x.shape[1]
        for n in (4, 9, 15)
    }
    assert widths == {8 + 4}


# =============================================================================
# Central placement (the grid-agnostic sensor set)
# =============================================================================
def test_central_placement_covers_the_slack_and_every_feeder():
    from pgml.assembly import node_phase_index
    from pgml.grids import synthetic_feeder
    from pgml.topology import slack_node_ids

    grid = synthetic_feeder(13, n_feeders=3)
    idx = node_phase_index(grid)
    rows = central_placement(grid, 0.2)

    slack_rows = {r for s in slack_node_ids(grid) for r in idx.rows(s)}
    assert slack_rows <= set(rows.tolist())
    assert rows.dtype == torch.int64
    assert rows.tolist() == sorted(set(rows.tolist()))

    # every feeder contributes at least one metered node (the default floor)
    node_of_row = idx.node_ids.tolist()
    metered = {int(node_of_row[r]) for r in rows.tolist()}
    from injection_hse.data.encodings import _feeder_partition

    _, feeders = _feeder_partition(grid)
    assert len(feeders) == 3
    for members in feeders:
        assert metered & set(members)


def test_central_placement_prefers_central_nodes_over_leaves():
    from pgml.assembly import node_phase_index
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(7, n_feeders=1)  # one chain: 0 (slack) - 1 - ... - 6
    idx = node_phase_index(grid)
    rows = central_placement(grid, 0.05)  # budget of one node beyond the slack
    metered = {int(idx.node_ids[r]) for r in rows.tolist()} - {0}
    # the chain's center (node 3 of 1..6) sees the whole feeder best
    assert metered == {3}


def test_central_placement_scales_with_the_fraction():
    from pgml.grids import synthetic_feeder

    grid = synthetic_feeder(19, n_feeders=2)
    few = central_placement(grid, 0.1).numel()
    many = central_placement(grid, 0.5).numel()
    assert few < many
