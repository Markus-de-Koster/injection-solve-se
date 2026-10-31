"""Switchable topology: per-variant physics, batch dispatch, and model conditioning.

One estimator trained across several switch configurations. The invariants under test:

- every configuration shares the union layout (rows, half-edges), so one placement and one
  normalizer serve all of them;
- a configuration's operators reproduce the network that configuration actually IS (the
  ``branch_states`` path == solving a grid whose switch is open);
- a mixed batch decodes each sample with ITS configuration, in the original sample order;
- the model is told which configuration a sample came from;
- a run WITHOUT a topology is untouched.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
    PhysicsConfig,
    TopologyConfig,
    TopologyVariantConfig,
)
from injection_hse.data import (
    TopologyBank,
    TopologySet,
    build_graph,
    enumerate_variants,
    merge_groups,
    switch_branch_ids,
    topology_groups,
)

ORDERS = [1, 3, 5]


@pytest.fixture(scope="module")
def switched_grid():
    """A small radial feeder with two normally-open tie switches."""
    from pgml.grids import synthetic_feeder

    return synthetic_feeder(13, n_feeders=3, tie_switches=2)


@pytest.fixture(scope="module")
def topology(switched_grid):
    """Every connected switch combination of that grid (4 for two independent ties)."""
    ids = switch_branch_ids(switched_grid)
    cfg = TopologyConfig(
        switch_ids=ids,
        variants=[
            TopologyVariantConfig(name=name, open_switches=open_ids)
            for name, open_ids in enumerate_variants(switched_grid, ids)
        ],
    )
    return TopologySet.from_config(cfg)


# =============================================================================
# the configuration set
# =============================================================================
def test_variants_enumerate_states(switched_grid, topology):
    ids = switch_branch_ids(switched_grid)
    assert len(ids) == 2
    # tie switches only ADD paths to a radial feeder, so all four combinations stay connected
    assert set(topology.names) == {"cc", "co", "oc", "oo"}
    assert topology.branch_states(topology.names.index("co")) == {
        ids[0]: 1.0,
        ids[1]: 0.0,
    }
    variant = topology.variant_grid(switched_grid, topology.names.index("co"))
    closed = {
        int(b.id): b.default_closed
        for b in variant.branches
        if hasattr(b, "default_closed")
    }
    assert closed == {ids[0]: True, ids[1]: False}


def test_union_layout_is_shared(switched_grid, topology):
    """Every variant keeps the union row/half-edge layout — the model input never changes."""
    graph = build_graph(topology.union_grid(switched_grid))
    bank = TopologyBank.build(
        topology,
        switched_grid,
        ORDERS,
        edge_branch_ids=graph.edge_branch_ids,
        dtype=torch.complex128,
        measure_current=True,
        decode="injection",
    )
    for op in bank.edge_ops:
        assert op.n_edges == int(graph.edge_index.shape[1])
        assert torch.equal(op.edge_src_rows, graph.edge_index[0])
    for solver in bank.solvers:
        assert solver.y_harmonic.shape[-1] == graph.num_nodes
    # the conduction channel is 1 everywhere except the open switches' own half-edges
    state = topology.edge_state(graph.edge_branch_ids)
    assert state.shape == (topology.n_variants, int(graph.edge_index.shape[1]))
    all_closed = topology.names.index("cc")
    assert bool(state[all_closed].all())
    one_open = topology.names.index("co")
    open_ids = graph.edge_branch_ids[state[one_open] == 0].unique().tolist()
    assert open_ids == [topology.switch_ids[1]]


def test_variant_physics_matches_the_open_switch_grid(switched_grid, topology):
    """``branch_states`` operators == solving the grid with that switch actually open."""
    from pgml.solver import solve_harmonic_flow

    graph = build_graph(topology.union_grid(switched_grid))
    bank = TopologyBank.build(
        topology,
        switched_grid,
        ORDERS,
        edge_branch_ids=graph.edge_branch_ids,
        dtype=torch.complex128,
        measure_current=True,
        decode="injection",
    )
    state = topology.edge_state(graph.edge_branch_ids)
    for k, name in enumerate(topology.names):
        v = solve_harmonic_flow(
            topology.variant_grid(switched_grid, k), ORDERS, dtype=torch.complex128
        ).v  # [H, N]
        solver = bank.solvers[k]
        i_true = solver.injection_targets(v.unsqueeze(0))
        v_slack = v[solver.fundamental_pos].index_select(-1, solver.slack_rows)
        v_solved = solver.solve(i_true, v_slack.unsqueeze(0))[0]
        assert torch.allclose(v_solved, v, rtol=1e-8, atol=1e-8), name
        # an open switch carries no current
        i_edges = bank.edge_ops[k](v)
        assert torch.count_nonzero(i_edges[:, state[k] == 0]) == 0, name


# =============================================================================
# batch dispatch
# =============================================================================
def test_groups_partition_and_merge_restores_order():
    topo = torch.tensor([2, 0, 1, 0, 2, 2])
    groups = topology_groups(topo, 3)
    assert [k for k, _ in groups] == [0, 1, 2]
    assert torch.equal(
        torch.cat([sel for _, sel in groups]).sort().values, torch.arange(6)
    )
    values = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    parts = [values.index_select(0, sel) for _, sel in groups]
    assert torch.equal(merge_groups(parts, groups), values)


def test_empty_variants_are_skipped():
    groups = topology_groups(torch.tensor([1, 1, 1]), 3)
    assert [k for k, _ in groups] == [1]


def test_merge_groups_is_differentiable():
    topo = torch.tensor([1, 0, 1])
    groups = topology_groups(topo, 2)
    x = torch.arange(3, dtype=torch.float64, requires_grad=True)
    merged = merge_groups(
        [x.index_select(0, sel) * (k + 1) for k, sel in groups], groups
    )
    merged.sum().backward()
    assert torch.equal(x.grad, torch.tensor([2.0, 1.0, 2.0], dtype=torch.float64))


# =============================================================================
# the training module
# =============================================================================
def _cfg(topology_cfg, *, conditioning="edge_and_global", decode="injection"):
    return ExperimentConfig(
        name="topology",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path="unused",
            topology=(
                None
                if topology_cfg is None
                else topology_cfg.model_copy(update={"conditioning": conditioning})
            ),
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2, conv="gine"),
        physics=PhysicsConfig(decode=decode),
        measurement=MeasurementConfig(voltage_nodes=[1], measure_current=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="full", kind="full", target_loss=1.0, max_epochs=1)
            ]
        ),
    )


def _module(switched_grid, topology, *, conditioning="edge_and_global"):
    """An SEModule wired to the topology bank, with normalizers fit on solved states."""
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import base_voltage_per_row
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import fit_injection_normalizer
    from injection_hse.train import SEModule

    graph = build_graph(topology.union_grid(switched_grid))
    bank = TopologyBank.build(
        topology,
        switched_grid,
        ORDERS,
        edge_branch_ids=graph.edge_branch_ids,
        dtype=torch.complex64,
        measure_current=True,
        decode="injection",
    )
    states = torch.stack(
        [
            solve_harmonic_flow(
                topology.variant_grid(switched_grid, k), ORDERS, dtype=torch.complex64
            ).v
            for k in range(topology.n_variants)
        ]
    )  # [K, H, N]
    encoding = build_encoding("trig")
    normalizer = HarmonicNormalizer().fit(encoding.encode(states).transpose(-3, -2))
    i_edges = bank.edge_ops[0](states)
    edge_normalizer = HarmonicNormalizer().fit(
        encoding.encode(i_edges).transpose(-3, -2)
    )
    cfg = _cfg(
        TopologyConfig(
            switch_ids=list(topology.switch_ids),
            variants=[
                TopologyVariantConfig(
                    name=name,
                    open_switches=[
                        int(b)
                        for s, b in enumerate(topology.switch_ids)
                        if topology.closed[k, s] == 0
                    ],
                )
                for k, name in enumerate(topology.names)
            ],
        ),
        conditioning=conditioning,
    )
    module = SEModule(
        cfg,
        n_nodes=int(states.shape[-1]),
        normalizer=normalizer,
        graph=graph,
        base_v=base_voltage_per_row(switched_grid),
        inj_normalizer=fit_injection_normalizer(bank.solvers[0], states, encoding),
        grid=switched_grid,
        edge_normalizer=edge_normalizer,
        topology=bank,
    )
    return module, states


def test_module_decodes_each_sample_with_its_own_configuration(switched_grid, topology):
    """A mixed batch equals decoding each configuration's samples on their own."""
    module, states = _module(switched_grid, topology)
    module.eval()
    order = torch.tensor([2, 0, 3, 1, 0])
    batch = {"v": states.index_select(0, order), "topology": order}
    mixed = module.infer(batch)["pred_v"]
    for i, k in enumerate(order.tolist()):
        single = module.infer({"v": states[k : k + 1], "topology": torch.tensor([k])})[
            "pred_v"
        ]
        assert torch.allclose(mixed[i], single[0], atol=1e-6)


def test_topology_channels_reach_the_model(switched_grid, topology):
    """The switch states enter the model input as node + edge channels."""
    module, states = _module(switched_grid, topology)
    lead, device = (topology.n_variants,), states.device
    topo = torch.arange(topology.n_variants)
    node_ch = module._build_node_topology(topo, lead, device)
    edge_ch = module._build_edge_topology(topo, lead, device)
    assert node_ch.shape == (topology.n_variants, topology.n_switches)
    assert torch.equal(node_ch, topology.closed.to(node_ch.dtype))
    assert edge_ch.shape[0] == topology.n_variants and edge_ch.shape[-1] == 1
    assert module._n_topology_features == topology.n_switches
    assert module._n_edge_topology_features == 1


@pytest.mark.parametrize(
    "conditioning,n_node,n_edge",
    [("edge_and_global", 2, 1), ("edge", 0, 1), ("global", 2, 0), ("none", 0, 0)],
)
def test_conditioning_selects_the_channels(
    switched_grid, topology, conditioning, n_node, n_edge
):
    module, _ = _module(switched_grid, topology, conditioning=conditioning)
    assert module._n_topology_features == n_node
    assert module._n_edge_topology_features == n_edge


def test_prediction_depends_on_the_configuration(switched_grid, topology):
    """The SAME state decoded as two different configurations gives different estimates."""
    module, states = _module(switched_grid, topology)
    module.eval()
    v = states[topology.names.index("cc")].unsqueeze(0)
    a = module.infer({"v": v, "topology": torch.tensor([topology.names.index("cc")])})
    b = module.infer({"v": v, "topology": torch.tensor([topology.names.index("oo")])})
    assert not torch.allclose(a["pred_v"], b["pred_v"], atol=1e-6)


def test_without_topology_the_module_is_unchanged(switched_grid):
    """No topology config -> no channels, no bank, the single-solver path."""
    from injection_hse.train import SEModule

    graph = build_graph(
        switched_grid
    )  # open ties dropped: the plain single-topology path
    module = SEModule(
        _cfg(None, decode="voltage"),
        n_nodes=graph.num_nodes,
        normalizer=_identity_normalizer(),
        graph=graph,
        grid=switched_grid,
        edge_op=_edge_op(switched_grid),
        edge_normalizer=_identity_normalizer(),
    )
    assert module._topology is None
    assert module._n_topology_features == 0
    assert module._n_edge_topology_features == 0
    assert (
        module._topology_index({"v": torch.zeros(2)}, (2,), torch.device("cpu")) is None
    )


def _identity_normalizer():
    from injection_hse.normalization import HarmonicNormalizer

    return HarmonicNormalizer().fit(torch.ones(4, 6, len(ORDERS), 3))


def _edge_op(grid):
    from injection_hse.data import BranchCurrentOperator

    return BranchCurrentOperator(grid, ORDERS, dtype=torch.complex64)
