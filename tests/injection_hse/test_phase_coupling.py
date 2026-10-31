"""The phase-coupling graph variant: byte-identity off, the physics it adds on.

The default (series-only) edge set connects phase ``k`` to phase ``k``, so a three-phase
feeder's node-phase rows fall into one component per phase. These tests pin the default
byte-for-byte, pin the added edge set to the off-diagonal sparsity pattern of the passive
network admittance (the physics it is supposed to mirror), and check that the widened edge
features reach the models and the multi-grid collate.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import PhaseCouplingConfig
from injection_hse.data import build_graph, edge_feature_dim
from injection_hse.data.coupling import COUPLING_FEATURE_DIM
from injection_hse.data.graph import EDGE_FEATURE_DIM

ORDERS = [1, 3, 5, 7, 9, 11, 13]

ON = PhaseCouplingConfig(enabled=True)


def _pyg_conv_available() -> bool:
    try:
        from torch_geometric.nn import SAGEConv

        SAGEConv(4, 4)
        return True
    except Exception:
        return False


pyg_required = pytest.mark.skipif(
    not _pyg_conv_available(),
    reason="torch-geometric conv construction unavailable (PyG 2.6.1 + Python 3.14).",
)


def _components(edge_index, n: int) -> list[set[int]]:
    """Connected components of an undirected edge list (no networkx dependency)."""
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[int] = set()
    out: list[set[int]] = []
    for start in range(n):
        if start in seen:
            continue
        stack, group = [start], set()
        seen.add(start)
        while stack:
            node = stack.pop()
            group.add(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(group)
    return out


def _undirected_pairs(edge_index) -> set[tuple[int, int]]:
    return {
        (min(a, b), max(a, b))
        for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist())
    }


# -- the default stays exactly what every existing checkpoint was trained on ---
def test_disabled_is_byte_identical(cigre_grid):
    plain = build_graph(cigre_grid, orders=ORDERS)
    for config in (None, PhaseCouplingConfig(), PhaseCouplingConfig(tolerance=1e-3)):
        other = build_graph(cigre_grid, orders=ORDERS, coupling=config)
        assert torch.equal(other.edge_index, plain.edge_index)
        assert torch.equal(other.edge_attr, plain.edge_attr)
        assert torch.equal(other.edge_branch_ids, plain.edge_branch_ids)
        assert torch.equal(other.x, plain.x)
    assert plain.edge_attr.shape[1] == EDGE_FEATURE_DIM


def test_the_coupling_graph_is_deterministic(cigre_grid):
    first = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    second = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_attr, second.edge_attr)
    assert torch.equal(first.edge_branch_ids, second.edge_branch_ids)


def test_edge_feature_dim_follows_the_config():
    assert edge_feature_dim() == EDGE_FEATURE_DIM
    assert edge_feature_dim(PhaseCouplingConfig()) == EDGE_FEATURE_DIM
    assert edge_feature_dim(ON) == EDGE_FEATURE_DIM + COUPLING_FEATURE_DIM


def test_single_phase_grid_keeps_its_edge_set():
    """A 1x1 impedance matrix has no off-diagonal: enabling it adds edges but no width lie."""
    from tests.injection_hse.tiny_grids import single_phase_chain

    grid = single_phase_chain()
    plain = build_graph(grid)
    coupled = build_graph(grid, coupling=ON)
    # nothing to couple on a single-phase chain — but the WIDTH follows the config, so a
    # corpus mixing such a grid with a three-phase one still has one edge-feature layout
    assert torch.equal(coupled.edge_index, plain.edge_index)
    assert coupled.edge_attr.shape == (
        plain.edge_attr.shape[0],
        EDGE_FEATURE_DIM + COUPLING_FEATURE_DIM,
    )
    assert float(coupled.edge_attr[:, EDGE_FEATURE_DIM:].abs().sum()) == 0.0


# -- what the variant adds -----------------------------------------------------
def test_series_prefix_is_preserved(cigre_grid):
    plain = build_graph(cigre_grid, orders=ORDERS)
    coupled = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    e = plain.edge_index.shape[1]
    assert coupled.edge_index.shape[1] > e
    assert torch.equal(coupled.edge_index[:, :e], plain.edge_index)
    assert torch.equal(coupled.edge_attr[:e, :EDGE_FEATURE_DIM], plain.edge_attr)
    assert torch.equal(coupled.edge_branch_ids[:e], plain.edge_branch_ids)
    # node features are untouched: the degree column stays the BRANCH degree
    assert torch.equal(coupled.x, plain.x)
    # the flag columns separate the two edge families
    assert float(coupled.edge_attr[:e, EDGE_FEATURE_DIM:].abs().sum()) == 0.0
    assert bool((coupled.edge_attr[e:, EDGE_FEATURE_DIM] == 1.0).all())
    assert torch.isfinite(coupled.edge_attr).all()


def test_half_edge_pair_convention_holds(cigre_grid):
    """Coupling edges follow the ADJACENT directed-pair layout of the series ones."""
    coupled = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    ei = coupled.edge_index
    assert torch.equal(ei[0, 0::2], ei[1, 1::2])
    assert torch.equal(ei[1, 0::2], ei[0, 1::2])
    assert torch.equal(coupled.edge_attr[0::2], coupled.edge_attr[1::2])
    assert torch.equal(coupled.edge_branch_ids[0::2], coupled.edge_branch_ids[1::2])


def test_default_graph_is_phase_disconnected_and_coupling_repairs_it(cigre_grid):
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre_grid)
    n = idx.size
    phase_of = idx.phase_codes.tolist()

    plain = build_graph(cigre_grid, orders=ORDERS)
    groups = _components(plain.edge_index, n)
    assert (
        len(groups) == 3
    )  # one per phase — a cross-phase pair is unreachable at any depth
    assert all(len({phase_of[r] for r in group}) == 1 for group in groups)

    coupled = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    assert len(_components(coupled.edge_index, n)) == 1


def test_edge_set_is_the_network_admittance_pattern(cigre_grid):
    """The union of series + coupling edges IS the off-diagonal sparsity of ``Y_net``."""
    from injection_hse._switches import materialize_estimator_grid
    from pgml.assembly import assemble_network_ybus, node_phase_index

    cigre_grid = materialize_estimator_grid(cigre_grid)
    idx = node_phase_index(cigre_grid)
    coupled = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    y = assemble_network_ybus(cigre_grid, [float(cigre_grid.base_frequency_hz)]).Y[0]
    magnitude = y.abs()
    magnitude.fill_diagonal_(0.0)
    rows, cols = torch.nonzero(magnitude > 0.0, as_tuple=True)
    expected = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(rows.tolist(), cols.tolist())
    }
    assert _undirected_pairs(coupled.edge_index) == expected
    assert idx.size == coupled.num_nodes


def test_coupling_edges_are_cross_phase_and_carry_their_branch(cigre_grid):
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre_grid)
    phase_of = idx.phase_codes.tolist()
    node_of = idx.node_ids.tolist()
    plain = build_graph(cigre_grid, orders=ORDERS)
    coupled = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    e = plain.edge_index.shape[1]

    src = coupled.edge_index[0, e:].tolist()
    dst = coupled.edge_index[1, e:].tolist()
    # every added edge crosses phases (the series set already carries every same-phase one)
    assert all(phase_of[a] != phase_of[b] for a, b in zip(src, dst))
    # the same-node flag agrees with the row layout
    same_node = coupled.edge_attr[e:, EDGE_FEATURE_DIM + 1]
    assert [bool(v) for v in (same_node > 0.5).tolist()] == [
        node_of[a] == node_of[b] for a, b in zip(src, dst)
    ]
    assert same_node.max() == 1.0 and same_node.min() == 0.0  # both families present
    # provenance: every added half-edge names a drawable branch of the grid
    drawable = set(plain.edge_branch_ids.tolist())
    assert set(coupled.edge_branch_ids[e:].tolist()) <= drawable


def test_tolerance_prunes_the_weak_couplings(cigre_grid):
    """The relative threshold is monotone: a bigger tolerance keeps fewer edges."""
    counts = [
        build_graph(
            cigre_grid,
            orders=ORDERS,
            coupling=PhaseCouplingConfig(enabled=True, tolerance=t),
        ).edge_index.shape[1]
        for t in (1e-9, 1e-1, 0.9)
    ]
    assert counts[0] >= counts[1] >= counts[2]
    assert counts[0] > counts[2]  # the CIGRE LV mutual couplings sit around 1e-1


# -- the models -----------------------------------------------------------------
@pyg_required
def test_gnn_forward_backward_with_coupling_edges(cigre_grid):
    from injection_hse.config import GNNConfig
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    n, f = g.num_nodes, len(ORDERS) * 3
    model = build_model(
        GNNConfig(hidden_dim=32, num_layers=2, conv="gine"),
        n_features=f,
        n_nodes=n,
        n_edge_static_features=edge_feature_dim(ON),
    )
    assert model.edge_encoder[0].in_features == edge_feature_dim(ON)
    feats = torch.randn(4, n, f, requires_grad=True)
    mask = torch.rand(4, n) > 0.5
    out = model(build_model_input(g, feats, mask))
    assert out.shape == (4, 1, n, f)
    out.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and float(sum(p.abs().sum() for p in grads)) > 0
    assert model.edge_encoder[0].weight.grad is not None


@pyg_required
def test_graphormer_forward_with_coupling_edges(cigre_grid):
    """The Graphormer reads only the resistance column, so the wider block is transparent."""
    from injection_hse.config import GraphormerConfig
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS, coupling=ON)
    n, f = g.num_nodes, len(ORDERS) * 3
    model = build_model(
        GraphormerConfig(
            hidden_dim=32, num_layers=1, num_heads=2, attention_bias="both"
        ),
        n_features=f,
        n_nodes=n,
        n_edge_static_features=edge_feature_dim(ON),
    )
    feats = torch.randn(2, n, f, requires_grad=True)
    mask = torch.rand(2, n) > 0.5
    out = model(build_model_input(g, feats, mask))
    assert out.shape == (2, 1, n, f)
    out.pow(2).mean().backward()
    assert feats.grad is not None


def test_experiment_config_rejects_coupling_with_measured_currents():
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        ExperimentConfig,
        MaskStageConfig,
        MeasurementConfig,
    )

    stages = CurriculumConfig(
        stages=[
            MaskStageConfig(name="s", kind="random", target_loss=1e-4, max_epochs=1)
        ]
    )
    with pytest.raises(ValueError, match="phase_coupling"):
        ExperimentConfig(
            data=DataConfig(phase_coupling=ON),
            measurement=MeasurementConfig(measure_current=True),
            curriculum=stages,
        )
    # either alone is fine
    ExperimentConfig(data=DataConfig(phase_coupling=ON), curriculum=stages)
    ExperimentConfig(
        measurement=MeasurementConfig(measure_current=True), curriculum=stages
    )


# -- the training module ---------------------------------------------------------
@pyg_required
def test_module_trains_and_reloads_on_the_coupling_graph(
    cigre_grid, se_dataset, tmp_path
):
    """End to end: the source builds the wider graph, the module sizes the GNN for it."""
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        ExperimentConfig,
        GNNConfig,
        MaskStageConfig,
    )
    from injection_hse.data import StateCollate, base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule

    data = DataConfig(source="parquet", dataset_path=str(se_dataset), phase_coupling=ON)
    source = build_data_source(data, grid=cigre_grid)
    graph = source.graph()
    assert graph.edge_attr.shape[1] == edge_feature_dim(ON)

    encoding = build_encoding("trig")
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    normalizer = HarmonicNormalizer().fit(encoding.encode(train_v).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="phase-coupling",
        seed=0,
        orders=list(source.orders),
        data=data,
        model=GNNConfig(hidden_dim=16, num_layers=2, conv="gine"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,
                    max_epochs=1,
                )
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=normalizer,
        graph=graph,
        base_v=base_voltage_per_row(cigre_grid),
    )
    assert module.model.edge_encoder[0].in_features == edge_feature_dim(ON)

    batch = StateCollate()([source.train_dataset()[i] for i in range(4)])
    loss = module.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)

    # the checkpoint rebuild derives the same graph and the same edge width from the config
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "coupled.ckpt"
    torch.save(checkpoint, path)
    reloaded = SEModule.from_checkpoint(path, grid=cigre_grid, device="cpu")
    assert reloaded.model.edge_encoder[0].in_features == edge_feature_dim(ON)
    assert torch.equal(
        reloaded.model.edge_encoder[0].weight,
        module.model.edge_encoder[0].weight,
    )
