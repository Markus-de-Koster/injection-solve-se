"""The ModelInput contract: build + shapes, edge plumbing, time folding, T=1 regression.

``test_t1_equals_old_behavior_graphormer`` is deliberately white-box: it replays the
Graphormer forward through ``Graphormer._compute_structural``, ``spatial_bias`` and the
blocks' rank-3 bias, as the exactness oracle of the ``T=1`` path. A Graphormer restructuring
(a renamed structural helper, a different bias rank, a new bias table) must update the
replay, not weaken the assertion.

The electrical attention bias is derived from ``edge_static`` (it inverts the
``log1p(|R|)`` column with ``expm1``), never from a ``Grid``, so a reloaded checkpoint
reproduces it without a grid object. If that edge column stops being ``log1p``-compressed,
the inversion silently changes scale; the bucket-equality test against
``injection_hse.data.pairwise_effective_resistance`` is what catches it.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import DNNConfig, GraphormerConfig
from injection_hse.io import build_model_input
from injection_hse.models import build_model

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _graph(n: int, e: int = 0):
    from torch_geometric.data import Data

    edge_index = (
        torch.randint(0, n, (2, e), dtype=torch.int64)
        if e
        else torch.zeros(2, 0, dtype=torch.int64)
    )
    d = Data(
        x=torch.randn(n, 8),
        edge_index=edge_index,
        edge_attr=torch.randn(e, 5),
    )
    d.num_nodes = n
    return d


def test_build_snapshot_lifts_t1():
    n, f, b = 10, 21, 4
    g = _graph(n, e=6)
    mi = build_model_input(g, torch.randn(b, n, f), torch.rand(b, n) > 0.5)
    assert mi.node_obs.shape == (b, 1, n, f)  # T=1 inserted
    assert mi.node_mask.shape == (b, 1, n)
    assert mi.node_static.shape == (n, 8)
    assert mi.edge_index.shape == (2, 6)
    assert mi.edge_static.shape == (6, 5)
    assert mi.n_steps == 1 and mi.n_nodes == n and mi.n_edges == 6
    assert mi.node_mask.dtype == torch.bool
    assert not mi.has_edge_obs


def test_build_sequence_keeps_time():
    n, f, b, t = 10, 21, 4, 5
    g = _graph(n, e=6)
    mi = build_model_input(
        g, torch.randn(b, t, n, f), torch.rand(b, t, n) > 0.5, has_time=True
    )
    assert mi.node_obs.shape == (b, t, n, f)
    assert mi.node_mask.shape == (b, t, n)
    assert mi.n_steps == t


def test_build_edge_dynamics_plumbing():
    n, f, b, e = 8, 21, 3, 6
    g = _graph(n, e=e)
    mi = build_model_input(
        g,
        torch.randn(b, n, f),
        torch.rand(b, n) > 0.5,
        edge_obs=torch.randn(b, e, 4),
        edge_mask=torch.rand(b, e) > 0.5,
    )
    assert mi.has_edge_obs
    assert mi.edge_obs.shape == (b, 1, e, 4)  # T=1 lifted
    assert mi.edge_mask.shape == (b, 1, e)
    assert mi.edge_mask.dtype == torch.bool


def test_edge_mask_without_edge_obs_raises():
    g = _graph(8, e=6)
    with pytest.raises(ValueError):
        build_model_input(
            g, torch.randn(2, 8, 21), torch.ones(2, 8), edge_mask=torch.ones(2, 6)
        )


def test_fold_time_into_batch_roundtrip():
    n, f, b, t, e = 8, 21, 3, 4, 6
    g = _graph(n, e=e)
    mi = build_model_input(
        g,
        torch.randn(b, t, n, f),
        torch.rand(b, t, n) > 0.5,
        has_time=True,
        edge_obs=torch.randn(b, t, e, 2),
        edge_mask=torch.rand(b, t, e) > 0.5,
    )
    folded, lead = mi.fold_time_into_batch()
    assert lead == (b, t)
    assert folded.node_obs.shape == (b * t, n, f)
    assert folded.node_mask.shape == (b * t, n)
    assert folded.edge_obs.shape == (b * t, e, 2)
    assert folded.edge_mask.shape == (b * t, e)
    # the entity ordering is preserved: unfold equals the original
    assert torch.equal(folded.node_obs.reshape(b, t, n, f), mi.node_obs)


def test_t1_equals_old_behavior_dnn(cigre_grid):
    """A T=1 ModelInput forward (squeezed) reproduces the old (features,mask) DNN logic."""
    from injection_hse.data import build_graph

    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    f = len(ORDERS) * 3
    b = 4
    torch.manual_seed(0)
    feats = torch.randn(b, n, f)
    mask = torch.rand(b, n) > 0.5
    model = build_model(DNNConfig(hidden_dims=[64, 64]), n_features=f, n_nodes=n).eval()

    new_out = model(build_model_input(g, feats, mask))[:, 0]  # squeeze T
    # replay the previous forward logic by hand (cat features+mask -> mlp -> reshape)
    mask_ch = mask.to(feats.dtype).unsqueeze(-1)
    x = torch.cat([feats, mask_ch], dim=-1).reshape(b, -1)
    old_out = model.mlp(x).reshape(b, n, f)
    assert torch.equal(new_out, old_out)


def test_t1_equals_old_behavior_graphormer(cigre_grid):
    from injection_hse.data import build_graph

    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    f = len(ORDERS) * 3
    b = 4
    torch.manual_seed(0)
    feats = torch.randn(b, n, f)
    mask = torch.rand(b, n) > 0.5
    gm = build_model(
        GraphormerConfig(hidden_dim=64, num_layers=2, num_heads=4),
        n_features=f,
        n_nodes=n,
    ).eval()

    new_out = gm(build_model_input(g, feats, mask))[:, 0]
    # replay: input_proj + degree_emb -> blocks(bias) -> head(norm_out)
    si, deg = type(gm)._compute_structural(g.edge_index, n)
    node_in = torch.cat(
        [feats, mask.float().unsqueeze(-1), g.x.unsqueeze(0).expand(b, n, -1)], dim=-1
    )
    h = gm.input_proj(node_in) + gm.degree_emb(deg).unsqueeze(0)
    bias = gm.spatial_bias(si).permute(2, 0, 1).unsqueeze(0)
    for blk in gm.blocks:
        h = blk(h, bias)
    old_out = gm.head(gm.norm_out(h))
    assert torch.allclose(new_out, old_out, atol=1e-5)


def test_to_moves_and_casts():
    g = _graph(8, e=4)
    mi = build_model_input(g, torch.randn(2, 8, 21), torch.ones(2, 8))
    moved = mi.to(dtype=torch.float64)
    assert moved.node_obs.dtype == torch.float64
    assert moved.node_static.dtype == torch.float64
    assert moved.node_mask.dtype == torch.bool  # masks not cast
    assert moved.edge_index.dtype == torch.int64  # index not cast


def test_model_input_is_frozen():
    g = _graph(8)
    mi = build_model_input(g, torch.randn(2, 8, 21), torch.ones(2, 8))
    with pytest.raises(Exception):
        mi.node_obs = torch.zeros(1)  # frozen dataclass
