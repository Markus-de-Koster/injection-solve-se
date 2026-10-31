"""GNN forward: shapes, gradient flow, device honoring, and the batched-edge math.

The PyG message-passing convs cannot be CONSTRUCTED on Python 3.14 with
torch-geometric 2.6.1 (its signature inspector reads ``typing.Union._name``, removed in 3.14).
Those tests skip cleanly when conv construction is unavailable; the batched-edge tiling (the
``injection_hse``-owned vectorized batching) is verified independently of PyG.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import GNNConfig


def _pyg_conv_available() -> bool:
    try:
        from torch_geometric.nn import SAGEConv

        SAGEConv(4, 4)
        return True
    except Exception:
        return False


PYG_OK = _pyg_conv_available()
pyg_required = pytest.mark.skipif(
    not PYG_OK,
    reason="torch-geometric conv construction unavailable (PyG 2.6.1 + Python 3.14).",
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def test_batched_edges_tiling():
    """The shared edge_index is tiled B times with per-copy offset i*N (no batch loop)."""
    from injection_hse.models.gnn import GNN

    gnn = GNN.__new__(GNN)  # bypass __init__ (no PyG needed for the pure-tensor helper)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)  # E=2, N=3
    big = gnn._batched_edges(edge_index, n=3, b=2)
    assert big.shape == (2, 4)
    # copy 0 unchanged; copy 1 shifted by N=3
    expected = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.int64)
    assert torch.equal(big, expected)


@pyg_required
@pytest.mark.parametrize("conv", ["gcn", "sage", "gin", "gine"])
def test_gnn_forward_backward(cigre_grid, conv):
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)
    n = g.num_nodes
    f = len(ORDERS) * 3
    b = 8
    model = build_model(
        GNNConfig(hidden_dim=64, num_layers=3, conv=conv), n_features=f, n_nodes=n
    )
    feats = torch.randn(b, n, f, requires_grad=True)
    mask = torch.rand(b, n) > 0.5
    out = model(build_model_input(g, feats, mask))
    assert out.shape == (b, 1, n, f)  # T=1 lifted
    loss = out.pow(2).mean()
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert float(grad) > 0
    assert feats.grad is not None and float(feats.grad.abs().sum()) > 0


@pyg_required
def test_gnn_consumes_edge_obs(cigre_grid):
    """gine folds the dynamic edge channel (lazy encoder built on first sight; grads flow)."""
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)
    n, e = g.num_nodes, g.edge_index.shape[1]
    f = len(ORDERS) * 3
    b = 4
    model = build_model(
        GNNConfig(hidden_dim=48, num_layers=2, conv="gine"), n_features=f, n_nodes=n
    )
    assert model.edge_obs_encoder is None  # no dynamic edge params until edge_obs seen
    mi = build_model_input(
        g,
        torch.randn(b, n, f),
        torch.rand(b, n) > 0.5,
        edge_obs=torch.randn(b, e, 4),
        edge_mask=torch.rand(b, e) > 0.5,
    )
    out = model(mi)
    assert out.shape == (b, 1, n, f)
    out.pow(2).mean().backward()
    assert model.edge_obs_encoder is not None  # lazily built
    assert float(model.edge_obs_encoder.weight.grad.abs().sum()) > 0


@pyg_required
def test_gnn_unsupported_conv_raises():
    from injection_hse.models import build_model

    with pytest.raises(NotImplementedError):
        build_model(
            GNNConfig(hidden_dim=32, num_layers=1, conv="gat"),
            n_features=21,
            n_nodes=132,
        )


@pyg_required
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_gnn_device_honoring(cigre_grid):
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)  # built on CPU
    n = g.num_nodes
    f = len(ORDERS) * 3
    model = build_model(
        GNNConfig(hidden_dim=32, num_layers=2, conv="gine"), n_features=f, n_nodes=n
    ).cuda()
    feats = torch.randn(4, n, f, device="cuda")
    mask = torch.ones(4, n, dtype=torch.bool, device="cuda")
    out = model(build_model_input(g, feats, mask))  # graph moved to CUDA inside forward
    assert out.device.type == "cuda"
    assert out.shape == (4, 1, n, f)


@pyg_required
def test_gnn_input_skip(cigre_grid):
    """input_skip doubles the head input (encoder embedding concatenated); off = old head."""
    from injection_hse.data import build_graph
    from injection_hse.io import build_model_input
    from injection_hse.models import build_model

    g = build_graph(cigre_grid, orders=ORDERS)
    n, f = g.num_nodes, len(ORDERS) * 3
    plain = build_model(GNNConfig(hidden_dim=64, num_layers=3), n_features=f, n_nodes=n)
    skip = build_model(
        GNNConfig(hidden_dim=64, num_layers=3, input_skip=True), n_features=f, n_nodes=n
    )
    assert plain.head.in_features == 64
    assert skip.head.in_features == 128
    feats = torch.randn(4, n, f, requires_grad=True)
    mask = torch.rand(4, n) > 0.5
    out = skip(build_model_input(g, feats, mask))
    assert out.shape == (4, 1, n, f)
    out.pow(2).mean().backward()
    assert feats.grad is not None and float(feats.grad.abs().sum()) > 0
