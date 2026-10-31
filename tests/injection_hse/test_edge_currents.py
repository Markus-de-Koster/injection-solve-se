"""BranchCurrentOperator: exactness vs pgml.assembly.branch_currents + graph alignment."""

from __future__ import annotations

import pytest
import torch

ORDERS = [1, 3, 5, 7, 9, 11, 13]


@pytest.fixture(scope="module")
def operator(cigre_grid):
    from injection_hse.data.currents import BranchCurrentOperator

    return BranchCurrentOperator(cigre_grid, ORDERS, dtype=torch.complex128)


def _random_state(cigre_grid, lead: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    from pgml.assembly import node_phase_index

    n = node_phase_index(cigre_grid).size
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn((*lead, len(ORDERS), n), generator=gen, dtype=torch.float64)
    im = torch.randn((*lead, len(ORDERS), n), generator=gen, dtype=torch.float64)
    return torch.complex(re, im)


def _direct_half_edge_currents(cigre_grid, v: torch.Tensor) -> torch.Tensor:
    """Reference: direct branch_currents on ``v``, mapped to the half-edge axis."""
    from injection_hse._switches import materialize_estimator_grid
    from pgml.assembly import branch_currents, node_phase_index
    from pgml.schemas.grid_schema import Switch

    cigre_grid = materialize_estimator_grid(cigre_grid)
    idx = node_phase_index(cigre_grid)
    f0 = float(cigre_grid.base_frequency_hz)
    by_id = {
        bc.branch_id: bc
        for bc in branch_currents(
            cigre_grid, v, [o * f0 for o in ORDERS], idx, dtype=torch.complex128
        )
    }
    cols = []
    for b in cigre_grid.branches:
        if not getattr(b, "in_service", True):
            continue
        if isinstance(b, Switch) and not b.default_closed:
            continue
        if int(b.from_node) == int(b.to_node):
            continue
        for k, (fph, tph) in enumerate(zip(b.from_phases, b.to_phases)):
            try:
                idx.row(int(b.from_node), fph)
                idx.row(int(b.to_node), tph)
            except KeyError:
                continue
            cols.append(by_id[int(b.id)].i_from[..., k])
            cols.append(by_id[int(b.id)].i_to[..., k])
    return torch.stack(cols, dim=-1)  # [*lead, H, E]


def test_matches_direct_branch_currents_snapshot(cigre_grid, operator):
    v = _random_state(cigre_grid, (3,))
    got = operator(v)
    ref = _direct_half_edge_currents(cigre_grid, v)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-9)


def test_matches_direct_branch_currents_sequence(cigre_grid, operator):
    v = _random_state(cigre_grid, (2, 4), seed=1)  # [B, T, H, N]
    got = operator(v)
    ref = _direct_half_edge_currents(cigre_grid, v.reshape(8, len(ORDERS), -1))
    torch.testing.assert_close(
        got.reshape(8, len(ORDERS), -1), ref, rtol=1e-10, atol=1e-9
    )


def test_edge_layout_matches_build_graph(cigre_grid, operator):
    from injection_hse.data import build_graph

    graph = build_graph(cigre_grid, orders=ORDERS)
    assert operator.n_edges == int(graph.edge_index.shape[1])
    assert torch.equal(operator.edge_src_rows, graph.edge_index[0].cpu())


def test_differentiable_in_v(cigre_grid, operator):
    v = _random_state(cigre_grid, (2,)).requires_grad_(True)
    out = operator(v)
    out.abs().sum().backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad.abs().sum())


def test_dtype_follows_input(cigre_grid):
    from injection_hse.data.currents import BranchCurrentOperator

    op64 = BranchCurrentOperator(cigre_grid, ORDERS, dtype=torch.complex64)
    v = _random_state(cigre_grid, (2,)).to(torch.complex64)
    out = op64(v)
    assert out.dtype == torch.complex64
    # complex128 input against complex64 weights promotes to the input precision
    out128 = op64(_random_state(cigre_grid, (2,)))
    assert out128.dtype == torch.complex128
