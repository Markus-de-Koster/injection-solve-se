"""GPU gate: the multi-grid injection decode reproduces its CPU result on CUDA.

The per-member dispatch builds its gather/scatter indices on the batch device and reads the
operators out of a :class:`~injection_hse.physics.MultiGridPhysicsBank` — a plain object Lightning does
NOT relocate with the module, so the training step moves it itself. This runs the same batch
through the decode on CPU and on CUDA and asserts the reconstructed voltages, the per-unit
current base and the derived branch currents agree (skips cleanly without CUDA).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

ORDERS = [1, 3]


class _StubSource:
    """The source surface :class:`MultiGridPhysicsBank` needs (grids + their orders)."""

    def __init__(self, grids) -> None:
        self._grids = list(grids)
        self.orders = list(ORDERS)

    @property
    def grids(self):
        return self._grids

    def bundle(self, index: int):
        from injection_hse.data import base_voltage_per_row, build_graph

        grid = self._grids[int(index)]
        graph = build_graph(grid, orders=list(ORDERS))
        return SimpleNamespace(
            key=f"toy_{int(index)}",
            grid=grid,
            graph=graph,
            n_edges=int(graph.edge_index.shape[1]),
            base_voltage=base_voltage_per_row(grid),
        )


def _setup(device):
    """A bank + a three-item batch (two members, interleaved) on ``device``."""
    from pgml.grids import synthetic_feeder

    from injection_hse.physics import MultiGridPhysicsBank
    from injection_hse.train import SEModule

    grids = [synthetic_feeder(4, n_feeders=2), synthetic_feeder(5, n_feeders=2)]
    bank = MultiGridPhysicsBank(_StubSource(grids), dtype=torch.complex128)
    bank.to(device)
    sizes = [int(bank.solver(i).y_fundamental.shape[-1]) for i in range(2)]
    edges = [int(bank.current_operator(i).n_edges) for i in range(2)]
    items = [0, 1, 0]

    ptr, edge_ptr = [0], [0]
    for member in items:
        ptr.append(ptr[-1] + sizes[member])
        edge_ptr.append(edge_ptr[-1] + edges[member])
    batch = {
        "ptr": torch.tensor(ptr, dtype=torch.int64, device=device),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.int64, device=device),
        "grid_index": torch.tensor(items, dtype=torch.int64, device=device),
    }
    stub = SimpleNamespace(_bank=bank, _fundamental_pos=ORDERS.index(1))
    stub._multigrid_segments = SEModule._multigrid_segments.__get__(stub, SEModule)
    decode = SEModule._multigrid_decode.__get__(stub, SEModule)
    return decode, stub, batch, ptr[-1]


def _inputs(n_total, device):
    generator = torch.Generator().manual_seed(0)
    shape = (len(ORDERS), n_total)
    i_pred = torch.complex(
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )
    v_abs = 230.0 * torch.complex(
        torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5,
        0.1 * torch.rand(shape, generator=generator, dtype=torch.float64),
    )
    mask = torch.zeros(n_total, dtype=torch.bool)
    mask[::2] = True
    return i_pred.to(device), v_abs.to(device), mask.to(device)


def test_multigrid_decode_cpu_matches_cuda():
    cuda = torch.device("cuda")
    decode_cpu, _, batch_cpu, n_total = _setup(torch.device("cpu"))
    i_pred, v_abs, mask = _inputs(n_total, torch.device("cpu"))
    reference = decode_cpu(i_pred, v_abs, mask, batch_cpu)

    decode_gpu, _, batch_gpu, n_gpu = _setup(cuda)
    assert n_gpu == n_total
    decoded = decode_gpu(i_pred.to(cuda), v_abs.to(cuda), mask.to(cuda), batch_gpu)
    assert decoded.device.type == "cuda"
    assert torch.allclose(
        decoded.cpu(), reference, rtol=0.0, atol=1e-8 * float(reference.abs().max())
    )


def test_multigrid_current_base_and_branch_currents_match_on_cuda():
    from injection_hse.train import SEModule, _gather_segments, _scatter_segments

    cuda = torch.device("cuda")
    results = {}
    for device in (torch.device("cpu"), cuda):
        _, stub, batch, n_total = _setup(device)
        i_pred, v_abs, _ = _inputs(n_total, device)
        base = SEModule._multigrid_i_base.__get__(stub, SEModule)(batch)
        segments = stub._multigrid_segments(batch)
        parts = [
            stub._bank.current_operator(index)(_gather_segments(v_abs, node_rows))
            for index, node_rows, _ in segments
        ]
        edges = _scatter_segments(
            parts,
            [edge_rows for _, _, edge_rows in segments],
            int(batch["edge_ptr"][-1]),
        )
        results[device.type] = (base.cpu(), edges.cpu())

    base_cpu, edges_cpu = results["cpu"]
    base_gpu, edges_gpu = results["cuda"]
    assert torch.allclose(base_gpu, base_cpu, rtol=1e-9, atol=0.0)
    assert torch.allclose(
        edges_gpu, edges_cpu, rtol=0.0, atol=1e-8 * float(edges_cpu.abs().max())
    )
