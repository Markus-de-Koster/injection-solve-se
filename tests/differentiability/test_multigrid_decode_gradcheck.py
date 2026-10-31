"""float64 gradcheck: the multi-grid injection decode's per-member dispatch.

A multi-grid batch carries the node-phase rows of several DIFFERENT grids on one concatenated
axis, so the physics decode gathers each member's item segments out of it, solves them with
that member's own ``Y(h)``, and scatters the results back into row order
(``injection_hse.train.SEModule._multigrid_decode``). Both directions have to be exactly differentiable:
a wrong permutation, an in-place scatter, or a gather that mixed two members would either kill
the gradient or route it to the wrong grid — and neither shows up as a forward error, since
every member's block is individually well formed.

The oracle is a float64 ``gradcheck`` of ``i_pred -> V`` over a three-item batch drawn from two
toy feeders (one member contributing two items, interleaved), plus the leakage check that the
gradient of one member's rows is exactly zero on the other member's injections.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from injection_hse.physics import MultiGridPhysicsBank
from injection_hse.train import SEModule

ORDERS = [1, 3]


class _StubSource:
    """The two attributes :class:`MultiGridPhysicsBank` needs of a data source.

    The gradient path under test is the dispatch + the per-member solve, so the toy grids are
    built directly instead of through a generated dataset (which would put parquet IO in a
    differentiability gate).
    """

    def __init__(self, grids) -> None:
        self._grids = list(grids)
        self.orders = list(ORDERS)

    @property
    def grids(self):
        return self._grids

    def bundle(self, index: int):
        return SimpleNamespace(grid=self._grids[int(index)])


def _decoder(anchor_v: float = 0.0):
    """A bank over two toy feeders + the bound decode, with the batch it dispatches on."""
    from pgml.grids import synthetic_feeder

    grids = [synthetic_feeder(4, n_feeders=2), synthetic_feeder(5, n_feeders=2)]
    bank = MultiGridPhysicsBank(
        _StubSource(grids), dtype=torch.complex128, anchor_v_weight=anchor_v
    )
    sizes = [int(bank.solver(i).y_fundamental.shape[-1]) for i in range(2)]
    edges = [12, 16]  # only their offsets are read (no current anchoring here)
    items = [0, 1, 0]  # member 0 twice, interleaved -> the scatter really permutes

    ptr = [0]
    edge_ptr = [0]
    for member in items:
        ptr.append(ptr[-1] + sizes[member])
        edge_ptr.append(edge_ptr[-1] + edges[member])
    batch = {
        "ptr": torch.tensor(ptr, dtype=torch.int64),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.int64),
        "grid_index": torch.tensor(items, dtype=torch.int64),
    }
    stub = SimpleNamespace(_bank=bank, _fundamental_pos=ORDERS.index(1))
    stub._multigrid_segments = SEModule._multigrid_segments.__get__(stub, SEModule)
    decode = SEModule._multigrid_decode.__get__(stub, SEModule)
    return decode, batch, ptr[-1]


def _boundary(n_total: int, *, seed: int = 0, scale: float = 230.0):
    """A deterministic complex state ``[H, N*]`` supplying the members' slack boundary."""
    generator = torch.Generator().manual_seed(seed)
    shape = (len(ORDERS), n_total)
    return scale * torch.complex(
        torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5,
        0.1 * torch.rand(shape, generator=generator, dtype=torch.float64),
    )


def test_gradcheck_multigrid_decode_wrt_the_predicted_injections():
    decode, batch, n_total = _decoder()
    v_abs = _boundary(n_total)
    mask = torch.ones(n_total, dtype=torch.bool)
    i_pred = torch.randn(
        (len(ORDERS), n_total), dtype=torch.complex128, requires_grad=True
    )

    def fn(i):
        # scaled to the amperes a per-unit injection maps to, so the solved voltages stay in
        # a range where the finite differences are meaningful
        return decode(i * 1e-2, v_abs, mask, batch)

    assert torch.autograd.gradcheck(fn, (i_pred,), eps=1e-6, atol=1e-6)


def test_gradcheck_multigrid_decode_wrt_the_measured_boundary():
    """The slack boundary is a MEASURED input; its gradient path must be exact too."""
    decode, batch, n_total = _decoder()
    mask = torch.ones(n_total, dtype=torch.bool)
    i_pred = torch.randn((len(ORDERS), n_total), dtype=torch.complex128) * 1e-2
    v_abs = _boundary(n_total, seed=1).requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda v: decode(i_pred, v, mask, batch), (v_abs,), eps=1e-6, atol=1e-6
    )


def test_gradcheck_multigrid_anchored_decode():
    """The measurement-anchored solve, dispatched the same way.

    The boundary is kept at unit scale here: the anchored solve is a reduced-correction least
    squares whose own residual is ~1e-10 RELATIVE, and against a 230 V state that error is
    larger than the 1e-6 perturbation's effect — a finite-difference resolution limit, not a
    gradient defect (the plain decode above is checked at the physical scale).
    """
    decode, batch, n_total = _decoder(anchor_v=1.0)
    v_abs = _boundary(n_total, seed=2, scale=1.0)
    mask = torch.zeros(n_total, dtype=torch.bool)
    mask[::2] = True
    i_pred = torch.randn(
        (len(ORDERS), n_total), dtype=torch.complex128, requires_grad=True
    )

    assert torch.autograd.gradcheck(
        lambda i: decode(i * 1e-2, v_abs, mask, batch),
        (i_pred,),
        eps=1e-6,
        atol=1e-6,
    )


def test_no_gradient_leaks_between_member_grids():
    """One item's voltages depend on NO other item's injections (the dispatch is disjoint)."""
    decode, batch, n_total = _decoder()
    v_abs = _boundary(n_total, seed=3)
    mask = torch.ones(n_total, dtype=torch.bool)
    i_pred = torch.randn(
        (len(ORDERS), n_total), dtype=torch.complex128, requires_grad=True
    )
    decoded = decode(i_pred * 1e-2, v_abs, mask, batch)

    ptr = batch["ptr"].tolist()
    lo, hi = ptr[1], ptr[2]  # the middle item (the other member)
    decoded[..., lo:hi].abs().sum().backward()
    grad = i_pred.grad.abs()
    assert float(grad[..., lo:hi].max()) > 0.0
    assert float(grad[..., :lo].max()) == 0.0
    assert float(grad[..., hi:].max()) == 0.0
