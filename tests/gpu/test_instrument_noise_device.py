"""GPU gate: the instrument error model draws the same errors for CUDA and CPU phasors
(skips cleanly without CUDA)."""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import InstrumentErrorConfig
from injection_hse.data import InstrumentNoise

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _model(seed: int = 0) -> InstrumentNoise:
    return InstrumentNoise(
        InstrumentErrorConfig(stable_fraction=0.5),
        quantity="voltage",
        nominal=torch.full((6,), 230.0),
        orders=[1, 5, 7],
        seed=seed,
    )


def test_cuda_phasors_draw_the_cpu_errors():
    gen = torch.Generator().manual_seed(0)
    mag = torch.tensor([230.0, 9.0, 1.0], dtype=torch.float64).reshape(3, 1)
    v = mag * torch.exp(
        1j * torch.rand((4, 5, 3, 6), generator=gen, dtype=torch.float64)
    )
    cpu = _model().apply(v)
    cuda = _model().apply(v.cuda())
    assert cuda.device.type == "cuda"
    assert torch.allclose(cuda.cpu(), cpu, rtol=1e-12, atol=1e-12)
    on_device = _model().draw_on_input_device().apply(v.cuda())
    assert on_device.device.type == "cuda"
    sig = _model().reading_sigma(v.abs().cuda())
    assert sig.device.type == "cuda"
