"""Split-real encodings of the complex node-phase-harmonic state.

Complex-valued networks are still poorly supported, so the complex state ``V`` is mapped to
real feature channels for the model and back. Two encodings, behind one interface so they
are interchangeable from config (both performed best in prior work):

- :class:`TrigEncoding` (default) — ``(m, cosθ, sinθ)``: a physically-meaningful magnitude
  channel (voltage drops/swells) plus a wrap-free direction (no ±180° discontinuity).
- :class:`CartesianEncoding` — ``(re, im)``: degrades gracefully near zero.

The LOSS is computed on the Cartesian reconstruction in BOTH cases (see
:meth:`StateEncoding.to_cartesian`): when a harmonic is ≈ 0 both reconstructed components
→ 0, so the meaningless angle of a non-existent signal is not penalized. Rationale:
``docs/pgl/index.md`` §3.

All ops are plain differentiable torch (real dtypes), run on CPU/CUDA unchanged, and act on
the trailing complex axis so any leading ``[*batch, N, H]`` layout broadcasts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class StateEncoding(ABC):
    """Map complex ``V`` ``[*shape]`` <-> real features ``[*shape, n_features]``."""

    #: number of real channels produced per complex value.
    n_features: int

    @abstractmethod
    def encode(self, v: Tensor) -> Tensor:
        """Complex ``[*shape]`` -> real features ``[*shape, n_features]``."""

    @abstractmethod
    def decode(self, feats: Tensor) -> Tensor:
        """Real features ``[*shape, n_features]`` -> complex ``[*shape]``."""

    @abstractmethod
    def to_cartesian(self, feats: Tensor) -> Tensor:
        """Real features ``[*shape, n_features]`` -> stacked ``(re, im)`` ``[*shape, 2]``.

        The loss space. Differentiable in the model-output features so gradients flow.
        """


class TrigEncoding(StateEncoding):
    """``(m, cosθ, sinθ)`` — magnitude + wrap-free direction (the default)."""

    n_features = 3

    def encode(self, v: Tensor) -> Tensor:
        m = v.abs()
        ang = torch.angle(v)  # angle(0) = 0 -> (cos, sin) = (1, 0); zero-safe
        return torch.stack([m, torch.cos(ang), torch.sin(ang)], dim=-1)

    def decode(self, feats: Tensor) -> Tensor:
        m, c, s = feats.unbind(dim=-1)
        return torch.complex(m * c, m * s)

    def to_cartesian(self, feats: Tensor) -> Tensor:
        m, c, s = feats.unbind(dim=-1)
        return torch.stack([m * c, m * s], dim=-1)


class CartesianEncoding(StateEncoding):
    """``(re, im)`` — graceful near zero; loss space is the encoding itself."""

    n_features = 2

    def encode(self, v: Tensor) -> Tensor:
        return torch.stack([v.real, v.imag], dim=-1)

    def decode(self, feats: Tensor) -> Tensor:
        re, im = feats.unbind(dim=-1)
        return torch.complex(re, im)

    def to_cartesian(self, feats: Tensor) -> Tensor:
        return feats


def build_encoding(kind: str) -> StateEncoding:
    """Construct a :class:`StateEncoding` from an :class:`~injection_hse.config.EncodingConfig` kind."""
    if kind == "trig":
        return TrigEncoding()
    if kind == "cartesian":
        return CartesianEncoding()
    raise ValueError(f"unknown encoding kind {kind!r} (use 'trig' or 'cartesian').")


def build_injection_encoding(cfg) -> StateEncoding:
    """The encoding of the INJECTION head, from an :class:`~injection_hse.config.EncodingConfig`.

    ``cfg.injection_kind`` when set, else ``cfg.kind`` (the head shares the state encoding).
    Every site that encodes or decodes predicted / target nodal injection currents — the
    injection normalizer fit, the decode of the model's raw output, the direct injection term
    — resolves the encoding here, so the parameterisation of the injection space is decided in
    one place. The voltage features always use :func:`build_encoding` on ``cfg.kind``.
    """
    kind = cfg.kind if cfg.injection_kind is None else cfg.injection_kind
    return build_encoding(kind)


__all__ = [
    "StateEncoding",
    "TrigEncoding",
    "CartesianEncoding",
    "build_encoding",
    "build_injection_encoding",
]
