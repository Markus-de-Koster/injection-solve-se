"""Datasheet error model of a power-quality instrument, split into a stable and a random part.

:class:`InstrumentNoise` draws the reading error of an instrument whose datasheet states
maximum errors per quantity (voltage, current) and order band (fundamental, harmonics)
(:class:`~injection_hse.config.InstrumentErrorConfig`). A limit is read as ``coverage`` standard
deviations. Each reading's regime follows its OWN magnitude:

- at or above the band's threshold (a fraction of the band's range, a multiple of the
  channel's nominal value), a
  PROPORTIONAL gain error and phase error; a share ``stable_fraction`` of their variance is
  held per scenario, channel and order (the instrument's calibration error, which cancels in
  a reading change), the rest is drawn per reading;
- at every level, an isotropic complex FLOOR referred to the band's range, drawn per
  reading (the instrument's resolution; below the threshold it is the whole error).

The held part is drawn per scenario from the model's stream by default. With a SITE KEY
(:attr:`InstrumentNoise.site`) it is a deterministic function of the seed and the key
instead, so readings of one site drawn in separate calls share the same gain and phase.

With per-real-component standard deviations ``σ_g`` (gain), ``σ_φ`` (phase, rad) and
``σ_f`` (floor), a reading ``y`` with regime indicator ``p`` (1 at or above the threshold)
has the marginal per-component variance::

    σ(y)² = ½·(σ_g² + σ_φ²)·p·|y|² + σ_f²

and the change ``y₂ - y₁`` of two readings of one scenario, channel and order, with
stable share ``s``::

    σ(Δ)² = ½·(σ_g² + σ_φ²)·[s·|p₂·y₂ - p₁·y₁|² + (1 - s)·(p₁·|y₁|² + p₂·|y₂|²)] + 2·σ_f²

The ½ spreads the anisotropic gain/phase error isotropically over the two components (the
convention of :func:`injection_hse.physics.anchor_measurement_sigma`); both are first order in the
error. :meth:`InstrumentNoise.reading_sigma` and :meth:`InstrumentNoise.change_sigma` return
them, so an estimator weighting by them agrees with the drawn errors by construction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from .measurement import base_voltage_per_row
from .noise import _EDGE_NOISE_SEED_OFFSET, _SeededStream

_MASK64 = (1 << 64) - 1


def _site_seed(seed: int, key: int) -> int:
    """A generator seed from a model seed and a site key (SplitMix64 finalizer)."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + int(key) + 0x5851F42D4C957F2D) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import InstrumentErrorConfig, NoiseConfig


class InstrumentNoise(_SeededStream):
    """Datasheet reading error of one quantity's channels, with a held and a random part.

    The sigma tensors are float64 on the CPU; :meth:`apply` moves them to the phasors'
    device.

    Parameters
    ----------
    spec:
        The instrument's error limits (:class:`~injection_hse.config.InstrumentErrorConfig`).
    quantity:
        ``"voltage"`` or ``"current"``: which bands of ``spec`` apply.
    nominal:
        float ``[R]`` (or a scalar, broadcast to one channel) nominal value of every channel
        in the units of the phasors the model perturbs: the line-to-neutral nominal voltage
        of a voltage row, the CT range of a current channel. Thresholds and floors refer
        to it through each band's ``fsr_over_nominal``.
    orders:
        The harmonic orders of the phasors' order axis; order 1 takes the fundamental band,
        every other order the harmonic band.
    seed:
        Seed of the draw stream (:class:`torch.Generator`; the same draws on CPU and GPU).
    """

    def __init__(
        self,
        spec: "InstrumentErrorConfig",
        *,
        quantity: Literal["voltage", "current"],
        nominal: Union[Tensor, float],
        orders: Sequence[int],
        seed: int = 0,
    ) -> None:
        if quantity not in ("voltage", "current"):
            raise ValueError(
                f"quantity must be 'voltage' or 'current', got {quantity!r}."
            )
        orders = tuple(int(o) for o in orders)
        if not orders:
            raise ValueError("an instrument noise model needs the harmonic orders.")
        beyond = [o for o in orders if o < 1 or o > int(spec.max_order)]
        if beyond:
            raise ValueError(
                f"orders {beyond} lie outside 1..{int(spec.max_order)}, where the "
                "instrument's datasheet states no error limit."
            )
        fundamental = getattr(spec, f"{quantity}_fundamental")
        harmonic = getattr(spec, f"{quantity}_harmonic")
        bands = [fundamental if o == 1 else harmonic for o in orders]
        k = float(spec.coverage)

        def per_order(values) -> Tensor:
            return torch.tensor(values, dtype=torch.float64)

        self._spec = spec
        self._quantity = quantity
        self._orders = orders
        self._gain = per_order([b.reading_pct / 100.0 / k for b in bands])  # [H]
        self._phase = per_order([math.radians(b.phase_deg) / k for b in bands])  # [H]
        # threshold and floor as fractions of the NOMINAL value: the band's range is
        # ``fsr_over_nominal`` times the nominal value
        self._floor_frac = per_order(
            [b.floor_pct / 100.0 / k * b.fsr_over_nominal for b in bands]
        )  # [H]
        self._threshold_frac = per_order(
            [b.threshold_pct / 100.0 * b.fsr_over_nominal for b in bands]
        )  # [H]
        self._stable = float(spec.stable_fraction)
        self._nominal = (
            torch.as_tensor(nominal, dtype=torch.float64).detach().cpu().reshape(-1)
        )  # [R]
        self._site: Union[int, Tensor, None] = None
        self._init_stream(seed)

    @classmethod
    def from_config(
        cls,
        cfg: "NoiseConfig",
        *,
        n_nodes: int,
        grid: Optional["Grid"],
        orders: Sequence[int],
    ) -> "InstrumentNoise":
        """The voltage-channel model of every node-phase row of ``grid``.

        ``cfg.instrument`` gives the limits; the nominal value of each row is its
        line-to-neutral base voltage (:func:`injection_hse.data.base_voltage_per_row`). A grid is
        required: without a nominal voltage neither the threshold nor the floor is defined.
        """
        if cfg.instrument is None:
            raise ValueError("noise.instrument is not set.")
        if grid is None:
            raise ValueError(
                "the instrument noise model refers its limits to each row's nominal "
                "voltage and needs the grid."
            )
        nominal = base_voltage_per_row(grid)
        if int(nominal.numel()) != int(n_nodes):
            raise ValueError(
                f"the grid has {int(nominal.numel())} node-phase rows, the model {n_nodes}."
            )
        return cls(
            cfg.instrument,
            quantity="voltage",
            nominal=nominal,
            orders=orders,
            seed=cfg.seed,
        )

    def for_edges(
        self,
        edge_src_rows: Tensor,
        *,
        current_range_a: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "InstrumentNoise":
        """The current-channel (CT) model of a half-edge fleet ``edge_src_rows`` ``[E]``.

        Every half-edge is read by the same instrument with the CT range
        ``current_range_a`` in amperes (:func:`injection_hse.data.grid_current_range`) as its nominal
        current; the current bands of the spec apply. The stream is independent of this
        model's (a distinct seed by default), as for :meth:`injection_hse.data.SensorNoise.for_edges`;
        the site key (:attr:`site`) is copied.
        """
        if current_range_a is None:
            raise ValueError(
                "the instrument noise model refers the current limits to the CT range; "
                "pass current_range_a."
            )
        n_edges = int(torch.as_tensor(edge_src_rows).numel())
        sibling = InstrumentNoise(
            self._spec,
            quantity="current",
            nominal=torch.full((n_edges,), float(current_range_a), dtype=torch.float64),
            orders=self._orders,
            seed=self._seed + _EDGE_NOISE_SEED_OFFSET if seed is None else int(seed),
        )
        sibling.site = self._site
        return sibling

    # -- the sigmas, read-only -----------------------------------------------------------
    @property
    def quantity(self) -> str:
        """``"voltage"`` or ``"current"``."""
        return self._quantity

    @property
    def orders(self) -> tuple[int, ...]:
        """The harmonic orders of the order axis the model perturbs."""
        return self._orders

    @property
    def nominal(self) -> Tensor:
        """float64 CPU ``[R]`` nominal value per channel (volts or amperes)."""
        return self._nominal

    @property
    def gain_sigma(self) -> Tensor:
        """float64 CPU ``[H]`` std of the proportional gain error (fraction of the reading)."""
        return self._gain

    @property
    def phase_sigma_rad(self) -> Tensor:
        """float64 CPU ``[H]`` std of the proportional phase error, in radians."""
        return self._phase

    @property
    def stable_fraction(self) -> float:
        """Share of the proportional error variance held per scenario, channel and order."""
        return self._stable

    @property
    def site(self) -> Union[int, Tensor, None]:
        """Site key of the held part: ``None`` (default), an ``int`` or int64 ``[S]``.

        ``None`` draws the held gain and phase per scenario from the model's stream (the
        first leading axis of :meth:`apply`). An ``int`` makes the whole call one site: the
        held part is shared over every leading axis and is a deterministic function of the
        seed and the key, so separate calls with the same key share it. An int64 tensor
        ``[S]`` gives the key of each scenario of the first leading axis (equal keys share
        the held part). :meth:`for_edges` copies the key; a key set afterwards must be set
        on the current sibling too. That model's own seed keeps its held part independent
        of the voltage channels'. The keyed draws are made on the CPU whatever the draw
        device.
        """
        return self._site

    @site.setter
    def site(self, key: Union[int, Tensor, Sequence[int], None]) -> None:
        if key is None or isinstance(key, int):
            self._site = key
            return
        keys = torch.as_tensor(key)
        if keys.dtype.is_floating_point or keys.is_complex() or keys.ndim > 1:
            raise ValueError("site must be None, an int or a 1-D integer tensor [S].")
        keys = keys.to(torch.int64).cpu()
        self._site = int(keys) if keys.ndim == 0 else keys

    def threshold(self, rows: Optional[Tensor] = None) -> Tensor:
        """float64 CPU ``[H, R]`` magnitude at and above which a reading is proportional.

        ``rows`` (int ``[R']``) selects channels; ``None`` = every channel.
        """
        return self._threshold_frac.reshape(-1, 1) * self._select(rows).reshape(1, -1)

    def floor_sigma(self, rows: Optional[Tensor] = None) -> Tensor:
        """float64 CPU ``[H, R]`` per-real-component std of the floor, in phasor units."""
        return self._floor_frac.reshape(-1, 1) * self._select(rows).reshape(1, -1)

    def _site_draw(self, key: int, h: int, r: int) -> Tensor:
        """float64 CPU ``[2, H, R]`` standard normals (gain, phase) of one site key."""
        gen = torch.Generator().manual_seed(_site_seed(self._seed, key))
        return torch.randn((2, h, r), generator=gen, dtype=torch.float64)

    def _held_normals(
        self, lead: tuple, h: int, r: int, device
    ) -> tuple[Tensor, Tensor]:
        """Standard normals ``(gain, phase)`` of the held part, broadcastable to the input."""
        key = self._site
        if key is None:
            held = (lead[0],) + (1,) * (len(lead) - 1) + (h, r) if lead else (h, r)
            gain = self._draw("randn", held, device).to(device)
            return gain, self._draw("randn", held, device).to(device)
        if isinstance(key, int):
            z = self._site_draw(key, h, r).to(device)
            shape = (1,) * len(lead) + (h, r)
            return z[0].reshape(shape), z[1].reshape(shape)
        if not lead or int(key.numel()) != lead[0]:
            raise ValueError(
                f"site keys [{int(key.numel())}] must match the first leading axis of "
                f"the readings {lead}."
            )
        unique, inverse = torch.unique(key, return_inverse=True)
        z = torch.stack([self._site_draw(int(k), h, r) for k in unique.tolist()])
        z = z.index_select(0, inverse).to(device)  # [S, 2, H, R]
        shape = (lead[0],) + (1,) * (len(lead) - 1) + (h, r)
        return z[:, 0].reshape(shape), z[:, 1].reshape(shape)

    def _select(self, rows: Optional[Tensor]) -> Tensor:
        if rows is None:
            return self._nominal
        return self._nominal.index_select(
            0, torch.as_tensor(rows).to(torch.int64).cpu()
        )

    def _terms(self, rows: Optional[Tensor], device) -> tuple[Tensor, Tensor, Tensor]:
        """``(threshold [H, R], floor² [H, R], ½(σ_g² + σ_φ²) [H, 1])`` on ``device``."""
        q = 0.5 * (self._gain**2 + self._phase**2)
        return (
            self.threshold(rows).to(device),
            self.floor_sigma(rows).pow(2).to(device),
            q.reshape(-1, 1).to(device),
        )

    def reading_sigma(self, magnitude: Tensor, rows: Optional[Tensor] = None) -> Tensor:
        """float64 per-real-component std of one reading's error, regime by its magnitude.

        ``magnitude`` is the real reading magnitude ``[..., H, R]`` (or ``[H]`` / ``[H, 1]``,
        broadcast over the channels) in the units of the phasors; ``rows`` selects the
        channels the last axis refers to. Returns ``[..., H, R]`` on ``magnitude``'s device;
        the held/random split does not change it.
        """
        m = torch.as_tensor(magnitude).to(torch.float64)
        m = m.reshape(-1, 1) if m.ndim < 2 else m
        thr, floor2, q = self._terms(rows, m.device)
        proportional = (m >= thr).to(torch.float64)
        return (q * proportional * m.pow(2) + floor2).sqrt()

    def change_sigma(
        self, before: Tensor, after: Tensor, rows: Optional[Tensor] = None
    ) -> Tensor:
        """float64 per-real-component std of the error of a reading CHANGE ``after - before``.

        ``before`` and ``after`` are complex readings ``[..., H, R]`` of the same scenario,
        channel and order (the regime of each follows its own magnitude); ``rows`` selects
        the channels. The held part of the proportional error enters through the change
        ``p₂·y₂ - p₁·y₁``, the random part and the floor of both readings add. Returns
        ``[..., H, R]`` on the readings' device.
        """
        y1 = torch.as_tensor(before).to(torch.complex128)
        y2 = torch.as_tensor(after).to(device=y1.device, dtype=torch.complex128)
        thr, floor2, q = self._terms(rows, y1.device)
        p1 = (y1.abs() >= thr).to(torch.float64)
        p2 = (y2.abs() >= thr).to(torch.float64)
        held = (p2 * y2 - p1 * y1).abs().pow(2)
        random = p1 * y1.abs().pow(2) + p2 * y2.abs().pow(2)
        s = self._stable
        return (q * (s * held + (1.0 - s) * random) + 2.0 * floor2).sqrt()

    # -- the draw ------------------------------------------------------------------------
    def apply(self, v: Tensor) -> Tensor:
        """Return a NOISY copy of the complex phasors ``v`` ``[*lead, H, R]``.

        The FIRST leading axis indexes independent scenarios; every further leading axis
        indexes the time steps of one scenario, over which the stable part is held (a
        ``[S, T, H, R]`` sequence batch holds it over ``T``; a ``[S, H, R]`` batch draws it
        per sample; pass one scenario's ``Q`` readings as ``[1, Q, H, R]``), unless a site
        key (:attr:`site`) fixes it across calls. A reading at or above its threshold is
        multiplied by ``(1 + g)·exp(jφ)`` with ``g`` and ``φ`` the sums of the held and the
        random proportional errors; every reading receives the complex floor. Draw order on
        the stream: held gain, held phase (if ``stable_fraction > 0`` and no site key is
        set), random gain, random phase (if ``stable_fraction < 1``), floor real part, floor
        imaginary part. Errors are formed in float64/complex128 and cast back to ``v.dtype``
        on ``v.device``; the clean ``v`` is not modified.
        """
        if v.numel() == 0:
            return v
        shape = tuple(v.shape)
        if v.dim() < 2:
            raise ValueError(f"v must be [*lead, H, R], got shape {shape}.")
        h, r = shape[-2], shape[-1]
        if h != len(self._orders) or r != int(self._nominal.numel()):
            raise ValueError(
                f"v has {h} orders x {r} channels; the model was built for "
                f"{len(self._orders)} x {int(self._nominal.numel())}."
            )
        lead = shape[:-2]
        device = v.device
        gain_sigma = self._gain.reshape(-1, 1).to(device)  # [H, 1]
        phase_sigma = self._phase.reshape(-1, 1).to(device)
        s = self._stable

        gain = torch.zeros((), dtype=torch.float64, device=device)
        phase = torch.zeros((), dtype=torch.float64, device=device)
        if s > 0.0:
            root = math.sqrt(s)
            z_gain, z_phase = self._held_normals(lead, h, r, device)
            gain = gain + z_gain * (root * gain_sigma)
            phase = phase + z_phase * (root * phase_sigma)
        if s < 1.0:
            root = math.sqrt(1.0 - s)
            gain = gain + self._draw("randn", shape, device).to(device) * (
                root * gain_sigma
            )
            phase = phase + self._draw("randn", shape, device).to(device) * (
                root * phase_sigma
            )
        floor = self.floor_sigma().to(device)  # [H, R]
        eps_re = self._draw("randn", shape, device).to(device)
        eps_im = self._draw("randn", shape, device).to(device)

        y = v.to(torch.complex128)
        proportional = y.abs() >= self.threshold().to(device)
        factor = torch.polar(1.0 + gain, phase)  # broadcasts to [*lead, H, R]
        factor = torch.where(proportional, factor, torch.ones_like(factor))
        out = y * factor + torch.complex(eps_re * floor, eps_im * floor)
        return out.to(v.dtype)


__all__ = ["InstrumentNoise"]
