"""Opt-in measurement noise on the OBSERVED phasors (keyed by the device accuracy class).

:class:`SensorNoise` perturbs each measured complex phasor with a MULTIPLICATIVE magnitude
error, an ADDITIVE phase error, and an ADDITIVE range-referenced FLOOR, drawn from
independent zero-mean Gaussians whose standard deviations come from the sensor's
:class:`~pgml.schemas.grid_schema.MeasurementDevice` accuracy class
(:class:`~injection_hse.config.NoiseConfig`). The floor is the part a purely multiplicative model
misses: an instrument's resolution is set by its RANGE, so a harmonic bin reading
millivolts against a 230 V range sees the floor, not a fraction of its own reading —
without it, noise studies systematically flatter the harmonic channels
(``injection_hse.analysis.feasibility`` quantifies why those are the binding ones). It acts on the
model INPUT only — the supervised targets and the reported metrics read the clean state —
so a run is a study of the estimator's robustness to instrument error, not of a corrupted
ground truth.

The same class perturbs both the metered VOLTAGE phasors and the metered BRANCH-CURRENT (CT)
phasors: the two are the readings of one measurement device, so a current channel inherits its
metering node's accuracy class (:meth:`SensorNoise.for_edges`). The CT error is drawn from an
INDEPENDENT stream (a distinct seed), so a run reflects a real instrument's separate voltage-
and current-transformer errors rather than one perfectly correlated perturbation.

The class -> sigma mapping is a pragmatic default (documented on
:class:`~injection_hse.config.AccuracyClassNoise`), NOT an IEC 61869 instrument-transformer error model.
The noise is deterministic given the config seed and vanishes exactly when disabled (nothing
is drawn), so a noise-off run is byte-identical to one built before noise existed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

from pgml.assembly import node_phase_index

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import NoiseConfig

# Seed offset for the branch-current (CT) noise stream, so the current error is drawn
# INDEPENDENTLY of the voltage error even though it shares each device's accuracy class.
_EDGE_NOISE_SEED_OFFSET = 1_000_003


class _SeededStream:
    """The seeded draw stream shared by the sensor-noise models.

    A CPU :class:`torch.Generator` seeded from ``seed`` draws every error, so the noise is
    identical on CPU and GPU (drawn on the CPU, moved to the phasors' device) and
    reproducible from the seed; :meth:`draw_on_input_device` switches to one generator per
    device, seeded alike.
    """

    def _init_stream(self, seed: int) -> None:
        self._seed = int(seed)
        self._gen = torch.Generator().manual_seed(self._seed)
        # Draw on the phasors' device instead of the CPU (:meth:`draw_on_input_device`); the
        # device generators are created lazily, one per device, from the same seed.
        self._on_input_device = False
        self._device_gens: dict = {}

    def reseed(self, seed: int) -> None:
        """Re-seed the draw generator (used to decorrelate ranks under distributed training)."""
        self._seed = int(seed)
        self._gen = torch.Generator().manual_seed(self._seed)
        self._device_gens = {}

    def draw_on_input_device(self, enabled: bool = True):
        """Draw the noise on the phasors' own device (``NoiseConfig.draw_device="input"``).

        Same distribution and seed, a device-specific stream; the default CPU draw stays
        device-independent. Returns ``self``.
        """
        self._on_input_device = bool(enabled)
        self._device_gens = {}
        return self

    def _draw(self, kind: str, shape, device) -> Tensor:
        """A float64 ``randn``/``rand`` draw of ``shape`` from this model's stream.

        On the CPU generator (then moved by the caller) unless :meth:`draw_on_input_device`
        is set and ``device`` is not the CPU, in which case it is drawn on ``device``.
        """
        fn = torch.randn if kind == "randn" else torch.rand
        dev = torch.device(device)
        if not self._on_input_device or dev.type == "cpu":
            return fn(shape, generator=self._gen, dtype=torch.float64)
        key = str(dev)
        gen = self._device_gens.get(key)
        if gen is None:
            gen = torch.Generator(device=dev).manual_seed(self._seed)
            self._device_gens[key] = gen
        return fn(shape, generator=gen, dtype=torch.float64, device=dev)


class SensorNoise(_SeededStream):
    """Per-node-phase-row magnitude + phase Gaussian noise on complex measured phasors.

    ``mag_sigma`` ``[N]`` is the standard deviation of the multiplicative magnitude error
    (fraction of the reading) and ``phase_sigma_rad`` ``[N]`` the standard deviation of the
    additive phase error (radians), one value per node-phase row. A CPU
    :class:`torch.Generator` seeds the draws so the noise is identical on CPU and GPU (drawn on
    CPU, moved to the phasor's device) and reproducible from the seed.
    """

    def __init__(
        self,
        mag_sigma: Tensor,
        phase_sigma_rad: Tensor,
        floor_sigma: Optional[Tensor] = None,
        floor_frac: Optional[Tensor] = None,
        *,
        order_floor_scale: Optional[Tensor] = None,
        seed: int = 0,
    ) -> None:
        self._mag = mag_sigma.to(torch.float64).cpu()  # [N]
        self._phase = phase_sigma_rad.to(torch.float64).cpu()  # [N]
        n = self._mag.shape[-1]
        self._floor = (
            floor_sigma.to(torch.float64).cpu()
            if floor_sigma is not None
            else torch.zeros(n, dtype=torch.float64)
        )  # [N] ABSOLUTE per-component std, units of the phasor
        self._floor_frac = (
            floor_frac.to(torch.float64).cpu()
            if floor_frac is not None
            else torch.zeros(n, dtype=torch.float64)
        )  # [N] fraction of the instrument range (re-ranged by :meth:`for_edges`)
        # [H] per-ORDER multiplier on the absolute floor. The floor is a fraction of the
        # instrument RANGE, and a range referenced to the fundamental is the wrong scale for a
        # harmonic subchannel: the same absolute floor is 0.05 % of a 230 V fundamental but
        # 17 % of a 0.7 V nineteenth, which drowns exactly the orders the estimator exists to
        # recover. ``None`` = one scale for every order (the fundamental-referenced form).
        self._order_floor = (
            order_floor_scale.to(torch.float64).cpu()
            if order_floor_scale is not None
            else None
        )
        self._init_stream(seed)
        # Magnitude-only (nominal-angle-filled) meter share and its RMS reading error; set by
        # :meth:`uniform` from ``NoiseConfig.angle_fill_fraction``. ``0`` = off.
        self._fill_fraction = 0.0
        self._fill_mag = 0.0

    @staticmethod
    def _order_floor_scale(cfg: "NoiseConfig", orders) -> Optional[Tensor]:
        """``[H]`` floor multiplier: 1 at the fundamental, ``harmonic_range`` above it.

        ``None`` when there is nothing to scale (no orders given, or the config asks for the
        fundamental-referenced floor at every order).
        """
        scale = float(cfg.harmonic_range)
        if not orders or scale == 1.0:
            return None
        return torch.tensor(
            [1.0 if int(o) == 1 else scale for o in orders], dtype=torch.float64
        )

    @classmethod
    def uniform(
        cls,
        cfg: "NoiseConfig",
        *,
        nominal: Optional[float] = None,
        orders: Optional[Sequence[int]] = None,
        seed_offset: int = 0,
    ) -> "SensorNoise":
        """One sigma set for EVERY row — the multi-grid form of the noise model.

        A multi-grid module has no single row layout (members differ in size) and a
        generated corpus carries no measurement-device metadata, so the per-row
        machinery of :meth:`from_config` has nothing to key on: every channel draws the
        ``default_class`` sigmas. The size-1 sigma tensors broadcast against any row
        count, which makes the model layout-independent; ``nominal`` is the instrument
        range in the units of the phasors the model will see (``1.0`` for a per-unit
        state — the floor then reads directly as a fraction of nominal), ``None``
        disables the floor (e.g. current channels whose range has no defined base on
        the per-unit path).
        """
        default = cfg.classes.get(cfg.default_class)
        if default is None:
            raise ValueError(
                f"noise.default_class {cfg.default_class!r} is not in noise.classes "
                f"{sorted(cfg.classes)}."
            )
        one = torch.ones(1, dtype=torch.float64)
        frac = one * (default.floor_pct / 100.0)
        floor = frac * float(nominal) if nominal is not None else one * 0.0
        noise = cls(
            one * (default.magnitude_pct / 100.0),
            one * math.radians(default.phase_deg),
            floor,
            frac,
            order_floor_scale=cls._order_floor_scale(cfg, orders),
            seed=cfg.seed + seed_offset,
        )
        fill_fraction = float(getattr(cfg, "angle_fill_fraction", 0.0))
        if fill_fraction > 0.0 and nominal is not None:
            # voltage channels only (``nominal`` set): the current channels keep the
            # default class
            fill_class = cfg.classes[cfg.angle_fill_class or cfg.default_class]
            noise._fill_fraction = fill_fraction
            noise._fill_mag = fill_class.magnitude_pct / 100.0
        return noise

    @classmethod
    def from_config(
        cls,
        cfg: "NoiseConfig",
        *,
        n_nodes: int,
        grid: Optional["Grid"] = None,
        orders: Optional[Sequence[int]] = None,
    ) -> "SensorNoise":
        """Build the per-row sigmas from the config + the grid's metered accuracy classes.

        Every row starts at the ``default_class`` sigma; a row belonging to an in-service
        voltage :class:`~pgml.schemas.grid_schema.MeasurementDevice` is overridden with that
        device's accuracy class (falling back to ``default_class`` for an unknown class label).
        Without a grid — a config-driven placement carrying no device metadata — every row uses
        ``default_class``.
        """
        default = cfg.classes.get(cfg.default_class)
        if default is None:
            raise ValueError(
                f"noise.default_class {cfg.default_class!r} is not in noise.classes "
                f"{sorted(cfg.classes)}."
            )
        mag = torch.full((n_nodes,), default.magnitude_pct / 100.0, dtype=torch.float64)
        phase = torch.full(
            (n_nodes,), math.radians(default.phase_deg), dtype=torch.float64
        )
        # The range-referenced FLOOR needs each row's instrument range (the nominal
        # line-to-neutral voltage); without a grid there is no voltage base, so
        # config-driven placements carry reading/phase noise only.
        floor_frac = torch.full(
            (n_nodes,), default.floor_pct / 100.0, dtype=torch.float64
        )
        floor_abs = torch.zeros(n_nodes, dtype=torch.float64)
        if grid is not None:
            idx = node_phase_index(grid)
            nodes_by_id = {int(n.id): n for n in grid.nodes}
            for nid, node in nodes_by_id.items():
                n_ph = len([p for p in node.phases if p.value != "n"])
                u_ln = float(node.u_rated_v) / (3.0**0.5 if n_ph >= 3 else 1.0)
                for row in idx.rows(nid):
                    floor_abs[row] = u_ln
            if grid.measurement_devices:
                for dev in grid.measurement_devices:
                    if not dev.in_service or "voltage" not in dev.measured_quantities:
                        continue
                    key = (
                        str(dev.accuracy_class)
                        if dev.accuracy_class is not None
                        else None
                    )
                    klass = cfg.classes.get(key, default)
                    sig_m = klass.magnitude_pct / 100.0
                    sig_p = math.radians(klass.phase_deg)
                    for row in idx.rows(int(dev.node)):
                        mag[row] = sig_m
                        phase[row] = sig_p
                        floor_frac[row] = klass.floor_pct / 100.0
        return cls(
            mag,
            phase,
            floor_frac * floor_abs,
            floor_frac,
            order_floor_scale=cls._order_floor_scale(cfg, orders),
            seed=cfg.seed,
        )

    def for_edges(
        self,
        edge_src_rows: Tensor,
        *,
        current_range_a: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "SensorNoise":
        """A sibling noise model for the branch-current (CT) channels of a half-edge fleet.

        Each directed half-edge inherits the accuracy class of the node that meters it — the CT
        shares its device with the voltage sensor — by gathering this model's per-row sigmas at
        the half-edges' source (observing-terminal) rows ``edge_src_rows`` ``[E]`` (the
        ``build_graph`` / :class:`~injection_hse.data.currents.BranchCurrentOperator` layout). The returned
        model draws from an INDEPENDENT stream (a distinct seed by default), so the current error
        is not perfectly correlated with the voltage error at the same location.

        ``current_range_a`` is the CT range in amperes that the class ``floor_pct`` refers
        to (see :func:`grid_current_range`); ``None`` disables the floor on the current
        channels (reading/phase noise only).
        """
        rows = edge_src_rows.to(torch.int64).cpu()
        edge_seed = self._seed + _EDGE_NOISE_SEED_OFFSET if seed is None else int(seed)
        frac = self._floor_frac.index_select(0, rows)
        floor = frac * float(current_range_a) if current_range_a is not None else None
        return SensorNoise(
            self._mag.index_select(0, rows),
            self._phase.index_select(0, rows),
            floor,
            frac,
            seed=edge_seed,
        )

    # -- the sigmas, read-only -----------------------------------------------------------
    # A consumer that must AGREE with the applied noise — the noise-calibrated anchor
    # weights of the injection decode (``injection_hse.physics.NetworkSolver.set_anchor_calibration``)
    # weight each channel by the inverse variance of the error this model draws for it —
    # reads the same tensors :meth:`apply` draws from, so the two cannot drift apart.
    @property
    def mag_sigma(self) -> Tensor:
        """float64 CPU ``[R]`` std of the MULTIPLICATIVE magnitude error (fraction of the reading).

        ``R`` is the channel count the model was built for — node-phase rows
        (:meth:`from_config`), half-edges (:meth:`for_edges`) or ``1`` (:meth:`uniform`,
        broadcasting over any row count).
        """
        return self._mag

    @property
    def phase_sigma_rad(self) -> Tensor:
        """float64 CPU ``[R]`` std of the ADDITIVE phase error, in radians."""
        return self._phase

    @property
    def floor_sigma(self) -> Tensor:
        """float64 CPU ``[R]`` ABSOLUTE per-real-component std of the range floor.

        In the units of the phasors the model perturbs, referenced to the FUNDAMENTAL
        instrument range (``floor_frac × range``); :attr:`order_floor_scale` multiplies it
        per order. Zero = no floor on that channel.
        """
        return self._floor

    @property
    def floor_frac(self) -> Tensor:
        """float64 CPU ``[R]`` floor as a FRACTION of the instrument range (the class value)."""
        return self._floor_frac

    @property
    def order_floor_scale(self) -> Optional[Tensor]:
        """float64 CPU ``[H]`` per-order multiplier on :attr:`floor_sigma`, or ``None``.

        ``None`` = one floor for every order (the fundamental-referenced form, also what a
        :meth:`for_edges` sibling uses).
        """
        return self._order_floor

    def apply(self, v: Tensor) -> Tensor:
        """Return a NOISY copy of the complex phasors ``v`` ``[*lead, H, N]``.

        Each phasor is scaled by ``1 + N(0, mag_sigma)``, rotated by ``N(0, phase_sigma)``,
        and shifted by a complex range-referenced floor ``N(0, floor) + j·N(0, floor)``
        (row-indexed sigmas broadcast over the leading and harmonic axes). Independent
        draws per ``(sample[, step], order, row)``; the clean ``v`` is not modified in
        place. A model whose floor is zero everywhere draws exactly the reading/phase
        streams (bit-compatible with the floor-free behaviour).
        """
        if v.numel() == 0:
            return v
        shape = tuple(v.shape)  # [*lead, H, N]
        mag_sigma = self._mag.view(*([1] * (v.dim() - 1)), -1)  # broadcast to [..., N]
        phase_sigma = self._phase.view(*([1] * (v.dim() - 1)), -1)
        eps_mag = self._draw("randn", shape, v.device)
        eps_phase = self._draw("randn", shape, v.device)
        mag_sigma = mag_sigma.to(eps_mag.device)
        phase_sigma = phase_sigma.to(eps_phase.device)
        scale = (1.0 + eps_mag * mag_sigma).to(v.device)
        rot = torch.polar(
            torch.ones(shape, dtype=torch.float64, device=v.device),
            (eps_phase * phase_sigma).to(v.device),
        ).to(v.dtype)
        out = v * scale.to(v.dtype) * rot
        if bool((self._floor > 0).any()):
            floor = self._floor.view(*([1] * (v.dim() - 1)), -1)
            if self._order_floor is not None and v.dim() >= 2:
                # [..., H, N]: the order axis is the second-to-last
                floor = floor * self._order_floor.view(*([1] * (v.dim() - 2)), -1, 1)
            eps_re = self._draw("randn", shape, v.device)
            eps_im = self._draw("randn", shape, v.device)
            floor = floor.to(eps_re.device)
            shift = torch.complex(eps_re * floor, eps_im * floor).to(v.device)
            out = out + shift.to(v.dtype)
        if self._fill_fraction > 0.0:
            out = self._angle_fill(v, out)
        return out

    def _angle_fill(self, v: Tensor, out: Tensor) -> Tensor:
        """Replace a random share of the rows by MAGNITUDE-ONLY, nominal-angle readings.

        Per leading index and row (shared across the order axis), a row becomes an RMS
        meter with probability ``_fill_fraction``: its reading is ``|v|·(1 + N(0, σ))`` at
        the row's NOMINAL angle. In the slack-referenced frame the multi-grid path draws its
        noise in, the nominal angle is zero for a row galvanically on the slack's side and
        a multiple of 30° behind a transformer with a vector-group shift (Dyn5: -150°), so
        the nominal angle is the true angle rounded to the nearest 30° — the load-dependent
        deviation from it (a few degrees at most at the fundamental) is what the filled
        reading loses. The remaining rows keep the phasor reading in ``out``.
        """
        shape = tuple(v.shape)  # [*lead, H, N]
        row_shape = shape[:-2] + (1, shape[-1])
        u = self._draw("rand", row_shape, v.device)
        eps = self._draw("randn", shape, v.device)
        filled_mask = (u < self._fill_fraction).to(v.device)
        v64 = v.to(torch.complex128)
        magnitude = v64.abs() * (1.0 + eps.to(v.device) * self._fill_mag)
        step = math.pi / 6.0
        nominal = torch.round(torch.angle(v64) / step) * step
        filled = torch.polar(magnitude, nominal).to(v.dtype)
        return torch.where(filled_mask, filled, out)


def grid_current_range(grid: "Grid") -> Optional[float]:
    """The CT range [A] the current-channel noise floor refers to.

    One aggregate figure for the whole grid — the total in-service injection rating over
    the three-phase nominal voltage — because per-branch CT ratings are not modelled.
    Aggregate is conservative for lateral CTs (their real ranges are smaller, so their
    real floors are lower), keeping a noise study pessimistic rather than flattering.
    ``None`` when the grid carries no rated injections (the floor stays off).
    """
    from pgml.schemas.grid_schema import Generator, Load

    total = sum(
        abs(float(a.p_nom_w or 0.0))
        for a in grid.appliances
        if isinstance(a, (Load, Generator)) and a.in_service
    )
    if total <= 0.0:
        return None
    u_ln = max(
        float(n.u_rated_v)
        / (3.0**0.5 if len([p for p in n.phases if p.value != "n"]) >= 3 else 1.0)
        for n in grid.nodes
    )
    return total / (3.0 * u_ln)


__all__ = ["SensorNoise", "grid_current_range"]
