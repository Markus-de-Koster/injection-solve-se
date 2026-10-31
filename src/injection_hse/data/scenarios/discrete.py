"""Discrete dominant emitters for the randomized snapshot recipe.

A load whose harmonic emission comes from a few large devices (an EV charger, an inverter
heat pump, a drive) does not emit a continuum: each device is off, idle or in one of a few
operating states, and in each state it injects one fixed spectrum. Over a batch of snapshots
the load's injection is therefore a MIXTURE of a few distinct phasors, which a uniform
per-order ratio (one independent draw per order) cannot express, because the orders of one
device switch together.

:class:`DiscreteEmitterConfig` declares such devices for the loads of a randomized batch, and
:func:`apply_discrete_emitters` adds them to a batch pgml has already drawn. It is the
snapshot counterpart of the sequence composition (:mod:`injection_hse.data.scenarios.composition`):
the devices are :class:`~injection_hse.data.scenarios.DeviceClassSpec` entries, matched to loads by
the same rules (load id > ``consumer_type`` > fallback), emitting through the same load
dependence and capped at the same IEC 61000-3-2 envelope. Per scenario each device draws
one state from the stationary distribution of the class's ``states``, the snapshot
equivalent of one step of the sequence model.

Per covered load ``l`` and solved order ``h``, with ``P_bg`` the active power pgml drew for
the load and ``r_bg`` its drawn injection (the background):

    P_l = P_bg + sum_d P_d,       P_d = on_d * lam_d * P_rated,d
    c_h = r_bg,h * P_bg + sum_d on_d * m_h,d(lam_d) * lam_d * P_rated,d * exp(j * phi_h,d(lam_d))
    I_h / I_1 = c_h / P_l

so a device's harmonic current is its own ratio times its own power, the background keeps
its absolute current, and the sum is expressed in pgml's injection convention (a ratio to
the load's fundamental current, phase relative to it). Active power stands in for the
current, as in the composition: the node voltage cancels in the ratio. ``lam_d`` is the
state's ``power_fraction`` (a class without states draws it from ``loading_mean`` per
scenario), ``m_h,d`` is the rated ratio times ``lam ** gamma`` times the state's
``spectrum_scale``, and ``phi_h,d`` the rated angle shifted by the phase slope; the affine
emission floor multiplies the phasor (:func:`~injection_hse.data.scenarios.affine_emission_correction`).
The device's fundamental ``P_d`` and ``Q_d = P_d * tan(acos(pf))`` are added to the load's
operating point.

Draws are seeded from the batch's ``seed`` and the config's ``name`` on streams of their
own, plain pseudo-random rather than pgml's sampling cube, so declaring emitters leaves
every pgml draw unchanged. Off the autograd tape, ``float64``, CPU.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, Optional

import torch
from pgml.assembly import phase_voltage_magnitude
from pgml.scenarios import SampledScenarios, nominal_power

from .emission_caps import iec61000_3_2_cap
from pgml.schemas.grid_schema import Load, StaticSpectrum
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from .config import DeviceClassSpec

if TYPE_CHECKING:  # emission.py imports this module for the config type
    from .emission import EmissionLawScenarioConfig

__all__ = [
    "EmitterCount",
    "EmitterRule",
    "DiscreteEmitterConfig",
    "apply_discrete_emitters",
]

_F64 = torch.float64

# Stream offsets on top of the batch seed and the config name's CRC-32: one stream for the
# device draws (identity), one for the per-scenario state draws.
_DEVICE_STREAM = 0x5BD1E995
_STATE_STREAM = 0x1B873593

# Net-fundamental floor [W]: below it the load injects only its background.
_P_FLOOR = 1e-9

# Reference power for the vectorised IEC 61000-3-2 cap: large enough that no Class A/B
# fraction is clamped at 1, so the fraction at any other power scales as P_REF / p.
_P_REF = 1.0e9


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmitterCount(_Base):
    """How many devices of one class a matched load carries, and how big they are.

    ``count`` is an inclusive ``[min, max]`` range drawn per load. ``size_basis`` sets the
    device's rated power: ``"rated"`` draws the class's ``rated_power_w`` in watts;
    ``"nameplate"`` draws ``nameplate_share`` of the load's nameplate ``|p_nom_w|``;
    ``"larger"`` takes the larger of the two, so one recipe gives a single household its
    physical device and an aggregate of many households a device group sized relative to
    the aggregate. ``off_probability`` switches the device off in a scenario (no power, no
    emission) independently of its state, which gives a class without ``states`` an
    on/off mixture.
    """

    class_name: str
    count: tuple[int, int] = (1, 1)
    size_basis: Literal["rated", "nameplate", "larger"] = "rated"
    nameplate_share: Optional[tuple[float, float]] = None
    off_probability: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check(self) -> "EmitterCount":
        lo, hi = self.count
        if lo < 0 or hi < lo:
            raise ValueError(
                f"EmitterCount {self.class_name!r} count must be 0 <= min <= max."
            )
        if (self.size_basis == "rated") != (self.nameplate_share is None):
            raise ValueError(
                f"EmitterCount {self.class_name!r}: nameplate_share is required for "
                f"size_basis={self.size_basis!r} and only then."
            )
        if self.nameplate_share is not None:
            lo_s, hi_s = self.nameplate_share
            if lo_s < 0.0 or hi_s < lo_s:
                raise ValueError(
                    f"EmitterCount {self.class_name!r} nameplate_share must be "
                    "0 <= low <= high."
                )
        return self


class EmitterRule(_Base):
    """The emitters of one consumer character.

    Matches a load by explicit ``load_ids`` if given, else by ``consumer_type`` (the
    load's own tag); a rule with both unset is the fallback for every load no other rule
    claims, which is how a grid whose loads carry no tag (or other tags) is covered. A
    load matched by no rule gets no emitters.
    """

    consumer_type: Optional[str] = None
    load_ids: Optional[list[int]] = None
    emitters: list[EmitterCount] = Field(min_length=1)


class DiscreteEmitterConfig(_Base):
    """Large discrete devices added to the loads of a randomized batch.

    Set as ``EmissionLawScenarioConfig.discrete_emitters``; applied after pgml's draw and
    after the emission laws (:func:`apply_discrete_emitters`).

    - ``classes``: the device classes (:class:`~injection_hse.data.scenarios.DeviceClassSpec`). A
      snapshot reads ``sign`` (``+1`` only), ``rated_power_w``, ``power_factor``,
      ``harmonic_magnitude`` / ``harmonic_phase_deg`` (rated ratio of the device's own
      fundamental current and angle, per order), ``emission_class`` (the cap), ``gamma``,
      ``phase_slope_deg``, ``emission_floor`` / ``emission_floor_phase_deg`` (the load
      dependence), ``states`` (the mixture: each state's ``weight`` is its probability,
      ``power_fraction`` the device's loading, ``spectrum_scale`` a multiplier on its
      ratios; a state with ``spectrum_scale`` 0 is an emission-free idle state) and, for a
      class without states, ``loading_mean`` (the per-scenario loading range). The
      temporal fields (activity preset, dwell, jitter) belong to the sequence model and
      are not read.
    - ``rules``: which loads carry which devices (:class:`EmitterRule`).
    - ``persistence``: ``"fixed"`` draws every device's presence, rated power, ratios,
      angles and load-dependence parameters once for the dataset (a device keeps its
      identity, only its state changes between snapshots); ``"scenario"`` redraws them in
      every scenario.
    - ``max_injection_pu``: cap on a load's total harmonic ratio, as in the composition.
    - ``name``: prefix of the recorded samples and part of the stream seed.

    The emitters inject at the orders the config's own ``h_mag`` / ``h_phase`` specs
    cover, so restricting those specs to the solved orders restricts the emitters with
    them. Orders a class lists no ratio for stay at the background there.
    """

    name: str = "emitters"
    classes: list[DeviceClassSpec] = Field(min_length=1)
    rules: list[EmitterRule] = Field(min_length=1)
    persistence: Literal["fixed", "scenario"] = "fixed"
    max_injection_pu: float = Field(default=3.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "DiscreteEmitterConfig":
        names = [c.name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("DiscreteEmitterConfig class names must be unique.")
        by_name = {c.name: c for c in self.classes}
        for rule in self.rules:
            for entry in rule.emitters:
                cls = by_name.get(entry.class_name)
                if cls is None:
                    raise ValueError(
                        f"EmitterRule references unknown class {entry.class_name!r}."
                    )
                if cls.sign != 1:
                    raise ValueError(
                        f"discrete emitter class {cls.name!r} must consume (sign=1): a "
                        "generating device behind a load meter is not supported here."
                    )
        return self


def _member_caps(
    orders: list[int],
    emission_class: Optional[str],
    p_phase_w: Tensor,
    u_ln_v: float,
    power_factor: float,
) -> Tensor:
    """IEC 61000-3-2 emission fraction per order for a tensor of device powers.

    The vectorised form of :func:`injection_hse.data.scenarios.emission_caps.iec61000_3_2_cap` (the
    IEC table where it lists the order, its envelope elsewhere; same single-phase-equivalent
    convention, same clamp at ``1``): Class A and B limits are
    amperes, so the fraction scales as ``1 / p``; Class C and D fractions do not depend
    on the power. ``emission_class=None`` resolves per device as the composition does
    (Class D inside its 75-600 W per-phase window, else Class A).

    Parameters
    ----------
    orders:
        Harmonic orders ``n_ord``.
    emission_class:
        ``"A"`` / ``"B"`` / ``"C"`` / ``"D"``, or ``None`` for the automatic choice.
    p_phase_w:
        Per-phase device power [W], any shape ``S``, ``float64``.
    u_ln_v, power_factor:
        Terminal line-to-neutral voltage [V] and displacement power factor.

    Returns
    -------
    Tensor
        ``[*S, n_ord]`` fractions of the device's fundamental current in ``[0, 1]``.
    """
    p = p_phase_w.to(_F64).unsqueeze(-1)

    def table(cls: str) -> Tensor:
        def at(p_ref: float) -> Tensor:
            return torch.tensor(
                [
                    iec61000_3_2_cap(
                        o,
                        emission_class=cls,
                        p_w=p_ref,
                        u_ln_v=u_ln_v,
                        power_factor=power_factor,
                    )
                    for o in orders
                ],
                dtype=_F64,
            )

        # Per order, a cap is either an ampere limit (the fraction falls as 1 / p: Class A
        # and B, and the envelope at orders a class table does not list) or a fraction that
        # does not depend on the power (Class C and D table entries); two reference powers
        # tell them apart.
        ref, ref2 = at(_P_REF), at(2.0 * _P_REF)
        ampere = torch.isclose(ref, 2.0 * ref2, rtol=1e-9, atol=0.0) & (ref > 0)
        scaled = (ref * _P_REF / p.clamp(min=1e-12)).clamp(max=1.0)
        return torch.where(ampere, scaled, ref.expand_as(scaled))

    if emission_class is not None:
        return table(emission_class)
    in_window = (p >= 75.0) & (p <= 600.0)
    return torch.where(in_window, table("D"), table("A"))


def _stream(seed: int, name: str, offset: int) -> torch.Generator:
    """A generator for one of the config's streams, from the batch seed and its name."""
    value = (int(seed) + zlib.crc32(name.encode("utf-8")) + offset) & 0x7FFFFFFF
    return torch.Generator().manual_seed(value)


def _uniform(gen: torch.Generator, rng, shape) -> Tensor:
    lo, hi = float(rng[0]), float(rng[1])
    return lo + (hi - lo) * torch.rand(shape, generator=gen, dtype=_F64)


def _order_ranges(table: dict, orders: list[int]) -> tuple[Tensor, Tensor]:
    """Per-order ``(low, high)`` arrays of a ``{order: (low, high)}`` table (0 if absent)."""
    lo = torch.tensor([float(table.get(o, (0.0, 0.0))[0]) for o in orders], dtype=_F64)
    hi = torch.tensor([float(table.get(o, (0.0, 0.0))[1]) for o in orders], dtype=_F64)
    return lo, hi


def _load_p(entry: Optional[dict], nominal: float, b: int) -> Tensor:
    """A load's total active power ``[B]`` as drawn (its nameplate if nothing varied it)."""
    if entry is not None and "p_per_phase_w" in entry:
        return torch.stack(
            [torch.as_tensor(x, dtype=_F64).expand(b) for x in entry["p_per_phase_w"]]
        ).sum(0)
    if entry is not None and "p_w" in entry:
        return torch.as_tensor(entry["p_w"], dtype=_F64).reshape(-1).expand(b)
    return torch.full((b,), float(nominal), dtype=_F64)


def _add_power(entry: Optional[dict], dp: Tensor, dq: Tensor, nominal) -> dict:
    """A load's operating-point entry with ``dp`` / ``dq`` ``[B]`` added.

    A per-phase entry gets an equal share on each phase, a balanced one the total; a load
    without an entry gets its nameplate plus the addition.
    """
    entry = dict(entry or {})
    if "p_per_phase_w" in entry or "q_per_phase_var" in entry:
        n = len(entry.get("p_per_phase_w") or entry.get("q_per_phase_var"))
        if "p_per_phase_w" in entry:
            entry["p_per_phase_w"] = [
                torch.as_tensor(x) + (dp / n).to(torch.as_tensor(x).dtype)
                for x in entry["p_per_phase_w"]
            ]
        else:
            entry["p_per_phase_w"] = [
                torch.as_tensor(float(p)) + dp / n for p in nominal.p_pp
            ]
        if "q_per_phase_var" in entry:
            entry["q_per_phase_var"] = [
                torch.as_tensor(x) + (dq / n).to(torch.as_tensor(x).dtype)
                for x in entry["q_per_phase_var"]
            ]
        else:
            entry["q_per_phase_var"] = [
                torch.as_tensor(float(q)) + dq / n for q in nominal.q_pp
            ]
        return entry
    p0 = entry.get("p_w", float(nominal.p_total))
    q0 = entry.get("q_var", float(nominal.q_total))
    entry["p_w"] = torch.as_tensor(p0, dtype=_F64) + dp
    entry["q_var"] = torch.as_tensor(q0, dtype=_F64) + dq
    return entry


def _stored_component(load: Load, order: int) -> tuple[float, float]:
    """``(mag_pu, phase_deg)`` of a load's stored static spectrum at ``order`` (else 0)."""
    spec = getattr(load, "spectrum", None)
    if isinstance(spec, StaticSpectrum):
        for c in spec.spectrum.components:
            if c.order == order:
                return float(c.magnitude_pu), float(c.phase_deg)
    return 0.0, 0.0


def apply_discrete_emitters(
    grid, sampled: SampledScenarios, config: "EmissionLawScenarioConfig"
) -> SampledScenarios:
    """Add the config's discrete emitters to a sampled batch (see the module doc).

    Parameters
    ----------
    grid:
        The grid the batch was drawn for; the covered loads are its in-service
        :class:`~pgml.schemas.grid_schema.Load` appliances with a matching rule.
    sampled:
        The batch after pgml's draw and the emission laws.
    config:
        The :class:`~injection_hse.data.scenarios.EmissionLawScenarioConfig` whose
        ``discrete_emitters`` is set (``None`` returns ``sampled`` unchanged).

    Returns
    -------
    pgml.scenarios.SampledScenarios
        A new batch (``sampled`` is not modified): every covered load's operating point
        carries its devices' power, its harmonic injection the summed spectrum at the
        config's injected orders, and the realized ``<spec>_mag`` / ``<spec>_phase``
        records of the ``h_mag`` specs are rewritten to what is injected. Added records
        (``name`` = ``discrete_emitters.name``, ``n_slot`` = the largest device count a
        load can carry):

        - ``<name>_mag`` / ``<name>_phase`` ``[B, n_load, n_ord]``: the realized
          injection of each covered load (ratio to its fundamental current, degrees);
        - ``<name>_state`` ``[B, n_load, n_slot]`` int64: each device's state index, ``-1``
          where the device is absent or off;
        - ``<name>_power_w`` ``[B, n_load, n_slot]``: each device's active power [W];
        - shared: ``<name>_device_ids`` ``[n_load]``, ``<name>_orders`` ``[n_ord]``,
          ``<name>_class_index`` ``[n_load, n_slot]`` (index into ``classes``, ``-1``
          for an unused slot).
    """
    from .composition import _rule_for_load
    from .emission import (
        _column,
        _realized,
        affine_emission_correction,
        phase_slope_shift,
    )

    em = config.discrete_emitters
    if em is None:
        return sampled
    orders = config.injected_orders()
    b = int(sampled.n_samples)
    loads = [
        a
        for a in grid.appliances
        if isinstance(a, Load)
        and a.in_service
        and _rule_for_load(em.rules, a) is not None
    ]
    if not orders or not loads:
        return sampled
    n_ord = len(orders)
    class_index = {c.name: i for i, c in enumerate(em.classes)}
    nominal = nominal_power(grid)
    nodes = {n.id: n for n in grid.nodes}
    dev_gen = _stream(config.seed, em.name, _DEVICE_STREAM)
    state_gen = _stream(config.seed, em.name, _STATE_STREAM)
    r = 1 if em.persistence == "fixed" else b  # rows of the identity draws

    # Device identity: presence, rated power, rated ratios and angles, load dependence.
    slots: list[tuple[int, int, dict]] = []  # (load index, class index, draws)
    for li, load in enumerate(loads):
        rule = _rule_for_load(em.rules, load)
        node = nodes[load.node]
        u_ln = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
        n_phases = max(1, len(load.phases))
        for entry in rule.emitters:
            cls = em.classes[class_index[entry.class_name]]
            n = torch.randint(
                entry.count[0], entry.count[1] + 1, (r,), generator=dev_gen
            )
            for k in range(entry.count[1]):
                rated = _uniform(dev_gen, cls.rated_power_w, (r,))
                if entry.size_basis != "rated":
                    share = _uniform(dev_gen, entry.nameplate_share, (r,))
                    share_w = share * abs(float(load.p_nom_w))
                    rated = (
                        share_w
                        if entry.size_basis == "nameplate"
                        else torch.maximum(rated, share_w)
                    )
                m_lo, m_hi = _order_ranges(cls.harmonic_magnitude, orders)
                a_lo, a_hi = _order_ranges(cls.harmonic_phase_deg, orders)
                u = torch.rand((r, n_ord), generator=dev_gen, dtype=_F64)
                mag = m_lo + (m_hi - m_lo) * u
                ang = a_lo + (a_hi - a_lo) * torch.rand(
                    (r, n_ord), generator=dev_gen, dtype=_F64
                )
                cap = _member_caps(
                    orders, cls.emission_class, rated / n_phases, u_ln, cls.power_factor
                )
                slots.append(
                    (
                        li,
                        class_index[cls.name],
                        {
                            "present": (k < n).to(_F64),
                            "rated": rated,
                            "mag": torch.minimum(mag, cap),
                            "ang": ang,
                            "gamma": _uniform(dev_gen, cls.gamma, (r, n_ord)),
                            "slope": _uniform(dev_gen, cls.phase_slope_deg, (r, n_ord)),
                            "floor": _uniform(dev_gen, cls.emission_floor, (r, n_ord)),
                            "floor_phase": _uniform(
                                dev_gen, cls.emission_floor_phase_deg, (r, n_ord)
                            ),
                            "off_p": entry.off_probability,
                        },
                    )
                )

    n_load = len(loads)
    n_slot = max(
        sum(e.count[1] for e in _rule_for_load(em.rules, ld).emitters) for ld in loads
    )
    state_rec = torch.full((b, n_load, n_slot), -1, dtype=torch.long)
    power_rec = torch.zeros((b, n_load, n_slot), dtype=_F64)
    class_rec = torch.full((n_load, n_slot), -1, dtype=torch.long)
    p_dev = torch.zeros((b, n_load), dtype=_F64)
    q_dev = torch.zeros((b, n_load), dtype=_F64)
    c_dev = torch.zeros((b, n_load, n_ord), dtype=torch.complex128)
    used = [0] * n_load

    # Per scenario: on/off and the operating state of every device.
    for li, ci, d in slots:
        cls = em.classes[ci]
        if cls.states:
            weights = torch.tensor([s.weight for s in cls.states], dtype=_F64)
            state = torch.multinomial(
                weights / weights.sum(), b, replacement=True, generator=state_gen
            )
            pfrac = torch.tensor([s.power_fraction for s in cls.states], dtype=_F64)[
                state
            ]
            sscale = torch.tensor([s.spectrum_scale for s in cls.states], dtype=_F64)[
                state
            ]
        else:
            state = torch.zeros(b, dtype=torch.long)
            pfrac = _uniform(state_gen, cls.loading_mean, (b,))
            sscale = torch.ones(b, dtype=_F64)
        on = d["present"].expand(b) * (
            torch.rand(b, generator=state_gen, dtype=_F64) >= d["off_p"]
        ).to(_F64)
        lam = pfrac.unsqueeze(-1)  # [B, 1]
        p = on * pfrac * d["rated"].expand(b)
        ratio = d["mag"] * lam.clamp(min=1e-9) ** d["gamma"] * sscale.unsqueeze(-1)
        angle = d["ang"] + phase_slope_shift(d["slope"], lam)
        phasor = (
            ratio
            * p.unsqueeze(-1)
            * torch.exp(1j * torch.deg2rad(angle))
            * affine_emission_correction(lam, d["floor"], d["floor_phase"])
        )
        slot = used[li]
        used[li] += 1
        class_rec[li, slot] = ci
        state_rec[:, li, slot] = torch.where(on > 0, state, torch.full_like(state, -1))
        power_rec[:, li, slot] = p
        p_dev[:, li] += p
        q_dev[:, li] += p * math.tan(math.acos(cls.power_factor))
        c_dev[:, li] += phasor

    # Sum with the background per load, in the injection convention.
    operating_point = dict(sampled.operating_point)
    injection = {cid: dict(per) for cid, per in sampled.harmonic_injection.items()}
    like = torch.zeros((), dtype=_F64)
    mag_rec = torch.zeros((b, n_load, n_ord), dtype=_F64)
    phase_rec = torch.zeros((b, n_load, n_ord), dtype=_F64)
    for li, load in enumerate(loads):
        entry = operating_point.get(load.id)
        p_bg = _load_p(entry, nominal[load.id].p_total, b)
        p_tot = p_bg + p_dev[:, li]
        dev = injection.setdefault(load.id, {})
        for k, order in enumerate(orders):
            mag0, ph0 = dev.get(order, _stored_component(load, order))
            mag0, ph0 = _column(mag0, b, like), _column(ph0, b, like)
            c = mag0 * p_bg * torch.exp(1j * torch.deg2rad(ph0)) + c_dev[:, li, k]
            active = p_tot.abs() > _P_FLOOR
            mag = torch.where(
                active,
                (c.abs() / p_tot.abs().clamp(min=_P_FLOOR)).clamp(
                    max=em.max_injection_pu
                ),
                mag0,
            )
            phase = torch.where(
                active,
                torch.rad2deg(torch.angle(c) - torch.angle(p_tot.to(torch.complex128))),
                ph0,
            )
            dev[order] = (mag, phase)
            mag_rec[:, li, k], phase_rec[:, li, k] = mag, phase
        operating_point[load.id] = _add_power(
            entry, p_dev[:, li], q_dev[:, li], nominal[load.id]
        )

    nm = em.name
    samples: dict = {}
    h_mag = {s.name: s for s in config.parameters if s.field == "h_mag"}
    for key, value in sampled.samples.items():  # realized records of the h_mag specs
        base, part = key, None
        if key.endswith("_mag"):
            base, part = key[: -len("_mag")], 0
        elif key.endswith("_phase"):
            base, part = key[: -len("_phase")], 1
        ids = sampled.shared_samples.get(f"{base}_device_ids")
        if part is not None and base in h_mag and ids is not None:
            samples[key] = _realized(
                injection, ids.tolist(), h_mag[base].orders, part, b, like
            )
        else:
            samples[key] = value
    samples[f"{nm}_mag"] = mag_rec
    samples[f"{nm}_phase"] = phase_rec
    samples[f"{nm}_state"] = state_rec
    samples[f"{nm}_power_w"] = power_rec
    shared = dict(sampled.shared_samples)
    shared[f"{nm}_device_ids"] = torch.tensor([ld.id for ld in loads], dtype=torch.long)
    shared[f"{nm}_orders"] = torch.tensor(orders, dtype=torch.long)
    shared[f"{nm}_class_index"] = class_rec
    return replace(
        sampled,
        operating_point=operating_point,
        harmonic_injection=injection,
        samples=samples,
        shared_samples=shared,
    )
