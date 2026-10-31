"""Statistical device-class composition of aggregated loads.

Turns an aggregated LV load (a household, an office, ...) into a sum of statistical
member devices drawn from a device-class library. Each member's per-step ACTIVITY drives
BOTH the fundamental power it draws AND the harmonic current it injects, through a
consistent load-to-spectrum mapping (magnitude ``mag_h(lam) = mag_h_rated * lam**gamma_h``
and phase ``ang_h(lam) = ang_h0 + s_h * (lam - 1)``). Summing the members' complex
currents per load produces an aggregate spectrum in which device diversity and phase
cancellation EMERGE rather than being imposed — so a state estimator can learn to
attribute an observed aggregate spectrum to a device mix (and, ideally, an error source).

The composition is engaged from :class:`CoherentSpectrumConfig` via its
``composition`` field; :func:`sample_device_composition` produces the per-load fundamental
operating point (``[B, T]``), the harmonic injection (``{id: {order: (mag[B, T],
phase[B, T])}}``), the class-attribution ground-truth samples and the realized aggregate
spectrum itself (``<name>_composed_mag`` / ``<name>_composed_phase``, so a written dataset
records what was injected). It SUPERSEDES the mode-bank fingerprint for the loads it
covers.

Generation model (per covered load, off the autograd tape, ``float64``, CPU, seeded)
------------------------------------------------------------------------------------
1. ROSTER (drawn once, persisted): a set of member devices, each with a rated power, a
   sign, per-order rated harmonic magnitude / phase, per-order load-dependence exponent
   / slope, a mean loading, activity stickiness and (multi-state classes) a small set of
   power/spectrum states. With ``scale_to_nominal`` the share-weighted installed capacity
   is rescaled to the load's ``p_nom_w``.
2. ACTIVITY ``a_d(t)``: a diurnal availability rate (a class preset, reusing the daily
   shapes of :mod:`injection_hse.data.scenarios.profiles`) scaled by a per-scenario shared latent
   (``behavioral`` for consumption, ``cloud`` for PV). A switching appliance realises it
   as an on/off Markov chain whose stationary occupancy equals the rate; a continuously
   modulated device (PV, base load) uses the rate directly.
3. LOADING ``lam_d(t) in [lam_min, 1]``: an AR(1)-smoothed fluctuation around the member's
   mean loading (single-state) or the current state's power fraction (multi-state).
4. CONTRIBUTIONS: power ``P_d = a_d * lam_d * P_rated * sign`` and a harmonic phasor
   current ``I_h,d = sign * ratio_h(lam_d) * (a_d * lam_d * P_rated) * exp(j*ang_h(lam_d))``
   (the device fundamental-current magnitude enters both numerator and denominator of the
   per-unit injection, so the node voltage cancels). Per load, ``P_agg = sum_d P_d`` (may
   go net-negative under PV — the Load then injects) and the summed harmonic phasor is
   normalised to the injection-dict convention: magnitude as a FRACTION of the aggregate
   fundamental current (capped at ``max_injection_pu`` near a net-zero fundamental), phase
   relative to the aggregate fundamental direction.

All sampling is off the autograd tape (plain ``float64`` draws with no ``requires_grad``);
the produced operating point / injection feed the solver, whose gradients flow to the grid
parameters. The Markov and AR(1) recurrences loop over the step axis (inherently
sequential); everything else is vectorised over ``[B, member, order, T]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor

from pgml.assembly import phase_voltage_magnitude
from pgml.schemas.grid_schema import Grid, Load

from pgml.scenarios import ar1_noise
from .emission import affine_emission_correction
from .emission_caps import iec61000_3_2_cap
from .config import (
    ActivityCalendar,
    CoherentSpectrumConfig,
    CompositionConfig,
    DeviceClassSpec,
    LoadProfileConfig,
)
from .profiles import (
    _HOUR_REF,
    _batch_time_axis,
    _pv_bell,
    _raw_daily,
    _weekly_factor,
)

_F64 = torch.float64

#: A load-matching rule: anything with ``load_ids`` and ``consumer_type``.
_Rule = TypeVar("_Rule")

# Derived-stream offsets, distinct from the fingerprint (``config.seed``), the
# operating-point cube and the profile: enabling a composition never disturbs the
# fingerprint / profile draws (a composition-free run stays byte-identical).
_COMPOSITION_SEED_OFFSET = 0xC2B2AE35  # 3266489909
_ROSTER_SEED_OFFSET = 0x27D4EB2F  # 668265261

# Default PV daylight-window parameters for the solar activity bell (a plain default;
# the composition has no LoadProfileConfig of its own).
_PV_DEFAULTS = LoadProfileConfig()

# Net-fundamental floor [W]: below it a load is treated as inactive (no injection),
# above it the max_injection_pu cap bounds the residual-THD blow-up.
_F_FLOOR = 1e-9


# =============================================================================
# Load-to-spectrum laws (small, directly unit-tested)
# =============================================================================
def _mag_law(mag_rated: Tensor, lam: Tensor, gamma: Tensor, sscale: Tensor) -> Tensor:
    """Load-dependent harmonic magnitude ``mag_rated * lam**gamma * sscale``.

    ``lam`` is clamped to ``> 0`` by the caller, so ``lam**gamma`` is finite for any
    ``gamma`` (a negative exponent grows the FRACTION at low load while the absolute
    current — scaled elsewhere by the loading — still falls)."""
    return mag_rated * lam.clamp(min=1e-9) ** gamma * sscale


def _phase_law(ang0: Tensor, slope: Tensor, lam: Tensor) -> Tensor:
    """Load-dependent harmonic phase ``ang0 + slope * (lam - 1)`` [deg]."""
    return ang0 + slope * (lam - 1.0)


def _emission_affine(lam: Tensor, floor: Tensor, delta_deg: Tensor) -> Tensor:
    """The affine emission law as a complex correction on the proportional phasor.

    The one definition lives in :func:`injection_hse.data.scenarios.affine_emission_correction`
    (shared with the randomized recipe); kept under this name for the composition path.
    """
    return affine_emission_correction(lam, floor, delta_deg)


def _urange_opt(gen: torch.Generator, rng: tuple, shape=()) -> Tensor:
    """``_urange``, but a zero-width range at zero draws nothing and consumes no RNG.

    Keeps the random stream — and therefore every existing dataset — identical unless the
    optional parameter this draws is actually configured.
    """
    if float(rng[0]) == 0.0 and float(rng[1]) == 0.0:
        return torch.zeros(shape, dtype=_F64)
    return _urange(gen, rng, shape)


def _aggregate_injection(
    c_agg: Tensor, f_agg: Tensor, max_injection_pu: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalise summed member currents to the injection-dict convention.

    Parameters
    ----------
    c_agg:
        Complex aggregate harmonic phasor per ``[..., order, ...]`` (member-summed).
    f_agg:
        Complex/real aggregate FUNDAMENTAL phasor per ``[...]`` (member-summed), one
        order axis narrower than ``c_agg``.
    max_injection_pu:
        Cap on the relative magnitude (bounds the residual-THD blow-up near a net-zero
        fundamental).

    Returns
    -------
    (mag, phase_deg, cap_binding)
        ``mag`` = ``|c_agg| / |f_agg|`` capped at ``max_injection_pu`` (0 where the net
        fundamental is below :data:`_F_FLOOR` — nothing active / exact cancellation);
        ``phase_deg`` = ``arg(c_agg) - arg(f_agg)`` [deg]; ``cap_binding`` is a float
        ``0/1`` flag marking where the cap clipped the magnitude.
    """
    f_mag = f_agg.abs()
    c_mag = c_agg.abs()
    active = (f_mag > _F_FLOOR).to(_F64)  # broadcast over the order axis below
    raw = c_mag / f_mag.clamp(min=_F_FLOOR).unsqueeze(-2)
    capped = raw.clamp(max=max_injection_pu)
    active_o = active.unsqueeze(-2)
    mag = capped * active_o
    phase = torch.rad2deg(torch.angle(c_agg) - torch.angle(f_agg).unsqueeze(-2))
    phase = phase * active_o
    cap_binding = ((raw > max_injection_pu).to(_F64)) * active_o
    return mag, phase, cap_binding


# =============================================================================
# Seeded RNG helpers (off the autograd tape)
# =============================================================================
def _markov_onoff(p_on: Tensor, stickiness: Tensor, gen: torch.Generator) -> Tensor:
    """Two-state on/off Markov path ``[B, M, T]`` (float ``0/1``) with target occupancy.

    ``p_on`` is the per-step target occupancy ``[B, M, T]`` and ``stickiness`` ``[M]`` the
    persistence: staying-on has probability ``s + (1-s)*p`` and turning-on ``(1-s)*p``, so
    the quasi-static stationary occupancy equals ``p`` while ``s`` sets the dwell."""
    b, m, t = p_on.shape
    s = stickiness.reshape(1, m)
    out = torch.empty((b, m, t), dtype=_F64)
    state = (torch.rand((b, m), generator=gen, dtype=_F64) < p_on[..., 0]).to(_F64)
    out[..., 0] = state
    for step in range(1, t):
        p = p_on[..., step]
        stay_on = s + (1.0 - s) * p
        turn_on = (1.0 - s) * p
        p_next = torch.where(state > 0.5, stay_on, turn_on)
        state = (torch.rand((b, m), generator=gen, dtype=_F64) < p_next).to(_F64)
        out[..., step] = state
    return out


def _cat_sample(cum_pi: Tensor, u: Tensor) -> Tensor:
    """Categorical sample per ``[B, M]`` from per-member cumulative weights ``[M, S]``."""
    idx = (u.unsqueeze(-1) >= cum_pi.unsqueeze(0)).sum(dim=-1)
    return idx.clamp(max=cum_pi.shape[-1] - 1)


def _markov_states(
    cum_pi: Tensor, dwell: Tensor, b: int, t: int, gen: torch.Generator
) -> Tensor:
    """Multi-state Markov path ``[B, M, T]`` (long) with stationary weights ``pi``.

    Each step stays with probability ``dwell`` (per member) else resamples from ``pi``
    (built from the state weights), so the stationary distribution is ``pi``. A
    single-state member (``cum_pi`` = ``[1, 0, ...]``) stays at index 0."""
    m = cum_pi.shape[0]
    d = dwell.reshape(1, m)
    path = torch.empty((b, m, t), dtype=torch.long)
    cur = _cat_sample(cum_pi, torch.rand((b, m), generator=gen, dtype=_F64))
    path[..., 0] = cur
    for step in range(1, t):
        stay = torch.rand((b, m), generator=gen, dtype=_F64) < d
        resample = _cat_sample(cum_pi, torch.rand((b, m), generator=gen, dtype=_F64))
        cur = torch.where(stay, cur, resample)
        path[..., step] = cur
    return path


def _activity_rate(preset: str, hour: Tensor, doy: Tensor) -> Tensor:
    """Diurnal availability rate in ``[0, 1]`` for an activity preset, shaped like ``hour``.

    Reuses the daily shapes of :mod:`injection_hse.data.scenarios.profiles`: ``"pv"`` is the solar
    bell (zero at night); ``"flat"`` is constant 1; every other preset normalises its
    raw daily shape by its own peak so occupancy-driven presets keep a floor while
    presets that vanish off-peak (office, EV, restaurant gaps) drop to ~0."""
    if preset == "flat":
        return torch.ones_like(hour)
    if preset == "pv":
        return _pv_bell(hour, doy, _PV_DEFAULTS).clamp(min=0.0)
    ref = _raw_daily(preset, _HOUR_REF)
    peak = ref.max().clamp(min=1e-12)
    return (_raw_daily(preset, hour) / peak).clamp(0.0, 1.0)


def _calendar_factor(
    preset: str, doy: Tensor, dow: Tensor, calendar: ActivityCalendar
) -> Tensor:
    """Weekly x seasonal multiplier of a preset's activity rate, shaped like ``doy``.

    ``pv`` follows the irradiance season alone; every consumption preset follows the
    consumption season, and all but ``"flat"`` (heating, constant base load) also the
    weekend dip of :class:`~injection_hse.data.scenarios.ActivityCalendar`.
    """
    if preset == "pv":
        return 1.0 + calendar.pv_seasonal_amplitude * torch.cos(
            2.0 * math.pi * (doy - calendar.pv_seasonal_peak_doy) / 365.25
        )
    amplitude = calendar.preset_seasonal_amplitude.get(
        preset, calendar.seasonal_amplitude
    )
    seasonal = 1.0 + amplitude * torch.cos(
        2.0 * math.pi * (doy - calendar.seasonal_peak_doy) / 365.25
    )
    if preset == "flat":
        return seasonal
    return seasonal * _weekly_factor(dow, calendar.weekend_contrast)


# =============================================================================
# Roster (drawn once per covered load, persisted)
# =============================================================================
@dataclass(frozen=True)
class _Roster:
    """Flattened member arrays across all covered loads (member axis length ``M``)."""

    load_idx: Tensor  # [M] long: index into agg_ids
    class_idx: Tensor  # [M] long: index into class_names
    p_rated: Tensor  # [M]
    sign: Tensor  # [M] +-1
    tan_phi: Tensor  # [M]
    mag_rated: Tensor  # [M, n_ord]
    ang0: Tensor  # [M, n_ord]
    gamma: Tensor  # [M, n_ord]
    slope: Tensor  # [M, n_ord]
    emission_floor: Tensor  # [M, n_ord]
    emission_floor_phase: Tensor  # [M, n_ord] deg
    lam_min: Tensor  # [M]
    loading_jitter: Tensor  # [M]
    loading_rho: Tensor  # [M]
    onoff_stickiness: Tensor  # [M]
    discrete: Tensor  # [M] bool
    use_cloud: Tensor  # [M] bool (PV -> cloud latent, else behavioral)
    preset: list  # [M] activity preset strings
    state_pfrac: Tensor  # [M, S]
    state_sscale: Tensor  # [M, S]
    state_cum_pi: Tensor  # [M, S]
    n_states: Tensor  # [M] long
    state_dwell: Tensor  # [M]
    agg_ids: list  # [n_agg] covered load ids
    class_names: list  # [n_class]
    roster_p_rated: Tensor  # [n_agg, n_class, max_count] padded sidecar


def _rule_for_load(rules: Sequence[_Rule], load: Load) -> _Rule | None:
    """The rule for a load: id override > consumer_type > fallback.

    ``rules`` is any list of rules with ``load_ids`` and ``consumer_type`` (the
    composition's :class:`ConsumerComposition` and the discrete emitters'
    :class:`~injection_hse.data.scenarios.EmitterRule` match the same way)."""
    for rule in rules:
        if rule.load_ids is not None and load.id in rule.load_ids:
            return rule
    ct = getattr(load.consumer_type, "value", load.consumer_type)
    for rule in rules:
        if rule.load_ids is not None:
            continue
        if rule.consumer_type is not None and rule.consumer_type == ct:
            return rule
    for rule in rules:
        if rule.load_ids is None and rule.consumer_type is None:
            return rule
    return None


def resolve_composed_ids(grid: Grid, comp: CompositionConfig) -> list[int]:
    """Ids of the loads the composition covers (selector ∩ loads with a matching rule).

    Loads matched by ``comp.selector`` (default: every in-service Load) but with no
    matching :class:`ConsumerComposition` rule are NOT composed — they fall through to
    the fingerprint machinery."""
    if comp.selector is not None:
        ids = comp.selector.resolve(grid)
        by_id = {a.id: a for a in grid.appliances}
        loads = [by_id[i] for i in ids if isinstance(by_id.get(i), Load)]
    else:
        loads = [a for a in grid.appliances if isinstance(a, Load) and a.in_service]
    return [ld.id for ld in loads if _rule_for_load(comp.compositions, ld) is not None]


def _urange(gen: torch.Generator, rng: tuple, shape=()) -> Tensor:
    """Uniform draw in ``[low, high]`` (0-d for ``shape=()``)."""
    lo, hi = float(rng[0]), float(rng[1])
    return lo + (hi - lo) * torch.rand(shape, generator=gen, dtype=_F64)


def _order_range_arrays(
    cls: DeviceClassSpec, orders: list, key: str
) -> tuple[Tensor, Tensor]:
    """Per-order ``(low, high)`` arrays for a magnitude/phase range dict (0 if absent)."""
    table = getattr(cls, key)
    lo = torch.zeros(len(orders), dtype=_F64)
    hi = torch.zeros(len(orders), dtype=_F64)
    for o, order in enumerate(orders):
        if order in table:
            lo[o], hi[o] = float(table[order][0]), float(table[order][1])
    return lo, hi


def _build_roster(
    grid: Grid, comp: CompositionConfig, composed_ids: list, orders: list, seed: int
) -> _Roster:
    """Draw the per-load device roster (Python loop over members; off-tape, once)."""
    gen = torch.Generator().manual_seed(int(seed))
    by_id = {a.id: a for a in grid.appliances}
    class_by_name = {c.name: c for c in comp.classes}
    class_names = comp.class_names()
    class_index = {n: i for i, n in enumerate(class_names)}
    n_class, n_ord = len(class_names), len(orders)

    # Per-class per-order range arrays (drawn per member below).
    mag_lohi = {
        c.name: _order_range_arrays(c, orders, "harmonic_magnitude")
        for c in comp.classes
    }
    ang_lohi = {
        c.name: _order_range_arrays(c, orders, "harmonic_phase_deg")
        for c in comp.classes
    }
    max_states = max((len(c.states) for c in comp.classes), default=0)
    n_state_slots = max(1, max_states)

    load_idx, class_idx = [], []
    p_rated, sign, tan_phi = [], [], []
    mag_rated, ang0, gamma, slope = [], [], [], []
    e_floor, e_floor_phase = [], []
    lam_min, ljit, lrho, stick = [], [], [], []
    discrete, use_cloud, preset = [], [], []
    state_pfrac, state_sscale, state_wpad, n_states, sdwell = [], [], [], [], []
    # sidecar: (agg_idx, class_idx) -> [p_rated, ...]
    sidecar: dict = {}

    nodes_by_id = {n.id: n for n in grid.nodes}
    for a_idx, cid in enumerate(composed_ids):
        load = by_id[cid]
        rule = _rule_for_load(comp.compositions, load)
        p_nom = abs(float(load.p_nom_w))
        node = nodes_by_id[load.node]
        u_ln = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
        n_load_phases = max(1, len(load.phases))
        members: list = []  # (class_idx, cls, share, p_raw_tensor)
        weighted_total = torch.zeros((), dtype=_F64)
        for cc in rule.classes:
            cls = class_by_name[cc.class_name]
            ci = class_index[cc.class_name]
            n = int(
                torch.randint(cc.count[0], cc.count[1] + 1, (), generator=gen).item()
            )
            for _ in range(n):
                p_raw = _urange(gen, cls.rated_power_w)
                # per-member draws (fixed order -> deterministic)
                ml, mh = mag_lohi[cls.name]
                al, ah = ang_lohi[cls.name]
                mrat = ml + (mh - ml) * torch.rand((n_ord,), generator=gen, dtype=_F64)
                a0 = al + (ah - al) * torch.rand((n_ord,), generator=gen, dtype=_F64)
                gam = _urange(gen, cls.gamma, (n_ord,))
                slp = _urange(gen, cls.phase_slope_deg, (n_ord,))
                efl = _urange_opt(gen, cls.emission_floor, (n_ord,))
                efp = _urange_opt(gen, cls.emission_floor_phase_deg, (n_ord,))
                lam_mean = _urange(gen, cls.loading_mean)
                st = _urange(gen, cls.on_off_dwell)
                sd = _urange(gen, cls.state_dwell)
                members.append((ci, cls, cc.power_share, p_raw))
                weighted_total = weighted_total + cc.power_share * p_raw

                load_idx.append(a_idx)
                class_idx.append(ci)
                sign.append(float(cls.sign))
                tan_phi.append(math.tan(math.acos(cls.power_factor)))
                mag_rated.append(mrat)
                ang0.append(a0)
                gamma.append(gam)
                slope.append(slp)
                e_floor.append(efl)
                e_floor_phase.append(efp)
                lam_min.append(cls.loading_min)
                ljit.append(cls.loading_jitter)
                lrho.append(cls.loading_rho)
                stick.append(st)
                discrete.append(bool(cls.discrete_activity))
                use_cloud.append(cls.activity_preset == "pv")
                preset.append(cls.activity_preset)
                sdwell.append(sd)
                # padded states
                pf = torch.zeros(n_state_slots, dtype=_F64)
                ss = torch.ones(n_state_slots, dtype=_F64)
                wp = torch.zeros(n_state_slots, dtype=_F64)
                if cls.states:
                    for k, state in enumerate(cls.states):
                        pf[k] = state.power_fraction
                        ss[k] = state.spectrum_scale
                        wp[k] = state.weight
                    n_states.append(len(cls.states))
                else:  # single-state: state 0 carries the member's mean loading
                    pf[0] = lam_mean
                    wp[0] = 1.0
                    n_states.append(1)
                state_pfrac.append(pf)
                state_sscale.append(ss)
                state_wpad.append(wp)

        alpha = (
            p_nom / weighted_total.clamp(min=_F_FLOOR)
            if (comp.scale_to_nominal and p_nom > 0.0)
            else torch.ones((), dtype=_F64)
        )
        # Per-member IEC 61000-3-2 emission cap, at the EFFECTIVE rated power (after
        # scale_to_nominal): a roster's aggregate can only emit what its individual
        # appliances are permitted to inject, keeping the composed spectra inside the
        # same physical envelope the randomized h_mag sampling references. Applied
        # here (not at draw time) because Class A/B absolute limits need the scaled
        # member power. Single-phase-equivalent power (member power / load phases),
        # matching ``iec61000_3_2_caps``; every order >= 2 has a cap (the standard's
        # envelope where its table lists none, :mod:`.emission_caps`).
        member_start = len(mag_rated) - len(members)
        for j, (ci, cls, share, p_raw) in enumerate(members):
            pr = alpha * share * p_raw
            p_rated.append(pr)
            sidecar.setdefault((a_idx, ci), []).append(pr)
            if bool((mag_rated[member_start + j] > 0).any()):
                p_phase = float(pr) / n_load_phases
                cls_eff = cls.emission_class or (
                    "D" if 75.0 <= p_phase <= 600.0 else "A"
                )
                cap = torch.tensor(
                    [
                        iec61000_3_2_cap(
                            o,
                            emission_class=cls_eff,
                            p_w=p_phase,
                            u_ln_v=u_ln,
                            power_factor=cls.power_factor,
                        )
                        for o in orders
                    ],
                    dtype=_F64,
                )
                mag_rated[member_start + j] = torch.minimum(
                    mag_rated[member_start + j], cap
                )

    m = len(load_idx)
    n_agg = len(composed_ids)
    if m == 0:
        empty = torch.zeros((0,), dtype=_F64)
        return _Roster(
            load_idx=torch.zeros((0,), dtype=torch.long),
            class_idx=torch.zeros((0,), dtype=torch.long),
            p_rated=empty,
            sign=empty,
            tan_phi=empty,
            mag_rated=torch.zeros((0, n_ord), dtype=_F64),
            ang0=torch.zeros((0, n_ord), dtype=_F64),
            gamma=torch.zeros((0, n_ord), dtype=_F64),
            slope=torch.zeros((0, n_ord), dtype=_F64),
            emission_floor=torch.zeros((0, n_ord), dtype=_F64),
            emission_floor_phase=torch.zeros((0, n_ord), dtype=_F64),
            lam_min=empty,
            loading_jitter=empty,
            loading_rho=empty,
            onoff_stickiness=empty,
            discrete=torch.zeros((0,), dtype=torch.bool),
            use_cloud=torch.zeros((0,), dtype=torch.bool),
            preset=[],
            state_pfrac=torch.zeros((0, n_state_slots), dtype=_F64),
            state_sscale=torch.ones((0, n_state_slots), dtype=_F64),
            state_cum_pi=torch.zeros((0, n_state_slots), dtype=_F64),
            n_states=torch.zeros((0,), dtype=torch.long),
            state_dwell=empty,
            agg_ids=list(composed_ids),
            class_names=class_names,
            roster_p_rated=torch.zeros((n_agg, n_class, 1), dtype=_F64),
        )

    wpad = torch.stack(state_wpad)  # [M, S]
    cum_pi = torch.cumsum(wpad / wpad.sum(dim=1, keepdim=True).clamp(min=1e-12), dim=1)

    max_count = max((len(v) for v in sidecar.values()), default=1)
    roster = torch.zeros((n_agg, n_class, max_count), dtype=_F64)
    for (a_idx, ci), vals in sidecar.items():
        for k, pr in enumerate(vals):
            roster[a_idx, ci, k] = pr

    return _Roster(
        load_idx=torch.tensor(load_idx, dtype=torch.long),
        class_idx=torch.tensor(class_idx, dtype=torch.long),
        p_rated=torch.stack(p_rated),
        sign=torch.tensor(sign, dtype=_F64),
        tan_phi=torch.tensor(tan_phi, dtype=_F64),
        mag_rated=torch.stack(mag_rated),
        ang0=torch.stack(ang0),
        gamma=torch.stack(gamma),
        slope=torch.stack(slope),
        emission_floor=torch.stack(e_floor),
        emission_floor_phase=torch.stack(e_floor_phase),
        lam_min=torch.tensor(lam_min, dtype=_F64),
        loading_jitter=torch.tensor(ljit, dtype=_F64),
        loading_rho=torch.tensor(lrho, dtype=_F64),
        onoff_stickiness=torch.stack(stick),
        discrete=torch.tensor(discrete, dtype=torch.bool),
        use_cloud=torch.tensor(use_cloud, dtype=torch.bool),
        preset=preset,
        state_pfrac=torch.stack(state_pfrac),
        state_sscale=torch.stack(state_sscale),
        state_cum_pi=cum_pi,
        n_states=torch.tensor(n_states, dtype=torch.long),
        state_dwell=torch.stack(sdwell),
        agg_ids=list(composed_ids),
        class_names=class_names,
        roster_p_rated=roster,
    )


# =============================================================================
# Temporal generation + aggregation
# =============================================================================
@dataclass(frozen=True)
class CompositionDraw:
    """Realized composition for the covered loads.

    Attributes
    ----------
    operating_point:
        ``{load_id: {"p_w": [B, T], "q_var": [B, T]}}`` — the aggregate fundamental
        (may be net-negative under PV).
    harmonic_injection:
        ``{load_id: {order: (mag[B, T], phase[B, T])}}`` — the aggregate spectrum in the
        solver's injection convention (magnitude as a fraction of the aggregate
        fundamental current, phase in degrees).
    samples:
        Class-attribution ground truth (see :func:`sample_device_composition`).
    composed_ids:
        The covered load ids (in aggregation order).
    """

    operating_point: dict
    harmonic_injection: dict
    samples: dict
    composed_ids: list


def _index_add(target: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """``index_add`` supporting complex (real/imag done separately)."""
    if torch.is_complex(source):
        real = target.real.index_add(dim, index, source.real)
        imag = target.imag.index_add(dim, index, source.imag)
        return torch.complex(real, imag)
    return target.index_add(dim, index, source)


def sample_device_composition(
    grid: Grid, config: CoherentSpectrumConfig
) -> CompositionDraw:
    """Sample the statistical device composition for a coherent config (reproducible).

    Parameters
    ----------
    grid:
        The reference grid; the loads covered by ``config.composition`` are composed.
    config:
        A :class:`CoherentSpectrumConfig` whose ``composition`` and ``start_time`` (or
        ``start_time_range``, a start per sequence) are set. The composition injects at ``config.orders`` and runs
        over ``config.n_scenarios`` × ``config.n_steps`` (``config.step_size_s``).

    Returns
    -------
    CompositionDraw
        Per-load operating point ``[B, T]``, harmonic injection ``[B, T]`` per order, and
        the class-attribution samples (``name`` = ``config.name``):

        - ``"<name>_class_p_w"`` ``[B, n_agg, n_class, T]`` — signed per-class active
          power contribution (the attribution label);
        - ``"<name>_class_active"`` ``[B, n_agg, n_class, T]`` (int64) — active member
          count per class per step;
        - ``"<name>_cap_binding"`` ``[B, n_agg, n_ord, T]`` — where the ``max_injection_pu``
          cap clipped the relative magnitude;
        - ``"<name>_agg_ids"`` ``[n_agg]`` (int64) — covered load ids;
        - ``"<name>_roster_p_rated"`` ``[n_agg, n_class, max_count]`` — the per-member
          rated powers (0-padded roster sidecar);
        - ``"<name>_composed_mag"`` / ``"<name>_composed_phase"`` ``[B, n_agg, n_ord, T]``
          — the REALIZED aggregate spectrum, i.e. the member-summed injection in per unit
          of the aggregate's own fundamental current and in degrees (the same pair the
          returned ``harmonic_injection`` carries), on the ``<name>_agg_ids`` device axis.

        The ``n_class`` axis is ordered as ``config.composition.classes`` (names via
        ``config.composition.class_names()``).
    """
    comp = config.composition
    if comp is None:  # pragma: no cover - guarded by the caller
        raise ValueError("sample_device_composition requires config.composition.")
    nm = config.name
    orders = list(config.orders)
    n_ord = len(orders)
    b, t = int(config.n_scenarios), int(config.n_steps)

    composed_ids = resolve_composed_ids(grid, comp)
    n_agg, n_class = len(composed_ids), len(comp.class_names())

    base_seed = comp.roster_seed if comp.roster_seed is not None else config.seed
    roster_seed = (int(base_seed) + _ROSTER_SEED_OFFSET) & 0x7FFFFFFF
    r = _build_roster(grid, comp, composed_ids, orders, roster_seed)

    # Empty roster: nothing composed -> no overrides.
    if r.p_rated.numel() == 0:
        return CompositionDraw({}, {}, {}, composed_ids)

    comp_seed = (int(base_seed) + _COMPOSITION_SEED_OFFSET) & 0x7FFFFFFF
    gen = torch.Generator().manual_seed(comp_seed)
    m = r.p_rated.shape[0]

    # calendar fields [S, T]: S = 1 (one shared window) or B (a start per sequence)
    _, hour, doy, dow = _batch_time_axis(config)
    s_seq = int(hour.shape[0])

    # Per-member diurnal availability rate [S, M, T] (grouped by preset), optionally
    # modulated by the weekly / seasonal calendar, then lifted by the configured duty
    # scale: the presets describe a device's own duty cycle, which puts an aggregate at a
    # small fraction of installed capacity — realistic for an average hour, but it never
    # visits the loaded states an estimator most needs to get right. The scale moves the
    # whole population up the same curve (the clamp keeps a rate a probability), leaving
    # the diurnal SHAPE and every other draw untouched.
    rate = torch.zeros((s_seq, m, t), dtype=_F64)
    for preset in set(r.preset):
        sel = torch.tensor([p == preset for p in r.preset], dtype=torch.bool)
        preset_rate = _activity_rate(preset, hour, doy)  # [S, T]
        if comp.calendar is not None:
            preset_rate = (
                preset_rate * _calendar_factor(preset, doy, dow, comp.calendar)
            ).clamp(0.0, 1.0)
        rate[:, sel] = preset_rate.unsqueeze(1)
    if comp.activity_scale != 1.0:
        rate = (rate * float(comp.activity_scale)).clamp(0.0, 1.0)

    # Per-scenario shared latents (co-variation): behavioral for consumption, cloud PV.
    z_beh = torch.randn((b, 1), generator=gen, dtype=_F64)
    z_cloud = torch.randn((b, 1), generator=gen, dtype=_F64)
    beh = (1.0 + comp.behavioral_coupling * z_beh).clamp(min=0.0).expand(b, m)
    cloud = (1.0 + comp.cloud_coupling * z_cloud).clamp(min=0.0).expand(b, m)
    latent = torch.where(r.use_cloud.unsqueeze(0), cloud, beh)  # [B, M]

    p_on = (rate * latent.unsqueeze(-1)).clamp(0.0, 1.0)  # [B, M, T]

    # Availability: on/off Markov for switching devices, continuous rate otherwise.
    on_markov = _markov_onoff(p_on, r.onoff_stickiness, gen)  # [B, M, T]
    avail = torch.where(r.discrete.reshape(1, m, 1), on_markov, p_on)  # [B, M, T]

    # Operating state (multi-state) + loading.
    path = _markov_states(r.state_cum_pi, r.state_dwell, b, t, gen)
    base_load = torch.gather(
        r.state_pfrac.unsqueeze(0).expand(b, m, r.state_pfrac.shape[1]), 2, path
    )  # [B, M, T]
    sscale = torch.gather(
        r.state_sscale.unsqueeze(0).expand(b, m, r.state_sscale.shape[1]), 2, path
    )
    e_lam = ar1_noise((b, m, t), r.loading_rho, gen)  # [B, M, T]
    lam = (base_load + r.loading_jitter.reshape(1, m, 1) * e_lam).clamp(min=0.0)
    lam = torch.maximum(lam, r.lam_min.reshape(1, m, 1)).clamp(max=1.0)  # [B, M, T]

    # Member fundamental power [W] and its "power-current" phasor (node voltage cancels).
    active_mag = avail * lam * r.p_rated.reshape(1, m, 1)  # [B, M, T] >= 0
    sign = r.sign.reshape(1, m, 1)
    p_member = sign * active_mag  # [B, M, T]
    q_member = p_member * r.tan_phi.reshape(1, m, 1)

    lam_o = lam.unsqueeze(2)  # [B, M, 1, T]
    mag_h = _mag_law(
        r.mag_rated.reshape(1, m, n_ord, 1),
        lam_o,
        r.gamma.reshape(1, m, n_ord, 1),
        sscale.unsqueeze(2),
    )  # [B, M, n_ord, T] fraction
    ang_h = _phase_law(
        r.ang0.reshape(1, m, n_ord, 1), r.slope.reshape(1, m, n_ord, 1), lam_o
    )  # [B, M, n_ord, T] deg
    scale_d = active_mag.unsqueeze(2)  # [B, M, 1, T]
    affine = _emission_affine(
        lam_o,
        r.emission_floor.reshape(1, m, n_ord, 1),
        r.emission_floor_phase.reshape(1, m, n_ord, 1),
    )  # [B, M, n_ord, T] complex, exactly 1 when the floor is zero
    c_member = (
        sign.unsqueeze(2)
        * mag_h
        * scale_d
        * torch.exp(1j * torch.deg2rad(ang_h))
        * affine
    )  # [B, M, n_ord, T] complex
    f_member = torch.complex(p_member, torch.zeros_like(p_member))  # [B, M, T]

    # Aggregate over members per load.
    li = r.load_idx
    c_agg = _index_add(
        torch.zeros((b, n_agg, n_ord, t), dtype=torch.complex128), 1, li, c_member
    )
    f_agg = _index_add(
        torch.zeros((b, n_agg, t), dtype=torch.complex128), 1, li, f_member
    )
    p_agg = _index_add(torch.zeros((b, n_agg, t), dtype=_F64), 1, li, p_member)
    q_agg = _index_add(torch.zeros((b, n_agg, t), dtype=_F64), 1, li, q_member)

    mag_agg, phase_agg, cap_binding = _aggregate_injection(
        c_agg, f_agg, comp.max_injection_pu
    )  # [B, n_agg, n_ord, T]

    # Per-class attribution ground truth (scatter members into [n_agg * n_class]).
    ac = r.load_idx * n_class + r.class_idx  # [M]
    class_p = _index_add(
        torch.zeros((b, n_agg * n_class, t), dtype=_F64), 1, ac, p_member
    ).reshape(b, n_agg, n_class, t)
    class_active = _index_add(
        torch.zeros((b, n_agg * n_class, t), dtype=_F64),
        1,
        ac,
        (avail > 1e-6).to(_F64),
    ).reshape(b, n_agg, n_class, t)

    operating_point = {
        cid: {"p_w": p_agg[:, a, :], "q_var": q_agg[:, a, :]}
        for a, cid in enumerate(composed_ids)
    }
    harmonic_injection = {
        cid: {
            order: (mag_agg[:, a, o, :], phase_agg[:, a, o, :])
            for o, order in enumerate(orders)
        }
        for a, cid in enumerate(composed_ids)
    }
    samples = {
        f"{nm}_class_p_w": class_p,  # [B, n_agg, n_class, T]
        f"{nm}_class_active": class_active.round().to(torch.long),
        f"{nm}_cap_binding": cap_binding,  # [B, n_agg, n_ord, T]
        f"{nm}_agg_ids": torch.tensor(composed_ids, dtype=torch.long),  # [n_agg]
        f"{nm}_roster_p_rated": r.roster_p_rated,  # [n_agg, n_class, max_count]
        # The realized aggregate spectrum itself: what the member sum injected, in the
        # same convention the fingerprint path records (per unit of the aggregate's own
        # fundamental current / degrees), so a written dataset is auditable without
        # re-running the roster or reconstructing I(h) = Y(h)*V(h) from the state.
        f"{nm}_composed_mag": mag_agg,  # [B, n_agg, n_ord, T]
        f"{nm}_composed_phase": phase_agg,  # [B, n_agg, n_ord, T]
    }
    return CompositionDraw(operating_point, harmonic_injection, samples, composed_ids)


__all__ = [
    "CompositionDraw",
    "sample_device_composition",
    "resolve_composed_ids",
]
