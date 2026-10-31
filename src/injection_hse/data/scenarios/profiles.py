"""Multi-scale synthetic load profiles for node-coherent scenario sequences.

Turns the constant per-scenario fundamental of a :class:`CoherentSpectrumConfig`
into a per-STEP time series. Each matched Load/Generator gets a multiplicative
factor composed on four time scales,

    ``factor(t) = f_seasonal(t) * f_weekly(t) * f_daily(t) * f_short(t)``,

applied to the device's per-scenario base ``P`` and ``Q`` together (constant power
factor). The daily shape is class-aware, keyed by each device's ``consumer_type``:

- ``household`` / ``heat_pump`` — low overnight, a small morning peak and a dominant
  evening peak;
- ``office`` / ``workshop`` — a smooth business-hours plateau;
- ``restaurant`` — twin lunch + dinner peaks;
- ``ev_charging`` — a late-evening charging peak;
- ``industrial_drive`` — a broad, shallow daytime plateau (near-flat base load);
- ``pv`` — a solar bell that is ZERO at night, with a daylight window and amplitude
  that widen in summer;
- everything else / unset — a neutral, near-flat default with a soft evening rise.

The shapes are smooth parametric curves (circular Gaussian bumps and sigmoid
plateaus), so they are continuous across midnight and differentiable in time. Every
consumption shape is zero-centred and scaled to a per-class daily depth, so the
composed factor oscillates around the device's base level; ``pv`` is a peak-1 bell
scaled by the seasonal amplitude.

Correlation model
-----------------
Per scenario, two SHARED latents make devices co-vary: a ``behavioral`` latent scales
the daily amplitude of every non-``pv`` device together, and a ``cloudiness`` latent
scales every ``pv`` device's output together. On top of them each device draws
idiosyncratic per-scenario values (overall level, amplitude multiplier, daily phase
offset) and a per-step AR(1) short-term term. See :class:`LoadProfileConfig`.

Reproducibility + tape hygiene
------------------------------
All draws come from a seeded RNG DERIVED from ``CoherentSpectrumConfig.seed`` with its
own offset — distinct from the fingerprint bank, the Markov path, the AR(1) jitter and
the operating-point cube — so enabling a profile leaves the harmonic fingerprint and
the raw parameter draws byte-identical. The factor is a plain ``float64`` tensor with
no ``requires_grad``; multiplying it onto a (possibly tensor-valued) base P/Q keeps the
solve differentiable w.r.t. that base while the sampling itself stays off the tape. All
sampling runs on CPU (like the rest of ``pgml.scenarios``); the tensors are promoted to
the solve device by :func:`~pgml.scenarios.run_scenarios`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Generator, Grid, Load

from pgml.errors import InputError
from pgml.scenarios import ar1_noise
from .config import CoherentSpectrumConfig, LoadProfileConfig, _epoch_s

_F64 = torch.float64

# Derived-stream offset for the profile RNG: distinct from the fingerprint RNG
# (seeded from ``config.seed``) and the operating-point cube (``_OP_CUBE_SEED_OFFSET``
# in ``harmonics``), so the profile draws never disturb either stream.
_PROFILE_SEED_OFFSET = 0x85EBCA6B  # 2246822507
# Derived-stream offset of the per-sequence start-time draw (``StartTimeRange``).
_START_TIME_SEED_OFFSET = 0x165667B1  # 374761393
# Upper bound on the candidate starts a ``StartTimeRange`` enumerates (a year at 10 s).
_MAX_START_CANDIDATES = 4_000_000

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_YEAR = 365.25 * _SECONDS_PER_DAY
# 1970-01-01 (epoch day 0) was a Thursday; Python weekday() has Monday=0, so adding
# 3 maps epoch day 0 to weekday index 3 (Thursday). Saturday=5, Sunday=6.
_EPOCH_WEEKDAY_OFFSET = 3.0

# Per-consumer-type daily preset key. Absent / unlisted types fall back to "neutral";
# "pv" is handled by the dedicated solar-bell path (never a consumption shape).
_PRESET_BY_TYPE = {
    "household": "household",
    "heat_pump": "household",
    "ev_charging": "ev",
    "office": "office",
    "workshop": "office",
    "restaurant": "restaurant",
    "industrial_drive": "industrial",
    "pv": "pv",
}

# Per-class daily DEPTH: the peak deviation of the (zero-centred, unit-peak) shape,
# i.e. how strongly the class swings over a day relative to its base level.
_TYPE_DEPTH = {
    "household": 0.60,
    "ev": 0.90,
    "office": 0.70,
    "restaurant": 0.70,
    "industrial": 0.25,
    "neutral": 0.10,
}

# Dense reference grid (hours in [0, 24)) used to zero-centre + unit-normalise each
# raw daily shape once, independent of the actual step grid.
_HOUR_REF = torch.linspace(0.0, 24.0, 241, dtype=_F64)[:-1]


def _cbump(hour: Tensor, mu: float, sigma: float) -> Tensor:
    """Circular Gaussian bump on the 24 h clock (continuous across midnight)."""
    d = torch.remainder(hour - mu + 12.0, 24.0) - 12.0
    return torch.exp(-0.5 * (d / sigma) ** 2)


def _sig(x: Tensor, w: float) -> Tensor:
    """Logistic ``sigmoid(x / w)`` — the building block of a smooth plateau."""
    return torch.sigmoid(x / w)


def _raw_daily(key: str, hour: Tensor) -> Tensor:
    """Raw (un-normalised) daily shape for a consumption preset key."""
    if key == "household":
        return 1.05 * _cbump(hour, 19.5, 2.2) + 0.45 * _cbump(hour, 7.0, 1.6)
    if key == "ev":
        return _cbump(hour, 22.0, 2.3)
    if key == "office":
        return _sig(hour - 8.0, 1.0) - _sig(hour - 18.0, 1.0)
    if key == "restaurant":
        return 0.7 * _cbump(hour, 13.0, 1.4) + 1.0 * _cbump(hour, 20.0, 2.0)
    if key == "industrial":
        return _sig(hour - 6.0, 1.5) - _sig(hour - 22.0, 1.5)
    # neutral: a soft evening rise around an otherwise flat day.
    return _cbump(hour, 18.0, 4.0)


def _daily_shape(key: str, hour: Tensor) -> Tensor:
    """Zero-centred daily shape scaled to the class depth (peak deviation).

    Normalises the raw shape on the dense reference grid to zero mean and unit peak
    deviation, then scales by the per-class depth, so ``1 + amplitude * shape`` swings
    by roughly ``+/- (amplitude * depth)`` around the base level over a day.
    """
    ref = _raw_daily(key, _HOUR_REF)
    m = ref.mean()
    peak = (ref - m).abs().max().clamp(min=1e-12)
    depth = _TYPE_DEPTH.get(key, _TYPE_DEPTH["neutral"])
    return depth * (_raw_daily(key, hour) - m) / peak


def _pv_bell(hour: Tensor, doy: Tensor, cfg: LoadProfileConfig) -> Tensor:
    """Solar bell: peak 1 at solar noon, zero at night, seasonally widening window.

    ``hour`` is ``[*batch, T]`` (device time-of-day, possibly phase-shifted); ``doy`` is
    the day of year, broadcastable against it (``[T]``, ``[1, T]`` or ``[B, T]``). The daylight half-width widens in summer; outside the window
    the bell is exactly 0.
    """
    daylen = cfg.pv_daylight_hours + cfg.pv_daylight_swing * torch.cos(
        2.0 * math.pi * (doy - cfg.pv_seasonal_peak_doy) / 365.25
    )
    half_width = (daylen / 2.0).clamp(min=0.5)  # shaped like doy
    x = (hour - 12.0) / half_width  # solar noon at 12:00
    return torch.cos(0.5 * math.pi * x).clamp(min=0.0) ** 1.3


def _calendar_fields(time_unix_s: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(hour_of_day, day_of_year, day_of_week)`` of epoch seconds, any shape (float64).

    The daily / weekly / seasonal fields are recovered from the UTC epoch by modular
    arithmetic: hour in ``[0, 24)``, day of year on a fixed 365.25-day period, and the
    continuous day of week (Monday 0 .. Sunday 6, fractional over the day).
    """
    hour = torch.remainder(time_unix_s, _SECONDS_PER_DAY) / 3600.0
    doy = torch.remainder(time_unix_s, _SECONDS_PER_YEAR) / _SECONDS_PER_DAY
    dow = torch.remainder(time_unix_s / _SECONDS_PER_DAY + _EPOCH_WEEKDAY_OFFSET, 7.0)
    return hour, doy, dow


def _time_axis(
    start_time: str, step_size_s: float, t: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Absolute time axis for the profile phases.

    Returns ``(time_unix_s, hour_of_day, day_of_year, day_of_week)`` as ``float64``
    ``[T]`` tensors. A naive (timezone-less) ``start_time`` is interpreted as UTC so the
    epoch is reproducible across machines; the daily / weekly / seasonal fields are
    recovered from that same UTC frame by modular arithmetic (a wall-clock ``12:00``
    anchor therefore lands the solar bell at hour 12).
    """
    dt = datetime.fromisoformat(start_time)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    base_epoch = dt.timestamp()
    time_unix_s = base_epoch + torch.arange(t, dtype=_F64) * float(step_size_s)
    return (time_unix_s, *_calendar_fields(time_unix_s))


def _start_candidates(config: CoherentSpectrumConfig) -> Tensor:
    """The admissible start instants of ``config.start_time_range`` (float64 epoch ``[C]``)."""
    rng = config.start_time_range
    t0, t1 = _epoch_s(rng.start), _epoch_s(rng.end)
    res = float(rng.resolution_s or config.step_size_s)
    n = int(math.ceil((t1 - t0) / res))
    if n > _MAX_START_CANDIDATES:
        raise InputError(
            f"StartTimeRange enumerates {n} candidate starts (more than "
            f"{_MAX_START_CANDIDATES}); raise resolution_s or narrow the range."
        )
    cand = t0 + torch.arange(n, dtype=_F64) * res
    cand = cand[cand < t1]
    keep = torch.ones(cand.shape, dtype=torch.bool)
    hour, _doy, dow = _calendar_fields(cand)
    if rng.hours is not None:
        lo, hi = (float(h) for h in rng.hours)
        keep &= (hour >= lo) & (hour < hi) if lo <= hi else (hour >= lo) | (hour < hi)
    if rng.weekdays is not None:
        allowed = torch.as_tensor(sorted(set(rng.weekdays)), dtype=torch.long)
        keep &= torch.isin(torch.floor(dow).to(torch.long), allowed)
    cand = cand[keep]
    if cand.numel() == 0:
        raise InputError(
            "StartTimeRange admits no start: no candidate in the range matches its hours "
            "and weekdays."
        )
    return cand


def sequence_start_times(config: CoherentSpectrumConfig) -> Optional[Tensor]:
    """Absolute start of every sequence of a coherent batch, float64 epoch seconds ``[B]``.

    ``config.start_time`` gives every sequence the same start; ``config.start_time_range``
    draws one per sequence (:class:`~injection_hse.data.scenarios.StartTimeRange`), uniformly or
    stratified over its admissible candidates, on a stream derived from ``config.seed``
    (deterministic for a fixed config, independent of every other draw of the batch).
    ``None`` for a timeless config (neither set). CPU tensor.
    """
    b = int(config.n_scenarios)
    rng = config.start_time_range
    if rng is None:
        if config.start_time is None:
            return None
        return torch.full((b,), _epoch_s(config.start_time), dtype=_F64)
    cand = _start_candidates(config)
    m = int(cand.numel())
    gen = torch.Generator().manual_seed(
        (int(config.seed) + _START_TIME_SEED_OFFSET) & 0x7FFFFFFF
    )
    if rng.sampling == "uniform":
        idx = torch.randint(0, m, (b,), generator=gen)
    else:
        u = (torch.arange(b, dtype=_F64) + torch.rand(b, generator=gen, dtype=_F64)) / b
        idx = torch.floor(u * m).to(torch.long).clamp(max=m - 1)
        idx = idx[torch.randperm(b, generator=gen)]
    return cand[idx]


def sequence_time_axis(config: CoherentSpectrumConfig) -> Optional[Tensor]:
    """Absolute per-step time axis of a coherent batch, float64 epoch seconds.

    ``[T]`` (``start_time`` + ``k * step_size_s``, shared by every sequence) for a config
    with one ``start_time``; ``[B, T]`` (each sequence from its own start of
    :func:`sequence_start_times`) for a config with a ``start_time_range``; ``None`` for a
    timeless config. This is the axis the generator records as ``samples["time_unix_s"]``
    and the profile / composition activity models follow. CPU tensor.
    """
    t = int(config.n_steps)
    if config.start_time_range is None:
        if config.start_time is None:
            return None
        return _time_axis(config.start_time, config.step_size_s, t)[0]
    steps = torch.arange(t, dtype=_F64) * float(config.step_size_s)
    return sequence_start_times(config).unsqueeze(-1) + steps  # [B, T]


def _batch_time_axis(
    config: CoherentSpectrumConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(time_unix_s, hour, doy, dow)`` of a timed config, the fields as ``[S, T]``.

    ``time_unix_s`` is the recorded axis of :func:`sequence_time_axis` (``[T]`` or
    ``[B, T]``); the calendar fields carry a leading sequence axis ``S`` of ``1`` (one
    shared window, broadcast over the batch) or ``B`` (a start per sequence), so the
    activity models broadcast them against their ``[B, ...]`` draws either way.
    """
    time_unix_s = sequence_time_axis(config)
    if time_unix_s is None:  # pragma: no cover - validated on the config
        raise ValueError("the time axis needs config.start_time or start_time_range.")
    fields = _calendar_fields(time_unix_s)
    if time_unix_s.ndim == 1:
        fields = tuple(f.unsqueeze(0) for f in fields)
    return (time_unix_s, *fields)


def _weekly_factor(dow: Tensor, contrast: float) -> Tensor:
    """Smooth weekday/weekend factor: ``~1`` on weekdays, ``1 - contrast`` on weekends.

    ``dow`` is a continuous day-of-week index (Monday 0 .. Sunday 6, fractional over the
    day). A circular Gaussian centred on the Sat/Sun midpoint gives a smooth weekend
    dip whose depth is ``contrast`` (a single aggregate contrast; positive means lower
    weekend consumption).
    """
    d = torch.remainder(dow - 5.5 + 3.5, 7.0) - 3.5  # circular distance to Sat/Sun mid
    weekend = torch.exp(-0.5 * (d / 0.75) ** 2)
    return 1.0 - contrast * weekend


@dataclass(frozen=True)
class ProfileDraw:
    """Realized profile factors for a coherent batch.

    Attributes
    ----------
    factor:
        ``[B, n_dev, T]`` float64 multiplicative factor per profiled device per step.
    device_ids:
        int64 ``[n_dev]`` profiled device ids (Load/Generator), in selector order.
    time_unix_s:
        float64 absolute epoch seconds per step: ``[T]`` (``start_time`` + ``k*dt``) for
        one shared window, ``[B, T]`` for per-sequence starts (``start_time_range``).
    """

    factor: Tensor
    device_ids: Tensor
    time_unix_s: Tensor


def _profiled_ids(grid: Grid, cfg: LoadProfileConfig) -> list[int]:
    """Ids of the devices the profile applies to (in appliance order)."""
    if cfg.selector is not None:
        return cfg.selector.resolve(grid)
    return [
        a.id
        for a in grid.appliances
        if isinstance(a, (Load, Generator)) and a.in_service
    ]


def load_profile_factors(grid: Grid, config: CoherentSpectrumConfig) -> ProfileDraw:
    """Draw the per-device, per-step profile factors for a coherent config.

    Parameters
    ----------
    grid:
        The reference grid; only the profiled Load/Generator devices are used.
    config:
        A :class:`CoherentSpectrumConfig` whose ``profile`` and ``start_time`` are set.

    Returns
    -------
    ProfileDraw
        ``factor`` ``[B, n_dev, T]`` (float64, non-negative), ``device_ids`` ``[n_dev]``,
        ``time_unix_s`` ``[T]`` (one shared window) or ``[B, T]`` (per-sequence starts).
    """
    cfg = config.profile
    if cfg is None:  # pragma: no cover - guarded by the caller
        raise ValueError("load_profile_factors requires config.profile to be set.")
    if not config.is_timed:  # pragma: no cover - validated on config
        raise ValueError(
            "load_profile_factors requires config.start_time or start_time_range."
        )

    ids = _profiled_ids(grid, cfg)
    by_id = {a.id: a for a in grid.appliances}
    b, t = int(config.n_scenarios), int(config.n_steps)
    n_dev = len(ids)

    # calendar fields [S, T]: S = 1 (one shared window) or B (a start per sequence)
    time_unix_s, hour, doy, dow = _batch_time_axis(config)

    if n_dev == 0:  # nothing matched: an empty factor bank (still shape-correct).
        return ProfileDraw(
            factor=torch.zeros((b, 0, t), dtype=_F64),
            device_ids=torch.zeros((0,), dtype=torch.long),
            time_unix_s=time_unix_s,
        )

    seed = (int(config.seed) + _PROFILE_SEED_OFFSET) & 0x7FFFFFFF
    gen = torch.Generator().manual_seed(seed)

    # Per-scenario shared latents (standard normal), clamped so the co-variation
    # scaling stays non-negative.
    z_behavior = torch.randn((b, 1), generator=gen, dtype=_F64)
    z_cloud = torch.randn((b, 1), generator=gen, dtype=_F64)
    behavioral_scale = (1.0 + cfg.behavioral_coupling * z_behavior).clamp(min=0.0)
    cloud_scale = (1.0 + cfg.cloud_coupling * z_cloud).clamp(min=0.0)

    # Per-scenario, per-device idiosyncratic draws.
    def _uniform(lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand((b, n_dev), generator=gen, dtype=_F64)

    level = _uniform(cfg.level_min, cfg.level_max)
    amp_jitter = _uniform(cfg.amplitude_jitter_min, cfg.amplitude_jitter_max)
    shift = _uniform(-cfg.phase_offset_hours, cfg.phase_offset_hours)

    # Effective per-device daily amplitude / pv output scale (shared latent x jitter).
    amp_daily = (cfg.daily_amplitude * behavioral_scale * amp_jitter).clamp(min=0.0)
    pv_scale = (cloud_scale * amp_jitter).clamp(min=0.0)

    # Per-device per-step AR(1) short-term term.
    e_short = ar1_noise((b, n_dev, t), cfg.short_rho, gen)  # [B, n_dev, T]

    # Slow (device-independent) modulators, [S, T].
    f_seasonal = 1.0 + cfg.seasonal_amplitude * torch.cos(
        2.0 * math.pi * (doy - cfg.seasonal_peak_doy) / 365.25
    )
    f_seasonal_pv = 1.0 + cfg.pv_seasonal_amplitude * torch.cos(
        2.0 * math.pi * (doy - cfg.pv_seasonal_peak_doy) / 365.25
    )
    f_weekly = _weekly_factor(dow, cfg.weekend_contrast)

    factors = []
    for d, cid in enumerate(ids):
        ctype = getattr(by_id[cid], "consumer_type", None)
        # ``consumer_type`` is a ConsumerType (str) enum or a plain string; key on its
        # string value (``ConsumerType.PV.value == "pv"``) so both forms resolve.
        ctype_key = getattr(ctype, "value", ctype)
        key = _PRESET_BY_TYPE.get(ctype_key, "neutral")
        hour_d = hour - shift[:, d : d + 1]  # [B, T]
        short_d = 1.0 + cfg.short_sigma * e_short[:, d, :]  # [B, T]
        if key == "pv":
            bell = _pv_bell(hour_d, doy, cfg)  # [B, T]
            fac = pv_scale[:, d : d + 1] * f_seasonal_pv * bell * short_d
        else:
            shape = _daily_shape(key, hour_d)  # [B, T]
            daily = 1.0 + amp_daily[:, d : d + 1] * shape
            fac = level[:, d : d + 1] * f_seasonal * f_weekly * daily * short_d
        factors.append(fac.clamp(min=0.0))

    factor = torch.stack(factors, dim=1)  # [B, n_dev, T]
    return ProfileDraw(
        factor=factor,
        device_ids=torch.as_tensor(ids, dtype=torch.long),
        time_unix_s=time_unix_s,
    )


def _mul(base, fac: Tensor) -> Tensor:
    """Multiply a per-scenario / scalar base P or Q by a ``[B, T]`` factor.

    ``base`` is a python float, a 0-d/``[B]`` tensor (a per-scenario draw), or a
    (tensor) nominal. Its batch is right-padded with a step axis so a ``[B]`` base
    broadcasts against ``[B, T]``; grad to a tensor base is preserved.
    """
    if isinstance(base, Tensor):
        b = base
        while b.ndim < fac.ndim:
            b = b.unsqueeze(-1)
        return b * fac
    return base * fac


def apply_load_profiles(
    grid: Grid, config: CoherentSpectrumConfig, operating_point: dict
) -> tuple[dict, dict]:
    """Lift a per-scenario operating point to a per-step ``[B, T]`` one via the profile.

    Each profiled device's base P/Q (its ``operating_point`` entry if present, else its
    nominal ``p_nom_w`` / ``q_nom_var``) is multiplied by the device's profile factor,
    so the totals gain the step axis (``p_w`` / ``q_var`` become ``[B, T]``; a per-phase
    base keeps its structure, each phase scaled by the same factor). A batched Source
    ``u_ref_scale`` (``[B]``, per-scenario) is promoted to ``[B, 1]`` so the slack
    reference broadcasts against the per-step ``[B, T]`` state (the scale stays constant
    over the sequence); other non-profiled entries pass through untouched.

    Parameters
    ----------
    grid, config:
        The reference grid and the coherent config (``config.profile`` set).
    operating_point:
        The per-scenario base operating point (``[B]`` draws or empty). NOT mutated —
        a new dict is returned.

    Returns
    -------
    (operating_point, samples)
        The lifted operating point and the profile ``samples`` — ``"<name>_profile_factor"``
        ``[B, n_dev, T]``, ``"<name>_profile_device_ids"`` ``[n_dev]``, and
        ``"time_unix_s"`` (``[T]``, or ``[B, T]`` with a ``start_time_range``).
    """
    draw = load_profile_factors(grid, config)
    by_id = {a.id: a for a in grid.appliances}
    nm = config.name

    op = {cid: dict(entry) for cid, entry in operating_point.items()}
    for d, cid in enumerate(draw.device_ids.tolist()):
        fac = draw.factor[:, d, :]  # [B, T]
        entry = op.get(cid, {})
        appliance = by_id[cid]
        new: dict = {}
        if "p_per_phase_w" in entry or "q_per_phase_var" in entry:
            if "p_per_phase_w" in entry:
                new["p_per_phase_w"] = [_mul(x, fac) for x in entry["p_per_phase_w"]]
            else:
                new["p_w"] = _mul(entry.get("p_w", appliance.p_nom_w), fac)
            if "q_per_phase_var" in entry:
                new["q_per_phase_var"] = [
                    _mul(x, fac) for x in entry["q_per_phase_var"]
                ]
            else:
                new["q_var"] = _mul(entry.get("q_var", appliance.q_nom_var), fac)
        else:
            new["p_w"] = _mul(entry.get("p_w", appliance.p_nom_w), fac)
            new["q_var"] = _mul(entry.get("q_var", appliance.q_nom_var), fac)
        op[cid] = new

    # A per-scenario source-voltage scale is [B]; the profiled state batch is [B, T].
    # Promote to [B, 1] so the ideal-slack reference broadcasts across the steps
    # (torch aligns trailing dims — a bare [B] would pair with T, not B).
    for entry in op.values():
        scale = entry.get("u_ref_scale")
        if scale is not None:
            t = torch.as_tensor(scale)
            if t.ndim == 1:
                entry["u_ref_scale"] = t.unsqueeze(-1)

    samples = {
        f"{nm}_profile_factor": draw.factor,  # [B, n_dev, T]
        f"{nm}_profile_device_ids": draw.device_ids,  # [n_dev]
        "time_unix_s": draw.time_unix_s,  # [T] or [B, T]
    }
    return op, samples


__all__ = [
    "ProfileDraw",
    "load_profile_factors",
    "apply_load_profiles",
    "sequence_start_times",
    "sequence_time_axis",
]
