"""Per-sequence start times and the composition's activity calendar.

Covers: the opt-in fields stay out of the serialized config (and its hash) while unset;
validation; a range admitting exactly one start reproduces the fixed-start draws exactly
(every other stream untouched, only the time axis gains its sequence dimension); the
``[B, T]`` axis shape, determinism, seed dependence and candidate filters; stratified
coverage; the weekly / seasonal calendar (neutral calendar = no calendar, weekend dip,
winter consumption peak, summer PV peak).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import Selector, config_hash
from injection_hse.data.scenarios import (
    ActivityCalendar,
    CoherentSpectrumConfig,
    CompositionConfig,
    LoadProfileConfig,
    StartTimeRange,
    sample_coherent_spectra,
    sample_device_composition,
    sequence_start_times,
    sequence_time_axis,
)

from .test_composition import _one_load_grid

DAY = 86400.0
YEAR = StartTimeRange(start="2024-01-01", end="2025-01-01")


def _epoch(stamp: str) -> float:
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp()


def _cfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5])
    kw.setdefault("n_steps", 6)
    kw.setdefault("n_scenarios", 8)
    kw.setdefault("seed", 0)
    kw.setdefault("step_size_s", 900.0)
    if "start_time_range" not in kw:
        kw.setdefault("start_time", "2024-06-21T16:00:00")
    kw.setdefault("composition", CompositionConfig())
    return CoherentSpectrumConfig(**kw)


def _equal(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    return a == b


# --- serialization + validation ----------------------------------------------
def test_unset_fields_are_not_serialized():
    cfg = _cfg()
    dumped = cfg.model_dump_json()
    assert "start_time_range" not in dumped and "calendar" not in dumped
    timed = _cfg(
        start_time_range=YEAR,
        composition=CompositionConfig(calendar=ActivityCalendar()),
    )
    assert "start_time_range" in timed.model_dump_json()
    assert "calendar" in timed.model_dump_json()
    assert CoherentSpectrumConfig.model_validate_json(timed.model_dump_json()) == timed
    assert config_hash(timed) != config_hash(cfg)


def test_start_time_and_range_are_exclusive():
    with pytest.raises(ValidationError, match="not both"):
        _cfg(start_time="2024-06-21T16:00:00", start_time_range=YEAR)


def test_range_satisfies_the_time_anchor_requirement():
    cfg = _cfg(start_time_range=YEAR, profile=LoadProfileConfig())
    assert cfg.is_timed and cfg.start_time is None


@pytest.mark.parametrize(
    "kw",
    [
        {"start": "2024-02-01", "end": "2024-01-01"},
        {"start": "2024-01-01", "end": "not a date"},
        {"start": "2024-01-01", "end": "2024-02-01", "weekdays": (7,)},
        {"start": "2024-01-01", "end": "2024-02-01", "weekdays": ()},
        {"start": "2024-01-01", "end": "2024-02-01", "hours": (0.0, 25.0)},
    ],
)
def test_range_validation(kw):
    with pytest.raises(ValidationError):
        StartTimeRange(**kw)


def test_a_range_without_admissible_start_raises():
    # 2024-01-01 is a Monday: a two-day range holds no Saturday
    rng = StartTimeRange(start="2024-01-01", end="2024-01-03", weekdays=(5,))
    with pytest.raises(Exception, match="admits no start"):
        sequence_start_times(_cfg(start_time_range=rng))


# --- equivalence with the fixed start ----------------------------------------
@pytest.mark.parametrize("kind", ["composition", "profile"])
def test_single_candidate_range_reproduces_the_fixed_start(kind):
    """A range admitting only the fixed start draws every stream identically; only the
    recorded time axis gains its sequence dimension."""
    grid = _one_load_grid("household", p_nom=4000.0)
    start = "2024-06-21T16:00:00"
    extra = (
        {}
        if kind == "composition"
        else {"composition": None, "profile": LoadProfileConfig()}
    )
    fixed = _cfg(start_time=start, **extra)
    one = StartTimeRange(start=start, end="2024-06-21T16:15:00")
    ranged = _cfg(start_time_range=one, **extra)
    a, b = sample_coherent_spectra(grid, fixed), sample_coherent_spectra(grid, ranged)

    tu_fixed = a.shared_samples["time_unix_s"]
    tu_ranged = b.samples["time_unix_s"]
    assert tu_fixed.shape == (6,) and tu_ranged.shape == (8, 6)
    assert torch.equal(tu_ranged, tu_fixed.expand(8, 6))
    assert "time_unix_s" not in b.shared_samples

    sa = {k: v for k, v in a.samples.items() if k != "time_unix_s"}
    sb = {k: v for k, v in b.samples.items() if k != "time_unix_s"}
    assert sa.keys() == sb.keys() and all(_equal(sa[k], sb[k]) for k in sa)
    for cid, entry in a.operating_point.items():
        assert all(_equal(v, b.operating_point[cid][f]) for f, v in entry.items())
    for cid, entry in a.harmonic_injection.items():
        assert all(_equal(v, b.harmonic_injection[cid][o]) for o, v in entry.items())


# --- the per-sequence axis ---------------------------------------------------
def test_axis_shape_determinism_and_seed():
    cfg = _cfg(start_time_range=YEAR, n_scenarios=16, n_steps=8)
    axis = sequence_time_axis(cfg)
    assert axis.shape == (16, 8) and axis.dtype == torch.float64
    assert torch.equal(axis, sequence_time_axis(cfg))
    assert torch.allclose(
        torch.diff(axis, dim=1), torch.full((16, 7), 900.0, dtype=torch.float64)
    )
    starts = axis[:, 0]
    t0, t1 = _epoch("2024-01-01T00:00:00"), _epoch("2025-01-01T00:00:00")
    assert bool(((starts >= t0) & (starts < t1)).all())
    assert torch.equal(
        torch.remainder(starts - t0, 900.0), torch.zeros(16, dtype=torch.float64)
    )
    assert len(set(starts.tolist())) > 12  # spread over the year
    other = sequence_time_axis(cfg.model_copy(update={"seed": 1}))
    assert not torch.equal(axis, other)


def test_fixed_start_axis_is_shared():
    cfg = _cfg()
    axis = sequence_time_axis(cfg)
    assert axis.shape == (6,)
    assert float(axis[0]) == _epoch("2024-06-21T16:00:00")
    assert torch.equal(
        sequence_start_times(cfg), torch.full((8,), float(axis[0]), dtype=torch.float64)
    )


def test_hours_and_weekdays_restrict_the_starts():
    rng = StartTimeRange(
        start="2024-01-01", end="2025-01-01", hours=(22.0, 6.0), weekdays=(5, 6)
    )
    starts = sequence_start_times(_cfg(start_time_range=rng, n_scenarios=64))
    hour = torch.remainder(starts, DAY) / 3600.0
    weekday = torch.floor(torch.remainder(starts / DAY + 3.0, 7.0))
    assert bool(((hour >= 22.0) | (hour < 6.0)).all())
    assert set(weekday.tolist()) <= {5.0, 6.0}


def test_stratified_starts_cover_every_stratum():
    b = 12
    rng = StartTimeRange(
        start="2024-01-01", end="2025-01-01", sampling="stratified", resolution_s=3600.0
    )
    starts = sequence_start_times(_cfg(start_time_range=rng, n_scenarios=b))
    t0 = _epoch("2024-01-01T00:00:00")
    m = 366 * 24  # hourly candidates in the leap year
    idx = np.sort(((starts.numpy() - t0) / 3600.0).round().astype(int))
    assert np.array_equal(idx * b // m, np.arange(b))


# --- the activity calendar ---------------------------------------------------
def test_neutral_calendar_equals_no_calendar():
    grid = _one_load_grid("household", p_nom=4000.0)
    neutral = ActivityCalendar(
        weekend_contrast=0.0, seasonal_amplitude=0.0, pv_seasonal_amplitude=0.0
    )
    a = sample_device_composition(grid, _cfg(start_time_range=YEAR))
    b = sample_device_composition(
        grid,
        _cfg(start_time_range=YEAR, composition=CompositionConfig(calendar=neutral)),
    )
    assert all(_equal(a.samples[k], b.samples[k]) for k in a.samples)


def _class_power(grid, class_name, start, *, n_steps, calendar, n_scenarios=64):
    cfg = _cfg(
        start_time=start,
        n_steps=n_steps,
        n_scenarios=n_scenarios,
        step_size_s=3600.0,
        composition=CompositionConfig(calendar=calendar),
    )
    d = sample_device_composition(grid, cfg)
    k = CompositionConfig().class_names().index(class_name)
    return d.samples["harmonics_class_p_w"][:, 0, k, :].abs().mean(dim=0)  # [T]


def test_calendar_weekend_dip():
    grid = _one_load_grid("household", p_nom=4000.0)
    cal = ActivityCalendar(weekend_contrast=0.6, seasonal_amplitude=0.0)
    # 2024-01-01 is a Monday: one hourly week, Saturday/Sunday are steps 120..167
    week = _class_power(
        grid, "base_linear", "2024-01-01T00:00:00", n_steps=168, calendar=cal
    )
    weekday, weekend = week[:120].reshape(5, 24), week[120:].reshape(2, 24)
    assert float(weekend.mean()) < 0.8 * float(weekday.mean())


def test_calendar_seasons():
    load = _one_load_grid("household", p_nom=4000.0)
    pv = _one_load_grid("pv", p_nom=5000.0)
    cal = ActivityCalendar(weekend_contrast=0.0, seasonal_amplitude=0.4)
    winter = _class_power(
        load, "base_linear", "2024-01-15T00:00:00", n_steps=24, calendar=cal
    )
    summer = _class_power(
        load, "base_linear", "2024-07-15T00:00:00", n_steps=24, calendar=cal
    )
    assert float(winter.mean()) > 1.2 * float(summer.mean())
    pv_dec = _class_power(
        pv, "pv_inverter", "2024-12-21T00:00:00", n_steps=24, calendar=cal
    )
    pv_jun = _class_power(
        pv, "pv_inverter", "2024-06-21T00:00:00", n_steps=24, calendar=cal
    )
    assert float(pv_jun.sum()) > 1.5 * float(pv_dec.sum())
    assert float(pv_jun[:3].max()) == 0.0  # still dark at night


def test_calendar_flat_heating_follows_the_season_not_the_weekday():
    grid = _one_load_grid("heat_pump", p_nom=6000.0)
    cal = ActivityCalendar(
        weekend_contrast=0.6, preset_seasonal_amplitude={"flat": 0.6}
    )
    winter = _class_power(
        grid, "heat_pump_inverter", "2024-01-15T00:00:00", n_steps=24, calendar=cal
    )
    summer = _class_power(
        grid, "heat_pump_inverter", "2024-07-15T00:00:00", n_steps=24, calendar=cal
    )
    assert float(winter.mean()) > 2.0 * float(summer.mean())
    week = _class_power(
        grid, "heat_pump_inverter", "2024-01-01T00:00:00", n_steps=168, calendar=cal
    )
    ratio = float(week[120:].mean()) / float(week[:120].mean())
    assert 0.9 < ratio < 1.1  # no weekend dip
    with pytest.raises(ValidationError):
        ActivityCalendar(preset_seasonal_amplitude={"pv": 0.2})
