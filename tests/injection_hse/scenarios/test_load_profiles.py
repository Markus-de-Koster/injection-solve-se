"""Time-varying fundamental load profiles for node-coherent sequences.

Covers the multi-scale synthetic profile generator: byte-identity when disabled,
determinism, the class-aware daily/weekly/seasonal physics (pv zero at night +
seasonal window, weekday/weekend contrast), the ``[B, T]`` solve == per-step loop
parity, and JSON + dataset round-trips of the new config and recorded outputs.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import (
    ParameterSpec,
    Selector,
    Uniform,
    run_scenarios,
    write_dataset,
)
from injection_hse.data.scenarios import (
    read_dataset,
    CoherentSpectrumConfig,
    LoadProfileConfig,
    sample_coherent_spectra,
)

from injection_hse.data.scenarios import (
    load_profile_factors,
)
from pgml.solver import solve_harmonic_flow

CDT = torch.complex128


def _ccfg(grid_selector="load", **kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component=grid_selector))
    kw.setdefault("orders", [3, 5])
    kw.setdefault("n_steps", 8)
    kw.setdefault("n_scenarios", 6)
    kw.setdefault("seed", 0)
    return CoherentSpectrumConfig(**kw)


# --- config validation + serialization -------------------------------------
def test_profile_requires_start_time():
    with pytest.raises(ValidationError, match="start_time"):
        _ccfg(profile=LoadProfileConfig())


def test_start_time_must_be_iso8601():
    with pytest.raises(ValidationError, match="ISO 8601"):
        _ccfg(profile=LoadProfileConfig(), start_time="not-a-date")


def test_profile_config_json_roundtrip():
    cfg = _ccfg(
        profile=LoadProfileConfig(
            daily_amplitude=0.8,
            behavioral_coupling=0.25,
            cloud_coupling=0.6,
            weekend_contrast=0.2,
            short_rho=0.7,
            short_sigma=0.1,
        ),
        start_time="2024-06-21T00:00:00",
    )
    back = CoherentSpectrumConfig.model_validate_json(cfg.model_dump_json())
    assert back == cfg
    assert back.profile is not None and back.start_time == "2024-06-21T00:00:00"


def test_profile_config_validators():
    with pytest.raises(ValidationError, match="level_max"):
        LoadProfileConfig(level_min=1.2, level_max=0.9)
    with pytest.raises(ValidationError, match="amplitude_jitter_max"):
        LoadProfileConfig(amplitude_jitter_min=1.5, amplitude_jitter_max=1.0)


# --- byte-identity + determinism -------------------------------------------
def test_profile_none_leaves_fingerprint_byte_identical(grid3):
    """Enabling a profile must not disturb the harmonic fingerprint or the raw
    parameter draws — the profile RNG is a distinct derived stream."""
    common = dict(
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=10,
        n_scenarios=6,
        seed=3,
        parameters=[
            ParameterSpec(
                name="pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.6, high=1.4),
                field="pq",
                mode="scale",
            )
        ],
    )
    s0 = sample_coherent_spectra(grid3, CoherentSpectrumConfig(**common))
    s1 = sample_coherent_spectra(
        grid3,
        CoherentSpectrumConfig(
            **common, profile=LoadProfileConfig(), start_time="2024-01-01T00:00:00"
        ),
    )
    for cid in (10, 11):
        for order in (3, 5):
            assert torch.equal(
                s0.harmonic_injection[cid][order][0],
                s1.harmonic_injection[cid][order][0],
            )
            assert torch.equal(
                s0.harmonic_injection[cid][order][1],
                s1.harmonic_injection[cid][order][1],
            )
    for key in ("harmonics_mode", "harmonics_mag", "harmonics_phase", "time_s", "pq"):
        assert torch.equal(s0.all_samples[key], s1.all_samples[key]), key


def test_profile_deterministic_per_seed(grid3):
    cfg = _ccfg(profile=LoadProfileConfig(), start_time="2024-01-01T00:00:00")
    a = load_profile_factors(grid3, cfg).factor
    b = load_profile_factors(grid3, cfg).factor
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_profile_seed_changes_factors(grid3):
    f0 = load_profile_factors(
        grid3, _ccfg(seed=0, profile=LoadProfileConfig(), start_time="2024-01-01")
    ).factor
    f1 = load_profile_factors(
        grid3, _ccfg(seed=1, profile=LoadProfileConfig(), start_time="2024-01-01")
    ).factor
    assert not torch.equal(f0, f1)


# --- class-aware physics ----------------------------------------------------
def test_pv_zero_at_night_and_positive_at_noon(grid_3ph):
    """A pv device's factor is exactly 0 across the night hours and > 0 at solar noon."""
    cfg = _ccfg(
        grid_selector="load",
        n_steps=24,
        n_scenarios=3,
        step_size_s=3600.0,
        profile=LoadProfileConfig(phase_offset_hours=0.0, short_sigma=0.0),
        start_time="2024-06-21T00:00:00",
    )
    fac = load_profile_factors(grid_3ph, cfg).factor  # [B, n_dev, T]
    night = fac[..., list(range(0, 5)) + list(range(20, 24))]
    assert float(night.max()) == 0.0
    assert float(fac[..., 12].min()) > 0.0


def test_pv_seasonal_window_and_amplitude(grid_3ph):
    """Summer has a wider daylight window and a higher peak than winter."""

    def _daylight(start):
        cfg = _ccfg(
            n_steps=24,
            n_scenarios=1,
            step_size_s=3600.0,
            profile=LoadProfileConfig(phase_offset_hours=0.0, short_sigma=0.0),
            start_time=start,
        )
        f = load_profile_factors(grid_3ph, cfg).factor[0, 0, :]
        return int((f > 0).sum()), float(f.max())

    s_hours, s_peak = _daylight("2024-06-21T00:00:00")
    w_hours, w_peak = _daylight("2024-12-21T00:00:00")
    assert s_hours > w_hours
    assert s_peak > w_peak


def test_weekday_weekend_contrast(grid3):
    """With only the weekly term active, weekend consumption is below the weekday level."""
    cfg = _ccfg(
        n_steps=7 * 24,
        n_scenarios=2,
        step_size_s=3600.0,
        profile=LoadProfileConfig(
            phase_offset_hours=0.0,
            short_sigma=0.0,
            daily_amplitude=0.0,
            seasonal_amplitude=0.0,
            weekend_contrast=0.2,
            level_min=1.0,
            level_max=1.0,
        ),
        start_time="2024-01-01T00:00:00",  # a Monday
    )
    day_mean = (
        load_profile_factors(grid3, cfg).factor[0, 0, :].reshape(7, 24).mean(dim=1)
    )
    assert float(day_mean[5:].mean()) < float(day_mean[:5].mean())


def test_factor_non_negative(grid3):
    cfg = _ccfg(
        n_steps=48,
        step_size_s=3600.0,
        profile=LoadProfileConfig(short_sigma=0.5, daily_amplitude=2.0),
        start_time="2024-01-01T00:00:00",
    )
    assert float(load_profile_factors(grid3, cfg).factor.min()) >= 0.0


# --- operating point shape + composition -----------------------------------
def test_operating_point_gains_step_axis(grid3):
    cfg = _ccfg(
        n_steps=8,
        n_scenarios=6,
        step_size_s=3600.0,
        profile=LoadProfileConfig(),
        start_time="2024-01-01T00:00:00",
    )
    s = sample_coherent_spectra(grid3, cfg)
    assert s.operating_point[10]["p_w"].shape == (6, 8)
    assert s.operating_point[10]["q_var"].shape == (6, 8)
    assert s.samples["harmonics_profile_factor"].shape == (6, 2, 8)
    assert s.all_samples["harmonics_profile_device_ids"].tolist() == [10, 11]
    assert s.all_samples["time_unix_s"].shape == (8,)


def test_profile_composes_with_pq_parameter(grid3):
    """A per-scenario pq base draw ([B]) is scaled by the per-step factor into [B, T]."""
    cfg = _ccfg(
        n_steps=8,
        n_scenarios=6,
        step_size_s=3600.0,
        parameters=[
            ParameterSpec(
                name="pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            )
        ],
        profile=LoadProfileConfig(),
        start_time="2024-01-01T00:00:00",
    )
    s = sample_coherent_spectra(grid3, cfg)
    factor = s.samples["harmonics_profile_factor"]  # [B, n_dev, T]
    base = s.samples["pq"][:, 0] * 2000.0  # load 10 base p_w [B] (nominal 2000 W)
    expected = base.unsqueeze(-1) * factor[:, 0, :]  # [B, T]
    torch.testing.assert_close(s.operating_point[10]["p_w"], expected)


# --- solve parity + persistence --------------------------------------------
def _parity_cfg():
    return _ccfg(
        orders=[3, 5],
        n_steps=6,
        n_scenarios=5,
        seed=2,
        step_size_s=3600.0,
        profile=LoadProfileConfig(),
        start_time="2024-03-10T00:00:00",
    )


def test_bt_solve_equals_per_step_loop(grid3):
    res = run_scenarios(grid3, _parity_cfg(), dtype=CDT)
    assert res.v.shape == (5, 6, 3, 3)  # [B, T, H, N]; H=[1,3,5], N=3
    op, inj = res.sampled.operating_point, res.sampled.harmonic_injection
    for b in range(5):
        for t in range(6):
            op1 = {c: {k: v[b, t] for k, v in e.items()} for c, e in op.items()}
            inj1 = {
                c: {o: (mp[0][b, t], mp[1][b, t]) for o, mp in od.items()}
                for c, od in inj.items()
            }
            r1 = solve_harmonic_flow(
                grid3,
                [1, 3, 5],
                operating_point=op1,
                harmonic_injection=inj1,
                dtype=CDT,
            )
            torch.testing.assert_close(res.v[b, t], r1.v, rtol=1e-7, atol=1e-9)


def test_chunked_equals_whole_with_profile(grid3):
    whole = run_scenarios(grid3, _parity_cfg(), dtype=CDT)
    chunked = run_scenarios(grid3, _parity_cfg(), dtype=CDT, chunk_size=2)
    assert chunked.v.shape == whole.v.shape
    torch.testing.assert_close(chunked.v, whole.v, rtol=1e-7, atol=1e-9)


def test_profiled_dataset_round_trip(grid3):
    res = run_scenarios(grid3, _parity_cfg(), dtype=CDT)
    with tempfile.TemporaryDirectory() as td:
        write_dataset(res, td, layout="wide")
        loaded = read_dataset(td)
    torch.testing.assert_close(loaded.v, res.v)
    torch.testing.assert_close(
        loaded.samples["harmonics_profile_factor"],
        res.sampled.samples["harmonics_profile_factor"],
    )
    # absolute epoch seconds survive as float64 (large-magnitude shared sample)
    torch.testing.assert_close(
        loaded.samples["time_unix_s"],
        res.sampled.all_samples["time_unix_s"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        loaded.samples["harmonics_profile_device_ids"],
        res.sampled.all_samples["harmonics_profile_device_ids"],
    )
    assert isinstance(loaded.config, CoherentSpectrumConfig)
    assert loaded.config.profile is not None


def test_profiled_solve_is_differentiable(grid3):
    """A scalar loss over a profiled coherent batch backprops to a grid line-R tensor."""
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    grid3.branches[0].series_resistance_ohm_per_m = r
    res = run_scenarios(grid3, _parity_cfg(), dtype=CDT)
    res.v.abs().sum().backward()
    assert (
        r.grad is not None and torch.isfinite(r.grad).all() and r.grad.abs().sum() > 0
    )


def test_profiled_solve_with_source_uref_scale(grid3):
    """A per-scenario source u_ref draw broadcasts against the per-step [B, T] state."""
    cfg = _ccfg(
        orders=[3, 5],
        n_steps=4,
        n_scenarios=3,
        seed=5,
        step_size_s=3600.0,
        profile=LoadProfileConfig(),
        start_time="2024-03-10T00:00:00",
        parameters=[
            ParameterSpec(
                name="pq",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="slack",
                selector=Selector(component="source"),
                distribution=Uniform(low=0.95, high=1.05),
                field="u_ref",
                mode="scale",
            ),
        ],
    )
    res = run_scenarios(grid3, cfg, dtype=CDT)
    assert res.v.shape == (3, 4, 3, 3)
    scale = res.sampled.operating_point[1]["u_ref_scale"]
    assert tuple(scale.shape) == (3, 1)  # [B, 1]: constant over the sequence
    # the slack row magnitude follows its scenario's scale at every step
    from pgml.schemas.grid_schema import Phase

    slack_row = res.index.row(1, Phase.A)
    v_slack = res.v[:, :, 0, slack_row].abs()  # [B, T] at the fundamental
    expected = 230.0 * scale.to(v_slack.dtype)
    torch.testing.assert_close(v_slack, expected.expand_as(v_slack))
    # per-step loop parity (mixed entry shapes: [B,T] power, [B,1] scale)
    op, inj = res.sampled.operating_point, res.sampled.harmonic_injection
    for b in range(3):
        for t in (0, 3):
            op1 = {}
            for c, e in op.items():
                op1[c] = {
                    k: (v[b, t] if v.shape[-1] == 4 else v[b, 0]) for k, v in e.items()
                }
            inj1 = {
                c: {o: (mp[0][b, t], mp[1][b, t]) for o, mp in od.items()}
                for c, od in inj.items()
            }
            r1 = solve_harmonic_flow(
                grid3,
                [1, 3, 5],
                operating_point=op1,
                harmonic_injection=inj1,
                dtype=CDT,
            )
            torch.testing.assert_close(res.v[b, t], r1.v, rtol=1e-7, atol=1e-9)
