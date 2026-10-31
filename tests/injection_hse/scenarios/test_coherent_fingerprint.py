"""The node-coherent fingerprint sampler: shapes, temporal structure, reproducibility.

Each matched device draws a small bank of base spectra and, over the step sequence, sticks
to one (Markov dwell) while wandering around it (AR(1) jitter) inside the per-order
emission reference. The engine only sees the resulting ``[B, T]`` injection batch.
"""

from __future__ import annotations

import torch

from pgml.scenarios import Selector, en50160_limit, run_scenarios
from injection_hse.data.scenarios import CoherentSpectrumConfig, sample_coherent_spectra


def _ccfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5, 7])
    kw.setdefault("n_steps", 24)
    kw.setdefault("n_scenarios", 8)
    kw.setdefault("n_modes", 2)
    kw.setdefault("seed", 0)
    return CoherentSpectrumConfig(**kw)


def test_coherent_shapes_and_timestamps(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(step_size_s=900.0))
    assert s.samples["harmonics_mode"].shape == (8, 2, 24)  # [B, n_dev, T]
    assert s.samples["harmonics_mag"].shape == (8, 2, 3, 24)  # [B, n_dev, n_ord, T]
    assert s.all_samples["harmonics_device_ids"].tolist() == [10, 11]
    torch.testing.assert_close(
        s.all_samples["time_s"][:3],
        torch.tensor([0.0, 900.0, 1800.0], dtype=torch.float64),
    )
    mag, phase = s.harmonic_injection[10][5]
    assert mag.shape == (8, 24)  # [B, T]


def test_coherent_stickiness_matches_dwell(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(n_scenarios=24, n_steps=60, dwell=0.9))
    mp = s.samples["harmonics_mode"]
    stay = (mp[..., 1:] == mp[..., :-1]).float().mean()
    assert abs(float(stay) - 0.9) < 0.05


def test_coherent_en50160_clamp(grid3):
    # harmonic_reference is opt-in now (the default is IEC 61000-3-2); pin EN 50160.
    s = sample_coherent_spectra(
        grid3, _ccfg(harmonic_reference="en50160", jitter_mag=0.5)
    )  # large jitter exercises clamp
    for order in (3, 5, 7):
        mag, _ = s.harmonic_injection[10][order]
        assert float(mag.max()) <= en50160_limit(order) + 1e-9
        assert float(mag.min()) >= 0.0


def test_coherent_single_mode_runs(grid3):
    s = sample_coherent_spectra(grid3, _ccfg(n_modes=1))
    assert int(s.samples["harmonics_mode"].abs().sum()) == 0  # all mode 0


def test_coherent_reproducible(grid3):
    a = sample_coherent_spectra(grid3, _ccfg())
    b = sample_coherent_spectra(grid3, _ccfg())
    torch.testing.assert_close(
        a.harmonic_injection[10][5][0], b.harmonic_injection[10][5][0]
    )
    torch.testing.assert_close(a.samples["harmonics_mode"], b.samples["harmonics_mode"])


def test_run_coherent_time_axis(grid3):
    res = run_scenarios(grid3, _ccfg(n_scenarios=4, n_steps=10))
    assert res.v.shape == (4, 10, 4, 3)  # [B, T, H, N]; H=[1,3,5,7], N=3 nodes
    assert res.frequencies_hz.tolist() == [50.0, 150.0, 250.0, 350.0]
