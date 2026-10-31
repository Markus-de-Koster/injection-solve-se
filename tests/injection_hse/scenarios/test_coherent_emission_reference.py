"""The fingerprint's emission reference and its held-out fingerprint bank.

Every drawn member magnitude is capped at the device's own IEC 61000-3-2 current-emission
fraction, so a fingerprint sequence stays inside the same physical envelope a randomized
per-device draw does. ``mode_bank_seed`` separates the device SIGNATURES from the temporal
realization, which is what an unseen-fingerprint test set needs.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import Selector, iec61000_3_2_device_caps
from injection_hse.data.scenarios import CoherentSpectrumConfig, sample_coherent_spectra


def _ccfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5, 7])
    kw.setdefault("n_steps", 24)
    kw.setdefault("n_scenarios", 8)
    kw.setdefault("seed", 0)
    return CoherentSpectrumConfig(**kw)


def test_coherent_default_reference_is_iec():
    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"), orders=[3, 5, 7], n_steps=4
    )
    assert cfg.harmonic_reference == "iec61000-3-2" and cfg.emission_class == "auto"
    with pytest.raises(ValidationError):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3],
            n_steps=4,
            harmonic_reference="en50160",
            emission_class="B",
        )


def test_coherent_iec_bounded_per_device(grid3):
    caps = iec61000_3_2_device_caps(grid3, [10, 11], [3, 5, 7])
    s = sample_coherent_spectra(
        grid3,
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3, 5, 7],
            n_steps=24,
            n_scenarios=8,
            jitter_mag=0.5,  # large jitter exercises the clamp
        ),
    )
    for cid in (10, 11):
        for order in (3, 5, 7):
            mag, _ = s.harmonic_injection[cid][order]
            assert float(mag.max()) <= caps[cid][order] + 1e-9


def test_mode_bank_seed_none_is_byte_identical(grid3):
    # The default (mode_bank_seed absent) draws the bank from the `seed` stream; an
    # explicit None must reproduce it exactly (byte-identical) -- no behavior change.
    a = sample_coherent_spectra(grid3, _ccfg())
    b = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=None))
    torch.testing.assert_close(
        a.all_samples["harmonics_mode_base_mag"],
        b.all_samples["harmonics_mode_base_mag"],
    )
    torch.testing.assert_close(a.samples["harmonics_mode"], b.samples["harmonics_mode"])
    torch.testing.assert_close(
        a.harmonic_injection[10][5][0], b.harmonic_injection[10][5][0]
    )


def test_mode_bank_seed_pins_distinct_bank(grid3):
    base = sample_coherent_spectra(grid3, _ccfg())
    held_out = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=999))
    # A distinct fingerprint bank while every other setting is shared.
    assert (
        base.all_samples["harmonics_mode_base_mag"]
        - held_out.all_samples["harmonics_mode_base_mag"]
    ).abs().max() > 1e-6
    # and it is itself reproducible.
    again = sample_coherent_spectra(grid3, _ccfg(mode_bank_seed=999))
    torch.testing.assert_close(
        held_out.all_samples["harmonics_mode_base_mag"],
        again.all_samples["harmonics_mode_base_mag"],
    )
