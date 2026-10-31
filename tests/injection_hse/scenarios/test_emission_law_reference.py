"""The randomized recipe reproduces recorded realized injections, number for number.

``data/emission_law_reference.json`` holds the harmonic injection, the realized
``<spec>_mag`` / ``<spec>_phase`` / ``<spec>_loading`` records and the law's own values as
the recipe produced them while the emission law was still applied inside pgml's sampler
(pgml 0.5.0, preset version 3). The law now lives in this package and is applied after
pgml's draw, which must not move a single number for the same seed: the three
persistence modes cover the per-scenario cube columns, the per-device streams and the
per-class streams the law's parameters are drawn from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pgml.grids import add_pv_systems, synthetic_feeder
from injection_hse.data.scenarios import (
    affine_emission_correction,
    phase_slope_shift,
    se_random_scenario_config,
)

_REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "emission_law_reference.json").read_text()
)
# Recorded and replayed in float64 on the CPU; the margin covers libm differences in
# exp / atan2 between platforms, nothing else.
_TOL = dict(rtol=1e-12, atol=1e-12)


def _t(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


@pytest.fixture(scope="module")
def pv_grid():
    grid = synthetic_feeder(6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


@pytest.mark.parametrize("persistence", ["scenario", "device", "class"])
def test_the_recipe_reproduces_the_recorded_injection(pv_grid, persistence):
    ref = _REFERENCE["recipes"][persistence]
    n = _REFERENCE["n_samples"]
    config = se_random_scenario_config(
        pv_grid,
        orders=[1, *_REFERENCE["orders"]],
        n_samples=n,
        seed=_REFERENCE["seed"],
        emission_persistence=persistence,
    )
    sampled = config.sample(pv_grid)
    assert {str(cid) for cid in sampled.harmonic_injection} == set(ref["injection"])
    for cid, per_order in sampled.harmonic_injection.items():
        assert {str(o) for o in per_order} == set(ref["injection"][str(cid)])
        for order, (mag, phase) in per_order.items():
            ref_mag, ref_phase = ref["injection"][str(cid)][str(order)]
            torch.testing.assert_close(
                torch.as_tensor(mag, dtype=torch.float64).expand(n), _t(ref_mag), **_TOL
            )
            torch.testing.assert_close(
                torch.as_tensor(phase, dtype=torch.float64).expand(n),
                _t(ref_phase),
                **_TOL,
            )
    for key, values in ref["samples"].items():
        torch.testing.assert_close(
            sampled.samples[key].to(torch.float64), _t(values), **_TOL
        )
    # the law is active in the recorded batch: the loading varies and moves the ratio
    loading = _t(ref["samples"]["load_spectrum_loading"])
    assert float(loading.min()) < 0.5 < float(loading.max())


def test_the_law_reproduces_its_recorded_values():
    ref = _REFERENCE["law"]
    lam = _t(ref["lam"])
    corr = affine_emission_correction(lam, _t(ref["floor"]), _t(ref["delta_deg"]))
    torch.testing.assert_close(corr.real, _t(ref["correction_re"]), **_TOL)
    torch.testing.assert_close(corr.imag, _t(ref["correction_im"]), **_TOL)
    torch.testing.assert_close(
        phase_slope_shift(torch.tensor(-25.0, dtype=torch.float64), lam),
        _t(ref["slope_shift"]),
        **_TOL,
    )
