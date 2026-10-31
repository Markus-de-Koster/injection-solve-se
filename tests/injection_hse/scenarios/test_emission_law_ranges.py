"""The calibrated emission ranges the randomized recipe feeds into the affine law.

The band each of the law's parameters is drawn from, and the guarantee that zeroing all
three recovers the proportional recipe exactly — the reference point every emission
ablation is measured against.
"""

from __future__ import annotations

import pytest
import torch

from pgml.grids import add_pv_systems, synthetic_feeder
from injection_hse.data.scenarios import se_random_scenario_config
from injection_hse.data.scenarios.presets import (
    EMISSION_FLOOR,
    EMISSION_FLOOR_PHASE_DEG,
    EMISSION_PHASE_SLOPE_DEG,
    SE_PRESET_VERSION,
)

ORDERS = [3, 5, 7]


@pytest.fixture(scope="module")
def pv_grid():
    """A small radial feeder with PV generators (no external dependency)."""
    grid = synthetic_feeder(6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


def test_the_recipe_carries_the_law_for_loads_and_inverters(pv_grid):
    assert SE_PRESET_VERSION == "3"
    cfg = se_random_scenario_config(pv_grid, orders=[1, *ORDERS], n_samples=8, seed=0)
    by_name = {spec.name: spec for spec in cfg.parameters}
    for group, component in (("load_emission", "load"), ("pv_emission", "generator")):
        for suffix, rng in (
            ("floor", EMISSION_FLOOR),
            ("floor_phase", EMISSION_FLOOR_PHASE_DEG),
            ("slope", EMISSION_PHASE_SLOPE_DEG),
        ):
            spec = by_name[f"{group}_{suffix}"]
            assert spec.field == "h_param" and spec.selector.component == component
            assert (spec.distribution.low, spec.distribution.high) == rng
            assert spec.orders == ORDERS and spec.per == "each"


def test_zero_ranges_recover_the_proportional_recipe_exactly(pv_grid):
    v2 = se_random_scenario_config(
        pv_grid,
        orders=[1, *ORDERS],
        n_samples=32,
        seed=0,
        emission_floor=(0.0, 0.0),
        emission_floor_phase_deg=(0.0, 0.0),
        phase_slope_deg=(0.0, 0.0),
    )
    assert not v2.emission_laws
    assert not any(spec.is_free_parameter for spec in v2.parameters)
    v3 = se_random_scenario_config(pv_grid, orders=[1, *ORDERS], n_samples=32, seed=0)
    assert sum(spec.is_free_parameter for spec in v3.parameters) == 6
    assert [law.spec_names() for law in v3.emission_laws] == [
        {role: f"{group}_{role}" for role in ("floor", "floor_phase", "slope")}
        for group in ("load_emission", "pv_emission")
    ]
    # the proportional recipe's draw has NO magnitude-to-loading relation, the law's has
    s2, s3 = v2.sample(pv_grid), v3.sample(pv_grid)
    for s, coupled in ((s2, False), (s3, True)):
        lam = s.samples["load_spectrum_loading"][:, 0]
        ratio = s.samples["load_spectrum_mag"][:, 0, 0] / s.samples["load_spectrum"][
            :, 0, 0
        ].clamp_min(1e-12)
        # ratio / drawn-fraction = cap x law; under the proportional recipe it is the
        # constant cap, under the law it grows as the device unloads
        spread = float(ratio.max() / ratio.min())
        assert (spread > 2.0) == coupled, (coupled, spread)
        if coupled:
            corr = torch.corrcoef(
                torch.stack([lam.to(torch.float64), ratio.to(torch.float64)])
            )[0, 1]
            assert corr < -0.5
