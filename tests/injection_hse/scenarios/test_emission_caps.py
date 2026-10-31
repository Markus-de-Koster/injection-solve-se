"""Emission caps and spectra at every analyser order (h2..h50, odd and even).

Pins: the IEC 61000-3-2 cap equals pgml's table value wherever the table lists the order
and follows the standard's ``1/h`` envelope elsewhere, so no order is silent; the
randomized recipe references a draw at such an order to the envelope instead of pgml's
zero and leaves every tabulated order untouched; the library continuation keeps the
tabulated bands and the draws at the orders it already covered; and the composed preset
runs at the driver's default orders 1..20 (formerly an error for h12-h20).
"""

from __future__ import annotations

import pytest
import torch

from pgml.grids import add_pv_systems, synthetic_feeder
from pgml.scenarios import (
    iec61000_3_2_device_caps,
    iec61000_3_2_fraction,
    iec61000_3_2_limits,
)
from pgml.scenarios import sample as pgml_sample
from injection_hse.data.scenarios import (
    ASYMMETRIC_EVEN_ORDER_RATIO,
    EVEN_ORDER_RATIO,
    CompositionConfig,
    DeviceClassSpec,
    continue_emission_bands,
    default_device_classes,
    iec61000_3_2_cap,
    iec61000_3_2_caps,
    se_coherent_scenario_config,
    se_random_scenario_config,
)

ALL_ORDERS = list(range(2, 51))


@pytest.fixture(scope="module")
def pv_grid():
    grid = synthetic_feeder(n_nodes=6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


# --- the cap ------------------------------------------------------------------
@pytest.mark.parametrize("cls", ["A", "B", "C", "D"])
def test_cap_is_the_table_value_where_listed_and_positive_everywhere(cls):
    kw = dict(emission_class=cls, p_w=300.0, u_ln_v=230.0, power_factor=0.95)
    listed = iec61000_3_2_limits(cls)["limits"]
    for h in ALL_ORDERS:
        cap = iec61000_3_2_cap(h, **kw)
        assert 0.0 < cap <= 1.0, (cls, h)
        if h in listed:
            assert cap == iec61000_3_2_fraction(h, **kw)


def test_cap_continues_the_standard_formulas():
    i1 = 2000.0 / 230.0  # a 2 kW Class A device, unity power factor
    kw = dict(emission_class="A", p_w=2000.0, u_ln_v=230.0)
    for h in (41, 43, 49):  # odd: 0.15 * 15 / h A, extrapolated above h39
        assert iec61000_3_2_cap(h, **kw) == pytest.approx(0.15 * 15 / h / i1, rel=1e-12)
    for h in (12, 20, 40, 50):  # even: 0.23 * 8 / h A (pgml tabulates to h10)
        assert iec61000_3_2_cap(h, **kw) == pytest.approx(0.23 * 8 / h / i1, rel=1e-12)
    # the envelope joins the table: h39 (listed) to h41 (continued) is the 1/h step
    assert iec61000_3_2_cap(41, **kw) / iec61000_3_2_cap(39, **kw) == pytest.approx(
        39 / 41, rel=1e-3
    )
    # Class D odd: 3.85 / h mA/W; Class D even: the Class A limit of the order
    d = dict(emission_class="D", p_w=300.0, u_ln_v=230.0)
    assert iec61000_3_2_cap(45, **d) == pytest.approx(3.85 / 45 * 0.230, rel=1e-12)
    assert iec61000_3_2_cap(12, **d) == iec61000_3_2_cap(
        12, **{**d, "emission_class": "A"}
    )
    # Class C odd: the flat 3 % continued
    assert iec61000_3_2_cap(45, emission_class="C", p_w=50.0, u_ln_v=230.0) == 0.03


def test_device_caps_match_pgml_at_listed_orders(pv_grid):
    from pgml.schemas.grid_schema import Generator, Load

    ids = [a.id for a in pv_grid.appliances if isinstance(a, (Load, Generator))]
    ours = iec61000_3_2_caps(pv_grid, ids, ALL_ORDERS)
    table = iec61000_3_2_device_caps(pv_grid, ids, ALL_ORDERS)
    for cid in ids:
        for h in ALL_ORDERS:
            if table[cid][h] > 0.0:
                assert ours[cid][h] == table[cid][h]
            assert ours[cid][h] > 0.0


# --- the randomized recipe ------------------------------------------------------
def test_random_recipe_references_unlisted_orders_and_keeps_listed_ones(pv_grid):
    full = se_random_scenario_config(
        pv_grid, orders=[1, 3, 12, 41], n_samples=6, seed=0
    )
    # without the emission law the realized ratio is the raw fraction times the cap
    cfg = full.with_parameters([p for p in full.parameters if p.field != "h_param"])
    raw = pgml_sample(pv_grid, cfg)  # pgml's batch: zero at h12 / h41
    drawn = cfg.sample(pv_grid)
    ids = drawn.shared_samples["load_spectrum_device_ids"].tolist()
    caps = iec61000_3_2_caps(pv_grid, ids, [12, 41])
    fraction = raw.samples["load_spectrum"]  # [B, n_dev, 3]
    before = raw.samples["load_spectrum_mag"]
    after = drawn.samples["load_spectrum_mag"]
    assert bool((before[..., 1:] == 0).all())
    for j, cid in enumerate(ids):
        for k, h in ((1, 12), (2, 41)):
            expect = fraction[:, j, k].double() * caps[cid][h]
            assert torch.allclose(after[:, j, k], expect, rtol=1e-12, atol=0.0)
            mag, _ = drawn.harmonic_injection[cid][h]
            assert torch.equal(torch.as_tensor(mag), after[:, j, k])
    # the listed order is pgml's draw
    assert torch.equal(after[..., 0], before[..., 0])
    # with the law, the unlisted orders emit as well
    assert bool((full.sample(pv_grid).samples["load_spectrum_mag"] > 0).all())


def test_a_listed_order_set_is_returned_untouched(pv_grid):
    from injection_hse.data.scenarios.emission import _cap_unlisted_orders

    cfg = se_random_scenario_config(pv_grid, orders=[1, 3, 5, 7], n_samples=4, seed=0)
    raw = pgml_sample(pv_grid, cfg)
    assert _cap_unlisted_orders(pv_grid, raw, cfg) is raw


# --- the device library ---------------------------------------------------------
def test_band_continuation_rules():
    cls = DeviceClassSpec(
        name="x",
        rated_power_w=(100.0, 200.0),
        harmonic_magnitude={3: (0.3, 0.6), 5: (0.1, 0.2), 19: (0.019, 0.038)},
        harmonic_phase_deg={3: (-10.0, 10.0), 19: (-90.0, 90.0)},
    )
    out = continue_emission_bands(cls, max_order=50, even_ratio=0.1)
    mag, phase = out.harmonic_magnitude, out.harmonic_phase_deg
    assert sorted(mag) == ALL_ORDERS
    assert mag[3] == (0.3, 0.6) and mag[19] == (0.019, 0.038)  # tabulated kept
    assert mag[7] == pytest.approx((0.1 * 5 / 7, 0.2 * 5 / 7))  # 1/h from h5
    assert mag[49] == pytest.approx((0.019 * 19 / 49, 0.038 * 19 / 49))
    assert mag[2] == pytest.approx((0.03, 0.06))  # h2 from h3 alone
    assert mag[4] == pytest.approx((0.1 * 0.2, 0.1 * 0.4))  # mean of h3 / h5
    assert mag[50] == pytest.approx(
        (
            0.1 * (0.019 * 19 / 49 + 0.019 * 19 / 51) / 2,
            0.1 * (0.038 * 19 / 49 + 0.038 * 19 / 51) / 2,
        )
    )
    assert phase[2] == (-10.0, 10.0) and phase[11] == (-10.0, 10.0)  # tie -> lower
    assert phase[12] == (-90.0, 90.0) and phase[50] == (-90.0, 90.0)
    silent = DeviceClassSpec(name="r", rated_power_w=(1.0, 2.0))
    assert continue_emission_bands(silent) is silent


def test_library_emits_at_every_order_to_h50():
    for cls in default_device_classes():
        if not cls.harmonic_magnitude:
            continue
        assert sorted(cls.harmonic_magnitude) == ALL_ORDERS, cls.name
        ratio = (
            ASYMMETRIC_EVEN_ORDER_RATIO
            if cls.name == "inverter_drive"
            else EVEN_ORDER_RATIO
        )
        lo3, hi3 = cls.harmonic_magnitude[3]
        assert cls.harmonic_magnitude[2] == pytest.approx((ratio * lo3, ratio * hi3))


def test_library_continuation_keeps_the_draws_at_the_orders_it_covered(pv_grid):
    """A run at the odd orders up to h19 draws what the library drew before."""
    orders = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    trimmed = [
        c.model_copy(
            update={
                "harmonic_magnitude": {
                    o: r for o, r in c.harmonic_magnitude.items() if o in orders
                },
                "harmonic_phase_deg": {
                    o: r for o, r in c.harmonic_phase_deg.items() if o in orders
                },
            }
        )
        for c in default_device_classes()
    ]
    kw = dict(orders=orders, n_scenarios=3, n_steps=4, seed=0)
    full = se_coherent_scenario_config(pv_grid, **kw).sample(pv_grid)
    old = se_coherent_scenario_config(
        pv_grid, composition=CompositionConfig(classes=trimmed), **kw
    ).sample(pv_grid)
    for key in ("coherent_composed_mag", "coherent_composed_phase", "coherent_mag"):
        assert torch.equal(full.samples[key], old.samples[key]), key


def test_composed_preset_runs_at_the_default_orders(pv_grid):
    """Orders 1..20 (the driver default) compose without a silent or unreferenced order."""
    cfg = se_coherent_scenario_config(
        pv_grid, orders=list(range(1, 21)), n_scenarios=2, n_steps=3, seed=0
    )
    s = cfg.sample(pv_grid)
    composed = s.samples["coherent_composed_mag"]  # [B, n_agg, n_ord, T]
    assert composed.shape[2] == 19
    assert bool((composed.amax(dim=(0, 1, 3)) > 0).all())
    fingerprint = s.samples["coherent_mag"]  # the PV generators
    assert bool((fingerprint.amax(dim=(0, 1, 3)) > 0).all())


def test_fingerprint_silent_orders_zero_only_those_orders(pv_grid):
    """Silent fingerprint orders are zero; every other draw is unchanged, the hash too."""
    from pgml.scenarios import config_hash

    kw = dict(orders=[1, 3, 5, 9], n_scenarios=2, n_steps=3, seed=0)
    base = se_coherent_scenario_config(pv_grid, **kw)
    silent = base.model_copy(update={"fingerprint_silent_orders": (3, 9)})
    a, b = base.sample(pv_grid), silent.sample(pv_grid)
    mag_a, mag_b = a.samples["coherent_mag"], b.samples["coherent_mag"]
    assert bool((mag_b[:, :, [0, 2]] == 0).all())
    assert torch.equal(mag_a[:, :, 1], mag_b[:, :, 1])
    assert torch.equal(
        a.samples["coherent_composed_mag"], b.samples["coherent_composed_mag"]
    )
    # a config that does not use the field serializes, and hashes, as before
    assert "fingerprint_silent_orders" not in base.model_dump()
    assert config_hash(silent) != config_hash(base)
