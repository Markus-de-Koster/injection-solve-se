"""The calibrated state-estimation scenario presets.

Covers: what the recipe puts in a config (reference, correlation, unbalance, slack draw,
per-order emission and phase diversity), that ANY requested order set flows through it,
the silent-order guard on a composed roster plus its escape hatch, and that the composed
generator emits at exactly the requested orders.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pgml.grids import add_pv_systems, synthetic_feeder
from pgml.scenarios import (
    Selector,
)
from injection_hse.data.scenarios import (
    ClassCount,
    CoherentSpectrumConfig,
    CompositionConfig,
    ConsumerComposition,
    DeviceClassSpec,
    composition_silent_orders,
    default_device_classes,
    resolve_composed_ids,
    sample_device_composition,
    se_coherent_scenario_config,
    se_random_scenario_config,
)

from injection_hse.data.scenarios.presets import PV_EMISSION_HIGH, _pv_emission_high

ORDERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


@pytest.fixture(scope="module")
def pv_grid():
    """A small radial feeder with PV generators (no external dependency)."""
    grid = synthetic_feeder(n_nodes=6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


# --- the recipe -------------------------------------------------------------
def test_random_recipe_carries_the_calibrated_shape(pv_grid):
    cfg = se_random_scenario_config(pv_grid, orders=ORDERS, n_samples=8, seed=0)
    by_name = {spec.name: spec for spec in cfg.parameters}

    load = by_name["load_scale"]
    assert load.correlation is not None and load.correlation.rho == 0.5
    assert [f.name for f in cfg.factors] == ["demand"]
    assert load.symmetry == "small_imbalance" and load.imbalance == 0.15
    assert (load.distribution.low, load.distribution.high) == (0.0, 1.0)

    # the slack boundary varies per scenario, mildly
    slack = by_name["source_scale"]
    assert slack.field == "u_ref" and slack.distribution.loc == 1.0
    assert slack.distribution.scale == pytest.approx(0.0333)

    # emission referenced to the appliance CURRENT standard, never the voltage levels
    spectrum = by_name["load_spectrum"]
    assert spectrum.harmonic_reference == "iec61000-3-2"
    assert (spectrum.distribution.low, spectrum.distribution.high) == (0.0, 2.0)
    assert spectrum.orders == [o for o in ORDERS if o > 1]


def test_every_injected_order_gets_a_magnitude_and_a_phase(pv_grid):
    """The requested order set decides the injection — no order is silently skipped."""
    for orders in ([1, 3, 5], [1, 2, 3, 4, 5], ORDERS, [1, 21, 23]):
        cfg = se_random_scenario_config(pv_grid, orders=orders, n_samples=4, seed=0)
        injected = {o for o in orders if o > 1}
        for field, selector in (
            ("h_mag", "load"),
            ("h_phase", "load"),
            ("h_mag", "generator"),
            ("h_phase", "generator"),
        ):
            covered = {
                o
                for spec in cfg.parameters
                if spec.field == field and spec.selector.component == selector
                for o in spec.orders
            }
            assert covered == injected, (orders, field, selector)


def test_phase_spans_widen_with_order_and_saturate(pv_grid):
    cfg = se_random_scenario_config(pv_grid, orders=ORDERS, n_samples=4, seed=0)
    spans = {
        o: spec.distribution.high
        for spec in cfg.parameters
        if spec.field == "h_phase" and spec.selector.consumer_type is None
        for o in spec.orders
    }
    assert spans[3] < spans[5] < spans[7] < spans[9] < spans[11] < spans[13]
    assert spans[13] == spans[19] == 180.0


def test_pv_emission_envelope_continues_beyond_the_table():
    """An order the measured table does not cover still emits, on a declining envelope."""
    assert _pv_emission_high(17) == PV_EMISSION_HIGH[17]  # the measured h17 bump
    assert 0.0 < _pv_emission_high(21) < _pv_emission_high(19)
    # even orders sit well below the neighbouring odd ones (symmetric converter)
    assert _pv_emission_high(4) < _pv_emission_high(5)


def test_a_fundamental_only_run_injects_nothing(pv_grid):
    cfg = se_random_scenario_config(pv_grid, orders=[1], n_samples=4, seed=0)
    assert not [spec for spec in cfg.parameters if spec.is_harmonic]
    assert [spec.name for spec in cfg.parameters] == [
        "load_scale",
        "pv_scale",
        "source_scale",
    ]


def test_an_order_beyond_the_iec_table_still_emits(pv_grid):
    """h41 has no IEC 61000-3-2 limit; the envelope references the draw there."""
    cfg = se_random_scenario_config(pv_grid, orders=[1, 12, 41], n_samples=4, seed=0)
    ratio = cfg.sample(pv_grid).samples["load_spectrum_mag"]  # [B, n_dev, 2]
    assert bool((ratio > 0).all())


def test_coherent_recipe_composes_at_the_high_activity_anchor(pv_grid):
    cfg = se_coherent_scenario_config(
        pv_grid, orders=ORDERS, n_scenarios=2, n_steps=3, seed=0
    )
    assert cfg.start_time == "2024-06-21T16:00:00"
    assert cfg.composition is not None and cfg.harmonic_reference == "iec61000-3-2"
    # the composition covers the loads, so the fingerprint addresses the generators
    assert cfg.selector.component == "generator" and cfg.profile is not None
    # the fundamental is the randomized recipe's, drawn once per sequence
    assert [spec.name for spec in cfg.parameters] == [
        "load_scale",
        "pv_scale",
        "source_scale",
    ]


def test_the_fingerprint_mode_is_timeless_and_load_borne(pv_grid):
    cfg = se_coherent_scenario_config(
        pv_grid, orders=ORDERS, n_scenarios=2, n_steps=3, seed=0, mode="fingerprint"
    )
    assert cfg.composition is None and cfg.start_time is None and cfg.profile is None
    assert cfg.selector.component == "load"


def test_a_coherent_sequence_needs_a_harmonic_order(pv_grid):
    from pgml.errors import InputError

    with pytest.raises(InputError, match="at least one harmonic order"):
        se_coherent_scenario_config(
            pv_grid, orders=[1], n_scenarios=2, n_steps=2, seed=0
        )


# --- the silent-order guard -------------------------------------------------
def _odd_only_roster() -> CompositionConfig:
    return CompositionConfig(
        classes=[
            DeviceClassSpec(
                name="odd_only",
                rated_power_w=(100.0, 200.0),
                harmonic_magnitude={3: (0.1, 0.2), 5: (0.05, 0.1)},
            )
        ],
        compositions=[ConsumerComposition(classes=[ClassCount(class_name="odd_only")])],
    )


def test_silent_orders_reports_what_no_class_emits():
    assert composition_silent_orders(default_device_classes(), [1, 3, 19]) == []
    assert composition_silent_orders(default_device_classes(), [1, 4, 21, 50]) == []
    roster = _odd_only_roster().classes
    assert composition_silent_orders(roster, [1, 3, 4, 7]) == [4, 7]


def test_a_roster_silent_at_a_solved_order_is_rejected():
    with pytest.raises(
        ValidationError, match=r"no device class emits at order\(s\) \[7\]"
    ):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3, 5, 7],
            n_steps=2,
            composition=_odd_only_roster(),
            start_time="2024-06-21T16:00:00",
        )


def test_declared_silence_is_allowed():
    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[3, 5, 7],
        n_steps=2,
        composition=_odd_only_roster(),
        start_time="2024-06-21T16:00:00",
        allow_silent_orders=(7,),
    )
    assert cfg.allow_silent_orders == (7,)


def test_a_reference_without_the_order_is_rejected_on_the_fingerprint_path():
    with pytest.raises(ValidationError, match="has no limit at order"):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3, 50],
            n_steps=2,
            harmonic_reference="en50160",
        )
    # the IEC 61000-3-2 reference caps every order
    CoherentSpectrumConfig(
        selector=Selector(component="load"), orders=[3, 41], n_steps=2
    )


def test_the_built_in_library_covers_the_full_odd_range():
    """The h15-h19 hole must be impossible in the recommended configuration."""
    cfg = CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[o for o in ORDERS if o > 1],
        n_steps=2,
        composition=CompositionConfig(),
        start_time="2024-06-21T16:00:00",
    )
    assert composition_silent_orders(cfg.composition.classes, list(cfg.orders)) == []


# --- the composed generator honours the requested orders --------------------
def test_composition_emits_at_exactly_the_requested_orders():
    from pgml.schemas.grid_schema import ConsumerType, Load

    grid = synthetic_feeder(n_nodes=4, n_feeders=1)
    for appliance in grid.appliances:
        if isinstance(appliance, Load):
            appliance.consumer_type = ConsumerType.HOUSEHOLD
    for orders in ([3, 5], [3, 9, 19]):
        cfg = CoherentSpectrumConfig(
            selector=Selector(component="generator"),
            orders=orders,
            n_steps=3,
            n_scenarios=2,
            composition=CompositionConfig(),
            start_time="2024-06-21T16:00:00",
        )
        draw = sample_device_composition(grid, cfg)
        composed = resolve_composed_ids(grid, cfg.composition)
        assert composed
        for device in composed:
            assert sorted(draw.harmonic_injection[device]) == orders
