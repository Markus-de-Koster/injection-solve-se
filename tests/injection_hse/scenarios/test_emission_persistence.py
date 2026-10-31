"""Per-device persistent emission draws (``ParameterSpec.per="fixed"`` and the preset's
``emission_persistence="device"``).

A device's harmonic signature — its emission fraction, angle, floor and slope — is a
property of the device, not of the moment. ``per="fixed"`` draws it once per component
and holds it across every scenario of the batch, from a stream seeded by the config seed
and the spec's name, without consuming a sampling dimension; the operating point keeps
varying per scenario. These tests pin the constancy, the seeding, the isolation from the
cube, and that the preset knob reaches every emission spec of loads and PV inverters.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.grids import add_pv_systems, synthetic_feeder
from pgml.scenarios import (
    ParameterSpec,
    ScenarioConfig,
    Selector,
    Uniform,
    sample,
)
from injection_hse.data.scenarios import (
    se_random_scenario_config,
)


@pytest.fixture(scope="module")
def pv_grid():
    grid = synthetic_feeder(n_nodes=6, n_feeders=2)
    assert add_pv_systems(grid, fraction=0.5) > 0
    return grid


def _cfg(per: str, seed: int = 3, n: int = 32) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=n,
        seed=seed,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="pq",
                mode="scale",
                per="each",
            ),
            ParameterSpec(
                name="hm",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                mode="absolute",
                per=per,
                orders=[3, 5],
            ),
        ],
    )


def test_a_fixed_draw_is_constant_across_scenarios_and_differs_across_devices(grid3):
    s = sample(grid3, _cfg("fixed"))
    hm = s.samples["hm"]  # [B, n_dev, n_ord]
    assert torch.all(hm == hm[:1])
    assert not torch.allclose(hm[0, 0], hm[0, 1])
    assert not torch.allclose(hm[0, :, 0], hm[0, :, 1])
    assert torch.all(s.samples["hm_mag"] == s.samples["hm_mag"][:1])
    # the operating point still varies per scenario
    assert s.samples["load_scale"].std(dim=0).min() > 0.0


def test_a_fixed_draw_is_seeded_and_consumes_no_cube_column(grid3):
    a, b = sample(grid3, _cfg("fixed")), sample(grid3, _cfg("fixed"))
    assert torch.equal(a.samples["hm"], b.samples["hm"])
    other = sample(grid3, _cfg("fixed", seed=4))
    assert not torch.allclose(a.samples["hm"], other.samples["hm"])
    # every OTHER draw is where it would be without the fixed spec
    alone = ScenarioConfig(
        n_samples=32, seed=3, parameters=[_cfg("fixed").parameters[0]]
    )
    assert torch.equal(
        sample(grid3, alone).samples["load_scale"], a.samples["load_scale"]
    )
    # ... which is not true of a per-scenario spec, which takes columns of the cube
    each = sample(grid3, _cfg("each"))
    assert not torch.equal(each.samples["hm"][0], each.samples["hm"][1])


def test_fixed_is_a_harmonic_option_only():
    with pytest.raises(ValidationError, match="fixed"):
        ParameterSpec(
            name="p",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),
            field="pq",
            mode="scale",
            per="fixed",
        )


def test_the_preset_knob_reaches_every_emission_spec(pv_grid):
    cfg = se_random_scenario_config(
        pv_grid,
        orders=[1, 3, 5, 7],
        n_samples=16,
        seed=0,
        emission_persistence="device",
    )
    harmonic = [s for s in cfg.parameters if s.is_harmonic]
    assert harmonic and all(s.per == "fixed" for s in harmonic)
    assert all(s.per != "fixed" for s in cfg.parameters if not s.is_harmonic)
    default = se_random_scenario_config(
        pv_grid, orders=[1, 3, 5, 7], n_samples=16, seed=0
    )
    assert all(s.per == "each" for s in default.parameters if s.is_harmonic)
    # sampled: the DRAWS are one per device for the whole batch ...
    s = cfg.sample(pv_grid)
    for key in (
        "load_spectrum",
        "load_emission_floor",
        "load_emission_slope",
        "pv_emission_floor",
    ):
        assert torch.all(s.samples[key] == s.samples[key][:1]), key
    # ... while the REALIZED emission still follows each scenario's own loading through
    # the law (a fixed signature, a moving operating point)
    assert not torch.all(
        s.samples["load_spectrum_mag"] == s.samples["load_spectrum_mag"][:1]
    )
    assert not torch.all(
        s.samples["load_spectrum_loading"] == s.samples["load_spectrum_loading"][:1]
    )


def test_a_class_draw_is_the_same_constant_in_every_dataset(grid3):
    """``per="class"``: one draw for all matched components, held across the batch and
    identical whatever the seed — a class constant."""
    a = sample(grid3, _cfg("class", seed=3))
    b = sample(grid3, _cfg("class", seed=4))
    hm = a.samples["hm"]
    assert hm.shape[1] == 1  # one draw for every matched component ...
    assert torch.all(hm == hm[:1])  # ... held across the scenarios ...
    mag = a.samples["hm_mag"]
    assert torch.allclose(
        mag[:, 0], mag[:, 1]
    )  # ... realized identically per device ...
    assert torch.equal(
        a.samples["hm"], b.samples["hm"]
    )  # ... and identical across seeds
    fixed = sample(grid3, _cfg("fixed", seed=3))
    assert not torch.equal(fixed.samples["hm"], a.samples["hm"])


def test_the_class_persistence_makes_the_law_a_class_constant(pv_grid):
    from pgml.schemas.grid_schema import Load

    cfg = se_random_scenario_config(
        pv_grid, orders=[1, 3, 5], n_samples=8, seed=0, emission_persistence="class"
    )
    named = {n for law in cfg.emission_laws for n in law.spec_names().values()}
    laws = [s for s in cfg.parameters if s.name in named]
    assert laws and all(s.per == "class" for s in laws)
    classes = {
        str(getattr(a.consumer_type, "value", a.consumer_type))
        for a in pv_grid.appliances
        if isinstance(a, Load) and a.in_service and a.consumer_type is not None
    }
    load_specs = [s for s in laws if s.selector.component == "load"]
    assert {s.selector.consumer_type for s in load_specs} - {None} == classes
    untyped = [
        int(a.id)
        for a in pv_grid.appliances
        if isinstance(a, Load) and a.in_service and a.consumer_type is None
    ]
    if untyped:
        by_ids = next(s for s in load_specs if s.selector.ids is not None)
        assert sorted(by_ids.selector.ids) == sorted(untyped)
    # the emission level stays a per-device, population-specific draw
    level = next(s for s in cfg.parameters if s.name == "load_spectrum")
    assert level.per == "fixed"
    # ... and two seeds share the class law but not the levels
    a = cfg.sample(pv_grid)
    b = cfg.model_copy(update={"seed": 1}).sample(pv_grid)
    law_name = next(s.name for s in laws if s.selector.component == "load")
    assert torch.equal(a.samples[law_name], b.samples[law_name])
    assert not torch.equal(a.samples["load_spectrum"], b.samples["load_spectrum"])
