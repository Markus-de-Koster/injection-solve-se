"""The load-dependent emission law of the randomized recipe.

Measured devices emit ``I_h(lam) = A_h + B_h * lam`` — a load-independent floor plus a
load-proportional part — so the ratio to the fundamental grows as a device unloads and
its emission angle rotates with loading. These tests pin what the randomized path must
satisfy: the shared law reduces exactly at zero and has the closed-form ratio inflation;
an :class:`EmissionLaw` folds its drawn parameters into the realized injection against
the loading each device's OWN power draw realised; a zero law is bit-identical to the
proportional draw; the law's specs are validated; and a config written with the sampler
fields of pgml 0.5.0 draws the same batch.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import Constant, ParameterSpec, Selector, Uniform
from pgml.scenarios import sample as sample_without_law
from injection_hse.data.scenarios import (
    LOADING_FLOOR,
    EmissionLaw,
    EmissionLawScenarioConfig,
    affine_emission_correction,
    phase_slope_shift,
)

ORDERS = [3, 5, 7]


def _law_config(
    n: int = 64, *, floor=(0.5, 0.5), delta=(120.0, 120.0), slope=(0.0, 0.0)
):
    """Loads at a drawn scale with a fixed fraction and the given law ranges."""

    def spec(name, field, rng, **kw):
        lo, hi = rng
        return ParameterSpec(
            name=name,
            selector=Selector(component="load"),
            distribution=Constant(value=lo) if lo == hi else Uniform(low=lo, high=hi),
            field=field,
            mode="absolute",
            per="each",
            orders=ORDERS,
            **kw,
        )

    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.0, high=1.0),
            field="pq",
            mode="scale",
            per="each",
        ),
        spec("hm", "h_mag", (0.2, 0.2)),
        spec("hp", "h_phase", (10.0, 10.0)),
        spec("floor", "h_param", floor),
        spec("floor_phase", "h_param", delta),
        spec("slope", "h_param", slope),
    ]
    return EmissionLawScenarioConfig(
        n_samples=n,
        seed=3,
        parameters=params,
        emission_laws=[
            EmissionLaw(floor="floor", floor_phase="floor_phase", slope="slope")
        ],
    )


def sample(grid, config):
    return config.sample(grid)


# =============================================================================
# The shared law
# =============================================================================
def test_the_shared_law_reduces_exactly_and_has_the_closed_form_ratio():
    lam = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    zero = torch.zeros_like(lam)
    assert torch.all(affine_emission_correction(lam, zero, zero) == 1.0 + 0.0j)
    one = torch.ones(1, dtype=torch.float64)
    for f in (0.43, 0.61, 0.73):
        for d in (0.0, 100.0, 150.0):
            assert torch.allclose(
                affine_emission_correction(one, one * f, one * d),
                torch.ones(1, dtype=torch.complex128),
            )
    f = 0.61
    got = affine_emission_correction(lam, torch.full_like(lam, f), zero).abs()
    assert torch.allclose(got, f / lam + (1.0 - f))
    # the explicit slope is zero at rating and linear in the loading
    assert float(phase_slope_shift(torch.tensor(25.0), torch.tensor(1.0))) == 0.0
    assert float(
        phase_slope_shift(torch.tensor(25.0), torch.tensor(0.5))
    ) == pytest.approx(-12.5)


# =============================================================================
# The specs fold the law into the realized injection
# =============================================================================
def test_the_realized_magnitude_follows_each_devices_own_loading(grid3):
    s = sample(grid3, _law_config())
    lam = s.samples["load_scale"]  # [B, n_dev] the drawn loading
    mag = s.samples["hm_mag"]  # [B, n_dev, n_ord] realized (post-law)
    used = s.samples["hm_loading"]  # [B, n_dev] the loading the law read
    assert torch.allclose(used, lam.clamp(min=LOADING_FLOOR).to(used.dtype))
    corr = affine_emission_correction(
        used, torch.full_like(used, 0.5), torch.full_like(used, 120.0)
    )
    expected = 0.2 * corr.abs().unsqueeze(-1).expand_as(mag)
    assert torch.allclose(mag.to(torch.float64), expected.to(torch.float64), rtol=1e-6)
    # unloading inflates the ratio: the least-loaded scenarios carry the largest ratio
    order = torch.argsort(lam[:, 0])
    assert mag[order[0], 0, 0] > mag[order[-1], 0, 0]
    # …and the injection the solver receives IS the realized value
    dev_id = int(s.all_samples["hm_device_ids"][0])
    inj_mag, inj_phase = s.harmonic_injection[dev_id][3]
    assert torch.allclose(inj_mag.to(torch.float64), mag[:, 0, 0].to(torch.float64))


def test_the_realized_phase_rotates_with_loading(grid3):
    s = sample(grid3, _law_config(slope=(30.0, 30.0)))
    used = s.samples["hm_loading"]
    ph = s.samples["hm_phase"]  # [B, n_dev, n_ord]
    corr = affine_emission_correction(
        used, torch.full_like(used, 0.5), torch.full_like(used, 120.0)
    )
    expected = 10.0 + torch.rad2deg(torch.angle(corr)) + 30.0 * (used - 1.0)
    assert torch.allclose(
        ph[..., 0].to(torch.float64), expected.to(torch.float64), atol=1e-9
    )
    # rated devices keep exactly the drawn angle; unloaded ones have moved
    assert torch.all((ph[..., 0] - 10.0).abs()[used >= 0.999] < 1e-6)
    assert torch.all((ph[..., 0] - 10.0).abs()[used < 0.5] > 1.0)


def test_a_zero_law_is_bit_identical_and_the_floor_keeps_it_finite(grid3):
    base = sample(grid3, _law_config(floor=(0.0, 0.0), delta=(0.0, 0.0)))
    same = sample(
        grid3, _law_config(floor=(0.0, 0.0), delta=(0.0, 0.0), slope=(0.0, 0.0))
    )
    assert torch.equal(base.samples["hm_mag"], same.samples["hm_mag"])
    assert torch.allclose(
        base.samples["hm_mag"], torch.full_like(base.samples["hm_mag"], 0.2)
    )
    assert torch.allclose(
        base.samples["hm_phase"], torch.full_like(base.samples["hm_phase"], 10.0)
    )
    # a device drawn at (near) zero load evaluates the law at the floor, never at 0
    s = sample(grid3, _law_config(floor=(0.9, 0.9)))
    assert torch.isfinite(s.samples["hm_mag"]).all()
    assert torch.isfinite(s.samples["hm_phase"]).all()
    bound = 0.2 * float(
        affine_emission_correction(
            torch.tensor(LOADING_FLOOR, dtype=torch.float64),
            torch.tensor(0.9, dtype=torch.float64),
            torch.tensor(120.0, dtype=torch.float64),
        ).abs()
    )
    assert float(s.samples["hm_mag"].max()) <= bound + 1e-9
    assert float(s.samples["hm_mag"].max()) > 0.2 * 5.0  # the floor does inflate it


def test_law_specs_are_validated():
    sel = Selector(component="load")

    def cfg(params, **law):
        return EmissionLawScenarioConfig(
            n_samples=4, parameters=params, emission_laws=[EmissionLaw(**law)]
        )

    def free(low, high, field="h_param"):
        return ParameterSpec(
            name="f",
            selector=sel,
            distribution=Uniform(low=low, high=high),
            field=field,
            mode="absolute",
            orders=[3],
        )

    with pytest.raises(ValidationError, match=r"must be drawn from \[0, 1\]"):
        cfg([free(0.0, 1.5)], floor="f")
    with pytest.raises(ValidationError, match="must name a field='h_param'"):
        cfg([free(0.0, 1.0)], floor="missing")
    with pytest.raises(ValidationError, match="must name a field='h_param'"):
        cfg([free(0.0, 1.0, field="h_phase")], slope="f")
    assert cfg([free(0.0, 1.5)], slope="f").emission_laws[0].spec_names() == {
        "slope": "f"
    }


def test_the_engine_sampler_alone_returns_the_batch_before_the_law(grid3):
    config = _law_config()
    before, after = sample_without_law(grid3, config), config.sample(grid3)
    assert torch.allclose(
        before.samples["hm_mag"], torch.full_like(before.samples["hm_mag"], 0.2)
    )
    assert "hm_loading" not in before.samples
    assert not torch.allclose(after.samples["hm_mag"], before.samples["hm_mag"])
    # every draw, including the law's own parameters, is the same batch
    for key in ("load_scale", "hm", "hp", "floor", "floor_phase", "slope"):
        assert torch.equal(before.samples[key], after.samples[key])
    assert list(after.samples).index("hm_loading") == (
        list(after.samples).index("hm_phase") + 1
    )


def test_a_config_with_the_sampler_fields_of_pgml_0_5_0_draws_the_same_batch(grid3):
    new = _law_config(slope=(-25.0, 25.0))
    legacy = new.model_dump(mode="json")
    del legacy["emission_laws"]
    for spec, field in zip(
        legacy["parameters"][-3:], ("h_floor", "h_floor_phase", "h_slope")
    ):
        spec["field"] = field
    migrated = EmissionLawScenarioConfig.model_validate(legacy)
    assert [s.field for s in migrated.parameters[-3:]] == ["h_param"] * 3
    assert [law.spec_names() for law in migrated.emission_laws] == [
        {"floor": "floor"},
        {"floor_phase": "floor_phase"},
        {"slope": "slope"},
    ]
    a, b = new.sample(grid3), migrated.sample(grid3)
    assert torch.equal(a.samples["hm_mag"], b.samples["hm_mag"])
    assert torch.equal(a.samples["hm_phase"], b.samples["hm_phase"])


# =============================================================================
# Dataset round trip
# =============================================================================
def test_a_dataset_reads_back_with_its_typed_config(grid3, tmp_path):
    import json

    from pgml.scenarios import write_dataset
    from injection_hse.data.scenarios import read_dataset, run_scenarios

    config = _law_config(n=4)
    result = run_scenarios(grid3, config, harmonic_orders=[1, *ORDERS])
    write_dataset(result, tmp_path / "new")
    loaded = read_dataset(tmp_path / "new")
    assert isinstance(loaded.config, EmissionLawScenarioConfig)
    assert loaded.config == config
    assert torch.equal(
        loaded.samples["hm_loading"], result.sampled.samples["hm_loading"]
    )

    # the same dataset as pgml 0.5.0 stored it: a ScenarioConfig with the law's fields
    write_dataset(result, tmp_path / "old")
    meta_path = tmp_path / "old" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = json.loads(meta["config_json"])
    del stored["emission_laws"]
    for spec, field in zip(
        stored["parameters"][-3:], ("h_floor", "h_floor_phase", "h_slope")
    ):
        spec["field"] = field
    meta["config_type"] = meta["config_class"] = "ScenarioConfig"
    meta["config_json"] = json.dumps(stored)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    legacy = read_dataset(tmp_path / "old")
    assert isinstance(legacy.config, EmissionLawScenarioConfig)
    again = legacy.config.sample(grid3)
    assert torch.equal(again.samples["hm_mag"], result.sampled.samples["hm_mag"])


def test_pruning_the_parameters_prunes_the_laws(grid3):
    config = _law_config(slope=(-25.0, 25.0))
    pruned = config.with_parameters(
        [s for s in config.parameters if s.name not in ("floor_phase", "slope")]
    )
    assert [law.spec_names() for law in pruned.emission_laws] == [{"floor": "floor"}]
    assert EmissionLawScenarioConfig.model_validate(pruned.model_dump()) == pruned
    bare = config.with_parameters(config.parameters[:3])
    assert bare.emission_laws == []
    assert torch.allclose(
        bare.sample(grid3).samples["hm_mag"], torch.tensor(0.2, dtype=torch.float64)
    )
