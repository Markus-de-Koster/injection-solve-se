"""Discrete dominant emitters on a randomized batch (``injection_hse.data.scenarios.discrete``).

Pins: an unset field changes neither the serialized config nor the batch; per scenario a
device is in one state of its class and the load's injection is the background plus the
device phasors over the summed power; ``fixed`` persistence keeps a device's spectrum across
scenarios; the IEC 61000-3-2 cap matches pgml's scalar function; rules match by id, type
and fallback; a batch with emitters solves and round-trips through a dataset.
"""

from __future__ import annotations

import math

import pytest
import torch
from pgml.scenarios import ParameterSpec, Selector

from injection_hse.data.scenarios.discrete import _member_caps
from injection_hse.data.scenarios import (
    DeviceClassSpec,
    DeviceState,
    DiscreteEmitterConfig,
    EmissionLawScenarioConfig,
    EmitterCount,
    EmitterRule,
    apply_emission_laws,
    read_dataset,
    run_scenarios,
)

ORDERS = [3, 5]


def _background(n: int = 256, seed: int = 0, **extra) -> EmissionLawScenarioConfig:
    """Loads at a random level with a small background emission at h3 and h5."""
    return EmissionLawScenarioConfig(
        n_samples=n,
        seed=seed,
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution={"kind": "uniform", "low": 0.2, "high": 1.0},
                field="pq",
                mode="scale",
            ),
            ParameterSpec(
                name="bg_mag",
                selector=Selector(component="load"),
                distribution={"kind": "uniform", "low": 0.01, "high": 0.03},
                field="h_mag",
                mode="absolute",
                orders=ORDERS,
            ),
            ParameterSpec(
                name="bg_phase",
                selector=Selector(component="load"),
                distribution={"kind": "normal", "loc": 0.0, "scale": 30.0},
                field="h_phase",
                mode="absolute",
                orders=ORDERS,
            ),
        ],
        **extra,
    )


def _ev(**overrides) -> DeviceClassSpec:
    values = dict(
        name="ev",
        rated_power_w=(7400.0, 11000.0),
        power_factor=0.99,
        emission_class="C",  # a fraction independent of power keeps the draws uncapped
        harmonic_magnitude={3: (0.01, 0.02), 5: (0.05, 0.08)},
        harmonic_phase_deg={3: (-10.0, 10.0), 5: (-10.0, 10.0)},
        states=[
            DeviceState(
                name="idle", power_fraction=0.01, spectrum_scale=0.0, weight=0.6
            ),
            DeviceState(name="charging", power_fraction=1.0, weight=0.4),
        ],
    )
    values.update(overrides)
    return DeviceClassSpec(**values)


def _emitters(**overrides) -> DiscreteEmitterConfig:
    values = dict(
        classes=[_ev()],
        rules=[
            EmitterRule(
                consumer_type="household", emitters=[EmitterCount(class_name="ev")]
            )
        ],
    )
    values.update(overrides)
    return DiscreteEmitterConfig(**values)


def test_unset_emitters_leave_the_serialized_config_and_the_batch_unchanged(grid3):
    from pgml.scenarios import sample

    cfg = _background(n=16)
    assert "discrete_emitters" not in cfg.model_dump_json()
    assert cfg.model_dump_json() == cfg.model_dump_json(exclude={"discrete_emitters"})
    a = cfg.sample(grid3)
    b = apply_emission_laws(grid3, sample(grid3, cfg), cfg)
    assert a.operating_point.keys() == b.operating_point.keys()
    for cid in a.harmonic_injection:
        for order, (mag, phase) in a.harmonic_injection[cid].items():
            assert torch.equal(mag, b.harmonic_injection[cid][order][0])
            assert torch.equal(phase, b.harmonic_injection[cid][order][1])
    assert a.samples.keys() == b.samples.keys()


def test_injection_is_background_plus_device_phasors_over_summed_power(grid3):
    cfg = _background(discrete_emitters=_emitters())
    plain = _background().sample(grid3)
    out = cfg.sample(grid3)
    assert out.shared_samples["emitters_device_ids"].tolist() == [10]
    assert out.shared_samples["emitters_orders"].tolist() == ORDERS
    state = out.samples["emitters_state"][:, 0, 0]
    power = out.samples["emitters_power_w"][:, 0, 0]
    # one state per scenario, drawn from the stationary weights (0.6 / 0.4)
    assert set(state.tolist()) == {0, 1}
    assert abs(float((state == 1).double().mean()) - 0.4) < 0.1
    # the device's power is added to the load's own draw
    p_bg = plain.operating_point[10]["p_w"]
    assert torch.allclose(
        out.operating_point[10]["p_w"], p_bg + power, rtol=0, atol=1e-9
    )
    for k, order in enumerate(ORDERS):
        mag0, ph0 = plain.harmonic_injection[10][order]
        mag, ph = out.harmonic_injection[10][order]
        c_bg = mag0 * p_bg * torch.exp(1j * torch.deg2rad(ph0))
        c = mag * (p_bg + power) * torch.exp(1j * torch.deg2rad(ph))
        device = c - c_bg
        idle = state == 0
        # idle: no emission, the background current is unchanged
        assert torch.allclose(
            device[idle].abs(),
            torch.zeros(int(idle.sum()), dtype=torch.float64),
            atol=1e-6,
        )
        # charging: a fixed device phasor (fixed persistence) of its rated ratio times power
        charging = ~idle
        ratio = device[charging].abs() / power[charging]
        lo, hi = _ev().harmonic_magnitude[order]
        assert float(ratio.min()) >= lo - 1e-9 and float(ratio.max()) <= hi + 1e-9
        assert float(ratio.std()) < 1e-9
        assert torch.equal(out.samples["emitters_mag"][:, 0, k], mag)
    # the h_mag spec's realized record follows the injection
    assert torch.equal(
        out.samples["bg_mag_mag"][:, 0, 1], out.harmonic_injection[10][5][0]
    )
    # the EV-charging load has no rule: untouched
    assert torch.equal(
        out.harmonic_injection[11][5][0], plain.harmonic_injection[11][5][0]
    )


def test_scenario_persistence_redraws_the_device(grid3):
    def ratios(persistence):
        cfg = _background(
            discrete_emitters=_emitters(
                classes=[_ev(states=[DeviceState(power_fraction=1.0)])],
                persistence=persistence,
            )
        )
        plain = _background().sample(grid3)
        out = cfg.sample(grid3)
        p_bg = plain.operating_point[10]["p_w"]
        power = out.samples["emitters_power_w"][:, 0, 0]
        mag0, ph0 = plain.harmonic_injection[10][5]
        mag, ph = out.harmonic_injection[10][5]
        device = mag * (p_bg + power) * torch.exp(
            1j * torch.deg2rad(ph)
        ) - mag0 * p_bg * torch.exp(1j * torch.deg2rad(ph0))
        return device.abs() / power, power

    fixed, p_fixed = ratios("fixed")
    redrawn, p_redrawn = ratios("scenario")
    assert float(fixed.std()) < 1e-9 and float(p_fixed.std()) < 1e-6
    assert float(redrawn.std()) > 1e-3 and float(p_redrawn.std()) > 100.0


def test_presence_off_probability_and_nameplate_share(grid3):
    cfg = _background(
        n=512,
        discrete_emitters=_emitters(
            rules=[
                EmitterRule(
                    consumer_type="household",
                    emitters=[
                        EmitterCount(
                            class_name="ev",
                            count=(2, 2),
                            size_basis="nameplate",
                            nameplate_share=(0.5, 0.5),
                            off_probability=0.5,
                        )
                    ],
                )
            ]
        ),
    )
    out = cfg.sample(grid3)
    state = out.samples["emitters_state"][:, 0]  # [B, 2]
    power = out.samples["emitters_power_w"][:, 0]
    assert out.shared_samples["emitters_class_index"].tolist() == [[0, 0]]
    off = state == -1
    assert abs(float(off.double().mean()) - 0.5) < 0.08
    assert torch.all(power[off] == 0.0)
    # charging draws exactly half the 2 kW nameplate
    assert torch.allclose(power[state == 1], torch.tensor(1000.0, dtype=torch.float64))


@pytest.mark.parametrize(
    ("share", "low", "high"),
    [((0.1, 0.2), 7400.0, 11000.0), ((10.0, 10.0), 20000.0, 20000.0)],
)
def test_larger_size_basis_takes_the_bigger_rating(grid3, share, low, high):
    single = [DeviceState(power_fraction=1.0)]
    cfg = _background(
        n=4,
        discrete_emitters=_emitters(
            classes=[_ev(states=single)],
            persistence="scenario",
            rules=[
                EmitterRule(
                    consumer_type="household",
                    emitters=[
                        EmitterCount(
                            class_name="ev", size_basis="larger", nameplate_share=share
                        )
                    ],
                )
            ],
        ),
    )
    power = cfg.sample(grid3).samples["emitters_power_w"][:, 0, 0]
    # a 2 kW household: the class rating wins at 10-20 %, the share at 10x
    assert float(power.min()) >= low - 1e-6 and float(power.max()) <= high + 1e-6


def test_rules_match_by_id_then_type_then_fallback(grid3):
    ev = _ev(states=[DeviceState(power_fraction=1.0)])
    hp = _ev(name="hp")
    cfg = _background(
        n=4,
        discrete_emitters=DiscreteEmitterConfig(
            classes=[ev, hp],
            rules=[
                EmitterRule(
                    consumer_type="household", emitters=[EmitterCount(class_name="ev")]
                ),
                EmitterRule(load_ids=[10], emitters=[EmitterCount(class_name="hp")]),
                EmitterRule(
                    emitters=[
                        EmitterCount(class_name="ev"),
                        EmitterCount(class_name="hp"),
                    ]
                ),
            ],
        ),
    )
    out = cfg.sample(grid3)
    assert out.shared_samples["emitters_device_ids"].tolist() == [10, 11]
    # load 10: the id rule wins over its type; load 11 (no household tag): the fallback
    assert out.shared_samples["emitters_class_index"].tolist() == [[1, -1], [0, 1]]


@pytest.mark.parametrize("emission_class", ["A", "B", "C", "D", None])
def test_vectorised_iec_cap_matches_the_shared_cap(emission_class):
    from injection_hse.data.scenarios.emission_caps import iec61000_3_2_cap

    orders = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    powers = torch.tensor(
        [10.0, 80.0, 300.0, 700.0, 3000.0, 25000.0], dtype=torch.float64
    )
    cap = _member_caps(orders, emission_class, powers, 230.0, 0.95)
    for i, p in enumerate(powers.tolist()):
        cls = emission_class or ("D" if 75.0 <= p <= 600.0 else "A")
        for k, order in enumerate(orders):
            want = iec61000_3_2_cap(
                order, emission_class=cls, p_w=p, u_ln_v=230.0, power_factor=0.95
            )
            assert math.isclose(float(cap[i, k]), want, rel_tol=1e-12, abs_tol=1e-15)


def test_streams_are_seeded_and_named(grid3):
    first = _background(discrete_emitters=_emitters()).sample(grid3)
    again = _background(discrete_emitters=_emitters()).sample(grid3)
    renamed = _background(discrete_emitters=_emitters(name="other")).sample(grid3)
    assert torch.equal(first.samples["emitters_state"], again.samples["emitters_state"])
    assert not torch.equal(
        first.samples["emitters_state"], renamed.samples["other_state"]
    )
    # pgml's own draws do not move
    assert torch.equal(first.samples["load_scale"], renamed.samples["load_scale"])


def test_config_validation():
    with pytest.raises(ValueError, match="unknown class"):
        _emitters(rules=[EmitterRule(emitters=[EmitterCount(class_name="nope")])])
    with pytest.raises(ValueError, match="nameplate_share is required"):
        EmitterCount(class_name="ev", size_basis="nameplate")
    with pytest.raises(ValueError, match="nameplate_share is required"):
        EmitterCount(class_name="ev", nameplate_share=(0.1, 0.2))
    with pytest.raises(ValueError, match="sign=1"):
        _emitters(classes=[_ev(sign=-1)])
    with pytest.raises(ValueError, match="h_mag / h_phase"):
        EmissionLawScenarioConfig(
            n_samples=2, parameters=[], discrete_emitters=_emitters()
        )


def test_batch_with_emitters_solves_and_round_trips(grid_3ph, tmp_path):
    from pgml.scenarios import write_dataset

    cfg = _background(
        n=8,
        discrete_emitters=_emitters(
            rules=[EmitterRule(emitters=[EmitterCount(class_name="ev", count=(1, 2))])]
        ),
    )
    result = run_scenarios(
        grid_3ph, cfg, calculation="harmonic", harmonic_orders=[1, *ORDERS]
    )
    assert bool(torch.as_tensor(result.converged).all())
    path = write_dataset(result, tmp_path / "ds")
    loaded = read_dataset(path)
    assert isinstance(loaded.config, EmissionLawScenarioConfig)
    assert loaded.config == cfg
    assert torch.equal(
        torch.as_tensor(loaded.samples["emitters_state"]),
        result.sampled.samples["emitters_state"],
    )
    assert torch.as_tensor(loaded.samples["emitters_class_index"]).shape == (2, 2)
