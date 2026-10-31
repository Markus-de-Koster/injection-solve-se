"""Statistical device-class composition of aggregated loads.

Covers: config validation + JSON/dataset round-trips, byte-identity of the fingerprint
when a composition covers a DISJOINT device, determinism + the distinct roster bank, the
load-to-spectrum laws (``lam**gamma`` magnitude, phase slope, phasor-sum cancellation,
the residual-THD cap), the class-aware activity physics (PV zero at night / negative in
the solar window; EV active in the evening not midday; a multi-state device's spectrum
tracking its power state), cross-device correlation, the attribution label shapes, and an
end-to-end ``[B, T, H, N]`` solve (chunk parity + differentiable).
"""

from __future__ import annotations

import math
import tempfile

import pytest
import torch
from pydantic import ValidationError

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.scenarios import (
    ParameterSpec,
    Selector,
    Uniform,
    run_scenarios,
    write_dataset,
)
from injection_hse.data.scenarios import (
    read_dataset,
    ClassCount,
    CoherentSpectrumConfig,
    CompositionConfig,
    ConsumerComposition,
    DeviceClassSpec,
    DeviceState,
    default_device_classes,
    resolve_composed_ids,
    sample_coherent_spectra,
    sample_device_composition,
)

from injection_hse.data.scenarios.composition import (
    _aggregate_injection,
    _mag_law,
    _phase_law,
)


CDT = torch.complex128
W = 2.0 * math.pi * 50.0


def _line(bid, a, b):
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[0.5]],
        series_inductance_h_per_m=[[0.5 / W]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


def _one_load_grid(consumer_type=None, p_nom=2000.0) -> Grid:
    """Single-phase 2-bus grid: source@1, one load@2 (id 10)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[_line(1, 1, 2)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=p_nom,
                q_nom_var=0.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type=consumer_type,
            ),
        ],
    )


def _ccfg(**kw) -> CoherentSpectrumConfig:
    kw.setdefault("selector", Selector(component="load"))
    kw.setdefault("orders", [3, 5])
    kw.setdefault("n_steps", 24)
    kw.setdefault("n_scenarios", 32)
    kw.setdefault("seed", 0)
    kw.setdefault("step_size_s", 3600.0)
    kw.setdefault("composition", CompositionConfig())
    kw.setdefault("start_time", "2024-06-21T00:00:00")
    return CoherentSpectrumConfig(**kw)


# --- config validation + serialization -------------------------------------
def test_composition_requires_start_time():
    with pytest.raises(ValidationError, match="composition requires start_time"):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3],
            n_steps=2,
            composition=CompositionConfig(),
        )


def test_class_names_must_be_unique():
    dup = [
        DeviceClassSpec(name="x", rated_power_w=(1.0, 2.0)),
        DeviceClassSpec(name="x", rated_power_w=(1.0, 2.0)),
    ]
    with pytest.raises(ValidationError, match="unique"):
        CompositionConfig(
            classes=dup,
            compositions=[ConsumerComposition(classes=[ClassCount(class_name="x")])],
        )


def test_unknown_class_reference_rejected():
    with pytest.raises(ValidationError, match="unknown class"):
        CompositionConfig(
            classes=[DeviceClassSpec(name="a", rated_power_w=(1.0, 2.0))],
            compositions=[ConsumerComposition(classes=[ClassCount(class_name="b")])],
        )


def test_device_class_validators():
    with pytest.raises(ValidationError, match="rated_power_w"):
        DeviceClassSpec(name="x", rated_power_w=(2.0, 1.0))
    with pytest.raises(ValidationError, match="loading_min"):
        DeviceClassSpec(
            name="x", rated_power_w=(1.0, 2.0), loading_min=0.9, loading_mean=(0.5, 1.0)
        )
    with pytest.raises(ValidationError, match="orders must be >= 2"):
        DeviceClassSpec(
            name="x", rated_power_w=(1.0, 2.0), harmonic_magnitude={1: (0.1, 0.2)}
        )


def test_composition_config_json_roundtrip():
    cfg = _ccfg(
        composition=CompositionConfig(
            scale_to_nominal=False,
            max_injection_pu=2.5,
            behavioral_coupling=0.4,
            roster_seed=7,
        ),
    )
    back = CoherentSpectrumConfig.model_validate_json(cfg.model_dump_json())
    assert back == cfg
    assert back.composition is not None
    # int-keyed harmonic dicts survive the JSON string-key coercion
    smps = next(c for c in back.composition.classes if c.name == "electronics_smps")
    source = next(c for c in default_device_classes() if c.name == "electronics_smps")
    assert set(smps.harmonic_magnitude) == set(source.harmonic_magnitude)
    assert all(isinstance(k, int) for k in smps.harmonic_magnitude)


# --- byte-identity + determinism -------------------------------------------
def _two_load_grid() -> Grid:
    """source@1, load 10 (household), load 11 (ev) on a 3-bus feeder."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[Node(id=i, u_rated_v=230.0, phases=(Phase.A,)) for i in (1, 2, 3)],
        branches=[_line(1, 1, 2), _line(2, 2, 3)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=0.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="household",
            ),
            Load(
                id=11,
                node=3,
                phases=(Phase.A,),
                p_nom_w=3000.0,
                q_nom_var=0.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="ev_charging",
            ),
        ],
    )


def test_composition_leaves_disjoint_fingerprint_byte_identical():
    """A composition covering load 10 must not disturb the fingerprint of load 11
    (the composition draws on distinct derived streams)."""
    grid = _two_load_grid()
    common = dict(
        selector=Selector(component="load", ids=[11]),
        orders=[3, 5],
        n_steps=8,
        n_scenarios=6,
        seed=4,
        step_size_s=3600.0,
    )
    s_no = sample_coherent_spectra(grid, CoherentSpectrumConfig(**common))
    s_comp = sample_coherent_spectra(
        grid,
        CoherentSpectrumConfig(
            **common,
            composition=CompositionConfig(
                selector=Selector(component="load", ids=[10])
            ),
            start_time="2024-06-21T00:00:00",
        ),
    )
    for order in (3, 5):
        for k in (0, 1):
            assert torch.equal(
                s_no.harmonic_injection[11][order][k],
                s_comp.harmonic_injection[11][order][k],
            )
    for key in ("harmonics_mode", "harmonics_mag", "harmonics_phase", "time_s"):
        assert torch.equal(s_no.all_samples[key], s_comp.all_samples[key]), key
    # the composition covered load 10 (excluded from the fingerprint)
    assert 10 not in s_comp.all_samples["harmonics_device_ids"].tolist()
    assert 10 in s_comp.harmonic_injection


def test_resolve_composed_ids_rules():
    grid = _two_load_grid()  # load 10 household, load 11 ev_charging
    # default library covers both consumer types
    assert resolve_composed_ids(grid, CompositionConfig()) == [10, 11]
    # a per-id override wins; a load with no matching rule + no fallback is uncovered
    comp = CompositionConfig(
        compositions=[
            ConsumerComposition(
                load_ids=[11], classes=[ClassCount(class_name="base_linear")]
            )
        ]
    )
    assert resolve_composed_ids(grid, comp) == [11]
    # a fallback (no consumer_type) then covers everything
    comp_fb = CompositionConfig(
        compositions=[
            ConsumerComposition(classes=[ClassCount(class_name="base_linear")])
        ]
    )
    assert resolve_composed_ids(grid, comp_fb) == [10, 11]


def test_composition_deterministic_per_seed():
    grid = _one_load_grid("household")
    a = sample_device_composition(grid, _ccfg())
    b = sample_device_composition(grid, _ccfg())
    for cid in a.operating_point:
        torch.testing.assert_close(
            a.operating_point[cid]["p_w"], b.operating_point[cid]["p_w"], rtol=0, atol=0
        )
    torch.testing.assert_close(
        a.samples["harmonics_class_p_w"],
        b.samples["harmonics_class_p_w"],
        rtol=0,
        atol=0,
    )


def test_roster_seed_changes_roster():
    grid = _one_load_grid("household")
    r0 = sample_device_composition(
        grid, _ccfg(composition=CompositionConfig(roster_seed=1))
    ).samples["harmonics_roster_p_rated"]
    r1 = sample_device_composition(
        grid, _ccfg(composition=CompositionConfig(roster_seed=2))
    ).samples["harmonics_roster_p_rated"]
    assert not torch.equal(r0, r1)


def test_seed_changes_temporal_realization():
    grid = _one_load_grid("household")
    p0 = sample_device_composition(grid, _ccfg(seed=0)).samples["harmonics_class_p_w"]
    p1 = sample_device_composition(grid, _ccfg(seed=1)).samples["harmonics_class_p_w"]
    assert not torch.equal(p0, p1)


# --- load-to-spectrum laws (unit) -------------------------------------------
def test_magnitude_follows_lam_gamma():
    mr = torch.tensor(0.4)
    lam = torch.tensor([0.25, 0.5, 1.0])
    mag = _mag_law(mr, lam, torch.tensor(-1.5), torch.tensor(1.0))
    torch.testing.assert_close(mag[0] / mag[2], torch.tensor(0.25**-1.5))
    # gamma = 0 -> constant ratio; spectrum_scale multiplies
    flat = _mag_law(mr, lam, torch.tensor(0.0), torch.tensor(2.0))
    torch.testing.assert_close(flat, torch.full_like(lam, 0.8))


def test_phase_follows_lam_slope():
    ph = _phase_law(torch.tensor(10.0), torch.tensor(20.0), torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(ph, torch.tensor([0.0, 10.0]))


def test_aggregate_is_phasor_sum_with_cancellation():
    # two members: 5th at 0 deg and 180 deg, equal fundamental -> cancel
    c = torch.zeros((1, 1, 1, 1), dtype=CDT)
    c[0, 0, 0, 0] = torch.exp(1j * torch.tensor(0.0, dtype=torch.float64)) + torch.exp(
        1j * torch.tensor(math.pi, dtype=torch.float64)
    )
    f = torch.tensor([[[2.0 + 0j]]], dtype=CDT)
    mag, _phase, cap = _aggregate_injection(c, f, 3.0)
    assert float(mag.abs().max()) < 1e-12
    assert float(cap.max()) == 0.0


def test_cap_binds_near_zero_fundamental():
    c = torch.tensor([[[[0.5 + 0j]]]], dtype=CDT)
    f_small = torch.tensor([[[1e-6 + 0j]]], dtype=CDT)
    mag, _p, cap = _aggregate_injection(c, f_small, 3.0)
    assert float(mag[0, 0, 0, 0]) == 3.0 and float(cap[0, 0, 0, 0]) == 1.0
    # truly inactive net fundamental -> no injection, no cap
    f_zero = torch.tensor([[[0.0 + 0j]]], dtype=CDT)
    mag0, _p0, cap0 = _aggregate_injection(c, f_zero, 3.0)
    assert float(mag0[0, 0, 0, 0]) == 0.0 and float(cap0[0, 0, 0, 0]) == 0.0


# --- class-aware activity physics -------------------------------------------
def test_pv_zero_at_night_negative_at_noon():
    grid = _one_load_grid("pv", p_nom=5000.0)
    d = sample_device_composition(grid, _ccfg())
    names = CompositionConfig().class_names()
    pv = names.index("pv_inverter")
    pv_p = d.samples["harmonics_class_p_w"][:, 0, pv, :]  # [B, T]
    night = pv_p[:, list(range(0, 5)) + list(range(21, 24))]
    assert float(night.abs().max()) == 0.0  # bell is exactly 0 at night
    assert float(pv_p[:, 12].mean()) < 0.0  # injecting at solar noon


def test_ev_active_evening_not_midday():
    grid = _one_load_grid("ev_charging", p_nom=8000.0)
    d = sample_device_composition(grid, _ccfg(n_steps=48))
    names = CompositionConfig().class_names()
    ev = names.index("ev_charger")
    act = d.samples["harmonics_class_active"][:, 0, ev, :].double()  # [B, T]
    hours = torch.arange(48) % 24
    evening = act[:, (hours >= 19) & (hours <= 23)].mean()
    midday = act[:, (hours >= 10) & (hours <= 14)].mean()
    assert float(evening) > 3.0 * float(midday)


def test_multistate_spectrum_tracks_power_state():
    """A multi-state device's harmonic fraction rises when its power state is low."""
    grid = _one_load_grid(p_nom=2000.0)
    drive = DeviceClassSpec(
        name="drive",
        emission_class="D",
        sign=1,
        rated_power_w=(2000.0, 2000.0),
        harmonic_magnitude={5: (0.3, 0.3)},
        harmonic_phase_deg={5: (0.0, 0.0)},
        gamma=(0.0, 0.0),
        activity_preset="flat",
        discrete_activity=False,
        loading_min=0.05,
        loading_mean=(1.0, 1.0),
        loading_jitter=0.0,
        states=[
            DeviceState(
                name="heat", power_fraction=0.9, spectrum_scale=0.1, weight=0.5
            ),
            DeviceState(
                name="spin", power_fraction=0.3, spectrum_scale=2.0, weight=0.5
            ),
        ],
        state_dwell=(0.7, 0.7),
    )
    comp = CompositionConfig(
        classes=[drive],
        compositions=[ConsumerComposition(classes=[ClassCount(class_name="drive")])],
        scale_to_nominal=False,
        behavioral_coupling=0.0,  # avail == flat rate == 1 exactly
        cloud_coupling=0.0,
    )
    cfg = _ccfg(
        orders=[5],
        n_steps=40,
        n_scenarios=20,
        seed=3,
        composition=comp,
        start_time="2024-01-01T00:00:00",
    )
    d = sample_device_composition(grid, cfg)
    p = d.samples["harmonics_class_p_w"][:, 0, 0, :]  # [B, T]
    mag = d.harmonic_injection[10][5][0]  # single member -> mag == member fraction
    heat = p > 1500.0  # heating state (0.9 * 2000)
    spin = p < 900.0  # inverter state (0.3 * 2000)
    torch.testing.assert_close(
        mag[heat].mean(), torch.tensor(0.03, dtype=torch.float64), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        mag[spin].mean(), torch.tensor(0.6, dtype=torch.float64), atol=1e-9, rtol=0
    )


def test_single_member_aggregate_follows_lam_gamma():
    """For a one-member load the aggregate 5th-harmonic fraction == m * lam**gamma,
    with lam recovered from the recorded per-class power."""
    grid = _one_load_grid(p_nom=1000.0)
    g = -1.2
    cls = DeviceClassSpec(
        name="c",
        emission_class="D",
        sign=1,
        rated_power_w=(1000.0, 1000.0),
        harmonic_magnitude={5: (0.3, 0.3)},
        harmonic_phase_deg={5: (0.0, 0.0)},
        gamma=(g, g),
        activity_preset="flat",
        discrete_activity=False,
        loading_min=0.1,
        loading_mean=(0.5, 0.5),
        loading_jitter=0.15,
        loading_rho=0.8,
    )
    comp = CompositionConfig(
        classes=[cls],
        compositions=[ConsumerComposition(classes=[ClassCount(class_name="c")])],
        scale_to_nominal=False,
        behavioral_coupling=0.0,
        cloud_coupling=0.0,
        max_injection_pu=100.0,  # keep the cap out of the law check
    )
    cfg = _ccfg(
        orders=[5],
        n_steps=30,
        n_scenarios=16,
        seed=1,
        composition=comp,
        start_time="2024-01-01T00:00:00",
    )
    d = sample_device_composition(grid, cfg)
    p = d.samples["harmonics_class_p_w"][:, 0, 0, :]  # avail(=1)*lam*1000
    lam = p / 1000.0
    mag = d.harmonic_injection[10][5][0]
    torch.testing.assert_close(mag, 0.3 * lam.clamp(min=1e-9) ** g, atol=1e-9, rtol=0)


def test_cap_binding_recorded_on_pv_cancelling_load():
    """A consuming + injecting pair with equal, always-on fundamentals drives the net
    fundamental to ~0, so the residual-harmonic cap binds and is recorded."""
    grid = _one_load_grid(p_nom=3000.0)
    load = DeviceClassSpec(
        name="load",
        sign=1,
        emission_class="D",
        rated_power_w=(3000.0, 3000.0),
        harmonic_magnitude={5: (0.2, 0.2)},
        harmonic_phase_deg={5: (0.0, 0.0)},
        activity_preset="flat",
        discrete_activity=False,
        loading_mean=(1.0, 1.0),
        loading_jitter=0.0,
        loading_min=0.5,
    )
    pv = DeviceClassSpec(
        name="pv",
        sign=-1,
        rated_power_w=(2850.0, 2850.0),  # net fundamental ~150 W
        activity_preset="flat",
        discrete_activity=False,
        loading_mean=(1.0, 1.0),
        loading_jitter=0.0,
        loading_min=0.5,
    )
    comp = CompositionConfig(
        classes=[load, pv],
        compositions=[
            ConsumerComposition(
                classes=[ClassCount(class_name="load"), ClassCount(class_name="pv")]
            )
        ],
        scale_to_nominal=False,
        behavioral_coupling=0.0,
        cloud_coupling=0.0,
        max_injection_pu=3.0,
    )
    cfg = _ccfg(
        orders=[5],
        n_steps=6,
        n_scenarios=4,
        composition=comp,
        start_time="2024-01-01T00:00:00",
    )
    d = sample_device_composition(grid, cfg)
    assert float(d.samples["harmonics_cap_binding"].max()) == 1.0


def test_behavioral_latent_correlates_consumption():
    """A strong behavioral latent makes two consuming loads' aggregate power co-vary."""
    grid = _two_load_grid()
    comp = CompositionConfig(behavioral_coupling=0.6, cloud_coupling=0.0)
    d = sample_device_composition(
        grid,
        _ccfg(selector=Selector(component="load"), n_scenarios=64, composition=comp),
    )
    p = d.samples["harmonics_class_p_w"].sum(dim=2)  # [B, n_agg, T] total per load
    tot = p.sum(dim=-1)  # [B, n_agg] energy per scenario per load
    a, b = tot[:, 0], tot[:, 1]
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert float(corr) > 0.2


# --- attribution labels + persistence ---------------------------------------
def test_label_shapes():
    grid = _two_load_grid()
    cfg = _ccfg(n_steps=8, n_scenarios=5)
    d = sample_device_composition(grid, cfg)
    n_class = len(CompositionConfig().class_names())
    assert d.samples["harmonics_class_p_w"].shape == (5, 2, n_class, 8)
    assert d.samples["harmonics_class_active"].shape == (5, 2, n_class, 8)
    assert d.samples["harmonics_class_active"].dtype == torch.long
    assert d.samples["harmonics_cap_binding"].shape == (5, 2, 2, 8)  # 2 orders
    assert d.samples["harmonics_agg_ids"].tolist() == [10, 11]
    assert d.samples["harmonics_roster_p_rated"].shape[:2] == (2, n_class)


def test_scale_to_nominal_matches_installed_capacity():
    grid = _one_load_grid("household", p_nom=4000.0)
    d = sample_device_composition(grid, _ccfg(composition=CompositionConfig()))
    roster = d.samples["harmonics_roster_p_rated"][0]  # [n_class, max_count]
    assert float(roster.sum()) == pytest.approx(4000.0, rel=1e-9)


def test_composed_dataset_round_trip():
    grid = _two_load_grid()
    res = run_scenarios(grid, _ccfg(n_steps=6, n_scenarios=5, seed=2), dtype=CDT)
    with tempfile.TemporaryDirectory() as td:
        write_dataset(res, td, layout="wide")
        loaded = read_dataset(td)
    torch.testing.assert_close(loaded.v, res.v)
    for key in (
        "harmonics_class_p_w",
        "harmonics_class_active",
        "harmonics_cap_binding",
    ):
        torch.testing.assert_close(
            loaded.samples[key], res.sampled.samples[key], rtol=0, atol=0
        )
    torch.testing.assert_close(
        loaded.samples["harmonics_agg_ids"],
        res.sampled.all_samples["harmonics_agg_ids"],
    )
    torch.testing.assert_close(
        loaded.samples["harmonics_roster_p_rated"],
        res.sampled.all_samples["harmonics_roster_p_rated"],
        rtol=0,
        atol=0,
    )
    assert isinstance(loaded.config, CoherentSpectrumConfig)
    assert loaded.config.composition is not None


# --- end-to-end solve -------------------------------------------------------
def test_end_to_end_solve_bt_hn():
    grid = _two_load_grid()
    res = run_scenarios(grid, _ccfg(n_steps=6, n_scenarios=5, seed=2), dtype=CDT)
    assert res.v.shape == (5, 6, 3, 3)  # [B, T, H=(1,3,5), N=3]
    assert res.converged


def test_chunked_equals_whole():
    grid = _two_load_grid()
    whole = run_scenarios(grid, _ccfg(n_steps=6, n_scenarios=6, seed=2), dtype=CDT)
    chunked = run_scenarios(
        grid, _ccfg(n_steps=6, n_scenarios=6, seed=2), dtype=CDT, chunk_size=2
    )
    torch.testing.assert_close(chunked.v, whole.v, rtol=1e-7, atol=1e-9)


def test_composition_supersedes_parameters_on_covered_load():
    """A pq parameter on a composed load is superseded by the composition; a
    non-composed device keeps its parameter draw (lifted to [B, T])."""
    grid = _two_load_grid()
    cfg = _ccfg(
        n_steps=4,
        n_scenarios=3,
        composition=CompositionConfig(selector=Selector(component="load", ids=[10])),
        parameters=[
            ParameterSpec(
                name="pq",
                selector=Selector(component="load", ids=[11]),
                distribution=Uniform(low=0.7, high=1.3),
                field="pq",
                mode="scale",
            )
        ],
    )
    s = sample_coherent_spectra(grid, cfg)
    # load 10: from composition ([B, T]); load 11: parameter draw lifted to [B, T]
    assert s.operating_point[10]["p_w"].shape == (3, 4)
    assert s.operating_point[11]["p_w"].shape == (3, 4)
    base = s.samples["pq"][:, 0] * 3000.0  # load 11 nominal
    torch.testing.assert_close(
        s.operating_point[11]["p_w"], base.unsqueeze(-1).expand(3, 4)
    )


def test_composed_solve_is_differentiable():
    grid = _two_load_grid()
    r = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    grid.branches[0].series_resistance_ohm_per_m = r
    res = run_scenarios(grid, _ccfg(n_steps=5, n_scenarios=4, seed=2), dtype=CDT)
    res.v.abs().sum().backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    assert float(r.grad.abs().sum()) > 0.0


def test_member_emission_capped_at_iec_fraction():
    """A drawn member ratio above the member's IEC 61000-3-2 emission fraction is
    clamped to it (at the EFFECTIVE scaled power); ratios below the cap are kept."""
    from injection_hse.data.scenarios.composition import (
        _build_roster,
    )
    from pgml.scenarios.iec61000_3_2 import (
        iec61000_3_2_fraction,
    )

    grid = _one_load_grid(p_nom=4000.0)
    comp = CompositionConfig(
        classes=[
            DeviceClassSpec(
                name="hot",
                rated_power_w=(1000.0, 1000.0),
                harmonic_magnitude={3: (0.9, 0.9)},
            ),
            DeviceClassSpec(
                name="mild",
                rated_power_w=(1000.0, 1000.0),
                harmonic_magnitude={3: (0.01, 0.01)},
            ),
        ],
        compositions=[
            ConsumerComposition(
                classes=[
                    ClassCount(class_name="hot", count=(1, 1), power_share=1.0),
                    ClassCount(class_name="mild", count=(1, 1), power_share=1.0),
                ]
            )
        ],
    )
    ids = resolve_composed_ids(grid, comp)
    roster = _build_roster(grid, comp, ids, [3], seed=0)
    # scale_to_nominal: two equal-share 1 kW members scale to 2 kW each (4 kW load)
    assert torch.allclose(roster.p_rated, torch.full((2,), 2000.0, dtype=torch.float64))
    cap = iec61000_3_2_fraction(3, emission_class="A", p_w=2000.0, u_ln_v=230.0)
    assert 0.0 < cap < 0.9
    hot, mild = (
        (0, 1)
        if float(roster.mag_rated[0, 0]) > float(roster.mag_rated[1, 0])
        else (1, 0)
    )
    assert float(roster.mag_rated[hot, 0]) == pytest.approx(cap, rel=1e-9)
    assert float(roster.mag_rated[mild, 0]) == pytest.approx(0.01, rel=1e-9)


def test_emission_floor_is_inert_at_zero_and_matches_the_measured_ratio_law():
    """The affine emission law reduces exactly, and inflates the ratio as measured.

    Measured devices emit a load-INDEPENDENT part, so ``I_h(lam) = A_h + B_h * lam`` and
    the ratio to the fundamental grows as the device unloads. The correction must leave
    the rated operating point untouched at any floor, and must vanish bit-for-bit when the
    floor is zero so a configuration that does not ask for it is unaffected.
    """
    import torch

    from injection_hse.data.scenarios.composition import (
        _emission_affine,
    )

    lam = torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    zero = torch.zeros_like(lam)

    # A zero floor is exactly the proportional law, not merely close to it.
    assert torch.all(_emission_affine(lam, zero, zero) == 1.0 + 0.0j)

    # The rated point is unchanged for any floor and any angle.
    one = torch.ones(1, dtype=torch.float64)
    for f in (0.0, 0.43, 0.61, 0.73):
        for d in (0.0, 120.0):
            rated = _emission_affine(one, one * f, one * d)
            assert torch.allclose(rated, torch.ones_like(rated))

    # Magnitude follows floor/lam + (1 - floor) — the measured ratio inflation.
    f = 0.61
    got = _emission_affine(lam, torch.full_like(lam, f), zero).abs()
    assert torch.allclose(got, f / lam + (1.0 - f))

    # The angle only moves when the two parts are not co-phased.
    assert torch.allclose(
        _emission_affine(lam, torch.full_like(lam, f), zero).angle(), zero
    )
    rotated = _emission_affine(
        lam, torch.full_like(lam, f), torch.full_like(lam, 120.0)
    )
    assert rotated.angle().abs().max() > math.radians(10.0)


def test_upstream_background_reaches_every_node_and_is_inert_when_unset():
    """The background is shared by the whole feeder, and absent unless configured.

    A device-side emission model can only produce a residual that is private to each
    device; what the bench measures is largely COMMON across devices behind one supply.
    The background source is what supplies that, so it has to (a) change the harmonic
    state at nodes far from where any device injects, (b) move them TOGETHER, and (c) do
    nothing at all when no order is configured.
    """
    import torch

    from pgml.scenarios import (
        BackgroundHarmonicConfig,
    )

    grid = _two_load_grid()
    cfg = _ccfg(n_steps=6, n_scenarios=5, seed=2)
    orders = [1, *cfg.orders]

    plain = run_scenarios(grid, cfg, dtype=CDT)
    empty = _ccfg(n_steps=6, n_scenarios=5, seed=2).model_copy(
        update={"background": BackgroundHarmonicConfig()}
    )
    assert torch.equal(run_scenarios(grid, empty, dtype=CDT).v, plain.v), (
        "an unconfigured background must leave the solve untouched"
    )

    order = int(cfg.orders[0])
    with_bg = _ccfg(n_steps=6, n_scenarios=5, seed=2).model_copy(
        update={
            "background": BackgroundHarmonicConfig(
                magnitude_pu={order: 0.03}, drift_std=0.4, drift_rho=0.9
            )
        }
    )
    got = run_scenarios(grid, with_bg, dtype=CDT)
    assert not torch.equal(got.v, plain.v), (
        "a configured background must reach the solve"
    )

    # the fundamental is preserved exactly: the source only acts at h > 1
    h1 = orders.index(1)
    assert torch.allclose(got.v[..., h1, :], plain.v[..., h1, :], atol=1e-9)

    # the background moves the nodes TOGETHER — that shared motion is its whole purpose
    hi = orders.index(order)
    delta = (got.v[..., hi, :] - plain.v[..., hi, :]).reshape(-1, got.v.shape[-1])
    assert delta.abs().min() > 0, "every node should see the upstream background"
    a = delta[:, 0].abs()
    corr = torch.corrcoef(torch.stack([a, delta[:, -1].abs()]).to(torch.float64))[0, 1]
    assert corr > 0.9, (
        f"node responses should share the upstream state, got r={corr:.3f}"
    )


def test_background_snapshot_scenarios_are_independent_and_solver_shaped():
    """A snapshot batch draws the background iid and keeps the ``[B, 1]`` step axis.

    The drift is a per-STEP process: a sequence walks it along the step axis, while a
    snapshot recipe has no step axis to walk, so its scenarios must be independent
    draws from the drift's stationary distribution — a level that is sticky across the
    scenario INDEX would correlate samples that are supposed to be i.i.d. The realized
    tensors carry the same ``[B, T]`` batch shape as the injection tensors, so the
    solve broadcasts per scenario instead of mixing the batch with itself.
    """
    from pgml.scenarios import (
        BackgroundHarmonicConfig,
    )
    from injection_hse.data.scenarios.coherent import (
        build_background_sources,
    )

    grid = _two_load_grid()
    cfg = BackgroundHarmonicConfig(
        magnitude_pu={5: 0.03}, drift_std=0.5, drift_rho=0.99
    )

    def _lag1(x):
        return float(torch.corrcoef(torch.stack([x[:-1], x[1:]]))[0, 1])

    snap = build_background_sources(
        grid, cfg, (4096, 1), torch.Generator().manual_seed(3)
    )
    level = snap[0].spectrum[5][0].squeeze(-1).log()
    assert abs(_lag1(level)) < 0.1, "snapshot levels must be iid across scenarios"

    seq = build_background_sources(
        grid, cfg, (2, 4096), torch.Generator().manual_seed(3)
    )
    assert _lag1(seq[0].spectrum[5][0][0].log()) > 0.9, (
        "a sequence walks the AR(1) along its step axis"
    )

    # End-to-end: the snapshot solve keeps the injections' [B, 1] batch — the
    # background must never broadcast a fresh axis against it.
    base = _ccfg(n_steps=1, n_scenarios=5, seed=2, composition=None)
    plain = run_scenarios(grid, base, dtype=CDT)
    with_bg = _ccfg(n_steps=1, n_scenarios=5, seed=2, composition=None).model_copy(
        update={
            "background": BackgroundHarmonicConfig(
                magnitude_pu={5: 0.03}, drift_std=0.4, drift_rho=0.9
            )
        }
    )
    got = run_scenarios(grid, with_bg, dtype=CDT)
    assert got.v.shape == plain.v.shape == (5, 1, 3, 3)
    orders = [1, *base.orders]
    h1, hi = orders.index(1), orders.index(5)
    assert torch.allclose(got.v[..., h1, :], plain.v[..., h1, :], atol=1e-9)
    assert not torch.equal(got.v[..., hi, :], plain.v[..., hi, :]), (
        "the background must reach the snapshot solve"
    )


def test_background_is_injected_once_per_source_node():
    """Each in-service ``Source`` node carries one injection of the SAME upstream state.

    A multi-feed grid sees one upstream network through several points of common
    coupling: every source node gets one ``NodeHarmonicSource`` with the same realized
    spectrum — the background exists once at the boundary, not once per device and not
    only at the first source — while an explicit ``node_id`` narrows the injection to
    that single node.
    """
    from pgml.scenarios import (
        BackgroundHarmonicConfig,
    )
    from injection_hse.data.scenarios.coherent import (
        build_background_sources,
    )

    base = _two_load_grid()
    second = base.appliances[0].model_copy(update={"id": 2, "node": 3})
    grid = Grid(
        base_frequency_hz=50.0,
        nodes=list(base.nodes),
        branches=list(base.branches),
        appliances=[*base.appliances, second],
    )

    cfg = BackgroundHarmonicConfig(magnitude_pu={5: 0.03}, drift_std=0.4)
    got = build_background_sources(grid, cfg, (8, 1), torch.Generator().manual_seed(1))
    assert [s.node_id for s in got] == [1, 3]
    assert all(s.kind == "voltage" for s in got)
    # one upstream state: the SAME realized spectrum tensor at every coupling point
    assert got[0].spectrum[5][0] is got[1].spectrum[5][0]

    only = build_background_sources(
        grid,
        BackgroundHarmonicConfig(magnitude_pu={5: 0.03}, node_id=3),
        (8, 1),
        torch.Generator().manual_seed(1),
    )
    assert [s.node_id for s in only] == [3]
