"""Recipe v4 of the workflow driver: grid-model options, the v4 scenario file, chunk sizing.

Pins: the device capacitance is one seeded wye shunt per load / PV generator sized per kW
and enters the estimator's ``Y``; the TB 575 source sets ``|Z| = 4`` ohm with R = X at
20 kV; the v4 recipe equals the preset with per-device emission persistence except the
emission angles (and, for the composed sequences, the class angle ranges); and an
operating-point-shunt solve is chunked only when its system tensors would not fit.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "run" / "examples" / "injection_hse"))

V4 = _REPO / "run" / "configs" / "pgl_se_scenario_v4.yaml"
CAPACITANCE = _REPO / "run" / "configs" / "pgl_device_capacitance.yaml"
ORDERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


@pytest.fixture(scope="module")
def cigre():
    import train_se_full_workflow as wf

    grid, _ = wf.build_grid(0.5)
    return grid


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        n_random=8,
        n_scenarios=4,
        n_steps=2,
        n_modes=2,
        dwell=0.9,
        seed=0,
        scenario_random=None,
        scenario_coherent=None,
        scenario_config=None,
        coherent_mode="composed",
        coherent_step_s=900.0,
        profile_start=None,
        activity_scale=1.0,
        behavioral_coupling=0.3,
        cloud_coupling=0.5,
        chunk_size=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


# =============================================================================
# Device capacitance
# =============================================================================
def test_device_capacitance_is_seeded_sized_per_kw_and_order_independent(cigre):
    from pgml.schemas.grid_schema import Generator, Load, ShuntAppliance

    import train_se_full_workflow as wf

    config = wf.load_device_capacitance(CAPACITANCE)
    grid = cigre.model_copy(deep=True)
    records = wf.add_device_capacitance(grid, config)
    loads = [a for a in cigre.appliances if isinstance(a, Load)]
    pvs = [a for a in cigre.appliances if isinstance(a, Generator)]
    assert len(records) == len(loads) + len(pvs)
    shunts = {a.id: a for a in grid.appliances if isinstance(a, ShuntAppliance)}
    devices = {a.id: a for a in cigre.appliances}
    ids = [a.id for a in grid.appliances]
    assert len(ids) == len(set(ids))
    for rec in records:
        device, shunt = devices[rec["device_id"]], shunts[rec["shunt_id"]]
        low, high = (
            config["pv_uf_per_kw"]
            if rec["kind"] == "pv"
            else config["load_uf_per_kw"]["default"]
        )
        assert low <= rec["uf_per_kw"] <= high
        assert shunt.node == device.node and shunt.phases == device.phases
        expected = rec["uf_per_kw"] * abs(device.p_nom_w) / 1e3 / len(device.phases)
        assert list(shunt.capacitance_f) == [pytest.approx(expected * 1e-6)] * 3
        assert list(shunt.conductance_s) == [0.0] * 3
    # the draw depends on (seed, device id) only: a grid without its first load draws
    # the same densities for everyone else
    reduced = cigre.model_copy(deep=True)
    reduced.appliances = [a for a in reduced.appliances if a.id != loads[0].id]
    again = {
        r["device_id"]: r["uf_per_kw"]
        for r in wf.add_device_capacitance(reduced, config)
    }
    assert all(again[r["device_id"]] == r["uf_per_kw"] for r in records[1:])
    other = wf.add_device_capacitance(
        cigre.model_copy(deep=True), {**config, "seed": 1}
    )
    assert other[0]["uf_per_kw"] != records[0]["uf_per_kw"]


def test_device_capacitance_enters_the_estimators_admittance(cigre):
    from pgml.schemas.grid_schema import Grid

    import train_se_full_workflow as wf
    from injection_hse.physics import NetworkSolver

    grid = cigre.model_copy(deep=True)
    records = wf.add_device_capacitance(grid, wf.load_device_capacitance(CAPACITANCE))
    reloaded = Grid.model_validate_json(grid.model_dump_json())
    plain = NetworkSolver.from_grid(cigre, [1, 5], dtype=torch.complex128)
    with_c = NetworkSolver.from_grid(reloaded, [1, 5], dtype=torch.complex128)
    # the capacitance is purely susceptive: the diagonal at the shunted rows rises by
    # j*h*w*C on every phase
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre)
    rec = records[0]
    node = next(a.node for a in cigre.appliances if a.id == rec["device_id"])
    row = int(torch.nonzero(idx.node_ids == node)[0])
    c_total = sum(
        r["c_per_phase_uf"] * 1e-6
        for r in records
        if next(a.node for a in cigre.appliances if a.id == r["device_id"]) == node
    )
    w5 = 2 * math.pi * 50.0 * 5
    delta = with_c.y_harmonic[0, row, row] - plain.y_harmonic[0, row, row]
    assert delta.imag == pytest.approx(w5 * c_total, rel=1e-9)
    assert abs(delta.real) < 1e-12 * abs(plain.y_harmonic[0, row, row])


def test_invalid_capacitance_configs_are_rejected(tmp_path):
    import train_se_full_workflow as wf

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"device_capacitance": {"load_uf_per_kw": {}}}))
    with pytest.raises(SystemExit, match="default"):
        wf.load_device_capacitance(path)
    path.write_text(
        yaml.safe_dump(
            {
                "device_capacitance": {
                    "load_uf_per_kw": {"default": [2.0, 1.0]},
                    "pv_uf_per_kw": [1.0, 4.0],
                }
            }
        )
    )
    with pytest.raises(SystemExit, match="low <= high"):
        wf.load_device_capacitance(path)


# =============================================================================
# Source impedance
# =============================================================================
def test_tb575_source_is_4_ohm_with_r_equal_x(cigre):
    from pgml.schemas.grid_schema import Source

    import train_se_full_workflow as wf

    grid = cigre.model_copy(deep=True)
    assert wf.apply_source_impedance(grid, "pgml") is None
    assert grid.model_dump_json() == cigre.model_dump_json()
    record = wf.apply_source_impedance(grid, "cigre-tb575")
    src = next(a for a in grid.appliances if isinstance(a, Source))
    r = src.resistance_ohm[0][0]
    x = 2 * math.pi * 50.0 * src.inductance_h[0][0]
    assert math.hypot(r, x) == pytest.approx(4.0, rel=1e-12)
    assert r == pytest.approx(x, rel=1e-12)
    assert src.resistance_ohm[0][1] == 0.0
    assert record["sources"][0]["u_rated_v"] == 20000.0


# =============================================================================
# The v4 scenario recipe
# =============================================================================
def _normal(spec) -> tuple:
    d = spec.distribution
    return d.kind, float(d.loc), float(d.scale)


def test_v4_random_section_against_the_device_persistent_preset(cigre):
    """v4 = the preset with per-device persistence, except the emission it re-sizes."""
    import train_se_full_workflow as wf
    from injection_hse.data.scenarios import default_device_classes, se_random_scenario_config

    rnd, _ = wf.load_scenario_config(V4)
    args = _args(scenario_random=rnd)
    v4 = wf.random_scenario_config(cigre, ORDERS, ORDERS[1:], args)
    preset = se_random_scenario_config(
        cigre,
        orders=ORDERS[1:],
        n_samples=args.n_random,
        seed=args.seed,
        emission_persistence="device",
    )
    by_name = {s.name: s for s in v4.parameters}
    # the operating point and the PV emission law are the preset's
    for spec in preset.parameters:
        if spec.field in ("pq", "u_ref") or spec.name.startswith("pv_emission_"):
            assert by_name[spec.name] == spec
    assert [law.floor for law in v4.emission_laws] == ["pv_emission_floor"]
    assert v4.emission_laws[0] in preset.emission_laws
    # aggregated loads: an absolute ratio of their own current per consumer type and
    # order, redrawn per snapshot, no IEC reference and no device law
    loads = [s for s in v4.parameters if s.name.startswith("load_spectrum_")]
    assert {s.selector.consumer_type for s in loads} == {
        "household",
        "office",
        "workshop",
    }
    assert sorted({o for s in loads for o in s.orders}) == ORDERS[1:]
    for spec in loads:
        assert spec.field == "h_mag" and spec.mode == "absolute" and spec.per == "each"
        assert spec.harmonic_reference is None
        assert spec.distribution.kind == "uniform" and spec.distribution.low == 0.0
    assert not any(s.name.startswith("load_emission_") for s in v4.parameters)
    # PV generators: the composition's pv_inverter class ranges, fixed per device
    pv_class = next(c for c in default_device_classes() if c.name == "pv_inverter")
    for order in ORDERS[1:]:
        spec = by_name[f"pv_spectrum_h{order}"]
        lo, hi = pv_class.harmonic_magnitude[order]
        assert (spec.distribution.low, spec.distribution.high) == (lo, hi)
        assert spec.per == "fixed"
    # angles: normal, fixed per device
    phases = {s.name: s for s in v4.parameters if s.field == "h_phase"}
    assert set(phases) == {
        "spectrum_phase_low",
        "spectrum_phase_high",
        "pv_phase_low",
        "pv_phase_high",
    }
    for name, spec in phases.items():
        low = name.endswith("_low")
        assert list(spec.orders) == ([3, 5, 7, 9] if low else [11, 13, 15, 17, 19])
        assert _normal(spec) == ("normal", 0.0, 32.5 if low else 65.0)
        assert spec.per == "fixed"


def test_v4_coherent_section_is_the_preset_with_matched_class_angles(cigre):
    import train_se_full_workflow as wf

    _, coh = wf.load_scenario_config(V4)
    v4 = wf.coherent_scenario_config(
        cigre, ORDERS, ORDERS[1:], _args(scenario_coherent=coh)
    )
    preset = wf.coherent_scenario_config(cigre, ORDERS, ORDERS[1:], _args())
    restored = v4.composition.model_copy(update={"classes": preset.composition.classes})
    assert (
        v4.model_copy(
            update={
                "composition": restored,
                "phase_distribution": preset.phase_distribution,
            }
        )
        == preset
    )
    assert v4.phase_distribution.kind == "normal"
    assert float(v4.phase_distribution.scale) == 32.5
    for cls in v4.composition.classes:
        for order, (lo, hi) in cls.harmonic_phase_deg.items():
            width = 55.0 if int(order) <= 9 else 105.0
            assert (lo, hi) == (-width, width)


def test_v4_angle_widths_match_the_normal_resultant_lengths():
    """The uniform half-widths reproduce the normal's mean resultant length to 0.01."""
    for sd, half in ((32.5, 55.0), (65.0, 105.0)):
        normal = math.exp(-(math.radians(sd) ** 2) / 2)
        uniform = math.sin(math.radians(half)) / math.radians(half)
        assert abs(normal - uniform) < 0.01


# =============================================================================
# Chunk sizing of an operating-point-shunt solve
# =============================================================================
def test_operating_point_solves_are_chunked_only_when_they_do_not_fit(
    cigre, monkeypatch
):
    import train_se_full_workflow as wf

    cfg = wf.coherent_scenario_config(cigre, ORDERS, ORDERS[1:], _args(n_scenarios=40))
    per_scenario = 3 * 2 * 9 * 132**2 * 16  # copies x T x H x N^2 x complex128
    dtype = torch.complex128
    cpu = torch.device("cpu")

    def chunk(free, **overrides):
        monkeypatch.setattr(wf, "_free_memory_bytes", lambda device: free)
        return wf.scenario_dependent_chunk(
            cfg, ORDERS, cigre, _args(**overrides), cpu, dtype
        )

    # the batch fits: the requested chunk is kept as it was (None = one batch)
    assert chunk(10**12) is None
    assert chunk(10**12, chunk_size=7) == 7
    # it does not: capped to half the free memory over three system tensors
    assert chunk(10 * per_scenario) == 5
    assert chunk(10 * per_scenario, chunk_memory_fraction=0.2) == 2
    assert chunk(10 * per_scenario, chunk_size=3) == 3
    # a scenario-independent Y(h) is never capped
    assert chunk(1, load_shunt="opendss", load_shunt_basis="nameplate") is None
    assert chunk(1, load_shunt="none") is None
    assert chunk(1, load_shunt="opendss", load_shunt_basis="operating_point") == 1


def test_the_cli_background_reaches_a_scenario_file_recipe(cigre):
    """``--background-distortion`` applies to a ``--scenario-config`` recipe as to the preset."""
    import train_se_full_workflow as wf

    rnd, coh = wf.load_scenario_config(V4)
    bg = dict(
        background_distortion="5:0.008,7:0.006",
        background_drift=0.3,
        background_drift_phase_deg=0.0,
        background_sc_mva=100.0,
    )
    random_cfg = wf.random_scenario_config(
        cigre, ORDERS, ORDERS[1:], _args(scenario_random=rnd, **bg)
    )
    coherent_cfg = wf.coherent_scenario_config(
        cigre, ORDERS, ORDERS[1:], _args(scenario_coherent=coh, **bg)
    )
    for cfg in (random_cfg, coherent_cfg):
        assert cfg.background is not None
        assert cfg.background.magnitude_pu == {5: 0.008, 7: 0.006}
        assert cfg.background.source_power_va == 100e6
    # without the option the file's recipe is unchanged
    plain = wf.coherent_scenario_config(
        cigre, ORDERS, ORDERS[1:], _args(scenario_coherent=coh)
    )
    assert plain.background is None
    # a file section with its own background and the CLI option are refused together
    coh_bg = coh.model_copy(update={"background": coherent_cfg.background})
    with pytest.raises(ValueError, match="set it in one place"):
        wf.coherent_scenario_config(
            cigre, ORDERS, ORDERS[1:], _args(scenario_coherent=coh_bg, **bg)
        )
