"""The realized spectrum of a composed aggregate is recorded as a sample column.

A composed aggregate has no draw behind it at all — its spectrum is the power-weighted sum
of its members, capped per member at the IEC 61000-3-2 envelope — so without these columns a
written dataset could only be audited by reconstructing ``I(h) = Y(h)*V(h)`` from the state.
The naming follows the one convention every generating path shares: ``<key>_mag`` /
``<key>_phase`` over the device axis ``<key>_device_ids`` or ``<key>_agg_ids``.
"""

from __future__ import annotations

import math
import tempfile

import torch

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
)
from pgml.scenarios import Selector, run_scenarios, write_dataset
from injection_hse.data.scenarios import (
    ClassCount,
    CoherentSpectrumConfig,
    CompositionConfig,
    ConsumerComposition,
    DeviceClassSpec,
    read_dataset,
    sample_coherent_spectra,
    sample_device_composition,
)

W = 2.0 * math.pi * 50.0


# =============================================================================
# Device composition
# =============================================================================
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


def _one_load_grid(p_nom=3000.0) -> Grid:
    """Single-phase 2-bus grid: source@1, one aggregated load@2 (id 10)."""
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
            ),
        ],
    )


def _steady_class(name, *, mag, phase_deg=0.0, rated=1000.0, sign=1) -> DeviceClassSpec:
    """An always-on, fully-loaded member: no activity, loading or law dynamics left."""
    return DeviceClassSpec(
        name=name,
        sign=sign,
        emission_class="D",
        rated_power_w=(rated, rated),
        harmonic_magnitude={5: (mag, mag)},
        harmonic_phase_deg={5: (phase_deg, phase_deg)},
        gamma=(0.0, 0.0),
        activity_preset="flat",
        discrete_activity=False,
        loading_min=1.0,
        loading_mean=(1.0, 1.0),
        loading_jitter=0.0,
    )


def _composed_cfg(classes, **kw) -> CoherentSpectrumConfig:
    comp = CompositionConfig(
        classes=classes,
        compositions=[
            ConsumerComposition(
                classes=[ClassCount(class_name=c.name) for c in classes]
            )
        ],
        scale_to_nominal=False,
        behavioral_coupling=0.0,
        cloud_coupling=0.0,
        **kw,
    )
    return CoherentSpectrumConfig(
        selector=Selector(component="load"),
        orders=[5],
        n_steps=4,
        n_scenarios=3,
        seed=0,
        step_size_s=3600.0,
        composition=comp,
        start_time="2024-06-21T12:00:00",
    )


def test_composed_aggregate_spectrum_is_recorded():
    """One member, no dynamics: the aggregate IS that member's drawn ratio and angle."""
    grid = _one_load_grid()
    config = _composed_cfg([_steady_class("c", mag=0.3, phase_deg=25.0)])

    draw = sample_device_composition(grid, config)

    mag = draw.samples["harmonics_composed_mag"]
    phase = draw.samples["harmonics_composed_phase"]
    assert mag.shape == (3, 1, 1, 4)  # [B, n_agg, n_ord, T]
    assert draw.samples["harmonics_agg_ids"].tolist() == [10]
    torch.testing.assert_close(mag, torch.full_like(mag, 0.3))
    torch.testing.assert_close(phase, torch.full_like(phase, 25.0))
    # and the columns are exactly the injection the solver is handed
    torch.testing.assert_close(mag[:, 0, 0, :], draw.harmonic_injection[10][5][0])
    torch.testing.assert_close(phase[:, 0, 0, :], draw.harmonic_injection[10][5][1])


def test_composed_aggregate_is_the_power_weighted_member_sum():
    """Two in-phase members: the aggregate ratio is their power-weighted mean."""
    grid = _one_load_grid()
    config = _composed_cfg(
        [
            _steady_class("strong", mag=0.4, rated=1000.0),
            _steady_class("weak", mag=0.1, rated=3000.0),
        ]
    )

    draw = sample_device_composition(grid, config)

    power = draw.samples["harmonics_class_p_w"][:, 0, :, :]  # [B, n_class, T]
    expected = (0.4 * power[:, 0, :] + 0.1 * power[:, 1, :]) / power.sum(dim=1)
    torch.testing.assert_close(
        draw.samples["harmonics_composed_mag"][:, 0, 0, :], expected
    )
    # 1 kW at 40 % + 3 kW at 10 % -> 17.5 % of the 4 kW aggregate
    torch.testing.assert_close(expected, torch.full_like(expected, 0.175))


def test_composed_records_the_capped_magnitude():
    """A near-cancelling PV member drives the cap; the CAPPED value is recorded."""
    grid = _one_load_grid()
    config = _composed_cfg(
        [
            _steady_class("load", mag=0.2, rated=3000.0),
            _steady_class("pv", mag=0.0, rated=2850.0, sign=-1),
        ],
        max_injection_pu=3.0,
    )

    draw = sample_device_composition(grid, config)

    binding = draw.samples["harmonics_cap_binding"]
    mag = draw.samples["harmonics_composed_mag"]
    assert float(binding.max()) == 1.0
    assert float(mag.max()) <= 3.0
    bound = mag[binding > 0.0]
    torch.testing.assert_close(bound, torch.full_like(bound, 3.0))


def test_composed_columns_survive_a_dataset_round_trip():
    grid = _one_load_grid()
    config = _composed_cfg([_steady_class("c", mag=0.3, phase_deg=25.0)])
    result = run_scenarios(grid, config, dtype=torch.complex128)

    with tempfile.TemporaryDirectory() as tmp:
        loaded = read_dataset(write_dataset(result, tmp))

    for key in ("harmonics_composed_mag", "harmonics_composed_phase"):
        torch.testing.assert_close(loaded.samples[key], result.sampled.samples[key])


# =============================================================================
# Coherent fingerprint and composition share one naming convention
# =============================================================================
def test_every_path_records_a_mag_phase_pair_on_a_device_axis():
    """A consumer reads one convention: <key>_mag / <key>_phase + the id column."""
    grid = _one_load_grid()

    fingerprint = sample_coherent_spectra(
        grid,
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[5],
            n_steps=4,
            n_scenarios=3,
            seed=0,
        ),
    )
    composed = sample_device_composition(
        grid, _composed_cfg([_steady_class("c", mag=0.3)])
    )

    for samples, key, ids in (
        (fingerprint.all_samples, "harmonics", "harmonics_device_ids"),
        (composed.samples, "harmonics_composed", "harmonics_agg_ids"),
    ):
        mag, phase = samples[f"{key}_mag"], samples[f"{key}_phase"]
        assert mag.shape == phase.shape
        assert mag.shape[1] == samples[ids].numel()  # [B, n_dev, n_ord, T]
        assert mag.shape[2] == 1 and mag.shape[-1] == 4
