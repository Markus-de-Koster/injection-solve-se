"""The fundamental operating point on top of a fingerprint sequence.

``CoherentSpectrumConfig.parameters`` reuses the engine's own ``ScenarioConfig`` machinery
(Sobol cube, copula correlation, per-phase symmetry, the source ``u_ref`` scale) for the
per-scenario fundamental, drawn on a stream distinct from the fingerprint RNG so the
realized harmonic injection does not move when a fundamental variation is added. Harmonic
fields are rejected there: the fingerprint owns the harmonics.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from pgml.scenarios import (
    Normal,
    ParameterSpec,
    Selector,
    Uniform,
    run_scenarios,
)
from injection_hse.data.scenarios import CoherentSpectrumConfig, sample_coherent_spectra


def _coherent(grid, parameters):
    return CoherentSpectrumConfig(
        name="c",
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=3,
        n_scenarios=8,
        n_modes=2,
        seed=0,
        parameters=parameters,
    )


def test_coherent_rejects_harmonic_parameters():
    with pytest.raises(ValidationError):
        CoherentSpectrumConfig(
            selector=Selector(component="load"),
            orders=[3],
            n_steps=2,
            parameters=[
                ParameterSpec(
                    name="h",
                    selector=Selector(component="load"),
                    distribution=Uniform(low=0.0, high=1.0),
                    field="h_mag",
                    orders=[3],
                )
            ],
        )


def test_coherent_parameters_vary_fundamental(grid3):
    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            mode="scale",
            per="each",
        ),
        ParameterSpec(
            name="source_scale",
            selector=Selector(component="source"),
            distribution=Normal(loc=1.0, scale=0.0333),
            field="u_ref",
            mode="scale",
            per="shared",
        ),
    ]
    with_params = run_scenarios(
        grid3,
        _coherent(grid3, params),
        harmonic_orders=[1, 3, 5],
        dtype=torch.complex128,
    )
    without = run_scenarios(
        grid3, _coherent(grid3, []), harmonic_orders=[1, 3, 5], dtype=torch.complex128
    )
    assert with_params.v.shape == without.v.shape  # [B, T, H, N]

    def band(v):  # per-row |V1| spread across scenarios+steps, averaged
        v1 = v[..., 0, :].abs().reshape(-1, v.shape[-1])
        return float(((v1.max(0).values - v1.min(0).values) / v1.mean(0)).mean())

    assert band(with_params.v) > 1e-2  # the fundamental now varies across scenarios
    assert band(without.v) < 1e-9  # the fixed-op fingerprint keeps |V1| constant

    # the coherent operating point + samples record the per-scenario draws ([B], not [B,T])
    assert set(with_params.sampled.operating_point[1]) == {"u_ref_scale"}
    assert with_params.sampled.samples["source_scale"].shape[0] == 8
    assert not without.sampled.operating_point


def test_same_config_seed_identical(grid3):
    cfg = _coherent(
        grid3,
        [
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.3, high=1.0),
                field="pq",
                per="each",
            )
        ],
    )
    a = run_scenarios(grid3, cfg, harmonic_orders=[1, 3, 5], dtype=torch.complex128)
    b = run_scenarios(grid3, cfg, harmonic_orders=[1, 3, 5], dtype=torch.complex128)
    assert torch.equal(a.v, b.v)
    for cid in a.sampled.operating_point:
        for k, v in a.sampled.operating_point[cid].items():
            assert torch.equal(v, b.sampled.operating_point[cid][k])


def test_parameters_do_not_perturb_fingerprint_stream(grid3):
    """The fingerprint RNG is a stream distinct from the operating-point cube: adding
    parameters leaves the realized harmonic_injection byte-identical."""
    empty = sample_coherent_spectra(grid3, _coherent(grid3, []))
    params = [
        ParameterSpec(
            name="load_scale",
            selector=Selector(component="load"),
            distribution=Uniform(low=0.3, high=1.0),
            field="pq",
            per="each",
        ),
        ParameterSpec(
            name="source_scale",
            selector=Selector(component="source"),
            distribution=Normal(loc=1.0, scale=0.0333),
            field="u_ref",
            per="shared",
        ),
    ]
    with_params = sample_coherent_spectra(grid3, _coherent(grid3, params))
    assert set(empty.harmonic_injection) == set(with_params.harmonic_injection)
    for cid, orders in empty.harmonic_injection.items():
        for o, (mag, phase) in orders.items():
            mag2, phase2 = with_params.harmonic_injection[cid][o]
            assert torch.equal(mag, mag2) and torch.equal(phase, phase2)
    # the fingerprint labels (mode path / base magnitudes) are also untouched
    assert torch.equal(empty.samples["c_mode"], with_params.samples["c_mode"])
    assert torch.equal(empty.samples["c_mag"], with_params.samples["c_mag"])
