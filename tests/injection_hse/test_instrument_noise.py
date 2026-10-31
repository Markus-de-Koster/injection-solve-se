"""The datasheet instrument error model: sigmas per regime, held part, changes, determinism."""

from __future__ import annotations

import math

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    InstrumentBand,
    InstrumentErrorConfig,
    MaskStageConfig,
    MeasurementConfig,
    MultiGridConfig,
    NoiseConfig,
    OptimConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import InstrumentNoise

ORDERS = [1, 3, 5]
U_NOM = 230.0
I_NOM = 80.0


def _voltage_model(spec=None, *, rows: int = 4, seed: int = 0) -> InstrumentNoise:
    return InstrumentNoise(
        spec or InstrumentErrorConfig(),
        quantity="voltage",
        nominal=torch.full((rows,), U_NOM),
        orders=ORDERS,
        seed=seed,
    )


def _readings(n: int, magnitudes, *, steps: int = 1, rows: int = 4) -> torch.Tensor:
    """``[n, steps, H, rows]`` complex128 readings of the given per-order magnitudes."""
    angles = torch.linspace(-2.0, 2.5, rows, dtype=torch.float64)
    mag = torch.tensor(magnitudes, dtype=torch.float64).reshape(-1, 1)
    y = torch.polar(mag.expand(-1, rows), angles.expand(len(magnitudes), -1))
    return y.expand(n, steps, -1, -1).clone()


def _component_sigma(err: torch.Tensor, dims) -> torch.Tensor:
    """Empirical per-real-component std (isotropic average) of complex errors."""
    return (err.abs().pow(2).mean(dim=dims) / 2.0).sqrt()


def test_default_preset_is_the_datasheet():
    spec = InstrumentErrorConfig()
    assert spec.coverage == 2.0 and spec.stable_fraction == 1.0
    assert spec.voltage_fundamental == InstrumentBand(
        reading_pct=0.1, phase_deg=0.02, floor_pct=0.01, threshold_pct=10.0
    )
    assert spec.voltage_harmonic == InstrumentBand(
        reading_pct=5.0, phase_deg=0.5, floor_pct=0.05, threshold_pct=1.0
    )
    assert spec.current_fundamental == InstrumentBand(
        reading_pct=0.1,
        phase_deg=0.1,
        floor_pct=0.01,
        threshold_pct=10.0,
        fsr_over_nominal=2.0,
    )
    assert spec.current_harmonic == InstrumentBand(
        reading_pct=5.0, phase_deg=0.5, floor_pct=0.15, threshold_pct=3.0
    )
    model = _voltage_model()
    assert torch.allclose(model.gain_sigma, torch.tensor([5e-4, 0.025, 0.025]).double())
    assert torch.allclose(
        model.phase_sigma_rad,
        torch.tensor(
            [math.radians(0.01), math.radians(0.25), math.radians(0.25)]
        ).double(),
    )
    floor = model.floor_sigma()[:, 0]
    # the fundamental floor gives the proportional error of the threshold reading
    assert torch.allclose(floor, torch.tensor([5e-5, 2.5e-4, 2.5e-4]).double() * U_NOM)
    assert torch.allclose(
        model.threshold()[:, 0], torch.tensor([0.1, 0.01, 0.01]).double() * U_NOM
    )
    # the current fundamental's threshold and floor refer to FSR = 2·I_nom
    edges = model.for_edges(torch.arange(2), current_range_a=I_NOM)
    assert torch.allclose(
        edges.threshold()[:, 0], torch.tensor([0.2, 0.03, 0.03]).double() * I_NOM
    )
    assert torch.allclose(
        edges.floor_sigma()[:, 0], torch.tensor([1e-4, 7.5e-4, 7.5e-4]).double() * I_NOM
    )


@pytest.mark.parametrize("stable", [1.0, 0.5, 0.0])
def test_empirical_sigma_matches_per_regime(stable):
    """Fundamental, a proportional harmonic and a floor harmonic, voltage and current."""
    spec = InstrumentErrorConfig(stable_fraction=stable)
    n = 20000
    # voltage: 230 V fundamental, 9 V (>= 2.3 V: proportional), 1 V (< 2.3 V: floor)
    v = _readings(n, [U_NOM, 9.0, 1.0])[:, 0]
    model = _voltage_model(spec, seed=4)
    emp = _component_sigma(model.apply(v) - v, 0)
    cfg = model.reading_sigma(v[0].abs())
    assert torch.allclose(emp, cfg, rtol=0.03)
    # the harmonic regimes: 2.5 % of 9 V (plus the floor) and the floor of 0.025 % U_nom
    harm_prop = math.sqrt(
        0.5 * (0.025**2 + math.radians(0.25) ** 2) * 81 + (U_NOM * 2.5e-4) ** 2
    )
    assert torch.allclose(cfg[1], torch.full((4,), harm_prop, dtype=torch.float64))
    assert torch.allclose(cfg[2], torch.full((4,), U_NOM * 2.5e-4, dtype=torch.float64))
    # current: 40 A fundamental (>= 16 A), 4 A at h=3 (>= 2.4 A), 1 A at h=5 (floor)
    edges = model.for_edges(torch.arange(4), current_range_a=I_NOM)
    i = _readings(n, [40.0, 4.0, 1.0])[:, 0]
    emp_i = _component_sigma(edges.apply(i) - i, 0)
    cfg_i = edges.reading_sigma(i[0].abs())
    assert torch.allclose(emp_i, cfg_i, rtol=0.03)
    fund = math.sqrt(
        0.5 * (5e-4**2 + math.radians(0.05) ** 2) * 1600 + (I_NOM * 1e-4) ** 2
    )
    assert math.isclose(float(cfg_i[0, 0]), fund, rel_tol=1e-12)
    assert torch.allclose(
        cfg_i[2], torch.full((4,), I_NOM * 7.5e-4, dtype=torch.float64)
    )


def test_regime_follows_each_reading():
    """One channel's readings straddle the threshold: each takes its own regime."""
    model = _voltage_model()
    thr = float(model.threshold()[1, 0])
    mags = torch.tensor(
        [[U_NOM, thr * 0.99, 1.0], [U_NOM, thr, 1.0]], dtype=torch.float64
    )
    sig = model.reading_sigma(mags.unsqueeze(-1).expand(-1, -1, 4))
    floor = float(model.floor_sigma()[1, 0])
    assert torch.allclose(sig[0, 1], torch.full((4,), floor, dtype=torch.float64))
    assert bool((sig[1, 1] > floor).all())


def test_held_part_is_constant_within_a_scenario_and_differs_between():
    """Without the floor, the error of a reading is its held gain and phase alone."""
    spec = InstrumentErrorConfig(
        voltage_harmonic=InstrumentBand(
            reading_pct=5.0, phase_deg=0.5, floor_pct=0.0, threshold_pct=1.0
        ),
        voltage_fundamental=InstrumentBand(
            reading_pct=0.1, phase_deg=0.02, floor_pct=0.0, threshold_pct=10.0
        ),
    )
    v = _readings(6, [U_NOM, 9.0, 4.0], steps=5)
    # the readings vary over the time steps; the held factor must not
    v = v * torch.linspace(0.8, 1.2, 5, dtype=torch.float64).reshape(1, 5, 1, 1)
    out = _voltage_model(spec, seed=2).apply(v)
    factor = out / v  # [S, T, H, R]
    assert torch.allclose(factor, factor[:, :1].expand_as(factor), rtol=0, atol=1e-12)
    assert bool((factor[0] != factor[1]).all())
    assert bool((factor[:, 0] != 1).all())


def test_floor_is_random_at_every_level():
    """Above the threshold the reading change still carries the floor of both readings."""
    v = _readings(4000, [U_NOM, 9.0, 1.0], steps=2)
    out = _voltage_model(seed=8).apply(v)
    err = out - v
    change = err[:, 1] - err[:, 0]  # the readings are equal: the held part cancels
    floor = _voltage_model().floor_sigma()
    assert torch.allclose(
        _component_sigma(change, 0), math.sqrt(2.0) * floor, rtol=0.05
    )


@pytest.mark.parametrize("stable", [1.0, 0.5, 0.0])
def test_change_variance_matches_the_helper(stable):
    spec = InstrumentErrorConfig(stable_fraction=stable)
    model = _voltage_model(spec, seed=5)
    v = _readings(20000, [U_NOM, 9.0, 1.0], steps=2)
    # after: a larger fundamental, a 3rd harmonic that grows, a 5th that crosses the
    # threshold upward (floor before, proportional after)
    v[:, 1] = v[:, 1] * torch.tensor([1.02, 1.6, 4.0], dtype=torch.float64).reshape(
        3, 1
    )
    err = model.apply(v) - v
    emp = _component_sigma(err[:, 1] - err[:, 0], 0)
    cfg = model.change_sigma(v[0, 0], v[0, 1])
    assert torch.allclose(emp, cfg, rtol=0.03)
    if stable == 1.0:
        # the held gain scales with the CHANGE: far below two independent readings
        indep = (
            model.reading_sigma(v[0, 0].abs()) ** 2
            + model.reading_sigma(v[0, 1].abs()) ** 2
        ).sqrt()
        assert bool((cfg[1] < 0.8 * indep[1]).all())
    if stable == 0.0:
        indep = (
            model.reading_sigma(v[0, 0].abs()) ** 2
            + model.reading_sigma(v[0, 1].abs()) ** 2
        ).sqrt()
        assert torch.allclose(cfg, indep, rtol=1e-12)


def test_determinism_under_a_seed():
    v = _readings(3, [U_NOM, 9.0, 1.0], steps=4)
    a = _voltage_model(seed=1).apply(v)
    b = _voltage_model(seed=1).apply(v)
    c = _voltage_model(seed=2).apply(v)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    model = _voltage_model(seed=1)
    first = model.apply(v)
    model.reseed(1)
    assert torch.equal(model.apply(v), first)
    # the CT stream is independent of the voltage stream
    edges = model.for_edges(torch.arange(4), current_range_a=I_NOM)
    assert not torch.equal(edges.apply(v) - v, first - v)


def test_input_dtype_and_shape_are_kept():
    v = _readings(2, [U_NOM, 9.0, 1.0])[:, 0].to(torch.complex64)
    out = _voltage_model().apply(v)
    assert out.dtype == torch.complex64 and out.shape == v.shape
    assert _voltage_model().apply(v[0]).shape == v[0].shape  # a single [H, R] reading


def test_refusals():
    with pytest.raises(ValueError, match="outside"):
        InstrumentNoise(
            InstrumentErrorConfig(),
            quantity="voltage",
            nominal=torch.ones(2),
            orders=[1, 51],
        )
    with pytest.raises(ValueError, match="current_range_a"):
        _voltage_model().for_edges(torch.arange(3))
    with pytest.raises(ValueError, match="orders"):
        _voltage_model().apply(torch.ones(2, 2, 4, dtype=torch.complex128))
    with pytest.raises(ValueError, match="grid"):
        InstrumentNoise.from_config(
            NoiseConfig(enabled=True, instrument=InstrumentErrorConfig()),
            n_nodes=4,
            grid=None,
            orders=ORDERS,
        )


def test_default_noise_config_keeps_the_accuracy_classes():
    """``instrument`` defaults off; a config without it reads as before."""
    cfg = NoiseConfig()
    assert cfg.instrument is None
    assert NoiseConfig.model_validate({"enabled": True, "seed": 3}).instrument is None
    on = NoiseConfig(enabled=True, instrument={})
    assert on.instrument == InstrumentErrorConfig()


def test_multigrid_rejects_the_instrument_model():
    def experiment(noise: NoiseConfig) -> ExperimentConfig:
        return ExperimentConfig(
            name="mg-instrument",
            orders=ORDERS,
            data=DataConfig(
                source="parquet",
                dataset_path="unused",
                multigrid=MultiGridConfig(task="random"),
            ),
            model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
            measurement=MeasurementConfig(slack_known=True),
            noise=noise,
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="full", kind="full", target_loss=1e-9, max_epochs=1
                    )
                ]
            ),
            optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
            train=TrainConfig(precision="32", accelerator="cpu", devices=1),
        )

    experiment(NoiseConfig(enabled=True))
    with pytest.raises(ValueError, match="noise.instrument"):
        experiment(NoiseConfig(enabled=True, instrument=InstrumentErrorConfig()))


def test_anchor_calibration_reads_the_instrument_sigma():
    """The noise-calibrated anchor weights use the instrument's per-reading sigma."""
    from injection_hse.config import PhysicsConfig
    from injection_hse.train.anchoring import anchor_calibration_sigmas

    model = _voltage_model()
    edges = model.for_edges(torch.arange(5), current_range_a=I_NOM)
    state_scale = torch.tensor([[U_NOM] * 4, [9.0, 1.0, 9.0, 1.0], [1.0] * 4]).double()
    edge_scale = torch.tensor([[40.0] * 5, [4.0] * 5, [1.0] * 5]).double()
    _, sig_v, sig_i = anchor_calibration_sigmas(
        PhysicsConfig(),
        state_scale=state_scale,
        injection_scale=torch.ones(3),
        edge_scale=edge_scale,
        voltage_noise=model,
        current_noise=edges,
    )
    assert torch.equal(sig_v, model.reading_sigma(state_scale))
    assert torch.equal(sig_i, edges.reading_sigma(edge_scale))
    # a per-unit feature-frame scale is converted to volts before the regime is decided
    _, sig_pu, _ = anchor_calibration_sigmas(
        PhysicsConfig(),
        state_scale=state_scale / U_NOM,
        injection_scale=torch.ones(3),
        voltage_noise=model,
        injection_base=torch.ones(3, 4),
        voltage_base=torch.full((4,), U_NOM, dtype=torch.float64),
    )
    assert torch.allclose(sig_pu, sig_v, rtol=1e-12)


def _floorless(stable: float = 1.0) -> InstrumentErrorConfig:
    """The preset without floors: a reading's error is its proportional part alone."""
    spec = InstrumentErrorConfig(stable_fraction=stable)
    update = {
        name: getattr(spec, name).model_copy(update={"floor_pct": 0.0})
        for name in (
            "voltage_fundamental",
            "voltage_harmonic",
            "current_fundamental",
            "current_harmonic",
        )
    }
    return spec.model_copy(update=update)


def test_a_site_key_shares_the_held_part_across_calls():
    v = _readings(3, [U_NOM, 9.0, 4.0], steps=2)
    model = _voltage_model(_floorless(), seed=6)
    model.site = 17
    first = model.apply(v) / v
    second = model.apply(v[:1, :1] * 1.1) / (v[:1, :1] * 1.1)  # a later, separate call
    # one site: every leading index and both calls carry the same held factor
    assert torch.allclose(first, first[:1, :1].expand_as(first), rtol=0, atol=1e-12)
    assert torch.allclose(second, first[:1, :1], rtol=0, atol=1e-12)
    # deterministic from the seed and the key; another key or seed draws another one
    fresh = _voltage_model(_floorless(), seed=6)
    fresh.site = 17
    assert torch.equal(fresh.apply(v), model.apply(v))
    other = _voltage_model(_floorless(), seed=6)
    other.site = 18
    assert bool((other.apply(v) / v != first).all())
    # the current sibling inherits the key but keeps its own held part
    edges = model.for_edges(torch.arange(4), current_range_a=I_NOM)
    assert edges.site == 17
    i = _readings(2, [40.0, 4.0, 2.0], steps=1)
    f_i = edges.apply(i) / i
    assert torch.allclose(f_i, f_i[:1].expand_as(f_i), rtol=0, atol=1e-12)
    assert bool((f_i[0, 0] != first[0, 0]).all())


def test_per_scenario_site_keys():
    v = _readings(4, [U_NOM, 9.0, 4.0], steps=3)
    model = _voltage_model(_floorless(), seed=2)
    model.site = torch.tensor([5, 9, 5, 2])
    factor = model.apply(v) / v
    assert torch.allclose(factor, factor[:, :1].expand_as(factor), rtol=0, atol=1e-12)
    assert torch.allclose(factor[0], factor[2], rtol=0, atol=1e-12)
    assert bool((factor[0] != factor[1]).all()) and bool((factor[1] != factor[3]).all())
    single = _voltage_model(_floorless(), seed=2)
    single.site = 9
    assert torch.allclose(single.apply(v[1:2]) / v[1:2], factor[1:2], atol=1e-12)
    with pytest.raises(ValueError, match="site keys"):
        model.apply(v[:3])
    with pytest.raises(ValueError, match="site"):
        model.site = torch.tensor([0.5])


def test_site_keyed_draws_keep_the_change_variance():
    """With a key, the held part is shared over the call and the change sigma holds."""
    n = 20000
    v = _readings(1, [U_NOM, 9.0, 1.0], steps=2).expand(n, -1, -1, -1).clone()
    v[:, 1] = v[:, 1] * torch.tensor([1.02, 1.6, 4.0], dtype=torch.float64).reshape(
        3, 1
    )
    model = _voltage_model(seed=5)
    model.site = torch.arange(n)
    err = model.apply(v) - v
    emp = _component_sigma(err[:, 1] - err[:, 0], 0)
    assert torch.allclose(emp, model.change_sigma(v[0, 0], v[0, 1]), rtol=0.03)
    emp_reading = _component_sigma(err[:, 0], 0)
    assert torch.allclose(emp_reading, model.reading_sigma(v[0, 0].abs()), rtol=0.03)


def test_the_class_a_preset():
    assert InstrumentErrorConfig.class_a() == InstrumentErrorConfig()
    spec = InstrumentErrorConfig.class_a(stable_fraction=0.5)
    assert spec.stable_fraction == 0.5 and spec.coverage == 2.0
    # a preset round-trips through a run config's dump
    assert InstrumentErrorConfig.model_validate(spec.model_dump()) == spec

