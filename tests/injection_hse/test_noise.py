"""Opt-in sensor noise: disabled = byte-identical, deterministic given seed, off the targets."""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    AccuracyClassNoise,
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
    NoiseConfig,
    OptimConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import SensorNoise

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def test_sensor_noise_deterministic_and_nontrivial():
    n = 6
    mag = torch.full((n,), 0.01)  # 1 % magnitude sigma
    phase = torch.full((n,), 0.01)  # ~0.57 deg
    v = torch.randn(4, 3, n, dtype=torch.complex64) + 1.0

    a = SensorNoise(mag, phase, seed=0).apply(v)
    b = SensorNoise(mag, phase, seed=0).apply(v)
    c = SensorNoise(mag, phase, seed=1).apply(v)
    assert torch.equal(a, b)  # deterministic given the seed
    assert not torch.equal(a, c)  # a different seed perturbs differently
    assert not torch.equal(a, v)  # the phasors actually moved
    # zero sigma is the identity (no perturbation)
    zero = SensorNoise(torch.zeros(n), torch.zeros(n), seed=0).apply(v)
    assert torch.allclose(zero, v)


def test_edge_noise_inherits_class_and_is_independent():
    """CT noise inherits each half-edge's metering-node class but draws from its own stream."""
    n = 6
    mag = torch.tensor([0.0, 0.05, 0.0, 0.0, 0.02, 0.0])
    phase = torch.tensor([0.0, 0.01, 0.0, 0.0, 0.005, 0.0])
    node = SensorNoise(mag, phase, seed=7)
    edge_src_rows = torch.tensor([1, 1, 4, 0, 4])  # half-edge -> source (metering) row
    edge = node.for_edges(edge_src_rows)

    # per-edge sigmas are the node sigmas gathered at the half-edge source rows
    assert torch.equal(edge._mag, mag.to(torch.float64)[edge_src_rows])
    assert torch.equal(edge._phase, phase.to(torch.float64)[edge_src_rows])

    i = torch.ones(3, 2, edge_src_rows.numel(), dtype=torch.complex64)
    a = node.for_edges(edge_src_rows).apply(i)
    b = node.for_edges(edge_src_rows).apply(i)
    assert torch.equal(a, b)  # deterministic per fresh instance (reproducible run)
    # the CT draw is independent of the voltage draw over the same locations
    v_draw = SensorNoise(mag, phase, seed=7).apply(
        torch.ones(3, 2, n, dtype=torch.complex64)
    )[..., edge_src_rows]
    assert not torch.equal(v_draw, a)
    # zero-sigma edges (their metering row is un-metered) are untouched
    assert torch.allclose(a[..., 3], i[..., 3])


def test_sensor_noise_from_config_default_class():
    cfg = NoiseConfig(
        enabled=True,
        default_class="0.5",
        classes={"0.5": AccuracyClassNoise(magnitude_pct=0.5, phase_deg=0.2)},
    )
    noise = SensorNoise.from_config(cfg, n_nodes=5, grid=None)
    assert torch.allclose(noise._mag, torch.full((5,), 0.005, dtype=torch.float64))
    # no grid -> no voltage base -> the range-referenced floor stays off
    assert torch.all(noise._floor == 0.0)


def test_range_floor_dominates_small_bins():
    """The additive floor perturbs a tiny harmonic bin where reading noise cannot."""
    n = 4
    mag = torch.full((n,), 0.001)  # 0.1 % of reading
    phase = torch.zeros(n)
    floor = torch.full((n,), 0.1)  # absolute per-component std (e.g. 0.05 % of 230 V)
    tiny = torch.full((8, 2, n), 1e-4, dtype=torch.complex128)  # a millivolt-scale bin
    clean = SensorNoise(mag, phase, seed=3).apply(tiny)
    floored = SensorNoise(mag, phase, floor, seed=3).apply(tiny)
    # reading noise alone moves the bin by ~1e-7; the floor moves it by ~0.1
    assert (clean - tiny).abs().max() < 1e-5
    assert (floored - tiny).abs().median() > 1e-2
    # zero floor is bit-compatible with the floor-free constructor (same streams)
    again = SensorNoise(mag, phase, torch.zeros(n), seed=3).apply(tiny)
    assert torch.equal(clean, again)


def test_uniform_model_is_layout_independent():
    """The multi-grid form: size-1 sigmas broadcast against any row count."""
    cfg = NoiseConfig(enabled=True, default_class="0.1", seed=5)
    noise = SensorNoise.uniform(cfg, nominal=1.0)
    for n in (7, 30):
        v = torch.full((2, 3, n), 0.001 + 0.0j, dtype=torch.complex128)  # tiny pu bins
        noisy = noise.apply(v)
        assert noisy.shape == v.shape
        # the 0.05 %-of-nominal floor dominates a 0.1 % reading error on a 1e-3 pu bin
        assert (noisy - v).abs().median() > 1e-4
    edges = SensorNoise.uniform(cfg, nominal=None, seed_offset=1)
    i = torch.randn(2, 3, 11, dtype=torch.complex128)
    assert (edges._floor == 0).all() and not torch.equal(edges.apply(i), i)


def test_multigrid_config_accepts_noise():
    from injection_hse.config import MultiGridConfig

    cfg = ExperimentConfig(
        name="mg-noise",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path="unused",
            multigrid=MultiGridConfig(task="random"),
            batch_size=4,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
        measurement=MeasurementConfig(slack_known=True),
        noise=NoiseConfig(enabled=True, default_class="0.1"),
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
    assert cfg.noise.enabled and cfg.data.multigrid is not None


def test_edge_floor_reranges_by_ct_range():
    n = 3
    noise = SensorNoise(
        torch.zeros(n),
        torch.zeros(n),
        torch.full((n,), 230.0 * 5e-4),
        torch.full((n,), 5e-4),
        seed=0,
    )
    edges = noise.for_edges(torch.tensor([0, 1, 2]), current_range_a=40.0)
    assert torch.allclose(edges._floor, torch.full((3,), 40.0 * 5e-4).double())
    off = noise.for_edges(torch.tensor([0, 1, 2]))
    assert torch.all(off._floor == 0.0)


def _cfg(noise: NoiseConfig, dataset_path, tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="noise-test",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset_path),
            batch_size=16,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        noise=noise,
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random_50",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )


def _build_module(cfg, grid):
    """Construct an SEModule the way ``train_experiment`` does (no fit)."""
    import lightning as L

    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule

    L.seed_everything(int(cfg.seed), workers=True)
    source = build_data_source(cfg.data, grid=grid)
    encoding = build_encoding(cfg.encoding.kind)
    train_ds = source.train_dataset()
    train_v = torch.stack([train_ds[i]["v"] for i in range(min(len(train_ds), 64))])
    feats = encoding.encode(train_v).transpose(-3, -2)
    normalizer = HarmonicNormalizer().fit(feats)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=normalizer,
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        grid=grid,
    )
    module.eval()
    return module, source


def _first_batch(source):
    from torch.utils.data import DataLoader

    from injection_hse.data import StateCollate

    loader = DataLoader(
        source.val_dataset(), batch_size=8, shuffle=False, collate_fn=StateCollate()
    )
    return next(iter(loader))


def test_noise_off_is_byte_identical(cigre_grid, se_dataset, tmp_path):
    """A module with noise disabled infers identically to one built before noise existed."""
    off = _cfg(NoiseConfig(enabled=False), se_dataset, tmp_path)
    module, source = _build_module(off, cigre_grid)
    assert module._noise is None  # nothing constructed when disabled
    batch = _first_batch(source)
    out1 = module.infer(batch)
    out2 = module.infer(batch)
    assert torch.equal(out1["pred_v"], out2["pred_v"])  # no stochastic input path


def test_noise_perturbs_input_not_target(cigre_grid, se_dataset, tmp_path):
    """Enabling noise changes the estimate but leaves the target (metric reference) clean."""
    on = _cfg(NoiseConfig(enabled=True, seed=0), se_dataset, tmp_path)
    module, source = _build_module(on, cigre_grid)
    assert module._noise is not None
    batch = _first_batch(source)

    off = _cfg(NoiseConfig(enabled=False), se_dataset, tmp_path)
    module_off, source_off = _build_module(off, cigre_grid)
    batch_off = _first_batch(source_off)

    out_on = module.infer(batch)
    out_off = module_off.infer(batch_off)
    # the ground-truth target is the clean state in BOTH runs (noise never touches it)
    assert torch.equal(out_on["target_v"], out_off["target_v"])
    # ...but the noisy input pushes the estimate away from the noise-free one
    assert not torch.equal(out_on["pred_v"], out_off["pred_v"])


# =============================================================================
# The harmonic subchannel's range (the floor's reference scale)
# =============================================================================
def test_the_floor_is_smaller_at_harmonic_orders():
    """A floor referenced to the fundamental for EVERY order buries the harmonics.

    One absolute floor is 0.05 % of a 230 V fundamental but ~17 % of a 0.7 V nineteenth, so
    the orders an estimator exists to recover would arrive dominated by an instrument error a
    real analyser's harmonic subchannel does not have.
    """
    import torch

    from injection_hse.config import NoiseConfig
    from injection_hse.data import SensorNoise

    orders = [1, 3, 19]
    cfg = NoiseConfig(enabled=True, default_class="0.1", harmonic_range=0.1)
    noise = SensorNoise.uniform(cfg, nominal=1.0, orders=orders)
    # a zero state isolates the additive floor from the multiplicative reading error
    v = torch.zeros((4096, len(orders), 1), dtype=torch.complex128)
    spread = noise.apply(v).abs().std(dim=0).squeeze(-1)
    assert float(spread[1] / spread[0]) == pytest.approx(0.1, rel=0.15)
    assert float(spread[2] / spread[0]) == pytest.approx(0.1, rel=0.15)


def test_harmonic_range_one_restores_the_flat_floor():
    import torch

    from injection_hse.config import NoiseConfig
    from injection_hse.data import SensorNoise

    orders = [1, 3, 19]
    cfg = NoiseConfig(enabled=True, default_class="0.1", harmonic_range=1.0)
    noise = SensorNoise.uniform(cfg, nominal=1.0, orders=orders)
    v = torch.zeros((4096, len(orders), 1), dtype=torch.complex128)
    spread = noise.apply(v).abs().std(dim=0).squeeze(-1)
    assert float(spread[1] / spread[0]) == pytest.approx(1.0, rel=0.15)


# =============================================================================
# Draw device: CPU generator (default) or the phasors' own device
# =============================================================================
def test_input_device_draws_on_cpu_match_the_default():
    """On CPU phasors the input-device option falls back to the CPU stream (byte-identical)."""
    n = 5
    mag = torch.full((n,), 0.01)
    phase = torch.full((n,), 0.005)
    floor = torch.full((n,), 0.02)
    v = torch.randn(3, 2, n, dtype=torch.complex64) + 1.0
    a = SensorNoise(mag, phase, floor, seed=3).apply(v)
    b = SensorNoise(mag, phase, floor, seed=3).draw_on_input_device().apply(v)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_input_device_draws_stay_on_the_gpu_and_keep_the_distribution():
    n = 64
    mag = torch.full((n,), 0.01)
    phase = torch.full((n,), 0.0)
    v = torch.ones(2000, 3, n, dtype=torch.complex64, device="cuda")
    noise = SensorNoise(mag, phase, seed=0).draw_on_input_device()
    out = noise.apply(v)
    assert out.device.type == "cuda"
    rel = (out.abs() - 1.0).double()
    assert abs(float(rel.mean())) < 1e-3
    assert abs(float(rel.std()) - 0.01) < 1e-3
    # deterministic per seed on the device
    again = SensorNoise(mag, phase, seed=0).draw_on_input_device().apply(v)
    assert torch.equal(out, again)


def test_config_draw_device_reaches_both_noise_models(cigre_grid, se_dataset, tmp_path):
    cfg = _cfg(NoiseConfig(enabled=True, seed=0, draw_device="input"), se_dataset, tmp_path)
    module, _ = _build_module(cfg, cigre_grid)
    assert module._noise._on_input_device
    default = _cfg(NoiseConfig(enabled=True, seed=0), se_dataset, tmp_path)
    module_default, _ = _build_module(default, cigre_grid)
    assert not module_default._noise._on_input_device
