"""Sensor noise on the multi-grid path + the anchor-randomization draw.

Both exist because a measurement model that is silently perfect makes an anchored solve look
better than any deployment can be. The noise plumbing had been built for the multi-grid path
but never reached its forward, so ``noise.enabled`` was accepted and ignored; these tests pin
the wiring and its boundaries — the model reads perturbed inputs, the loss target and the
anchor targets do not move.
"""

from __future__ import annotations

import math

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MultiGridConfig,
    NoiseConfig,
    TrainConfig,
)
from injection_hse.data import build_data_source
from injection_hse.data.multigrid import (
    MultiGridGenerationConfig,
    SyntheticFeederSource,
    generate_multigrid_dataset,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.masking import build_masker
from injection_hse.train import (
    ANCHOR_LOG_DECADES,
    SEModule,
    _anchor_weight_from,
    fit_multigrid_normalizer,
    make_datamodule,
)

ORDERS = [1, 3]


# =============================================================================
# The per-batch anchor draw
# =============================================================================
def test_uniform_draw_interpolates_the_weight():
    draw = _anchor_weight_from("uniform")
    assert draw((0.0, 20.0), 0.0) == pytest.approx(0.0)
    assert draw((0.0, 20.0), 0.5) == pytest.approx(10.0)
    assert draw((0.0, 20.0), 1.0) == pytest.approx(20.0)


def test_log_uniform_spans_decades_below_the_maximum():
    """u=0 lands ANCHOR_LOG_DECADES below the top, u=1 on it, the midpoint is geometric."""
    draw = _anchor_weight_from("log_uniform")
    high = 20.0
    floor = high * 10.0**-ANCHOR_LOG_DECADES
    assert draw((0.0, high), 0.0) == pytest.approx(floor)
    assert draw((0.0, high), 1.0) == pytest.approx(high)
    assert draw((0.0, high), 0.5) == pytest.approx(math.sqrt(floor * high))


def test_log_uniform_makes_weak_anchor_batches_common():
    """The point of the draw: the regime that can teach the injection head is reachable.

    Under a uniform draw over 0~20 only 5 % of batches fall below weight 1 — in the rest the
    measurements pin the state and the predicted injections carry no gradient.
    """
    grid = [i / 10_000 for i in range(10_001)]

    def weak_share(kind):
        draw = _anchor_weight_from(kind)
        return sum(draw((0.0, 20.0), u) < 1.0 for u in grid) / len(grid)

    assert weak_share("uniform") == pytest.approx(0.05, abs=0.01)
    assert weak_share("log_uniform") > 0.5


def test_a_range_with_a_positive_floor_keeps_it():
    assert _anchor_weight_from("log_uniform")((0.5, 50.0), 0.0) == pytest.approx(0.5)


def test_an_unknown_draw_is_rejected():
    with pytest.raises(ValueError, match="anchor_weight_draw"):
        _anchor_weight_from("gaussian")


# =============================================================================
# Noise reaches the model input and nothing else
# =============================================================================
@pytest.fixture(scope="module")
def root(tmp_path_factory):
    out = tmp_path_factory.mktemp("multigrid_noise")
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(out),
            n_scenarios=4,
            orders=ORDERS,
            val_fraction=0.34,
            sources=[SyntheticFeederSource(n_grids=4, node_range=(4, 6), n_feeders=2)],
        )
    )
    return out


def _module(root, *, noise: bool) -> SEModule:
    cfg = ExperimentConfig(
        name="multigrid-noise",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        noise=NoiseConfig(enabled=noise, default_class="1"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="frac_50",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        train=TrainConfig(accelerator="cpu", devices=1, enable_progress_bar=False),
    )
    source = build_data_source(cfg.data)
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, TrigEncoding(), cfg),
        grid_keys=[bundle.key for bundle in source.grids],
    )
    module.set_masker(build_masker("random", observed_fraction=0.5))
    module.eval()
    return module


@pytest.fixture(scope="module")
def batch(root):
    source = build_data_source(
        DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        )
    )
    dm = make_datamodule(source, batch_size=2, num_workers=0)
    dm.setup("fit")
    return next(iter(dm.val_dataloader()))


def test_noise_perturbs_the_model_input_but_not_the_target(root, batch, monkeypatch):
    """The forward reads perturbed features while the loss target stays the clean state."""
    module = _module(root, noise=True)
    assert module._noise is not None, "noise must be built on the multi-grid path"
    seen = []
    original = module._encode_normalize
    monkeypatch.setattr(
        module,
        "_encode_normalize",
        lambda v: (seen.append(v.detach().clone()), original(v))[1],
    )
    module._multigrid_step(batch, train=False)
    # first call = the clean target, second = the noisy input: same shape, different values
    assert len(seen) >= 2
    assert seen[0].shape == seen[1].shape
    assert not torch.allclose(seen[0], seen[1])


def _estimate(module, batch):
    """One forward under a FIXED mask draw — the masker owns its own RNG stream, so two
    calls otherwise observe different rows and nothing downstream is comparable."""
    state = module._mask_gen.get_state()
    try:
        return module._multigrid_step(batch, train=False)[2]
    finally:
        module._mask_gen.set_state(state)


def test_noise_off_leaves_the_forward_untouched(root, batch):
    """A noise-disabled module reproduces the pre-noise forward exactly."""
    off = _module(root, noise=False)
    assert off._noise is None
    torch.testing.assert_close(_estimate(off, batch), _estimate(off, batch))


def _injection_module(root, *, noise: bool) -> SEModule:
    """An anchored injection-decode module — the configuration the anchor question is about."""
    from injection_hse.config import MeasurementConfig, PhysicsConfig
    from injection_hse.physics import (
        MultiGridPhysicsBank,
        fit_multigrid_edge_normalizer,
        fit_multigrid_injection_normalizer,
    )

    cfg = ExperimentConfig(
        name="multigrid-noise-anchor",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(decode="injection"),
        measurement=MeasurementConfig(measure_current=True, slack_known=True),
        noise=NoiseConfig(enabled=noise, default_class="1"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="placement",
                    kind="placement",
                    target_loss=1e-9,
                    max_epochs=1,
                    anchor_voltage_weight=5.0,
                    anchor_current_weight=5.0,
                )
            ]
        ),
        train=TrainConfig(accelerator="cpu", devices=1, enable_progress_bar=False),
    )
    source = build_data_source(cfg.data)
    encoding = TrigEncoding()
    bank = MultiGridPhysicsBank(
        source,
        list(ORDERS),
        dtype=torch.complex128,
        anchor_v_weight=5.0,
        anchor_i_weight=5.0,
    )
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, encoding, cfg),
        grid_keys=[b.key for b in source.grids],
        bank=bank,
        inj_normalizer=fit_multigrid_injection_normalizer(bank, source, encoding, cfg),
        edge_normalizer=fit_multigrid_edge_normalizer(bank, source, encoding, cfg),
    )
    module.set_masker(
        build_masker("placement", observed_rows=torch.zeros(0, dtype=torch.int64))
    )
    module.set_stage_anchor(v_weight=5.0, i_weight=5.0)
    module.eval()
    return module


def test_the_anchored_solve_is_constrained_by_the_NOISY_measurement(
    root, batch, monkeypatch
):
    """The solve must see the meter's reading, not the truth — and the SAME reading the
    model saw. A clean anchor is a solve reading ground truth, which is exactly the
    artificially perfect floor the noise model exists to remove."""
    module = _injection_module(root, noise=True)
    seen = {}
    original = module._multigrid_decode

    def spy(i_pred, v_meas, *args, **kwargs):
        seen["v_meas"] = v_meas.detach().clone()
        return original(i_pred, v_meas, *args, **kwargs)

    monkeypatch.setattr(module, "_multigrid_decode", spy)
    module._multigrid_step(batch, train=False)
    assert "v_meas" in seen, "the injection decode did not run"
    truth = batch["v"].to(seen["v_meas"].dtype)
    assert not torch.allclose(seen["v_meas"], truth)
    # a MEASUREMENT, not a different state: the perturbation stays within the noise class
    assert (
        float((seen["v_meas"].abs() - truth.abs()).abs().max() / truth.abs().max())
        < 0.1
    )


def test_a_clean_run_anchors_on_the_exact_state(root, batch, monkeypatch):
    """The complement: with noise off the anchor target IS the reference state."""
    module = _injection_module(root, noise=False)
    seen = {}
    original = module._multigrid_decode
    monkeypatch.setattr(
        module,
        "_multigrid_decode",
        lambda i, v, *a, **k: (
            seen.setdefault("v", v.detach().clone()),
            original(i, v, *a, **k),
        )[1],
    )
    module._multigrid_step(batch, train=False)
    torch.testing.assert_close(seen["v"], batch["v"].to(seen["v"].dtype))


def test_noise_changes_the_estimate(root, batch):
    """Whole-forward evidence, not just the feature tensors: the estimate must move."""
    clean = _module(root, noise=False)
    noisy = _module(root, noise=True)
    noisy.load_state_dict(clean.state_dict())  # same weights, only the inputs differ
    assert not torch.allclose(_estimate(clean, batch), _estimate(noisy, batch))
