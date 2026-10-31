"""The single-grid per-unit feature frame (``NormalizationConfig.per_unit``).

A feeder's node-phase rows do not share one voltage class, so a per-order statistic fit in
ABSOLUTE volts is set by that row population rather than by the variation between scenarios,
and the fundamental's per-scenario signal ends up a few percent of the feature channel's
scale. ``per_unit`` moves the FRAME the model sees — the state over the per-row base voltage,
the injections and branch currents over the per-unit current base ``s(h)·V_ref²/V_base`` — while the
physics keeps solving, anchoring and reporting in absolute units. The invariants under test:

- the default (``False``) is byte-identical: same parameter count, identical loss;
- the state round-trips through the frame, and a voltage decode returns VOLTS;
- the injection decode's head predicts per-unit currents and the solve consumes AMPERES, so
  the true injections decode back to the true state; a checkpoint reload reproduces the loss;
- the noise-calibrated anchor weights of a single-voltage-class grid are the absolute-frame
  ones — the two bases enter the weight only through their ratio, which is the grid's operator
  scale there, exactly as it is per member in the multi-grid bank;
- the physics-derived input priors and the measured-current channels train in frame;
- the multi-grid capability contract rejects the field (that path is always per-unit).
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
    MultiGridConfig,
    NoiseConfig,
    NormalizationConfig,
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.train.layout import _flatten_features, _to_nhc, _unflatten_features

ORDERS = [1, 3, 5]


class _ConstantModel(torch.nn.Module):
    """A model emitting a FIXED feature block: probes the decode, not the network."""

    def __init__(self, out: torch.Tensor) -> None:
        super().__init__()
        self.out = out
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=out.dtype))

    def forward(self, model_input):  # noqa: D401 - the StateEstimator call contract
        return self.out + self.bias


@pytest.fixture(scope="module")
def setup():
    """A small feeder, solved states and the operators every module below shares."""
    from pgml.grids import synthetic_feeder
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import base_voltage_per_row, build_graph
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.physics import NetworkSolver

    grid = synthetic_feeder(6, n_feeders=2)
    v = solve_harmonic_flow(grid, ORDERS, dtype=torch.complex128).v  # [H, N]
    states = torch.stack([v * (1 + 0.01 * k) for k in range(8)])  # [B, H, N]
    base_v = base_voltage_per_row(grid)
    edge_op = BranchCurrentOperator(grid, ORDERS, dtype=torch.complex128)
    solver = NetworkSolver.from_grid(
        grid,
        ORDERS,
        dtype=torch.complex128,
        anchor_v_weight=1.0,
        anchor_i_weight=1.0,
        edge_op=edge_op,
    )
    return {
        "grid": grid,
        "graph": build_graph(grid),
        "states": states,
        "base_v": base_v,
        "solver": solver,
        "edge_op": edge_op,
        "i_base": solver.i_base(base_v),
    }


def _cfg(
    *,
    per_unit: bool,
    decode: str = "voltage",
    priors=(),
    measure_current: bool = False,
    anchor: float = 0.0,
    weighting: str = "manual",
    noise: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="per-unit",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        normalization=NormalizationConfig(per_unit=per_unit),
        physics=PhysicsConfig(
            decode=decode,
            input_priors=list(priors),
            anchor_voltage_weight=anchor,
            anchor_weighting=weighting,
        ),
        measurement=MeasurementConfig(
            slack_known=True, measure_current=measure_current
        ),
        noise=NoiseConfig(enabled=noise, seed=3),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,
                    max_epochs=1,
                )
            ]
        ),
        train=TrainConfig(dtype="complex128"),
    )


def _normalizers(setup, cfg):
    """The state / injection / edge normalizers, each fit in ``cfg``'s frame."""
    from injection_hse.data import per_unit_state
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import fit_injection_normalizer

    enc = TrigEncoding()
    states, base_v = setup["states"], setup["base_v"]
    per_unit = cfg.normalization.per_unit
    i_base = setup["i_base"] if per_unit else None
    v_feat = per_unit_state(states, base_v) if per_unit else states
    normalizer = HarmonicNormalizer().fit(enc.encode(v_feat).transpose(-3, -2))
    inj_normalizer = fit_injection_normalizer(setup["solver"], states, enc, base=i_base)
    edge_normalizer = None
    if cfg.measurement.measure_current:
        i_train = setup["edge_op"](states)
        if i_base is not None:
            i_train = i_train / i_base.index_select(-1, setup["edge_op"].edge_src_rows)
        edge_normalizer = HarmonicNormalizer().fit(
            enc.encode(i_train).transpose(-3, -2)
        )
    return normalizer, inj_normalizer, edge_normalizer


def _solver(setup, cfg):
    """A FRESH solver per module — the anchoring state and the calibration live on it."""
    from injection_hse.physics import NetworkSolver

    currents = cfg.measurement.measure_current
    return NetworkSolver.from_grid(
        setup["grid"],
        ORDERS,
        dtype=torch.complex128,
        anchor_v_weight=1.0,
        anchor_i_weight=1.0 if currents else 0.0,
        edge_op=setup["edge_op"] if currents else None,
    )


def _module(setup, cfg, *, masker="full"):
    from injection_hse.masking import build_masker
    from injection_hse.train import SEModule

    normalizer, inj_normalizer, edge_normalizer = _normalizers(setup, cfg)
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=int(setup["states"].shape[-1]),
        normalizer=normalizer,
        graph=setup["graph"],
        base_v=setup["base_v"],
        solver=_solver(setup, cfg) if cfg.needs_solver else None,
        inj_normalizer=inj_normalizer if cfg.needs_solver else None,
        edge_op=setup["edge_op"] if cfg.measurement.measure_current else None,
        edge_normalizer=edge_normalizer,
        grid=setup["grid"],
    )
    module.set_masker(
        build_masker("full")
        if masker == "full"
        else build_masker(masker, observed_fraction=0.5)
    )
    return module


# =============================================================================
# Default OFF = byte-identical
# =============================================================================
def test_the_default_is_byte_identical(setup):
    """A config that never names the field equals one naming its default, and both modules
    hold the same parameters and produce the same loss on the same batch."""
    legacy = _cfg(per_unit=False)
    dump = legacy.model_dump()
    dump["normalization"].pop("per_unit")
    pre_feature = ExperimentConfig.model_validate(dump)
    assert pre_feature == legacy

    a, b = _module(setup, pre_feature), _module(setup, legacy)
    assert a._per_unit is False
    assert sum(p.numel() for p in a.model.parameters()) == sum(
        p.numel() for p in b.model.parameters()
    )
    batch = {"v": setup["states"]}
    loss_a = a._step(batch, train=True)[0]
    loss_b = b._step(batch, train=True)[0]
    assert torch.equal(loss_a.detach(), loss_b.detach())
    # the seams are the identity without the frame
    v = setup["states"]
    assert torch.equal(a._to_feature_frame(v), v)
    assert torch.equal(a._from_feature_frame(v), v)
    assert a._injection_base() is None


def test_the_frame_is_a_behaviour_field():
    """The normalizers were fit in it — a later pass may not swap the frame silently."""
    from injection_hse.train import CFG_BEHAVIOUR_FIELDS

    assert "normalization.per_unit" in CFG_BEHAVIOUR_FIELDS


# =============================================================================
# The voltage frame: round trip, and the estimate comes back in volts
# =============================================================================
def test_the_state_round_trips_through_the_feature_frame(setup):
    """input -> features -> decode reproduces the state, and the features sit near 1 pu."""
    module = _module(setup, _cfg(per_unit=True))
    v = setup["states"]
    v_pu = module._to_feature_frame(v)
    assert torch.allclose(module._from_feature_frame(v_pu), v, rtol=1e-12, atol=1e-9)
    # a per-unit state is O(1) while the absolute one is kilovolts: the whole point
    assert 0.5 < float(v_pu.abs().select(-2, 0).mean()) < 2.0
    assert float(v.abs().select(-2, 0).mean()) > 1e3

    feats = module._encode_normalize(v)  # [B, N, H, C]
    back = module._from_feature_frame(
        module.encoding.decode(module.normalizer.inverse_transform(feats)).transpose(
            -2, -1
        )
    )
    assert torch.allclose(back, v, rtol=1e-8, atol=1e-6)


def test_the_voltage_decode_returns_volts_and_takes_a_step(setup):
    """With the model output forced to the true state's features the estimate IS the state
    (so the decode multiplies the base back), and a real model trains one optimizer step."""
    module = _module(setup, _cfg(per_unit=True))
    v = setup["states"]
    module.model = _ConstantModel(_flatten_features(module._encode_normalize(v)))
    _, _, pred_v, target_v, _, _, _ = module._step({"v": v}, train=False)
    assert torch.allclose(pred_v, v, rtol=1e-8, atol=1e-6)
    assert torch.equal(target_v, v)  # the target stays in volts too
    # ... and that is kilovolts, not per-unit
    assert float(pred_v.detach().abs().max()) > 1e3

    trained = _module(setup, _cfg(per_unit=True), masker="random")
    opt = trained.configure_optimizers()
    optimizer = opt["optimizer"] if isinstance(opt, dict) else opt
    loss = trained._step({"v": v}, train=True)[0]
    loss.backward()
    optimizer.step()
    grads = [p.grad for p in trained.model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0


# =============================================================================
# The injection frame: the head predicts per-unit, the solve consumes amperes
# =============================================================================
def _injection_features(module, i_amperes):
    """The true injections as the model's raw output block ``[B, N, H·C_out]``."""
    return _flatten_features(
        module.inj_normalizer.transform(
            _to_nhc(module.inj_encoding.encode(module._to_injection_frame(i_amperes)))
        )
    )


def test_the_injection_decode_scales_the_prediction_back_to_amperes(setup):
    """Forced to the true per-unit injections, the decode reconstructs the true state."""
    module = _module(setup, _cfg(per_unit=True, decode="injection"))
    v = setup["states"]
    i_true = setup["solver"].injection_targets(v)  # amperes
    assert torch.allclose(
        module._from_injection_frame(module._to_injection_frame(i_true)),
        i_true,
        rtol=1e-12,
        atol=1e-9,
    )
    # the per-unit injections are O(1)-ish, the amperes are not
    assert float(module._to_injection_frame(i_true).abs().max()) < 1e2
    assert float(i_true.abs().max()) > 1e2

    module.model = _ConstantModel(_injection_features(module, i_true))
    _, _, pred_v, target_v, _, _, _ = module._step({"v": v}, train=False)
    assert torch.allclose(pred_v, target_v, rtol=1e-6, atol=1e-6)


def test_the_injection_frame_survives_a_checkpoint_round_trip(setup, tmp_path):
    """The config + the normalizers travel, so a reloaded module reproduces the loss."""
    from injection_hse.masking import build_masker
    from injection_hse.train import SEModule

    module = _module(setup, _cfg(per_unit=True, decode="injection"))
    batch = {"v": setup["states"]}
    with torch.no_grad():
        before = module._step(batch, train=False)[0]

    ckpt = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(ckpt)
    path = tmp_path / "per_unit.ckpt"
    torch.save(ckpt, path)

    reloaded = SEModule.from_checkpoint(path, grid=setup["grid"])
    assert reloaded._per_unit is True
    assert reloaded._injection_base() is not None
    torch.testing.assert_close(
        reloaded._injection_base(), setup["i_base"], rtol=1e-12, atol=0.0
    )
    reloaded.set_masker(build_masker("full"))
    with torch.no_grad():
        after = reloaded._step(batch, train=False)[0]
    torch.testing.assert_close(after, before)


def test_the_calibrated_anchor_weights_keep_only_the_base_ratio(setup):
    """``σ_I = σ_I_pu·i_base`` and ``σ_y = σ_y_pu·V_base``: the weight keeps only the RATIO of
    the two bases, ``i_base[h,r]/V_base[r]``, which on this single-voltage-class feeder is the
    operator scale ``s(h)`` at every row — so the voltage weight is the base-free
    ``(σ_I_pu·s(h)/σ_y_pu)²`` and the CT channels land on the same dimensionless trust. The
    single-grid mirror of what the multi-grid bank does per member."""
    module = _module(
        setup,
        _cfg(
            per_unit=True,
            decode="injection",
            measure_current=True,
            anchor=1.0,
            weighting="noise_calibrated",
        ),
    )
    solver = module.solver
    assert solver.anchor_calibrated
    rho = float(module.cfg.physics.prior_injection_rel_sigma)
    clean = float(module.cfg.physics.clean_sensor_rel_sigma)
    from injection_hse.train.anchoring import channel_scale

    scale = solver.operator_scale()  # [H]
    s_inj = module.inj_normalizer.aux_scale.to(torch.float64)
    s_v = channel_scale(module.normalizer)  # [H, N] per-unit, each row's own
    s_e = channel_scale(module.edge_normalizer)  # [H, E]

    cal_v = solver.anchor_voltage_calibration  # [H, N]
    expected_v = ((rho * s_inj * scale)[:, None] / (clean * s_v)) ** 2  # base-free
    torch.testing.assert_close(cal_v, expected_v)
    cal_i = solver.anchor_current_calibration  # [H, E]
    expected_i = ((rho * s_inj)[:, None] / (clean * s_e)) ** 2  # base-free
    torch.testing.assert_close(cal_i, expected_i)


# =============================================================================
# The frame reaches every input block
# =============================================================================
@pytest.mark.parametrize("priors", [["kcl"], ["linear_state"], ["kcl", "linear_state"]])
def test_the_input_priors_train_in_frame(setup, priors):
    """Each prior lands in ITS normalizer's space and one optimizer step runs."""
    module = _module(setup, _cfg(per_unit=True, priors=priors), masker="random")
    v = setup["states"]
    loss = module._step({"v": v}, train=True)[0]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0

    if "kcl" in priors:
        # fully observed, the decoded kcl channels are the true injections IN AMPERES:
        # the per-unit base applied on the way in comes back off on the way out
        full = _module(setup, _cfg(per_unit=True, priors=["kcl"]))
        n, h = full.n_nodes, full.n_harmonics
        c_inj = full.inj_encoding.n_features
        mask = torch.ones(int(v.shape[0]), n, dtype=torch.bool)
        block = full.input_priors(v, mask, None, None, None, None, None)
        assert block.shape == (int(v.shape[0]), n, h * c_inj + 1)
        kappa = full._from_injection_frame(
            full.inj_encoding.decode(
                full.inj_normalizer.inverse_transform(
                    _unflatten_features(block[..., : h * c_inj], h, c_inj)
                )
            ).transpose(-2, -1)
        )
        expected = setup["solver"].injection_targets(v)
        assert torch.allclose(kappa, expected, rtol=1e-5, atol=1e-6)


def test_the_measured_currents_are_edge_features_in_frame(setup):
    """The CT channels divide by the half-edge current base while the raw ``i_edges`` handed
    to the anchored solve stay in amperes; one optimizer step runs."""
    module = _module(
        setup,
        _cfg(per_unit=True, decode="injection", measure_current=True, anchor=1.0),
        masker="random",
    )
    v = setup["states"]
    b, n = int(v.shape[0]), module.n_nodes
    mask = torch.ones(b, n, dtype=torch.bool)
    edge_obs, edge_mask, _, i_edges = module._edge_observations(
        v, mask, None, (b,), v.device
    )
    torch.testing.assert_close(i_edges, setup["edge_op"](v))  # amperes, untouched
    e_base = module._edge_current_base()
    expected = module.edge_normalizer.transform(
        _to_nhc(module.encoding.encode(i_edges / e_base))
    )
    torch.testing.assert_close(
        edge_obs, expected.reshape(*expected.shape[:-2], -1), rtol=1e-10, atol=1e-12
    )
    assert bool(edge_mask.all())

    loss = module._step({"v": v}, train=True)[0]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0


def test_per_unit_currents_without_a_solver_are_refused(setup):
    """The per-unit CURRENT base is the network's operator scale — a module built without
    the solver must fail loudly, not scale its edge channels by something else."""
    from injection_hse.train import SEModule

    cfg = _cfg(per_unit=True, measure_current=True)
    assert cfg.needs_solver  # the entry points build it for exactly this reason
    normalizer, _, edge_normalizer = _normalizers(setup, cfg)
    with pytest.raises(ValueError, match="per-unit current base"):
        SEModule(
            cfg,
            n_nodes=int(setup["states"].shape[-1]),
            normalizer=normalizer,
            graph=setup["graph"],
            base_v=setup["base_v"],
            edge_op=setup["edge_op"],
            edge_normalizer=edge_normalizer,
            grid=setup["grid"],
        )


# =============================================================================
# End to end: the entry point fits every normalizer in the frame
# =============================================================================
def test_train_experiment_fits_every_normalizer_in_the_frame(
    cigre_grid, se_dataset, tmp_path
):
    """One real run: the state normalizer is fit per-unit, the solver is built for the
    per-unit CURRENT base even under a direct-voltage decode, and the estimate the module
    returns is still in volts."""
    from injection_hse.config import DNNConfig, OptimConfig
    from injection_hse.data import StateCollate, build_data_source
    from injection_hse.train import SEModule, train_experiment

    cfg = ExperimentConfig(
        name="per-unit-e2e",
        seed=0,
        orders=[1, 3, 5, 7, 9, 11, 13],
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[32], dropout=0.0),
        normalization=NormalizationConfig(per_unit=True),
        measurement=MeasurementConfig(
            voltage_nodes=["Bus R4"], measure_current=True, slack_known=True
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
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
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    module = train_experiment(cfg, grid=cigre_grid)
    assert module._per_unit is True
    # the solver exists for the current base alone (the decode is the direct voltage one)
    assert module.solver is not None and module.cfg.physics.decode == "voltage"
    assert module._injection_base() is not None
    # a per-unit fit puts the fundamental's scale near 1, not near the base voltage
    assert float(module.normalizer.cartesian_scale[0]) < 10.0
    assert float(module.base_v.max()) > 100.0

    source = build_data_source(cfg.data, grid=cigre_grid)
    batch = StateCollate()([source.val_dataset()[i] for i in range(4)])
    # a deterministic mask on both sides, so the comparison below is about the frame
    from injection_hse.masking import build_masker

    module.set_masker(build_masker("full"))
    module.eval()
    out = module.infer(batch)
    # the estimate is in VOLTS: its magnitude sits within an order of the row base
    ratio = out["pred_v"].abs().select(-2, 0) / module.base_v.to(
        out["pred_v"].real.dtype
    )
    assert 0.1 < float(ratio.mean()) < 10.0

    ckpt = tmp_path / "ckpt" / "stage_masked.ckpt"
    reloaded = SEModule.from_checkpoint(ckpt, grid=cigre_grid, source=source)
    assert reloaded._per_unit is True
    reloaded.set_masker(build_masker("full"))
    torch.testing.assert_close(reloaded.infer(batch)["pred_v"], out["pred_v"])


# =============================================================================
# The multi-grid capability contract
# =============================================================================
def test_the_multigrid_validator_rejects_the_field():
    """The multi-grid path is ALWAYS per-unit (``data.multigrid.per_unit``), so a knob that
    changes nothing there must fail rather than look applied."""
    dump = _cfg(per_unit=True).model_dump()
    dump["data"]["multigrid"] = MultiGridConfig().model_dump()
    with pytest.raises(ValueError, match="normalization.per_unit"):
        ExperimentConfig.model_validate(dump)


def test_the_field_is_classified():
    from injection_hse.config import MULTIGRID_IGNORED, unclassified_multigrid_fields

    assert "normalization.per_unit" in MULTIGRID_IGNORED
    assert unclassified_multigrid_fields() == []
