"""The loss preconditioner and the prior-residual decode (``LossConfig.preconditioner``,
``PhysicsConfig.injection_prior_residual``).

The injection decode's loss is an exact quadratic in the head's output whose Hessian is
grid-constant and badly conditioned (half the directions flat on CIGRE LV), and a
closed-form linear prior fitted from a few hundred snapshots beats a network trained from
scratch. These tests pin the two answers: the preconditioner leaves every forward pass
byte-identical and multiplies the gradient at the head's output by the inverse curvature
(exactly, per order, on the non-slack rows), whitening the well-determined directions to
unit curvature; the residual decode starts training at the prior's own estimate, reads
only the placement rows a stage observes, and both are refused where they have no
meaning (the voltage decode, a non-affine head, a run without a placement).
"""

from __future__ import annotations

import lightning as L
import pytest
import torch
from pydantic import ValidationError

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    EncodingConfig,
    ExperimentConfig,
    LossConfig,
    MaskStageConfig,
    MeasurementConfig,
    PhysicsConfig,
)
from injection_hse.encoding import TrigEncoding

SENSOR_NODES = ("Bus R4", "Bus C5", "Bus I2")


@pytest.fixture(scope="module")
def setup(cigre_grid, se_dataset):
    from injection_hse.data import MeasurementModel, StateCollate, build_data_source
    from injection_hse.normalization import HarmonicNormalizer

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    enc = TrigEncoding()
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(64)])
    norm = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    mm = MeasurementModel.resolve(cigre_grid, list(SENSOR_NODES), slack_known=True)
    return {
        "source": source,
        "train_v": train_v,
        "normalizer": norm,
        "batch": batch,
        "observed_rows": mm.observed_rows(cigre_grid),
    }


def _config(setup, **overrides) -> ExperimentConfig:
    kw = dict(
        name="loss-geometry",
        seed=0,
        orders=list(setup["source"].orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        encoding=EncodingConfig(injection_kind="cartesian"),
        physics=PhysicsConfig(decode="injection", anchor_voltage_weight=1.0),
        measurement=MeasurementConfig(
            voltage_nodes=list(SENSOR_NODES), slack_known=True
        ),
        loss=LossConfig(),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="placement",
                    kind="placement",
                    target_loss=1e-12,
                    max_epochs=1,
                )
            ]
        ),
    )
    kw.update(overrides)
    return ExperimentConfig(**kw)


def _module(setup, grid, cfg):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.encoding import build_injection_encoding
    from injection_hse.masking import build_masker
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule, attach_injection_prior, attach_loss_preconditioner

    source = setup["source"]
    solver = NetworkSolver.from_grid(grid, list(source.orders), anchor_v_weight=1.0)
    inj_norm = fit_injection_normalizer(
        solver, setup["train_v"], build_injection_encoding(cfg.encoding)
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=setup["normalizer"],
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        solver=solver,
        inj_normalizer=inj_norm,
        grid=grid,
    )
    attach_injection_prior(module, cfg, setup["train_v"], setup["observed_rows"])
    attach_loss_preconditioner(module, cfg, setup["observed_rows"])
    module.set_masker(build_masker("placement", observed_rows=setup["observed_rows"]))
    module.set_stage_anchor(v_weight=1.0, i_weight=0.0)
    return module


# =============================================================================
# The preconditioner
# =============================================================================
def test_the_forward_pass_is_untouched_and_the_gradient_is_not(cigre_grid, setup):
    plain = _module(setup, cigre_grid, _config(setup))
    pre = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="placement"))
    )
    assert plain.loss_preconditioner is None and pre.loss_preconditioner is not None
    pre.load_state_dict(plain.state_dict())
    batch = setup["batch"]
    l_plain = plain._step(batch, train=True)[0]
    l_pre = pre._step(batch, train=True)[0]
    assert torch.equal(l_plain.detach(), l_pre.detach())
    g_plain = torch.autograd.grad(l_plain, list(plain.model.parameters()))
    g_pre = torch.autograd.grad(l_pre, list(pre.model.parameters()))
    assert all(torch.isfinite(g).all() for g in g_pre)
    assert not all(torch.allclose(a, b) for a, b in zip(g_plain, g_pre))
    # eval / inference never sees it
    out_plain, out_pre = plain.infer(batch), pre.infer(batch)
    assert torch.equal(out_plain["pred_v"], out_pre["pred_v"])


def test_the_backward_applies_the_matrix_per_order_on_the_non_slack_rows(
    cigre_grid, setup
):
    mod = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="placement"))
    )
    pre = mod.loss_preconditioner
    n, h, c = int(mod.n_nodes), int(mod.n_harmonics), int(mod.n_out_channels)
    torch.manual_seed(1)
    z = torch.randn((3, n, h, c), dtype=torch.float64, requires_grad=True)
    g = torch.randn((3, n, h, c), dtype=torch.float64)
    (pre(z) * g).sum().backward()
    rows = pre.rows
    r = int(rows.numel())
    for k in range(h):
        gk = torch.cat([g[:, rows, k, 0], g[:, rows, k, 1]], dim=-1)  # [3, 2R]
        expected = gk @ pre.matrices[k].T
        got = torch.cat([z.grad[:, rows, k, 0], z.grad[:, rows, k, 1]], dim=-1)
        assert torch.allclose(got, expected, atol=1e-10)
    # the slack rows keep the plain gradient
    keep = torch.ones(n, dtype=torch.bool).index_fill(0, rows, False)
    assert torch.equal(z.grad[:, keep], g[:, keep])
    assert r == n - int(keep.sum())


def test_the_matrices_whiten_the_loss_curvature(cigre_grid, setup):
    from injection_hse.train.preconditioner import _curvature_maps

    mod = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="placement"))
    )
    pre = mod.loss_preconditioner
    maps, rows = _curvature_maps(mod, setup["observed_rows"], mode="placement")
    assert torch.equal(rows, pre.rows.cpu())
    eps = pre.floor * pre.sigma_max
    for k in range(maps.shape[0]):
        hess = maps[k].T @ maps[k]  # [2R, 2R]
        p = pre.matrices[k].cpu()
        assert torch.allclose(p, p.T, atol=1e-9)
        # P is positive definite ...
        assert float(torch.linalg.eigvalsh(p).min()) > 0.0
        # ... and P·H has unit curvature (in σ_max² units) on every well-determined
        # direction, and never more than that on a flat one
        _, sv, vt = torch.linalg.svd(maps[k], full_matrices=False)
        curv = torch.einsum("ij,jk,ki->i", vt, p @ hess, vt.T) / pre.sigma_max**2
        strong = sv > 10.0 * eps
        assert strong.any()
        assert torch.all(curv[strong] > 0.99) and torch.all(curv[strong] < 1.0 + 1e-6)
        assert torch.all(curv <= 1.0 + 1e-6)
    assert sum(pre.n_flat) > 0
    assert "placement" in pre.summary()


def test_the_plain_mode_reads_the_slack_only_decode(cigre_grid, setup):
    placement = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="placement"))
    )
    plain = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="plain"))
    )
    assert not torch.allclose(
        placement.loss_preconditioner.spectra, plain.loss_preconditioner.spectra
    )
    assert plain.loss_preconditioner.mode == "plain"


def test_the_checkpoint_records_the_preconditioner_and_infers_without_it(
    cigre_grid, setup, tmp_path
):
    from injection_hse.masking import build_masker
    from injection_hse.train import SEModule

    mod = _module(
        setup, cigre_grid, _config(setup, loss=LossConfig(preconditioner="placement"))
    )
    ckpt: dict = {"state_dict": mod.state_dict()}
    mod.on_save_checkpoint(ckpt)
    assert ckpt["loss_preconditioner"]["mode"] == "placement"
    assert ckpt["loss_preconditioner"]["n_flat"] == mod.loss_preconditioner.n_flat
    path = tmp_path / "pre.ckpt"
    torch.save(ckpt, path)
    back = SEModule.from_checkpoint(path, grid=cigre_grid)
    assert back.loss_preconditioner is None
    # the sensor placement is a run-time choice, not checkpoint state: install the same one
    back.set_masker(build_masker("placement", observed_rows=setup["observed_rows"]))
    assert torch.allclose(
        back.infer(setup["batch"])["pred_v"], mod.infer(setup["batch"])["pred_v"]
    )


# =============================================================================
# The prior residual
# =============================================================================
def test_the_residual_decode_starts_exactly_at_the_prior(cigre_grid, setup):
    from injection_hse.physics import solve_injection
    from injection_hse.analysis.baselines import measured_inputs

    cfg = _config(
        setup,
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, injection_prior_residual=True
        ),
    )
    mod = _module(setup, cigre_grid, cfg)
    assert mod._prior_residual and mod.injection_prior is not None
    # the output layer starts at zero, so the network contributes nothing
    last = [m for m in mod.model.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert float(last.weight.abs().max()) == 0.0
    batch = setup["batch"]
    out = mod.infer(batch)
    v = batch["v"].to(mod._complex_dtype)
    mask = out["mask"]
    v_meas, _, _ = measured_inputs(mod, v, mask)
    i_prior = mod.injection_prior.injections(mod, v_meas)
    with torch.no_grad():
        v_prior = solve_injection(mod, i_prior, v, mask=mask)
    assert torch.allclose(out["pred_v"], v_prior, rtol=1e-5, atol=1e-6)
    # ... and a training step moves it away from the prior with a finite gradient
    loss = mod._step(batch, train=True)[0]
    grads = torch.autograd.grad(loss, list(mod.model.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().max()) > 0.0 for g in grads)


def test_inside_a_sparser_stage_the_residual_reads_only_observed_rows(
    cigre_grid, setup
):
    cfg = _config(
        setup,
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, injection_prior_residual=True
        ),
    )
    mod = _module(setup, cigre_grid, cfg)
    teacher = mod.injection_prior
    batch = setup["batch"]
    v = batch["v"].to(mod._complex_dtype)
    n = int(mod.n_nodes)
    full = torch.zeros((v.shape[0], n), dtype=torch.bool)
    full[:, torch.as_tensor(setup["observed_rows"])] = True
    # hide the first sensor bus's rows for the odd samples
    rows = torch.as_tensor(setup["observed_rows"])
    hidden = rows[:3]
    sparse = full.clone()
    sparse[1::2, hidden] = False
    at_placement = teacher.injections(mod, v, sample_mask=full)
    partial = teacher.injections(mod, v, sample_mask=sparse)
    assert torch.allclose(at_placement, teacher.injections(mod, v))
    assert torch.allclose(partial[0::2], at_placement[0::2])
    assert not torch.allclose(partial[1::2], at_placement[1::2])
    assert torch.isfinite(partial).all()
    avail = teacher.feature_availability(sparse)
    assert avail.shape[0] == v.shape[0] and avail.dtype == torch.bool
    assert int(avail[1].sum()) < int(avail[0].sum()) == int(avail.shape[1])
    # the module's own residual call is the masked one
    same = mod._prior_residual_injection(v, None, sparse, None)
    assert torch.allclose(same.to(torch.complex128), partial.to(torch.complex128))


def test_the_injection_space_terms_see_the_full_injection(cigre_grid, setup):
    """With the residual, distillation compares (prior + network) with the prior — so at the
    zero-initialised start the distillation term is exactly zero."""
    cfg = _config(
        setup,
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, injection_prior_residual=True
        ),
        loss=LossConfig(prior_distillation_weight=1.0),
    )
    mod = _module(setup, cigre_grid, cfg)
    mod._step(setup["batch"], train=False)
    assert float(mod._loss_components["prior_distillation"]) == pytest.approx(
        0.0, abs=1e-12
    )


def test_the_analysis_decode_seam_carries_the_residual(cigre_grid, setup):
    """``injection_hse.analysis`` decodes the raw network output itself (attribution, occlusion, the
    band ablation); with the residual on it must reach the same state as the training step,
    while the model-free rows of the band ablation stay what they are."""
    from injection_hse.analysis.attribution import _decode_state, _model_output, _masked_features
    from injection_hse.analysis.occlusion import estimator_band_ablation
    from injection_hse.data import base_voltage_per_row

    cfg = _config(
        setup,
        physics=PhysicsConfig(
            decode="injection", anchor_voltage_weight=1.0, injection_prior_residual=True
        ),
    )
    mod = _module(setup, cigre_grid, cfg)
    batch = setup["batch"]
    v = batch["v"].to(mod._complex_dtype)
    out = mod.infer(batch)
    mask = out["mask"]
    with torch.no_grad():
        flat = _model_output(mod, _masked_features(mod, v, mask), mask)
        via_analysis = _decode_state(mod, flat, v, mask=mask)
    assert torch.allclose(via_analysis, out["pred_v"], rtol=1e-5, atol=1e-6)
    table = estimator_band_ablation(
        mod,
        v,
        mask,
        base_v=base_voltage_per_row(cigre_grid),
        device="cpu",
        linear_prior=mod.injection_prior.prior,
    ).to_dicts()
    rows = {r["variant"]: r for r in table}
    # at the zero-initialised start the deployed estimate IS the prior's ...
    assert rows["baseline"]["mae_h1"] == pytest.approx(
        rows["linear_prior"]["mae_h1"], rel=1e-4, abs=1e-9
    )
    # ... and the model-free references are untouched by the residual
    plain = _module(setup, cigre_grid, _config(setup))
    ref = {
        r["variant"]: r
        for r in estimator_band_ablation(
            plain,
            v,
            mask,
            base_v=base_voltage_per_row(cigre_grid),
            device="cpu",
            linear_prior=mod.injection_prior.prior,
        ).to_dicts()
    }
    for name in ("anchors_only", "linear_prior"):
        assert rows[name]["mae_h1"] == pytest.approx(ref[name]["mae_h1"], rel=1e-6)


# =============================================================================
# Refusals
# =============================================================================
def test_the_config_refuses_what_has_no_meaning(setup):
    with pytest.raises(ValidationError, match="decode='injection'"):
        _config(
            setup,
            physics=PhysicsConfig(decode="voltage"),
            loss=LossConfig(preconditioner="plain"),
        )
    with pytest.raises(ValidationError, match="Cartesian"):
        _config(
            setup,
            encoding=EncodingConfig(injection_kind="trig"),
            loss=LossConfig(preconditioner="plain"),
        )
    with pytest.raises(ValidationError, match="decode='injection'"):
        _config(
            setup,
            physics=PhysicsConfig(decode="voltage", injection_prior_residual=True),
        )
    with pytest.raises(ValidationError, match="placement"):
        _config(
            setup,
            physics=PhysicsConfig(decode="injection", injection_prior_residual=True),
            measurement=MeasurementConfig(voltage_nodes=[], slack_known=True),
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="full", kind="full", target_loss=1e-12, max_epochs=1
                    )
                ]
            ),
        )
    with pytest.raises(ValidationError):
        _config(
            setup, loss=LossConfig(preconditioner="placement", preconditioner_floor=0.0)
        )
