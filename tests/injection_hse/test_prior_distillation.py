"""The prior-distillation term (``LossConfig.prior_distillation_weight``).

The injection decode's supervision is flat in the injection directions the measurements
already determine, and the TRUE injections are not identifiable from a sparse sensor set, so
neither the voltage term nor the direct injection term builds the injection map. This term
supervises the predicted injections against a fitted LINEAR INJECTION PRIOR's predictions —
a label that is by construction a function of the measurements.

These tests pin what the term must satisfy: weight ``0`` never evaluates a teacher and is the
previous loss exactly; a positive weight adds a finite, non-zero component that changes the
loss and back-propagates to the model; the teacher reads ITS OWN fitted placement rather than
whatever mask a stage drew, so the label is the same function of the data in every stage; the
fitted map travels with the checkpoint, so a reload distils against the same teacher; the
per-stage override anneals the term; a positive weight with no teacher installed is refused
rather than silently trained through; and the multi-grid path rejects the knob it cannot read.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    GNNConfig,
    LossConfig,
    MaskStageConfig,
    MeasurementConfig,
    MultiGridConfig,
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.encoding import TrigEncoding

#: The deployed placement of these tests: three sensor buses plus the metered slack.
SENSOR_NODES = ("Bus R4", "Bus C5", "Bus I2")


@pytest.fixture(scope="module")
def setup(cigre_grid, se_dataset):
    """Data source + fitted voltage normalizer + one collated batch (shared)."""
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


def _config(setup, *, weight: float, decode: str = "injection") -> ExperimentConfig:
    """The experiment config of a distilling run at the deployed placement."""
    return ExperimentConfig(
        name="prior-distillation",
        seed=0,
        orders=list(setup["source"].orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode=decode, anchor_voltage_weight=1.0),
        measurement=MeasurementConfig(
            voltage_nodes=list(SENSOR_NODES), slack_known=True
        ),
        loss=LossConfig(prior_distillation_weight=weight),
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


def _module(setup, grid, *, weight: float, with_prior: bool = True):
    """An injection-decode module at the deployed placement, teacher installed."""
    from injection_hse.data import base_voltage_per_row
    from injection_hse.masking import build_masker
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule, attach_injection_prior

    source = setup["source"]
    orders = list(source.orders)
    solver = NetworkSolver.from_grid(grid, orders, anchor_v_weight=1.0)
    inj_norm = fit_injection_normalizer(solver, setup["train_v"], TrigEncoding())
    cfg = _config(setup, weight=weight)
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
    if with_prior:
        attach_injection_prior(
            module,
            cfg.model_copy(
                update={
                    "loss": cfg.loss.model_copy(
                        update={"prior_distillation_weight": 1.0}
                    )
                }
            ),
            setup["train_v"],
            setup["observed_rows"],
        )
    module.set_masker(build_masker("placement", observed_rows=setup["observed_rows"]))
    return module


# =============================================================================
# The fitted map records its placement
# =============================================================================
def test_the_fit_records_the_placement_it_read(cigre_grid, setup, tmp_path):
    """``observed_rows`` is the fixed placement the fit read, and it survives a round trip."""
    from injection_hse.analysis import LinearInjectionPrior

    mod = _module(setup, cigre_grid, weight=0.0)
    prior = mod.injection_prior.prior
    assert prior.observed_rows is not None
    assert sorted(prior.observed_rows.tolist()) == sorted(
        int(r) for r in setup["observed_rows"]
    )
    # a run without metered branch currents has no edge block in its features
    assert prior.observed_edge_rows is None
    restored = LinearInjectionPrior.load(prior.save(tmp_path / "linear_prior.npz"))
    assert restored.observed_rows.tolist() == prior.observed_rows.tolist()
    assert restored.observed_edge_rows is None


# =============================================================================
# The loss term
# =============================================================================
def test_zero_weight_evaluates_no_teacher(cigre_grid, setup):
    """Weight ``0`` is the previous behaviour: no component, no teacher call."""
    mod = _module(setup, cigre_grid, weight=0.0)
    calls = _count_teacher_calls(mod)
    loss = mod._step(setup["batch"], train=True)[0]
    assert torch.isfinite(loss)
    assert calls[0] == 0
    assert set(mod._loss_components) == {"voltage"}
    assert torch.equal(mod._loss_components["voltage"], loss.detach())


def _count_teacher_calls(module) -> list[int]:
    """Wrap the teacher so every label evaluation is counted."""
    counter = [0]
    teacher = module.injection_prior
    original = teacher.injections

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    teacher.injections = counted
    return counter


def test_the_term_changes_the_loss_and_reaches_the_model(cigre_grid, setup):
    """``loss = voltage + w·distill``, with a finite, non-zero gradient into the model."""
    mod = _module(setup, cigre_grid, weight=0.0)
    batch = setup["batch"]
    params = [p for p in mod.model.parameters() if p.requires_grad]
    mod._mask_gen.manual_seed(1)
    base = mod._step(batch, train=True)[0]
    g_base = torch.autograd.grad(base, params)

    mod.cfg = mod.cfg.model_copy(
        update={
            "loss": mod.cfg.loss.model_copy(update={"prior_distillation_weight": 0.5})
        }
    )
    calls = _count_teacher_calls(mod)
    mod._mask_gen.manual_seed(1)
    loss = mod._step(batch, train=True)[0]
    assert calls[0] == 1
    parts = mod._loss_components
    assert set(parts) == {"voltage", "prior_distillation"}
    for name, value in parts.items():
        assert torch.isfinite(value), name
        assert float(value) > 0.0, name
    assert torch.equal(parts["voltage"], base.detach())
    assert torch.allclose(loss, parts["voltage"] + 0.5 * parts["prior_distillation"])
    assert not torch.allclose(loss, base)

    g_loss = torch.autograd.grad(loss, params)
    g_term = [2.0 * (b - a) for a, b in zip(g_base, g_loss)]
    assert all(torch.isfinite(g).all() for g in g_term)
    assert sum(float(g.abs().sum()) for g in g_term) > 0.0


def test_the_teacher_reads_its_own_placement(cigre_grid, setup):
    """The label is the same function of the data whatever mask the stage drew.

    A distillation target computed from the stage's mask would change meaning between the
    curriculum's random stages and its placement stage; the teacher reads the sensors the map
    was fitted on instead.
    """
    from injection_hse.masking import build_masker

    mod = _module(setup, cigre_grid, weight=1.0)
    v = setup["batch"]["v"].to(mod._complex_dtype)
    fitted = mod._prior_distillation_target_features(v, None)

    mod.set_masker(build_masker("full"))
    assert torch.equal(mod._prior_distillation_target_features(v, None), fitted)


def test_the_label_is_the_priors_prediction_not_the_truth(cigre_grid, setup):
    """The term's label is the prior's estimate of ``Y·V``, measurably not ``Y·V`` itself."""
    mod = _module(setup, cigre_grid, weight=1.0)
    v = setup["batch"]["v"].to(mod._complex_dtype)
    prior_feats = mod._prior_distillation_target_features(v, None)
    true_feats = mod._injection_target_features(v, None)
    assert prior_feats.shape == true_feats.shape
    assert not torch.allclose(prior_feats, true_feats)
    # ... but it is an ESTIMATE of it: the two agree far better than a zero prediction does
    rows = torch.as_tensor(mod.injection_prior.target_rows)
    truth = mod.solver.injection_targets(v).index_select(-1, rows)
    estimate = mod.injection_prior.injections(mod, v).index_select(-1, rows)
    assert float((estimate - truth).abs().mean()) < float(truth.abs().mean())


# =============================================================================
# Reproducibility: the teacher travels with the checkpoint
# =============================================================================
def test_the_teacher_travels_with_the_checkpoint(cigre_grid, setup, tmp_path):
    """A reload distils against the same fitted map — the objective reproduces exactly."""
    from injection_hse.train import SEModule

    mod = _module(setup, cigre_grid, weight=0.5)
    mod._mask_gen.manual_seed(3)
    loss = mod._step(setup["batch"], train=True)[0]

    ckpt: dict = {"state_dict": mod.state_dict()}
    mod.on_save_checkpoint(ckpt)
    assert "injection_prior" in ckpt
    path = tmp_path / "stage_placement.ckpt"
    torch.save(ckpt, path)

    reloaded = SEModule.from_checkpoint(path, grid=cigre_grid)
    assert reloaded.injection_prior is not None
    assert torch.equal(reloaded.injection_prior.coef, mod.injection_prior.coef)
    assert torch.equal(
        reloaded.injection_prior.node_mask, mod.injection_prior.node_mask
    )
    reloaded.set_masker(mod._masker)
    reloaded.train()
    reloaded._mask_gen.manual_seed(3)
    assert torch.allclose(
        reloaded._step(setup["batch"], train=True)[0], loss, rtol=1e-6, atol=1e-9
    )


# =============================================================================
# The anneal
# =============================================================================
def test_the_per_stage_weight_anneals_the_term(cigre_grid, setup):
    """``MaskStageConfig.prior_distillation_weight`` overrides the config's, ``0`` included."""
    mod = _module(setup, cigre_grid, weight=1.0)
    assert mod._distillation_weight() == pytest.approx(1.0)
    mod.set_stage_distillation(0.0)
    assert mod._distillation_weight() == pytest.approx(0.0)
    calls = _count_teacher_calls(mod)
    mod._step(setup["batch"], train=True)
    assert calls[0] == 0
    assert "prior_distillation" not in mod._loss_components
    mod.set_stage_distillation(None)
    assert mod._distillation_weight() == pytest.approx(1.0)


def test_the_stage_manager_installs_the_stage_weight(cigre_grid, setup):
    """The stage's override reaches the module through the ``StageManager``."""
    from injection_hse.train import StageManager

    mod = _module(setup, cigre_grid, weight=1.0)
    stages = [
        MaskStageConfig(
            name="placement",
            kind="placement",
            target_loss=1e-12,
            max_epochs=1,
            prior_distillation_weight=0.0,
        )
    ]
    cfg = mod.cfg.model_copy(update={"curriculum": CurriculumConfig(stages=stages)})
    manager = StageManager(cfg, observed_rows=setup["observed_rows"])
    stage = cfg.curriculum.stages[0]
    mod.set_stage_distillation(
        stage.prior_distillation_weight
        if stage.prior_distillation_weight is not None
        else manager.cfg.loss.prior_distillation_weight
    )
    assert mod._distillation_weight() == pytest.approx(0.0)


def test_the_schedule_parser_stamps_the_stages():
    """``parse_stage_weights`` / ``apply_stage_weights`` express the anneal in the config."""
    from injection_hse.curriculum import apply_stage_weights, parse_stage_weights

    schedule = parse_stage_weights(
        "placement:1,placement_anneal:0", flag="--prior-distillation-schedule"
    )
    assert schedule == {"placement": 1.0, "placement_anneal": 0.0}
    stages = [
        MaskStageConfig(name=name, kind="placement", target_loss=1.0, max_epochs=1)
        for name in ("placement", "placement_anneal", "other")
    ]
    stamped, unmatched = apply_stage_weights(
        stages, schedule, field="prior_distillation_weight"
    )
    assert [s.prior_distillation_weight for s in stamped] == [1.0, 0.0, None]
    assert unmatched == set()
    with pytest.raises(SystemExit):
        parse_stage_weights("placement", flag="--flag")
    with pytest.raises(SystemExit):
        parse_stage_weights("placement:-1", flag="--flag")


# =============================================================================
# Guards
# =============================================================================
def test_a_missing_teacher_is_refused(cigre_grid, setup):
    """A positive weight with no fitted prior raises rather than training another objective."""
    mod = _module(setup, cigre_grid, weight=0.5, with_prior=False)
    assert mod.injection_prior is None
    with pytest.raises(ValueError, match="no linear injection prior is installed"):
        mod._step(setup["batch"], train=True)


def test_the_weight_without_the_injection_decode_is_ignored(setup):
    """The term needs a predicted injection: the voltage decode warns and drops it."""
    from injection_hse.train import SEModule

    source = setup["source"]
    cfg = _config(setup, weight=0.5, decode="voltage")
    with pytest.warns(UserWarning, match="prior_distillation_weight"):
        module = SEModule(
            cfg,
            n_nodes=source.n_nodes,
            normalizer=setup["normalizer"],
            graph=source.graph(),
        )
    assert module._distillation_weight() == 0.0


def test_a_voltage_decode_run_cannot_fit_a_teacher(cigre_grid, setup):
    """``attach_injection_prior`` names the decode rather than fitting an unusable map."""
    from injection_hse.train import SEModule, attach_injection_prior

    source = setup["source"]
    cfg = _config(setup, weight=1.0, decode="voltage")
    with pytest.warns(UserWarning):
        module = SEModule(
            cfg,
            n_nodes=source.n_nodes,
            normalizer=setup["normalizer"],
            graph=source.graph(),
        )
    with pytest.raises(ValueError, match="physics.decode='injection'"):
        attach_injection_prior(module, cfg, setup["train_v"], setup["observed_rows"])


def test_the_history_collector_records_the_component():
    """The training-history parquet carries the distillation term beside the others."""
    from injection_hse.evaluation.history import _DIAGNOSTICS

    assert "train_prior_distillation_loss" in _DIAGNOSTICS
    assert "val_prior_distillation_loss" in _DIAGNOSTICS


def test_validation_logs_the_component(cigre_grid, setup):
    """``val_prior_distillation_loss`` is published beside the other loss components."""
    captured: dict[str, float] = {}

    def _capture(name, value, **_kwargs):
        captured[name] = float(value.detach() if torch.is_tensor(value) else value)

    mod = _module(setup, cigre_grid, weight=0.5)
    mod.log = _capture  # type: ignore[method-assign]
    mod.validation_step(setup["batch"], 0)
    assert captured["val_prior_distillation_loss"] > 0.0

    off = _module(setup, cigre_grid, weight=0.0)
    captured.clear()
    off.log = _capture  # type: ignore[method-assign]
    off.validation_step(setup["batch"], 0)
    assert "val_prior_distillation_loss" not in captured


def test_multigrid_rejects_a_non_default_weight(tmp_path):
    """The linear prior is ONE grid's map, so a cross-grid run refuses the knob."""
    base = dict(
        name="mg",
        seed=0,
        orders=[1, 3],
        model=GNNConfig(hidden_dim=8, num_layers=1),
        physics=PhysicsConfig(decode="injection"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="full", kind="full", target_loss=1.0, max_epochs=1)
            ]
        ),
        train=TrainConfig(accelerator="cpu", devices=1),
    )
    data = DataConfig(
        dataset_path=str(tmp_path), multigrid=MultiGridConfig(placement_fraction=0.5)
    )
    with pytest.raises(ValueError, match="prior_distillation_weight"):
        ExperimentConfig(
            **base, data=data, loss=LossConfig(prior_distillation_weight=1.0)
        )
    with pytest.raises(ValueError, match="prior_distillation_weight"):
        ExperimentConfig(
            **{
                **base,
                "curriculum": CurriculumConfig(
                    stages=[
                        MaskStageConfig(
                            name="full",
                            kind="full",
                            target_loss=1.0,
                            max_epochs=1,
                            prior_distillation_weight=1.0,
                        )
                    ]
                ),
            },
            data=data,
        )
