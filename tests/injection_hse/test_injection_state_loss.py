"""Direct supervision of the predicted nodal injections, and the fixed-anchor regime.

The injection decode predicts one nodal injection current per node-phase ROW and the network
solve turns it into voltages; by default only those solved voltages are supervised, so the
injections are learned indirectly (the solve is blind to any injection error the network law
maps onto a negligible voltage change). ``LossConfig.injection_state_loss_split`` blends in a
SECOND supervised term directly on the predicted injections, against the true
``I(h) = Y(h)·V_true(h)`` in the normalized injection space.

These tests pin what such a blend must satisfy: a zero split is EXACTLY the previous
voltage-only loss (no extra term, no extra compute), a positive split combines two finite,
non-zero components and back-propagates to the model through BOTH, and the decode's
measurement anchoring runs at FIXED weights with no per-batch randomness — plus the workflow
flag that collapses a ranged anchor schedule to that fixed regime.
"""

from __future__ import annotations

import pytest
import torch


def _with_split(cfg, split: float, *, normalize: bool = True):
    """The same experiment config with a different injection/voltage supervision split."""
    return cfg.model_copy(
        update={
            "loss": cfg.loss.model_copy(
                update={
                    "injection_state_loss_split": float(split),
                    "injection_state_loss_normalize": bool(normalize),
                }
            )
        }
    )


@pytest.fixture(scope="module")
def inj_setup(cigre_grid, se_dataset):
    """Data source + fitted voltage/injection normalizers + one collated batch (shared)."""
    from injection_hse.config import DataConfig
    from injection_hse.data import StateCollate, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    enc = build_encoding("trig")
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    norm = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    return {
        "source": source,
        "encoding": enc,
        "train_v": train_v,
        "normalizer": norm,
        "batch": batch,
    }


def _module(setup, grid, *, split: float, anchor_v: float = 0.0):
    """An injection-decode :class:`~injection_hse.train.SEModule` at the given supervision split."""
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        LossConfig,
        MaskStageConfig,
        PhysicsConfig,
    )
    from injection_hse.data import base_voltage_per_row
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    source = setup["source"]
    orders = list(source.orders)
    solver = NetworkSolver.from_grid(grid, orders, anchor_v_weight=anchor_v)
    inj_norm = fit_injection_normalizer(solver, setup["train_v"], setup["encoding"])
    cfg = ExperimentConfig(
        name="injection-split",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode="injection", anchor_voltage_weight=anchor_v),
        loss=LossConfig(injection_state_loss_split=split),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    return SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=setup["normalizer"],
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        solver=solver,
        inj_normalizer=inj_norm,
        grid=grid,  # the identifiability diagnostic derives its row set from the grid
    )


def test_zero_split_is_the_voltage_only_loss(cigre_grid, inj_setup):
    """Split 0 computes the voltage term and nothing else — the pre-split behaviour exactly.

    The regression guard for the default: the reported loss must be BIT-identical to the
    voltage term the blended run reports as its own component. Only the voltage part is
    published (the injection term is never evaluated), and it equals the reported loss.
    """
    mod = _module(inj_setup, cigre_grid, split=0.0)
    batch = inj_setup["batch"]
    voltage_only = mod._step(batch, train=True)[0].detach()
    assert torch.isfinite(voltage_only)
    assert set(mod._loss_components) == {"voltage"}
    assert torch.equal(mod._loss_components["voltage"], voltage_only)

    mod.cfg = _with_split(mod.cfg, 0.5, normalize=False)
    blended = mod._step(batch, train=True)[0].detach()
    parts = mod._loss_components
    assert torch.equal(parts["voltage"], voltage_only)
    assert torch.allclose(
        blended, 0.5 * parts["voltage"] + 0.5 * parts["injection_state"]
    )


def test_split_reaches_the_model_through_both_terms(cigre_grid, inj_setup):
    """At split 0.5 both components are finite and non-zero, and each back-propagates.

    The blended gradient is ``0.5·g_voltage + 0.5·g_injection``; subtracting half the
    voltage-only gradient isolates the injection term's contribution, which must be
    non-zero — the direct term genuinely trains the model, it is not a passive readout.
    """
    mod = _module(inj_setup, cigre_grid, split=0.0)
    batch = inj_setup["batch"]
    params = [p for p in mod.model.parameters() if p.requires_grad]
    g_voltage = torch.autograd.grad(mod._step(batch, train=True)[0], params)

    mod.cfg = _with_split(mod.cfg, 0.5)
    blended = mod._step(batch, train=True)[0]
    parts = mod._loss_components
    for name, value in parts.items():
        assert torch.isfinite(value), name
        assert float(value) > 0.0, name
    g_blended = torch.autograd.grad(blended, params)

    g_injection = [2.0 * (b - 0.5 * v) for b, v in zip(g_blended, g_voltage)]
    assert all(torch.isfinite(g).all() for g in g_injection)
    assert sum(float(g.abs().sum()) for g in g_injection) > 0.0
    assert not all(
        torch.allclose(b, 0.5 * v) for b, v in zip(g_blended, g_voltage)
    )  # the blend is not merely a rescaled voltage gradient


def test_split_without_the_injection_decode_is_ignored(cigre_grid, inj_setup):
    """The term needs a predicted injection: the voltage decode warns and drops it."""
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        LossConfig,
        MaskStageConfig,
        PhysicsConfig,
    )
    from injection_hse.train import SEModule

    source = inj_setup["source"]
    cfg = ExperimentConfig(
        name="voltage-decode",
        seed=0,
        orders=list(source.orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode="voltage"),
        loss=LossConfig(injection_state_loss_split=0.5),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )
    with pytest.warns(UserWarning, match="injection_state_loss_split"):
        mod = SEModule(
            cfg,
            n_nodes=source.n_nodes,
            normalizer=inj_setup["normalizer"],
            graph=source.graph(),
        )
    loss = mod._step(inj_setup["batch"], train=True)[0]
    assert torch.isfinite(loss)
    assert set(mod._loss_components) == {"voltage"}
    assert torch.equal(mod._loss_components["voltage"], loss.detach())


def test_fixed_anchor_weights_have_no_per_batch_randomness(cigre_grid, inj_setup):
    """A stage anchored at FIXED weights is deterministic: one batch, one loss, twice.

    The per-batch weight randomization is the only stochastic element of the anchored
    decode (masks are deterministic in a fully observed stage), so repeating the same
    batch separates the two regimes. The comparison watches the VOLTAGE component: it is
    the one the anchored solve produces, while the injection term reads the predicted
    currents before any solve and is anchor-independent by construction.
    """

    def _run(mod, batch):
        total = mod._step(batch, train=True)[0].detach()
        return total, mod._loss_components["voltage"].clone()

    mod = _module(inj_setup, cigre_grid, split=0.5, anchor_v=1.0)
    batch = inj_setup["batch"]
    mod.set_stage_anchor(v_weight=1.0, i_weight=0.0)
    assert mod._anchor_ranges is None
    first, first_v = _run(mod, batch)
    second, second_v = _run(mod, batch)
    assert torch.equal(first, second)
    assert torch.equal(first_v, second_v)

    mod.set_stage_anchor(v_weight=1.0, i_weight=0.0, v_range=(0.1, 100.0))
    _, ranged_first_v = _run(mod, batch)
    _, ranged_second_v = _run(mod, batch)
    assert not torch.equal(ranged_first_v, ranged_second_v)


def test_anchor_fixed_collapses_ranges_to_midpoints():
    """``--anchor-fixed`` rewrites a ranged schedule as its midpoints: weights, no ranges."""
    from injection_hse.config import MaskStageConfig
    from injection_hse.curriculum import apply_anchor_schedule, parse_anchor_schedule

    spec = "random_25:0~1/0~0.1,placement:10/0.5"
    ranged = parse_anchor_schedule(spec)
    assert ranged["random_25"] == ((0.0, 1.0), (0.0, 0.1))
    fixed = parse_anchor_schedule(spec, fixed=True)
    assert fixed["random_25"] == ((0.5, 0.5), (0.05, 0.05))
    assert fixed["placement"] == ((10.0, 10.0), (0.5, 0.5))  # already fixed, unchanged

    stages = [
        MaskStageConfig(
            name=name,
            kind="random",
            observed_fraction=0.25,
            target_loss=1e-3,
            max_epochs=1,
        )
        for name in ("random_25", "placement")
    ]
    updated, unmatched = apply_anchor_schedule(stages, fixed)
    assert not unmatched
    by_name = {s.name: s for s in updated}
    assert by_name["random_25"].anchor_voltage_weight == 0.5
    assert by_name["random_25"].anchor_current_weight == 0.05
    assert by_name["placement"].anchor_voltage_weight == 10.0
    assert all(
        s.anchor_voltage_range is None and s.anchor_current_range is None
        for s in updated
    )
    # without the flag the same string keeps its ranges — the flag is what removes them
    ranged_stages, _ = apply_anchor_schedule(stages, ranged)
    assert {s.name: s for s in ranged_stages}["random_25"].anchor_voltage_range == (
        0.0,
        1.0,
    )


def test_fixed_schedule_installs_a_constant_solver_weight():
    """A fixed ``stage:V/C`` entry reaches the solver as a constant: training draws nothing."""
    from injection_hse.config import MaskStageConfig
    from injection_hse.train import SEModule
    from injection_hse.curriculum import apply_anchor_schedule, parse_anchor_schedule

    stages, _ = apply_anchor_schedule(
        [
            MaskStageConfig(
                name="placement", kind="placement", target_loss=1e-3, max_epochs=1
            )
        ],
        parse_anchor_schedule("placement:10/0.1"),
    )
    stage = stages[0]
    assert stage.anchor_voltage_range is None and stage.anchor_current_range is None

    class _Solver:  # minimal stand-in: the runtime scalars + the current capability
        anchor_v_weight = 0.0
        anchor_i_weight = 0.0
        can_anchor_current = True

    mod = SEModule.__new__(SEModule)
    mod.solver = _Solver()
    mod._bank = None  # single-grid run: no multi-grid operator bank to dial
    mod._anchor_gen = torch.Generator().manual_seed(0)
    mod._anchor_draw = "uniform"  # normally resolved from PhysicsConfig at construction
    mod.set_stage_anchor(
        v_weight=stage.anchor_voltage_weight,
        i_weight=stage.anchor_current_weight,
        v_range=stage.anchor_voltage_range,
        i_range=stage.anchor_current_range,
    )
    assert mod._anchor_ranges is None
    for _ in range(4):
        mod._apply_batch_anchor(train=True)
        assert mod.solver.anchor_v_weight == 10.0
        assert mod.solver.anchor_i_weight == 0.1


def test_the_split_is_the_realized_share(cigre_grid, inj_setup):
    """With normalization the configured split IS the fraction the injection term carries.

    The two residuals are not commensurate — the injection one runs orders of magnitude
    above the voltage one — so a raw blend at a nominal ``s`` puts far more than ``s`` of
    the loss on the injections, and the knob behaves as a switch rather than a dial.
    Rescaling the injection term onto the voltage term's running scale restores the meaning:
    the realized share must equal ``s``, whatever ``s`` is.
    """
    batch = inj_setup["batch"]
    for split in (0.1, 0.5):
        mod = _module(inj_setup, cigre_grid, split=split)
        mod._step(batch, train=True)
        assert mod._loss_share is not None
        assert float(mod._loss_share) == pytest.approx(split, rel=1e-5)


def test_the_raw_blend_overweights_the_injection_term(cigre_grid, inj_setup):
    """The pathology the normalization fixes, pinned so a regression is visible.

    Without rescaling, a nominal 10 % split hands the injection term the overwhelming
    majority of the loss purely because its residual is larger. The exact factor is
    grid- and stage-dependent; what must hold is that the realized share is far above the
    configured one, and that normalization removes the gap.
    """
    batch = inj_setup["batch"]
    raw = _module(inj_setup, cigre_grid, split=0.1)
    raw.cfg = _with_split(raw.cfg, 0.1, normalize=False)
    raw._step(batch, train=True)
    parts = raw._loss_components
    assert float(parts["injection_state"]) > 10.0 * float(parts["voltage"])
    assert float(raw._loss_share) > 0.5  # nominal 0.1 -> the injections dominate

    normalized = _module(inj_setup, cigre_grid, split=0.1)
    normalized._step(batch, train=True)
    assert float(normalized._loss_share) == pytest.approx(0.1, rel=1e-5)


def test_normalization_keeps_the_voltage_term_at_its_own_scale(cigre_grid, inj_setup):
    """The blend stays in the voltage term's units, so lr / grad-clip keep their meaning.

    At the moment the running scale is seeded, ``scale·injection`` equals the voltage term
    by construction, so the blended loss equals the voltage-only loss — a run that turns the
    split on does not silently change the size of its own objective (which a raw blend does,
    by orders of magnitude).
    """
    mod = _module(inj_setup, cigre_grid, split=0.5)
    batch = inj_setup["batch"]
    blended = mod._step(batch, train=True)[0].detach()
    assert torch.allclose(blended, mod._loss_components["voltage"], rtol=1e-5)


def test_the_share_holds_as_the_two_terms_drift_apart(cigre_grid, inj_setup):
    """The realized share stays at ``s`` however far the raw magnitudes diverge.

    The voltage term falls by orders of magnitude within a stage while the injection term
    barely moves, so any scale carried across batches lags and the share overshoots. Pinning
    the factor to the batch's own ratio holds the split exactly — checked as the model (and
    with it the loss magnitudes) is perturbed between steps.
    """
    mod = _module(inj_setup, cigre_grid, split=0.5)
    batch = inj_setup["batch"]
    for _ in range(4):
        mod._step(batch, train=True)
        assert float(mod._loss_share) == pytest.approx(0.5, rel=1e-5)
        with torch.no_grad():  # move the model, and with it the two terms' magnitudes
            for p in mod.model.parameters():
                p.mul_(1.5)


def test_cancelling_rows_are_the_colocated_load_and_generator_nodes(cigre_grid):
    """The diagnostic's row set is exactly the buses carrying BOTH a load and a generator.

    Those are the rows whose injection the voltages cannot resolve: only the SUM reaches the
    network, so +20 kW beside -20 kW and +2 kW beside -2 kW give the identical voltage
    profile. The set must be derived from the grid, not configured.
    """
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Generator, Load
    from injection_hse.train import _cancelling_injection_rows

    loads = {int(a.node) for a in cigre_grid.appliances if isinstance(a, Load)}
    gens = {int(a.node) for a in cigre_grid.appliances if isinstance(a, Generator)}
    shared = loads & gens
    assert shared, "the CIGRE LV fixture is expected to co-locate PV with load"

    index = node_phase_index(cigre_grid)
    expected = sorted({r for n in shared for r in index.rows(n)})
    assert _cancelling_injection_rows(cigre_grid).tolist() == expected
    # a grid with no generator has no ambiguity to report
    plain = cigre_grid.model_copy(
        update={
            "appliances": [
                a for a in cigre_grid.appliances if not isinstance(a, Generator)
            ]
        }
    )
    assert _cancelling_injection_rows(plain).numel() == 0


def test_validation_reports_injection_identifiability_per_order(cigre_grid, inj_setup):
    """Validation publishes the per-order injection error on the ambiguous rows vs the rest.

    This is the measurement that answers whether the estimator learns to tell a load from a
    generator by its spectral signature: the fundamental is expected to stay ambiguous while
    the harmonic orders become identifiable, which is only visible if both groups are
    reported per order.
    """
    from injection_hse.masking import build_masker

    mod = _module(inj_setup, cigre_grid, split=0.5)
    # the diagnostic is about ESTIMATED rows, so it needs a stage that leaves some unobserved
    mod.set_masker(build_masker("random", observed_fraction=0.5))
    mod._step(inj_setup["batch"], train=False)
    diag = mod._injection_diag
    assert diag, "the CIGRE fixture has cancelling rows, so the diagnostic must report"
    for order in mod.orders:
        for group in ("cancel", "other"):
            key = f"inj_rel_{group}_h{int(order)}"
            assert key in diag
            assert diag[key] >= 0.0

    # a fully observed stage has nothing estimated: report nothing rather than NaNs
    mod.set_masker(build_masker("full"))
    mod._step(inj_setup["batch"], train=False)
    assert mod._injection_diag == {}
    # a training step must NOT pay for the diagnostic
    mod._injection_diag = {}
    mod._step(inj_setup["batch"], train=True)
    assert mod._injection_diag == {}
