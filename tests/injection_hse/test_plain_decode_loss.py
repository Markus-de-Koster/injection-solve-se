"""The plain-decode prior term (``LossConfig.plain_decode_weight``).

With measurement anchoring on, the voltage loss is evaluated on the anchored solve, which is
flat in the predicted injections wherever the measurements pin the state: the network gets
no gradient there while the run reports the solve's accuracy. The term adds the SAME voltage
loss on the PLAIN (slack-only) decode of the same predicted injections, so the network is
supervised as a prior whatever the anchored output does.

These tests pin what the term must satisfy: weight ``0`` performs no extra solve and is the
previous loss exactly; without anchoring the plain decode IS the decode and nothing is solved
twice; a positive weight under anchoring adds a finite, non-zero component that changes the
loss and back-propagates to the model; validation reports the plain decode's own unobserved
per-unit MAE beside the anchored one; and the multi-grid path runs the term through its
operator bank.
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
    MultiGridConfig,
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.encoding import TrigEncoding


def _with_weight(cfg, weight: float):
    """The same experiment config with a different plain-decode weight."""
    return cfg.model_copy(
        update={"loss": cfg.loss.model_copy(update={"plain_decode_weight": weight})}
    )


def _count_plain_solves(solver) -> list[int]:
    """Wrap ``solver._solve_plain`` so every slack-only solve is counted."""
    counter = [0]
    original = solver._solve_plain

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    solver._solve_plain = counted
    return counter


# =============================================================================
# Single grid
# =============================================================================
@pytest.fixture(scope="module")
def setup(cigre_grid, se_dataset):
    """Data source + fitted voltage normalizer + one collated batch (shared)."""
    from injection_hse.data import StateCollate, build_data_source
    from injection_hse.normalization import HarmonicNormalizer

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    enc = TrigEncoding()
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    norm = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    return {"source": source, "train_v": train_v, "normalizer": norm, "batch": batch}


def _module(setup, grid, *, weight: float, anchor_v: float, split: float = 0.0):
    """An injection-decode module: anchored at ``anchor_v``, plain-decode weight ``weight``.

    Built under a partially observed stage (the plain metrics under test are the
    unobserved-row ones, vacuous on a fully observed stage).
    """
    from injection_hse.data import base_voltage_per_row
    from injection_hse.masking import build_masker
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    source = setup["source"]
    orders = list(source.orders)
    solver = NetworkSolver.from_grid(grid, orders, anchor_v_weight=anchor_v)
    inj_norm = fit_injection_normalizer(solver, setup["train_v"], TrigEncoding())
    cfg = ExperimentConfig(
        name="plain-decode",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode="injection", anchor_voltage_weight=anchor_v),
        loss=LossConfig(plain_decode_weight=weight, injection_state_loss_split=split),
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
    module.set_masker(build_masker("random", observed_fraction=0.5))
    return module


def test_zero_weight_performs_no_extra_solve(cigre_grid, setup):
    """Weight ``0`` is the previous behaviour: one anchored solve, no plain one, no term."""
    mod = _module(setup, cigre_grid, weight=0.0, anchor_v=1.0)
    assert mod.solver.anchoring
    plain_calls = _count_plain_solves(mod.solver)
    loss = mod._step(setup["batch"], train=True)[0]
    assert torch.isfinite(loss)
    assert plain_calls[0] == 0
    assert set(mod._loss_components) == {"voltage"}
    assert torch.equal(mod._loss_components["voltage"], loss.detach())
    assert mod._plain_pred_v is None


def test_without_anchoring_nothing_is_solved_twice(cigre_grid, setup):
    """Anchors off: the plain decode IS the decode — one solve, the term is the voltage term."""
    mod = _module(setup, cigre_grid, weight=0.5, anchor_v=0.0)
    assert not mod.solver.anchoring
    plain_calls = _count_plain_solves(mod.solver)
    loss = mod._step(setup["batch"], train=True)[0]
    assert plain_calls[0] == 1
    assert set(mod._loss_components) == {"voltage"}
    assert torch.equal(mod._loss_components["voltage"], loss.detach())


def test_the_term_changes_the_loss_and_reaches_the_model(cigre_grid, setup):
    """Anchors on: ONE extra plain solve, ``loss = voltage + w·plain``, non-zero gradient.

    The plain term's gradient is isolated by subtracting the weight-0 gradient of the same
    module on the same batch; it must be finite and non-zero — the term trains the network,
    it is not a passive readout.
    """
    mod = _module(setup, cigre_grid, weight=0.0, anchor_v=1.0)
    batch = setup["batch"]
    params = [p for p in mod.model.parameters() if p.requires_grad]
    # identical masks for the two evaluations: the mask stream advances per step
    mod._mask_gen.manual_seed(1)
    base = mod._step(batch, train=True)[0]
    g_base = torch.autograd.grad(base, params)

    mod.cfg = _with_weight(mod.cfg, 0.5)
    plain_calls = _count_plain_solves(mod.solver)
    mod._mask_gen.manual_seed(1)
    loss = mod._step(batch, train=True)[0]
    assert plain_calls[0] == 1
    parts = mod._loss_components
    assert set(parts) == {"voltage", "plain_decode"}
    for name, value in parts.items():
        assert torch.isfinite(value), name
        assert float(value) > 0.0, name
    assert torch.equal(parts["voltage"], base.detach())
    assert torch.allclose(loss, parts["voltage"] + 0.5 * parts["plain_decode"])
    assert not torch.allclose(loss, base)

    g_loss = torch.autograd.grad(loss, params)
    g_plain = [2.0 * (b - a) for a, b in zip(g_base, g_loss)]
    assert all(torch.isfinite(g).all() for g in g_plain)
    assert sum(float(g.abs().sum()) for g in g_plain) > 0.0


def test_the_term_stacks_on_the_injection_blend(cigre_grid, setup):
    """``loss = blended + w·plain`` — the split's share is measured on the blend alone."""
    mod = _module(setup, cigre_grid, weight=0.5, anchor_v=1.0, split=0.3)
    loss = mod._step(setup["batch"], train=True)[0]
    parts = mod._loss_components
    assert set(parts) == {"voltage", "injection_state", "plain_decode"}
    assert torch.isfinite(loss)
    assert mod._loss_share is not None
    assert 0.0 < float(mod._loss_share) < 1.0
    # the blend is normalized onto the voltage term's scale, so it equals the voltage term
    # in value; the plain term is added on top
    assert torch.allclose(
        loss, parts["voltage"] + 0.5 * parts["plain_decode"], rtol=1e-5, atol=1e-7
    )


def test_observed_row_weight_reweights_only_the_plain_term(cigre_grid, setup):
    """``plain_decode_observed_weight``: unobserved rows weigh 1, observed rows ``λ``.

    ``λ = 1`` is the plain term over every row (the default row selection under
    ``supervise_rows="all"``); another ``λ`` changes the plain component alone, which
    equals the row-weighted mean of the same per-row losses.
    """
    mod = _module(setup, cigre_grid, weight=0.5, anchor_v=1.0)
    batch = setup["batch"]

    def parts_at(lam):
        mod.cfg = mod.cfg.model_copy(
            update={
                "loss": mod.cfg.loss.model_copy(
                    update={"plain_decode_observed_weight": lam}
                )
            }
        )
        mod._mask_gen.manual_seed(1)
        mod._step(batch, train=True)
        return dict(mod._loss_components), mod._plain_rows

    default, rows_default = parts_at(None)
    one, rows_one = parts_at(1.0)
    four, rows_four = parts_at(4.0)
    assert rows_default is None
    assert torch.equal(rows_one, torch.ones_like(rows_one))
    assert torch.allclose(one["plain_decode"], default["plain_decode"], rtol=1e-6)
    assert torch.equal(four["voltage"], default["voltage"])
    observed = rows_four == 4.0
    assert bool(observed.any()) and bool((~observed).any())
    assert torch.equal(rows_four[~observed], torch.ones_like(rows_four[~observed]))
    assert not torch.allclose(four["plain_decode"], one["plain_decode"])


def test_row_weights_are_a_weighted_mean():
    """The loss's masked mean with float row weights is ``Σ w·l / Σ w``."""
    from injection_hse.loss import StateEstimationLoss

    per_node = torch.tensor([[1.0, 2.0, 3.0]])
    weights = torch.tensor([[1.0, 4.0, 1.0]])
    got = StateEstimationLoss._reduce(per_node, weights)
    assert torch.allclose(got, torch.tensor((1.0 + 8.0 + 3.0) / 6.0))
    assert torch.allclose(
        StateEstimationLoss._reduce(per_node, weights.bool()), per_node.mean()
    )


def _logged(module, batch) -> dict[str, float]:
    """``{metric: value}`` of one validation step, captured from the module's ``log`` calls."""
    captured: dict[str, float] = {}

    def _capture(name, value, **_kwargs):
        captured[name] = float(value.detach() if torch.is_tensor(value) else value)

    module.log = _capture  # type: ignore[method-assign]
    module.validation_step(batch, 0)
    return captured


def test_validation_reports_the_plain_decode_quality(cigre_grid, setup):
    """``val_pu_mae_unobs_plain`` (+ per order) and ``val_plain_decode_loss`` are published.

    The plain decode is a different estimate from the anchored one, so its unobserved
    per-unit MAE differs from ``val_pu_mae_unobs``; without the term neither column exists.
    """
    mod = _module(setup, cigre_grid, weight=0.5, anchor_v=1.0)
    batch = setup["batch"]
    logged = _logged(mod, batch)
    assert logged["val_plain_decode_loss"] > 0.0
    assert logged["val_pu_mae_unobs_plain"] > 0.0
    assert logged["val_pu_mae_unobs_plain"] != pytest.approx(logged["val_pu_mae_unobs"])
    for order in mod.orders:
        assert logged[f"val_pu_mae_unobs_plain_h{order}"] > 0.0
    assert logged["val_pu_mae_unobs_plain"] == pytest.approx(
        sum(logged[f"val_pu_mae_unobs_plain_h{o}"] for o in mod.orders)
        / len(mod.orders),
        rel=1e-5,
    )

    off = _module(setup, cigre_grid, weight=0.0, anchor_v=1.0)
    logged_off = _logged(off, batch)
    assert "val_pu_mae_unobs_plain" not in logged_off
    assert "val_plain_decode_loss" not in logged_off


def test_the_history_collector_records_the_plain_columns():
    """The training-history parquet carries the plain term and metric beside the others."""
    from injection_hse.evaluation.history import _DIAGNOSTICS, _ORDER_PREFIXES

    assert "train_plain_decode_loss" in _DIAGNOSTICS
    assert "val_plain_decode_loss" in _DIAGNOSTICS
    assert "val_pu_mae_unobs_plain" in _DIAGNOSTICS
    assert "val_pu_mae_unobs_plain_h" in _ORDER_PREFIXES


def test_the_weight_without_the_injection_decode_is_ignored(setup):
    """The term needs a predicted injection: the voltage decode warns and drops it."""
    from injection_hse.train import SEModule

    source = setup["source"]
    cfg = ExperimentConfig(
        name="voltage-decode",
        seed=0,
        orders=list(source.orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        physics=PhysicsConfig(decode="voltage"),
        loss=LossConfig(plain_decode_weight=0.5),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )
    with pytest.warns(UserWarning, match="plain_decode_weight"):
        mod = SEModule(
            cfg,
            n_nodes=source.n_nodes,
            normalizer=setup["normalizer"],
            graph=source.graph(),
        )
    loss = mod._step(setup["batch"], train=True)[0]
    assert torch.isfinite(loss)
    assert set(mod._loss_components) == {"voltage"}


# =============================================================================
# Many grids: the term runs through the operator bank
# =============================================================================
MG_ORDERS = (1, 3, 5)


@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    """A tiny multi-grid dataset with harmonic injection (three synthetic feeders)."""
    from injection_hse.data.multigrid import (
        MultiGridGenerationConfig,
        SyntheticFeederSource,
        generate_multigrid_dataset,
    )

    root = tmp_path_factory.mktemp("multigrid_plain_decode")
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(root),
            n_scenarios=3,
            orders=MG_ORDERS,
            val_fraction=0.3,
            sources=[SyntheticFeederSource(n_grids=3, node_range=(4, 6), n_feeders=2)],
        )
    )
    return root


def _multigrid_module(root, *, weight: float, anchor_v: float):
    from injection_hse.data import MultiGridDataSource
    from injection_hse.masking import build_masker
    from injection_hse.physics import MultiGridPhysicsBank, fit_multigrid_injection_normalizer
    from injection_hse.train import SEModule, fit_multigrid_normalizer

    cfg = ExperimentConfig(
        name="multigrid-plain-decode",
        seed=0,
        orders=list(MG_ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(decode="injection", anchor_voltage_weight=anchor_v),
        loss=LossConfig(plain_decode_weight=weight),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        train=TrainConfig(
            accelerator="cpu", devices=1, dtype="complex128", enable_progress_bar=False
        ),
    )
    source = MultiGridDataSource(root, placement_fraction=0.3)
    bank = MultiGridPhysicsBank(
        source, dtype=torch.complex128, anchor_v_weight=anchor_v
    )
    encoding = TrigEncoding()
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, encoding, cfg),
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        inj_normalizer=fit_multigrid_injection_normalizer(bank, source, encoding, cfg),
    )
    module.set_masker(build_masker("random", observed_fraction=0.5))
    return module, source, bank


def test_the_multigrid_path_runs_the_term_through_the_bank(multigrid_root):
    """A multi-grid batch under anchoring: the plain term is added per member and trains."""
    from injection_hse.data import MultiGridCollate

    module, source, bank = _multigrid_module(multigrid_root, weight=0.5, anchor_v=1.0)
    dataset = source.train_dataset()
    batch = MultiGridCollate(source)([dataset[i] for i in range(2)])
    counters = [
        _count_plain_solves(bank.solver(int(g)))
        for g in sorted(set(batch["grid_index"].tolist()))
    ]
    loss = module._step(batch, train=True)[0]
    assert torch.isfinite(loss)
    assert set(module._loss_components) == {"voltage", "plain_decode"}
    assert all(c[0] == 1 for c in counters)  # one plain solve per distinct member
    parts = module._loss_components
    assert torch.allclose(loss, parts["voltage"] + 0.5 * parts["plain_decode"])
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0

    logged = _logged(module, batch)
    assert logged["val_pu_mae_unobs_plain"] > 0.0
    assert logged["val_plain_decode_loss"] > 0.0
    for order in MG_ORDERS:
        assert f"val_pu_mae_unobs_plain_h{order}" in logged


def test_the_multigrid_zero_weight_is_untouched(multigrid_root):
    """Weight ``0`` on the multi-grid path: the anchored solve alone, no plain component."""
    from injection_hse.data import MultiGridCollate

    module, source, bank = _multigrid_module(multigrid_root, weight=0.0, anchor_v=1.0)
    dataset = source.train_dataset()
    batch = MultiGridCollate(source)([dataset[i] for i in range(2)])
    counters = [
        _count_plain_solves(bank.solver(int(g)))
        for g in sorted(set(batch["grid_index"].tolist()))
    ]
    loss = module._step(batch, train=True)[0]
    assert torch.isfinite(loss)
    assert set(module._loss_components) == {"voltage"}
    assert all(c[0] == 0 for c in counters)
