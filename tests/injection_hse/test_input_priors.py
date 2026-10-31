"""Physics-derived node input priors (``PhysicsConfig.input_priors``).

The estimator's message passing carries a few hops, while the voltage at an unobserved row
is a global function of the measurements, and ``I = Y·V`` multiplies neighbour voltages by
admittances that additive messages do not represent. The priors feed both quantities in as
per-row input channels. The invariants under test:

- the default (empty list) is byte-identical: same parameter count, identical loss;
- ``kcl`` at full observability IS ``Y·V_in`` per order (decoded back through the injection
  normalizer), with the observed-neighbourhood fraction at 1;
- ``linear_state`` at full observability with a large anchor weight reproduces the measured
  state at the observed rows, the model trains one optimizer step, and the checkpoint
  round-trip rebuilds the solver and infers;
- a multi-grid batch runs both priors per member through the operator bank;
- under a switchable topology the ``kcl`` prior uses each sample's own configuration.
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
    PhysicsConfig,
    TopologyConfig,
    TopologyVariantConfig,
    TrainConfig,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.train.layout import _unflatten_features

ORDERS = [1, 3, 5]


@pytest.fixture(scope="module")
def setup():
    """A small feeder, solved states, fitted normalizers and the shared operators."""
    from pgml.grids import synthetic_feeder
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import base_voltage_per_row, build_graph
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer

    grid = synthetic_feeder(6, n_feeders=2)
    graph = build_graph(grid)
    v = solve_harmonic_flow(grid, ORDERS, dtype=torch.complex128).v  # [H, N]
    states = torch.stack([v * (1 + 0.01 * k) for k in range(8)])  # [B, H, N]
    enc = TrigEncoding()
    normalizer = HarmonicNormalizer().fit(enc.encode(states).transpose(-3, -2))
    solver = NetworkSolver.from_grid(
        grid, ORDERS, dtype=torch.complex128, anchor_v_weight=1.0
    )
    inj_normalizer = fit_injection_normalizer(solver, states, enc)
    return {
        "grid": grid,
        "graph": graph,
        "states": states,
        "normalizer": normalizer,
        "solver": solver,
        "inj_normalizer": inj_normalizer,
        "base_v": base_voltage_per_row(grid),
    }


def _cfg(priors, *, decode="voltage", prior_wv=1.0, prior_wi=1.0) -> ExperimentConfig:
    return ExperimentConfig(
        name="input-priors",
        seed=0,
        orders=ORDERS,
        data=DataConfig(dataset_path="x"),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(
            decode=decode,
            input_priors=list(priors),
            prior_anchor_voltage_weight=prior_wv,
            prior_anchor_current_weight=prior_wi,
            anchor_voltage_weight=1.0 if decode == "injection" else 0.0,
        ),
        measurement=MeasurementConfig(slack_known=True),
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


def _module(setup, cfg):
    from injection_hse.masking import build_masker
    from injection_hse.train import SEModule

    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=int(setup["states"].shape[-1]),
        normalizer=setup["normalizer"],
        graph=setup["graph"],
        base_v=setup["base_v"],
        solver=setup["solver"],
        inj_normalizer=setup["inj_normalizer"],
        grid=setup["grid"],
    )
    module.set_masker(build_masker("random", observed_fraction=0.5))
    return module


# =============================================================================
# Default OFF = byte-identical
# =============================================================================
def test_the_default_is_byte_identical(setup):
    """A config that never names the fields equals one naming their defaults exactly."""
    legacy = _cfg([])
    dump = legacy.model_dump()
    for field in (
        "input_priors",
        "prior_anchor_voltage_weight",
        "prior_anchor_current_weight",
    ):
        dump["physics"].pop(field)
    pre_feature = ExperimentConfig.model_validate(dump)
    assert pre_feature == legacy

    a = _module(setup, pre_feature)
    b = _module(setup, legacy)
    assert a._n_prior_features == 0
    assert sum(p.numel() for p in a.model.parameters()) == sum(
        p.numel() for p in b.model.parameters()
    )
    batch = {"v": setup["states"]}
    loss_a = a._step(batch, train=True)[0]
    loss_b = b._step(batch, train=True)[0]
    assert torch.equal(loss_a.detach(), loss_b.detach())
    mask = torch.ones(8, a.n_nodes, dtype=torch.bool)
    assert a.input_priors(setup["states"], mask, None, None, None, None, None) is None


def test_a_duplicate_prior_is_rejected():
    with pytest.raises(ValueError, match="twice"):
        PhysicsConfig(input_priors=["kcl", "kcl"])


def test_linear_state_requires_the_measured_slack():
    dump = _cfg(["linear_state"]).model_dump()
    dump["measurement"]["slack_known"] = False
    with pytest.raises(ValueError, match="slack_known"):
        ExperimentConfig.model_validate(dump)


def test_a_prior_without_a_solver_is_refused(setup):
    """The priors are model INPUT width — a missing solver must fail at construction."""
    from injection_hse.train import SEModule

    cfg = _cfg(["kcl"])
    with pytest.raises(ValueError, match="needs the network solver"):
        SEModule(
            cfg,
            n_nodes=int(setup["states"].shape[-1]),
            normalizer=setup["normalizer"],
            graph=setup["graph"],
            base_v=setup["base_v"],
            grid=setup["grid"],
        )


# =============================================================================
# ``kcl``: the implied injection of the observed neighbourhood
# =============================================================================
def test_kcl_at_full_observability_is_the_true_injection(setup):
    """Fully observed: the decoded kcl channels equal ``Y·V_in`` and ``f_r = 1``."""
    module = _module(setup, _cfg(["kcl"]))
    v = setup["states"]
    b, n = int(v.shape[0]), module.n_nodes
    h, c_inj = module.n_harmonics, module.inj_encoding.n_features
    mask = torch.ones(b, n, dtype=torch.bool)
    block = module.input_priors(v, mask, None, None, None, None, None)  # [B, N, Fp]
    assert block.shape == (b, n, h * c_inj + 1)
    kappa_feats = _unflatten_features(block[..., : h * c_inj], h, c_inj)
    kappa = module.inj_encoding.decode(
        module.inj_normalizer.inverse_transform(kappa_feats.to(torch.float64))
    ).transpose(-2, -1)  # [B, H, N]
    expected = setup["solver"].injection_targets(v)
    assert torch.allclose(kappa, expected, rtol=1e-5, atol=1e-8)
    assert torch.allclose(block[..., -1], torch.ones(b, n, dtype=block.dtype))


def test_kcl_under_a_partial_mask_reports_the_coverage(setup):
    """A partially observed neighbourhood: ``f_r`` in [0, 1], exactly 1 only where full."""
    module = _module(setup, _cfg(["kcl"]))
    v = setup["states"][:2]
    n = module.n_nodes
    mask = torch.zeros(2, n, dtype=torch.bool)
    mask[:, ::2] = True
    block = module.input_priors(v, mask, None, None, None, None, None)
    fraction = block[..., -1]
    assert float(fraction.min()) >= 0.0 and float(fraction.max()) <= 1.0
    coupling = setup["solver"].coupling_mask()
    expected = mask.to(torch.float64) @ coupling.to(torch.float64).T
    expected = expected / coupling.sum(dim=-1).clamp_min(1).to(torch.float64)
    assert torch.allclose(fraction.to(torch.float64), expected)


# =============================================================================
# ``linear_state``: the anchored zero-injection solve
# =============================================================================
def test_linear_state_with_large_anchors_matches_the_observed_rows(setup):
    """A large voltage anchor pins the prior state to the measurements where observed."""
    module = _module(setup, _cfg(["linear_state"], prior_wv=1e8))
    v = setup["states"][:4]
    n = module.n_nodes
    mask = torch.zeros(4, n, dtype=torch.bool)
    mask[:, ::2] = True
    mask[:, module._slack_rows] = True
    block = module.input_priors(v, mask, None, None, None, None, None)  # [B, N, H·C]
    h, c = module.n_harmonics, module.n_channels
    feats = _unflatten_features(block, h, c)
    v_prior = module.encoding.decode(
        module.normalizer.inverse_transform(feats.to(torch.float64))
    ).transpose(-2, -1)  # [B, H, N]
    observed = mask.unsqueeze(1).expand_as(v_prior)
    residual = (v_prior - v).abs()[observed]
    scale = v.abs()[observed].mean()
    assert float(residual.max()) < 1e-3 * float(scale)


def test_the_prior_solve_restores_the_decode_weights(setup):
    """The prior installs its weights for ITS solve and restores the stage's afterwards."""
    module = _module(setup, _cfg(["linear_state"], decode="injection", prior_wv=123.0))
    solver = module.solver
    before = (solver.anchor_v_weight, solver.anchor_i_weight)
    v = setup["states"][:2]
    mask = torch.ones(2, module.n_nodes, dtype=torch.bool)
    module.input_priors(v, mask, None, None, None, None, None)
    assert (solver.anchor_v_weight, solver.anchor_i_weight) == before


def test_one_optimizer_step_and_the_checkpoint_round_trip(setup, tmp_path):
    """Both priors train one step; ``from_checkpoint`` rebuilds the solver and infers."""
    from injection_hse.train import SEModule

    module = _module(setup, _cfg(["kcl", "linear_state"]))
    batch = {"v": setup["states"]}
    opt = torch.optim.Adam(module.model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in module.model.parameters()]
    loss = module._step(batch, train=True)[0]
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert any(
        not torch.equal(a, p.detach())
        for a, p in zip(before, module.model.parameters())
    )

    path = tmp_path / "stage_m.ckpt"
    ckpt = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(ckpt)
    torch.save(ckpt, path)
    reloaded = SEModule.from_checkpoint(path, grid=setup["grid"])
    assert reloaded.cfg.physics.input_priors == ["kcl", "linear_state"]
    assert (
        reloaded.solver is not None
    )  # rebuilt for the priors despite the voltage decode
    assert reloaded._n_prior_features == module._n_prior_features
    assert all(
        torch.equal(a, b)
        for a, b in zip(module.state_dict().values(), reloaded.state_dict().values())
    )
    module.eval()
    out = reloaded.infer(batch)
    assert out["pred_v"].shape == setup["states"].shape
    # same masker + same mask stream -> the same estimate: the prior block is rebuilt
    # identically from the checkpoint's config and normalizers
    from injection_hse.masking import build_masker

    reloaded.set_masker(build_masker("random", observed_fraction=0.5))
    module._mask_gen.manual_seed(7)
    reloaded._mask_gen.manual_seed(7)
    assert torch.allclose(
        module.infer(batch)["pred_v"], reloaded.infer(batch)["pred_v"]
    )


# =============================================================================
# Many grids: both priors through the operator bank (MULTIGRID_CONSUMED)
# =============================================================================
MG_ORDERS = (1, 3, 5)


@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    from injection_hse.data.multigrid import (
        MultiGridGenerationConfig,
        SyntheticFeederSource,
        generate_multigrid_dataset,
    )

    root = tmp_path_factory.mktemp("multigrid_input_priors")
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


def test_the_multigrid_step_runs_both_priors_through_the_bank(multigrid_root):
    from injection_hse.data import MultiGridCollate, MultiGridDataSource
    from injection_hse.masking import build_masker
    from injection_hse.physics import MultiGridPhysicsBank, fit_multigrid_injection_normalizer
    from injection_hse.train import SEModule, fit_multigrid_normalizer

    cfg = ExperimentConfig(
        name="multigrid-input-priors",
        seed=0,
        orders=list(MG_ORDERS),
        data=DataConfig(
            dataset_path=str(multigrid_root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(decode="voltage", input_priors=["kcl", "linear_state"]),
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
        train=TrainConfig(accelerator="cpu", devices=1, dtype="complex128"),
    )
    source = MultiGridDataSource(multigrid_root, placement_fraction=0.3)
    bank = MultiGridPhysicsBank(source, dtype=torch.complex128, anchor_v_weight=1.0)
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
    dataset = source.train_dataset()
    batch = MultiGridCollate(source)([dataset[i] for i in range(2)])
    assert module._n_prior_features > 0
    loss = module._step(batch, train=True)[0]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0
    # the block itself: per member, in the run's frame — finite and correctly shaped
    out = module.infer(batch)
    assert out["pred_v"].shape[-1] == int(batch["ptr"][-1])


# =============================================================================
# Switchable topology: each sample's own configuration
# =============================================================================
def test_the_kcl_prior_uses_each_samples_own_configuration():
    """Per-variant dispatch: the mixed batch's kcl block equals the per-variant one."""
    from pgml.grids import synthetic_feeder
    from pgml.solver import solve_harmonic_flow

    from injection_hse.data import (
        TopologyBank,
        TopologySet,
        base_voltage_per_row,
        build_graph,
        enumerate_variants,
        switch_branch_ids,
    )
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import fit_injection_normalizer
    from injection_hse.train import SEModule

    grid = synthetic_feeder(13, n_feeders=3, tie_switches=2)
    ids = switch_branch_ids(grid)
    topo_cfg = TopologyConfig(
        switch_ids=ids,
        variants=[
            TopologyVariantConfig(name=name, open_switches=open_ids)
            for name, open_ids in enumerate_variants(grid, ids)
        ],
    )
    topology = TopologySet.from_config(topo_cfg)
    graph = build_graph(topology.union_grid(grid))
    bank = TopologyBank.build(
        topology,
        grid,
        ORDERS,
        edge_branch_ids=graph.edge_branch_ids,
        dtype=torch.complex128,
        decode="voltage",
        build_solvers=True,
    )
    states = torch.stack(
        [
            solve_harmonic_flow(
                topology.variant_grid(grid, k), ORDERS, dtype=torch.complex128
            ).v
            for k in range(topology.n_variants)
        ]
    )  # [K, H, N]
    encoding = TrigEncoding()
    normalizer = HarmonicNormalizer().fit(encoding.encode(states).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="topology-priors",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="unused", topology=topo_cfg),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(decode="voltage", input_priors=["kcl"]),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="full", kind="full", target_loss=1.0, max_epochs=1)
            ]
        ),
        train=TrainConfig(dtype="complex128"),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=int(states.shape[-1]),
        normalizer=normalizer,
        graph=graph,
        base_v=base_voltage_per_row(grid),
        inj_normalizer=fit_injection_normalizer(bank.solvers[0], states, encoding),
        grid=grid,
        topology=bank,
    )

    # perturb each sample off its solved equilibrium: at the solved state a closed tie
    # between balanced feeders carries (nearly) no current, so Y_cc·V and Y_oo·V would
    # coincide and a wrong dispatch could pass unnoticed. Off equilibrium the switch rows'
    # implied injections differ by g_switch·ΔV — variants become clearly distinguishable.
    order = torch.tensor([2, 0, 3, 1, 0])
    generator = torch.Generator().manual_seed(0)
    perturbation = 1.0 + 0.05 * torch.randn(
        (len(order), *states.shape[1:]), generator=generator, dtype=torch.float64
    )
    v = states.index_select(0, order) * perturbation
    mask = torch.ones(len(order), module.n_nodes, dtype=torch.bool)
    mixed = module.input_priors(v, mask, None, None, None, None, order)
    h, c_inj = module.n_harmonics, module.inj_encoding.n_features
    kappa = module.inj_encoding.decode(
        module.inj_normalizer.inverse_transform(
            _unflatten_features(mixed[..., : h * c_inj], h, c_inj).to(torch.float64)
        )
    ).transpose(-2, -1)  # [B, H, N]
    for i, k in enumerate(order.tolist()):
        expected = bank.solvers[k].injection_targets(v[i : i + 1])[0]
        assert torch.allclose(kappa[i], expected, rtol=1e-4, atol=1e-6), (
            f"sample {i} (variant {k}) did not use its own configuration's Y(h)"
        )
        # a foreign configuration's operator implies a DIFFERENT injection off equilibrium
        other = bank.solvers[(k + 1) % topology.n_variants].injection_targets(
            v[i : i + 1]
        )[0]
        assert not torch.allclose(kappa[i], other, rtol=1e-4, atol=1e-6)
    # the same off-equilibrium state under two configurations yields two prior blocks
    a = module.input_priors(v[:1], mask[:1], None, None, None, None, torch.tensor([2]))
    b = module.input_priors(
        v[:1], mask[:1], None, None, None, None, torch.tensor([topology.n_variants - 1])
    )
    assert not torch.allclose(a, b)
