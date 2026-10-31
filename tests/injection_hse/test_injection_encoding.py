"""A parameterisation of its own for the injection head (``EncodingConfig.injection_kind``).

The injection decode's head predicts the nodal injection currents ``I(h) = Y(h)·V(h)``, a
map that is LINEAR in Cartesian coordinates and strongly nonlinear in the polar
``(|I|, cosθ, sinθ)`` encoding the voltage features use (whose ``log1p``-standardised
magnitude must additionally hit an exact zero on a zero-injection row).
:attr:`injection_hse.config.EncodingConfig.injection_kind` lets the head emit per-order standardised
``(re, im)`` currents while the voltage input keeps the trig encoding — which makes the
model's OUTPUT width differ from its input width.

These tests pin what that must satisfy: ``None`` and an explicit ``"trig"`` build the
identical module (the default stays byte-identical); a Cartesian head trains, round-trips
through a checkpoint with the right output width, and infers; every spatial model (and the
temporal wrapper around one) emits the output width it was built for; the multi-grid path
honours the field through the operator bank.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    EncodingConfig,
    ExperimentConfig,
    GNNConfig,
    GraphormerConfig,
    MaskStageConfig,
    MultiGridConfig,
    PhysicsConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.encoding import CartesianEncoding, TrigEncoding, build_injection_encoding
from injection_hse.io import build_model_input
from injection_hse.models import build_model


# =============================================================================
# Single grid: the CIGRE LV fixtures
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


def _config(setup, *, injection_kind, kind="trig", model=None, anchor_v=0.0):
    return ExperimentConfig(
        name="injection-encoding",
        seed=0,
        orders=list(setup["source"].orders),
        data=DataConfig(dataset_path="x"),
        model=model if model is not None else DNNConfig(hidden_dims=[32]),
        encoding=EncodingConfig(kind=kind, injection_kind=injection_kind),
        physics=PhysicsConfig(decode="injection", anchor_voltage_weight=anchor_v),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="m", kind="full", target_loss=1e-12, max_epochs=1)
            ]
        ),
    )


def _module(setup, grid, cfg):
    """An injection-decode :class:`~injection_hse.train.SEModule` under ``cfg`` (seeded)."""
    from injection_hse.data import base_voltage_per_row
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.train import SEModule

    source = setup["source"]
    solver = NetworkSolver.from_grid(
        grid, list(source.orders), anchor_v_weight=cfg.physics.anchor_voltage_weight
    )
    inj_norm = fit_injection_normalizer(
        solver, setup["train_v"], build_injection_encoding(cfg.encoding)
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
        grid=grid,
    )


def test_none_and_trig_build_the_identical_module(cigre_grid, setup):
    """``injection_kind=None`` IS ``kind``: same parameters, same loss, bit for bit."""
    a = _module(setup, cigre_grid, _config(setup, injection_kind=None))
    b = _module(setup, cigre_grid, _config(setup, injection_kind="trig"))
    assert isinstance(a.inj_encoding, TrigEncoding)
    assert isinstance(b.inj_encoding, TrigEncoding)
    assert a.n_out_channels == b.n_out_channels == a.n_channels == 3
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    batch = setup["batch"]
    la = a._step(batch, train=True)[0].detach()
    lb = b._step(batch, train=True)[0].detach()
    assert torch.isfinite(la)
    assert torch.equal(la, lb)


def test_cartesian_head_trains_and_round_trips_through_a_checkpoint(
    cigre_grid, setup, tmp_path
):
    """A Cartesian injection head: 2 output channels per order, one optimizer step, reload.

    The model's input keeps the 3-channel trig encoding; its output carries ``H × 2``
    channels. After one optimizer step the module is written as a checkpoint and rebuilt by
    :meth:`~injection_hse.train.SEModule.from_checkpoint` from the config it carries — the rebuilt
    head has the same width, the same weights and infers the same state.
    """
    from injection_hse.train import SEModule

    cfg = _config(setup, injection_kind="cartesian")
    mod = _module(setup, cigre_grid, cfg)
    assert isinstance(mod.encoding, TrigEncoding)
    assert isinstance(mod.inj_encoding, CartesianEncoding)
    assert mod.n_channels == 3 and mod.n_out_channels == 2
    assert mod.inj_normalizer.mean.shape == (mod.n_harmonics, 2)
    assert mod.model.mlp[-1].out_features == mod.n_nodes * mod.n_harmonics * 2

    batch = setup["batch"]
    opt = torch.optim.Adam(mod.model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in mod.model.parameters()]
    loss = mod._step(batch, train=True)[0]
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in mod.model.parameters()
        if p.grad is not None
    )
    opt.step()
    assert any(
        not torch.equal(a, p.detach()) for a, p in zip(before, mod.model.parameters())
    )

    path = tmp_path / "stage_m.ckpt"
    ckpt = {"state_dict": mod.state_dict()}
    mod.on_save_checkpoint(ckpt)
    torch.save(ckpt, path)
    reloaded = SEModule.from_checkpoint(path, grid=cigre_grid)
    assert reloaded.cfg.encoding.injection_kind == "cartesian"
    assert isinstance(reloaded.inj_encoding, CartesianEncoding)
    assert reloaded.n_out_channels == 2
    assert all(
        torch.equal(a, b)
        for a, b in zip(mod.state_dict().values(), reloaded.state_dict().values())
    )

    mod.eval()
    h, n = mod.n_harmonics, mod.n_nodes
    out_a = mod.infer(batch)
    out_b = reloaded.infer(batch)
    assert out_a["pred_v"].shape == (batch["v"].shape[0], h, n)
    assert out_a["pred_v"].is_complex()
    assert torch.allclose(out_a["pred_v"], out_b["pred_v"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "model_cfg",
    [
        DNNConfig(hidden_dims=[16]),
        GNNConfig(hidden_dim=16, num_layers=2, conv="sage"),
        GNNConfig(hidden_dim=16, num_layers=2, conv="gine", input_skip=True),
        GraphormerConfig(hidden_dim=16, num_layers=1, num_heads=2),
        TemporalConfig(
            spatial=GNNConfig(hidden_dim=16, num_layers=1, conv="sage"),
            hidden_dim=8,
            num_layers=1,
        ),
    ],
    ids=["dnn", "gnn-sage", "gnn-gine-skip", "graphormer", "temporal-gnn"],
)
def test_every_model_emits_the_output_width_it_was_built_for(setup, model_cfg):
    """``n_out_features`` widens (or narrows) only the final projection.

    The node observations carry ``H × 3`` trig channels, the output ``H × 2`` Cartesian
    ones; the GNN's ``input_skip`` head, the Graphormer / DNN heads and the temporal wrapper
    (whose recurrence lives on the INPUT width) all honour it.
    """
    source = setup["source"]
    graph = source.graph()
    n = source.n_nodes
    h = len(source.orders)
    model = build_model(
        model_cfg,
        n_features=h * 3,
        n_nodes=n,
        n_static_features=int(graph.x.shape[1]),
        n_out_features=h * 2,
    )
    assert model.n_out_features == h * 2
    steps = 3 if isinstance(model_cfg, TemporalConfig) else 1
    node_obs = torch.randn(2, steps, n, h * 3)
    mask = torch.rand(2, steps, n) > 0.5
    mi = build_model_input(graph, node_obs, mask, has_time=True)
    out = model(mi)
    assert out.shape == (2, steps, n, h * 2)
    assert torch.isfinite(out).all()

    # the default (None) is the in = out contract, byte-identical to before
    same = build_model(
        model_cfg,
        n_features=h * 3,
        n_nodes=n,
        n_static_features=int(graph.x.shape[1]),
    )
    assert same.n_out_features == h * 3
    assert same(mi).shape == (2, steps, n, h * 3)


def test_injection_kind_is_a_behaviour_field_of_the_stage_manager(setup):
    """A pass may not silently switch the injection head's width on a built module."""
    from injection_hse.train import assert_cfg_swappable

    a = _config(setup, injection_kind=None)
    b = _config(setup, injection_kind="cartesian")
    with pytest.raises(ValueError) as excinfo:
        assert_cfg_swappable(a, b)
    assert "encoding.injection_kind" in str(excinfo.value)


# =============================================================================
# Many grids: the field reaches the operator-bank path
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

    root = tmp_path_factory.mktemp("multigrid_injection_encoding")
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


def _multigrid_config(root, *, injection_kind):
    return ExperimentConfig(
        name="multigrid-injection-encoding",
        seed=0,
        orders=list(MG_ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        encoding=EncodingConfig(kind="trig", injection_kind=injection_kind),
        physics=PhysicsConfig(decode="injection"),
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


def test_the_multigrid_path_honours_the_injection_encoding(multigrid_root):
    """One multi-grid training step with a Cartesian injection head, through the bank.

    The pooled injection normalizer is fit in the injection encoding (2 channels), the model
    emits ``H × 2`` channels over the concatenated node axis, and the loss back-propagates.
    """
    from injection_hse.data import MultiGridCollate, MultiGridDataSource
    from injection_hse.masking import build_masker
    from injection_hse.physics import MultiGridPhysicsBank, fit_multigrid_injection_normalizer
    from injection_hse.train import SEModule, fit_multigrid_normalizer

    cfg = _multigrid_config(multigrid_root, injection_kind="cartesian")
    source = MultiGridDataSource(multigrid_root, placement_fraction=0.3)
    bank = MultiGridPhysicsBank(source, dtype=torch.complex128)
    inj_encoding = build_injection_encoding(cfg.encoding)
    assert isinstance(inj_encoding, CartesianEncoding)
    inj_norm = fit_multigrid_injection_normalizer(bank, source, inj_encoding, cfg)
    assert inj_norm.mean.shape == (len(MG_ORDERS), 2)

    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, TrigEncoding(), cfg),
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        inj_normalizer=inj_norm,
    )
    assert module.n_channels == 3 and module.n_out_channels == 2
    module.set_masker(build_masker("random", observed_fraction=0.5))

    dataset = source.train_dataset()
    batch = MultiGridCollate(source)([dataset[i] for i in range(2)])
    loss, mask, pred_v, target_v, _, _, _ = module._step(batch, train=True)
    assert torch.isfinite(loss)
    assert pred_v.shape == target_v.shape == (len(MG_ORDERS), int(batch["ptr"][-1]))
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0
