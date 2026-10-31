"""Measured branch currents as model input: edge pipeline, masking, checkpoint round-trip."""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
)
from injection_hse.masking import PlacementMasker, RandomMasker

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _cfg(**measurement) -> ExperimentConfig:
    return ExperimentConfig(
        name="edge-current",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="unused"),
        model=GNNConfig(hidden_dim=32, num_layers=2, conv="gine"),
        measurement=MeasurementConfig(
            voltage_nodes=["Bus R4", "Bus C5"], **measurement
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="full", kind="full", target_loss=1.0, max_epochs=1)
            ]
        ),
    )


@pytest.fixture(scope="module")
def edge_setup(cigre_grid, se_dataset):
    """Shared source + fitted normalizers + operator for the current-input tests."""
    from injection_hse.data import ParquetDataset
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer

    source = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2)
    train_ds = source.train_dataset()
    train_v = torch.stack([train_ds[i]["v"] for i in range(32)])  # [Bt, H, N]
    encoding = build_encoding("trig")
    normalizer = HarmonicNormalizer().fit(encoding.encode(train_v).transpose(-3, -2))

    edge_op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=torch.complex64)
    i_train = edge_op(train_v.to(torch.complex64))
    edge_normalizer = HarmonicNormalizer().fit(
        encoding.encode(i_train).transpose(-3, -2)
    )
    v_src = train_v.to(torch.complex64).index_select(-1, edge_op.edge_src_rows)
    power_normalizer = HarmonicNormalizer().fit(
        encoding.encode(v_src * i_train.conj()).transpose(-3, -2)
    )
    return {
        "source": source,
        "train_v": train_v,
        "normalizer": normalizer,
        "edge_op": edge_op,
        "edge_normalizer": edge_normalizer,
        "power_normalizer": power_normalizer,
    }


def _module(cfg, cigre_grid, s, **kwargs):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.train import SEModule

    return SEModule(
        cfg,
        n_nodes=s["source"].n_nodes,
        normalizer=s["normalizer"],
        graph=s["source"].graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
        edge_op=s["edge_op"],
        edge_normalizer=s["edge_normalizer"],
        **kwargs,
    )


def test_step_with_currents_trains_and_grads_flow(cigre_grid, edge_setup):
    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mod.set_masker(RandomMasker(0.5))
    batch = {"v": edge_setup["train_v"][:4]}
    loss, mask, pred_v, target_v, _, _, _ = mod._step(batch, train=True)
    assert torch.isfinite(loss)
    loss.backward()
    # the EAGERLY-built edge-observation encoder participates in the forward/backward
    w = mod.model.edge_obs_encoder.weight
    assert w.grad is not None and torch.isfinite(w.grad).all()
    # ... and its parameters are registered (state_dict + optimizer) at construction
    assert "model.edge_obs_encoder.weight" in mod.state_dict()
    out = mod.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(w) in opt_params


def test_step_with_currents_sequence_batch(cigre_grid, edge_setup):
    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mod.set_masker(RandomMasker(0.5))
    v = edge_setup["train_v"][:6].reshape(2, 3, len(ORDERS), -1)  # [B, T, H, N]
    loss, mask, pred_v, *_ = mod._step({"v": v}, train=True)
    assert torch.isfinite(loss)
    assert pred_v.shape == v.shape


def test_edge_mask_derives_from_node_mask(cigre_grid, edge_setup):
    """All-incident default: a half-edge is observed iff its source row is observed."""
    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mod.set_masker(RandomMasker(0.3))
    v = edge_setup["train_v"][:3].to(torch.complex64)
    mask, order_mask = mod._draw_mask((3,), v.device)
    edge_obs, edge_mask, edge_order_mask, _ = mod._edge_observations(
        v, mask, order_mask, (3,), v.device
    )
    src = edge_setup["edge_op"].edge_src_rows
    assert torch.equal(edge_mask, mask[:, src])
    assert edge_order_mask is None  # homogeneous run
    # unobserved edges carry zero features (masked like the node observations)
    assert edge_obs[~edge_mask].abs().sum() == 0
    assert edge_obs.shape[-1] == len(ORDERS) * 3  # trig C=3


def test_placement_with_explicit_edge_rows(cigre_grid, edge_setup):
    """A device-based placement's explicit CT set overrides the all-incident default."""
    from injection_hse.data import MeasurementModel

    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mm = MeasurementModel(["Bus R4", "Bus C5"])
    rows = mm.observed_rows(cigre_grid)
    explicit = torch.tensor([0, 5, 11], dtype=torch.int64)
    mod.set_masker(PlacementMasker(rows, observed_edge_rows=explicit))
    v = edge_setup["train_v"][:2].to(torch.complex64)
    mask, order_mask = mod._draw_mask((2,), v.device)
    _, edge_mask, _, _ = mod._edge_observations(v, mask, order_mask, (2,), v.device)
    expected = torch.zeros_like(edge_mask)
    expected[:, explicit] = True
    assert torch.equal(edge_mask, expected)


def test_measure_power_doubles_edge_channels(cigre_grid, edge_setup):
    cfg = _cfg(measure_current=True, measure_power=True)
    mod = _module(
        cfg, cigre_grid, edge_setup, power_normalizer=edge_setup["power_normalizer"]
    )
    mod.set_masker(RandomMasker(0.5))
    v = edge_setup["train_v"][:2].to(torch.complex64)
    mask, order_mask = mod._draw_mask((2,), v.device)
    edge_obs, _, _, _ = mod._edge_observations(v, mask, order_mask, (2,), v.device)
    assert edge_obs.shape[-1] == len(ORDERS) * 3 * 2  # current + power blocks
    loss = mod._step({"v": edge_setup["train_v"][:2]}, train=True)[0]
    assert torch.isfinite(loss)


def test_missing_edge_pieces_raise(cigre_grid, edge_setup):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.train import SEModule

    cfg = _cfg(measure_current=True)
    with pytest.raises(ValueError, match="measure_current"):
        SEModule(
            cfg,
            n_nodes=edge_setup["source"].n_nodes,
            normalizer=edge_setup["normalizer"],
            graph=edge_setup["source"].graph(),
            base_v=base_voltage_per_row(cigre_grid),
        )


def test_train_experiment_currents_hetero_end_to_end(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    """Full curriculum with currents + heterogeneous fleet: trains, checkpoints, reloads."""
    import lightning as L

    from injection_hse import train as train_mod
    from injection_hse.config import TrainConfig
    from injection_hse.train import SEModule

    cfg = ExperimentConfig(
        name="currents-hetero",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=32, num_layers=2, conv="gine"),
        measurement=MeasurementConfig(
            voltage_nodes=["Bus R4"],
            fundamental_nodes=["Bus C5"],
            measure_current=True,
            heterogeneous_orders=True,
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="mixed",
                    kind="random",
                    observed_fraction=0.5,
                    fundamental_fraction=0.3,
                    target_loss=1.0,
                    max_epochs=2,
                ),
                MaskStageConfig(
                    name="placement",
                    kind="placement",
                    target_loss=1.0,
                    max_epochs=2,
                ),
            ]
        ),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )

    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)
    train_mod.train_experiment(cfg, grid=cigre_grid)

    ckpt = tmp_path / "ckpt" / "stage_placement.ckpt"
    assert ckpt.is_file()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "edge_normalizer" in payload
    reloaded = SEModule.from_checkpoint(ckpt, grid=cigre_grid)
    assert reloaded.edge_op is not None and reloaded._hetero


def test_checkpoint_round_trip_with_currents(cigre_grid, edge_setup, tmp_path):
    from injection_hse.masking import FullMasker
    from injection_hse.train import SEModule

    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mod.set_masker(FullMasker())
    ckpt = {"state_dict": mod.state_dict()}
    mod.on_save_checkpoint(ckpt)
    assert "edge_normalizer" in ckpt
    path = tmp_path / "edge.ckpt"
    torch.save(ckpt, path)

    reloaded = SEModule.from_checkpoint(
        path, grid=cigre_grid, source=edge_setup["source"]
    )
    reloaded.set_masker(FullMasker())
    batch = {"v": edge_setup["train_v"][:2]}
    a = mod.infer(batch)["pred_v"]
    b = reloaded.infer(batch)["pred_v"]
    torch.testing.assert_close(a, b)


def test_upstream_edge_mask_one_ct_per_location(cigre_grid):
    """The one-CT budget keeps, per non-slack node, exactly its phase count of
    half-edges (the branch toward the slack); the slack keeps its incident feeders."""
    from collections import Counter

    from pgml.assembly import node_phase_index
    from pgml.topology import slack_node_ids

    from injection_hse.data import build_graph, upstream_edge_mask

    graph = build_graph(cigre_grid)
    keep = upstream_edge_mask(cigre_grid, graph.edge_index)
    idx = node_phase_index(cigre_grid)
    src_nodes = idx.node_ids[graph.edge_index[0]]
    kept = Counter(int(n) for n, k in zip(src_nodes.tolist(), keep.tolist()) if k)
    slack = {int(s) for s in slack_node_ids(cigre_grid)}
    phases = {n.id: len(n.phases) for n in cigre_grid.nodes}
    for nid, count in kept.items():
        if nid in slack:
            assert count > phases[nid]  # boundary meter: all incident feeder heads
        else:
            assert count == phases[nid]  # one branch toward the slack, per phase
    # every energized non-slack node with a metered half-edge appears
    assert len(kept) > 30


def test_upstream_channels_restrict_edge_mask(cigre_grid, edge_setup):
    """current_channels='upstream' restricts the derived edge mask to the
    toward-slack half-edges; 'incident' (default) keeps every connected branch."""
    s = edge_setup
    module_inc = _module(_cfg(measure_current=True), cigre_grid, s)
    cfg_up = _cfg(measure_current=True)
    cfg_up.measurement.current_channels = "upstream"
    module_up = _module(cfg_up, cigre_grid, s)
    v = s["train_v"][:2].to(torch.complex64)
    mask = torch.ones(2, module_inc.n_nodes, dtype=torch.bool)
    _, em_inc, _, _ = module_inc._edge_observations(v, mask, None, (2,), v.device)
    _, em_up, _, _ = module_up._edge_observations(v, mask, None, (2,), v.device)
    assert int(em_up.sum()) < int(em_inc.sum())
    assert bool((em_up & ~em_inc).sum() == 0)  # upstream is a strict subset


def test_analysis_seam_honors_upstream_channels(cigre_grid, edge_setup):
    """The analysis attribution seam derives the SAME edge mask as the training
    step for an upstream-budget module (the probe must not feed incident channels
    a one-CT model never trained on)."""
    from injection_hse.analysis.attribution import _edge_features

    s = edge_setup
    cfg_up = _cfg(measure_current=True)
    cfg_up.measurement.current_channels = "upstream"
    module_up = _module(cfg_up, cigre_grid, s)
    v = s["train_v"][:2].to(torch.complex64)
    mask = torch.ones(2, module_up.n_nodes, dtype=torch.bool)
    _, em_train, _, _ = module_up._edge_observations(v, mask, None, (2,), v.device)
    _, em_probe, _ = _edge_features(module_up, v, mask)
    assert torch.equal(em_probe, em_train)


# =============================================================================
# Under sensor noise a CT reads the TRUE current with its own error only
# =============================================================================
def _noisy_current_module(cigre_grid, s):
    from injection_hse.config import NoiseConfig

    cfg = _cfg(measure_current=True).model_copy(
        update={"noise": NoiseConfig(enabled=True, seed=0)}
    )
    mod = _module(cfg, cigre_grid, s)
    assert mod._noise is not None and mod._edge_noise is not None
    mod._edge_noise = None  # isolate the derivation from the CT's own error draw
    return mod, s["train_v"][:4].to(torch.complex64)


def test_noisy_voltages_do_not_leak_into_measured_currents(cigre_grid, edge_setup):
    mod, v = _noisy_current_module(cigre_grid, edge_setup)
    v_in = mod._noise.apply(v)
    assert not torch.equal(v_in, v)
    mask = torch.ones(v.shape[0], v.shape[-1], dtype=torch.bool)
    *_, i_edges = mod._edge_observations(
        v_in, mask, None, (v.shape[0],), v.device, v_true=v
    )
    assert torch.equal(i_edges, mod.edge_op(v))
    # what the old derivation produced: the voltage error amplified across short branches
    i_from_noisy = mod.edge_op(v_in)
    rel = (i_from_noisy - i_edges).abs() / i_edges.abs().clamp_min(1e-9)
    assert float(rel.median()) > 0.01


def test_baseline_measured_inputs_read_true_currents(cigre_grid, edge_setup):
    from injection_hse.analysis.baselines import measured_inputs

    mod, v = _noisy_current_module(cigre_grid, edge_setup)
    mask = torch.ones(v.shape[0], v.shape[-1], dtype=torch.bool)
    v_meas, i_edges, _ = measured_inputs(mod, v, mask, noisy=True)
    assert not torch.equal(v_meas, v)
    assert torch.equal(i_edges, mod.edge_op(v))


def test_currents_are_read_at_the_loaded_precision(cigre_grid, edge_setup):
    """A complex64 run derives its CT readings from the state AS LOADED (complex128).

    A feeder-head current at the 20 kV slack is the admittance times a voltage difference
    far below what complex64 resolves at 20 kV, so deriving it after the cast to the
    training dtype returns rounding noise instead of the current.
    """
    from injection_hse.analysis.baselines import measured_inputs

    mod, _ = _noisy_current_module(cigre_grid, edge_setup)
    assert mod._complex_dtype == torch.complex64
    v = edge_setup["train_v"][:4].to(torch.complex128)
    mask = torch.ones(v.shape[0], v.shape[-1], dtype=torch.bool)
    exact = mod.edge_op(v)  # complex128: the weights are promoted to the input dtype
    *_, i_edges = mod._edge_observations(
        v.to(torch.complex64), mask, None, (v.shape[0],), v.device, v_true=v
    )
    assert i_edges.dtype == torch.complex64
    torch.testing.assert_close(i_edges, exact.to(torch.complex64))
    _, i_base, _ = measured_inputs(mod, v, mask)
    torch.testing.assert_close(i_base, exact.to(torch.complex64))
    # the derivation after the cast misses the currents of the branches at the 20 kV rows
    from injection_hse.data import base_voltage_per_row

    mv_rows = torch.nonzero(base_voltage_per_row(cigre_grid) > 1000.0).reshape(-1)
    slack = torch.isin(mod._edge_src_rows, mv_rows)
    assert bool(slack.any())
    lossy = mod.edge_op(v.to(torch.complex64)).to(torch.complex128)
    fund = mod.orders.index(1)
    err_lossy = (lossy - exact)[:, fund][:, slack].abs().max()
    err_kept = (i_edges.to(torch.complex128) - exact)[:, fund][:, slack].abs().max()
    assert float(err_lossy) > 100.0 * float(err_kept)


# =============================================================================
# Loss rows: every row (default) or the unobserved rows only
# =============================================================================
def test_supervise_unobserved_rows_restricts_the_loss(cigre_grid, edge_setup):
    from injection_hse.config import LossConfig

    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    mask = torch.zeros(2, edge_setup["train_v"].shape[-1], dtype=torch.bool)
    mask[:, :5] = True
    assert mod._supervision_rows(mask) is None  # default: every row
    cfg_u = cfg.model_copy(update={"loss": LossConfig(supervise_rows="unobserved")})
    mod_u = _module(cfg_u, cigre_grid, edge_setup)
    rows = mod_u._supervision_rows(mask)
    assert rows.shape == mask.shape and torch.equal(rows, ~mask)
    # a fully observed batch falls back to every row
    assert mod_u._supervision_rows(torch.ones_like(mask)) is None
    mod_u.set_masker(RandomMasker(0.5))
    loss, *_ = mod_u._step({"v": edge_setup["train_v"][:4]}, train=True)
    assert torch.isfinite(loss)
    loss.backward()


def test_adam_eps_reaches_the_optimizer(cigre_grid, edge_setup):
    from injection_hse.config import OptimConfig

    cfg = _cfg(measure_current=True)
    mod = _module(cfg, cigre_grid, edge_setup)
    default = mod.configure_optimizers()
    opt = default["optimizer"] if isinstance(default, dict) else default
    opt = opt[0] if isinstance(opt, (list, tuple)) else opt
    assert opt.param_groups[0]["eps"] == 1e-8
    cfg_e = cfg.model_copy(update={"optim": OptimConfig(name="adamw", eps=1e-12)})
    mod_e = _module(cfg_e, cigre_grid, edge_setup)
    out = mod_e.configure_optimizers()
    opt_e = out["optimizer"] if isinstance(out, dict) else out
    opt_e = opt_e[0] if isinstance(opt_e, (list, tuple)) else opt_e
    assert opt_e.param_groups[0]["eps"] == 1e-12
