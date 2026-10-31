"""Heterogeneous sensor fleets: fundamental-only monitors, per-order masks, metrics."""

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
from injection_hse.masking import FullMasker, PlacementMasker, RandomMasker

ORDERS = [1, 3, 5, 7, 9, 11, 13]
H = len(ORDERS)


# =============================================================================
# maskers
# =============================================================================
def test_random_masker_order_mask_off_is_none():
    m = RandomMasker(0.5)
    mask = m.mask(4, 10, generator=torch.Generator().manual_seed(0))
    assert m.order_mask(mask, H, fundamental_pos=0) is None


def test_random_masker_demotes_observed_rows_to_fundamental():
    gen = torch.Generator().manual_seed(0)
    m = RandomMasker(0.5, fundamental_fraction=0.5)
    mask = m.mask(16, 20, generator=gen)
    om = m.order_mask(mask, H, fundamental_pos=0, generator=gen)
    assert om.shape == (16, 20, H)
    # unobserved rows observe nothing; observed rows are full OR fundamental-only
    assert not om[~mask].any()
    full = om.all(dim=-1)
    fund_only = om[..., 0] & ~om[..., 1:].any(dim=-1)
    assert ((full | fund_only) == mask).all()
    n_demoted = int((fund_only & mask).sum())
    assert 0 < n_demoted < int(mask.sum())  # some demoted, not all


def test_random_masker_no_fundamental_solved_demotes_to_nothing():
    gen = torch.Generator().manual_seed(0)
    m = RandomMasker(0.5, fundamental_fraction=1.0)
    mask = m.mask(4, 10, generator=gen)
    om = m.order_mask(mask, 3, fundamental_pos=None, generator=gen)
    assert not om.any()  # every observed row demoted, no fundamental to keep


def test_placement_masker_order_capability():
    rows = torch.tensor([1, 3])
    cap = torch.ones(6, H, dtype=torch.bool)
    cap[3] = False
    cap[3, 0] = True  # row 3: fundamental-only sensor
    m = PlacementMasker(rows, order_capability=cap)
    mask = m.mask(2, 6)
    om = m.order_mask(mask, H, fundamental_pos=0)
    assert om[:, 1].all()  # full-harmonic sensor row
    assert om[:, 3, 0].all() and not om[:, 3, 1:].any()  # fundamental-only row
    assert not om[:, 0].any()  # unobserved row
    with pytest.raises(ValueError, match="orders"):
        m.order_mask(mask, H + 1, fundamental_pos=0)


def test_full_masker_order_mask_is_none():
    m = FullMasker()
    assert m.order_mask(m.mask(2, 4), H, fundamental_pos=0) is None


# =============================================================================
# MeasurementModel: fundamental nodes, capability, device-based edges
# =============================================================================
def test_measurement_model_fundamental_nodes(cigre_grid):
    from injection_hse.data import MeasurementModel
    from pgml.assembly import node_phase_index

    mm = MeasurementModel(["Bus R4"], fundamental_nodes=["Bus C5"], slack_known=True)
    idx = node_phase_index(cigre_grid)
    rows = mm.observed_rows(cigre_grid)
    c5 = next(n.id for n in cigre_grid.nodes if n.name == "Bus C5")
    assert set(idx.rows(c5)) <= set(rows.tolist())  # smart-meter rows are observed

    cap = mm.order_capability(cigre_grid, ORDERS)
    assert cap is not None and cap.shape == (idx.size, H)
    # fundamental-only rows cover order 1 only; full-harmonic rows cover everything
    assert cap[idx.rows(c5)][:, 0].all() and not cap[idx.rows(c5)][:, 1:].any()
    r4 = next(n.id for n in cigre_grid.nodes if n.name == "Bus R4")
    assert cap[idx.rows(r4)].all()
    # the metered slack stays full-harmonic
    from pgml.topology import slack_node_ids

    for sid in slack_node_ids(cigre_grid):
        assert cap[idx.rows(sid)].all()


def test_measurement_model_homogeneous_capability_is_none(cigre_grid):
    from injection_hse.data import MeasurementModel

    mm = MeasurementModel(["Bus R4", "Bus C5"])
    assert mm.order_capability(cigre_grid, ORDERS) is None
    assert mm.observed_edges(cigre_grid) is None  # config-driven: all-incident default


def test_measurement_model_from_grid_devices(cigre_grid):
    from injection_hse.data import MeasurementModel
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import (
        CurrentChannel,
        MeasuredQuantity,
        MeasurementDevice,
    )
    from pgml.topology import slack_node_ids

    grid = cigre_grid.model_copy(deep=True)
    idx = node_phase_index(grid)
    r4 = next(n.id for n in grid.nodes if n.name == "Bus R4")
    c5 = next(n.id for n in grid.nodes if n.name == "Bus C5")
    incident_r4 = [b for b in grid.branches if r4 in (int(b.from_node), int(b.to_node))]
    assert incident_r4
    devices = [
        # full PQ meter at R4, CTs restricted to ONE incident branch
        MeasurementDevice(
            id=1,
            node=r4,
            measured_quantities=(MeasuredQuantity.VOLTAGE, MeasuredQuantity.CURRENT),
            current_channels=[CurrentChannel(branch=int(incident_r4[0].id))],
        ),
        # fundamental-only smart meter at C5, all incident branches metered
        MeasurementDevice(
            id=2,
            node=c5,
            measured_quantities=(MeasuredQuantity.VOLTAGE, MeasuredQuantity.CURRENT),
            max_harmonic_order=1,
        ),
    ]
    # meter every slack so slack_known holds
    for k, sid in enumerate(slack_node_ids(grid)):
        devices.append(
            MeasurementDevice(
                id=100 + k,
                node=int(sid),
                measured_quantities=(
                    MeasuredQuantity.VOLTAGE,
                    MeasuredQuantity.CURRENT,
                ),
            )
        )
    grid.attach_measurement_devices(devices)

    mm = MeasurementModel.from_grid(grid)
    assert mm.slack_known
    assert c5 in mm.fundamental_nodes and r4 in mm.voltage_nodes
    cap = mm.order_capability(grid, ORDERS)
    assert cap[idx.rows(c5)][:, 1:].sum() == 0

    edges = mm.observed_edges(grid)
    assert edges is not None and edges.numel() > 0
    # resolve against the graph layout: observed half-edges at R4 belong to the
    # single CT'd branch only; C5's half-edges cover every incident branch.
    from injection_hse.data import build_graph

    graph = build_graph(grid, orders=ORDERS)
    src_rows = graph.edge_index[0]
    r4_rows = set(idx.rows(r4))
    c5_rows = set(idx.rows(c5))
    r4_edges_observed = [e for e in edges.tolist() if int(src_rows[e]) in r4_rows]
    assert r4_edges_observed  # the CT'd branch is covered
    # every observed R4 half-edge must belong to the CT'd branch — count the
    # total incident half-edges and check the restriction bites.
    r4_all_edges = [
        e for e in range(graph.edge_index.shape[1]) if int(src_rows[e]) in r4_rows
    ]
    assert len(r4_edges_observed) < len(r4_all_edges)
    c5_edges_observed = [e for e in edges.tolist() if int(src_rows[e]) in c5_rows]
    c5_all_edges = [
        e for e in range(graph.edge_index.shape[1]) if int(src_rows[e]) in c5_rows
    ]
    assert sorted(c5_edges_observed) == sorted(c5_all_edges)


# =============================================================================
# metrics: per-order observed/unobserved split
# =============================================================================
def test_state_errors_order_mask_split():
    from injection_hse.metrics import state_errors

    orders = [1, 3]
    target = torch.ones(1, 2, 2, dtype=torch.complex64)
    pred = target.clone()
    pred[0, 1, 0] += 1.0  # error only at (order 3, row 0)
    mask = torch.tensor([[True, False]])  # row 0 observed (location level)

    # homogeneous: the error is at an OBSERVED row
    errs = state_errors(pred, target, mask=mask, orders=orders)
    assert errs[3]["mae_obs"] > 0 and errs[3]["mae_unobs"] == 0

    # heterogeneous: row 0 is a fundamental-only smart meter -> at order 3 the row
    # counts as UNOBSERVED (its harmonics are estimated, not measured)
    order_mask = torch.tensor([[[True, False], [False, False]]])  # [B, N, H]
    errs = state_errors(pred, target, mask=mask, orders=orders, order_mask=order_mask)
    assert errs[3]["mae_obs"] == 0 and errs[3]["mae_unobs"] > 0
    assert errs[1]["mae_obs"] == 0 and errs[1]["mae_unobs"] == 0  # order 1 exact


# =============================================================================
# SEModule: heterogeneous end-to-end step
# =============================================================================
def _hetero_cfg(**measurement) -> ExperimentConfig:
    return ExperimentConfig(
        name="hetero",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="unused"),
        model=GNNConfig(hidden_dim=32, num_layers=2, conv="gine"),
        measurement=MeasurementConfig(
            voltage_nodes=["Bus R4"],
            fundamental_nodes=["Bus C5"],
            heterogeneous_orders=True,
            **measurement,
        ),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="r",
                    kind="random",
                    observed_fraction=0.5,
                    fundamental_fraction=0.5,
                    target_loss=1.0,
                    max_epochs=1,
                )
            ]
        ),
    )


@pytest.fixture(scope="module")
def hetero_setup(cigre_grid, se_dataset):
    from injection_hse.data import ParquetDataset
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer

    source = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.2)
    train_ds = source.train_dataset()
    train_v = torch.stack([train_ds[i]["v"] for i in range(32)])
    encoding = build_encoding("trig")
    normalizer = HarmonicNormalizer().fit(encoding.encode(train_v).transpose(-3, -2))
    return {"source": source, "train_v": train_v, "normalizer": normalizer}


def test_hetero_step_random_stage(cigre_grid, hetero_setup):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.train import SEModule

    cfg = _hetero_cfg()
    mod = SEModule(
        cfg,
        n_nodes=hetero_setup["source"].n_nodes,
        normalizer=hetero_setup["normalizer"],
        graph=hetero_setup["source"].graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
    )
    # node token gains H per-order channels: F + mask + H + static(8)
    assert mod.model.encoder[0].in_features == H * 3 + 1 + H + 8
    mod.set_masker(RandomMasker(0.5, fundamental_fraction=0.5))
    loss, mask, pred_v, target_v, _, _, order_mask = mod._step(
        {"v": hetero_setup["train_v"][:4]}, train=True
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert order_mask is not None and order_mask.shape == (*mask.shape, H)
    # demotion only strikes observed rows; the metered slack stays full-harmonic
    assert not order_mask[~mask].any()
    slack_rows = mod._forced_obs_rows
    assert order_mask[:, slack_rows, :].all()


def test_hetero_placement_capability_flows(cigre_grid, hetero_setup):
    from injection_hse.data import MeasurementModel, base_voltage_per_row
    from injection_hse.train import SEModule
    from pgml.assembly import node_phase_index

    cfg = _hetero_cfg()
    mm = MeasurementModel(
        cfg.measurement.voltage_nodes,
        fundamental_nodes=cfg.measurement.fundamental_nodes,
        slack_known=True,
    )
    mod = SEModule(
        cfg,
        n_nodes=hetero_setup["source"].n_nodes,
        normalizer=hetero_setup["normalizer"],
        graph=hetero_setup["source"].graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
    )
    mod.set_masker(
        PlacementMasker(
            mm.observed_rows(cigre_grid),
            order_capability=mm.order_capability(cigre_grid, ORDERS),
        )
    )
    mask, order_mask = mod._draw_mask((3,), torch.device("cpu"))
    idx = node_phase_index(cigre_grid)
    c5 = next(n.id for n in cigre_grid.nodes if n.name == "Bus C5")
    c5_rows = torch.tensor(idx.rows(c5))
    assert order_mask[:, c5_rows, 0].all()  # fundamental observed
    assert not order_mask[:, c5_rows, 1:].any()  # harmonics estimated


def test_hetero_with_currents_transfers_order_capability_to_edges(
    cigre_grid, hetero_setup, se_dataset
):
    from injection_hse.data import base_voltage_per_row
    from injection_hse.data.currents import BranchCurrentOperator
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule

    cfg = _hetero_cfg(measure_current=True)
    encoding = build_encoding("trig")
    edge_op = BranchCurrentOperator(cigre_grid, ORDERS, dtype=torch.complex64)
    i_train = edge_op(hetero_setup["train_v"].to(torch.complex64))
    edge_normalizer = HarmonicNormalizer().fit(
        encoding.encode(i_train).transpose(-3, -2)
    )
    mod = SEModule(
        cfg,
        n_nodes=hetero_setup["source"].n_nodes,
        normalizer=hetero_setup["normalizer"],
        graph=hetero_setup["source"].graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
        edge_op=edge_op,
        edge_normalizer=edge_normalizer,
    )
    mod.set_masker(RandomMasker(0.5, fundamental_fraction=0.5))
    v = hetero_setup["train_v"][:3].to(torch.complex64)
    mask, order_mask = mod._draw_mask((3,), v.device)
    edge_obs, edge_mask, edge_order_mask, _ = mod._edge_observations(
        v, mask, order_mask, (3,), v.device
    )
    assert edge_order_mask is not None
    # a fundamental-only metering row reads fundamental currents only
    src = edge_op.edge_src_rows
    assert torch.equal(edge_order_mask, order_mask[:, src, :] & edge_mask.unsqueeze(-1))
    loss = mod._step({"v": hetero_setup["train_v"][:3]}, train=True)[0]
    assert torch.isfinite(loss)


# =============================================================================
# config validation
# =============================================================================
def test_config_validators():
    with pytest.raises(ValueError, match="heterogeneous_orders"):
        MeasurementConfig(voltage_nodes=["a"], fundamental_nodes=["b"])
    with pytest.raises(ValueError, match="measure_current"):
        MeasurementConfig(measure_power=True)
    with pytest.raises(ValueError, match="BOTH"):
        MeasurementConfig(
            voltage_nodes=["a"],
            fundamental_nodes=["a"],
            heterogeneous_orders=True,
        )
    with pytest.raises(ValueError, match="fundamental_fraction"):
        ExperimentConfig(
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="r",
                        kind="random",
                        observed_fraction=0.5,
                        fundamental_fraction=0.5,
                        target_loss=1.0,
                        max_epochs=1,
                    )
                ]
            )
        )
