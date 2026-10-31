"""Data layer: measurement-row resolution, graph build, and the parquet dataset."""

from __future__ import annotations

import pytest
import torch

from injection_hse.data import (
    MeasurementModel,
    ParquetDataset,
    build_graph,
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def test_measurement_rows_resolve_by_name(cigre_grid):
    from pgml.assembly import node_phase_index
    from pgml.topology import slack_node_ids

    idx = node_phase_index(cigre_grid)
    mm = MeasurementModel(["Bus R4", "Bus C5"], slack_known=True)
    rows = mm.observed_rows(cigre_grid)

    # one observed-row block per sensor node, every phase of that node, PLUS the
    # metered slack rows (the slack is a measurement like any other)
    r4 = [n.id for n in cigre_grid.nodes if n.name == "Bus R4"][0]
    c5 = [n.id for n in cigre_grid.nodes if n.name == "Bus C5"][0]
    slack = [r for s in slack_node_ids(cigre_grid) for r in idx.rows(s)]
    expected = sorted(set(idx.rows(r4) + idx.rows(c5) + slack))
    assert rows.dtype == torch.int64
    assert rows.tolist() == expected


def test_measurement_rows_resolve_by_id(cigre_grid):
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre_grid)
    node_id = cigre_grid.nodes[5].id
    mm = MeasurementModel([node_id], slack_known=False)
    assert mm.observed_rows(cigre_grid).tolist() == sorted(idx.rows(node_id))


def test_measurement_unknown_node_raises(cigre_grid):
    with pytest.raises(ValueError):
        MeasurementModel(["does-not-exist"]).observed_rows(cigre_grid)


def test_slack_rows_cover_every_source(cigre_grid):
    from pgml.assembly import node_phase_index
    from pgml.topology import slack_node_ids

    idx = node_phase_index(cigre_grid)
    mm = MeasurementModel([], slack_known=True)
    expected = sorted({r for s in slack_node_ids(cigre_grid) for r in idx.rows(s)})
    assert mm.slack_rows(cigre_grid).tolist() == expected
    # deprecated alias kept for callers of the old boundary naming
    assert mm.known_boundary_rows(cigre_grid).tolist() == expected
    assert MeasurementModel([], slack_known=False).slack_rows(cigre_grid).numel() == 0
    # the metered slack is part of the observed rows (a measurement, not a side input)
    assert mm.observed_rows(cigre_grid).tolist() == expected


def test_measurement_model_from_grid_devices(cigre_grid):
    from pgml.schemas.grid_schema import MeasurementDevice
    from pgml.topology import slack_node_ids

    grid = cigre_grid.model_copy(deep=True)
    slack = slack_node_ids(grid)[0]
    sensor = [n.id for n in grid.nodes if n.name == "Bus R4"][0]
    grid.attach_measurement_devices(
        [
            MeasurementDevice(id=1, node=slack, accuracy_class="0.2S"),
            MeasurementDevice(id=2, node=sensor),
        ]
    )
    mm = MeasurementModel.from_grid(grid)
    assert mm.slack_known  # every slack node carries a voltage device
    assert sorted(mm.voltage_nodes) == sorted({slack, sensor})

    # without a device at the slack, the sensor set cannot provide the boundary
    grid2 = cigre_grid.model_copy(deep=True)
    grid2.attach_measurement_devices([MeasurementDevice(id=1, node=sensor)])
    assert not MeasurementModel.from_grid(grid2).slack_known


def test_measurement_model_resolve_prefers_the_installed_devices(cigre_grid):
    """A configured sensor set resolves to the DEVICE placement when it names those meters.

    Training, evaluation and deployment all resolve the placement through this one rule; a
    device-based placement meters ONE branch per meter, a config-driven one every incident
    branch, so a mismatch would feed the deployed model an edge set it never trained on.
    """
    from pgml.schemas.grid_schema import (
        CurrentChannel,
        MeasuredQuantity,
        MeasurementDevice,
    )
    from pgml.topology import slack_node_ids

    grid = cigre_grid.model_copy(deep=True)
    slack = slack_node_ids(grid)[0]
    sensor = [n.id for n in grid.nodes if n.name == "Bus R4"][0]
    metered = next(
        int(b.id)
        for b in grid.branches
        if sensor in (int(b.from_node), int(b.to_node))
        and int(b.from_node) != int(b.to_node)
    )
    grid.attach_measurement_devices(
        [
            MeasurementDevice(id=1, node=slack),
            MeasurementDevice(
                id=2,
                node=sensor,
                measured_quantities=(
                    MeasuredQuantity.VOLTAGE,
                    MeasuredQuantity.CURRENT,
                ),
                current_channels=[CurrentChannel(branch=metered)],
            ),
        ]
    )
    resolved = MeasurementModel.resolve(grid, [slack, sensor])
    assert resolved._device_based
    # ... and its CT channels are the metered half-edges, not every incident branch
    edges = resolved.observed_edges(grid)
    assert edges is not None and edges.numel() == len(grid.nodes[0].phases)

    # a sensor set that is NOT the installed fleet keeps the configured (all-incident) model
    other = [n.id for n in grid.nodes if n.name == "Bus C5"][0]
    fallback = MeasurementModel.resolve(grid, [slack, sensor, other])
    assert not fallback._device_based
    assert fallback.observed_edges(grid) is None
    # ... also when the sensors are given by NAME
    by_name = MeasurementModel.resolve(grid, ["Bus R4", "Bus C5"])
    assert not by_name._device_based
    assert torch.equal(by_name.observed_rows(grid), fallback.observed_rows(grid))

    # a grid without instrumentation is unaffected
    assert not MeasurementModel.resolve(cigre_grid, [slack, sensor])._device_based


def test_measurement_model_resolve_restricts_to_a_meter_subset(cigre_grid):
    """A placement over PART of the installed meters stays device-based.

    An estimator may use a subset of the instrumentation as its inputs, the rest serving as
    independent validation channels. The placement then carries the CT channels of ITS
    meters only — falling back to all-incident edges would demand CTs the deployment does
    not meter, and the validation instruments' channels must never leak into the input set.
    """
    from pgml.schemas.grid_schema import (
        CurrentChannel,
        MeasuredQuantity,
        MeasurementDevice,
    )
    from pgml.topology import slack_node_ids

    grid = cigre_grid.model_copy(deep=True)
    slack = slack_node_ids(grid)[0]
    sensor = [n.id for n in grid.nodes if n.name == "Bus R4"][0]
    validation = [n.id for n in grid.nodes if n.name == "Bus C5"][0]

    def channel_of(node: int) -> int:
        return next(
            int(b.id)
            for b in grid.branches
            if node in (int(b.from_node), int(b.to_node))
            and int(b.from_node) != int(b.to_node)
        )

    quantities = (MeasuredQuantity.VOLTAGE, MeasuredQuantity.CURRENT)
    grid.attach_measurement_devices(
        [
            MeasurementDevice(id=1, node=slack),
            MeasurementDevice(
                id=2,
                node=sensor,
                measured_quantities=quantities,
                current_channels=[CurrentChannel(branch=channel_of(sensor))],
            ),
            MeasurementDevice(
                id=3,
                node=validation,
                measured_quantities=quantities,
                current_channels=[CurrentChannel(branch=channel_of(validation))],
            ),
        ]
    )

    resolved = MeasurementModel.resolve(grid, [slack, sensor])
    assert resolved._device_based
    assert sorted(resolved.voltage_nodes) == sorted([slack, sensor])
    # its CT channels are the INPUT meter's alone — the validation CT contributes nothing
    edges = resolved.observed_edges(grid)
    assert edges is not None and edges.numel() == len(grid.nodes[0].phases)
    from injection_hse.data.currents import _half_edge_slots
    from pgml.assembly import node_phase_index

    slots, _src = _half_edge_slots(grid, node_phase_index(grid))
    assert {slots[e][0] for e in edges.tolist()} == {channel_of(sensor)}

    # a configured node with NO meter still falls back to the all-incident model
    unmetered = [n.id for n in grid.nodes if n.name == "Bus R15"][0]
    fallback = MeasurementModel.resolve(grid, [slack, sensor, unmetered])
    assert not fallback._device_based


def test_build_graph_row_aligned(cigre_grid):
    from pgml.assembly import node_phase_index

    idx = node_phase_index(cigre_grid)
    g = build_graph(cigre_grid, orders=ORDERS)
    assert g.num_nodes == idx.size
    assert g.x.shape == (idx.size, 8)
    assert g.x.dtype == torch.float32
    assert torch.isfinite(g.x).all()
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.dtype == torch.int64
    assert g.edge_attr.shape == (g.edge_index.shape[1], 5)
    assert torch.isfinite(g.edge_attr).all()
    # edges are symmetric (undirected pairs) and reference valid rows
    assert g.edge_index.max() < idx.size
    # the slack node's rows carry the is_slack flag
    from pgml.evaluation.topology import slack_node_id

    slack_rows = idx.rows(slack_node_id(cigre_grid))
    assert (g.x[slack_rows, 5] == 1.0).all()


def test_parquet_dataset_split_and_shapes(cigre_grid, se_dataset):
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    assert ds.orders == ORDERS
    n = ds.n_nodes
    tr, va = ds.train_dataset(), ds.val_dataset()
    assert len(tr) + len(va) == 256
    assert len(va) == round(0.1 * 256)
    item = tr[0]
    assert item["v"].shape == (len(ORDERS), n)
    assert item["v"].dtype == torch.complex64
    # the shared graph aligns with the state width
    assert ds.graph().num_nodes == n


def test_parquet_split_is_deterministic(cigre_grid, se_dataset):
    a = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    b = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    ia = torch.stack([a.val_dataset()[i]["index"] for i in range(len(a.val_dataset()))])
    ib = torch.stack([b.val_dataset()[i]["index"] for i in range(len(b.val_dataset()))])
    assert torch.equal(ia, ib)
    # a different seed gives a different split
    c = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=1)
    ic = torch.stack([c.val_dataset()[i]["index"] for i in range(len(c.val_dataset()))])
    assert not torch.equal(ia, ic)


def test_grid_mismatch_raises(cigre_grid, se_dataset):
    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.oracles import cigre_lv_full_grid

    # a single-phase-equivalent grid has a different node-phase row layout than the
    # 3-phase dataset, so the dataset must reject it (row alignment is a hard requirement).
    other, _ = cigre_lv_full_grid(phase_mode=PhaseMode.SINGLE_PHASE_EQUIV)
    with pytest.raises(ValueError):
        ParquetDataset(str(se_dataset), grid=other)


def test_parquet_dataset_rejects_permuted_phase_codes(
    cigre_grid, se_dataset, monkeypatch
):
    """A dataset whose phase codes differ from the grid layout is rejected (node_ids alone
    do not pin a row — the (node, phase) pair must match)."""
    import dataclasses

    import injection_hse.data.scenarios as scenarios

    loaded = scenarios.read_dataset(str(se_dataset))
    # keep node_ids intact (so the node-id check passes and the PHASE-code check is exercised)
    # but corrupt one row's phase code, mimicking a grid with a different per-node phase order.
    pc = loaded.phase_codes.clone()
    pc[0] = (int(pc[0]) + 1) % 4
    permuted = dataclasses.replace(loaded, phase_codes=pc)
    monkeypatch.setattr(scenarios, "read_dataset", lambda _path: permuted)

    with pytest.raises(ValueError, match="phase code"):
        ParquetDataset(str(se_dataset), grid=cigre_grid)


def test_snapshot_dataset_is_t1(cigre_grid, se_dataset):
    """A [B,H,N] snapshot dataset reports T=1 and yields [H,N] items (unchanged path)."""
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    assert ds.n_steps == 1
    assert ds.train_dataset()[0]["v"].shape == (len(ORDERS), ds.n_nodes)


def test_sequence_dataset_preserves_time(cigre_grid, se_sequence_dataset):
    """A [B,T,H,N] sequence dataset keeps T: each item is the scenario's [T,H,N] sequence."""
    ds = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, val_fraction=0.25, seed=0
    )
    assert ds.n_steps == 4
    item = ds.train_dataset()[0]["v"]
    assert item.shape == (4, len(ORDERS), ds.n_nodes)  # [T, H, N]
    assert item.dtype == torch.complex64


def test_sequence_split_is_by_scenario(cigre_grid, se_sequence_dataset):
    """The split is by SCENARIO (B axis): no scenario id appears in both train and val."""
    ds = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, val_fraction=0.25, seed=0
    )
    tr, va = ds.train_dataset(), ds.val_dataset()
    train_ids = {int(tr[i]["index"]) for i in range(len(tr))}
    val_ids = {int(va[i]["index"]) for i in range(len(va))}
    assert train_ids.isdisjoint(val_ids)  # no scenario leaks across the split
    assert len(train_ids) + len(val_ids) == 12  # all scenarios accounted for
    assert len(val_ids) == round(0.25 * 12)
