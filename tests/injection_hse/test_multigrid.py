"""Multi-grid data path: generation + manifest, the grid-level split, and heterogeneous batching."""

from __future__ import annotations

import pytest
import torch

from injection_hse.data import (
    GraphBatchMasker,
    MultiGridCollate,
    MultiGridDataSource,
    per_unit_state,
    physical_state,
    placement_mask,
    segment_random_mask,
)
from injection_hse.data.multigrid import (
    MultiGridGenerationConfig,
    MultiGridManifest,
    SyntheticFeederSource,
    generate_multigrid_dataset,
    read_grid,
)

N_GRIDS = 6
N_SCENARIOS = 4


@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    """A tiny 6-grid multi-grid dataset (synthetic feeders, fundamental only)."""
    root = tmp_path_factory.mktemp("multigrid")
    config = MultiGridGenerationConfig(
        output_root=str(root),
        n_scenarios=N_SCENARIOS,
        val_fraction=0.34,
        sources=[
            SyntheticFeederSource(n_grids=N_GRIDS, node_range=(4, 9), n_feeders=2)
        ],
    )
    generate_multigrid_dataset(config)
    return root


@pytest.fixture(scope="module")
def multigrid_source(multigrid_root):
    return MultiGridDataSource(multigrid_root)


def _items_from_distinct_grids(source, split="train", count=3):
    """One item from each of ``count`` different member grids of a split."""
    dataset = source.train_dataset() if split == "train" else source.val_dataset()
    seen: dict[int, dict] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        seen.setdefault(int(item["grid"]), item)
        if len(seen) == count:
            break
    assert len(seen) == count, "the fixture must provide several member grids"
    return [seen[k] for k in sorted(seen)]


# =============================================================================
# Generation + manifest
# =============================================================================
def test_manifest_round_trips(multigrid_root):
    manifest = MultiGridManifest.read(multigrid_root)
    assert manifest.version == 5
    assert len(manifest.grids) == N_GRIDS
    assert not manifest.failures
    assert manifest.orders == (1,)
    assert manifest.config.n_scenarios == N_SCENARIOS
    # a snapshot-only corpus records no sequence dataset
    assert not manifest.has_coherent
    assert all(entry.n_coherent_scenarios is None for entry in manifest.grids)

    reloaded = MultiGridManifest.model_validate_json(manifest.model_dump_json())
    assert reloaded == manifest
    # the leakage block exists even when no external source contributed
    assert reloaded.leakage.pgg == []
    assert reloaded.leakage.ding0_excluded_districts == ()


def test_manifest_records_the_generation_provenance(multigrid_root):
    """What produced the corpus: its config fingerprint, the code, the shipped tables."""
    from pgml.scenarios import config_hash
    from injection_hse.data.scenarios import DEVICE_LIBRARY_VERSION, SE_PRESET_VERSION

    manifest = MultiGridManifest.read(multigrid_root)
    assert manifest.config_hash == config_hash(manifest.config)
    assert manifest.provenance["pgml_version"] and manifest.provenance["git_sha"]
    # the recipe and the device library move the data without moving the config, so this
    # package stamps its own revisions beside the engine's provenance
    assert manifest.device_library_version == DEVICE_LIBRARY_VERSION
    assert manifest.se_preset_version == SE_PRESET_VERSION
    for table in ("en50160", "iec61000_3_2"):
        assert manifest.standards[table]["sha256"]
        assert manifest.standards[table]["override"] is False
    # every member dataset says which ideal-switch model solved it: this package's
    # front door materialises an ideal switch as a finite contact
    import json

    from injection_hse._switches import ESTIMATOR_IDEAL_SWITCH_OHM

    metas = list(multigrid_root.rglob("meta.json"))
    assert metas
    for meta_path in metas:
        extra = json.loads(meta_path.read_text())["extra_provenance"]
        assert extra["ideal_switch_ohm"] == ESTIMATOR_IDEAL_SWITCH_OHM
        assert extra["se_preset_version"] == SE_PRESET_VERSION


def test_a_manifest_without_provenance_still_loads():
    """An older corpus (version 2, no provenance block) reads as 'not recorded'."""
    older = {
        "version": 2,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "schema_version": "0.0.5",
        "seed": 0,
        "orders": [1],
        "config": {
            "output_root": "x",
            "sources": [{"kind": "synthetic", "n_grids": 1}],
        },
    }
    manifest = MultiGridManifest.model_validate(older)
    assert manifest.config_hash is None and manifest.provenance == {}
    assert manifest.device_library_version is None and manifest.standards == {}


def test_manifest_entries_describe_their_dataset(multigrid_root):
    from pgml.assembly import node_phase_index
    from injection_hse.data.scenarios import read_dataset

    from injection_hse.data.multigrid import dataset_dir

    manifest = MultiGridManifest.read(multigrid_root)
    for entry in manifest.grids:
        loaded = read_dataset(dataset_dir(multigrid_root, entry.key))
        assert loaded.v.shape[0] == entry.n_scenarios
        assert loaded.v.shape[-1] == entry.n_rows
        grid = read_grid(multigrid_root, entry.key)
        assert int(node_phase_index(grid).size) == entry.n_rows
        assert len(grid.nodes) == entry.n_nodes


def test_persisted_grid_round_trips_to_the_same_row_layout(multigrid_root):
    from pgml.topology import layout_fingerprint

    manifest = MultiGridManifest.read(multigrid_root)
    for entry in manifest.grids:
        grid = read_grid(multigrid_root, entry.key)
        assert layout_fingerprint(grid) == entry.layout_fingerprint


def test_splits_are_disjoint_by_grid(multigrid_root):
    manifest = MultiGridManifest.read(multigrid_root)
    train = {e.key for e in manifest.split("train")}
    val = {e.key for e in manifest.split("val")}
    test = {e.key for e in manifest.split("test")}

    assert train and val  # the generated grids split into both
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == {e.key for e in manifest.grids}


def test_generation_is_reproducible_from_config_and_seed(tmp_path):
    """Same config + seed -> the same members, seeds and solved states."""
    from injection_hse.data.scenarios import read_dataset

    from injection_hse.data.multigrid import dataset_dir

    def _run(name):
        config = MultiGridGenerationConfig(
            output_root=str(tmp_path / name),
            n_scenarios=2,
            sources=[SyntheticFeederSource(n_grids=2, node_range=(4, 5))],
        )
        return generate_multigrid_dataset(config)

    first, second = _run("a"), _run("b")
    assert [e.key for e in first.grids] == [e.key for e in second.grids]
    assert [e.split for e in first.grids] == [e.split for e in second.grids]
    for entry in first.grids:
        a = read_dataset(dataset_dir(tmp_path / "a", entry.key)).v
        b = read_dataset(dataset_dir(tmp_path / "b", entry.key)).v
        assert torch.equal(a, b)


def test_parallel_generation_matches_the_serial_result(tmp_path):
    from injection_hse.data.scenarios import read_dataset

    from injection_hse.data.multigrid import dataset_dir

    def _run(name, workers):
        config = MultiGridGenerationConfig(
            output_root=str(tmp_path / name),
            n_scenarios=2,
            sources=[SyntheticFeederSource(n_grids=3, node_range=(4, 6))],
        )
        return generate_multigrid_dataset(config, workers=workers)

    serial, parallel = _run("serial", 1), _run("parallel", 3)
    assert [e.key for e in serial.grids] == [e.key for e in parallel.grids]
    for entry in serial.grids:
        a = read_dataset(dataset_dir(tmp_path / "serial", entry.key)).v
        b = read_dataset(dataset_dir(tmp_path / "parallel", entry.key)).v
        assert torch.equal(a, b)


def test_generation_leaves_the_callers_torch_threads_alone(tmp_path):
    """A library call must not silently reconfigure the process it runs in.

    The per-grid job pins itself to one torch thread because the POOL provides the
    parallelism; on the serial path that job runs in the caller's process, where shrinking the
    thread pool would slow down (and subtly re-order the BLAS reductions of) everything the
    caller does afterwards.
    """
    before = torch.get_num_threads()
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(tmp_path / "threads"),
            n_scenarios=1,
            sources=[SyntheticFeederSource(n_grids=1, node_range=(4, 5))],
        )
    )
    assert torch.get_num_threads() == before


def test_a_failing_grid_is_recorded_not_raised(tmp_path):
    from injection_hse.data.multigrid import MultiGridGenerationConfig, _generate_one, _Task

    config = MultiGridGenerationConfig(
        output_root=str(tmp_path),
        sources=[SyntheticFeederSource(n_grids=1)],
    )
    task = _Task(
        key="broken",
        kind="synthetic",
        source="synthetic:broken",
        cell=None,
        seed=0,
        coherent_seed=1,
        mode_bank_seed=2,
        grid_json="{not a grid}",
        out_dir=str(tmp_path / "broken"),
        config=config,
    )
    result = _generate_one(task)
    assert "failure" in result
    assert result["failure"]["key"] == "broken"


def test_split_rule_holds_out_real_grids_and_whole_cells():
    from injection_hse.data.multigrid import _assign_splits

    rows = [
        {"kind": "pgg", "cell": c, "key": f"g{i}"}
        for i, c in enumerate([0, 0, 1, 2, 3])
    ]
    rows += [
        {"kind": "simbench", "cell": None, "key": "s0"},
        {"kind": "ding0", "cell": None, "key": "d0"},
    ]
    _assign_splits(rows, val_fraction=0.5, seed=0)

    by_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        by_split[row["split"]].add(row["key"])
    assert by_split["test"] == {"s0", "d0"}  # real grids only

    train_cells = {r["cell"] for r in rows if r["split"] == "train"}
    val_cells = {r["cell"] for r in rows if r["split"] == "val"}
    assert val_cells and not (train_cells & val_cells)  # whole cells held out


# =============================================================================
# The source
# =============================================================================
def test_source_exposes_members_and_refuses_single_grid_accessors(multigrid_source):
    manifest = multigrid_source.manifest
    assert len(multigrid_source.grids) == len(manifest.grids)
    assert multigrid_source.orders == [1]
    assert multigrid_source.n_steps == 1

    sizes = {bundle.n_nodes for bundle in multigrid_source.grids}
    assert len(sizes) > 1, "the point of the source is grids of different size"

    for accessor in ("n_nodes", "grid"):
        with pytest.raises(ValueError, match="multi-grid"):
            getattr(multigrid_source, accessor)
    with pytest.raises(ValueError, match="multi-grid"):
        multigrid_source.graph()


def test_source_datasets_follow_the_grid_level_split(multigrid_source):
    train_grids = set(multigrid_source.split_indices("train"))
    val_grids = set(multigrid_source.split_indices("val"))
    assert train_grids and val_grids and not (train_grids & val_grids)

    train = multigrid_source.train_dataset()
    assert len(train) == len(train_grids) * N_SCENARIOS
    assert {int(train[i]["grid"]) for i in range(len(train))} == train_grids

    val = multigrid_source.val_dataset()
    assert {int(val[i]["grid"]) for i in range(len(val))} == val_grids


def test_source_collate_is_the_multi_grid_one(multigrid_source):
    assert isinstance(multigrid_source.collate(), MultiGridCollate)


def test_per_unit_states_are_dimensionless(multigrid_source):
    states = multigrid_source.per_unit_states(split="train", max_scenarios_per_grid=2)
    assert len(states) == len(multigrid_source.split_indices("train"))
    for state in states:
        assert state.shape[0] == 2
        # a solved feeder sits within a few percent of nominal in per-unit
        assert 0.5 < float(state.abs().max()) < 1.5


# =============================================================================
# Collate: the frozen batch contract
# =============================================================================
def test_collate_concatenates_node_axes_with_ptr_and_batch(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    batch = MultiGridCollate(multigrid_source)(items)
    bundles = [multigrid_source.bundle(int(i)) for i in batch["grid_index"]]
    sizes = [b.n_nodes for b in bundles]
    n_total = sum(sizes)

    assert batch["v"].shape == (1, n_total)  # [H, N*]
    assert batch["v"].is_complex()
    assert batch["ptr"].tolist() == [0, sizes[0], sizes[0] + sizes[1], n_total]
    assert batch["batch"].tolist() == [b for b, n in enumerate(sizes) for _ in range(n)]
    assert batch["node_static"].shape == (n_total, bundles[0].graph.x.shape[1])
    assert batch["base_voltage"].shape == (n_total,)
    assert batch["index"].shape == (3,)

    # each item's block is that member's own state / features, unmodified
    for b, (bundle, item) in enumerate(zip(bundles, items)):
        lo, hi = int(batch["ptr"][b]), int(batch["ptr"][b + 1])
        assert torch.equal(batch["v"][..., lo:hi], item["v"])
        assert torch.equal(batch["node_static"][lo:hi], bundle.graph.x)
        assert torch.equal(batch["base_voltage"][lo:hi], bundle.base_voltage)


def test_collate_offsets_edges_into_the_batched_node_axis(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    batch = MultiGridCollate(multigrid_source)(items)
    bundles = [multigrid_source.bundle(int(i)) for i in batch["grid_index"]]

    edge_sizes = [b.n_edges for b in bundles]
    assert batch["edge_index"].shape == (2, sum(edge_sizes))
    assert batch["edge_ptr"].tolist() == [
        0,
        edge_sizes[0],
        edge_sizes[0] + edge_sizes[1],
        sum(edge_sizes),
    ]
    for b, bundle in enumerate(bundles):
        lo, hi = int(batch["edge_ptr"][b]), int(batch["edge_ptr"][b + 1])
        offset = int(batch["ptr"][b])
        assert torch.equal(
            batch["edge_index"][:, lo:hi], bundle.graph.edge_index + offset
        )
        assert torch.equal(batch["edge_static"][lo:hi], bundle.graph.edge_attr)
        assert torch.equal(
            batch["edge_branch_ids"][lo:hi], bundle.graph.edge_branch_ids
        )

    # every half-edge stays inside its own item's node block
    node_of_edge = batch["batch"][batch["edge_index"][0]]
    assert torch.equal(node_of_edge, batch["edge_batch"])
    assert torch.equal(batch["batch"][batch["edge_index"][1]], batch["edge_batch"])


def test_source_builds_every_member_with_the_phase_coupling_edge_set(multigrid_root):
    """One edge-feature layout across the corpus, whatever a given member happens to couple."""
    from injection_hse.config import PhaseCouplingConfig
    from injection_hse.data import edge_feature_dim

    coupling = PhaseCouplingConfig(enabled=True)
    width = edge_feature_dim(coupling)
    plain = MultiGridDataSource(multigrid_root)
    coupled = MultiGridDataSource(multigrid_root, coupling=coupling)
    for key in range(len(plain.grids)):
        base, wide = plain.bundle(key), coupled.bundle(key)
        assert wide.graph.edge_attr.shape[1] == width
        e = base.n_edges
        # the series half-edges lead, unchanged, and every added one is flagged
        assert torch.equal(wide.graph.edge_index[:, :e], base.graph.edge_index)
        assert torch.equal(
            wide.graph.edge_attr[:e, : base.graph.edge_attr.shape[1]],
            base.graph.edge_attr,
        )
        assert float(wide.graph.edge_attr[:e, -2:].abs().sum()) == 0.0
        assert bool((wide.graph.edge_attr[e:, -2] == 1.0).all())

    items = _items_from_distinct_grids(coupled, count=3)
    batch = MultiGridCollate(coupled)(items)
    assert batch["edge_static"].shape == (batch["edge_index"].shape[1], width)
    assert batch["edge_batch"].numel() == batch["edge_index"].shape[1]
    assert batch["edge_branch_ids"].numel() == batch["edge_index"].shape[1]
    node_of_edge = batch["batch"][batch["edge_index"][0]]
    assert torch.equal(node_of_edge, batch["edge_batch"])


def test_collate_maps_slack_rows_to_global_indices(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    batch = MultiGridCollate(multigrid_source)(items)
    bundles = [multigrid_source.bundle(int(i)) for i in batch["grid_index"]]

    expected = torch.cat(
        [b.slack_rows + int(off) for b, off in zip(bundles, batch["ptr"][:-1])]
    )
    assert torch.equal(batch["slack_rows"], expected)
    # a slack row is a slack row of its own item, not of the batch
    assert bool(
        torch.all(batch["node_static"][batch["slack_rows"], 5] == 1.0)
    )  # is_slack feature


def test_collate_carries_a_fixed_placement_as_global_rows(multigrid_root):
    source = MultiGridDataSource(multigrid_root, placement_fraction=0.3)
    items = _items_from_distinct_grids(source, count=2)
    batch = MultiGridCollate(source)(items)

    assert "observed_rows" in batch
    observed = set(batch["observed_rows"].tolist())
    assert set(batch["slack_rows"].tolist()) <= observed
    mask = placement_mask(batch["observed_rows"], int(batch["ptr"][-1]))
    assert int(mask.sum()) == len(observed)


def test_collate_rejects_mismatched_harmonic_axes(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=2)
    items[1] = dict(items[1])
    items[1]["v"] = items[1]["v"].unsqueeze(0)  # a sequence item among snapshots
    with pytest.raises(ValueError, match=r"\[T, H\]"):
        MultiGridCollate(multigrid_source)(items)


# =============================================================================
# Masking in the concatenated space
# =============================================================================
def test_segment_mask_observes_the_fraction_of_each_graph():
    ptr = torch.tensor([0, 10, 30, 34], dtype=torch.int64)
    generator = torch.Generator().manual_seed(0)
    mask = segment_random_mask(ptr, fraction=0.5, generator=generator)

    assert mask.shape == (34,)
    counts = [int(mask[ptr[i] : ptr[i + 1]].sum()) for i in range(3)]
    assert counts == [5, 10, 2]  # per GRAPH, not a global top-k


def test_segment_mask_keeps_at_least_one_row_per_graph():
    ptr = torch.tensor([0, 3, 9], dtype=torch.int64)
    mask = segment_random_mask(
        ptr, fraction=0.01, generator=torch.Generator().manual_seed(1)
    )
    assert int(mask[:3].sum()) == 1 and int(mask[3:].sum()) == 1


def test_segment_mask_forces_the_slack_rows_observed():
    ptr = torch.tensor([0, 8, 16], dtype=torch.int64)
    forced = torch.tensor([0, 1, 8], dtype=torch.int64)
    mask = segment_random_mask(
        ptr,
        fraction=0.25,
        forced_rows=forced,
        generator=torch.Generator().manual_seed(2),
    )
    assert bool(mask[forced].all())


def test_graph_batch_masker_masks_a_real_batch(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    collate = MultiGridCollate(
        multigrid_source,
        masker=GraphBatchMasker(0.5),
        generator=torch.Generator().manual_seed(0),
    )
    batch = collate(items)
    ptr = batch["ptr"]
    for b in range(3):
        size = int(ptr[b + 1] - ptr[b])
        observed = int(batch["node_mask"][ptr[b] : ptr[b + 1]].sum())
        assert observed >= max(1, round(0.5 * size))
    assert bool(batch["node_mask"][batch["slack_rows"]].all())


def test_masker_draws_are_deterministic_given_the_generator():
    ptr = torch.tensor([0, 7, 19], dtype=torch.int64)
    masker = GraphBatchMasker(0.4)
    a = masker.mask(ptr, generator=torch.Generator().manual_seed(7))
    b = masker.mask(ptr, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# =============================================================================
# Per-unit conversion (the multi-grid normalization decision)
# =============================================================================
def test_per_unit_round_trip_is_the_identity(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    batch = MultiGridCollate(multigrid_source)(items)

    pu = per_unit_state(batch["v"], batch["base_voltage"])
    assert torch.allclose(
        physical_state(pu, batch["base_voltage"]), batch["v"], rtol=1e-6, atol=1e-6
    )
    # per-unit collapses the voltage classes onto one scale
    assert float(pu.abs().max()) < 1.5


def test_per_unit_puts_two_voltage_classes_on_one_scale():
    v = torch.tensor([[400.0 + 0j, 20000.0 + 0j]])
    base = torch.tensor([400.0, 20000.0], dtype=torch.float64)
    pu = per_unit_state(v, base)
    assert torch.allclose(pu.real, torch.ones(1, 2))
    assert pu.dtype == v.dtype


# =============================================================================
# The slack-referenced angle frame (the multi-grid angle decision)
# =============================================================================
def _mismatched_phase_grid():
    """A 2-node grid whose slack node carries phase A only, the far node A and B.

    The B row therefore has no same-phase slack to reference — the fallback case.
    """
    from pgml.schemas.grid_schema import Grid, Line, Node, Phase, Source

    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=0, u_rated_v=400.0, phases=(Phase.A,)),
            Node(id=1, u_rated_v=400.0, phases=(Phase.A, Phase.B)),
        ],
        branches=[
            Line(
                id=10,
                from_node=0,
                to_node=1,
                from_phases=(Phase.A,),
                to_phases=(Phase.A,),
                length_m=100.0,
                series_resistance_ohm_per_m=[[3.0e-4]],
                series_inductance_h_per_m=[[1.0e-6]],
                shunt_capacitance_f_per_m=[[1.0e-11]],
            )
        ],
        appliances=[
            Source(
                id=100,
                node=0,
                phases=(Phase.A,),
                u_ref_v=[230.0],
                u_angle_deg=[0.0],
                resistance_ohm=[[0.01]],
                inductance_h=[[1.0e-4]],
            )
        ],
    )


def test_slack_reference_rows_point_at_the_same_phase_slack(multigrid_source):
    from pgml.assembly import node_phase_index

    from injection_hse.data import MeasurementModel, slack_reference_rows

    for bundle in multigrid_source.grids[:3]:
        rows = slack_reference_rows(bundle.grid)
        slack = MeasurementModel([], slack_known=True).slack_rows(bundle.grid)
        codes = node_phase_index(bundle.grid).phase_codes.to(torch.int64)

        assert rows.dtype == torch.int64 and rows.shape == codes.shape
        assert set(rows.tolist()) <= set(slack.tolist())
        # a fully three-phase feeder references every row against its OWN phase
        assert torch.equal(codes[rows], codes)
        assert torch.equal(rows, bundle.reference_rows)


def test_slack_reference_rows_fall_back_to_the_first_slack():
    from pgml.assembly import node_phase_index

    from injection_hse.data import slack_reference_rows

    grid = _mismatched_phase_grid()
    codes = node_phase_index(grid).phase_codes.to(torch.int64)
    rows = slack_reference_rows(grid)

    assert rows.shape == (3,)  # (0,a), (1,a), (1,b)
    assert rows[:2].tolist() == [0, 0]  # the phase-a rows reference the slack row
    assert int(rows[2]) == 0  # phase b has no slack row -> the FIRST slack row
    assert int(codes[2]) != int(codes[0])


def test_slack_reference_rows_need_a_source():
    from injection_hse.data import slack_reference_rows

    grid = _mismatched_phase_grid()
    with pytest.raises(ValueError, match="no in-service Source"):
        slack_reference_rows(grid.model_copy(update={"appliances": []}))


def _reference_case(seed=0, orders=(1, 3, 5), n=6):
    """A random complex state ``[S, H, N]`` plus a two-phase reference-row vector."""
    from injection_hse.data import slack_referenced_state

    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(4, len(orders), n, dtype=torch.complex128, generator=generator)
    rows = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)[:n]
    return v, rows, list(orders), slack_referenced_state(v, rows, orders=list(orders))


def test_the_frame_zeroes_the_reference_angle_and_keeps_magnitudes():
    v, rows, orders, rotated = _reference_case()
    fundamental = rotated[:, orders.index(1), :]

    # a rotation touches the angle only (magnitude-exact to float64 rounding)
    assert torch.allclose(rotated.abs(), v.abs(), rtol=1e-15, atol=0.0)
    # every row's reference row sits at angle 0 in the new frame
    assert float(fundamental.index_select(-1, rows).angle().abs().max()) < 1e-12
    assert rotated.dtype == v.dtype and rotated.shape == v.shape


def test_the_frame_scales_the_rotation_with_the_harmonic_order():
    """Order ``h`` rotates by ``exp(-1j*h*phi)`` — a shift of the TIME reference."""
    v, rows, orders, rotated = _reference_case()
    phi = v[:, orders.index(1), :].index_select(-1, rows).angle()
    for position, order in enumerate(orders):
        expected = v[:, position, :] * torch.polar(torch.ones_like(phi), -order * phi)
        assert torch.allclose(rotated[:, position, :], expected, atol=1e-12)


def test_the_frame_removes_a_grids_global_angle_convention():
    """THE headline: a grid-wide time-reference shift leaves the frame unchanged."""
    from injection_hse.data import slack_referenced_state

    v, rows, orders, rotated = _reference_case()
    alpha = 0.9871  # this grid's arbitrary global angle convention
    order_axis = torch.tensor(orders, dtype=torch.float64).reshape(-1, 1)
    shift = torch.polar(
        torch.ones(len(orders), v.shape[-1], dtype=torch.float64),
        order_axis * alpha * torch.ones(len(orders), v.shape[-1], dtype=torch.float64),
    )
    shifted = slack_referenced_state(v * shift, rows, orders=orders)
    assert torch.allclose(shifted, rotated, atol=1e-12)

    # a fundamental-only state (the corpora generated so far) is the plain global rotation
    single = v[:, orders.index(1) : orders.index(1) + 1, :]
    turned = single * torch.exp(torch.tensor(1j * alpha, dtype=torch.complex128))
    assert torch.allclose(
        slack_referenced_state(turned, rows, orders=[1]),
        slack_referenced_state(single, rows, orders=[1]),
        atol=1e-12,
    )


def test_the_frame_is_idempotent():
    """Rotating an already-referenced state changes nothing (no double rotation)."""
    from injection_hse.data import slack_referenced_state

    _, rows, orders, rotated = _reference_case()
    assert torch.allclose(
        slack_referenced_state(rotated, rows, orders=orders), rotated, atol=1e-12
    )


def test_a_zero_reference_row_passes_through_unrotated():
    from injection_hse.data import slack_referenced_state

    v, rows, orders, _ = _reference_case()
    v[:, :, 0] = 0.0  # a de-energized reference row
    rotated = slack_referenced_state(v, torch.zeros_like(rows), orders=orders)
    assert bool(torch.isfinite(rotated).all())
    assert torch.equal(rotated, v)


def test_the_frame_honors_dtype_and_needs_the_fundamental():
    from injection_hse.data import slack_referenced_state

    v, rows, orders, _ = _reference_case()
    assert slack_referenced_state(v.to(torch.complex64), rows, orders=orders).dtype == (
        torch.complex64
    )
    with pytest.raises(ValueError, match="FUNDAMENTAL"):
        slack_referenced_state(v, rows, orders=[3, 5, 7])
    with pytest.raises(ValueError, match="reference_rows"):
        slack_referenced_state(v, rows[:2], orders=orders)


def test_collate_maps_the_reference_rows_to_global_indices(multigrid_source):
    items = _items_from_distinct_grids(multigrid_source, count=3)
    batch = MultiGridCollate(multigrid_source)(items)
    bundles = [multigrid_source.bundle(int(i)) for i in batch["grid_index"]]

    expected = torch.cat(
        [b.reference_rows + int(off) for b, off in zip(bundles, batch["ptr"][:-1])]
    )
    assert torch.equal(batch["reference_rows"], expected)
    assert batch["reference_rows"].shape == (int(batch["ptr"][-1]),)
    # a row never references another member's slack
    assert torch.equal(batch["batch"][batch["reference_rows"]], batch["batch"])
    assert set(batch["reference_rows"].tolist()) <= set(batch["slack_rows"].tolist())


def test_per_unit_states_are_fit_in_the_configured_frame(multigrid_source):
    from injection_hse.data import slack_referenced_state

    referenced = multigrid_source.per_unit_states(
        split="train", max_scenarios_per_grid=2
    )  # "slack" is the default
    absolute = multigrid_source.per_unit_states(
        split="train", max_scenarios_per_grid=2, angle_reference="absolute"
    )
    indices = multigrid_source.split_indices("train")
    assert len(referenced) == len(absolute) == len(indices)

    for state, raw, index in zip(referenced, absolute, indices):
        bundle = multigrid_source.bundle(index)
        assert torch.allclose(state.abs(), raw.abs())  # magnitudes are frame-free
        assert not torch.allclose(state, raw)  # the angles are not
        assert torch.allclose(
            state,
            slack_referenced_state(
                raw, bundle.reference_rows, orders=multigrid_source.orders
            ),
        )

    with pytest.raises(ValueError, match="angle_reference"):
        multigrid_source.per_unit_states(split="train", angle_reference="nonsense")


# =============================================================================
# The LV-busbar angle frame (the local reference meter)
# =============================================================================
def _substation_grid(*, tie=False, neutral_feeder=False, isolated=False):
    """MV slack -> MV bus -> TRANSFORMER -> LV busbar (node 2) -> LV feeder (node 3).

    A second transformer further out (node 3 -> node 4) must never win the reference; with
    ``tie`` a THIRD one sits at the same hop count as the first but with a lower branch id
    (the tie-break case). ``neutral_feeder`` gives the LV feeder node a neutral row the
    busbar does not carry (the phase-fallback case), ``isolated`` replaces the MV line with
    an OPEN switch so no transformer is reachable at all.
    """
    from pgml.schemas.grid_schema import (
        Grid,
        Line,
        Node,
        Phase,
        Source,
        Switch,
        Transformer,
        WindingConnection,
    )

    abc = (Phase.A, Phase.B, Phase.C)

    def line(id_, a, b):
        return Line(
            id=id_,
            from_node=a,
            to_node=b,
            from_phases=abc,
            to_phases=abc,
            length_m=100.0,
            series_resistance_ohm_per_m=[
                [3.0e-4 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
            series_inductance_h_per_m=[
                [1.0e-6 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
            shunt_capacitance_f_per_m=[
                [1.0e-11 if i == j else 0.0 for j in range(3)] for i in range(3)
            ],
        )

    def transformer(id_, a, b, *, u_from, u_to):
        return Transformer(
            id=id_,
            from_node=a,
            to_node=b,
            from_phases=abc,
            to_phases=abc,
            s_rated_va=4.0e5,
            u_rated_from_v=u_from,
            u_rated_to_v=u_to,
            from_connection=WindingConnection.DELTA,
            to_connection=WindingConnection.WYE_GROUNDED,
            series_resistance_ohm=0.01,
            series_inductance_h=1.0e-4,
        )

    nodes = [
        Node(id=0, u_rated_v=20.0e3, phases=abc),
        Node(id=1, u_rated_v=20.0e3, phases=abc),
        Node(id=2, u_rated_v=400.0, phases=abc),
        Node(
            id=3,
            u_rated_v=400.0,
            phases=(*abc, Phase.N) if neutral_feeder else abc,
        ),
        Node(id=4, u_rated_v=230.0, phases=abc),
    ]
    branches = [
        Switch(
            id=10,
            from_node=0,
            to_node=1,
            from_phases=abc,
            to_phases=abc,
            default_closed=False,
        )
        if isolated
        else line(10, 0, 1),
        transformer(200, 1, 2, u_from=20.0e3, u_to=400.0),
        line(30, 2, 3),
        transformer(400, 3, 4, u_from=400.0, u_to=230.0),
    ]
    if tie:
        nodes.append(Node(id=5, u_rated_v=400.0, phases=abc))
        branches.append(transformer(100, 1, 5, u_from=20.0e3, u_to=400.0))
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=branches,
        appliances=[
            Source(
                id=1000,
                node=0,
                phases=abc,
                u_ref_v=(11547.0, 11547.0, 11547.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [0.16 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [5.0e-3 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
        ],
    )


def _rows_of_node(grid, node_id):
    from pgml.assembly import node_phase_index

    return torch.nonzero(
        node_phase_index(grid).node_ids.to(torch.int64) == int(node_id), as_tuple=False
    ).flatten()


def test_reference_bus_rows_point_at_the_lv_side_of_the_nearest_transformer():
    from pgml.assembly import node_phase_index

    from injection_hse.data import reference_bus_rows, slack_reference_rows

    grid = _substation_grid()
    codes = node_phase_index(grid).phase_codes.to(torch.int64)
    rows = reference_bus_rows(grid)

    assert rows.dtype == torch.int64 and rows.shape == codes.shape
    # every row references the LV BUSBAR (node 2), at its own phase
    assert set(rows.tolist()) == set(_rows_of_node(grid, 2).tolist())
    assert torch.equal(codes[rows], codes)
    # the more distant transformer (node 3 -> node 4) never supplies the reference
    assert not set(rows.tolist()) & set(_rows_of_node(grid, 4).tolist())
    # and this is a DIFFERENT frame from the slack one — the transformer sits between them
    assert not torch.equal(rows, slack_reference_rows(grid))


def test_reference_bus_rows_break_a_hop_tie_by_branch_id():
    """Two transformers at the same hop count: the lower branch id decides."""
    from injection_hse.data import reference_bus_rows

    grid = _substation_grid(tie=True)
    rows = reference_bus_rows(grid)
    assert set(rows.tolist()) == set(
        _rows_of_node(grid, 5).tolist()
    )  # branch 100 < 200


def test_reference_bus_rows_fall_back_to_the_busbars_first_row():
    from pgml.assembly import node_phase_index

    from injection_hse.data import reference_bus_rows

    grid = _substation_grid(neutral_feeder=True)
    codes = node_phase_index(grid).phase_codes.to(torch.int64)
    busbar = _rows_of_node(grid, 2)
    rows = reference_bus_rows(grid)

    busbar_codes = set(codes[busbar].tolist())
    neutral = [i for i in range(codes.numel()) if int(codes[i]) not in busbar_codes]
    assert neutral, "the fixture must carry a phase the busbar does not"
    for row in neutral:
        assert int(rows[row]) == int(busbar[0])  # the busbar's FIRST row


def test_reference_bus_rows_equal_the_slack_rows_without_a_transformer(
    multigrid_source,
):
    """A grid with no in-service transformer has no local busbar: the two frames coincide."""
    from injection_hse.data import reference_bus_rows, slack_reference_rows

    for bundle in multigrid_source.grids[:3]:
        assert torch.equal(
            reference_bus_rows(bundle.grid), slack_reference_rows(bundle.grid)
        )
    grid = _mismatched_phase_grid()
    assert torch.equal(reference_bus_rows(grid), slack_reference_rows(grid))
    # an OPEN switch carries no path, so a transformer behind one is not reachable either
    isolated = _substation_grid(isolated=True)
    assert torch.equal(reference_bus_rows(isolated), slack_reference_rows(isolated))


def test_reference_bus_rows_need_a_source():
    from injection_hse.data import reference_bus_rows

    grid = _substation_grid()
    with pytest.raises(ValueError, match="no in-service Source"):
        reference_bus_rows(grid.model_copy(update={"appliances": []}))


def _clocked_state(grid, orders, *, shift_rad, seed=0):
    """A random state ``[S, H, N]`` plus a copy whose LV island is time-shifted.

    The shift is what a different transformer model would put between the slack and the LV
    network (order ``h`` rotated by ``exp(1j*h*alpha)``) — a modelling decision ABOVE the LV
    busbar, which the LV-busbar frame must not see.
    """
    from pgml.assembly import node_phase_index

    generator = torch.Generator().manual_seed(seed)
    n = int(node_phase_index(grid).size)
    v = torch.randn(3, len(orders), n, dtype=torch.complex128, generator=generator)
    lv = node_phase_index(grid).node_ids.to(torch.int64) >= 2
    turn = torch.exp(
        1j * shift_rad * torch.tensor(orders, dtype=torch.float64).reshape(-1, 1)
    )
    clocked = v.clone()
    clocked[..., lv] = clocked[..., lv] * turn.to(v.dtype)
    return v, clocked, lv


def test_the_lv_busbar_frame_is_blind_to_the_transformer_above_it():
    """THE motivation: the LV state in this frame does not move with the LV island's clock."""
    from injection_hse.data import (
        reference_bus_rows,
        slack_reference_rows,
        slack_referenced_state,
    )

    orders = [1, 5]
    grid = _substation_grid()
    v, clocked, lv = _clocked_state(grid, orders, shift_rad=-2.618)  # a -150 deg clock
    busbar = reference_bus_rows(grid)
    slack = slack_reference_rows(grid)

    referenced = slack_referenced_state(v, busbar, orders=orders)
    moved = slack_referenced_state(clocked, busbar, orders=orders)
    assert torch.allclose(moved[..., lv], referenced[..., lv], atol=1e-12)
    # the slack frame keeps the clock in exactly those rows
    assert not torch.allclose(
        slack_referenced_state(clocked, slack, orders=orders)[..., lv],
        slack_referenced_state(v, slack, orders=orders)[..., lv],
    )


def test_the_lv_busbar_frame_still_removes_a_grids_global_convention():
    """The frame property is unchanged: only the reference bus moved."""
    from pgml.assembly import node_phase_index

    from injection_hse.data import reference_bus_rows, slack_referenced_state

    orders = [1, 5]
    grid = _substation_grid()
    v, _, _ = _clocked_state(grid, orders, shift_rad=0.0, seed=1)
    rows = reference_bus_rows(grid)
    alpha = 0.9871  # this grid's arbitrary global angle convention
    shift = torch.exp(
        1j * alpha * torch.tensor(orders, dtype=torch.float64).reshape(-1, 1)
    ).to(v.dtype)

    referenced = slack_referenced_state(v, rows, orders=orders)
    assert torch.allclose(
        slack_referenced_state(v * shift, rows, orders=orders), referenced, atol=1e-12
    )
    assert torch.allclose(referenced.abs(), v.abs(), rtol=1e-15, atol=0.0)
    # the LV busbar itself sits at angle 0 (a meter reads its own reference as zero)
    fundamental = referenced[:, orders.index(1), :]
    busbar = _rows_of_node(grid, 2)
    assert float(fundamental.index_select(-1, busbar).angle().abs().max()) < 1e-12
    assert int(node_phase_index(grid).node_ids[int(rows[0])]) == 2


def test_the_source_carries_the_configured_frames_rows(
    multigrid_root, multigrid_source
):
    from injection_hse.data import reference_bus_rows

    source = MultiGridDataSource(multigrid_root, angle_reference="lv_busbar")
    assert source.angle_reference == "lv_busbar"
    assert multigrid_source.angle_reference == "slack"
    for bundle in source.grids:
        assert torch.equal(bundle.reference_rows, reference_bus_rows(bundle.grid))

    items = _items_from_distinct_grids(source, count=2)
    batch = MultiGridCollate(source)(items)
    expected = torch.cat(
        [
            source.bundle(int(g)).reference_rows + int(off)
            for g, off in zip(batch["grid_index"], batch["ptr"][:-1])
        ]
    )
    assert torch.equal(batch["reference_rows"], expected)


def test_an_unknown_angle_frame_is_rejected(multigrid_root):
    with pytest.raises(ValueError, match="angle_reference"):
        MultiGridDataSource(multigrid_root, angle_reference="nonsense")


def test_per_unit_states_refuse_a_foreign_row_frame(multigrid_source):
    """The bundles carry ONE frame's rows; a different one would rotate against them."""
    with pytest.raises(ValueError, match="lv_busbar"):
        multigrid_source.per_unit_states(split="train", angle_reference="lv_busbar")


# =============================================================================
# Config wiring
# =============================================================================
def test_build_data_source_returns_a_multigrid_source(multigrid_root):
    from injection_hse.config import DataConfig, MultiGridConfig, StructuralEncodingConfig
    from injection_hse.data import build_data_source

    cfg = DataConfig(
        dataset_path=str(multigrid_root),
        multigrid=MultiGridConfig(max_grids=2),
        structural_encoding=StructuralEncodingConfig(kind="rwse", rwse_steps=3),
    )
    source = build_data_source(cfg)  # no grid: the members carry their own

    assert isinstance(source, MultiGridDataSource)
    assert len(source.grids) <= 3 * 2  # capped per split
    assert source.grids[0].graph.node_encoding_dim == 3
    assert source.per_unit is True
    assert source.angle_reference == "slack"  # the default frame


def test_the_angle_frame_round_trips_from_the_config_into_the_source(multigrid_root):
    from pydantic import ValidationError

    from injection_hse.config import DataConfig, MultiGridConfig
    from injection_hse.data import build_data_source

    cfg = DataConfig(
        dataset_path=str(multigrid_root),
        multigrid=MultiGridConfig(angle_reference="lv_busbar", max_grids=1),
    )
    assert DataConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert build_data_source(cfg).angle_reference == "lv_busbar"
    with pytest.raises(ValidationError):
        MultiGridConfig(angle_reference="lv-busbar")


# =============================================================================
# ding0 district selection
# =============================================================================
def test_ding0_district_whitelist_limits_the_enumeration(monkeypatch):
    """`districts` selects exactly the named districts; `exclude_districts` still wins."""
    import sys
    import types

    from pgml.grids import synthetic_feeder

    from injection_hse.data.multigrid import Ding0Source, _collect_ding0

    class _Record:
        def __init__(self, grid_id):
            self.grid_id = grid_id
            self.source = f"/nowhere/{grid_id}"

    walked: list[str] = []

    def _iter_lv_grids(path):
        walked.append(path.name)
        yield "lv1", synthetic_feeder(4)
        yield "lv2", synthetic_feeder(5)

    fake = types.ModuleType("pghub.loaders.ding0")
    fake.iter_lv_grids = _iter_lv_grids
    monkeypatch.setitem(sys.modules, "pghub.loaders.ding0", fake)
    monkeypatch.setattr(
        "injection_hse.data.multigrid._hub_records",
        lambda dataset, data_root, **kw: [_Record(g) for g in ("a", "b", "c", "d")],
    )

    picked = _collect_ding0(
        Ding0Source(districts=("b", "c"), exclude_districts=("c",), max_per_district=1)
    )
    assert [c.source for c in picked] == ["ding0:b_lv1"]
    assert walked == ["b"]  # non-whitelisted districts are never opened
