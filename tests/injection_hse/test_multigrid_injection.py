"""The multi-grid INJECTION path: the operator bank, the device axes, the per-unit currents.

A multi-grid batch has no single ``Y(h)`` and no single device layout, so serving the
injection decode / the localization head across grids needs two things:

- :class:`injection_hse.physics.MultiGridPhysicsBank` — one network solver per member, index-aligned
  with the source's bundles, plus the per-unit CURRENT base that makes injections comparable
  across voltage classes;
- the DEVICE axis of :class:`injection_hse.data.MultiGridCollate` — the per-device target concatenated
  the way the node axis is, with its own ``ptr``/``batch`` and GLOBAL device rows.

The load-bearing properties checked here: the bank's per-order operator scale is the SAME
quantity measurement anchoring already used (it was factored out, not recomputed differently),
the device bookkeeping reproduces the per-member items exactly, a batch without the injection
flag is byte-identical to what a voltage-only run has always seen, and the injection targets
are formed in the ABSOLUTE frame — ``Y`` does not commute with the slack-frame rotation.
"""

from __future__ import annotations

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
    NoiseConfig,
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.data import (
    MultiGridCollate,
    MultiGridDataSource,
    slack_frame_rotation,
    slack_referenced_state,
)
from injection_hse.data.multigrid import (
    GRID_FILE,
    GridEntry,
    MultiGridGenerationConfig,
    MultiGridManifest,
    SyntheticFeederSource,
    dataset_dir,
    generate_multigrid_dataset,
)
from injection_hse.encoding import CartesianEncoding, TrigEncoding
from injection_hse.masking import build_masker
from injection_hse.physics import (
    MultiGridPhysicsBank,
    NetworkSolver,
    fit_multigrid_edge_normalizer,
    fit_multigrid_injection_normalizer,
)
from injection_hse.train import SEModule, fit_multigrid_normalizer

ORDERS = (1, 3, 5)
N_GRIDS = 3
N_SCENARIOS = 3
#: key of the hand-registered member whose scenarios carry NO harmonic injection
QUIET_KEY = "synthetic_900_quiet"


# =============================================================================
# Fixtures
# =============================================================================
def _register_injection_free_member(root) -> None:
    """Add a member whose dataset drives no per-device harmonic injection.

    The generator's shared scenario template gives every member a per-load current spectrum,
    so an empty :class:`~injection_hse.data.InjectionSpec` (``D_b = 0``) has to be built explicitly: the
    same solve at the same orders, with only the load-scale parameter. It is the member the
    device-axis bookkeeping must survive.
    """
    from pgml.grids import synthetic_feeder
    from pgml.scenarios import (
        ParameterSpec,
        ScenarioConfig,
        Selector,
        Uniform,
        run_scenarios,
        write_dataset,
    )
    from pgml.topology import layout_fingerprint

    grid = synthetic_feeder(5, n_feeders=2)
    out_dir = dataset_dir(root, QUIET_KEY)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / GRID_FILE).write_text(grid.model_dump_json(), encoding="utf-8")
    result = run_scenarios(
        grid,
        ScenarioConfig(
            n_samples=N_SCENARIOS,
            seed=7,
            parameters=[
                ParameterSpec(
                    name="load_scale",
                    selector=Selector(component="load"),
                    distribution=Uniform(low=0.5, high=1.0),
                    field="pq",
                    mode="scale",
                    per="each",
                )
            ],
        ),
        calculation="harmonic",
        harmonic_orders=list(ORDERS),
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )
    write_dataset(result, out_dir)

    manifest = MultiGridManifest.read(root)
    entry = GridEntry(
        key=QUIET_KEY,
        split="train",
        source="synthetic:5",
        kind="synthetic",
        n_nodes=len(grid.nodes),
        n_rows=int(result.v.shape[-1]),
        n_scenarios=int(result.v.shape[0]),
        layout_fingerprint=layout_fingerprint(grid),
    )
    manifest.model_copy(update={"grids": [*manifest.grids, entry]}).write(root)


@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    """A tiny multi-grid dataset WITH harmonic injection, plus one injection-free member."""
    root = tmp_path_factory.mktemp("multigrid_injection")
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(root),
            n_scenarios=N_SCENARIOS,
            orders=ORDERS,
            val_fraction=0.3,
            sources=[
                SyntheticFeederSource(n_grids=N_GRIDS, node_range=(4, 6), n_feeders=2)
            ],
        )
    )
    _register_injection_free_member(root)
    return root


@pytest.fixture(scope="module")
def source(multigrid_root):
    return MultiGridDataSource(
        multigrid_root, placement_fraction=0.3, with_injection=True
    )


@pytest.fixture(scope="module")
def bank(source):
    # complex128 so the frame algebra is checked at the solve precision, not float32 noise
    return MultiGridPhysicsBank(source, dtype=torch.complex128)


def _items(source, keys):
    """One item per named member (its first scenario), in the given order."""
    dataset = source.train_dataset()
    wanted = {source.bundle(i).key: i for i in range(len(source.grids))}
    picked: dict[int, dict] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        picked.setdefault(int(item["grid"]), item)
    return [picked[wanted[key]] for key in keys]


def _injecting_keys(source, n=2):
    return [b.key for b in source.grids if b.n_devices][:n]


# =============================================================================
# The bank
# =============================================================================
def test_bank_is_index_aligned_with_the_members(source, bank):
    assert len(bank) == len(source.grids)
    assert bank.orders == list(source.orders)
    for i, bundle in enumerate(source.grids):
        solver = bank.solver(i)
        assert solver is bank.solver(i)  # cached, built once
        assert int(solver.y_fundamental.shape[-1]) == bundle.n_nodes
        assert torch.equal(solver.slack_rows, bundle.slack_rows.to(torch.int64))


def test_operator_scale_equals_the_anchoring_computation(source, bank):
    """The factored scale is the SAME quantity the anchored solve used before.

    ``_assemble_anchoring`` computed the median singular value of the SOLVED operator inline;
    it now reads :meth:`NetworkSolver.operator_scale`, so both the reference computation and
    an anchoring solver must agree with it exactly.
    """
    solver = bank.solver(0)
    scale = solver.operator_scale()
    assert scale.shape == (len(bank.orders),)
    assert scale.dtype == torch.float64
    assert bool((scale > 0).all())

    expected = []
    hi = 0
    for h in solver.orders:
        if h == 1:
            free = solver._free_rows
            y = solver.y_fundamental[0].index_select(0, free).index_select(1, free)
        else:
            y = solver.y_harmonic[hi]
            hi += 1
        expected.append(torch.linalg.svdvals(y).median().to(torch.float64))
    assert torch.equal(scale, torch.stack(expected))

    anchored = NetworkSolver.from_grid(
        source.bundle(0).grid,
        bank.orders,
        dtype=torch.complex128,
        anchor_v_weight=1.0,
    )
    assert torch.equal(anchored._v_scale, scale)


def test_reassembly_refreshes_the_cached_scale(bank):
    """A re-assembled operator invalidates the cache (the calibration path re-reads it)."""
    solver = bank.solver(0)
    first = solver.operator_scale()
    solver.assemble()
    assert solver._v_scale is None
    assert torch.equal(solver.operator_scale(), first)


def test_i_base_is_the_member_power_base_over_its_row_voltages(source, bank):
    """``i_base[h,n] = s(h)·V_ref² / V_base[n]`` per member, with that member's OWN reference
    voltage — a power base per row voltage, so the base scales inversely with the class."""
    from injection_hse.physics import reference_base_voltage

    for i, bundle in enumerate(source.grids):
        base = bank.i_base(i)
        assert base.shape == (len(bank.orders), bundle.n_nodes)
        assert base.dtype == torch.float64
        assert bool((base > 0).all())
        assert not base.requires_grad
        v_base = bundle.base_voltage.to(torch.float64)
        v_ref = reference_base_voltage(v_base)
        scale = bank.solver(i).operator_scale().unsqueeze(-1)
        assert torch.allclose(base, scale * v_ref**2 / v_base)
        # the bank keeps no second formula: it is the single-grid solver's own base
        assert torch.equal(base, bank.solver(i).i_base(bundle.base_voltage))


def test_device_current_base_gathers_the_first_valid_phase_row(source, bank):
    key = _injecting_keys(source, 1)[0]
    i = next(j for j in range(len(source.grids)) if source.grids[j].key == key)
    spec = source.bundle(i).injection_spec
    base = bank.device_current_base(i)
    assert base.shape == (spec.n_devices, len(source.injection_orders))
    pos = [bank.orders.index(o) for o in spec.orders]
    for d in range(spec.n_devices):
        row = int(spec.device_rows[d, 0])
        for k, p in enumerate(pos):
            assert float(base[d, k]) == pytest.approx(float(bank.i_base(i)[p, row]))


def test_an_injection_free_member_has_no_device_base(source, bank):
    i = next(j for j in range(len(source.grids)) if source.grids[j].key == QUIET_KEY)
    assert source.bundle(i).injection_spec.is_empty
    assert source.bundle(i).n_devices == 0
    assert bank.device_current_base(i).shape == (0, len(source.injection_orders))


def test_the_current_operator_is_built_only_on_demand(source):
    """The heaviest per-member object is never built by a run that does not meter currents."""
    lazy = MultiGridPhysicsBank(source, dtype=torch.complex128)
    lazy.solver(0)
    lazy.i_base(0)
    assert all(op is None for op in lazy._edge_ops)
    op = lazy.current_operator(0)
    assert op is lazy.current_operator(0)  # cached
    assert op.n_edges == source.bundle(0).n_edges
    assert all(op is None for op in lazy._edge_ops[1:])


def test_bank_to_moves_built_operators(source, bank):
    """``to`` is a no-op onto the device the operators are already on (the CPU guard path)."""
    assert bank.to(torch.device("cpu")) is bank
    assert bank.device == torch.device("cpu")
    assert bank.solver(0).slack_rows.device == torch.device("cpu")
    assert bank.i_base(0).device == torch.device("cpu")


# =============================================================================
# The collate's device axis
# =============================================================================
def test_device_axes_bookkeeping(source, bank):
    keys = [b.key for b in source.grids][:3]
    items = _items(source, keys)
    bundles = [source.bundle(int(item["grid"])) for item in items]
    batch = MultiGridCollate(source, with_injection=True, bank=bank)(items)

    counts = [b.n_devices for b in bundles]
    n_dev = sum(counts)
    n_orders = len(source.injection_orders)
    assert batch["device_ptr"].tolist() == [
        sum(counts[:k]) for k in range(len(counts) + 1)
    ]
    assert batch["device_batch"].tolist() == [
        b for b, c in enumerate(counts) for _ in range(c)
    ]
    assert batch["injection_target"].shape == (n_dev, n_orders, 1)
    assert batch["injection_target"].dtype == torch.float32
    assert batch["device_rows"].shape[0] == n_dev
    assert batch["device_rows_mask"].shape == batch["device_rows"].shape

    # GLOBAL rows: member-local + the item's node offset, pad slots -1
    for b, bundle in enumerate(bundles):
        lo, hi = int(batch["device_ptr"][b]), int(batch["device_ptr"][b + 1])
        offset = int(batch["ptr"][b])
        rows = batch["device_rows"][lo:hi]
        mask = batch["device_rows_mask"][lo:hi]
        if bundle.n_devices == 0:
            assert lo == hi
            continue
        width = int(bundle.device_rows.shape[1])
        assert torch.equal(mask[:, :width], bundle.device_rows_mask)
        assert torch.equal(
            rows[:, :width][mask[:, :width]],
            bundle.device_rows[bundle.device_rows_mask] + offset,
        )
        assert bool((rows[~mask] == -1).all())
    assert bool(
        (batch["device_rows"][batch["device_rows_mask"]] < int(batch["ptr"][-1])).all()
    )


def test_an_injection_free_member_contributes_no_devices(source, bank):
    keys = [*_injecting_keys(source, 2), QUIET_KEY]
    items = _items(source, keys)
    batch = MultiGridCollate(source, with_injection=True, bank=bank)(items)
    counts = (batch["device_ptr"][1:] - batch["device_ptr"][:-1]).tolist()
    assert counts[-1] == 0
    assert sum(counts) == int(batch["injection_target"].shape[-3])
    assert int(batch["device_batch"].max()) == len(items) - 2  # no device tagged to it


def test_the_collated_target_is_in_the_per_unit_current_base(source, bank):
    keys = _injecting_keys(source, 2)
    items = _items(source, keys)
    batch = MultiGridCollate(source, with_injection=True, bank=bank)(items)
    start = 0
    for item in items:
        g = int(item["grid"])
        raw = item["injection_target"]  # amperes, as persisted
        n_dev = raw.shape[-3]
        collated = batch["injection_target"][start : start + n_dev]
        base = bank.device_current_base(g).to(torch.float32).unsqueeze(-1)
        assert torch.allclose(collated * base, raw, rtol=1e-5, atol=0.0)
        assert not torch.allclose(collated, raw)  # the scaling is not a no-op
        start += n_dev


def test_injection_axes_need_the_bank(source):
    with pytest.raises(ValueError, match="per-unit current base"):
        MultiGridCollate(source, with_injection=True)


def test_the_source_collate_follows_the_attached_bank(multigrid_root, bank):
    fresh = MultiGridDataSource(multigrid_root, placement_fraction=0.3)
    assert fresh.collate().with_injection is False
    attached = MultiGridDataSource(multigrid_root, placement_fraction=0.3)
    attached.attach_physics_bank(MultiGridPhysicsBank(attached))
    assert attached.collate().with_injection is True


def test_disagreeing_injected_orders_are_rejected(source, monkeypatch):
    """One head needs ONE order axis; a member injecting elsewhere must fail loudly."""

    class _OtherOrders:
        is_empty = False
        orders = [7]

    bundles = source.grids
    target = next(b for b in bundles if b.n_devices)
    patched = [
        b
        if b is not target
        else b.__class__(**{**b.__dict__, "injection_spec": _OtherOrders()})
        for b in bundles
    ]
    monkeypatch.setattr(source, "_bundles", patched)
    monkeypatch.setattr(source, "_injection_orders", None)
    with pytest.raises(ValueError, match="one order axis"):
        source.injection_orders


# =============================================================================
# The injection normalizer (and its frame)
# =============================================================================
def _config(root, **overrides):
    cfg = dict(
        name="multigrid-injection",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
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
        train=TrainConfig(accelerator="cpu", devices=1, enable_progress_bar=False),
    )
    cfg.update(overrides)
    return ExperimentConfig(**cfg)


def test_injection_normalizer_is_finite_and_per_unit(source, bank, multigrid_root):
    normalizer = fit_multigrid_injection_normalizer(
        bank, source, TrigEncoding(), _config(multigrid_root)
    )
    assert normalizer.mean.shape == (len(ORDERS), 3)
    assert torch.isfinite(normalizer.mean).all()
    assert torch.isfinite(normalizer.std).all()
    assert bool((normalizer.std > 0).all())
    assert torch.isfinite(normalizer.cartesian_scale).all()


def test_injection_targets_are_formed_in_the_absolute_frame(source, bank):
    """``Y`` does not commute with the per-row slack rotation — the order matters.

    The targets must be ``Y·V_abs`` and the RESULT rotated, not ``Y·(D V_abs)``.
    """
    i = next(j for j in range(len(source.grids)) if source.grids[j].n_devices)
    bundle = source.bundle(i)
    solver = bank.solver(i)
    rows = bundle.reference_rows
    v_abs = bundle.source.states[:2].to(torch.complex128)

    i_abs = solver.injection_targets(v_abs)
    rotated_current = slack_referenced_state(i_abs, rows, orders=bank.orders)
    from_rotated_state = solver.injection_targets(
        slack_referenced_state(v_abs, rows, orders=bank.orders)
    )

    # the two orders of operation are genuinely different (a row-dependent diagonal)
    assert len(set(rows.tolist())) > 1
    assert not torch.allclose(rotated_current, from_rotated_state)
    # magnitudes are rotation-invariant either way; the phase is not
    assert torch.allclose(rotated_current.abs(), i_abs.abs())

    # ... and the normalizer fit uses the first (the CURRENT is rotated, not the state)
    encoding = TrigEncoding()
    expected = encoding.encode(
        rotated_current / bank.i_base(i).to(torch.float64)
    ).transpose(-3, -2)
    fitted = _fit_one_member(bank, source, i, encoding)
    assert torch.allclose(fitted, expected)


def _fit_one_member(bank, source, index, encoding):
    """The per-member feature block the multi-grid injection fit builds (first 2 scenarios)."""
    bundle = source.bundle(index)
    i_target = bank.solver(index).injection_targets(bundle.source.states[:2])
    i_pu = i_target / bank.i_base(index).to(i_target.real.dtype)
    return encoding.encode(
        slack_referenced_state(i_pu, bundle.reference_rows, orders=bank.orders)
    ).transpose(-3, -2)


def _absolute_config(root):
    return _config(
        root,
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(
                placement_fraction=0.3, angle_reference="absolute"
            ),
            batch_size=2,
            num_workers=0,
        ),
    )


def test_the_angle_frame_reaches_the_injection_fit(source, bank, multigrid_root):
    """``angle_reference`` is a real choice here: the Cartesian statistics move with it."""
    cartesian = CartesianEncoding()
    slack = fit_multigrid_injection_normalizer(
        bank, source, cartesian, _config(multigrid_root)
    )
    absolute = fit_multigrid_injection_normalizer(
        bank, source, cartesian, _absolute_config(multigrid_root)
    )
    assert not torch.allclose(slack.mean, absolute.mean)


def test_the_frame_leaves_the_injection_magnitudes_alone(source, bank, multigrid_root):
    """The rotation has unit modulus, so the magnitude statistics are frame-free."""
    trig = TrigEncoding()
    slack = fit_multigrid_injection_normalizer(
        bank, source, trig, _config(multigrid_root)
    )
    absolute = fit_multigrid_injection_normalizer(
        bank, source, trig, _absolute_config(multigrid_root)
    )
    assert torch.allclose(slack.mean[:, 0], absolute.mean[:, 0])
    assert torch.allclose(slack.cartesian_scale, absolute.cartesian_scale)


# =============================================================================
# Config guards
# =============================================================================
def test_the_physics_rejections_are_lifted(multigrid_root):
    """The per-member bank serves these; the validator must no longer refuse them."""
    assert (
        _config(
            multigrid_root,
            physics=PhysicsConfig(decode="injection"),
            loss=LossConfig(predict_injection=True, injection_state_loss_split=0.5),
            measurement=MeasurementConfig(measure_current=True, slack_known=True),
        ).physics.decode
        == "injection"
    )


def test_multigrid_accepts_sensor_noise(multigrid_root):
    """Noise is implemented on this path now (uniform default-class model), so the config
    must build — the forward applies it to the model inputs only."""
    cfg = _config(multigrid_root, noise=NoiseConfig(enabled=True))
    assert cfg.noise.enabled


def test_the_remaining_rejections_still_fire(multigrid_root):
    with pytest.raises(ValueError, match="heterogeneous_orders"):
        _config(
            multigrid_root,
            measurement=MeasurementConfig(heterogeneous_orders=True),
        )
    with pytest.raises(ValueError, match="dnn"):
        _config(multigrid_root, model=DNNConfig(hidden_dims=[8]))


def test_the_injection_decode_needs_a_metered_slack(multigrid_root):
    with pytest.raises(ValueError, match="slack_known"):
        _config(
            multigrid_root,
            physics=PhysicsConfig(decode="injection"),
            measurement=MeasurementConfig(slack_known=False),
        )


def test_calibration_across_a_bank_is_rejected(multigrid_root):
    from injection_hse.config import CalibrationKnobConfig

    with pytest.raises(ValueError, match="calibrate"):
        _config(
            multigrid_root,
            physics=PhysicsConfig(
                decode="injection",
                calibrate=[
                    CalibrationKnobConfig(
                        target="line", id=10001, field="series_resistance_ohm_per_m"
                    )
                ],
            ),
        )


# =============================================================================
# The per-member physics decode
# =============================================================================
def _fresh_bank(source, *, anchor_v=0.0, anchor_i=0.0):
    """A bank of its own, so one test's anchor schedule cannot leak into another."""
    return MultiGridPhysicsBank(
        source,
        dtype=torch.complex128,
        anchor_v_weight=anchor_v,
        anchor_i_weight=anchor_i,
    )


def _mixed_batch(source, bank, *, with_injection=True):
    """A batch INTERLEAVING two members, so the per-member gather/scatter really permutes."""
    dataset = source.train_dataset()
    by_grid: dict[int, list[dict]] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        by_grid.setdefault(int(item["grid"]), []).append(item)
    injecting = [g for g in sorted(by_grid) if source.bundle(g).n_devices]
    assert len(injecting) >= 2, "the fixture must provide two injecting train members"
    assert len(by_grid[injecting[0]]) >= 2, "one member must contribute two scenarios"
    a, b = injecting[0], injecting[1]
    items = [by_grid[a][0], by_grid[b][0], by_grid[a][1]]
    collate = MultiGridCollate(source, with_injection=with_injection, bank=bank)
    return collate(items)


def _injection_module(root, source, bank, *, masker="random", **overrides):
    """An injection-decode :class:`SEModule` over the bank, at the solve precision."""
    overrides.setdefault("physics", PhysicsConfig(decode="injection"))
    overrides.setdefault(
        "train",
        TrainConfig(
            accelerator="cpu",
            devices=1,
            dtype="complex128",
            enable_progress_bar=False,
        ),
    )
    cfg = _config(root, **overrides)
    encoding = TrigEncoding()
    kwargs: dict = {}
    if cfg.physics.decode == "injection":
        kwargs["inj_normalizer"] = fit_multigrid_injection_normalizer(
            bank, source, encoding, cfg
        )
    if cfg.loss.predict_injection:
        kwargs["injection_orders"] = source.injection_orders
    if cfg.measurement.measure_current:
        kwargs["edge_normalizer"] = fit_multigrid_edge_normalizer(
            bank, source, encoding, cfg
        )
        if cfg.measurement.measure_power:
            kwargs["power_normalizer"] = fit_multigrid_edge_normalizer(
                bank, source, encoding, cfg, quantity="power"
            )
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, encoding, cfg),
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        **kwargs,
    )
    module.set_masker(build_masker(masker, observed_fraction=0.5))
    return module


def _random_injection(batch, *, seed=0):
    """A deterministic complex injection current over the batch's node axis ``[H, N*]``."""
    generator = torch.Generator().manual_seed(seed)
    shape = (len(ORDERS), int(batch["ptr"][-1]))
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )


def _reference_decode(bank, batch, i_pred, v_abs, *, mask=None, i_edges=None):
    """The oracle: solve every ITEM alone with its own member solver, then concatenate."""
    fundamental = ORDERS.index(1)
    ptr = batch["ptr"].tolist()
    edge_ptr = batch["edge_ptr"].tolist()
    outs = []
    for item, index in enumerate(batch["grid_index"].tolist()):
        lo, hi = ptr[item], ptr[item + 1]
        solver = bank.solver(int(index))
        i_seg = i_pred[..., lo:hi].unsqueeze(0)  # [1, H, N_g]
        v_seg = v_abs[..., lo:hi].unsqueeze(0)
        v_slack = (
            v_abs.select(-2, fundamental)[..., lo:hi]
            .index_select(-1, solver.slack_rows)
            .unsqueeze(0)
        )
        if solver.anchoring:
            e_lo, e_hi = edge_ptr[item], edge_ptr[item + 1]
            out = solver.solve(
                i_seg,
                v_slack,
                v_meas=v_seg,
                v_mask=None if mask is None else mask[..., lo:hi].unsqueeze(0),
                i_meas=None
                if i_edges is None
                else i_edges[..., e_lo:e_hi].unsqueeze(0),
                edge_mask=(
                    None
                    if i_edges is None or mask is None
                    else mask.index_select(-1, batch["edge_index"][0])[
                        ..., e_lo:e_hi
                    ].unsqueeze(0)
                ),
            )
        else:
            out = solver.solve(i_seg, v_slack)
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=-1)


def test_multigrid_decode_equals_each_members_own_solve(source, multigrid_root):
    """The dispatch is a bookkeeping device: batched == every member solved alone."""
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    i_pred = _random_injection(batch)
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)

    decoded = module._multigrid_decode(i_pred, v_abs, mask, batch)
    expected = _reference_decode(bank, batch, i_pred, v_abs)
    assert decoded.shape == expected.shape
    assert torch.allclose(
        decoded, expected, rtol=0.0, atol=1e-12 * expected.abs().max()
    )

    # the batch really does interleave two members (so the scatter permutation is exercised)
    assert batch["grid_index"].tolist() != sorted(batch["grid_index"].tolist())


def test_multigrid_decode_matches_the_voltage_anchored_member_solve(
    source, multigrid_root
):
    bank = _fresh_bank(source, anchor_v=1.0)
    module = _injection_module(multigrid_root, source, bank)
    module.set_stage_anchor(v_weight=2.0, i_weight=0.0)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    i_pred = _random_injection(batch, seed=1)
    mask = torch.zeros(int(batch["ptr"][-1]), dtype=torch.bool)
    mask[::3] = True
    mask[batch["slack_rows"]] = True

    decoded = module._multigrid_decode(i_pred, v_abs, mask, batch)
    expected = _reference_decode(bank, batch, i_pred, v_abs, mask=mask)
    assert torch.allclose(
        decoded, expected, rtol=0.0, atol=1e-12 * expected.abs().max()
    )
    # the anchor genuinely moved the state away from the slack-only solve
    plain = _fresh_bank(source)
    assert not torch.allclose(decoded, _reference_decode(plain, batch, i_pred, v_abs))


def test_multigrid_decode_matches_the_current_anchored_member_solve(
    source, multigrid_root
):
    bank = _fresh_bank(source, anchor_v=1.0, anchor_i=1.0)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        measurement=MeasurementConfig(measure_current=True, slack_known=True),
    )
    module.set_stage_anchor(v_weight=1.0, i_weight=0.5)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    i_pred = _random_injection(batch, seed=2)
    mask = torch.zeros(int(batch["ptr"][-1]), dtype=torch.bool)
    mask[::2] = True
    mask[batch["slack_rows"]] = True

    segments = module._multigrid_segments(batch)
    i_base = module._multigrid_i_base(batch).to(torch.float64)
    _, edge_mask, i_edges = module._multigrid_edge_observations(
        v_abs, None, mask, batch, segments, i_base, v_abs.device
    )
    decoded = module._multigrid_decode(
        i_pred, v_abs, mask, batch, i_edges=i_edges, edge_mask=edge_mask
    )
    expected = _reference_decode(bank, batch, i_pred, v_abs, mask=mask, i_edges=i_edges)
    assert torch.allclose(
        decoded, expected, rtol=0.0, atol=1e-10 * expected.abs().max()
    )


def test_the_decoded_state_satisfies_the_member_network_law(source, multigrid_root):
    """Physics by CONSTRUCTION, per member segment, in the ABSOLUTE frame."""
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    i_pred = _random_injection(batch, seed=3)
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)
    decoded = module._multigrid_decode(i_pred, v_abs, mask, batch)

    ptr = batch["ptr"].tolist()
    for item, index in enumerate(batch["grid_index"].tolist()):
        lo, hi = ptr[item], ptr[item + 1]
        solver = bank.solver(int(index))
        v_seg = decoded[..., lo:hi]
        i_seg = i_pred[..., lo:hi]
        hi_pos = 0
        for pos, order in enumerate(ORDERS):
            if order == 1:
                y = solver.y_fundamental[0]
                # the slack rows are held at the measurement; the FREE rows carry the law
                free = solver._free_rows
                residual = (y @ v_seg[pos])[free] - i_seg[pos][free]
            else:
                y = solver.y_harmonic[hi_pos]
                hi_pos += 1
                residual = y @ v_seg[pos] - i_seg[pos]
            scale = (y.abs().max() * v_seg[pos].abs().max()).clamp_min(1.0)
            assert float(residual.abs().max() / scale) < 1e-10

        # the fundamental slack rows reproduce the measured boundary exactly
        fundamental = ORDERS.index(1)
        slack = solver.slack_rows
        assert torch.allclose(
            v_seg[fundamental].index_select(-1, slack),
            v_abs[fundamental, lo:hi].index_select(-1, slack),
        )


# =============================================================================
# The angle frame around the decode
# =============================================================================
def test_the_decode_is_run_in_the_absolute_frame(source, multigrid_root):
    """Rotating the ANSWER of the absolute decode == decoding unrotated slack-frame inputs.

    And the frame-blind alternative — handing the rotated injections straight to ``Y`` — is a
    different state, which is exactly why the step unrotates first.
    """
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    rotation = slack_frame_rotation(v_abs, batch["reference_rows"], orders=list(ORDERS))
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)
    i_abs = _random_injection(batch, seed=4)
    i_slack = i_abs * rotation  # what the model predicts, in its own frame

    from_absolute = module._multigrid_decode(i_abs, v_abs, mask, batch) * rotation
    from_slack = (
        module._multigrid_decode(i_slack * rotation.conj(), v_abs, mask, batch)
        * rotation
    )
    scale = float(from_absolute.abs().max())
    assert torch.allclose(from_slack, from_absolute, rtol=0.0, atol=1e-12 * scale)

    # the rotation is a real, row-dependent diagonal on this batch ...
    assert len(set(batch["reference_rows"].tolist())) > 1
    assert float((rotation - 1.0).abs().max()) > 1e-3
    # ... and it does NOT commute with Y: skipping the unrotation changes the answer
    naive = module._multigrid_decode(i_slack, v_abs, mask, batch)
    assert not torch.allclose(naive, from_absolute, rtol=1e-6, atol=1e-6 * scale)


def test_the_injection_step_is_blind_to_a_members_angle_convention(
    source, multigrid_root
):
    """A member re-solved against a shifted time reference must give the SAME estimate.

    A global time shift rotates order ``h`` by ``exp(1j*h*alpha)`` — the freedom every data
    source's angle convention leaves. With the frame applied at one seam and the physics run
    in the absolute frame, it cancels exactly: features, the decoded injection, the solve
    (whose boundary and RHS scale together) and the re-rotated estimate all come back.
    """
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)

    shifted = dict(batch)
    lo, hi = int(batch["ptr"][1]), int(batch["ptr"][2])
    alpha = 0.7853
    turn = torch.exp(
        1j * alpha * torch.tensor(ORDERS, dtype=torch.float64).reshape(-1, 1)
    )
    state = batch["v"].clone()
    state[..., lo:hi] = state[..., lo:hi] * turn.to(state.dtype)
    shifted["v"] = state

    rng = module._mask_gen.get_state()
    reference = module._step(batch, train=False)
    module._mask_gen.set_state(rng)
    moved = module._step(shifted, train=False)

    assert float(reference[0]) == pytest.approx(float(moved[0]), rel=1e-9)
    scale = float(reference[2].abs().max())
    assert torch.allclose(reference[2], moved[2], rtol=0.0, atol=1e-9 * scale)
    assert torch.allclose(reference[3], moved[3], rtol=0.0, atol=1e-9 * scale)


# =============================================================================
# The training step
# =============================================================================
def test_the_injection_step_solves_and_backpropagates(source, multigrid_root):
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)
    loss, mask, pred_v, target_v, inj_loss, phys_loss, order_mask = module._step(
        batch, train=True
    )
    n = int(batch["ptr"][-1])
    assert loss.ndim == 0 and bool(torch.isfinite(loss))
    assert pred_v.shape == (len(ORDERS), n) and pred_v.is_complex()
    assert target_v.shape == pred_v.shape
    assert mask.shape == (n,)
    assert inj_loss is None and phys_loss is None and order_mask is None
    assert module._loss_components.keys() == {"voltage"}

    # the estimate is in VOLTS and in the run's slack frame: its fundamental reproduces the
    # measured boundary at every member's slack rows (the solve holds them fixed).
    fundamental = ORDERS.index(1)
    slack = batch["slack_rows"]
    assert torch.allclose(
        pred_v[fundamental].index_select(-1, slack),
        target_v[fundamental].index_select(-1, slack),
        atol=1e-9 * float(target_v.abs().max()),
    )

    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().max()) > 0.0 for g in grads)


def test_the_injection_state_split_logs_both_terms(source, multigrid_root):
    """The blend is NORMALIZED (the multi-grid path applies the same rescaling as the
    single-grid one), so the split is the share the injection term actually carries."""
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        loss=LossConfig(injection_state_loss_split=0.4),
    )
    batch = _mixed_batch(source, bank)
    loss, _, _, _, _, _, _ = module._step(batch, train=True)
    assert module._loss_components.keys() == {"voltage", "injection_state"}
    parts = module._loss_components
    # normalized: the scaled injection term equals the voltage term, so the blend is the
    # voltage term itself and the realized share IS the configured split.
    assert float(loss) == pytest.approx(float(parts["voltage"]), rel=1e-9)
    assert float(module._loss_share) == pytest.approx(0.4, rel=1e-9)
    assert all(bool(torch.isfinite(v)) for v in parts.values())


def test_the_blend_target_is_the_members_own_rotated_injection(source, multigrid_root):
    """The label: ``Y_g·V_abs`` per member, in the per-unit base, rotated into the frame."""
    bank = _fresh_bank(source)
    module = _injection_module(multigrid_root, source, bank)
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    rotation = slack_frame_rotation(v_abs, batch["reference_rows"], orders=list(ORDERS))
    segments = module._multigrid_segments(batch)
    i_base = module._multigrid_i_base(batch)
    feats = module._multigrid_injection_target_features(
        v_abs, rotation, batch, segments, i_base
    )

    ptr = batch["ptr"].tolist()
    for item, index in enumerate(batch["grid_index"].tolist()):
        lo, hi = ptr[item], ptr[item + 1]
        solver = bank.solver(int(index))
        expected = solver.injection_targets(v_abs[..., lo:hi])
        expected = expected / bank.i_base(int(index))
        expected = expected * rotation[..., lo:hi]
        # compare the CURRENT the features encode (a near-zero injection has no meaningful
        # angle, so the trig channels of the encoded feature are not a stable comparison)
        recovered = (
            TrigEncoding()
            .decode(module.inj_normalizer.inverse_transform(feats[lo:hi]))
            .transpose(-2, -1)
        )  # [N_g, H] -> [H, N_g]
        assert torch.allclose(
            recovered, expected, rtol=0.0, atol=1e-9 * float(expected.abs().max())
        )
    assert not feats.requires_grad  # a label, never a graph


def test_the_absolute_frame_injection_step_runs(source, multigrid_root):
    """``angle_reference="absolute"`` skips both rotations and decodes the raw solver frame."""
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        data=DataConfig(
            dataset_path=str(multigrid_root),
            multigrid=MultiGridConfig(
                placement_fraction=0.3, angle_reference="absolute"
            ),
            batch_size=2,
            num_workers=0,
        ),
    )
    batch = _mixed_batch(source, bank)
    loss, _, pred_v, target_v, _, _, _ = module._step(batch, train=False)
    assert bool(torch.isfinite(loss))
    assert torch.equal(target_v, batch["v"].to(target_v.dtype))
    fundamental = ORDERS.index(1)
    slack = batch["slack_rows"]
    assert torch.allclose(
        pred_v[fundamental].index_select(-1, slack),
        target_v[fundamental].index_select(-1, slack),
        atol=1e-9 * float(target_v.abs().max()),
    )


def test_the_lv_busbar_frame_decode_uses_the_batchs_reference_rows(multigrid_root):
    """The opt-in local-reference frame runs the SAME boundary rotation, over the same key.

    The source is built for ``"lv_busbar"``, so the batch's ``reference_rows`` are that
    frame's; the decode must unrotate with exactly them (rotating the answer of the absolute
    decode reproduces it) and the step must still hold every member's measured slack.
    """
    source = MultiGridDataSource(
        multigrid_root,
        placement_fraction=0.3,
        with_injection=True,
        angle_reference="lv_busbar",
    )
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        data=DataConfig(
            dataset_path=str(multigrid_root),
            multigrid=MultiGridConfig(
                placement_fraction=0.3, angle_reference="lv_busbar"
            ),
            batch_size=2,
            num_workers=0,
        ),
    )
    batch = _mixed_batch(source, bank)
    expected_rows = torch.cat(
        [
            source.bundle(int(g)).reference_rows + int(off)
            for g, off in zip(batch["grid_index"], batch["ptr"][:-1])
        ]
    )
    assert torch.equal(batch["reference_rows"], expected_rows)

    v_abs = batch["v"].to(torch.complex128)
    rotation = slack_frame_rotation(v_abs, batch["reference_rows"], orders=list(ORDERS))
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)
    i_abs = _random_injection(batch, seed=5)
    i_framed = i_abs * rotation  # what the model predicts, in its own frame

    from_absolute = module._multigrid_decode(i_abs, v_abs, mask, batch) * rotation
    from_framed = (
        module._multigrid_decode(i_framed * rotation.conj(), v_abs, mask, batch)
        * rotation
    )
    scale = float(from_absolute.abs().max())
    assert torch.allclose(from_framed, from_absolute, rtol=0.0, atol=1e-12 * scale)

    loss, _, pred_v, target_v, _, _, _ = module._step(batch, train=False)
    assert bool(torch.isfinite(loss))
    assert torch.equal(
        target_v, slack_referenced_state(v_abs, batch["reference_rows"], orders=ORDERS)
    )
    fundamental = ORDERS.index(1)
    slack = batch["slack_rows"]
    assert torch.allclose(
        pred_v[fundamental].index_select(-1, slack),
        target_v[fundamental].index_select(-1, slack),
        atol=1e-9 * float(target_v.abs().max()),
    )


# =============================================================================
# The localization head over the concatenated device axis
# =============================================================================
def test_the_localization_head_matches_the_members_alone(source, multigrid_root):
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root, source, bank, loss=LossConfig(predict_injection=True)
    )
    assert module.injection_head is not None
    assert module.injection_head.n_orders == len(source.injection_orders)

    batch = _mixed_batch(source, bank)
    n = int(batch["ptr"][-1])
    generator = torch.Generator().manual_seed(11)
    feats = torch.randn((n, len(ORDERS) * 3), generator=generator, dtype=torch.float64)
    batched = module.injection_head(
        feats, batch["device_rows"], batch["device_rows_mask"]
    )
    assert batched.shape == (
        int(batch["device_ptr"][-1]),
        len(source.injection_orders),
        1,
    )

    ptr = batch["ptr"].tolist()
    device_ptr = batch["device_ptr"].tolist()
    for item, index in enumerate(batch["grid_index"].tolist()):
        lo, hi = ptr[item], ptr[item + 1]
        d_lo, d_hi = device_ptr[item], device_ptr[item + 1]
        bundle = source.bundle(int(index))
        # the pooled rows of this item's devices lie inside its own node segment
        rows = batch["device_rows"][d_lo:d_hi]
        valid = batch["device_rows_mask"][d_lo:d_hi]
        assert bool(((rows[valid] >= lo) & (rows[valid] < hi)).all())
        assert bool((batch["device_batch"][d_lo:d_hi] == item).all())
        alone = module.injection_head(
            feats[lo:hi], bundle.device_rows, bundle.device_rows_mask
        )
        assert torch.allclose(batched[d_lo:d_hi], alone, atol=1e-12)


def test_the_head_loss_is_added_on_top_of_the_state_loss(source, multigrid_root):
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        loss=LossConfig(predict_injection=True, injection_loss_weight=2.0),
    )
    batch = _mixed_batch(source, bank)
    loss, _, _, _, inj_loss, _, _ = module._step(batch, train=True)
    assert inj_loss is not None and bool(torch.isfinite(inj_loss))
    assert float(loss) == pytest.approx(
        float(module._loss_components["voltage"]) + 2.0 * float(inj_loss), rel=1e-9
    )
    loss.backward()
    head_grads = [
        p.grad for p in module.injection_head.parameters() if p.grad is not None
    ]
    assert head_grads and any(float(g.abs().max()) > 0.0 for g in head_grads)


def test_the_head_needs_the_members_injected_orders(source, multigrid_root):
    with pytest.raises(ValueError, match="injection_orders"):
        SEModule(
            _config(multigrid_root, loss=LossConfig(predict_injection=True)),
            normalizer=fit_multigrid_normalizer(
                source, TrigEncoding(), _config(multigrid_root)
            ),
            grid_keys=[bundle.key for bundle in source.grids],
        )


# =============================================================================
# Measured branch currents across members
# =============================================================================
def test_the_edge_features_are_per_unit_and_rotated_at_the_source_row(
    source, multigrid_root
):
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        measurement=MeasurementConfig(measure_current=True, slack_known=True),
    )
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    rotation = slack_frame_rotation(v_abs, batch["reference_rows"], orders=list(ORDERS))
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)
    segments = module._multigrid_segments(batch)
    i_base = module._multigrid_i_base(batch)
    edge_obs, edge_mask, i_edges = module._multigrid_edge_observations(
        v_abs, rotation, mask, batch, segments, i_base, v_abs.device
    )
    e_total = int(batch["edge_index"].shape[1])
    assert i_edges.shape == (len(ORDERS), e_total)
    assert edge_mask.shape == (e_total,)
    assert edge_obs.shape == (e_total, len(ORDERS) * 3)

    # the raw currents are each member's own operator applied to its own ABSOLUTE segment
    ptr = batch["ptr"].tolist()
    edge_ptr = batch["edge_ptr"].tolist()
    for item, index in enumerate(batch["grid_index"].tolist()):
        lo, hi = ptr[item], ptr[item + 1]
        e_lo, e_hi = edge_ptr[item], edge_ptr[item + 1]
        expected = bank.current_operator(int(index))(v_abs[..., lo:hi])
        assert torch.allclose(i_edges[..., e_lo:e_hi], expected, atol=1e-12)

    # the features are those currents in the per-unit base, rotated at the SOURCE row
    src = batch["edge_index"][0]
    expected_feat = (i_edges / i_base.index_select(-1, src)) * rotation.index_select(
        -1, src
    )
    expected = module.edge_normalizer.transform(
        TrigEncoding().encode(expected_feat).transpose(-3, -2)
    ).reshape(e_total, -1)
    assert torch.allclose(edge_obs, expected, atol=1e-10)


def test_the_upstream_budget_restricts_every_member(source, multigrid_root):
    from injection_hse.data import upstream_edge_mask

    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        measurement=MeasurementConfig(
            measure_current=True, current_channels="upstream", slack_known=True
        ),
    )
    batch = _mixed_batch(source, bank)
    v_abs = batch["v"].to(torch.complex128)
    mask = torch.ones(int(batch["ptr"][-1]), dtype=torch.bool)
    segments = module._multigrid_segments(batch)
    i_base = module._multigrid_i_base(batch)
    _, edge_mask, _ = module._multigrid_edge_observations(
        v_abs, None, mask, batch, segments, i_base, v_abs.device
    )
    expected = torch.cat(
        [
            upstream_edge_mask(
                source.bundle(int(g)).grid, source.bundle(int(g)).graph.edge_index
            )
            for g in batch["grid_index"].tolist()
        ]
    )
    assert torch.equal(edge_mask, expected)
    assert not bool(edge_mask.all())  # the budget really drops half-edges


def test_the_edge_normalizer_pools_the_members_in_per_unit(source, multigrid_root):
    cfg = _config(multigrid_root)
    bank = _fresh_bank(source)
    normalizer = fit_multigrid_edge_normalizer(bank, source, TrigEncoding(), cfg)
    assert normalizer.mean.shape == (len(ORDERS), 3)
    assert torch.isfinite(normalizer.mean).all()
    assert bool((normalizer.std > 0).all())
    power = fit_multigrid_edge_normalizer(
        bank, source, TrigEncoding(), cfg, quantity="power"
    )
    assert torch.isfinite(power.mean).all()
    with pytest.raises(ValueError, match="quantity"):
        fit_multigrid_edge_normalizer(
            bank, source, TrigEncoding(), cfg, quantity="voltage"
        )


# =============================================================================
# End to end
# =============================================================================
def _experiment_config(root, tmp_path, **overrides):
    overrides.setdefault(
        "train",
        TrainConfig(
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
        ),
    )
    return _config(root, **overrides)


def test_train_experiment_trains_the_multigrid_injection_decode(
    multigrid_root, tmp_path
):
    """The whole run: bank + pooled injection normalizer + device axes + blended loss."""
    from injection_hse.train import evaluate_multigrid, train_experiment

    cfg = _experiment_config(
        multigrid_root,
        tmp_path,
        physics=PhysicsConfig(decode="injection"),
        loss=LossConfig(predict_injection=True, injection_state_loss_split=0.3),
    )
    module = train_experiment(cfg)
    assert module._bank is not None and module.injection_head is not None
    assert module.inj_normalizer is not None

    trained_source = module._bank.source
    assert trained_source.physics_bank is module._bank  # the device axes are collated
    batch = next(iter(_loader(trained_source, batch_size=2)))
    loss, _, pred_v, target_v, inj_loss, _, _ = module._step(batch, train=False)
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(inj_loss))
    assert module._loss_components.keys() == {"voltage", "injection_state"}
    assert pred_v.shape == target_v.shape and pred_v.is_complex()

    report = evaluate_multigrid(module, trained_source, split="val", batch_size=2)
    assert report["n_items"] and report["pu_mae"] == report["pu_mae"]


def test_train_experiment_runs_the_current_anchored_multigrid_decode(
    multigrid_root, tmp_path
):
    """Measured branch currents as input AND as anchors, per member."""
    from injection_hse.train import train_experiment

    cfg = _experiment_config(
        multigrid_root,
        tmp_path,
        physics=PhysicsConfig(
            decode="injection",
            anchor_voltage_weight=1.0,
            anchor_current_weight=0.1,
        ),
        measurement=MeasurementConfig(measure_current=True, slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                    anchor_voltage_weight=2.0,
                    anchor_current_weight=0.5,
                )
            ]
        ),
    )
    module = train_experiment(cfg)
    assert module.edge_normalizer is not None
    bank = module._bank
    # the stage's schedule reached every member, and the build capability is the stage's max
    assert bank.anchor_v_weight == 2.0 and bank.anchor_i_weight == 0.5
    for index in range(len(bank)):
        solver = bank.solver(index)
        assert solver.anchor_v_weight == 2.0 and solver.anchor_i_weight == 0.5
        assert solver._c_map is not None  # the V -> branch-current map was materialised

    batch = next(iter(_loader(module._bank.source, batch_size=2)))
    loss, _, pred_v, _, _, _, _ = module._step(batch, train=False)
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(pred_v).all())


def test_a_member_built_after_the_stage_inherits_its_anchor(source):
    """Lazy members must decode at the STAGE's strength, not the build capability."""
    bank = _fresh_bank(source, anchor_v=10.0, anchor_i=0.0)
    bank.set_anchor_weights(v_weight=3.0, i_weight=0.0)
    fresh = bank.solver(len(source.grids) - 1)
    assert fresh.anchor_v_weight == 3.0
    assert fresh.anchoring
    assert (
        fresh._v_scale is not None
    )  # the anchored solve's per-order scale is available
    with pytest.raises(ValueError, match="branch-current map"):
        bank.set_anchor_weights(v_weight=1.0, i_weight=1.0)


def _loader(source, *, batch_size):
    from torch.utils.data import DataLoader

    return DataLoader(
        source.train_dataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=source.collate(),
    )


def test_a_device_free_batch_keeps_the_head_on_the_graph(source, multigrid_root):
    """A batch drawn only from injection-free members has nothing to localize.

    ``D* = 0``: the per-device MSE is undefined (an empty mean is NaN), so no injection loss
    is formed — but the head must still take part in the backward, or a distributed step would
    find its parameters unused.
    """
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root, source, bank, loss=LossConfig(predict_injection=True)
    )
    batch = MultiGridCollate(source, with_injection=True, bank=bank)(
        _items(source, [QUIET_KEY])
    )
    assert int(batch["device_ptr"][-1]) == 0

    loss, _, _, _, inj_loss, _, _ = module._step(batch, train=True)
    assert inj_loss is None and bool(torch.isfinite(loss))
    loss.backward()
    head_grads = [
        p.grad for p in module.injection_head.parameters() if p.grad is not None
    ]
    assert head_grads and all(float(g.abs().max()) == 0.0 for g in head_grads)


def test_the_injection_decode_needs_the_operator_bank(source, multigrid_root):
    """The gap the bank closes: a multi-grid injection config must not fall back to voltage."""
    cfg = _config(multigrid_root, physics=PhysicsConfig(decode="injection"))
    with pytest.raises(ValueError, match="operator bank"):
        SEModule(
            cfg,
            normalizer=fit_multigrid_normalizer(source, TrigEncoding(), cfg),
            grid_keys=[bundle.key for bundle in source.grids],
        )


def test_measured_currents_need_the_operator_bank(source, multigrid_root):
    cfg = _config(
        multigrid_root,
        measurement=MeasurementConfig(measure_current=True, slack_known=True),
    )
    bank = _fresh_bank(source)
    with pytest.raises(ValueError, match="MultiGridPhysicsBank"):
        SEModule(
            cfg,
            normalizer=fit_multigrid_normalizer(source, TrigEncoding(), cfg),
            grid_keys=[bundle.key for bundle in source.grids],
            edge_normalizer=fit_multigrid_edge_normalizer(
                bank, source, TrigEncoding(), cfg
            ),
        )


def test_measured_currents_feed_the_voltage_decode_too(source, multigrid_root):
    """Branch currents are an INPUT: they compose with the direct voltage decode as well."""
    bank = _fresh_bank(source)
    module = _injection_module(
        multigrid_root,
        source,
        bank,
        physics=PhysicsConfig(decode="voltage"),
        measurement=MeasurementConfig(
            measure_current=True, measure_power=True, slack_known=True
        ),
    )
    assert module.power_normalizer is not None
    batch = _mixed_batch(source, bank)
    loss, _, pred_v, target_v, _, _, _ = module._step(batch, train=True)
    assert bool(torch.isfinite(loss))
    assert pred_v.shape == target_v.shape
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0.0 for g in grads)
