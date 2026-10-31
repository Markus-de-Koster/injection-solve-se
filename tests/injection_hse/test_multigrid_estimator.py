"""The estimator side of a multi-grid run: models, the training step, and the evaluation.

The data path (source / collate / masking / per-unit) is covered in ``test_multigrid.py``;
here the batch it produces is fed to the MODELS and to :class:`~injection_hse.train.SEModule`.

The load-bearing property is that a batched forward equals running each member grid ALONE
and concatenating the outputs: a batch of grids is one disjoint graph, so any leakage across
member boundaries (a message over an offset edge, an attention weight between two grids, a
pooled statistic) would show up as a difference. Every model is checked against that oracle.
"""

from __future__ import annotations

from types import SimpleNamespace

import lightning as L
import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    GNNConfig,
    GraphormerConfig,
    MaskStageConfig,
    MultiGridConfig,
    PhysicsConfig,
    StructuralEncodingConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import (
    MultiGridCollate,
    MultiGridDataSource,
    per_unit_state,
    slack_referenced_state,
)
from injection_hse.data.multigrid import (
    MultiGridGenerationConfig,
    MultiGridManifest,
    SyntheticFeederSource,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.io import build_model_input
from injection_hse.models import build_model
from injection_hse.train import SEModule, evaluate_multigrid, fit_multigrid_normalizer

N_GRIDS = 7
N_SCENARIOS = 3
ORDERS = [1]


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    """A tiny multi-grid dataset whose last member is marked as the held-out TEST grid.

    A synthetic-only dataset has no real grids, so the generator assigns no test split; the
    manifest is rewritten here to hold one member out, which is what the test-split
    (unseen-topology) evaluation path consumes.
    """
    root = tmp_path_factory.mktemp("multigrid_estimator")
    config = MultiGridGenerationConfig(
        output_root=str(root),
        n_scenarios=N_SCENARIOS,
        val_fraction=0.3,
        sources=[
            SyntheticFeederSource(n_grids=N_GRIDS, node_range=(4, 9), n_feeders=2)
        ],
    )
    from injection_hse.data.multigrid import generate_multigrid_dataset

    manifest = generate_multigrid_dataset(config)
    grids = list(manifest.grids)
    grids[-1] = grids[-1].model_copy(update={"split": "test"})
    manifest.model_copy(update={"grids": grids}).write(root)
    assert MultiGridManifest.read(root).split("test")
    return root


@pytest.fixture(scope="module")
def source(multigrid_root):
    return MultiGridDataSource(multigrid_root, placement_fraction=0.3)


@pytest.fixture(scope="module")
def batch(source):
    """One collated batch drawn from three DIFFERENT member grids."""
    dataset = source.train_dataset()
    seen: dict[int, dict] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        seen.setdefault(int(item["grid"]), item)
        if len(seen) == 3:
            break
    assert len(seen) == 3, "the fixture must provide several member grids"
    collated = MultiGridCollate(source)([seen[k] for k in sorted(seen)])
    sizes = (collated["ptr"][1:] - collated["ptr"][:-1]).tolist()
    assert len(set(sizes)) > 1, "the members must differ in size"
    return collated


def _model_input(batch, mask, features):
    """A multi-grid ``ModelInput`` over the concatenated node axis."""
    graph = SimpleNamespace(
        x=batch["node_static"],
        edge_index=batch["edge_index"],
        edge_attr=batch["edge_static"],
    )
    return build_model_input(
        graph,
        features,
        mask,
        node_batch=batch["batch"],
        node_ptr=batch["ptr"],
        edge_batch=batch["edge_batch"],
    )


def _member_input(source, batch, item, mask, features):
    """The same input restricted to ONE member (its own graph, no index vectors)."""
    bundle = source.bundle(int(batch["grid_index"][item]))
    lo, hi = int(batch["ptr"][item]), int(batch["ptr"][item + 1])
    return build_model_input(bundle.graph, features[lo:hi], mask[lo:hi]), (lo, hi)


def _features(batch, *, seed=0):
    """Deterministic node features + mask over the batched node axis."""
    n = int(batch["ptr"][-1])
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, len(ORDERS) * 3, generator=generator)
    mask = torch.rand(n, generator=generator) > 0.4
    return features, mask


def _assert_batched_equals_individual(model, source, batch, *, atol=1e-5):
    features, mask = _features(batch)
    with torch.no_grad():
        batched = model(_model_input(batch, mask, features))[0]  # [N*, F]
        for item in range(int(batch["grid_index"].numel())):
            member_input, (lo, hi) = _member_input(source, batch, item, mask, features)
            alone = model(member_input)[0]
            assert torch.allclose(batched[lo:hi], alone, atol=atol), f"member {item}"
    return batched


# =============================================================================
# Models: no leakage across member graphs
# =============================================================================
def test_gnn_batched_forward_equals_the_members_alone(source, batch):
    model = build_model(
        GNNConfig(hidden_dim=16, num_layers=3),
        n_features=len(ORDERS) * 3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    ).eval()
    out = _assert_batched_equals_individual(model, source, batch)
    assert out.shape == (int(batch["ptr"][-1]), len(ORDERS) * 3)
    assert bool(torch.isfinite(out).all())


def test_temporal_gnn_batched_forward_equals_the_members_alone(source, batch):
    model = build_model(
        TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=2), hidden_dim=16),
        n_features=len(ORDERS) * 3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    ).eval()
    _assert_batched_equals_individual(model, source, batch)


@pytest.mark.parametrize("attention_bias", ["spd", "electrical", "both"])
def test_graphormer_batched_forward_equals_the_members_alone(
    source, batch, attention_bias
):
    model = build_model(
        GraphormerConfig(
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            attention_bias=attention_bias,
        ),
        n_features=len(ORDERS) * 3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    ).eval()
    _assert_batched_equals_individual(model, source, batch)


def test_graphormer_electrical_bias_builds_its_own_table(batch):
    spd = build_model(
        GraphormerConfig(hidden_dim=16, num_layers=1, num_heads=4),
        n_features=3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    )
    both = build_model(
        GraphormerConfig(
            hidden_dim=16, num_layers=1, num_heads=4, attention_bias="both"
        ),
        n_features=3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    )
    names = {name for name, _ in spd.named_parameters()}
    assert "spatial_bias.weight" in names and "electrical_bias.weight" not in names
    both_names = {name for name, _ in both.named_parameters()}
    assert {"spatial_bias.weight", "electrical_bias.weight"} <= both_names


def test_electrical_buckets_match_the_grids_effective_resistance(source):
    """The bias derives the SAME electrical distance the structural encodings compute.

    The model reads its series resistances back out of ``edge_static`` (``expm1`` of the
    ``log1p`` column) rather than from a grid object, so the check is that the bucketed result
    equals the one :func:`injection_hse.data.pairwise_effective_resistance` derives from the grid itself.
    """
    from injection_hse.data import pairwise_effective_resistance
    from injection_hse.models.graphormer import Graphormer

    bundle = source.grids[0]
    n = bundle.n_nodes
    from_edges = Graphormer._compute_electrical(
        bundle.graph.edge_index, bundle.graph.edge_attr, n
    )
    reference = pairwise_effective_resistance(bundle.grid)

    far = Graphormer.NUM_ELECTRICAL - 1
    unreachable = reference < 0.0
    assert torch.equal(from_edges == far, unreachable)
    assert torch.equal(from_edges == 0, reference == 0.0)
    # the finite pairs land in the same log bucket
    span = Graphormer.NUM_ELECTRICAL - 2
    decades = Graphormer.ELECTRICAL_DECADES
    expected = 1 + torch.floor(
        (torch.log10(reference.to(torch.float64).clamp_min(1e-30)) + decades)
        / decades
        * span
    ).to(torch.int64)
    finite = (~unreachable) & (reference > 0.0)
    assert torch.equal(from_edges[finite], expected.clamp(1, span)[finite])


def test_graphormer_attention_cannot_reach_another_member(source, batch):
    """A member's output must not move when ANOTHER member's observations change."""
    model = build_model(
        GraphormerConfig(hidden_dim=16, num_layers=2, num_heads=4),
        n_features=len(ORDERS) * 3,
        n_nodes=0,
        n_static_features=int(batch["node_static"].shape[1]),
    ).eval()
    features, mask = _features(batch)
    lo, hi = int(batch["ptr"][1]), int(batch["ptr"][2])
    perturbed = features.clone()
    perturbed[lo:hi] += 5.0
    with torch.no_grad():
        base = model(_model_input(batch, mask, features))[0]
        moved = model(_model_input(batch, mask, perturbed))[0]
    assert torch.allclose(base[:lo], moved[:lo], atol=1e-6)
    assert torch.allclose(base[hi:], moved[hi:], atol=1e-6)
    assert not torch.allclose(base[lo:hi], moved[lo:hi])


# =============================================================================
# Config guard
# =============================================================================
def _config(
    root,
    *,
    model=None,
    encoding=None,
    stages=None,
    angle_reference="slack",
    **overrides,
):
    cfg = dict(
        name="multigrid",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(
                placement_fraction=0.3, angle_reference=angle_reference
            ),
            structural_encoding=encoding or StructuralEncodingConfig(),
            batch_size=3,
            num_workers=0,
        ),
        model=model or GNNConfig(hidden_dim=16, num_layers=2),
        curriculum=CurriculumConfig(
            stages=stages
            or [
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


def test_fixed_topology_dnn_with_multigrid_is_rejected(multigrid_root):
    with pytest.raises(ValueError, match="dnn"):
        _config(multigrid_root, model=DNNConfig(hidden_dims=[8]))


def test_single_grid_config_is_unaffected():
    """The guard only fires for a multi-grid run (the single-grid contract is untouched)."""
    cfg = ExperimentConfig(
        name="single",
        data=DataConfig(dataset_path="x"),
        physics=PhysicsConfig(decode="injection"),
        curriculum=CurriculumConfig(
            stages=[MaskStageConfig(name="s", target_loss=1.0, max_epochs=1)]
        ),
    )
    assert cfg.physics.decode == "injection"


# =============================================================================
# SEModule
# =============================================================================
def _module(root, source, **kwargs):
    cfg = _config(root, **kwargs)
    normalizer = fit_multigrid_normalizer(source, TrigEncoding(), cfg)
    module = SEModule(
        cfg,
        normalizer=normalizer,
        grid_keys=[bundle.key for bundle in source.grids],
    )
    module.set_masker(
        __import__("injection_hse.masking", fromlist=["build_masker"]).build_masker(
            "random", observed_fraction=0.5
        )
    )
    return module


def test_normalizer_is_fit_on_per_unit_states(source, multigrid_root):
    cfg = _config(multigrid_root)
    normalizer = fit_multigrid_normalizer(source, TrigEncoding(), cfg)
    assert normalizer.mean.shape == (len(ORDERS), 3)
    # the magnitude statistic sits at the per-unit scale (log1p of ~1), not at 400 V
    assert 0.1 < float(normalizer.mean[0, 0]) < 1.5


def test_training_step_runs_and_backpropagates(source, multigrid_root, batch):
    module = _module(multigrid_root, source)
    loss, mask, pred_v, target_v, inj, phys, order_mask = module._step(
        batch, train=True
    )
    n = int(batch["ptr"][-1])
    assert loss.ndim == 0 and bool(torch.isfinite(loss))
    assert pred_v.shape == (len(ORDERS), n) and pred_v.is_complex()
    assert target_v.shape == pred_v.shape
    assert mask.shape == (n,)
    assert inj is None and phys is None and order_mask is None

    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_the_estimate_is_returned_in_volts(source, multigrid_root, batch):
    """Per-unit is the FEATURE space; the state and its metrics stay in volts."""
    module = _module(multigrid_root, source)
    _, _, pred_v, target_v, _, _, _ = module._step(batch, train=False)
    expected = slack_referenced_state(
        batch["v"].to(target_v.dtype),
        batch["reference_rows"],
        orders=list(ORDERS),
    )
    assert torch.equal(target_v, expected)  # volts, in the run's angle frame
    pu = per_unit_state(pred_v.detach(), batch["base_voltage"])
    assert float(pu.abs().max()) < float(pred_v.detach().abs().max())


# =============================================================================
# The slack-referenced angle frame
# =============================================================================
def test_the_step_puts_inputs_targets_and_estimate_in_the_slack_frame(
    source, multigrid_root, batch
):
    """One seam: the state is rotated once, before features, targets and metrics."""
    module = _module(multigrid_root, source)
    loss, _, pred_v, target_v, _, _, _ = module._step(batch, train=True)

    # magnitudes survive the frame, the reference rows' fundamental angle is zeroed
    raw = batch["v"].to(target_v.dtype)
    assert torch.allclose(target_v.abs(), raw.abs(), rtol=1e-6, atol=1e-6)
    fundamental = target_v[ORDERS.index(1)]
    assert float(fundamental[batch["reference_rows"]].angle().abs().max()) < 1e-6
    # the estimate comes back in the same frame (it is compared against this target)
    assert pred_v.shape == target_v.shape and pred_v.is_complex()
    assert bool(torch.isfinite(loss))


def test_training_and_validation_run_in_the_slack_frame(source, multigrid_root, batch):
    """A full training_step + validation_step on the synthetic corpus, slack frame."""
    module = _module(multigrid_root, source)
    logged: dict[str, float] = {}
    module.log = lambda name, value, **_kw: logged.__setitem__(
        name, float(value.detach() if torch.is_tensor(value) else value)
    )  # type: ignore[method-assign]

    module.on_train_epoch_start()
    loss = module.training_step(batch, 0)
    assert bool(torch.isfinite(loss)) and loss.requires_grad
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)

    module.on_validation_epoch_start()
    val_loss = module.validation_step(batch, 0)
    assert bool(torch.isfinite(val_loss))
    assert all(v == v for v in (logged["val_pu_mae"], logged["val_pu_mae_unobs"]))


def test_the_step_is_blind_to_a_members_global_angle_convention(
    source, multigrid_root, batch
):
    """The motivation, end to end: re-solving a member against a rotated source EMF —
    the same physical state in another angle convention — must not move the loss."""
    module = _module(multigrid_root, source)
    turned = dict(batch)
    item = 1
    lo, hi = int(batch["ptr"][item]), int(batch["ptr"][item + 1])
    rotated = batch["v"].clone()
    # ORDERS is fundamental-only here, so a global convention shift is one rotation
    rotated[..., lo:hi] *= torch.exp(torch.tensor(1j * 0.7853, dtype=rotated.dtype))
    turned["v"] = rotated

    state = module._mask_gen.get_state()
    reference = module._step(batch, train=False)
    module._mask_gen.set_state(state)
    shifted = module._step(turned, train=False)

    # equal up to the float32 rounding of the rotation (a frame that did NOT remove the
    # convention would move this member's angles by the full 45 degrees applied above, i.e.
    # by a fraction of the state itself). The tolerance is measured against the state SCALE,
    # not per element: the rounding of a ~1e4 V pipeline lands on every element alike, so an
    # element that happens to sit near zero must not be held to a relative bound.
    scale = float(reference[3].abs().max())
    assert float(reference[0]) == pytest.approx(float(shifted[0]), rel=1e-4)
    assert torch.allclose(reference[2], shifted[2], rtol=0.0, atol=1e-3 * scale)
    assert torch.allclose(reference[3], shifted[3], rtol=0.0, atol=1e-5 * scale)


def test_the_absolute_frame_keeps_the_raw_solver_state(source, multigrid_root, batch):
    """``angle_reference="absolute"`` is the untouched behaviour: no rotation at all."""
    module = _module(multigrid_root, source, angle_reference="absolute")
    _, _, _, target_v, _, _, _ = module._step(batch, train=False)
    assert torch.equal(target_v, batch["v"].to(target_v.dtype))


def test_training_and_validation_run_in_the_lv_busbar_frame(multigrid_root):
    """The opt-in local-reference frame: the same step, over the batch's reference rows.

    The synthetic members carry no transformer, so their LV-busbar reference IS the slack one
    (the documented coincidence) — what this exercises is the plumbing: the source builds the
    configured frame's rows, the collate emits them under the one key, and the step consumes
    them.
    """
    source = MultiGridDataSource(
        multigrid_root, placement_fraction=0.3, angle_reference="lv_busbar"
    )
    module = _module(multigrid_root, source, angle_reference="lv_busbar")
    logged: dict[str, float] = {}
    module.log = lambda name, value, **_kw: logged.__setitem__(
        name, float(value.detach() if torch.is_tensor(value) else value)
    )  # type: ignore[method-assign]

    dataset = source.train_dataset()
    seen: dict[int, dict] = {}
    for i in range(len(dataset)):
        seen.setdefault(int(dataset[i]["grid"]), dataset[i])
        if len(seen) == 3:
            break
    batch = MultiGridCollate(source)([seen[k] for k in sorted(seen)])

    module.on_train_epoch_start()
    loss = module.training_step(batch, 0)
    assert bool(torch.isfinite(loss)) and loss.requires_grad
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)

    module.on_validation_epoch_start()
    assert bool(torch.isfinite(module.validation_step(batch, 0)))
    assert all(v == v for v in (logged["val_pu_mae"], logged["val_pu_mae_unobs"]))

    # the supervision target is the state expressed over THIS frame's reference rows
    _, _, _, target_v, _, _, _ = module._step(batch, train=False)
    assert torch.equal(
        target_v,
        slack_referenced_state(
            batch["v"].to(target_v.dtype),
            batch["reference_rows"],
            orders=list(ORDERS),
        ),
    )


def test_the_two_frames_agree_on_an_already_referenced_batch(
    source, multigrid_root, batch
):
    """The rotation is applied EXACTLY once (a double rotation would move the state).

    Fed a batch whose state is already expressed in the slack frame, the ``"slack"`` module
    must produce the same loss and estimate as the ``"absolute"`` one — its rotation is then
    the identity.
    """
    referenced = dict(batch)
    referenced["v"] = slack_referenced_state(
        batch["v"], batch["reference_rows"], orders=list(ORDERS)
    )
    module = _module(multigrid_root, source)
    twin = _module(multigrid_root, source, angle_reference="absolute")
    twin.load_state_dict(module.state_dict())  # same weights ...
    twin.normalizer.load_state_dict(module.normalizer.state_dict())  # ... and scales

    loss, _, pred_v, target_v, _, _, _ = module._step(referenced, train=False)
    twin_loss, _, twin_pred, twin_target, _, _, _ = twin._step(referenced, train=False)

    assert torch.allclose(target_v, twin_target, atol=1e-6)
    assert torch.equal(twin_target, referenced["v"].to(twin_target.dtype))
    assert float(loss) == pytest.approx(float(twin_loss))
    assert torch.allclose(pred_v, twin_pred, atol=1e-6)


def test_the_normalizer_magnitude_statistics_are_frame_free(source, multigrid_root):
    """Rotating the state cannot move a per-harmonic MAGNITUDE statistic."""
    referenced = fit_multigrid_normalizer(
        source, TrigEncoding(), _config(multigrid_root)
    )
    absolute = fit_multigrid_normalizer(
        source, TrigEncoding(), _config(multigrid_root, angle_reference="absolute")
    )
    assert torch.allclose(referenced.mean[:, 0], absolute.mean[:, 0], atol=1e-9)
    assert torch.allclose(referenced.std[:, 0], absolute.std[:, 0], atol=1e-9)


def test_evaluate_multigrid_scores_in_the_module_frame(source, multigrid_root):
    """The reported pu-MAE is measured between an estimate and a target of ONE frame."""
    module = _module(multigrid_root, source)
    report = evaluate_multigrid(module, source, split="val", batch_size=2)
    assert report["n_items"] and report["pu_mae"] == report["pu_mae"]

    dataset = source.val_dataset()
    single = MultiGridCollate(source)([dataset[0]])
    estimate = module.infer(single)
    expected = slack_referenced_state(
        single["v"].to(estimate["target_v"].dtype),
        single["reference_rows"],
        orders=list(ORDERS),
    )
    assert torch.equal(estimate["target_v"], expected)


def test_mask_observes_a_fraction_of_every_member(source, multigrid_root, batch):
    module = _module(multigrid_root, source)
    _, mask, _, _, _, _, _ = module._step(batch, train=True)
    ptr = batch["ptr"].tolist()
    for item in range(len(ptr) - 1):
        size = ptr[item + 1] - ptr[item]
        observed = int(mask[ptr[item] : ptr[item + 1]].sum())
        assert observed >= max(1, round(0.5 * size))
    assert bool(mask[batch["slack_rows"]].all())


def test_bus_mask_observes_all_phases_together(source, multigrid_root, batch):
    """A physical voltage meter either observes every present phase or none of them."""
    data = DataConfig(
        dataset_path=str(multigrid_root),
        multigrid=MultiGridConfig(random_mask_unit="bus", placement_fraction=0.3),
        batch_size=3,
        num_workers=0,
    )
    module = _module(multigrid_root, source, data=data)
    _, mask, _, _, _, _, _ = module._step(batch, train=True)
    for bus in range(int(batch["bus_ptr"][-1])):
        phases = mask[batch["row_bus"] == bus]
        assert bool(phases.all()) or not bool(phases.any())
    assert bool(mask[batch["slack_rows"]].all())


def test_placement_stage_uses_the_per_member_placement(source, multigrid_root, batch):
    from injection_hse.masking import build_masker

    module = _module(multigrid_root, source)
    module.set_masker(
        build_masker("placement", observed_rows=torch.zeros(0, dtype=torch.int64))
    )
    _, mask, _, _, _, _, _ = module._step(batch, train=False)
    expected = torch.zeros(int(batch["ptr"][-1]), dtype=torch.bool)
    expected[batch["observed_rows"]] = True
    expected[batch["slack_rows"]] = True
    assert torch.equal(mask, expected)


def test_validation_logs_pooled_and_per_grid_metrics(source, multigrid_root, batch):
    module = _module(multigrid_root, source)
    captured: dict[str, float] = {}

    def _capture(name, value, **_kwargs):
        captured[name] = float(value.detach() if torch.is_tensor(value) else value)

    module.log = _capture  # type: ignore[method-assign]
    module.on_validation_epoch_start()
    module.validation_step(batch, 0)
    module._log_grid_metrics()

    assert "val_pu_mae" in captured and "val_pu_mae_unobs" in captured
    assert f"val_pu_mae_h{ORDERS[0]}" in captured
    keys = [source.bundle(int(g)).key for g in batch["grid_index"]]
    for key in keys:
        assert f"val_pu_mae_grid_{key}" in captured
    assert captured["val_pu_mae_worst_grid"] == pytest.approx(
        max(captured[f"val_pu_mae_grid_{k}"] for k in keys)
    )


def test_per_grid_metric_matches_state_errors_on_one_member(source, multigrid_root):
    """The per-member number is the same all-rows pu-MAE the pooled metric reports."""
    from injection_hse.metrics import state_errors

    dataset = source.train_dataset()
    single = MultiGridCollate(source)([dataset[0]])
    module = _module(multigrid_root, source)
    _, mask, pred_v, target_v, _, _, _ = module._step(single, train=False)
    n = int(single["ptr"][-1])
    errs = state_errors(
        pred_v.reshape(-1, len(ORDERS), n),
        target_v.reshape(-1, len(ORDERS), n),
        mask=mask.reshape(-1, n),
        orders=list(ORDERS),
        base_v=single["base_voltage"],
    )
    module.on_validation_epoch_start()
    module._accumulate_grid_errors(single, pred_v, target_v)
    total, count = module._val_grid_error
    seen = count > 0
    assert int(seen.sum()) == 1
    value = float((total[seen] / count[seen]).item())
    assert value == pytest.approx(float(errs["mae_all"]), rel=1e-5)


@pytest.mark.parametrize(
    "model, encoding",
    [
        (
            TemporalConfig(
                spatial=GNNConfig(hidden_dim=16, num_layers=2), hidden_dim=16
            ),
            StructuralEncodingConfig(),
        ),
        (
            GraphormerConfig(hidden_dim=16, num_layers=2, num_heads=4),
            StructuralEncodingConfig(kind="rwse", rwse_steps=4, base="normalized"),
        ),
        (
            GraphormerConfig(
                hidden_dim=16, num_layers=2, num_heads=4, attention_bias="electrical"
            ),
            StructuralEncodingConfig(kind="electrical", base="normalized"),
        ),
    ],
)
def test_one_epoch_fit_logs_per_grid_validation(
    multigrid_root, tmp_path, model, encoding
):
    """A real Lightning epoch over the multi-grid loaders, for each shipped arm."""
    from injection_hse.train import make_datamodule

    encoded_source = MultiGridDataSource(
        multigrid_root, placement_fraction=0.3, encoding=encoding
    )
    cfg = _config(
        multigrid_root,
        model=model,
        encoding=encoding,
        train=TrainConfig(
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            enable_progress_bar=False,
        ),
    )
    normalizer = fit_multigrid_normalizer(encoded_source, TrigEncoding(), cfg)
    module = SEModule(
        cfg,
        normalizer=normalizer,
        grid_keys=[bundle.key for bundle in encoded_source.grids],
    )
    dm = make_datamodule(encoded_source, batch_size=2)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=dm)
    metrics = trainer.callback_metrics
    assert torch.isfinite(metrics["val_pu_mae"])
    assert "val_pu_mae_worst_grid" in metrics
    per_grid = [k for k in metrics if k.startswith("val_pu_mae_grid_")]
    assert per_grid, "every validated member grid must be reported"


# =============================================================================
# Evaluation (held-out topologies)
# =============================================================================
def test_evaluate_multigrid_reports_pooled_and_per_grid(source, multigrid_root):
    module = _module(multigrid_root, source)
    report = evaluate_multigrid(module, source, split="val", batch_size=2)

    assert report["split"] == "val"
    assert report["n_items"] == len(source.val_dataset())
    assert report["pu_mae"] > 0.0
    val_keys = {source.bundle(i).key for i in source.split_indices("val")}
    assert set(report["per_grid"]) == val_keys
    assert report["worst_grid"] in val_keys
    for entry in report["per_grid"].values():
        assert entry["n_nodes"] > 0 and entry["n_items"] > 0


def test_evaluate_multigrid_scores_the_held_out_test_grids(source, multigrid_root):
    """The unseen-topology path: the manifest's test split, never trained on."""
    module = _module(multigrid_root, source)
    report = evaluate_multigrid(module, source, split="test", batch_size=2)
    test_keys = {source.bundle(i).key for i in source.split_indices("test")}
    assert test_keys, "the fixture holds one member out as the test split"
    assert set(report["per_grid"]) == test_keys
    assert report["n_items"] == N_SCENARIOS * len(test_keys)


def test_make_datamodule_uses_the_sources_collate(source):
    from injection_hse.train import make_datamodule

    dm = make_datamodule(source, batch_size=2)
    assert isinstance(dm._collate, MultiGridCollate)
    loaded = next(iter(dm.train_dataloader()))
    assert {"ptr", "batch", "edge_batch", "base_voltage"} <= set(loaded)


def test_a_multigrid_checkpoint_reloads_from_the_corpus_not_one_grid(
    source, multigrid_root, tmp_path
):
    """One grid cannot rebuild a many-grid input; the corpus source rebuilds it exactly."""
    module = _module(multigrid_root, source)
    path = tmp_path / "multigrid.ckpt"
    checkpoint = {
        "state_dict": module.state_dict(),
        "pgl_config": module.cfg.model_dump(),
    }
    module.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="data.multigrid"):
        SEModule.from_checkpoint(path, grid=source.grids[0].grid)

    rebuilt = SEModule.from_checkpoint(path, source=source)
    assert rebuilt.cfg.data.multigrid is not None
    for reference, restored in zip(
        module.state_dict().values(), rebuilt.state_dict().values()
    ):
        assert torch.equal(reference, restored)


def test_normalizer_travels_with_the_checkpoint(source, multigrid_root):
    module = _module(multigrid_root, source)
    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)
    assert "normalizer" in checkpoint
    assert checkpoint["pgl_config"]["data"]["multigrid"]["per_unit"] is True
