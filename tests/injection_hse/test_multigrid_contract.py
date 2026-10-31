"""The multi-grid CAPABILITY CONTRACT: what that path reads, and what it refuses.

Two parallel training paths (one grid vs many) share one :class:`~injection_hse.config.ExperimentConfig`,
so the dangerous failure mode is a field that is accepted and then silently not applied. These
tests hold the contract:

- every config field is classified as consumed or ignored on the multi-grid path — the
  completeness walk fails on any field a future change adds without classifying it;
- an IGNORED field holding a non-default value is rejected, naming the field and the reason;
- the DATA-split fields are ignored by construction (the split is by grid, from the corpus
  manifest) while ``orders`` selects a genuine subset of the corpus' spectrum;
- the injection/voltage loss blend is NORMALIZED exactly as on the single-grid path, and the
  realized share is logged — the diagnostic that says whether the split means what it says.

A new :mod:`injection_hse.config` field fails the completeness walk until it is classified in
``MULTIGRID_CONSUMED`` or ``MULTIGRID_IGNORED``. Classify it by reading what the multi-grid
path actually reads (``SEModule._multigrid_step`` and its helpers,
``injection_hse.data.build_multigrid_source``); an IGNORED field with a non-default value is then
rejected by the ``ExperimentConfig`` validator, so callers on that path must stop setting it.

A corpus built from ``SyntheticFeederSource`` alone has NO test split: only real grids
(simbench / ding0) are routed to test. A test that needs one rewrites the manifest after
generation (``manifest.model_copy(update={"grids": ...}).write(root)`` with one entry's
``split`` set to ``"test"``).
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    MULTIGRID_CONSUMED,
    MULTIGRID_IGNORED,
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    GNNConfig,
    LossConfig,
    MaskStageConfig,
    MeasurementConfig,
    MultiGridConfig,
    PhysicsConfig,
    TopologyConfig,
    TopologyVariantConfig,
    TrainConfig,
    unclassified_multigrid_fields,
)
from injection_hse.data import MultiGridCollate, MultiGridDataSource, build_data_source
from injection_hse.data.multigrid import (
    MultiGridGenerationConfig,
    SyntheticFeederSource,
    generate_multigrid_dataset,
)
from injection_hse.encoding import TrigEncoding
from injection_hse.masking import build_masker
from injection_hse.physics import MultiGridPhysicsBank, fit_multigrid_injection_normalizer
from injection_hse.train import SEModule, fit_multigrid_normalizer

ORDERS = (1, 3, 5)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def multigrid_root(tmp_path_factory):
    """A tiny corpus of synthetic feeders with harmonic injection."""
    root = tmp_path_factory.mktemp("multigrid_contract")
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(root),
            n_scenarios=3,
            orders=ORDERS,
            val_fraction=0.3,
            sources=[SyntheticFeederSource(n_grids=3, node_range=(4, 6), n_feeders=2)],
        )
    )
    return root


def _config(root, **overrides) -> ExperimentConfig:
    fields = dict(
        name="multigrid-contract",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=8, num_layers=1),
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
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _data(root, **overrides) -> DataConfig:
    fields = dict(
        dataset_path=str(root),
        multigrid=MultiGridConfig(placement_fraction=0.3),
        batch_size=2,
        num_workers=0,
    )
    fields.update(overrides)
    return DataConfig(**fields)


# =============================================================================
# The classification is COMPLETE (this is what makes the contract a contract)
# =============================================================================
def test_every_config_field_is_classified():
    """A field in neither set is an undeclared multi-grid behaviour — fail until classified.

    This is the guard that survives future config growth: adding a field to any sub-config
    breaks this test until its multi-grid behaviour is decided and written down.
    """
    assert unclassified_multigrid_fields() == []


def test_the_two_classes_are_disjoint():
    assert not (set(MULTIGRID_CONSUMED) & set(MULTIGRID_IGNORED))
    assert all(reason.strip() for reason in MULTIGRID_IGNORED.values())


# =============================================================================
# An IGNORED field must FAIL LOUDLY when it is set
# =============================================================================
@pytest.mark.parametrize(
    "overrides, field",
    [
        (dict(data_kwargs=dict(val_fraction=0.25)), "data.val_fraction"),
        (dict(data_kwargs=dict(test_fraction=0.2)), "data.test_fraction"),
        (dict(data_kwargs=dict(split_seed=7)), "data.split_seed"),
        (dict(data_kwargs=dict(source="online")), "data.source"),
        (
            dict(data_kwargs=dict(scenario_config_path="scenario.yaml")),
            "data.scenario_config_path",
        ),
        (
            dict(measurement=MeasurementConfig(voltage_nodes=["Bus R4"])),
            "measurement.voltage_nodes",
        ),
        (
            dict(
                measurement=MeasurementConfig(
                    heterogeneous_orders=True, fundamental_nodes=["Bus R4"]
                )
            ),
            "measurement.fundamental_nodes",
        ),
        (
            dict(physics=PhysicsConfig(calibration_weight=1e-3)),
            "physics.calibration_weight",
        ),
    ],
)
def test_a_non_default_ignored_field_is_rejected(multigrid_root, overrides, field):
    data_kwargs = overrides.pop("data_kwargs", {})
    with pytest.raises(ValueError) as excinfo:
        _config(multigrid_root, data=_data(multigrid_root, **data_kwargs), **overrides)
    message = str(excinfo.value)
    assert field in message
    # the reason travels with the rejection — the user learns WHY, not just THAT
    assert MULTIGRID_IGNORED[field].split(";")[0][:40] in message


def test_a_stage_level_ignored_field_is_rejected(multigrid_root):
    """The walk reaches list elements: a per-stage field is checked on every stage."""
    stages = [
        MaskStageConfig(
            name="frac_50",
            kind="random",
            observed_fraction=0.5,
            fundamental_fraction=0.5,
            target_loss=1e-3,
            max_epochs=1,
        )
    ]
    with pytest.raises(ValueError, match="fundamental_fraction"):
        _config(
            multigrid_root,
            curriculum=CurriculumConfig(stages=stages),
            measurement=MeasurementConfig(heterogeneous_orders=True),
        )


def test_a_switchable_topology_is_rejected_by_the_same_mechanism(multigrid_root):
    with pytest.raises(ValueError, match="data.topology"):
        _config(
            multigrid_root,
            data=_data(
                multigrid_root,
                topology=TopologyConfig(
                    switch_ids=[1],
                    variants=[TopologyVariantConfig(name="closed")],
                ),
            ),
        )


def test_default_valued_ignored_fields_pass(multigrid_root):
    """Only a NON-DEFAULT value is a lie; the defaults describe what actually happens."""
    cfg = _config(multigrid_root)
    assert cfg.data.val_fraction == DataConfig().val_fraction
    assert cfg.data.split_seed == DataConfig().split_seed
    assert cfg.physics.calibrate == []


def test_the_structural_model_check_still_fires(multigrid_root):
    with pytest.raises(ValueError, match="dnn"):
        _config(multigrid_root, model=DNNConfig(hidden_dims=[8]))


def test_a_single_grid_config_is_untouched_by_the_contract(multigrid_root):
    """None of this applies without ``data.multigrid`` — the single-grid path is unchanged."""
    cfg = ExperimentConfig(
        name="single",
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(multigrid_root),
            val_fraction=0.25,
            test_fraction=0.2,
            split_seed=7,
        ),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"]),
        physics=PhysicsConfig(calibration_weight=1e-3),
        curriculum=CurriculumConfig(stages=[]),
    )
    assert cfg.data.val_fraction == pytest.approx(0.25)


# =============================================================================
# The dropped DataConfig fields: split by grid, orders as a real subset
# =============================================================================
def test_the_source_serves_every_corpus_order_by_default(multigrid_root):
    source = build_data_source(_data(multigrid_root))
    assert source.orders == list(ORDERS)
    item = source.train_dataset()[0]
    assert int(item["v"].shape[-2]) == len(ORDERS)


def test_requested_orders_select_a_subset_of_the_corpus(multigrid_root):
    """``orders`` is applied to the DATA, not just relabelled onto the harmonic axis."""
    full = build_data_source(_data(multigrid_root))
    subset = build_data_source(_data(multigrid_root), orders=[1, 5])

    assert subset.orders == [1, 5]
    full_item = full.train_dataset()[0]
    sub_item = subset.train_dataset()[0]
    assert int(sub_item["v"].shape[-2]) == 2
    keep = torch.tensor([ORDERS.index(1), ORDERS.index(5)])
    assert torch.equal(sub_item["v"], full_item["v"].index_select(-2, keep))
    # the graph follows the selected axis too
    assert subset.grids[0].graph.x.shape[0] == full.grids[0].graph.x.shape[0]


def test_an_order_the_corpus_lacks_is_named_in_the_error(multigrid_root):
    with pytest.raises(ValueError) as excinfo:
        build_data_source(_data(multigrid_root), orders=[1, 3, 7, 11])
    message = str(excinfo.value)
    assert "7" in message and "11" in message


def test_the_split_stays_the_manifests_grid_level_one(multigrid_root):
    """No scenario-level split leaks in: every scenario of a train member trains."""
    source = build_data_source(_data(multigrid_root))
    train_keys = {source.bundle(i).key for i in source.split_indices("train")}
    val_keys = {source.bundle(i).key for i in source.split_indices("val")}
    assert train_keys and val_keys and not (train_keys & val_keys)
    per_member = [
        len(source.bundle(i).source.train_dataset())
        for i in source.split_indices("train")
    ]
    assert len(source.train_dataset()) == sum(per_member)


# =============================================================================
# The supervised blend is NORMALIZED on the multi-grid path too
# =============================================================================
@pytest.fixture(scope="module")
def source(multigrid_root):
    return MultiGridDataSource(multigrid_root, placement_fraction=0.3)


def _module(root, source, *, split: float, normalize: bool) -> SEModule:
    cfg = _config(
        root,
        physics=PhysicsConfig(decode="injection"),
        loss=LossConfig(
            injection_state_loss_split=split,
            injection_state_loss_normalize=normalize,
        ),
        train=TrainConfig(
            accelerator="cpu",
            devices=1,
            dtype="complex128",
            enable_progress_bar=False,
        ),
    )
    encoding = TrigEncoding()
    bank = MultiGridPhysicsBank(source, dtype=torch.complex128)
    module = SEModule(
        cfg,
        normalizer=fit_multigrid_normalizer(source, encoding, cfg),
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        inj_normalizer=fit_multigrid_injection_normalizer(bank, source, encoding, cfg),
    )
    module.set_masker(build_masker("random", observed_fraction=0.5))
    return module


def _batch(source):
    dataset = source.train_dataset()
    items = [dataset[i] for i in range(min(3, len(dataset)))]
    return MultiGridCollate(source)(items)


def test_the_blend_honors_injection_state_loss_normalize(multigrid_root, source):
    """The same split must not mean a different loss on the two paths.

    Raw, the injection residual runs orders of magnitude above the voltage one, so a nominal
    0.1 realizes almost all of the loss on the injections. Normalized, the injection term is
    rescaled onto the voltage term's scale, so the blend stays the size of the voltage-only
    loss and the split is the share it says it is.
    """
    split = 0.1
    normalized = _module(multigrid_root, source, split=split, normalize=True)
    raw = _module(multigrid_root, source, split=split, normalize=False)
    raw.load_state_dict(
        normalized.state_dict()
    )  # the SAME weights, only the blend differs
    batch = _batch(source)

    torch.manual_seed(0)
    loss_n, *_ = normalized._step(batch, train=False)
    torch.manual_seed(0)
    loss_r, *_ = raw._step(batch, train=False)

    parts_n = normalized._loss_components
    parts_r = raw._loss_components
    assert parts_n.keys() == parts_r.keys() == {"voltage", "injection_state"}
    # identical terms (same weights, same batch) -> the difference IS the normalization
    assert float(parts_n["voltage"]) == pytest.approx(float(parts_r["voltage"]))
    assert float(parts_n["injection_state"]) == pytest.approx(
        float(parts_r["injection_state"])
    )
    assert float(loss_n) != pytest.approx(float(loss_r))

    # normalized: the scaled term equals the voltage term, so the blend is that term
    assert float(loss_n) == pytest.approx(float(parts_n["voltage"]), rel=1e-9)
    # raw: the plain (1-s)V + sI blend the pre-normalization runs produced
    assert float(loss_r) == pytest.approx(
        (1.0 - split) * float(parts_r["voltage"])
        + split * float(parts_r["injection_state"]),
        rel=1e-9,
    )
    # ... and the realized share is what the split promises only in the normalized case
    assert float(normalized._loss_share) == pytest.approx(split, rel=1e-9)
    assert float(raw._loss_share) > 0.5


def test_the_share_is_logged_on_a_multigrid_validation_step(multigrid_root, source):
    """``val_injection_share`` is the metric that reveals a mis-scaled blend — it must exist."""
    module = _module(multigrid_root, source, split=0.1, normalize=True)
    logged: dict[str, float] = {}
    module.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    module.validation_step(_batch(source), 0)

    assert "val_injection_share" in logged
    assert logged["val_injection_share"] == pytest.approx(0.1, rel=1e-6)
    assert {"val_voltage_loss", "val_injection_state_loss"} <= set(logged)


def test_no_share_without_the_split(multigrid_root, source):
    """Voltage-only supervision logs no share (there is no second term to report)."""
    module = _module(multigrid_root, source, split=0.0, normalize=True)
    logged: dict[str, float] = {}
    module.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    module.validation_step(_batch(source), 0)

    assert module._loss_share is None
    assert "val_injection_share" not in logged
    assert "val_voltage_loss" in logged
