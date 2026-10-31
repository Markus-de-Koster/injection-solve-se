"""What a checkpoint says about itself: the code that wrote it, and the config that trained it.

Two guards on the run record:

- ``on_save_checkpoint`` stamps the CODE provenance (commit + dirty flag + library versions).
  Config and seed alone do not identify a run — the code between two identical configs may
  have changed — and reading it back stays optional, so older checkpoints still load.
- the :class:`~injection_hse.train.StageManager` hands its own config to the module it runs (so a
  two-task curriculum's sequence checkpoints record the sequence pass). That is only honest for fields the
  module reads per step; a field the module DERIVED state from at construction would leave it
  behaving as before while reporting the new config, so the swap is checked and names the
  offending field.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    EncodingConfig,
    ExperimentConfig,
    GNNConfig,
    LossConfig,
    MaskStageConfig,
    MeasurementConfig,
    NoiseConfig,
    OptimConfig,
    TrainConfig,
)
from injection_hse.train import SEModule, StageManager, _assert_cfg_swappable

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _cfg(dataset, **overrides) -> ExperimentConfig:
    fields = dict(
        name="provenance",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset),
            batch_size=32,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=8, num_layers=1),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="masked",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=1e-3, scheduler="none"),
        train=TrainConfig(precision="32", accelerator="cpu", devices=1),
    )
    fields.update(overrides)
    return ExperimentConfig(**fields)


@pytest.fixture
def module(cigre_grid, se_dataset):
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import _to_nhc

    cfg = _cfg(se_dataset)
    source = build_data_source(cfg.data, grid=cigre_grid)
    encoding = build_encoding(cfg.encoding.kind)
    states = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    return SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=HarmonicNormalizer().fit(_to_nhc(encoding.encode(states))),
        graph=source.graph(),
        base_v=base_voltage_per_row(cigre_grid),
        grid=cigre_grid,
    )


# =============================================================================
# Code provenance
# =============================================================================
def test_the_checkpoint_records_the_code_that_wrote_it(module):
    import json

    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)

    provenance = checkpoint["pgml_provenance"]
    assert set(provenance) >= {
        "git_sha",
        "git_dirty",
        "git_source",
        "pgml_version",
        "torch_version",
    }
    assert provenance["torch_version"] == torch.__version__
    json.dumps(provenance)  # JSON-ready: it travels into run manifests unchanged


def test_a_checkpoint_without_provenance_still_loads(module, cigre_grid, tmp_path):
    """Reading the stamp is optional — a checkpoint written before it existed must reload."""
    from injection_hse.normalization import HarmonicNormalizer

    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    del checkpoint["pgml_provenance"]
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)

    restored = SEModule.from_checkpoint(path, grid=cigre_grid, device="cpu")
    assert isinstance(restored.normalizer, HarmonicNormalizer)
    assert restored.cfg.name == module.cfg.name


# =============================================================================
# The StageManager's config swap
# =============================================================================
def test_run_behaviour_fields_may_differ_between_passes(se_dataset):
    """The two-task curriculum's legitimate case: name, blend, stages, split fractions."""
    a = _cfg(se_dataset)
    b = _cfg(
        se_dataset,
        name="pass-b",
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=8,
            val_fraction=0.3,
            num_workers=0,
        ),
        loss=LossConfig(injection_loss_weight=5.0, injection_state_loss_split=0.2),
        optim=OptimConfig(name="adamw", lr=1e-4, scheduler="cosine"),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="placement", kind="placement", target_loss=1e-9, max_epochs=1
                )
            ]
        ),
    )
    _assert_cfg_swappable(a, b)  # does not raise


@pytest.mark.parametrize(
    "overrides, field",
    [
        (dict(model=GNNConfig(hidden_dim=32, num_layers=1)), "model"),
        (dict(orders=[1, 3, 5]), "orders"),
        (dict(encoding=EncodingConfig(kind="cartesian")), "encoding.kind"),
        (dict(encoding=EncodingConfig(time_features=True)), "encoding.time_features"),
        (dict(seed=13), "seed"),
        (
            dict(train=TrainConfig(accelerator="cpu", devices=1, dtype="complex128")),
            "train.dtype",
        ),
        (
            dict(loss=LossConfig(magnitude_term_weight=0.5)),
            "loss.magnitude_term_weight",
        ),
        (dict(loss=LossConfig(predict_injection=True)), "loss.predict_injection"),
        (dict(noise=NoiseConfig(enabled=True)), "noise"),
        (
            dict(
                measurement=MeasurementConfig(
                    voltage_nodes=["Bus R4"], measure_current=True
                )
            ),
            "measurement.measure_current",
        ),
    ],
)
def test_a_behaviour_deriving_field_cannot_be_swapped(se_dataset, overrides, field):
    a = _cfg(se_dataset)
    b = _cfg(se_dataset, **overrides)
    with pytest.raises(ValueError) as excinfo:
        _assert_cfg_swappable(a, b)
    assert field in str(excinfo.value)


def test_the_stage_manager_refuses_the_swap_before_training(
    module, cigre_grid, se_dataset, tmp_path
):
    """The guard fires on ``run``, before a single epoch is spent."""
    from injection_hse.data import build_data_source
    from injection_hse.train import make_datamodule

    other = _cfg(
        se_dataset,
        name="other",
        model=GNNConfig(hidden_dim=32, num_layers=1),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    datamodule = make_datamodule(
        build_data_source(other.data, grid=cigre_grid),
        batch_size=other.data.batch_size,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="model"):
        StageManager(other).run(module, datamodule, logger=False)
    # the module kept ITS config: a refused swap leaves nothing half-applied
    assert module.cfg.name == "provenance"


def test_the_stage_manager_adopts_a_compatible_config(module, se_dataset, tmp_path):
    """A compatible config IS adopted (the checkpoint then records the pass that ran)."""
    incoming = _cfg(
        se_dataset,
        name="pass-b",
        loss=LossConfig(injection_loss_weight=5.0),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )
    manager = StageManager(incoming)
    _assert_cfg_swappable(module.cfg, manager.cfg)  # what run() checks first
    module.cfg = manager.cfg
    checkpoint: dict = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["pgl_config"]["name"] == "pass-b"
