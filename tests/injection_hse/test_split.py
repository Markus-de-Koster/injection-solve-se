"""Three-way (train/val/test) split + non-converged dropping.

Proves the two-way (val/train) partition stays byte-identical when ``test_fraction=0`` (so no
existing config / checkpoint changes), that a carved test split is disjoint by scenario and
never shrinks val, that non-converged scenarios are dropped by default, and that the test split
is NEVER read during training.
"""

from __future__ import annotations

import dataclasses

import torch

import injection_hse.data.scenarios as scenarios
from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    TemporalConfig,
    TrainConfig,
)
from injection_hse.data import ParquetDataset
from injection_hse.data.sources import _split_indices

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _indices(ds) -> list[int]:
    return sorted(int(ds[i]["index"]) for i in range(len(ds)))


def test_split_indices_byte_identical_when_no_test():
    """``test_fraction=0`` reproduces the historical two-way val/train split exactly."""
    n, vf, seed = 256, 0.1, 0
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = round(vf * n)
    val_ref = perm[:n_val].sort().values
    train_ref = perm[n_val:].sort().values

    train, val, test = _split_indices(n, vf, seed)  # test_fraction defaults to 0.0
    assert torch.equal(val, val_ref)
    assert torch.equal(train, train_ref)
    assert test.numel() == 0


def test_parquet_two_way_split_byte_identical(cigre_grid, se_dataset):
    """The parquet source's val/train indices are unchanged at ``test_fraction=0``."""
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(256, generator=gen)
    n_val = round(0.1 * 256)
    assert _indices(ds.val_dataset()) == sorted(perm[:n_val].tolist())
    assert _indices(ds.train_dataset()) == sorted(perm[n_val:].tolist())
    assert len(ds.test_dataset()) == 0


def test_three_way_split_disjoint_and_val_unchanged(cigre_grid, se_dataset):
    """A test split is disjoint by scenario, covers everything, and leaves val identical."""
    two = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    three = ParquetDataset(
        str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0, test_fraction=0.2
    )
    tr = set(_indices(three.train_dataset()))
    va = set(_indices(three.val_dataset()))
    te = set(_indices(three.test_dataset()))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert tr | va | te == set(range(256))
    assert len(te) == round(0.2 * 256)
    # carving the test set off the SAME permutation leaves the leading val slice untouched
    assert _indices(three.val_dataset()) == _indices(two.val_dataset())
    # ...and only shrinks the training set
    assert tr < set(_indices(two.train_dataset()))


def _dataset_with_failures(se_dataset, failed):
    """Monkeypatch-able ``read_dataset`` returning the real dataset tagged with failures."""
    loaded = scenarios.read_dataset(str(se_dataset))
    return dataclasses.replace(loaded, failed_scenarios=tuple(failed))


def test_failed_scenarios_dropped_by_default(cigre_grid, se_dataset, monkeypatch):
    failed = (3, 7, 42)
    patched = _dataset_with_failures(se_dataset, failed)
    monkeypatch.setattr(scenarios, "read_dataset", lambda _p: patched)

    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    assert ds.n_dropped_failed == len(failed)
    present = set(_indices(ds.train_dataset())) | set(_indices(ds.val_dataset()))
    assert present.isdisjoint(failed)  # no failed scenario in any split
    assert len(present) == 256 - len(failed)


def test_keep_failed_retains_them(cigre_grid, se_dataset, monkeypatch):
    failed = (3, 7, 42)
    patched = _dataset_with_failures(se_dataset, failed)
    monkeypatch.setattr(scenarios, "read_dataset", lambda _p: patched)

    ds = ParquetDataset(
        str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0, keep_failed=True
    )
    assert ds.n_dropped_failed == 0
    present = set(_indices(ds.train_dataset())) | set(_indices(ds.val_dataset()))
    assert failed[0] in present and present == set(range(256))


def test_no_failures_leaves_split_byte_identical(cigre_grid, se_dataset):
    """A dataset with no failures is unaffected by the default drop (byte-identical)."""
    ds = ParquetDataset(str(se_dataset), grid=cigre_grid, val_fraction=0.1, seed=0)
    assert ds.n_dropped_failed == 0
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(256, generator=gen)
    n_val = round(0.1 * 256)
    assert _indices(ds.val_dataset()) == sorted(perm[:n_val].tolist())


class _SpySource:
    """Delegates to a real DataSource, counting how often the TEST split is requested."""

    def __init__(self, inner):
        self._inner = inner
        self.test_calls = 0

    def test_dataset(self):
        self.test_calls += 1
        return self._inner.test_dataset()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_test_split_never_read_during_training(
    cigre_grid, se_dataset, tmp_path, monkeypatch
):
    """The held-out test split is untouched by training / early stop / checkpointing."""
    # the run entry point and the data-source call it resolves live in the same module,
    # so the spy below has to replace the name THERE
    import injection_hse.train.loop as train_mod

    cfg = ExperimentConfig(
        name="split-audit",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(se_dataset),
            batch_size=32,
            val_fraction=0.2,
            test_fraction=0.2,
            num_workers=0,
        ),
        model=TemporalConfig(spatial=GNNConfig(hidden_dim=16, num_layers=1)),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random_50",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        optim=OptimConfig(name="adamw", lr=2e-3, scheduler="none"),
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )

    real_build = train_mod.build_data_source
    spy_holder = {}

    def _spy_build(data_cfg, *, grid):
        spy = _SpySource(real_build(data_cfg, grid=grid))
        spy_holder["spy"] = spy
        return spy

    monkeypatch.setattr(train_mod, "build_data_source", _spy_build)
    module = train_mod.train_experiment(cfg, grid=cigre_grid)
    assert spy_holder["spy"].test_calls == 0

    # the held-out split is available for an explicit, post-training evaluation (and only then)
    monkeypatch.setattr(train_mod, "build_data_source", real_build)
    from injection_hse.evaluation import predict_dataset

    pred = predict_dataset(
        module,
        source_or_path=str(se_dataset),
        grid=cigre_grid,
        measurement=cfg.measurement,
        split="test",
        device="cpu",
    )
    # non-empty (test_fraction=0.2) and disjoint by scenario from what training saw
    assert pred.n_samples == round(0.2 * 256)
