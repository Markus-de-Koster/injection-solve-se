"""Layer grouping, the gradient-norm collector, and checkpoint weight deltas."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from injection_hse.analysis import (
    GradientNormCollector,
    checkpoint_weight_deltas,
    parameter_groups,
    read_gradient_norms,
)


def test_parameter_groups_temporal_gnn(trained_module):
    groups = parameter_groups(trained_module)
    names = set(groups)
    assert {"temporal.memory", "temporal.context_proj", "spatial.encoder"} <= names
    assert {"spatial.conv_0", "spatial.conv_1", "spatial.head"} <= names
    # every parameter lands in exactly one group
    covered = [p for group in groups.values() for p in group]
    assert sorted(covered) == sorted(n for n, _ in trained_module.named_parameters())


class _Tiny(L.LightningModule):
    """A minimal LightningModule with two parameter groups for the collector."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 4)
        self.head = torch.nn.Linear(4, 1)

    def training_step(self, batch, _):
        return self.head(self.encoder(batch[0])).sum()

    def validation_step(self, batch, _):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def _loader():
    return DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)


def test_gradient_norm_collector_spans_fits(tmp_path):
    collector = GradientNormCollector(every_n_steps=1)
    module = _Tiny()
    for _ in range(2):  # two "stages"
        trainer = L.Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            callbacks=[collector],
            num_sanity_val_steps=0,
        )
        trainer.fit(module, _loader(), _loader())

    df = collector.to_frame()
    # 2 stages x 2 epochs x 2 optimizer steps (2 batches, no accumulation) x 2 groups
    assert df.height == 16
    assert set(df["group"].to_list()) == {"encoder", "head"}
    assert sorted(set(df["stage_index"].to_list())) == [0, 1]
    assert np.all(np.isfinite(df["grad_norm"].to_numpy()))
    assert np.all(df["param_norm"].to_numpy() > 0)
    # the step counter is monotonic across the two fits
    steps = df.filter(df["group"] == "encoder")["step"].to_list()
    assert steps == sorted(steps) and len(set(steps)) == len(steps)

    path = collector.write_parquet(tmp_path / "gradient_norms.parquet")
    assert read_gradient_norms(path).height == 16


def test_accumulation_samples_at_optimizer_step(tmp_path):
    # 4 micro-batches/epoch, accumulate 2 -> 2 optimizer steps/epoch; the collector
    # must sample per OPTIMIZER step (not per backward), so one epoch yields 2 x 2 groups.
    collector = GradientNormCollector(every_n_steps=1)
    module = _Tiny()
    loader = DataLoader(TensorDataset(torch.randn(16, 2)), batch_size=4)
    trainer = L.Trainer(
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
        callbacks=[collector],
        num_sanity_val_steps=0,
    )
    trainer.fit(module, loader, _loader())
    df = collector.to_frame()
    # 4 batches / accumulate 2 = 2 optimizer steps; x 2 groups
    assert df.height == 4
    assert sorted(df.filter(df["group"] == "encoder")["step"].to_list()) == [0, 1]


def test_checkpoint_weight_deltas(trained_run):
    _, _, run_dir = trained_run
    ckpts = [
        Path(run_dir) / "checkpoints" / "stage_random_50.ckpt",
        Path(run_dir) / "checkpoints" / "stage_placement.ckpt",
    ]
    frame = checkpoint_weight_deltas(ckpts)
    assert set(frame["stage"].to_list()) == {"placement"}
    assert {"temporal.memory", "spatial.encoder", "spatial.head"} <= set(
        frame["group"].to_list()
    )
    deltas = frame["rel_delta"].to_numpy()
    assert np.all(np.isfinite(deltas)) and np.all(deltas >= 0)
    # one epoch of training moves every trained group at least a little
    assert deltas.max() > 0
