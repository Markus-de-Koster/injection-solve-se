"""``TrainConfig.deterministic``: the reproducibility switch reaches torch and Lightning.

A seeded run is not a reproducible run: TF32 matmuls and non-deterministic CUDA kernels
round differently between executions. The flag must (1) reach every ``L.Trainer`` the
StageManager builds, (2) put torch into deterministic-algorithms mode for the duration of
training, and (3) force the float32 matmul precision off the TF32 path. Off must remain
byte-identical to a config written before the field existed.
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    MaskStageConfig,
    MeasurementConfig,
    OptimConfig,
    TrainConfig,
)

ORDERS = [1, 3, 5, 7, 9, 11, 13]


@pytest.fixture()
def _restore_determinism_state():
    matmul = torch.get_float32_matmul_precision()
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision(matmul)


def _config(dataset_path, tmp_path, deterministic) -> ExperimentConfig:
    return ExperimentConfig(
        name="determinism",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset_path),
            batch_size=64,
            val_fraction=0.2,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[32], dropout=0.0),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
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
        train=TrainConfig(
            precision="32",
            accelerator="cpu",
            devices=1,
            mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
            checkpoint_dir=str(tmp_path / "ckpt"),
            deterministic=deterministic,
        ),
    )


def _run(cfg, cigre_grid, monkeypatch) -> list[dict]:
    import lightning as L

    from injection_hse import train as train_mod

    seen: list[dict] = []
    orig_trainer = L.Trainer

    def _trainer(*args, **kwargs):
        seen.append(dict(kwargs))
        kwargs["num_sanity_val_steps"] = 0
        return orig_trainer(*args, **kwargs)

    monkeypatch.setattr(train_mod.L, "Trainer", _trainer)
    train_mod.train_experiment(cfg, grid=cigre_grid)
    return seen


def test_deterministic_reaches_trainer_and_torch(
    cigre_grid, se_dataset, tmp_path, monkeypatch, _restore_determinism_state
):
    cfg = _config(se_dataset, tmp_path, deterministic=True)
    seen = _run(cfg, cigre_grid, monkeypatch)
    assert seen and all(k["deterministic"] is True for k in seen)
    # process-wide state after training: deterministic kernels on, TF32 path off
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision() == "highest"


def test_warn_mode_passes_through(
    cigre_grid, se_dataset, tmp_path, monkeypatch, _restore_determinism_state
):
    cfg = _config(se_dataset, tmp_path, deterministic="warn")
    seen = _run(cfg, cigre_grid, monkeypatch)
    assert seen and all(k["deterministic"] == "warn" for k in seen)


def test_off_leaves_global_state_untouched(
    cigre_grid, se_dataset, tmp_path, monkeypatch, _restore_determinism_state
):
    torch.use_deterministic_algorithms(False)
    cfg = _config(se_dataset, tmp_path, deterministic=False)
    seen = _run(cfg, cigre_grid, monkeypatch)
    # False maps to None: Lightning must not touch torch's global state either way
    assert seen and all(k["deterministic"] is None for k in seen)
    assert not torch.are_deterministic_algorithms_enabled()
