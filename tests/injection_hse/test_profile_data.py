"""Profile-sequence data: ``time_unix_s`` + profile-factor ground truth through the data layer.

A PROFILED coherent dataset (``CoherentSpectrumConfig`` with a ``LoadProfileConfig``) stamps
each step with an absolute ``time_unix_s`` and records the per-device profile factors. These
checks verify the injection_hse data layer surfaces both (the ``ParquetDataset`` accessors + the
``StateCollate`` batch key) and that a non-profiled / snapshot dataset carries neither. The
example workflow's profiled-sequence generation + config assembly are exercised too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from injection_hse.data import ParquetDataset, ProfileFactors, StateCollate

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "run" / "examples" / "injection_hse"))

ORDERS = [1, 3, 5]
INJECTED = [3, 5]
N_SCEN, N_STEPS = 4, 6


@pytest.fixture(scope="module")
def profile_dataset(cigre_grid, tmp_path_factory):
    """A tiny PROFILED coherent dataset (``[B, T, H, N]`` + ``time_unix_s``) -> directory path."""
    from pgml.scenarios import (
        Selector,
        run_scenarios,
        write_dataset,
    )
    from injection_hse.data.scenarios import (
        CoherentSpectrumConfig,
        LoadProfileConfig,
    )

    out = tmp_path_factory.mktemp("profile_ds")
    cfg = CoherentSpectrumConfig(
        name="prof",
        selector=Selector(component="load"),
        orders=INJECTED,
        n_steps=N_STEPS,
        n_scenarios=N_SCEN,
        n_modes=2,
        seed=0,
        step_size_s=3600.0,
        profile=LoadProfileConfig(),
        start_time="2024-06-21T00:00:00",
    )
    result = run_scenarios(
        cigre_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


def test_parquet_exposes_time_and_profile_factors(cigre_grid, profile_dataset):
    ds = ParquetDataset(str(profile_dataset), grid=cigre_grid, orders=ORDERS)
    # absolute per-step epoch axis
    tu = ds.time_unix_s
    assert tu is not None and tu.shape == (N_STEPS,)
    assert tu.dtype == torch.float64  # epoch seconds kept at full precision
    # the hourly step grid: 3600 s apart, anchored at the ISO start
    assert torch.allclose(
        tu[1:] - tu[:-1], torch.full((N_STEPS - 1,), 3600.0, dtype=torch.float64)
    )

    pf = ds.profile_factors()
    assert isinstance(pf, ProfileFactors)
    n_dev = pf.device_ids.numel()
    assert n_dev > 0
    assert pf.factor.shape == (N_SCEN, n_dev, N_STEPS)
    assert torch.equal(pf.time_unix_s, tu)
    assert (pf.factor >= 0).all()  # profile factors are non-negative multipliers


def test_snapshot_and_sequence_have_no_time(
    cigre_grid, se_dataset, se_sequence_dataset
):
    # a randomized SNAPSHOT dataset: no time, no profile factors
    snap = ParquetDataset(str(se_dataset), grid=cigre_grid)
    assert snap.time_unix_s is None
    assert snap.profile_factors() is None
    # a NON-profiled coherent SEQUENCE dataset: still no absolute timestamp (only a relative
    # time_s, which is NOT time_unix_s) and no profile factors
    seq = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, orders=[1, 3, 5, 7, 9, 11, 13]
    )
    assert seq.time_unix_s is None
    assert seq.profile_factors() is None


def test_collate_carries_time_unix_s(cigre_grid, profile_dataset):
    ds = ParquetDataset(str(profile_dataset), grid=cigre_grid, orders=ORDERS)
    train = ds.train_dataset()
    items = [train[i] for i in range(min(3, len(train)))]
    assert all("time_unix_s" in it for it in items)
    batch = StateCollate()(items)
    # the shared step axis is carried once (not stacked over the batch)
    assert "time_unix_s" in batch
    assert batch["time_unix_s"].shape == (N_STEPS,)
    assert torch.equal(batch["time_unix_s"], ds.time_unix_s)


# =============================================================================
# example workflow: profiled-sequence generation + config assembly
# =============================================================================
def _wf():
    import train_se_full_workflow as wf

    return wf


def test_profiled_sequence_stage_assembly():
    wf = _wf()
    args = SimpleNamespace(
        target_placement=1e-2, epochs=2, placement_lr=None, placement_scheduler=None
    )
    stages = wf._profiled_sequence_stages(args)
    assert len(stages) == 1
    assert stages[0].name == "profile_placement" and stages[0].kind == "placement"
    assert stages[0].target_loss == 1e-2 and stages[0].max_epochs == 2


def test_base_config_time_feature_flag():
    wf = _wf()
    base = SimpleNamespace(
        seed=0,
        batch_size=8,
        val_fraction=0.2,
        test_fraction=0.0,
        keep_failed=False,
        num_workers=0,
        decode="voltage",
        lr=1e-3,
        scheduler="none",
        grad_clip=0.0,
        effective_batch=None,
        devices=1,
        patience=0,
        min_delta=0.0,
        accelerator="cpu",
        mlflow_uri="file:./mlruns",
        no_progress=True,
    )
    stages = wf._profiled_sequence_stages(
        SimpleNamespace(
            target_placement=1e-2, epochs=1, placement_lr=None, placement_scheduler=None
        )
    )
    # time features OFF by default (byte-identical two-task run)
    off = SimpleNamespace(**vars(base))
    cfg_off = wf._base_config(off, "ds", ORDERS, stages, ".", "run-off")
    assert cfg_off.encoding.time_features is False
    # ON when the profiled-sequence pass is requested
    on = SimpleNamespace(**vars(base), time_features=True)
    cfg_on = wf._base_config(on, "ds", ORDERS, stages, ".", "run-on")
    assert cfg_on.encoding.time_features is True


def test_workflow_generate_profile_dataset(cigre_grid, tmp_path):
    wf = _wf()
    args = SimpleNamespace(
        n_profile_scenarios=N_SCEN,
        profile_steps=N_STEPS,
        profile_step_s=3600.0,
        profile_start="2024-06-21T00:00:00",
        n_modes=2,
        dwell=0.9,
        seed=0,
        chunk_size=None,
        scenario_coherent=None,
    )
    path = wf.generate_profile_dataset(
        cigre_grid,
        ORDERS,
        INJECTED,
        args,
        tmp_path,
        torch.device("cpu"),
        torch.complex64,
    )
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS)
    assert ds.time_unix_s is not None and ds.time_unix_s.shape == (N_STEPS,)
    assert ds.profile_factors() is not None


def test_build_curriculum_wires_the_profiled_pass(cigre_grid, profile_dataset):
    # the three-task plumbing: build_curriculum returns a profiled-sequence datamodule and the ONE module
    # carries the time-feature input width (Tasks A/B feed zeros). Construction only — no fit.
    wf = _wf()
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        EncodingConfig,
        ExperimentConfig,
        GNNConfig,
        MaskStageConfig,
        MeasurementConfig,
        PhysicsConfig,
        TemporalConfig,
        TrainConfig,
    )

    def _cfg(name):
        return ExperimentConfig(
            name=name,
            seed=0,
            orders=ORDERS,
            data=DataConfig(
                source="parquet",
                dataset_path=str(profile_dataset),
                batch_size=4,
                val_fraction=0.25,
            ),
            model=TemporalConfig(
                spatial=GNNConfig(hidden_dim=16, num_layers=2, conv="sage"),
                hidden_dim=16,
            ),
            encoding=EncodingConfig(time_features=True),
            physics=PhysicsConfig(decode="voltage"),
            measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="placement",
                        kind="placement",
                        target_loss=1e-9,
                        max_epochs=1,
                    )
                ]
            ),
            train=TrainConfig(accelerator="cpu"),
        )

    cfg_a, cfg_b, cfg_c = _cfg("A"), _cfg("B"), _cfg("C")
    module, dm_a, dm_b, dm_c, rows, _extras = wf.build_curriculum(
        cigre_grid, cfg_a, cfg_b, ORDERS, cfg_c=cfg_c
    )
    assert dm_c is not None
    assert module._n_time_features == 7
    assert rows.numel() > 0  # the placement rows resolved
    # dm_c is absent when the profiled-sequence pass is not requested
    *_, dm_c_none, _rows, _ex = wf.build_curriculum(cigre_grid, cfg_a, cfg_b, ORDERS)
    assert dm_c_none is None
