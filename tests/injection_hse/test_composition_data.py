"""Composed coherent data through the injection_hse data layer + the composed workflow curriculum.

A COMPOSED coherent dataset (:class:`injection_hse.data.scenarios.CompositionConfig`) decomposes each covered
load into statistical member devices whose per-step activity moves the fundamental power AND the
harmonic spectrum jointly. Unlike the fingerprint data it carries an absolute time axis (the
periodic time features) and per-class attribution labels — but it records those labels, not a
``time_unix_s`` array, so the injection_hse data layer DERIVES the absolute axis from the config.

These checks verify (1) the ``ParquetDataset`` surfaces the derived ``time_unix_s`` + the
``composition_labels()`` accessor (and that a snapshot / fingerprint sequence carries neither),
and (2) the example workflow's composed sequence/C generation, the time-feature flag, and the
separate cosine-anneal stage assemble correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from injection_hse.data import CompositionLabels, ParquetDataset, StateCollate

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "run" / "examples" / "injection_hse"))

ORDERS = [1, 3, 5]
INJECTED = [3, 5]
N_SCEN, N_STEPS = 4, 4


@pytest.fixture(scope="module")
def composed_dataset(cigre_grid, tmp_path_factory):
    """A tiny COMPOSED coherent dataset (``[B, T, H, N]`` + attribution labels) -> directory."""
    from pgml.scenarios import (
        Selector,
        run_scenarios,
        write_dataset,
    )
    from injection_hse.data.scenarios import (
        CoherentSpectrumConfig,
        CompositionConfig,
    )

    out = tmp_path_factory.mktemp("composed_ds")
    cfg = CoherentSpectrumConfig(
        name="comp",
        selector=Selector(component="load"),
        orders=INJECTED,
        n_steps=N_STEPS,
        n_scenarios=N_SCEN,
        seed=0,
        step_size_s=900.0,
        composition=CompositionConfig(),
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


# =============================================================================
# data layer: derived time axis + composition-label accessor
# =============================================================================
def test_composed_dataset_derives_time_axis(cigre_grid, composed_dataset):
    ds = ParquetDataset(str(composed_dataset), grid=cigre_grid, orders=ORDERS)
    tu = ds.time_unix_s
    # composed data has NO time_unix_s sample -> derived from start_time + step_size_s
    assert tu is not None and tu.shape == (N_STEPS,)
    assert tu.dtype == torch.float64
    # 2024-06-21T00:00:00Z epoch + a 900 s grid
    assert float(tu[0]) == pytest.approx(1_718_928_000.0)
    assert torch.allclose(
        tu[1:] - tu[:-1], torch.full((N_STEPS - 1,), 900.0, dtype=torch.float64)
    )


def test_composition_labels_accessor(cigre_grid, composed_dataset):
    ds = ParquetDataset(str(composed_dataset), grid=cigre_grid, orders=ORDERS)
    cl = ds.composition_labels()
    assert isinstance(cl, CompositionLabels)
    n_agg = int(cl.agg_ids.numel())
    n_class = len(cl.class_names)
    assert n_agg > 0 and n_class == 7  # the built-in device classes
    assert cl.class_p_w.shape == (N_SCEN, n_agg, n_class, N_STEPS)
    assert cl.class_active.shape == (N_SCEN, n_agg, n_class, N_STEPS)
    assert cl.class_active.dtype == torch.int64
    assert cl.cap_binding.shape[:2] == (N_SCEN, n_agg)
    assert cl.roster_p_rated.shape[:2] == (n_agg, n_class)
    assert cl.class_names[0] == "base_linear"  # the fixed class ordering
    # the label time axis matches the dataset's derived axis
    assert torch.equal(cl.time_unix_s, ds.time_unix_s)


def test_collate_carries_composed_time_axis(cigre_grid, composed_dataset):
    ds = ParquetDataset(str(composed_dataset), grid=cigre_grid, orders=ORDERS)
    train = ds.train_dataset()
    items = [train[i] for i in range(min(3, len(train)))]
    batch = StateCollate()(items)
    assert "time_unix_s" in batch and batch["time_unix_s"].shape == (N_STEPS,)
    assert torch.equal(batch["time_unix_s"], ds.time_unix_s)


def test_snapshot_and_fingerprint_have_no_composition(
    cigre_grid, se_dataset, se_sequence_dataset
):
    # a randomized SNAPSHOT: no time axis, no composition labels
    snap = ParquetDataset(str(se_dataset), grid=cigre_grid)
    assert snap.time_unix_s is None and snap.composition_labels() is None
    # a fingerprint SEQUENCE (no start_time): timeless, no composition labels
    seq = ParquetDataset(
        str(se_sequence_dataset), grid=cigre_grid, orders=[1, 3, 5, 7, 9, 11, 13]
    )
    assert seq.time_unix_s is None and seq.composition_labels() is None


# =============================================================================
# example workflow: composed sequence/C generation + curriculum assembly
# =============================================================================
def _wf():
    import train_se_full_workflow as wf

    return wf


def _gen_args(**over):
    base = dict(
        coherent_mode="composed",
        n_scenarios=N_SCEN,
        n_steps=N_STEPS,
        n_modes=2,
        dwell=0.9,
        seed=0,
        coherent_step_s=900.0,
        n_profile_scenarios=N_SCEN,
        profile_steps=6,
        profile_step_s=3600.0,
        profile_start="2024-06-21T00:00:00",
        chunk_size=None,
        scenario_coherent=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_workflow_generate_coherent_composed(cigre_grid, tmp_path):
    wf = _wf()
    path = wf.generate_coherent_dataset(
        cigre_grid,
        ORDERS,
        INJECTED,
        _gen_args(),
        tmp_path,
        torch.device("cpu"),
        torch.complex64,
    )
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS)
    # composed the sequence pass carries an absolute time axis + attribution labels
    assert ds.time_unix_s is not None and ds.time_unix_s.shape == (N_STEPS,)
    assert ds.composition_labels() is not None


def test_workflow_generate_coherent_fingerprint(cigre_grid, tmp_path):
    wf = _wf()
    path = wf.generate_coherent_dataset(
        cigre_grid,
        ORDERS,
        INJECTED,
        _gen_args(coherent_mode="fingerprint"),
        tmp_path,
        torch.device("cpu"),
        torch.complex64,
    )
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS)
    # fingerprint the sequence pass is timeless (no start_time) and has no composition labels
    assert ds.time_unix_s is None
    assert ds.composition_labels() is None


def test_workflow_generates_the_profiled_composed_pass(cigre_grid, tmp_path):
    wf = _wf()
    path = wf.generate_profile_dataset(
        cigre_grid,
        ORDERS,
        INJECTED,
        _gen_args(),
        tmp_path,
        torch.device("cpu"),
        torch.complex64,
    )
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS)
    # the profiled-sequence pass: composition (loads) + profile (uncomposed appliances) -> BOTH label sets
    assert ds.time_unix_s is not None and ds.time_unix_s.shape == (6,)
    assert ds.composition_labels() is not None
    assert ds.profile_factors() is not None


def test_anneal_stage_assembly():
    from injection_hse import curriculum

    wf = _wf()
    st = wf._anneal_stage(SimpleNamespace(anneal_epochs=150, placement_lr=1e-4))
    assert st.name == "placement_anneal" and st.kind == "placement"
    assert st.scheduler == "cosine" and st.max_epochs == 150
    assert st.lr == pytest.approx(1e-4)
    # a negligible target so the stage runs its full cosine budget, not an early target stop
    assert st.target_loss == curriculum.ANNEAL_TARGET_FLOOR
    # placement stage is constant-lr by default (the anneal owns the schedule) and comes first
    b_stages = wf._sequence_stages(
        SimpleNamespace(
            target_full=1e-3,
            target_placement=1e-2,
            epochs=2,
            placement_lr=None,
            placement_scheduler=None,
        )
    )
    assert b_stages[-1].name == "placement" and b_stages[-1].scheduler is None
