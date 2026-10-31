"""Per-sequence start times through the data layer and the time-feature input.

A coherent dataset generated with a ``start_time_range`` stores ``time_unix_s`` per scenario
(``[B, T]``). These checks verify the loader keeps it per scenario, items carry their own
scenario's axis, the collate stacks it to ``[B, T]``, and the module turns it into the
time-feature input PER SAMPLE (equal to the shared-axis path when all rows coincide).
"""

from __future__ import annotations

import torch

from injection_hse.data import ParquetDataset, StateCollate

ORDERS = [1, 3, 5]
N_SCEN, N_STEPS = 6, 3


def _composed_dataset(grid, out, **time):
    from pgml.scenarios import Selector, write_dataset

    from injection_hse.data.scenarios import (
        CoherentSpectrumConfig,
        CompositionConfig,
        run_scenarios,
    )

    cfg = CoherentSpectrumConfig(
        name="comp",
        selector=Selector(component="load"),
        orders=[3, 5],
        n_steps=N_STEPS,
        n_scenarios=N_SCEN,
        seed=0,
        step_size_s=900.0,
        composition=CompositionConfig(),
        **time,
    )
    result = run_scenarios(
        grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide"), cfg


def test_per_sequence_time_axis_through_the_data_layer(cigre_grid, tmp_path):
    from injection_hse.data.scenarios import StartTimeRange, sequence_time_axis

    rng = StartTimeRange(start="2024-01-01", end="2025-01-01")
    path, cfg = _composed_dataset(cigre_grid, tmp_path / "ds", start_time_range=rng)
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS, val_fraction=0.34)
    expected = sequence_time_axis(cfg)
    assert ds.time_unix_s.shape == (N_SCEN, N_STEPS)
    assert torch.equal(ds.time_unix_s, expected)
    assert ds.composition_labels().time_unix_s.shape == (N_SCEN, N_STEPS)

    train = ds.train_dataset()
    items = [train[i] for i in range(len(train))]
    for it in items:
        assert "time_unix_s" not in it and "sample_time_unix_s" not in it
        assert torch.equal(it["sequence_time_unix_s"], expected[int(it["index"])])
    batch = StateCollate()(items)
    assert batch["time_unix_s"].shape == (len(items), N_STEPS)
    assert torch.equal(batch["time_unix_s"], expected[batch["index"]])


def test_fixed_start_dataset_keeps_the_shared_axis(cigre_grid, tmp_path):
    path, _ = _composed_dataset(
        cigre_grid, tmp_path / "ds", start_time="2024-06-21T16:00:00"
    )
    ds = ParquetDataset(str(path), grid=cigre_grid, orders=ORDERS)
    assert ds.time_unix_s.shape == (N_STEPS,)
    items = [ds.train_dataset()[i] for i in range(2)]
    assert all("sequence_time_unix_s" not in it for it in items)
    assert StateCollate()(items)["time_unix_s"].shape == (N_STEPS,)


def test_module_reads_the_time_features_per_sample(cigre_grid):
    from injection_hse.data import build_graph
    from injection_hse.time_features import time_features

    from .test_time_features import _module, _seq_batch

    n = build_graph(cigre_grid).num_nodes
    v = _seq_batch(n, b=4, t=3)
    mod = _module(cigre_grid, v, time_features_on=True)
    tu = 1.7e9 + torch.arange(4, dtype=torch.float64)[:, None] * 5e6
    tu = tu + torch.arange(3, dtype=torch.float64) * 900.0  # [B, T], distinct rows
    feats = mod._build_node_time({"time_unix_s": tu}, (4, 3), torch.device("cpu"))
    assert feats.shape == (4, 3, 7)
    assert torch.equal(feats, time_features(tu).to(mod._real_dtype))

    # a [B, T] axis whose rows coincide estimates exactly like the shared [T] axis
    shared = tu[0]
    state = mod._mask_gen.get_state()
    out_shared = mod.infer({"v": v, "time_unix_s": shared})["pred_v"]
    mod._mask_gen.set_state(state)
    out_rows = mod.infer({"v": v, "time_unix_s": shared.expand(4, 3)})["pred_v"]
    assert torch.equal(out_shared, out_rows)
