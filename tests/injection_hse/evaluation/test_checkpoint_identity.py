"""A checkpoint carries the grid's row-layout identity; reload validates it.

``SEModule.on_save_checkpoint`` stores ``pgml.topology.layout_fingerprint(grid)`` (plus the
node-id / phase-code lists for a legible message). ``SEModule.from_checkpoint`` refuses a grid
whose layout differs, and warns (does not crash) on a legacy checkpoint that predates the
fingerprint. Uses the small ``eval_grid`` / ``eval_dataset`` session fixtures.
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
)
from injection_hse.train import SEModule

ORDERS = [1, 3, 5]
SENSORS = ["Bus R4", "Bus C5"]


def _build_checkpoint(grid, dataset, tmp_path):
    """A genuine checkpoint (normalizer + config + fingerprint via ``on_save_checkpoint``).

    Constructs the module the way training does (fits the normalizer on the train split) but
    skips the fit itself — the identity payload is independent of the trained weights.
    """
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import _to_nhc

    cfg = ExperimentConfig(
        name="ckpt-identity",
        seed=0,
        orders=ORDERS,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset),
            batch_size=8,
            val_fraction=0.25,
            num_workers=0,
        ),
        model=DNNConfig(hidden_dims=[16]),
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="placement", kind="placement", target_loss=1e-9, max_epochs=1
                )
            ]
        ),
    )
    source = build_data_source(cfg.data, grid=grid)
    enc = build_encoding(cfg.encoding.kind)
    train_ds = source.train_dataset()
    tv = torch.stack([train_ds[i]["v"] for i in range(min(len(train_ds), 16))])
    norm = HarmonicNormalizer().fit(_to_nhc(enc.encode(tv)))
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=norm,
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        grid=grid,
    )
    payload = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(payload)
    path = tmp_path / "stage.ckpt"
    torch.save(payload, path)
    return str(path)


def _node_swapped(grid):
    """The same grid with its first two nodes reordered (a DIFFERENT row layout, same N)."""
    nodes = list(grid.nodes)
    reordered = [nodes[1], nodes[0], *nodes[2:]]
    return grid.model_copy(update={"nodes": reordered})


def test_checkpoint_stores_layout_fingerprint(eval_grid, eval_dataset, tmp_path):
    from pgml.topology import layout_fingerprint

    path = _build_checkpoint(eval_grid, eval_dataset, tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["grid_fingerprint"] == layout_fingerprint(eval_grid)
    assert len(payload["grid_node_ids"]) == len(payload["grid_phase_codes"])


def test_reload_same_grid_ok(eval_grid, eval_dataset, tmp_path):
    path = _build_checkpoint(eval_grid, eval_dataset, tmp_path)
    module = SEModule.from_checkpoint(path, grid=eval_grid, device="cpu")
    assert list(module.orders) == ORDERS
    assert not module.training  # eval mode


def test_reload_node_swapped_grid_raises(eval_grid, eval_dataset, tmp_path):
    from pgml.topology import layout_fingerprint

    path = _build_checkpoint(eval_grid, eval_dataset, tmp_path)
    swapped = _node_swapped(eval_grid)
    # guard: the reordering genuinely changes the layout fingerprint (same N, different order)
    assert layout_fingerprint(swapped) != layout_fingerprint(eval_grid)
    with pytest.raises(ValueError, match="row layout"):
        SEModule.from_checkpoint(path, grid=swapped, device="cpu")


def test_reload_legacy_checkpoint_warns(eval_grid, eval_dataset, tmp_path):
    path = _build_checkpoint(eval_grid, eval_dataset, tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # simulate a checkpoint saved before grid identity was recorded
    for key in ("grid_fingerprint", "grid_node_ids", "grid_phase_codes"):
        payload.pop(key, None)
    legacy = tmp_path / "legacy.ckpt"
    torch.save(payload, legacy)
    with pytest.warns(UserWarning, match="no grid fingerprint"):
        module = SEModule.from_checkpoint(str(legacy), grid=eval_grid, device="cpu")
    assert list(module.orders) == ORDERS
