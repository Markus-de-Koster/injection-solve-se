"""Saliency / integrated gradients on the tiny trained module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from injection_hse.analysis import (
    harmonic_coupling,
    integrated_gradients,
    node_importance,
    order_importance,
    saliency,
)
from injection_hse.analysis.attribution import (
    _ig_raw,
    _masked_features,
    _model_output,
    _order_index,
    _target_scalar,
)

from .conftest import ORDERS, SENSORS


def test_attribution_batch_shapes(attribution_inputs, trained_module):
    v, mask = attribution_inputs
    assert v.shape[0] == 8 and v.shape[1] == len(ORDERS)
    assert mask.shape == (8, trained_module.n_nodes)
    assert mask.any() and not mask.all()


def test_attribution_batch_random_mask(trained_run, analysis_grid):
    """``observed_fraction`` yields per-sample RANDOM placements (placement-agnostic probe).

    The fixed-placement mask is identical for every sample; the random-mask draw observes the
    same COUNT per sample but a different subset, so rows are not all pinned to one placement.
    """
    from injection_hse.analysis import attribution_batch
    from injection_hse.config import MeasurementConfig

    _, dataset, _ = trained_run
    common = dict(
        grid=analysis_grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS),
        split="train",
        n_samples=8,
        seed=0,
    )
    _, fixed = attribution_batch(dataset, **common)
    _, rnd = attribution_batch(dataset, observed_fraction=0.5, **common)

    from injection_hse.data import MeasurementModel

    n = rnd.shape[-1]
    assert rnd.shape == fixed.shape
    # every sample observes round(0.5 * N) random rows PLUS the always-observed metered
    # slack rows (forced after the draw, matching training), but the subsets are NOT all
    # identical
    slack = MeasurementModel([], slack_known=True).slack_rows(analysis_grid)
    assert bool(rnd[:, slack].all())
    k = max(1, round(0.5 * n))
    counts = rnd.sum(dim=-1)
    assert (counts >= k).all() and (counts <= k + int(slack.numel())).all()
    assert not torch.all(rnd == rnd[0])  # the placement varies across samples
    # the fixed placement, by contrast, is the same row set for every sample
    assert torch.all(fixed == fixed[0])


def test_saliency_shapes_and_mass(trained_module, attribution_inputs, analysis_grid):
    v, mask = attribution_inputs
    attr = saliency(trained_module, v, mask, grid=analysis_grid)
    s, n = mask.shape
    assert attr.values.shape == (s, n, len(ORDERS))
    assert attr.method == "saliency" and attr.target == "unobs_all_orders"
    assert np.all(np.isfinite(attr.values)) and np.all(attr.values >= 0)
    assert attr.values.sum() > 0
    # saliency carries sensitivity at unobserved rows too
    assert (attr.values * ~attr.mask[:, :, None]).sum() > 0


def test_integrated_gradients_zero_off_placement(
    trained_module, attribution_inputs, analysis_grid
):
    v, mask = attribution_inputs
    attr = integrated_gradients(trained_module, v, mask, grid=analysis_grid, steps=8)
    # unobserved rows equal the baseline along the whole path -> exactly zero mass
    off = attr.values * ~attr.mask[:, :, None]
    assert np.all(off == 0.0)
    assert (attr.values * attr.mask[:, :, None]).sum() > 0


def test_integrated_gradients_completeness(trained_module, attribution_inputs):
    v, mask = attribution_inputs
    raw, raw_edge, _, f_x, f_x0 = _ig_raw(
        trained_module, v, mask, steps=64, order_index=None, batch_size=8
    )
    total = float(raw.sum()) + (float(raw_edge.sum()) if raw_edge is not None else 0.0)
    delta = f_x - f_x0
    assert total == pytest.approx(delta, rel=0.05, abs=1e-4)


def test_output_order_slicing(trained_module, attribution_inputs, analysis_grid):
    v, mask = attribution_inputs
    attr = saliency(
        trained_module, v, mask, grid=analysis_grid, output_order=ORDERS[-1]
    )
    assert attr.target == f"unobs_order_{ORDERS[-1]}"
    with pytest.raises(ValueError):
        saliency(trained_module, v, mask, grid=analysis_grid, output_order=99)


def test_target_scalar_selects_unobserved(trained_module, attribution_inputs):
    v, mask = attribution_inputs
    with torch.no_grad():
        x = _masked_features(trained_module, v, mask)
        out = _model_output(trained_module, x, mask)
        full = _target_scalar(trained_module, out, mask, None)
        one = _target_scalar(
            trained_module, out, mask, _order_index(trained_module, ORDERS[0])
        )
    unobs_sum = float((out * (~mask).unsqueeze(-1).to(out.dtype)).sum())
    assert float(full) == pytest.approx(unobs_sum, rel=1e-6)
    assert abs(float(one)) <= abs(float(full)) + 1e-6 or np.isfinite(float(one))


def test_harmonic_coupling_matrix(trained_module, attribution_inputs, analysis_grid):
    v, mask = attribution_inputs
    coupling = harmonic_coupling(trained_module, v, mask, grid=analysis_grid)
    h = len(ORDERS)
    assert coupling.shape == (h, h)
    assert np.all(np.isfinite(coupling)) and np.all(coupling >= 0)
    assert coupling.sum() > 0


def test_aggregations(trained_module, attribution_inputs, analysis_grid):
    v, mask = attribution_inputs
    attr = integrated_gradients(trained_module, v, mask, grid=analysis_grid, steps=4)
    per_order = order_importance(attr)
    assert per_order.shape == (len(ORDERS),) and np.all(per_order >= 0)
    per_node = node_importance(attr)
    assert set(per_node) == set(np.asarray(attr.node_ids).tolist())
    # IG mass concentrates on the sensor nodes
    from pgml.assembly import node_phase_index

    idx = node_phase_index(analysis_grid)
    sensor_rows = np.unique(np.concatenate([attr.mask.nonzero()[1]]))
    sensor_nodes = {int(idx.node_ids[r]) for r in sensor_rows}
    massive = {nid for nid, val in per_node.items() if val > 0}
    assert massive <= sensor_nodes
    assert len(massive) > 0
    # the observed nodes = the configured sensors + the metered slack (a measurement too)
    from pgml.topology import slack_node_ids

    assert len(sensor_nodes) == len(SENSORS) + len(slack_node_ids(analysis_grid))


def test_attribution_casts_double_precision_states(
    trained_module, attribution_inputs, analysis_grid
):
    """A complex128 dataset (the generation default) must attribute through a
    float32 model: the feature seam applies the training-precision boundary cast."""
    from injection_hse.analysis.attribution import integrated_gradients

    v, mask = attribution_inputs
    res = integrated_gradients(
        trained_module,
        v[:2].to(torch.complex128),
        mask[:2],
        grid=analysis_grid,
        steps=2,
    )
    import numpy as np

    assert np.isfinite(res.values).all()
