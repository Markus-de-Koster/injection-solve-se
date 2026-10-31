"""Sensor-drop and input-order occlusion on the tiny trained modules."""

from __future__ import annotations

import numpy as np
import pytest

from injection_hse.analysis import (
    estimator_band_ablation,
    estimator_order_coupling,
    order_occlusion,
    order_occlusion_error,
    sensor_occlusion,
)

from .conftest import ORDERS, SENSORS


def test_sensor_occlusion_frame(trained_module, trained_run, analysis_grid):
    _, dataset, _ = trained_run
    frame = sensor_occlusion(
        trained_module,
        dataset,
        grid=analysis_grid,
        sensor_names=SENSORS,
        include_pairs=True,
    )
    # baseline + one row per dropped single + one per dropped pair
    assert frame.height == 1 + len(SENSORS) + 3
    assert frame["dropped"][0] == "none"
    assert frame["n_sensors"][0] == len(SENSORS)
    for col in ["mae", "mae_fund", "mae_harm"] + [f"mae_h{o}" for o in ORDERS]:
        vals = frame[col].to_numpy()
        assert np.all(np.isfinite(vals)) and np.all(vals > 0)
    # dropping a sensor reduces the observed set
    assert set(frame["n_sensors"].to_list()) == {1, 2, 3}


def test_order_occlusion_matrix(trained_module, attribution_inputs):
    v, mask = attribution_inputs
    matrix = order_occlusion(trained_module, v, mask)
    h = len(ORDERS)
    assert matrix.shape == (h, h)
    assert np.all(np.isfinite(matrix)) and np.all(matrix >= 0)
    # zeroing an observed input order must move the outputs somewhere
    assert matrix.sum() > 0


def test_order_occlusion_error(trained_module, attribution_inputs, analysis_grid):
    from injection_hse.data import base_voltage_per_row

    v, mask = attribution_inputs
    frame = order_occlusion_error(
        trained_module, v, mask, base_v=base_voltage_per_row(analysis_grid)
    )
    # a baseline row + one per input order
    assert frame.height == 1 + len(ORDERS)
    assert frame["order"].to_list() == ["none"] + [str(o) for o in ORDERS]
    for col in ("mae", "mae_fund", "mae_harm"):
        assert np.all(np.isfinite(frame[col].to_numpy()))
    # the baseline delta is exactly zero; every occluded order is measured vs it
    base_row = frame.filter(frame["order"] == "none")
    assert float(base_row["delta_mae"][0]) == 0.0
    assert float(base_row["mae"][0]) > 0


def test_order_occlusion_current_seam(trained_current_module, attribution_inputs):
    """The current-measuring module probes with edge inputs; per-block occlusion
    variants differ from the joint one (the current block carries signal)."""
    v, mask = attribution_inputs
    joint = order_occlusion(trained_current_module, v, mask)
    v_only = order_occlusion(trained_current_module, v, mask, occlude=("voltage",))
    h = len(ORDERS)
    assert joint.shape == v_only.shape == (h, h)
    assert np.all(np.isfinite(joint)) and joint.sum() > 0
    assert not np.allclose(joint, v_only)
    with pytest.raises(ValueError):
        order_occlusion(trained_current_module, v, mask, occlude=("power",))


def test_attribution_edge_fields(
    trained_current_module, trained_module, attribution_inputs, analysis_grid
):
    from injection_hse.analysis import order_importance, saliency

    v, mask = attribution_inputs
    attr = saliency(trained_current_module, v, mask, grid=analysis_grid)
    assert attr.edge_values is not None and attr.edge_observed is not None
    s = v.shape[0]
    assert attr.edge_values.shape[0] == s and attr.edge_values.shape[-1] == len(ORDERS)
    assert np.all(np.isfinite(attr.edge_values))
    cur = order_importance(attr, inputs="current")
    both = order_importance(attr, inputs="all")
    assert cur.sum() > 0 and np.all(both >= cur - 1e-12)
    # a voltage-only module carries no edge attribution and zero current mass
    attr_v = saliency(trained_module, v, mask, grid=analysis_grid)
    assert attr_v.edge_values is None
    assert order_importance(attr_v, inputs="current").sum() == 0.0


def test_estimator_order_coupling(
    trained_current_module, attribution_inputs, analysis_grid
):
    from injection_hse.data import base_voltage_per_row

    v, mask = attribution_inputs
    shift, delta_mae, mae_base = estimator_order_coupling(
        trained_current_module,
        v,
        mask,
        base_v=base_voltage_per_row(analysis_grid),
    )
    h = len(ORDERS)
    assert shift.shape == delta_mae.shape == (h, h)
    assert mae_base.shape == (h,)
    assert np.all(np.isfinite(shift)) and np.all(shift >= 0) and shift.sum() > 0
    assert np.all(np.isfinite(delta_mae)) and np.all(np.isfinite(mae_base))
    assert np.all(mae_base > 0)


def test_estimator_band_ablation(
    trained_current_module, trained_module, attribution_inputs, analysis_grid
):
    from injection_hse.data import base_voltage_per_row

    v, mask = attribution_inputs
    base_v = base_voltage_per_row(analysis_grid)
    frame = estimator_band_ablation(trained_current_module, v, mask, base_v=base_v)
    assert frame["variant"].to_list() == [
        "baseline",
        "no_harmonics",
        "no_fundamental",
        "anchors_only",
        "model_only",
    ]
    for o in ORDERS:
        vals = frame[f"mae_h{o}"].to_numpy()
        assert np.all(np.isfinite(vals)) and np.all(vals >= 0)
    # the voltage-decode module has no injection decode to ablate
    with pytest.raises(ValueError):
        estimator_band_ablation(trained_module, v, mask, base_v=base_v)


def test_estimator_band_ablation_noisy_rows(
    trained_current_module, attribution_inputs, analysis_grid
):
    """A module carrying a sensor-noise model gains the three noisy variants,
    scored against the clean truth (values finite, distinct from the clean rows)."""
    from injection_hse.config import NoiseConfig
    from injection_hse.data import SensorNoise, base_voltage_per_row

    v, mask = attribution_inputs
    base_v = base_voltage_per_row(analysis_grid)
    mod = trained_current_module
    saved = (mod._noise, mod._edge_noise, mod.cfg.noise)
    try:
        mod.cfg.noise = NoiseConfig(enabled=True, default_class="1")
        mod._noise = SensorNoise.from_config(
            mod.cfg.noise, n_nodes=mod.n_nodes, grid=analysis_grid
        )
        mod._edge_noise = mod._noise.for_edges(mod._edge_src_rows)
        frame = estimator_band_ablation(mod, v, mask, base_v=base_v)
    finally:
        mod._noise, mod._edge_noise, mod.cfg.noise = saved
    rows = frame["variant"].to_list()
    for name in ("baseline_noisy", "anchors_only_noisy", "no_harmonics_noisy"):
        assert name in rows
    cols = [c for c in frame.columns if c.startswith("mae_h")]
    for name in rows:
        vals = frame.filter(frame["variant"] == name).select(cols).to_numpy()
        assert np.isfinite(vals).all() and (vals >= 0).all()
    clean = frame.filter(frame["variant"] == "baseline").select(cols).to_numpy()
    noisy = frame.filter(frame["variant"] == "baseline_noisy").select(cols).to_numpy()
    assert not np.allclose(clean, noisy)


def test_estimator_band_ablation_linear_prior_row(
    trained_current_module, attribution_inputs, linear_prior_inputs, analysis_grid
):
    """A fitted linear injection prior adds one row decoded through the SAME anchored
    solve as ``anchors_only`` — the same decode under a fitted prior instead of a zero one."""
    from injection_hse.analysis import fit_linear_injection_prior
    from injection_hse.data import base_voltage_per_row

    module = trained_current_module
    v_fit, mask_fit = linear_prior_inputs
    prior = fit_linear_injection_prior(module, v_fit, mask_fit)

    v, mask = attribution_inputs
    base_v = base_voltage_per_row(analysis_grid)
    frame = estimator_band_ablation(module, v, mask, base_v=base_v, linear_prior=prior)
    assert frame["variant"].to_list() == [
        "baseline",
        "no_harmonics",
        "no_fundamental",
        "anchors_only",
        "model_only",
        "linear_prior",
    ]
    cols = [f"mae_h{o}" for o in ORDERS]
    row = frame.filter(frame["variant"] == "linear_prior").select(cols).to_numpy()
    assert np.isfinite(row).all() and (row >= 0).all()
    # a fitted prior is a different estimate from the zero prior it replaces
    zero = frame.filter(frame["variant"] == "anchors_only").select(cols).to_numpy()
    assert not np.allclose(row, zero)
    # omitting the prior leaves the table exactly as it was
    without = estimator_band_ablation(module, v, mask, base_v=base_v)
    assert "linear_prior" not in without["variant"].to_list()
    base_cols = without.filter(without["variant"] == "anchors_only").select(cols)
    assert np.allclose(base_cols.to_numpy(), zero)


def _variant(frame, name):
    cols = [f"mae_h{o}" for o in ORDERS]
    return frame.filter(frame["variant"] == name).select(cols).to_numpy()[0]


def test_evaluation_batch_draws_whole_sequences_from_the_split(
    analysis_sequence_dataset, analysis_grid
):
    import torch

    from injection_hse.analysis import evaluation_batch
    from injection_hse.config import DataConfig, MeasurementConfig
    from injection_hse.data import MeasurementModel, build_data_source

    source = build_data_source(
        DataConfig(
            source="parquet",
            dataset_path=str(analysis_sequence_dataset),
            val_fraction=0.25,
            test_fraction=0.25,
        ),
        grid=analysis_grid,
    )
    test = source.test_dataset()
    measurement = MeasurementConfig(voltage_nodes=SENSORS, slack_known=True)
    v, mask, time_unix_s = evaluation_batch(
        source, grid=analysis_grid, measurement=measurement, split="test", n_samples=99
    )
    steps = source.n_steps
    assert steps > 1 and v.shape[:2] == (len(test), steps)
    test_states = [test[i]["v"] for i in range(len(test))]
    for sample in v:  # every drawn sequence is one whole TEST scenario
        assert any(torch.equal(sample, ref) for ref in test_states)
    rows = MeasurementModel.resolve(
        analysis_grid, SENSORS, slack_known=True
    ).observed_rows(analysis_grid)
    assert mask.shape == (len(test), v.shape[-1]) and mask[:, rows].all()
    assert int(mask.sum()) == len(test) * int(rows.numel())
    assert time_unix_s is None  # a timeless coherent dataset
    with pytest.raises(ValueError, match="empty"):
        evaluation_batch(
            build_data_source(
                DataConfig(
                    source="parquet", dataset_path=str(analysis_sequence_dataset)
                ),
                grid=analysis_grid,
            ),
            grid=analysis_grid,
            measurement=measurement,
            split="test",
        )


def test_estimator_band_ablation_scores_sequences_as_sequences(
    trained_current_module, analysis_sequence_dataset, analysis_grid
):
    """On ``[B, T, H, N]`` the network reads whole sequences (its temporal memory runs) and
    its baseline row is the deployed estimate ``SEModule.infer`` returns; every per-step
    variant equals the same states scored as independent snapshots."""
    import torch

    from injection_hse.analysis import evaluation_batch
    from injection_hse.config import MeasurementConfig
    from injection_hse.data import base_voltage_per_row
    from injection_hse.masking import build_masker

    module = trained_current_module
    v, mask, _ = evaluation_batch(
        analysis_sequence_dataset,
        grid=analysis_grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        split="train",
        n_samples=3,
    )
    b, t, h, n = v.shape
    base_v = base_voltage_per_row(analysis_grid)
    seq = estimator_band_ablation(module, v, mask, base_v=base_v)
    flat_mask = mask.unsqueeze(1).expand(b, t, n).reshape(b * t, n)
    flat = estimator_band_ablation(
        module, v.reshape(b * t, h, n), flat_mask, base_v=base_v
    )
    # per-step variants: the same states, the same scores
    np.testing.assert_allclose(
        _variant(seq, "anchors_only"), _variant(flat, "anchors_only"), rtol=1e-6
    )
    # the network row is not the per-step snapshot reading of a temporal model
    assert not np.allclose(_variant(seq, "baseline"), _variant(flat, "baseline"))

    # ... it is what the deployed estimator returns on these sequences
    saved = module._masker
    try:
        module.set_masker(
            build_masker("placement", observed_rows=mask[0].nonzero()[:, 0])
        )
        with torch.no_grad():
            out = module.infer({"v": v})
    finally:
        module.set_masker(saved)
    assert torch.equal(out["mask"], mask.unsqueeze(1).expand(b, t, n))
    err = (out["pred_v"] - v.to(out["pred_v"].dtype)).abs() / base_v.view(1, 1, 1, -1)
    unobs = ~out["mask"].unsqueeze(-2)  # [B, T, 1, N]
    mae = (err * unobs).sum(dim=(0, 1, 3)) / unobs.sum()
    np.testing.assert_allclose(
        _variant(seq, "baseline"), mae.double().numpy(), rtol=1e-4
    )
