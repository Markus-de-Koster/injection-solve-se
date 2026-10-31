"""Non-learned baselines: the trivial references the learned estimator is scored against."""

from __future__ import annotations

import numpy as np

from injection_hse.analysis import (
    collect_baseline_metrics,
    persistence_prediction,
    score_prediction,
    training_mean_prediction,
    zero_harmonic_prediction,
)
from injection_hse.config import MeasurementConfig

from .conftest import ORDERS, SENSORS

_METRIC_KEYS = ("mae", "rmse", "mae_fund", "mae_harm", "rel_harm", "thd_mae")


def test_score_prediction_has_shared_columns(balanced_prediction):
    row = score_prediction(balanced_prediction)
    assert set(_METRIC_KEYS) <= set(row)
    assert np.isfinite(row["mae"]) and row["mae"] > 0
    assert np.isfinite(row["thd_mae"])


def test_zero_harmonic_predicts_zero_above_fundamental(balanced_prediction):
    pred = zero_harmonic_prediction(balanced_prediction)
    for hi, order in enumerate(ORDERS):
        if order == 1:
            assert np.any(pred.pred_v[:, hi, :] != 0)  # a flat fundamental profile
        else:
            assert np.all(pred.pred_v[:, hi, :] == 0)  # exactly zero harmonics
    # the fundamental target is untouched (baselines never rewrite the ground truth)
    assert np.array_equal(pred.target_v, balanced_prediction.target_v)


def test_training_mean_is_constant_across_samples(balanced_prediction):
    train_target = balanced_prediction.target_v  # stand-in training targets
    pred = training_mean_prediction(balanced_prediction, train_target)
    # every sample gets the SAME estimate (the per-row, per-order training mean)
    assert np.allclose(pred.pred_v, pred.pred_v[0][None])
    assert np.allclose(pred.pred_v[0], train_target.mean(axis=0))


def test_persistence_shifts_by_one_step(balanced_prediction):
    b, t, h, n = 3, 4, len(ORDERS), balanced_prediction.target_v.shape[-1]
    seq = np.arange(b * t * h * n, dtype=np.complex128).reshape(b, t, h, n)
    # fold the reference to S = b*t so the shapes line up
    import dataclasses

    ref = dataclasses.replace(
        balanced_prediction,
        target_v=seq.reshape(b * t, h, n),
        pred_v=np.zeros((b * t, h, n), dtype=np.complex128),
        mask=np.zeros((b * t, n), dtype=bool),
    )
    pred = persistence_prediction(seq, ref).pred_v.reshape(b, t, h, n)
    assert np.array_equal(pred[:, 0], seq[:, 0])  # first step copies itself
    assert np.array_equal(pred[:, 1:], seq[:, :-1])  # step t <- truth at t-1


def test_collect_baseline_metrics_snapshot(analysis_grid, analysis_dataset):
    frame = collect_baseline_metrics(
        analysis_dataset,
        grid=analysis_grid,
        measurement=MeasurementConfig(voltage_nodes=SENSORS, slack_known=True),
        split="val",
    )
    variants = set(frame["variant"].to_list())
    # snapshot data -> the two applicable baselines; persistence is skipped (needs time)
    assert variants == {"baseline_zero_harmonic", "baseline_training_mean"}
    for col in ("n_params", "epochs", *_METRIC_KEYS):
        assert col in frame.columns
    assert (frame["n_params"] == 0).all()  # non-learned
    assert np.all(np.isfinite(frame["mae"].to_numpy()))


# =============================================================================
# The linear injection prior (fitted, closed-form, decoded like the estimator)
# =============================================================================
def test_true_injections_decode_back_to_the_true_state(
    trained_current_module, linear_prior_inputs
):
    """The decode the prior is scored through is exact: feeding it the TRUE injections
    reproduces the true state, so any error the prior shows is the prior's own."""
    import torch

    from injection_hse.analysis.attribution import _edge_features
    from injection_hse.physics import solve_injection

    module = trained_current_module
    v, mask = linear_prior_inputs
    module.eval()
    with torch.no_grad():
        i_true = module.solver.injection_targets(v).clone()
        i_true[..., module.solver.slack_rows] = 0.0  # the free boundary current
        _, edge_mask, i_edges = _edge_features(module, v, mask)
        pred = solve_injection(
            module, i_true, v, mask=mask, edge_mask=edge_mask, i_edges=i_edges
        )
    base_v = torch.as_tensor(trained_current_module.base_v)
    err = ((pred - v.to(pred.dtype)).abs() / base_v).max()
    assert float(err) < 1e-4  # the module's complex64 training precision, not the solve


def test_linear_prior_layout_and_zero_rows(
    trained_current_module, linear_prior_inputs, analysis_grid
):
    from injection_hse.analysis import (
        LINEAR_PRIOR_RIDGE,
        fit_linear_injection_prior,
        linear_prior_features,
    )
    from injection_hse.data import MeasurementModel

    module = trained_current_module
    v, mask = linear_prior_inputs
    prior = fit_linear_injection_prior(module, v, mask)
    assert prior.ridge == LINEAR_PRIOR_RIDGE
    assert prior.n_train == v.shape[0]
    assert prior.noisy is False
    assert prior.orders == tuple(ORDERS)
    assert prior.n_rows == module.n_nodes

    # the fit predicts every NON-slack row that is not a switch tie (electrically the
    # slack — physics.exclude_tied_rows), over all orders, real + imaginary
    slack = module.solver.slack_rows.numpy()
    tied = module._tied_rows.numpy()
    assert set(prior.rows.tolist()) == (
        set(range(module.n_nodes)) - set(slack.tolist()) - set(tied.tolist())
    )
    assert prior.n_targets == len(ORDERS) * prior.rows.size * 2

    # the features are exactly the observed measurements the estimator reads
    feats = linear_prior_features(module, v.to(module._complex_dtype), mask)
    n_obs = int(MeasurementModel(SENSORS).observed_rows(analysis_grid).numel())
    n_edges = int(mask.index_select(-1, module._edge_src_rows)[0].sum())
    assert feats.shape == (v.shape[0], len(ORDERS) * 2 * (n_obs + n_edges))
    assert prior.n_features == feats.shape[1]

    # the slack rows carry no predicted injection (the solve never reads them)
    injections = prior.predict(feats.numpy())
    assert injections.shape == (v.shape[0], len(ORDERS), module.n_nodes)
    assert np.all(injections[:, :, slack] == 0)


def test_linear_prior_leaves_never_injecting_rows_negligible(
    trained_current_module, linear_prior_inputs
):
    """A row that carries no device has a target that is zero to the solve's roundoff, so
    the fit predicts a proportionally negligible injection there — the device-row
    restriction follows from the data and is not imposed."""
    import torch

    from injection_hse.analysis import fit_linear_injection_prior, linear_prior_features

    module = trained_current_module
    v, mask = linear_prior_inputs
    prior = fit_linear_injection_prior(module, v, mask)
    with torch.no_grad():
        rms = module.solver.injection_targets(v).abs().pow(2).mean(dim=(0, 1)).sqrt()
    rms = rms.numpy()
    rms[module.solver.slack_rows.numpy()] = 0.0
    quiet = rms <= 1e-6 * rms.max()  # rows no appliance injects into
    assert quiet.any() and not quiet.all()
    feats = linear_prior_features(module, v.to(module._complex_dtype), mask)
    injections = prior.predict(feats.numpy())
    loud = np.abs(injections[:, :, ~quiet]).mean()
    assert np.abs(injections[:, :, quiet]).max() < 1e-6 * loud


def test_linear_prior_save_load_roundtrip(
    trained_current_module, linear_prior_inputs, tmp_path
):
    from injection_hse.analysis import (
        LinearInjectionPrior,
        fit_linear_injection_prior,
        linear_prior_features,
    )

    module = trained_current_module
    v, mask = linear_prior_inputs
    prior = fit_linear_injection_prior(module, v, mask)
    path = prior.save(tmp_path / "linear_prior.npz")
    back = LinearInjectionPrior.load(path)
    assert back.orders == prior.orders and back.n_train == prior.n_train
    assert back.ridge == prior.ridge and back.noisy == prior.noisy
    feats = linear_prior_features(module, v.to(module._complex_dtype), mask).numpy()
    assert np.array_equal(back.predict(feats), prior.predict(feats))


def test_linear_prior_rejects_a_varying_placement(
    trained_current_module, linear_prior_inputs
):
    import pytest

    from injection_hse.analysis import fit_linear_injection_prior

    module = trained_current_module
    v, mask = linear_prior_inputs
    varying = mask.clone()
    varying[0] = ~varying[0]
    with pytest.raises(ValueError, match="FIXED sensor placement"):
        fit_linear_injection_prior(module, v, varying)


def test_linear_prior_requires_the_injection_decode(trained_module, attribution_inputs):
    import pytest

    from injection_hse.analysis import fit_linear_injection_prior

    v, mask = attribution_inputs
    with pytest.raises(ValueError, match="injection decode"):
        fit_linear_injection_prior(trained_module, v, mask)


def test_collect_baseline_metrics_linear_injection_row(
    analysis_grid, trained_current_run, trained_current_module
):
    _, dataset, _ = trained_current_run
    measurement = MeasurementConfig(voltage_nodes=SENSORS, slack_known=True)
    with_module = collect_baseline_metrics(
        dataset,
        grid=analysis_grid,
        measurement=measurement,
        split="val",
        module=trained_current_module,
    )
    assert "baseline_linear_injection" in with_module["variant"].to_list()
    row = with_module.filter(with_module["variant"] == "baseline_linear_injection")
    for col in _METRIC_KEYS:
        assert np.isfinite(row[col][0])
    assert row["n_params"][0] == 0
    # no module -> the row is skipped, the existing baselines are unchanged
    without = collect_baseline_metrics(
        dataset, grid=analysis_grid, measurement=measurement, split="val"
    )
    assert "baseline_linear_injection" not in without["variant"].to_list()
