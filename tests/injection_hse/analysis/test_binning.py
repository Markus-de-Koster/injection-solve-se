"""Physical scenario features + binned errors on a controlled ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from injection_hse.analysis import binned_errors, scenario_features

from .conftest import H3_FRACTION, H5_FRACTION


@pytest.fixture(scope="module")
def features(balanced_prediction):
    return scenario_features(balanced_prediction)


def test_scenario_features_schema(balanced_prediction, features):
    assert features.columns == [
        "sample",
        "thd_max_pct",
        "thd_mean_pct",
        "vuf_max_pct",
        "loading_rel_max",
        "vdev_max_pct",
    ]
    assert features.height == balanced_prediction.n_samples


def test_thd_exact_on_controlled_state(features):
    expected = 100.0 * float(np.hypot(H3_FRACTION, H5_FRACTION))
    assert features["thd_max_pct"].to_numpy() == pytest.approx(expected, rel=1e-6)
    assert features["thd_mean_pct"].to_numpy() == pytest.approx(expected, rel=1e-6)


def test_vuf_zero_for_balanced_state(features):
    assert np.all(features["vuf_max_pct"].to_numpy() < 1e-8)


def test_vdev_zero_at_nominal(features):
    assert np.all(features["vdev_max_pct"].to_numpy() < 1e-8)


def test_loading_finite_and_normalized(features):
    loading = features["loading_rel_max"].to_numpy()
    assert np.all(np.isfinite(loading))
    assert np.all(loading > 0)
    # identical samples -> every sample sits at its branch p95 exactly
    assert loading == pytest.approx(1.0, rel=1e-6)


def test_binned_errors_partition_and_stats(balanced_prediction, features):
    # spread the binning feature so quantile bins are non-degenerate
    spread = features.with_columns(thd_max_pct=np.linspace(1.0, 9.0, features.height))
    table = binned_errors(
        balanced_prediction, spread, by="thd_max_pct", bins=4, min_count=1
    )
    assert int(table["n"].sum()) == balanced_prediction.n_samples
    for col in ("mae_mean", "mae_p50", "mae_p90", "mae_fund_mean", "mae_harm_mean"):
        assert np.all(np.isfinite(table[col].to_numpy()))
        assert np.all(table[col].to_numpy() > 0)


def test_binned_errors_rejects_misaligned_features(balanced_prediction, features):
    with pytest.raises(ValueError):
        binned_errors(balanced_prediction, features.head(3), by="thd_max_pct")
