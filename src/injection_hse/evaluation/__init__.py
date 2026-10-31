"""injection_hse.evaluation — state-estimation model evaluation plots (actual vs predicted).

The ML-side counterpart of :mod:`pgml.evaluation` (which compares the solver to reference
libraries): here the comparison is the TRAINED estimator's output against the ground-truth
state, plus diagnostics of where and at which harmonic it struggles. Framework-agnostic plot
functions (matplotlib SVG + plotly HTML) consuming a plain-numpy
:class:`~injection_hse.evaluation.inference.Prediction`; the existing :mod:`pgml.evaluation` containers
and figures are reused wherever possible. Everything runs OFF the autograd tape (the module is
evaluated under ``no_grad`` and detached to numpy for plotting).

Layout
------
- ``inference``     run a trained module over a dataset -> :class:`Prediction` (+ checkpoint load).
- ``state_plots``   actual-vs-predicted harmonic profiles (2D / interactive / 3D), sensors marked.
- ``error_heatmap`` per-order x node-phase pu-``|error|`` heatmap, sensor rows highlighted.
- ``history``       :class:`HistoryCollector` (spans the two-task fit calls), the MLflow
                    file-store reader for a run that was not collected, and the loss-curve
                    plot with its per-term / per-order decomposition.
- ``insight_plots`` parity / per-order error / error-vs-distance / error CDF (+ worst sample).
- ``report``        :func:`evaluate_run` — one call: manifest + checkpoint -> the full plot set.

Requires the training stack (lightning + torch-geometric, core dependencies of
``power-grid-learn``) plus the plotting stack (``power-grid-learn[viz]``:
matplotlib/plotly/networkx).
"""

from __future__ import annotations

from .error_heatmap import plot_error_heatmap
from .history import (
    HistoryCollector,
    mlflow_history,
    plot_loss_components,
    plot_loss_curves,
    read_history,
    read_mlflow_metrics,
    read_training_history,
    stage_manifest,
)
from .inference import (
    Prediction,
    load_se_module,
    predict_dataset,
    prediction_metrics,
    prediction_metrics_by_group,
)
from .loss_heatmap import LossTermMaps, loss_term_maps, plot_loss_heatmap
from .insight_plots import (
    plot_error_by_order,
    plot_error_cdf,
    plot_error_vs_sensor_distance,
    plot_parity_per_harmonic,
    worst_scenario,
)
from .report import evaluate_run
from .state_plots import harmonic_profile_from_voltages, plot_state_vs_prediction

__all__ = [
    # inference
    "Prediction",
    "load_se_module",
    "predict_dataset",
    "prediction_metrics",
    "prediction_metrics_by_group",
    # state plots
    "harmonic_profile_from_voltages",
    "plot_state_vs_prediction",
    # error heatmap
    "plot_error_heatmap",
    # loss heatmap (which supervised term, which band, which injection node)
    "LossTermMaps",
    "loss_term_maps",
    "plot_loss_heatmap",
    # history (the collected parquet and the MLflow file store, one reader)
    "HistoryCollector",
    "read_history",
    "read_mlflow_metrics",
    "mlflow_history",
    "stage_manifest",
    "read_training_history",
    "plot_loss_curves",
    "plot_loss_components",
    # insight plots
    "plot_parity_per_harmonic",
    "plot_error_by_order",
    "plot_error_vs_sensor_distance",
    "plot_error_cdf",
    "worst_scenario",
    # report
    "evaluate_run",
]
