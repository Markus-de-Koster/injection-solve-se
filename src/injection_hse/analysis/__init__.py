"""injection_hse.analysis — model-interpretability analysis of trained state estimators.

Explains HOW a trained harmonic state estimator works, complementing the
accuracy-focused :mod:`injection_hse.evaluation`: which input harmonic orders and nodes
carry the information (gradient attribution, occlusion), which model components
learn during which curriculum stage (per-layer gradient dynamics, checkpoint
weight deltas), and in which physical operating conditions the estimator degrades (error binned by
THD, voltage unbalance, line loading, voltage deviation).

Figures are camera-ready for a double-column IEEE A4 conference template and
written as PDF + SVG pairs (:mod:`injection_hse.analysis.style`).

Layout
------
- ``style``       IEEE column geometry, rcParams context, PDF+SVG save helper.
- ``binning``     physical scenario features + binned unobserved-error statistics.
- ``attribution`` input-gradient saliency, integrated gradients, harmonic coupling.
- ``baselines``   the non-learned references (zero-harmonic, training-mean, persistence)
  plus the LINEAR INJECTION PRIOR — the closed-form least-squares estimate of the nodal
  injections from the estimator's own measurements, decoded through the estimator's own
  anchored network solve.
- ``occlusion``   sensor-drop and input-order-zeroing degradation.
- ``gradients``   per-layer gradient norms during training + checkpoint weight deltas.
- ``report``      :func:`analyze_run` — manifest + checkpoint -> the paper figure set;
  ``python -m injection_hse.analysis <run_dir>`` is the command-line entry.

Requires the training stack (lightning, a core dependency of ``power-grid-learn``) plus
the plotting stack (``power-grid-learn[viz]``) for the figures.
"""

from __future__ import annotations

from .baselines import (
    LINEAR_PRIOR_RIDGE,
    LinearInjectionPrior,
    collect_baseline_metrics,
    fit_linear_injection_prior,
    linear_injection_prediction,
    linear_prior_features,
    linear_prior_injections,
    measured_inputs,
    persistence_prediction,
    score_prediction,
    training_mean_prediction,
    zero_harmonic_prediction,
)
from .attribution import (
    Attribution,
    attribution_batch,
    evaluation_batch,
    harmonic_coupling,
    integrated_gradients,
    node_importance,
    order_importance,
    saliency,
)
from .binning import binned_errors, sample_features, scenario_features
from .gradients import (
    GradientNormCollector,
    checkpoint_weight_deltas,
    parameter_groups,
    read_gradient_norms,
)
from .occlusion import (
    estimator_band_ablation,
    estimator_order_coupling,
    order_occlusion,
    order_occlusion_error,
    sensor_occlusion,
)
from .report import (
    analyze_run,
    plot_binned_errors,
    plot_estimator_coupling,
    plot_gradient_dynamics,
    plot_harmonic_coupling,
    plot_node_importance,
    plot_order_importance,
    plot_order_occlusion,
    plot_sensor_occlusion,
    plot_weight_deltas,
)
from .style import (
    IEEE_COLUMN_IN,
    IEEE_PAGE_IN,
    SERIES_COLORS,
    categorical_colors,
    figure_size,
    paper_style,
    save_paper_figure,
    series_style,
)

__all__ = [
    # style
    "IEEE_COLUMN_IN",
    "IEEE_PAGE_IN",
    "SERIES_COLORS",
    "series_style",
    "categorical_colors",
    "figure_size",
    "paper_style",
    "save_paper_figure",
    # binning
    "scenario_features",
    "sample_features",
    "binned_errors",
    # attribution
    "Attribution",
    "attribution_batch",
    "evaluation_batch",
    "saliency",
    "integrated_gradients",
    "harmonic_coupling",
    "order_importance",
    "node_importance",
    # occlusion
    "sensor_occlusion",
    "order_occlusion",
    "order_occlusion_error",
    "estimator_order_coupling",
    "estimator_band_ablation",
    # gradients
    "parameter_groups",
    "GradientNormCollector",
    "read_gradient_norms",
    "checkpoint_weight_deltas",
    # report
    "plot_order_importance",
    "plot_order_occlusion",
    "plot_harmonic_coupling",
    "plot_estimator_coupling",
    "plot_node_importance",
    "plot_sensor_occlusion",
    "plot_gradient_dynamics",
    "plot_weight_deltas",
    "plot_binned_errors",
    "analyze_run",
    # baselines (non-learned references)
    "score_prediction",
    "zero_harmonic_prediction",
    "training_mean_prediction",
    "persistence_prediction",
    "LINEAR_PRIOR_RIDGE",
    "LinearInjectionPrior",
    "measured_inputs",
    "linear_prior_features",
    "fit_linear_injection_prior",
    "linear_prior_injections",
    "linear_injection_prediction",
    "collect_baseline_metrics",
]
