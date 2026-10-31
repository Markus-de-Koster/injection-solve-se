"""Training: the Lightning module, the curriculum stage manager, and the run entry point.

Two decode modes are selected by :attr:`~injection_hse.config.PhysicsConfig.decode`:

- ``"voltage"`` (default) — the model predicts the normalized voltage features directly
  and the loss supervises against the true state.
- ``"injection"`` — the model predicts the nodal INJECTION currents; the differentiable
  network solve ``V(h) = solve(Y(h), I_pred)`` reconstructs the voltages so the output
  satisfies the network law ``Y(h)·V(h) = I(h)`` by construction.

In both modes the curriculum drives one ``Trainer.fit`` per masking stage (``full`` →
``random`` ladder → ``placement`` finetune), continuing weights across stages; each stage
trains until the validation per-unit all-rows MAE reaches its ``target_loss``
(:class:`TargetLossEarlyStop`) or ``max_epochs`` is hit. A checkpoint is forced at every
stage boundary.

Lightning does NOT touch the autograd graph; the physics-informed injection decode and any
``pgg`` generation losses call ``pgml`` and gradients flow as in raw PyTorch. Use
``precision="32"`` on the complex physics path (no AMP). MLflow logs config + metrics +
artifacts (the fitted normalizer + config travel with the checkpoint for exact reproduction).

:func:`train_experiment` runs ONE config end to end. A full run is several passes over one
module (the task curriculum) — that contract is :mod:`injection_hse.workflow`, and this package is its
command line::

    python -m injection_hse.train --config <run.yaml> [--device cuda] [--out <run dir>]
    python -m injection_hse.train --json-schema | --example

Layout
------
- ``module``     :class:`SEModule` — the estimator, the single-grid step, the Lightning hooks.
- ``multigrid``  the ``[H, N*]`` cross-grid mirror of that step (a mixin :class:`SEModule`
  inherits), plus :func:`fit_multigrid_normalizer` and :func:`evaluate_multigrid`.
- ``stages``     :class:`StageManager`, :class:`TargetLossEarlyStop`, and the config-swap
  guard :func:`assert_cfg_swappable`.
- ``checkpoint`` the shared checkpoint-to-live-module reconstruction.
- ``loop``       :func:`train_experiment`, :func:`make_datamodule`, :func:`probe_step_memory`,
  :func:`mlflow_logger`.
- ``priors``     :mod:`injection_hse.train.priors` — the solver-level primitives behind the
  physics-derived input priors (:attr:`~injection_hse.config.PhysicsConfig.input_priors`), assembled
  per batch by :meth:`SEModule.input_priors`.
- ``distillation`` :mod:`injection_hse.train.distillation` — the LINEAR-PRIOR distillation objective
  (:attr:`~injection_hse.config.LossConfig.prior_distillation_weight`): the fitted prior as an
  on-device teacher (:class:`InjectionPriorTeacher`) and the per-pass fit on the pass's
  own train states (:func:`attach_injection_prior`, :func:`prior_fit_states`).
- ``layout`` / ``dtypes`` / ``anchoring``  the feature-layout, precision and anchor-weight
  helpers both paths share; ``anchoring`` also builds the noise-calibrated anchor weights
  (:func:`install_anchor_calibration`, :func:`anchor_calibration_sigmas`).
- ``cli``        :func:`main`, the ``python -m injection_hse.train`` entry point.
"""

from __future__ import annotations

try:  # lightning is a core dependency of power-grid-learn, but keep import errors legible
    import lightning as L  # noqa: F401  (the package's handle onto the Trainer class)
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "injection_hse.train requires lightning (a dependency of power-grid-learn); reinstall the "
        "package (`pip install power-grid-learn`)."
    ) from exc

from .anchoring import (
    ANCHOR_LOG_DECADES,
    anchor_build_weights,
    anchor_calibration_sigmas,
    install_anchor_calibration,
)
from .checkpoint import load_multigrid_checkpoint
from .cli import main
from .distillation import (
    InjectionPriorTeacher,
    attach_injection_prior,
    distillation_requested,
    prior_fit_states,
    prior_requested,
)
from .dtypes import training_dtypes
from .loop import make_datamodule, mlflow_logger, probe_step_memory, train_experiment
from .module import SEModule
from .multigrid import MultiGridMixin, evaluate_multigrid, fit_multigrid_normalizer
from .preconditioner import (
    LossPreconditioner,
    attach_loss_preconditioner,
    build_loss_preconditioner,
    preconditioner_requested,
)
from .stages import (
    CFG_BEHAVIOUR_FIELDS,
    StageManager,
    TargetLossEarlyStop,
    assert_cfg_swappable,
)

# Names that were module-level before the package split and are reached through
# ``injection_hse.train`` by the analysis probes, the GPU parity tests and the differentiability
# gates: the feature-layout helpers, the multi-grid segment gather/scatter, the anchor
# draw law, the identifiability row set, the stage's best-weight keeper and the config
# guard's old spelling.
from .anchoring import _anchor_ranges  # noqa: F401
from .anchoring import _anchor_weight_from  # noqa: F401
from .layout import _BatchGraph  # noqa: F401
from .layout import _flatten_features  # noqa: F401
from .layout import _gather_segments  # noqa: F401
from .layout import _scatter_segments  # noqa: F401
from .layout import _to_nhc  # noqa: F401
from .layout import _unflatten_features  # noqa: F401
from .module import _cancelling_injection_rows  # noqa: F401
from .stages import _assert_cfg_swappable  # noqa: F401
from .stages import _BestWeightKeeper  # noqa: F401

__all__ = [
    "ANCHOR_LOG_DECADES",
    "CFG_BEHAVIOUR_FIELDS",
    "InjectionPriorTeacher",
    "LossPreconditioner",
    "MultiGridMixin",
    "SEModule",
    "StageManager",
    "TargetLossEarlyStop",
    "anchor_build_weights",
    "anchor_calibration_sigmas",
    "assert_cfg_swappable",
    "attach_injection_prior",
    "prior_fit_states",
    "attach_loss_preconditioner",
    "build_loss_preconditioner",
    "distillation_requested",
    "preconditioner_requested",
    "prior_requested",
    "evaluate_multigrid",
    "fit_multigrid_normalizer",
    "install_anchor_calibration",
    "load_multigrid_checkpoint",
    "main",
    "make_datamodule",
    "mlflow_logger",
    "probe_step_memory",
    "train_experiment",
    "training_dtypes",
]
