"""injection_hse — power-grid-learn: harmonic state estimation on pgml-generated data.

The learning package of the suite. Trains state-estimation models (DNN baseline, GNN,
Graphormer) to recover the full complex harmonic state of a grid from a few measurements,
on data produced by the differentiable physics core :mod:`pgml`.

The design record, the model I/O schema and the known limitations are part of the
published documentation (https://power-grid-ml.readthedocs.io).

Lightweight, dependency-free-of-heavy-ML pieces (``config``, ``curriculum``, ``encoding``,
``masking``, ``io``, ``workflow``) are re-exported here; the training/model/data layers
(``injection_hse.train`` / ``injection_hse.models`` / ``injection_hse.data``) pull in the training stack (lightning +
torch-geometric + mlflow, core dependencies of ``power-grid-learn``) and are imported
explicitly from their submodules.

A training run is described by ``injection_hse.workflow.TrainingRunConfig`` (the ordered curriculum
passes over one module) and executed by ``injection_hse.workflow.run_training`` or its command line
``python -m injection_hse.train --config <run.yaml>``.
"""

from __future__ import annotations

from . import config, curriculum, encoding, io, masking, time_features, workflow
from .config import ExperimentConfig
from .encoding import (
    CartesianEncoding,
    StateEncoding,
    TrigEncoding,
    build_encoding,
    build_injection_encoding,
)
from .io import ModelInput, build_model_input
from .masking import build_masker
from .time_features import TIME_FEATURE_DIM
from .workflow import TrainingRunConfig
from ._switches import ESTIMATOR_IDEAL_SWITCH_OHM, materialize_estimator_grid

#: Distribution version of ``power-grid-learn`` (independent of the pgml engine version
#: it is built on; the compatible pgml range is declared in ``pyproject.toml``).
__version__ = "0.1.0"

__all__ = [
    "config",
    "curriculum",
    "encoding",
    "io",
    "masking",
    "time_features",
    "workflow",
    "ExperimentConfig",
    "TrainingRunConfig",
    "ModelInput",
    "build_model_input",
    "StateEncoding",
    "TrigEncoding",
    "CartesianEncoding",
    "build_encoding",
    "build_injection_encoding",
    "build_masker",
    "TIME_FEATURE_DIM",
    "ESTIMATOR_IDEAL_SWITCH_OHM",
    "materialize_estimator_grid",
    "__version__",
]
