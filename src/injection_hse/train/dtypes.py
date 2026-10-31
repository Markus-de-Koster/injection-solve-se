"""Training precision: the configured dtype string -> the torch dtypes the pipeline runs in."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config import ExperimentConfig


#: TRAINING precision (``TrainConfig.dtype``) -> the real / complex torch dtypes the SE
#: pipeline runs in. The loaded complex state is cast to the complex dtype at the module
#: boundary and the model parameters live in the matching real dtype; ``"complex64"`` (the
#: default) is float32 params, byte-identical to a run written before the field existed.
_TRAIN_REAL_DTYPE = {"complex64": torch.float32, "complex128": torch.float64}
_TRAIN_COMPLEX_DTYPE = {"complex64": torch.complex64, "complex128": torch.complex128}


def training_dtypes(train_dtype: str) -> tuple[torch.dtype, torch.dtype]:
    """``TrainConfig.dtype`` string -> ``(real_dtype, complex_dtype)`` torch dtypes.

    ``"complex64"`` -> ``(float32, complex64)`` (today's default); ``"complex128"`` ->
    ``(float64, complex128)``. The public helper drivers use to build the physics solver /
    generation-vs-training dtype at the matching precision.
    """
    try:
        return _TRAIN_REAL_DTYPE[train_dtype], _TRAIN_COMPLEX_DTYPE[train_dtype]
    except KeyError as exc:  # pragma: no cover - guarded by the config Literal
        raise ValueError(
            f"unknown training dtype {train_dtype!r} (use 'complex64' or 'complex128')."
        ) from exc


def _lightning_precision(cfg: ExperimentConfig) -> str:
    """The Lightning ``Trainer`` precision for a config's training dtype.

    ``dtype="complex128"`` needs float64 model parameters + inputs, so Lightning runs in
    double precision (``"64"``) regardless of the configured ``precision`` (an AMP/complex
    setting that is meaningless for a double-precision study). ``dtype="complex64"`` keeps the
    configured ``precision`` (default ``"32"``) — byte-identical to before.
    """
    if cfg.train.dtype == "complex128":
        return "64"
    return cfg.train.precision
