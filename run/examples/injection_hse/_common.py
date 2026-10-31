"""Shared plumbing for the ``injection_hse`` (state-estimation) examples — configure once, reuse.

Collects the boilerplate every SE example repeated — output-directory resolution, the
CUDA/runtime setup, device selection, and the common command-line arguments — so an example
only declares what is genuinely specific to it.

Output paths are anchored to the repository root (via ``__file__``), NOT the current working
directory, so ``pixi run python run/examples/injection_hse/foo.py`` writes under ``data/pgl/`` rather
than polluting the source tree. Set ``PGML_EXPERIMENTS`` to redirect (e.g. a cluster scratch
dir).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

# run/examples/injection_hse/_common.py -> the repository root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def output_dir(name: str) -> Path:
    """Default output directory for an example, anchored at the repository root (not the cwd).

    ``data/pgl/<name>`` under the untracked data root by default; ``$PGML_EXPERIMENTS/<name>``
    when ``PGML_EXPERIMENTS`` is set (the cluster run dir writes its outputs flat). Independent
    of where the script is launched from.
    """
    env = os.environ.get("PGML_EXPERIMENTS")
    root = Path(env) if env else _REPO_ROOT / "data" / "injection_hse"
    return root / name


def configure_runtime(device: torch.device) -> None:
    """Quiet Lightning's INFO chatter and enable Tensor-Core matmul on a capable GPU.

    - Drops Lightning's INFO log level to WARNING so the litlogger "Tip" and GPU-detection
      notices do not clutter the run (warnings/errors still show).
    - On CUDA, sets ``float32_matmul_precision='high'`` (TF32) — a safe speed-up for the float32
      ML model: it affects only float32 matmuls, NOT the complex64/128 physics solve or the
      float64 gradient checks, and silences Lightning's Tensor-Core notice. A run with
      ``TrainConfig.deterministic`` set overrides this back to ``'highest'`` when training
      starts (TF32 rounding is not run-to-run reproducible).
    """
    import lightning  # noqa: F401  (import FIRST so its __init__ INFO-level set does not undo us)

    for name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")


def resolve_device(spec: str) -> torch.device:
    """``"auto"`` -> CUDA if available else CPU; else the named device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def parse_phases(spec):
    """Parse a ``"a,b,c"`` phase list into ``[Phase, ...]`` (or ``None``)."""
    if not spec:
        return None
    from pgml.schemas.grid_schema import Phase

    return [Phase(tok.strip().lower()) for tok in spec.split(",") if tok.strip()]


DTYPES = {"complex64": torch.complex64, "complex128": torch.complex128}


def add_common_args(
    ap: argparse.ArgumentParser,
    *,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-3,
    grad_clip: float = 1.0,
    scheduler: str = "none",
) -> argparse.ArgumentParser:
    """Add the run/optimisation arguments shared by the SE examples (defaults overridable).

    Adds ``--device/--seed/--dtype/--epochs/--batch-size/--lr/--grad-clip/--scheduler/
    --val-fraction/--num-workers/--chunk-size/--no-progress``; an example adds only its own
    data-generation / model knobs on top.
    """
    ap.add_argument(
        "--device", default="auto", help="'auto' (CUDA if available), 'cpu', or 'cuda'."
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dtype", default="complex64", choices=list(DTYPES), help="Solve precision."
    )
    ap.add_argument(
        "--epochs", type=int, default=epochs, help="Max epochs per curriculum stage."
    )
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=grad_clip,
        help="Gradient-norm clip (0 disables). Guards against the loss-spike collapse.",
    )
    ap.add_argument(
        "--scheduler",
        default=scheduler,
        choices=["none", "cosine", "plateau"],
        help="LR schedule (cosine anneals once over the stage's max_epochs).",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Plateau early stopping: end a stage when the validation all-rows pu-MAE has "
        "not improved by >= --min-delta for this many consecutive validation epochs (0 "
        "disables; --epochs stays the hard cap). Stops a stage once it stops learning.",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum val pu-MAE improvement counted as progress by --patience.",
    )
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Solve the scenario batch in slices of at most this many scenarios during data "
        "generation (streams the dense system so a large batch fits GPU memory).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live Lightning progress bar (metrics still go to MLflow + history).",
    )
    return ap


def import_se_helpers():
    """Import ``add_pv_systems`` / ``build_config`` whether run as a script or a package.

    A script run puts ``run/examples/injection_hse/`` on ``sys.path`` (sibling import); a package/test import
    resolves ``run.examples.injection_hse`` from the repo root. Return ``(add_pv_systems, build_config)``.
    """
    try:
        from pgml.grids import add_pv_systems
        from injection_hse.data.scenarios import se_benchmark_scenario_config as build_config
    except ModuleNotFoundError:
        from pgml.grids import add_pv_systems  # pragma: no cover
        from injection_hse.data.scenarios import se_benchmark_scenario_config as build_config
    return add_pv_systems, build_config


__all__ = [
    "output_dir",
    "configure_runtime",
    "resolve_device",
    "parse_phases",
    "add_common_args",
    "import_se_helpers",
    "DTYPES",
]
