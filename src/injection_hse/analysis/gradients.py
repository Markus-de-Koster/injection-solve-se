"""Per-layer training dynamics: gradient norms during training, weight motion after.

Two complementary views of WHICH model component learns WHEN across the
measurement-scarcity curriculum:

- :class:`GradientNormCollector` — a Lightning callback (one shared instance
  across every curriculum stage, the :class:`~injection_hse.evaluation.HistoryCollector`
  pattern) sampling per-layer-group gradient and parameter L2 norms during
  training. Requires an instrumented run.
- :func:`checkpoint_weight_deltas` — post-hoc: the relative weight motion of
  each layer group between consecutive stage checkpoints. Works on ANY finished
  run's ``checkpoints/stage_*.ckpt`` files, no retraining.

Layer groups are canonical names derived from the parameter names
(:func:`parameter_groups`): ``temporal.memory`` / ``temporal.context_proj`` for
the recurrent wrapper, ``spatial.encoder`` / ``spatial.conv_<k>`` /
``spatial.block_<k>`` / ``spatial.mlp_<k>`` / ``spatial.head`` for the wrapped
spatial model (unprefixed for a bare spatial model), plus ``injection_head``
and ``grid_calibration`` when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

try:  # lightning is a core dependency of power-grid-learn; keep the import legible if missing
    import lightning as L
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "injection_hse.analysis.gradients requires lightning (a dependency of power-grid-learn); "
        "reinstall the package."
    ) from exc

if TYPE_CHECKING:  # heavy / optional imports kept off the import path
    import polars as pl

_FIELDS = ("stage_index", "step", "group", "grad_norm", "param_norm")


def _group_of(name: str) -> str:
    """Canonical layer group of one ``named_parameters`` entry."""
    parts = name.split(".")
    if parts and parts[0] == "model":
        parts = parts[1:]
    if parts[0] in ("memory", "context_proj"):  # TemporalEstimator wrapper
        return f"temporal.{parts[0]}"
    prefix = ""
    if parts[0] == "spatial":
        prefix = "spatial."
        parts = parts[1:]
    head = parts[0]
    if head in ("convs", "norms"):
        return f"{prefix}conv_{parts[1]}"
    if head == "blocks":
        return f"{prefix}block_{parts[1]}"
    if head == "mlp" and len(parts) > 1 and parts[1].isdigit():
        return f"{prefix}mlp_{parts[1]}"
    if head in ("degree_emb", "spatial_bias", "electrical_bias"):
        return f"{prefix}structural_emb"
    if head in ("edge_encoder", "edge_obs_encoder"):
        return f"{prefix}edge_encoder"
    if head in ("head", "norm_out"):
        return f"{prefix}head"
    return prefix + head


def parameter_groups(module: torch.nn.Module) -> dict[str, list[str]]:
    """Canonical layer groups over a module's ``named_parameters``.

    Maps each group name (module docstring) to the parameter names it covers,
    in encounter order. Works on an :class:`~injection_hse.train.SEModule` (whose model
    parameters are prefixed ``model.``) and on a bare
    :class:`~injection_hse.models.base.StateEstimator` alike.
    """
    groups: dict[str, list[str]] = {}
    for name, _ in module.named_parameters():
        groups.setdefault(_group_of(name), []).append(name)
    return groups


class GradientNormCollector(L.Callback):
    """Sample per-layer-group gradient/parameter L2 norms during training.

    One instance is shared across every curriculum stage of both task passes
    (attached via ``StageManager.run(..., extra_callbacks=[collector])`` or the
    workflow's gradient-collection option): ``on_fit_start`` increments the
    stage index; ``on_before_optimizer_step`` — every ``every_n_steps``
    OPTIMIZER steps, after any gradient accumulation and before clipping —
    appends one record per layer group ``{stage_index, step (monotonic across
    stages), group, grad_norm, param_norm}``. Sampling at the optimizer step
    (not each backward) makes the norms correct under gradient accumulation:
    they reflect the full accumulated gradient the optimizer actually applies.
    Export with :meth:`to_frame` / :meth:`write_parquet`
    (``gradient_norms.parquet``).
    """

    def __init__(self, *, every_n_steps: int = 10) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1.")
        self.every_n_steps = int(every_n_steps)
        self.stage_index: int = -1
        self.records: list[dict] = []
        self._step: int = -1
        self._groups: dict[str, list[str]] = {}

    def on_fit_start(self, trainer, pl_module) -> None:  # noqa: D401
        self.stage_index += 1
        self._groups = parameter_groups(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self._step += 1
        if self._step % self.every_n_steps:
            return
        params = dict(pl_module.named_parameters())
        for group, names in self._groups.items():
            grad_sq = 0.0
            param_sq = 0.0
            for name in names:
                p = params[name]
                param_sq += float(p.detach().norm() ** 2)
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().norm() ** 2)
            self.records.append(
                {
                    "stage_index": self.stage_index,
                    "step": self._step,
                    "group": group,
                    "grad_norm": grad_sq**0.5,
                    "param_norm": param_sq**0.5,
                }
            )

    # -- export ---------------------------------------------------------------
    def to_frame(self) -> "pl.DataFrame":
        """The accumulated samples as a polars ``DataFrame`` (one row per group per sample)."""
        import polars as pl

        if not self.records:
            return pl.DataFrame({f: [] for f in _FIELDS})
        return pl.DataFrame(self.records).select(_FIELDS)

    def write_parquet(self, path) -> Path:
        """Write the samples to ``path`` (parquet); returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().write_parquet(p)
        return p


def read_gradient_norms(path) -> "pl.DataFrame":
    """Reload a gradient-norm parquet written by :meth:`GradientNormCollector.write_parquet`."""
    import polars as pl

    return pl.read_parquet(path)


def checkpoint_weight_deltas(ckpt_paths: list) -> "pl.DataFrame":
    """Relative per-group weight motion between consecutive stage checkpoints.

    For each consecutive checkpoint pair the per-group relative delta
    ``‖w_next − w_prev‖ / ‖w_prev‖`` (L2 over all the group's parameters) is
    attributed to the stage that PRODUCED the later checkpoint — which
    component a stage actually trained. Reads the checkpoints' ``state_dict``
    directly (no module rebuild), so it works post-hoc on any finished run.

    ``ckpt_paths`` must be the boundary checkpoints in curriculum order (e.g.
    the sorted ``checkpoints/stage_*.ckpt`` list returned by
    :meth:`~injection_hse.train.StageManager.run`); the stage name is taken from the file
    stem (``stage_<name>``). Returns rows ``{stage, group, rel_delta}``.
    """
    import polars as pl

    if len(ckpt_paths) < 2:
        raise ValueError("need at least two checkpoints to compute weight deltas.")

    states: list[tuple[str, dict]] = []
    for path in ckpt_paths:
        p = Path(path)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        name = p.stem.removeprefix("stage_")
        states.append((name, ckpt["state_dict"]))

    records: list[dict] = []
    for (_, prev), (stage, nxt) in zip(states[:-1], states[1:]):
        shared = [
            k
            for k in prev
            if k in nxt
            and prev[k].dtype.is_floating_point
            and prev[k].shape == nxt[k].shape
        ]
        groups: dict[str, list[str]] = {}
        for k in shared:
            groups.setdefault(_group_of(k), []).append(k)
        for group, keys in groups.items():
            delta_sq = 0.0
            prev_sq = 0.0
            for k in keys:
                delta_sq += float((nxt[k] - prev[k]).norm() ** 2)
                prev_sq += float(prev[k].norm() ** 2)
            rel = (delta_sq**0.5) / (prev_sq**0.5 + 1e-12)
            records.append({"stage": stage, "group": group, "rel_delta": rel})
    return pl.DataFrame(records)


__all__ = [
    "parameter_groups",
    "GradientNormCollector",
    "read_gradient_norms",
    "checkpoint_weight_deltas",
]
