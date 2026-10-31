"""The measurement-scarcity curriculum: stage ladders, anchoring schedules, resume.

A state estimator is not trained at its deployment scarcity directly. It walks a LADDER of
:class:`~injection_hse.config.MaskStageConfig` stages, from every node observed down to the handful of
metered locations a real feeder carries, continuing one module's weights across every stage
(:class:`injection_hse.train.StageManager` runs them; :mod:`injection_hse.workflow` runs the task passes).

This module owns the SHAPE of that ladder, independent of any command line:

- the canonical snapshot / sequence / profiled-sequence ladders and the final cosine-anneal stage
  (:func:`snapshot_stages`, :func:`sequence_stages`, :func:`profiled_sequence_stages`,
  :func:`anneal_stage`, all over the generic :func:`build_stages`);
- the per-stage early-stop target, ramped GEOMETRICALLY across the combined ladder
  (:func:`log_ramp`), so the switch between task passes is not a discontinuity in what a
  stage is asked to reach;
- the measurement-anchoring schedule (:func:`parse_anchor_schedule`,
  :func:`apply_anchor_schedule`) — the per-stage weights that move the injection decode from
  physics-led (dense stages) to measurement-dominant (the sparse placement);
- resuming a ladder from a stage checkpoint (:func:`resume_stage_split`,
  :func:`apply_init_checkpoint`).

Stage NAMES are part of the ladder argument, not baked into the builders: they identify a
checkpoint (``stage_<name>.ckpt``) and key the anchoring schedule, so a study that has
already written checkpoints keeps its own spelling by passing its own ladder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .config import MaskStageConfig

#: One ladder rung: ``(stage name, masker kind, observed fraction)``. The fraction is ``None``
#: for the ``full`` and ``placement`` kinds, which do not draw a random observation set.
Rung = tuple[str, str, Optional[float]]

#: The SNAPSHOT ladder — the harmonic-flow surrogate on randomized snapshots: every node observed, then
#: half, then a quarter.
SNAPSHOT_LADDER: tuple[Rung, ...] = (
    ("full", "full", None),
    ("random_50", "random", 0.5),
    ("random_25", "random", 0.25),
)

#: The SEQUENCE ladder — device patterns on coherent sequences: a tenth of the nodes, then the fixed
#: sensor placement (the deployment question).
SEQUENCE_LADDER: tuple[Rung, ...] = (
    ("random_10", "random", 0.10),
    ("placement", "placement", None),
)

#: Rungs of the COMBINED snapshot + sequence ladder — the denominator of the target ramp, so a
#: pass built on its own still places its stages at their positions in the whole curriculum.
CURRICULUM_LENGTH = len(SNAPSHOT_LADDER) + len(SEQUENCE_LADDER)

#: Early-stop target of the cosine-anneal stage. Negligible on purpose: the anneal runs its
#: full epoch budget (or the plateau guard) rather than an early target stop — the placement
#: stage before it already reached whatever the ramp asked for, so a shared target would end
#: the anneal on its first epoch.
ANNEAL_TARGET_FLOOR = 1e-9


def parse_orders(spec: Optional[str]) -> list[int]:
    """Harmonic orders from a comma-separated spec; ``None`` -> ``1..20``.

    The fundamental (order 1) must be included — it anchors the state and the per-unit
    metrics — and at least one order above 1 must remain (the injected harmonic spectrum).
    Duplicates are dropped, the result is sorted. Raises :class:`SystemExit` naming the
    offending spec, so a mistyped command line ends before any data is touched.
    """
    if spec is None:
        return list(range(1, 21))
    try:
        orders = sorted({int(tok) for tok in spec.split(",") if tok.strip()})
    except ValueError as exc:
        raise SystemExit(
            f"--orders must be comma-separated integers, got {spec!r}."
        ) from exc
    if not orders or orders[0] < 1:
        raise SystemExit(f"--orders must be positive integers, got {spec!r}.")
    if 1 not in orders:
        raise SystemExit("--orders must include the fundamental (order 1).")
    if len(orders) < 2:
        raise SystemExit("--orders needs at least one harmonic order above 1.")
    return orders


def log_ramp(target_full: float, target_placement: float, i: int, n: int) -> float:
    """Geometric ramp of a per-stage early-stop target from full-observation to placement.

    ``i`` is the stage's position in a ladder of ``n`` rungs; rung ``0`` returns
    ``target_full`` and rung ``n-1`` returns ``target_placement``.
    """
    t = i / max(1, n - 1)
    return target_full * (target_placement / target_full) ** t


def build_stages(
    ladder: Sequence[Rung],
    *,
    epochs: int,
    target_full: float,
    target_placement: float,
    ladder_length: Optional[int] = None,
    offset: int = 0,
    fundamental_fraction: Optional[float] = None,
    placement_lr: Optional[float] = None,
    placement_scheduler: Optional[str] = None,
) -> list[MaskStageConfig]:
    """Turn a ladder of rungs into :class:`~injection_hse.config.MaskStageConfig` stages.

    Each rung becomes one stage with ``epochs`` as its epoch cap and a target loss taken
    from :func:`log_ramp` at position ``offset + i`` of a ladder of ``ladder_length`` rungs
    (default: the length of ``ladder`` itself). A task pass built on its own therefore keeps
    the targets it would have inside the combined curriculum — pass the combined length and
    the pass's start offset.

    ``fundamental_fraction`` applies to ``random`` rungs only (the share of observed rows
    demoted to fundamental-only smart meters); ``placement_lr`` / ``placement_scheduler``
    apply to ``placement`` rungs only (the finetune's optimizer override, ``None`` =
    inherit the run's :class:`~injection_hse.config.OptimConfig`).
    """
    n_total = len(ladder) if ladder_length is None else int(ladder_length)
    return [
        MaskStageConfig(
            name=name,
            kind=kind,
            observed_fraction=fraction,
            fundamental_fraction=(fundamental_fraction if kind == "random" else None),
            target_loss=log_ramp(target_full, target_placement, offset + i, n_total),
            max_epochs=epochs,
            lr=placement_lr if kind == "placement" else None,
            scheduler=placement_scheduler if kind == "placement" else None,
        )
        for i, (name, kind, fraction) in enumerate(ladder)
    ]


def snapshot_stages(
    *,
    epochs: int,
    target_full: float,
    target_placement: float,
    ladder: Sequence[Rung] = SNAPSHOT_LADDER,
    ladder_length: int = CURRICULUM_LENGTH,
    fundamental_fraction: Optional[float] = None,
) -> list[MaskStageConfig]:
    """The SNAPSHOT pass: the dense-to-sparse ladder on the randomized snapshots.

    The first rungs of the combined curriculum, so the target ramp starts at
    ``target_full`` and continues into :func:`sequence_stages`.
    """
    return build_stages(
        ladder,
        epochs=epochs,
        target_full=target_full,
        target_placement=target_placement,
        ladder_length=ladder_length,
        fundamental_fraction=fundamental_fraction,
    )


def sequence_stages(
    *,
    epochs: int,
    target_full: float,
    target_placement: float,
    ladder: Sequence[Rung] = SEQUENCE_LADDER,
    ladder_length: int = CURRICULUM_LENGTH,
    offset: int = len(SNAPSHOT_LADDER),
    fundamental_fraction: Optional[float] = None,
    placement_lr: Optional[float] = None,
    placement_scheduler: Optional[str] = None,
) -> list[MaskStageConfig]:
    """The SEQUENCE pass: the scarce ladder on the coherent sequences down to the placement.

    Continues the target ramp where the snapshot pass left it (``offset`` rungs in), and ends at
    ``target_placement`` — the deployment scarcity the run is judged at.
    """
    return build_stages(
        ladder,
        epochs=epochs,
        target_full=target_full,
        target_placement=target_placement,
        ladder_length=ladder_length,
        offset=offset,
        fundamental_fraction=fundamental_fraction,
        placement_lr=placement_lr,
        placement_scheduler=placement_scheduler,
    )


def profiled_sequence_stages(
    *,
    epochs: int,
    target_loss: float,
    name: str = "profile_placement",
    lr: Optional[float] = None,
    scheduler: Optional[str] = None,
) -> list[MaskStageConfig]:
    """The profiled-sequence pass: one placement finetune on the PROFILED (time-stamped) sequences.

    A single stage continuing the sequence pass's weights with the periodic time features enabled,
    so the model conditions the placement estimate on the diurnal / weekly / seasonal
    profile structure. The scarcity no longer moves, hence the flat ``target_loss``.
    """
    return [
        MaskStageConfig(
            name=name,
            kind="placement",
            observed_fraction=None,
            target_loss=target_loss,
            max_epochs=epochs,
            lr=lr,
            scheduler=scheduler,
        )
    ]


def anneal_stage(
    *,
    epochs: int,
    name: str = "placement_anneal",
    lr: Optional[float] = None,
) -> MaskStageConfig:
    """The final cosine-anneal placement finetune — the LAST stage of the curriculum.

    A dedicated stage AFTER the constant-rate placement stage: the SAME placement mask, a
    fresh optimizer at the placement learning rate, cosine-annealed ONCE over ``epochs``
    down toward zero. Separating the anneal lets the placement stage make progress at a
    CONSTANT rate (ending on its plateau or its target) and only THEN polish with a decaying
    one — instead of annealing across the whole placement stage, which drives the rate to
    ~0 early and freezes progress while validation is still descending.
    """
    return MaskStageConfig(
        name=name,
        kind="placement",
        observed_fraction=None,
        target_loss=ANNEAL_TARGET_FLOOR,
        max_epochs=epochs,
        lr=lr,
        scheduler="cosine",
    )


def parse_anchor_schedule(spec, *, fixed: bool = False):
    """Parse an anchor-schedule spec -> ``{stage_name: ((vlo, vhi), (ilo, ihi))}``.

    Entry forms per side: a single weight or a per-batch randomization RANGE ``lo~hi``; the
    two sides separate with ``/`` (``name:spec`` applies one spec to BOTH anchors). Examples:
    ``placement:1/0.1`` (fixed, voltage trusted tenfold over the CTs),
    ``random_25:0~1/0~0.1`` (every training batch draws one uniform sample scaling both
    ranges; validation runs at the midpoints). ``None`` / empty -> ``{}``. Raises
    :class:`SystemExit` on a malformed entry or a negative weight (scientific notation like
    ``1e4`` is accepted).

    ``fixed`` collapses every range to its MIDPOINT, so a ranged schedule string reads as the
    fixed-weight regime verbatim: the stages then carry a single weight per side and nothing
    is drawn per batch.
    """
    if not spec:
        return {}

    def _side(name, s):
        parts = s.split("~")
        if len(parts) > 2:
            raise SystemExit(
                f"--anchor-schedule weight for {name!r} must be 'w' or 'lo~hi': {s!r}."
            )
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError:
            raise SystemExit(
                f"--anchor-schedule weight for {name!r} is not a number: {s!r}."
            )
        lo, hi = (vals[0], vals[0]) if len(vals) == 1 else vals
        if not (0 <= lo <= hi):
            raise SystemExit(
                f"--anchor-schedule weight for {name!r} needs 0 <= lo <= hi: {s!r}."
            )
        if fixed:
            mid = (lo + hi) / 2.0
            return (mid, mid)
        return (lo, hi)

    out: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise SystemExit(
                f"--anchor-schedule entry {entry!r} must be 'stage_name:spec' "
                "or 'stage_name:vspec/ispec'."
            )
        name, _, w = entry.partition(":")
        parts = w.split("/")
        if len(parts) > 2:
            raise SystemExit(
                f"--anchor-schedule weight for {name!r} must be 'spec' or "
                f"'vspec/ispec': {w!r}."
            )
        sides = tuple(_side(name, p) for p in parts)
        out[name.strip()] = sides if len(sides) == 2 else (sides[0], sides[0])
    return out


def apply_anchor_schedule(stages, schedule):
    """Set each stage's per-stage anchor weights from ``schedule``.

    ``schedule`` maps a stage NAME to a ``(voltage, current)`` weight pair (the voltage
    anchor is auto-scaled by the per-order admittance downstream; the current anchor takes
    its weight directly). Stages not named keep their inherited (``None``) weights. Returns
    ``(stages, unmatched)`` — a new stage list plus the schedule keys that matched no stage,
    for the caller to warn about.

    A side whose bounds coincide — a fixed ``stage:V/C`` entry, or any range collapsed by
    the ``fixed`` option of :func:`parse_anchor_schedule` — becomes a FIXED stage weight with
    no range, so the solve runs at one strength and nothing is drawn per batch.
    """
    matched: set[str] = set()
    updated = []
    for stage in stages:
        if stage.name in schedule:
            (vlo, vhi), (ilo, ihi) = schedule[stage.name]
            update = {
                "anchor_voltage_weight": (vlo + vhi) / 2.0,
                "anchor_current_weight": (ilo + ihi) / 2.0,
                "anchor_voltage_range": (vlo, vhi) if vhi > vlo else None,
                "anchor_current_range": (ilo, ihi) if ihi > ilo else None,
            }
            updated.append(stage.model_copy(update=update))
            matched.add(stage.name)
        else:
            updated.append(stage)
    return updated, set(schedule) - matched


def parse_stage_weights(spec, *, flag: str) -> dict:
    """Parse a ``name:weight,...`` per-stage schedule -> ``{stage_name: weight}``.

    The generic form behind a per-stage scalar override (the prior-distillation anneal,
    ``--prior-distillation-schedule``): one non-negative weight per named stage, e.g.
    ``placement:1,placement_anneal:0``. ``None`` / empty -> ``{}``. Raises
    :class:`SystemExit` naming ``flag`` on a malformed entry or a negative weight.
    """
    if not spec:
        return {}
    out: dict[str, float] = {}
    for entry in str(spec).split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise SystemExit(f"{flag} entry {entry!r} must be 'stage_name:weight'.")
        name, _, raw = entry.partition(":")
        try:
            weight = float(raw)
        except ValueError:
            raise SystemExit(
                f"{flag} weight for {name.strip()!r} is not a number: {raw!r}."
            ) from None
        if weight < 0:
            raise SystemExit(
                f"{flag} weight for {name.strip()!r} must be >= 0, got {weight}."
            )
        out[name.strip()] = weight
    return out


def apply_stage_weights(stages, schedule, *, field: str):
    """Set ``field`` on each stage named in ``schedule``; return ``(stages, unmatched)``.

    Stages not named keep their inherited (``None``) value. ``unmatched`` are the schedule
    keys that matched no stage, for the caller to warn about — the same contract as
    :func:`apply_anchor_schedule`.
    """
    matched: set[str] = set()
    updated = []
    for stage in stages:
        if stage.name in schedule:
            updated.append(stage.model_copy(update={field: schedule[stage.name]}))
            matched.add(stage.name)
        else:
            updated.append(stage)
    return updated, set(schedule) - matched


def resume_stage_split(
    *task_ladders: Sequence[str],
    init_from: Optional[str],
    start_stage: Optional[str],
) -> tuple[list[str], ...]:
    """The stage names each task pass still has to TRAIN when resuming from a checkpoint.

    Takes one stage-name list per pass (snapshots, sequences, and the profiled sequences
    when enabled) and
    returns them filtered, in the same order. ``start_stage`` names the first stage to
    train; when omitted it is inferred from the ``init_from`` checkpoint filename
    (``stage_<name>.ckpt`` -> the stage AFTER ``<name>`` in the combined ladder across every
    given pass). Without either, everything trains. Raises with the full ladder when a name
    is unknown, or when the checkpoint already covers the whole ladder (nothing left to
    train).
    """
    ladder = [name for names in task_ladders for name in names]
    if start_stage is None:
        if init_from is None:
            return tuple(list(names) for names in task_ladders)
        done = Path(init_from).stem.removeprefix("stage_")
        if done not in ladder:
            raise SystemExit(
                f"--init-from checkpoint stage {done!r} is not in the curriculum ladder "
                f"{ladder}; pass --start-stage to name the first stage to train."
            )
        pos = ladder.index(done) + 1
        if pos == len(ladder):
            raise SystemExit(
                f"--init-from checkpoint {done!r} is the FINAL stage — nothing left to "
                "train. Pass --start-stage to retrain from an earlier stage."
            )
        start_stage = ladder[pos]
    if start_stage not in ladder:
        raise SystemExit(f"--start-stage {start_stage!r} not in the ladder {ladder}.")
    keep = ladder[ladder.index(start_stage) :]
    return tuple([n for n in names if n in keep] for names in task_ladders)


def apply_init_checkpoint(module, path) -> None:
    """Load a stage checkpoint's weights + fitted normalizer(s) into a fresh module.

    The module keeps THIS run's config (stages, optimizer, data paths); only the model
    parameters and the normalizers travel from the checkpoint — the normalizers must match
    the weights (the features the parameters were trained against), so freshly fitted ones
    are overwritten. The checkpoint must come from the same model architecture and harmonic
    orders; a mismatch fails with a clear error before any training.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_orders = ckpt.get("pgl_config", {}).get("orders")
    if ckpt_orders is not None and list(ckpt_orders) != list(module.orders):
        raise SystemExit(
            f"--init-from checkpoint was trained on orders {ckpt_orders}, this run uses "
            f"{list(module.orders)} — the model dimensions cannot match. Use --orders "
            f"{','.join(str(o) for o in ckpt_orders)} or a matching checkpoint."
        )
    try:
        module.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        raise SystemExit(
            f"--init-from checkpoint does not fit this run's model "
            f"(architecture/grid mismatch): {exc}"
        ) from exc
    module.on_load_checkpoint(ckpt)  # restore the fitted normalizer(s)


__all__ = [
    "ANNEAL_TARGET_FLOOR",
    "CURRICULUM_LENGTH",
    "Rung",
    "SNAPSHOT_LADDER",
    "SEQUENCE_LADDER",
    "anneal_stage",
    "apply_anchor_schedule",
    "apply_init_checkpoint",
    "apply_stage_weights",
    "build_stages",
    "log_ramp",
    "parse_anchor_schedule",
    "parse_orders",
    "parse_stage_weights",
    "resume_stage_split",
    "snapshot_stages",
    "sequence_stages",
    "profiled_sequence_stages",
]
