"""Training-history collection across the two-task curriculum + the loss-curve plot.

:class:`HistoryCollector` is a Lightning callback that accumulates one record per validation
epoch ACROSS multiple ``Trainer.fit`` calls — so a single instance, attached to both
the snapshot and sequence :class:`~injection_hse.train.StageManager` passes, captures the whole run with a
monotonic global epoch and a stage index that increments at every ``fit``. The history is
written to parquet (``training_history.parquet``); :func:`plot_loss_curves` renders the
train/val loss with a vertical line at each stage boundary and a band at the A→B task switch.

A run that was not collected — anything trained outside the workflow, or fetched from a
cluster — still has its curves in the MLflow FILE store it logged to.
:func:`read_mlflow_metrics` parses that store without a server and :func:`mlflow_history`
reshapes it into the SAME per-epoch frame the collector writes, so one reader
(:func:`read_training_history`, parquet first) and one plot serve both sources.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pgml.evaluation.style import save_html

try:  # lightning is a core dependency of power-grid-learn; keep the import legible if missing
    import lightning as L
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "injection_hse.evaluation.history requires lightning (a dependency of power-grid-learn); "
        "reinstall the package."
    ) from exc

_METRICS = ("train_loss", "val_loss", "val_pu_mae", "val_pu_mae_unobs")
_FIELDS = ("global_epoch", "stage_index", *_METRICS)

# Diagnostics recorded next to the headline metrics whenever the module logs them, so the
# reported loss can be read apart into WHAT it is made of and WHERE the error sits. The
# reported ``*_loss`` is the blend of the supervised terms — a run with a positive
# ``loss.injection_state_loss_split`` can drive its total down while the voltage term (the
# one the per-unit metrics track) stands still, which is indistinguishable in the total
# alone. ``val_pu_mae_obs`` is the anchored solve's agreement with its own measurements,
# the complement of the estimated-row quality. ``*_plain_decode_loss`` is the prior term on
# the slack-only decode (``loss.plain_decode_weight``) and ``val_pu_mae_unobs_plain`` that
# decode's own estimated-row quality — what the network achieves beside the anchored solve.
# ``*_prior_distillation_loss`` is the injection head's error against the LINEAR PRIOR's
# injections (``loss.prior_distillation_weight``) — how much of the optimal linear map the
# network has copied, which is the quantity that term is driving.
_DIAGNOSTICS = (
    "train_voltage_loss",
    "train_injection_state_loss",
    "train_plain_decode_loss",
    "train_prior_distillation_loss",
    "val_voltage_loss",
    "val_injection_state_loss",
    "val_plain_decode_loss",
    "val_prior_distillation_loss",
    "train_injection_share",
    "val_injection_share",
    "val_pu_mae_obs",
    "val_pu_mae_unobs_fund",
    "val_pu_mae_unobs_harm",
    "val_pu_mae_unobs_plain",
)
# Per-ORDER columns are discovered from the logged metrics (their names carry the order), so
# the history follows whatever harmonic set the run solves. The ``inj_rel_*`` pair is the
# injection identifiability at the nodes where a load and a generator share a bus (their
# fundamentals cancel; only a spectral signature can separate them) versus everywhere else.
_ORDER_PREFIXES = (
    "val_pu_mae_h",
    "val_pu_mae_unobs_h",
    "val_pu_mae_unobs_plain_h",
    "val_rel_unobs_h",
    "val_inj_rel_cancel_h",
    "val_inj_rel_other_h",
)


class HistoryCollector(L.Callback):
    """Accumulate per-validation-epoch metrics across multiple ``Trainer.fit`` calls.

    One instance is shared across every curriculum stage of BOTH the snapshot and sequence
    :class:`~injection_hse.train.StageManager` passes (attached via ``StageManager.run(...,
    extra_callbacks=[collector])``). ``on_fit_start`` increments the stage index (so each
    stage's ``fit`` is a distinct ``stage_index``, starting at 0); ``on_validation_epoch_end``
    appends ``{global_epoch (monotonic), stage_index, train_loss, val_loss, val_pu_mae,
    val_pu_mae_unobs}`` (the Lightning sanity-check epoch is skipped). ``train_loss`` is the
    epoch MEAN, read from the module's :attr:`~injection_hse.train.SEModule.train_epoch_loss` (validation
    runs after every training batch, so the mean is complete) — Lightning's
    ``callback_metrics["train_loss"]`` at this point is only the last minibatch; the val
    metrics come from ``callback_metrics``. A module that exposes no ``train_epoch_loss`` falls
    back to the logged ``train_loss``. ``val_pu_mae`` is the all-rows early-stop monitor;
    ``val_pu_mae_unobs`` is the headline unobserved quality. The records survive across ``fit``
    calls because the callback object is reused.

    Every diagnostic the module logs is recorded alongside those: the supervised loss split
    into its ``voltage`` / ``injection_state`` parts, the observed-row per-unit MAE, the
    fundamental-vs-harmonics band pair, and one column per solved order
    (``val_pu_mae_unobs_h3``, ``val_rel_unobs_h3``, …). They are what makes a loss curve
    readable — which term is falling, and in which frequency band the estimate is actually
    improving. The diagnostic columns are discovered from the logged metric names (so they
    follow the run's orders); the plain decode's ``val_pu_mae_unobs_plain`` /
    ``val_pu_mae_unobs_plain_h{h}`` and the ``plain_decode`` loss terms are recorded the same
    way. A module that logs none of them yields only the four headline columns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage_index: int = -1
        self.global_epoch: int = -1
        self.records: list[dict] = []

    def on_fit_start(self, trainer, pl_module) -> None:  # noqa: D401
        self.stage_index += 1

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        import math

        if getattr(trainer, "sanity_checking", False):
            return
        metrics = trainer.callback_metrics
        # The training loss is the epoch MEAN, read from the module (validation runs after
        # every training batch, so the module's accumulator is complete). Lightning's
        # ``callback_metrics["train_loss"]`` here is only the LAST minibatch (one epoch stale
        # for the aggregated variant), which is exactly the value this used to record. Fall
        # back to it for a plain module that exposes no ``train_epoch_loss``.
        train_loss = getattr(pl_module, "train_epoch_loss", None)
        if train_loss is None:
            val = metrics.get("train_loss")
            train_loss = float(val) if val is not None else None
        elif not math.isfinite(float(train_loss)):
            train_loss = None
        # skip the validation epoch before any training has produced a loss
        if train_loss is None and metrics.get("val_loss") is None:
            return
        self.global_epoch += 1
        rec = {"global_epoch": self.global_epoch, "stage_index": self.stage_index}
        rec["train_loss"] = (
            float(train_loss) if train_loss is not None else float("nan")
        )
        for key in ("val_loss", "val_pu_mae", "val_pu_mae_unobs"):
            val = metrics.get(key)
            rec[key] = float(val) if val is not None else float("nan")
        for key in _DIAGNOSTICS:
            if key in metrics:
                rec[key] = float(metrics[key])
        for key in metrics:
            if key.startswith(_ORDER_PREFIXES) and key[-1].isdigit():
                rec[str(key)] = float(metrics[key])
        self.records.append(rec)

    # -- export ---------------------------------------------------------------
    def to_frame(self):
        """The accumulated history as a polars ``DataFrame`` (one row per validation epoch).

        The headline columns come first, then whichever diagnostic / per-order columns the run
        logged (an epoch missing one — a stage that reports a metric another does not — gets a
        null there rather than a ragged frame).

        Built COLUMN-wise against an explicit schema, never row-wise: a diagnostic that only
        starts partway through a run (the identifiability metrics report nothing while a stage
        is fully observed, i.e. for the whole first stage) is null in every early row, and
        row-wise construction infers such a column's type from the first ``infer_schema_length``
        rows only — it would type it Null and then fail on the first real value, at
        ``write_parquet``, after the whole run has trained.
        """
        import polars as pl

        seen: list[str] = []
        for rec in self.records:
            seen.extend(k for k in rec if k not in _FIELDS and k not in seen)
        fields = (*_FIELDS, *sorted(seen))
        schema = {
            f: (pl.Int64 if f in ("global_epoch", "stage_index") else pl.Float64)
            for f in fields
        }
        data = {f: [rec.get(f) for rec in self.records] for f in fields}
        return pl.DataFrame(data, schema=schema)

    def write_parquet(self, path) -> Path:
        """Write the history to ``path`` (parquet); returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().write_parquet(p)
        return p


def read_history(path):
    """Reload a training-history parquet written by :meth:`HistoryCollector.write_parquet`."""
    import polars as pl

    return pl.read_parquet(path)


# =============================================================================
# The other source: the MLflow file store
# =============================================================================
#: MLflow series that are bookkeeping rather than run quality (their per-epoch value says
#: nothing about the model), and the per-step training loss the epoch mean replaces.
_MLFLOW_SKIP: frozenset[str] = frozenset({"epoch", "train_loss_epoch"})

#: The metric MLflow logs at every validation epoch's end that corresponds to the parquet
#: history's ``train_loss`` (the epoch MEAN; the bare ``train_loss`` series is per batch).
_MLFLOW_TRAIN_EPOCH = "train_loss_epoch"


def read_mlflow_metrics(mlruns, *, run_id: Optional[str] = None):
    """Parse an MLflow FILE store (``mlruns/``) into a tidy metric table — no server.

    The file store keeps one plain-text file per metric under
    ``<mlruns>/<experiment>/<run>/metrics/<name>``, each line
    ``<timestamp_ms> <value> <step>``. This reads them directly, which is what makes a report
    usable on a run FETCHED from a cluster (the tracking URI in its config points at a
    directory that no longer exists).

    Every curriculum stage fits a fresh Lightning ``Trainer``, so the logged ``step``
    RESTARTS at each stage boundary. Segmenting each metric on its OWN restarts is not
    enough: a metric that logs no point during a stage (a short stage cut by the
    target-loss early stop) never sees that restart and its segment indices slide out of
    step with every other metric. TIMESTAMPS, unlike steps, are monotone across the whole
    run, so the boundaries are taken once from a REFERENCE series — the metric with the
    most restarts, ties broken by the most points — and every point of every metric is
    dated against them (``segment`` = how many boundaries precede it). A restart is a
    step that DROPS; a step merely repeated (Lightning logs ``epoch`` twice at the same
    step) stays inside its stage. ``global_step`` then offsets each segment past the
    longest step seen in it across all metrics, so metrics logged at different rates
    share one axis.

    A two-task run that keeps one logger across both passes is ONE tracked run: its
    segments are the concatenated curricula of both task configs, so its stage labels are
    the stage names of both configs, in order.

    Parameters
    ----------
    mlruns:
        The store root. A missing / empty / metric-free store yields an EMPTY table
        rather than an error — a run without tracking must still produce its report.
    run_id:
        Which run to read; by default the most recently started one (ties broken by the
        directory's modification time).

    Returns
    -------
    polars.DataFrame
        ``metric, step, value, timestamp_ms, segment, global_step``, ordered by metric
        and then by the logging order in the file.
    """
    import polars as pl

    schema = {
        "metric": pl.Utf8,
        "step": pl.Int64,
        "value": pl.Float64,
        "timestamp_ms": pl.Int64,
        "segment": pl.Int64,
        "global_step": pl.Int64,
    }
    directory = _resolve_mlflow_run(mlruns, run_id=run_id)
    if directory is None:
        return pl.DataFrame(schema=schema)

    series: dict[str, list[tuple[int, float, int]]] = {}
    for path in sorted((directory / "metrics").glob("*")):
        if not path.is_file():
            continue
        points: list[tuple[int, float, int]] = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append((int(parts[0]), float(parts[1]), int(parts[2])))
            except ValueError:  # a partially flushed line
                continue
        if points:
            series[path.name] = points
    if not series:
        return pl.DataFrame(schema=schema)

    # one canonical set of stage boundaries for the whole run, in TIMESTAMPS: the
    # series that saw the most restarts (the most stages) dates all the others
    restarts = {name: _step_restarts(points) for name, points in series.items()}
    reference = max(
        sorted(series), key=lambda name: (len(restarts[name]), len(series[name]))
    )
    boundaries = sorted(restarts[reference])

    segments: dict[str, list[int]] = {}
    span: dict[int, int] = {}
    for name, points in series.items():
        marks = [bisect_right(boundaries, timestamp) for timestamp, _v, _s in points]
        segments[name] = marks
        for (_ts, _value, step), segment in zip(points, marks):
            span[segment] = max(span.get(segment, 0), step)
    offset: dict[int, int] = {}
    running = 0
    for segment in sorted(span):
        offset[segment] = running
        running += span[segment] + 1

    records: list[dict] = []
    for name in sorted(series):
        for (timestamp, value, step), segment in zip(series[name], segments[name]):
            records.append(
                {
                    "metric": name,
                    "step": step,
                    "value": value,
                    "timestamp_ms": timestamp,
                    "segment": segment,
                    "global_step": offset[segment] + step,
                }
            )
    return pl.DataFrame(records, schema=schema)


def _step_restarts(points: Sequence[tuple[int, float, int]]) -> list[int]:
    """Timestamps at which a metric's ``step`` DROPS — one per fresh ``Trainer`` fit.

    ``points`` are the ``(timestamp_ms, value, step)`` triples of one metric in logging
    order. A drop, not a repeat: Lightning logs some metrics (``epoch``) twice at the
    SAME step within one stage, and two stages may well end on the same step, so only a
    decrease is evidence that the step counter was reset.
    """
    timestamps: list[int] = []
    previous: Optional[int] = None
    for timestamp, _value, step in points:
        if previous is not None and step < previous:
            timestamps.append(timestamp)
        previous = step
    return timestamps


def _resolve_mlflow_run(mlruns, *, run_id: Optional[str] = None) -> Optional[Path]:
    """The run directory to read from a file store (``None`` when there is none)."""
    root = Path(mlruns)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for experiment in sorted(root.iterdir()):
        if not experiment.is_dir() or experiment.name.startswith("."):
            continue
        for run in sorted(experiment.iterdir()):
            if run.is_dir() and (run / "metrics").is_dir():
                if run_id is None or run.name == run_id:
                    candidates.append(run)
    if not candidates:
        return None

    def _started(run: Path) -> tuple[int, float]:
        meta = run / "meta.yaml"
        start = 0
        if meta.exists():
            for line in meta.read_text().splitlines():
                if line.startswith("start_time:"):
                    try:
                        start = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        start = 0
                    break
        return start, run.stat().st_mtime

    return max(candidates, key=_started)


def _empty_history():
    """A zero-row history frame carrying the headline columns (the typed empty case)."""
    import polars as pl

    return pl.DataFrame(
        schema={
            f: (pl.Int64 if f in ("global_epoch", "stage_index") else pl.Float64)
            for f in _FIELDS
        }
    )


def mlflow_history(metrics):
    """An MLflow metric table (:func:`read_mlflow_metrics`) as a per-epoch history frame.

    The same shape :class:`HistoryCollector` writes — one row per VALIDATION epoch, a
    monotonic ``global_epoch``, a ``stage_index`` that increments at every stage boundary,
    and one column per logged metric — so a collected run and a merely tracked one plot
    through the same :func:`plot_loss_curves` / :func:`plot_loss_components`.

    The rows are anchored on the validation points: every run-global step at which a
    ``val_*`` metric was logged (a store with no validation metric at all falls back to the
    steps of every series, so a train-only run still yields curves). ``train_loss`` is the
    epoch MEAN — the ``train_loss_epoch`` series the module publishes at each validation
    epoch's end — falling back to the last per-batch ``train_loss`` at or before the anchor
    within the same stage. Lightning's own ``epoch`` counter is dropped.

    A metric that logged nothing at an anchor is null there rather than carried forward: a
    stage that reports a diagnostic another does not must read as absent, not as stale.

    Returns an empty (typed) frame for an empty metric table.
    """
    import polars as pl

    if metrics.height == 0:
        return _empty_history()
    # per metric: {run-global step: value} plus the segment each step belongs to
    values: dict[str, dict[int, float]] = {}
    segment_of: dict[int, int] = {}
    val_steps: set[int] = set()
    all_steps: set[int] = set()
    for row in metrics.iter_rows(named=True):
        name = str(row["metric"])
        step = int(row["global_step"])
        segment_of[step] = int(row["segment"])
        all_steps.add(step)
        if name.startswith("val_"):
            val_steps.add(step)
        if name in _MLFLOW_SKIP and name != _MLFLOW_TRAIN_EPOCH:
            continue
        values.setdefault(name, {})[step] = float(row["value"])

    anchors = sorted(val_steps or all_steps)
    if not anchors:
        return _empty_history()
    train_steps = sorted(values.get("train_loss", {}))
    epoch_loss = values.pop(_MLFLOW_TRAIN_EPOCH, {})

    records: list[dict] = []
    for global_epoch, step in enumerate(anchors):
        record = {"global_epoch": global_epoch, "stage_index": segment_of[step]}
        for name, points in values.items():
            if step in points:
                record[name] = points[step]
        train_loss = epoch_loss.get(step, record.get("train_loss"))
        if train_loss is None and train_steps:
            # the anchor carries no training point of its own: take the last one before it,
            # but never across a stage boundary (that value belongs to another fit)
            position = bisect_right(train_steps, step) - 1
            if (
                position >= 0
                and segment_of[train_steps[position]] == record["stage_index"]
            ):
                train_loss = values["train_loss"][train_steps[position]]
        record["train_loss"] = train_loss
        records.append(record)

    seen: list[str] = []
    for record in records:
        seen.extend(k for k in record if k not in _FIELDS and k not in seen)
    fields = (*_FIELDS, *sorted(seen))
    schema = {
        f: (pl.Int64 if f in ("global_epoch", "stage_index") else pl.Float64)
        for f in fields
    }
    return pl.DataFrame(
        {f: [record.get(f) for record in records] for f in fields}, schema=schema
    )


def stage_manifest(stage_names: Sequence[str]) -> dict:
    """Curriculum stage NAMES as the manifest structure the plots label their guides from.

    An MLflow file store records no stage names — the curriculum config does. This wraps a
    list of them (in curriculum order, index-aligned with the history's ``stage_index``) in
    the ``{"stages": {index: {...}}}`` shape :func:`plot_loss_curves` reads, so a run without
    a workflow ``manifest.json`` still gets named stage boundaries.
    """
    return {
        "stages": {
            str(i): {"stage_name": str(name)} for i, name in enumerate(stage_names)
        }
    }


def read_training_history(run_dir, *, mlruns=None, stage_names: Sequence[str] = ()):
    """A finished run's training history and the stage labels for it, from whichever source.

    ``training_history.parquet`` when the run was collected (the workflow writes it), else
    the MLflow file store — ``mlruns`` when given, otherwise ``<run_dir>/mlruns`` — reshaped
    to the same frame by :func:`mlflow_history`. Neither present yields an empty frame, which
    every plot here treats as "no figure" rather than an error.

    The labels come from the run's ``manifest.json`` when it has one (that is also where the
    task blocks come from); ``stage_names`` supplies them otherwise, e.g. from the
    curriculum config of a multi-grid run, which writes no such manifest.

    Returns
    -------
    tuple
        ``(history, manifest)`` — a polars ``DataFrame`` and the label dict, ready for
        :func:`plot_loss_curves`.
    """
    root = Path(run_dir)
    manifest: dict = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    if not manifest.get("stages") and stage_names:
        manifest = {**manifest, **stage_manifest(stage_names)}
    parquet = root / "training_history.parquet"
    if parquet.is_file():
        return read_history(parquet), manifest
    store = Path(mlruns) if mlruns is not None else root / "mlruns"
    return mlflow_history(read_mlflow_metrics(store)), manifest


def _stage_boundaries(history) -> list[tuple[int, int]]:
    """``[(stage_index, first_global_epoch)]`` — the first epoch of each stage, in order."""
    out: list[tuple[int, int]] = []
    seen: set[int] = set()
    stages = history["stage_index"].to_list()
    epochs = history["global_epoch"].to_list()
    for si, ep in zip(stages, epochs):
        if int(si) not in seen:
            seen.add(int(si))
            out.append((int(si), int(ep)))
    return out


def _stage_label(manifest: dict, stage_index: int) -> str:
    """Concise pass-prefixed label for a stage from the manifest.

    ``snap·full``, ``snap·r50%``, ``seq·place``, ``prof·anneal`` — the pass prefix
    disambiguates same-kind stages across passes (every pass's finetune is a
    ``placement`` kind), and an anneal stage is named by its schedule rather
    than its mask. A manifest without per-stage pass labels falls back to the bare
    short form.
    """
    stages = manifest.get("stages", {})
    info = stages.get(str(stage_index), stages.get(stage_index, {}))
    if not info:
        return f"stage {stage_index}"
    name = info.get("stage_name") or ""
    kind = info.get("kind", "")
    frac = info.get("observed_fraction")
    if "anneal" in name:
        short = "anneal"
    elif kind == "random" and frac is not None:
        short = f"r{frac:.0%}"
    elif kind == "placement":
        short = "place"
    elif kind == "full":
        short = "full"
    else:
        short = name or f"stage {stage_index}"
    task = info.get("task")
    prefix = PASS_PREFIXES.get(task, task)
    return f"{prefix}·{short}" if task else short


def _paper_stage_label(manifest: dict, stage_index: int) -> str:
    """Observability-focused stage label for the compact paper figure.

    ``100%`` / ``50%`` / ``25%`` / ``10%`` for the mask fractions,
    ``placement`` / ``anneal`` for the finetune stages — the pass prefix is
    dropped because the pass bands already shade the figure.
    """
    stages = manifest.get("stages", {})
    info = stages.get(str(stage_index), stages.get(stage_index, {}))
    name = info.get("stage_name") or ""
    kind = info.get("kind", "")
    frac = info.get("observed_fraction")
    if "anneal" in name:
        return "anneal"
    if kind == "full":
        return "100%"
    if kind == "random" and frac is not None:
        return f"{frac:.0%}"
    if kind == "placement":
        return "placement"
    return name or f"stage {stage_index}"


#: Display names of the curriculum passes a run manifest labels ``"A"`` / ``"B"`` / ``"C"``.
#: The stored letters are data (an older manifest carries them), so only the display is
#: spelled out: a reader of a figure should not have to know the letters.
PASS_NAMES = {"A": "snapshots", "B": "sequences", "C": "profiled sequences"}
#: The same, abbreviated for a per-stage axis annotation (``seq·r10%``).
PASS_PREFIXES = {"A": "snap", "B": "seq", "C": "prof"}


def _pass_name(task: str) -> str:
    """Display name of a manifest pass label (unknown labels pass through)."""
    return PASS_NAMES.get(task, task)


def _pass_blocks(manifest: dict) -> list[tuple[int, str]]:
    """``[(first_stage_index, pass_label)]`` for each consecutive pass block, in order.

    Empty when the manifest carries no per-stage ``task`` fields (older runs) —
    the caller then falls back to the single ``switch_stage_index`` band.
    """
    stages = manifest.get("stages", {})
    task_of = {
        int(k): v.get("task")
        for k, v in stages.items()
        if isinstance(v, dict) and v.get("task")
    }
    blocks: list[tuple[int, str]] = []
    prev = None
    for si in sorted(task_of):
        if task_of[si] != prev:
            blocks.append((si, task_of[si]))
            prev = task_of[si]
    return blocks


def _logsafe(values) -> np.ndarray:
    """Values as a float array with non-finite / non-positive entries set to NaN.

    A log y-axis cannot draw a zero or negative point, and matplotlib would otherwise connect
    the first real point to one plotted at ``-inf``. The vacuous unobserved pu-MAE in a fully
    observed stage (no unobserved rows) is exactly such a leading 0; masking it to NaN makes
    the unobserved curve START at its first measured point. matplotlib skips NaNs in a line.
    """
    a = np.asarray(values, dtype=float)
    a[~np.isfinite(a) | (a <= 0.0)] = np.nan
    return a


def _draw_stage_marks(
    ax, boundaries, epochs, manifest: dict, *, paper: bool = False, labels: bool = True
) -> None:
    """Shade the task blocks and mark every stage boundary on ``ax``.

    One shaded band per TASK after the first (A stays unshaded), so every dataset switch is
    unmistakable; older manifests without per-stage tasks fall back to the single A→B band. A
    vertical line marks the first epoch of each stage, annotated with the stage's short label
    (``labels=False`` for a stacked panel whose sibling above it already carries them).
    """
    switch_stage = manifest.get("switch_stage_index")
    blocks = _pass_blocks(manifest)
    if not blocks and switch_stage is not None:
        blocks = [(0, "A"), (int(switch_stage), "B")]
    switch_indices: set[int] = set()
    if epochs and len(blocks) > 1:
        first_ep = {si: e for si, e in boundaries}
        shades = ["0.88", "0.78", "0.68"]
        for i, (si, task) in enumerate(blocks[1:]):
            start = first_ep.get(si)
            if start is None:
                continue
            switch_indices.add(si)
            nxt = blocks[i + 2][0] if i + 2 < len(blocks) else None
            end = first_ep.get(nxt, max(epochs)) if nxt is not None else max(epochs)
            ax.axvspan(
                start,
                end,
                color=shades[min(i, len(shades) - 1)],
                alpha=0.5,
                zorder=0,
                label=_pass_name(task),
            )

    ymax = ax.get_ylim()[1]
    for si, e in boundaries:
        is_switch = si in switch_indices
        ax.axvline(
            e,
            color="tab:red" if is_switch else "0.4",
            linestyle="-" if is_switch else "--",
            linewidth=2.0 if is_switch else 1.0,
            zorder=1,
        )
        if not labels:
            continue
        ax.annotate(
            _paper_stage_label(manifest, si) if paper else _stage_label(manifest, si),
            xy=(e, ymax),
            xytext=(2, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=7 if paper else 6,
            color="tab:red" if is_switch else "0.3",
        )


def plot_loss_curves(
    history,
    manifest: Optional[dict] = None,
    *,
    out_svg: Optional[str] = None,
    out_html: Optional[str] = None,
    out_pdf: Optional[str] = None,
    curves: str = "all",
    paper: bool = False,
):
    """Train/val loss vs cumulative epoch with stage boundaries + the A→B task-switch band.

    THE loss-curve figure of a training run, whichever source the history came from: the
    collected ``training_history.parquet`` or the MLflow file store reshaped by
    :func:`mlflow_history` (:func:`read_training_history` picks).

    The x-axis is the epoch counted CONTINUOUSLY across every curriculum stage of both task
    fits (each stage early-stops at its per-stage target, so the total is well below
    ``max_epochs`` × n_stages). ``history`` is a polars ``DataFrame`` (from :func:`read_history`
    or :meth:`HistoryCollector.to_frame`); ``manifest`` supplies the per-stage labels and the
    A→B switch stage index (``None`` / no ``stages`` entry leaves the boundaries unlabelled —
    see :func:`stage_manifest` for naming them from a curriculum config). Vertical lines mark
    the first epoch of each stage; the sequence half is shaded so the dataset switch is
    unmistakable. Loss is on a log y-axis (non-positive points — e.g. the vacuous unobserved
    MAE before any node is masked — are dropped so a curve starts at its first measured
    point). No internal title (caption externally).

    ``curves`` selects the traces: ``"all"`` (train/val loss + val pu-MAE all-rows/unobserved) or
    ``"loss"`` (train + val loss only). A multi-grid run's ``val_pu_mae_worst_grid`` is drawn
    alongside them when the history carries it — its gap to the pooled curve is the cross-grid
    spread. ``paper=True`` renders a compact wide 1-column figure
    (fits one column of a two-column paper) with print-sized fonts, vertical
    observability-focused stage labels (``100%``/``50%``/.../``placement``/``anneal``), and the
    y-label pulled into the tick-label column. The matplotlib figure is written to ``out_svg`` and/or
    ``out_pdf``; an interactive plotly HTML is written to ``out_html`` only when given (the stage
    annotations overlap in HTML — prefer the SVG). The matplotlib ``Figure`` is returned, or
    ``None`` for an empty history (a run with no tracking contributes no figure).
    """
    import matplotlib.pyplot as plt

    manifest = manifest or {}
    if history.height == 0:
        return None
    ep = history["global_epoch"].to_list()
    train = _logsafe(history["train_loss"].to_list())
    val = _logsafe(history["val_loss"].to_list())
    show_mae = curves != "loss"
    boundaries = _stage_boundaries(history)
    switch_stage = manifest.get("switch_stage_index")

    rc = (
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
        if paper
        else {}
    )
    figsize = (4.8, 2.0) if paper else (8.4, 4.6)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ms = 3 if paper else 4
        ax.plot(ep, train, color="tab:blue", marker=".", ms=ms, label="train loss")
        ax.plot(ep, val, color="tab:orange", marker=".", ms=ms, label="val loss")
        if show_mae:
            if "val_pu_mae" in history.columns:
                ax.plot(
                    ep,
                    _logsafe(history["val_pu_mae"].to_list()),
                    color="tab:red",
                    marker=".",
                    ms=ms,
                    alpha=0.8,
                    label="val pu-MAE (all rows)",
                )
            ax.plot(
                ep,
                _logsafe(history["val_pu_mae_unobs"].to_list()),
                color="tab:green",
                marker=".",
                ms=ms,
                alpha=0.7,
                label="val pu-MAE (unobs.)",
            )
            # multi-grid runs only: the WORST member's error, whose gap to the pooled curve
            # is the cross-grid spread (a single-grid history has no such column)
            if "val_pu_mae_worst_grid" in history.columns:
                ax.plot(
                    ep,
                    _logsafe(history["val_pu_mae_worst_grid"].to_list()),
                    color="tab:purple",
                    marker=".",
                    ms=ms,
                    alpha=0.7,
                    label="val pu-MAE (worst grid)",
                )
        ax.set_yscale("log")
        _draw_stage_marks(ax, boundaries, ep, manifest, paper=paper)
        ax.set_xlabel("Epoch (cumulative across stages)")
        # the compact figure pulls the y-label into the tick-label column (the sparse
        # log-decade ticks leave the vertical room), saving horizontal figure space
        ax.set_ylabel(
            "Loss" if not show_mae else "Loss / pu-MAE",
            labelpad=-12 if paper else 4,
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, loc="best")

        if out_svg is not None or out_pdf is not None:
            from pgml.evaluation.style import save_figure

            if out_svg is not None:
                save_figure(fig, out_svg)
            if out_pdf is not None:
                save_figure(fig, out_pdf)

    if out_html is not None:
        _plotly_loss_curves(history, manifest, boundaries, switch_stage, out_html)

    return fig


def _order_columns(history, prefix: str) -> list[tuple[int, str]]:
    """``[(order, column)]`` of the per-order columns named ``<prefix><order>``, ascending."""
    out: list[tuple[int, str]] = []
    for col in history.columns:
        if col.startswith(prefix) and col[len(prefix) :].isdigit():
            out.append((int(col[len(prefix) :]), col))
    return sorted(out)


def plot_loss_components(
    history,
    manifest: dict,
    *,
    out_svg: Optional[str] = None,
    out_pdf: Optional[str] = None,
):
    """Decompose the training curve: which loss term falls, and which harmonic band improves.

    The reported loss is a blend (:attr:`injection_hse.config.LossConfig.injection_state_loss_split`),
    and its per-unit state metrics average over harmonic orders whose magnitudes differ by
    decades — so a falling total loss is compatible with a state estimate that is standing
    still, and an aggregate MAE hides a regression confined to one band. Three stacked panels
    over the same cumulative-epoch axis as :func:`plot_loss_curves` separate the two effects:

    1. **supervised loss terms** — the reported total next to its ``voltage``,
       ``injection_state``, ``prior_distillation`` and ``plain_decode`` parts, so the term
       actually driving the total is visible;
    2. **per-order per-unit MAE on the estimated rows** — each order's contribution to the
       headline ``val_pu_mae_unobs``, which the fundamental dominates;
    3. **per-order relative error** — each order against its OWN signal scale, where a badly
       estimated small harmonic reads large instead of vanishing into the per-unit average.

    ``history`` is a polars ``DataFrame``; columns absent from an older run are simply not
    drawn. Returns the matplotlib ``Figure``, or ``None`` when the history carries none of the
    diagnostic columns (nothing to decompose).
    """
    import matplotlib.pyplot as plt

    cols = set(history.columns)
    pu_orders = _order_columns(history, "val_pu_mae_unobs_h")
    rel_orders = _order_columns(history, "val_rel_unobs_h")
    loss_parts = [
        (c, lbl)
        for c, lbl in (
            ("val_loss", "total (reported)"),
            ("val_voltage_loss", "voltage term"),
            ("val_injection_state_loss", "injection term"),
            ("val_prior_distillation_loss", "prior-distillation term"),
            ("val_plain_decode_loss", "plain-decode term"),
        )
        if c in cols
    ]
    if not pu_orders and not rel_orders and len(loss_parts) < 2:
        return None

    ep = history["global_epoch"].to_list()
    boundaries = _stage_boundaries(history)
    # one panel per available decomposition, stacked over the shared epoch axis
    panels = [p for p in (loss_parts, pu_orders, rel_orders) if p]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8.4, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    axes = list(axes[:, 0])
    term_colors = {
        "total (reported)": "0.25",
        "voltage term": "tab:blue",
        "plain-decode term": "tab:green",
        "prior-distillation term": "tab:red",
    }
    cmap = plt.get_cmap("viridis")
    # the stage names annotate the TOP panel only; the boundary lines repeat in every panel
    annotated = False

    if loss_parts:
        ax = axes.pop(0)
        for col, label in loss_parts:
            ax.plot(
                ep,
                _logsafe(history[col].to_list()),
                color=term_colors.get(label, "tab:orange"),
                lw=1.2,
                label=label,
            )
        ax.set_ylabel("Validation loss")
        _finish_panel(ax, boundaries, ep, manifest, labels=True)
        annotated = True

    for orders, ylabel in ((pu_orders, "pu-MAE (unobs.)"), (rel_orders, "rel. error")):
        if not orders:
            continue
        ax = axes.pop(0)
        for i, (order, col) in enumerate(orders):
            ax.plot(
                ep,
                _logsafe(history[col].to_list()),
                color=cmap(i / max(len(orders) - 1, 1)),
                lw=1.1,
                label=f"h{order}",
            )
        ax.set_ylabel(ylabel)
        _finish_panel(ax, boundaries, ep, manifest, labels=not annotated)
        annotated = True
    fig.axes[-1].set_xlabel("Epoch (cumulative across stages)")
    fig.tight_layout()

    if out_svg is not None or out_pdf is not None:
        from pgml.evaluation.style import save_figure

        if out_svg is not None:
            save_figure(fig, out_svg)
        if out_pdf is not None:
            save_figure(fig, out_pdf)
    return fig


def _finish_panel(ax, boundaries, epochs, manifest: dict, *, labels: bool) -> None:
    """Log axis, stage marks, grid and legend — the shared look of a decomposition panel."""
    ax.set_yscale("log")
    _draw_stage_marks(ax, boundaries, epochs, manifest, labels=labels)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="best", fontsize=8, ncol=2)


def _plotly_loss_curves(history, manifest, boundaries, switch_stage, out_html) -> None:
    """The interactive loss-curve figure (plotly HTML)."""
    import plotly.graph_objects as go

    ep = history["global_epoch"].to_list()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ep,
            y=history["train_loss"].to_list(),
            mode="lines+markers",
            name="train loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ep, y=history["val_loss"].to_list(), mode="lines+markers", name="val loss"
        )
    )
    if "val_pu_mae" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=ep,
                y=history["val_pu_mae"].to_list(),
                mode="lines+markers",
                name="val pu-MAE (all rows, monitor)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ep,
            y=history["val_pu_mae_unobs"].to_list(),
            mode="lines+markers",
            name="val pu-MAE (unobs)",
        )
    )
    for si, e in boundaries:
        is_switch = switch_stage is not None and si == int(switch_stage)
        fig.add_vline(
            x=e,
            line=dict(
                color="red" if is_switch else "gray",
                dash="solid" if is_switch else "dash",
                width=2 if is_switch else 1,
            ),
            annotation_text=_stage_label(manifest, si),
            annotation_position="top",
        )
    if switch_stage is not None:
        switch_ep = next((e for si, e in boundaries if si == int(switch_stage)), None)
        if switch_ep is not None and ep:
            fig.add_vrect(
                x0=switch_ep,
                x1=max(ep),
                fillcolor="lightgray",
                opacity=0.3,
                line_width=0,
                annotation_text=_pass_name("B"),
                annotation_position="top left",
            )
    fig.update_layout(
        title="Training history (measurement-scarcity curriculum)",
        xaxis_title="Global epoch",
        yaxis_title="Loss / pu-MAE",
        yaxis_type="log",
        width=950,
        height=560,
    )
    save_html(fig, out_html)


__all__ = [
    "HistoryCollector",
    "mlflow_history",
    "plot_loss_components",
    "plot_loss_curves",
    "read_history",
    "read_mlflow_metrics",
    "read_training_history",
    "stage_manifest",
]
