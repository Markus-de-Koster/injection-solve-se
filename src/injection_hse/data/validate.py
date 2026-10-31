"""Dataset validation: does a generated dataset contain what its configuration claims?

A state estimator is trained on whatever was generated, not on what the recipe describes.
Between the two sits a whole class of silent failures — an emission reference read in the
wrong units, a solved harmonic order nothing ever injects at, a coherent window anchored in
the dead of night — that no solver error and no convergence flag reports, because every
scenario converges perfectly to the wrong population. This module closes that gap: it reads
a written dataset back, measures the physical quantities the data is supposed to carry, and
compares them against three layers of expectation, most specific first:

1. **Self-declared** — the dataset's own ``meta.json`` configuration. A quantity the config
   declares varied must actually vary in the persisted samples, and its realized envelope
   must not collapse far below the declared support.
2. **Norms** — DIN EN 50160: the per-order harmonic voltage compatibility levels (above
   h25, which the standard does not tabulate, the IEC 61000-2-2 pattern, labelled as an
   extrapolation: :func:`harmonic_voltage_limits`), total harmonic distortion ``<= 8 %``,
   voltage unbalance ``<= 2 %``, fundamental deviation ``+/-10 %``.
3. **Measured populations** (optional) — a tracked expectations file
   (``run/configs/expected_ranges.yaml``, ``{quantity: {order: {p50, p95, hard_min,
   hard_max}}}``) seeded from device-certification and lab-rack numbers. A realized median
   off by more than 3x warns, more than 10x fails.

On top of those sit the hygiene checks that need no reference at all: non-finite states,
non-converged scenarios, and the **dead-order check** — a SOLVED harmonic order whose
``|V_h|`` is numerically zero everywhere is an order the run pays for and learns nothing
from, and it is named explicitly in the verdict.

The injected currents are audited from two sources, and they are NOT the same quantity:

- ``i_h_emission_pct`` — the REALIZED per-device emission ``|I_h| / |I_1|`` the generator
  injected, read straight from the persisted sample columns (``<key>_mag`` /
  ``<key>_phase``, written by every generating path of :mod:`pgml.scenarios`). Exact and
  free, and the definition measured device populations are tabulated in.
- ``i_h_ratio_pct`` — the NODAL NET injection at the device rows, RECONSTRUCTED from the
  state through the network law ``I(h) = Y(h)*V(h)``
  (:class:`injection_hse.physics.NetworkSolver`). It is what the network actually saw at a bus, so
  it also contains every other device at that node and the shunt current the network
  itself draws — a larger population than one device's emission.

``injection_source`` selects between them: ``"auto"`` (default) prefers the persisted
columns and reconstructs only for a dataset written without them, ``"both"`` measures both
(the only way to compare the two definitions on one dataset), and ``"persisted"`` /
``"reconstructed"`` pin one. Which source a report used is stated in its ``notes``.

The thresholds are calibrated on an LV distribution benchmark: they must catch a
decade-scale error (an emission reference read in the wrong units, a coherent window in the
dead band, a dead solved order) without firing on every legitimate difference between
grids. Three consequences follow. Only per-DEVICE emission fractions transfer between grids
as measured expectations; THD / VUF levels are properties of the network and of the solved
order set, so the expectations file states hard maxima for them only. The
snapshots-vs-sequences cross check of :func:`validate_run` can fire on a synthetic MV feeder,
where a composed household-scale device roster is diluted against MV-sized loads — a grid
class outside the calibration, not a generator regression; check the grid class before
touching a threshold. And the split-envelope comparison of :func:`validate_corpus` is only
meaningful on a corpus of realistic size; on a few-member smoke corpus its WARNs are noise.
The ratio thresholds are inclusive (``>=``), so a ratio of exactly
:data:`REFERENCE_FAIL_FACTOR` fails — percentile interpolation can land a nominal 10x
mis-scaling on either side of that boundary.

Three entry points, one report type:

- :func:`validate_dataset` — one dataset directory (the ``pgml.scenarios.write_dataset``
  layout);
- :func:`validate_corpus` — a multi-grid corpus root (every manifest member, plus the
  train/val/test split-envelope comparison);
- :func:`validate_run` — a workflow run holding ``dataset_random`` / ``dataset_coherent`` /
  ``dataset_profile``: each dataset, plus the per-order snapshots-vs-sequences median ratio that
  makes a systematically mis-scaled task visible immediately.

Each writes ``validation.json`` (statistics, verdicts and the validated configuration
fingerprint), ``validation_summary.csv`` and a ``validation`` figure into the directory it
validated. Everything here is analysis-side: plain numpy on detached tensors, never on the
autograd tape.

CLI::

    python -m injection_hse.data.validate <dataset|corpus|run dir> [--reference ranges.yaml] [--strict]
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_log = logging.getLogger(__name__)

#: Statistic columns of every reported series (the shape the multi-grid distribution
#: report uses, plus the standard deviation a "declared varied" check needs).
STAT_KEYS = ("count", "min", "p1", "p25", "median", "p75", "p99", "max", "std")

#: Written artifacts, in the directory that was validated.
REPORT_NAME = "validation.json"
SUMMARY_NAME = "validation_summary.csv"
FIGURE_STEM = "validation"

#: Phase code of a neutral row (:func:`pgml.assembly.node_phase_index` layout). Neutral
#: rows carry a near-zero voltage by design and would make every ratio meaningless.
NEUTRAL_CODE = 3

#: A row whose fundamental magnitude falls below this fraction of its base voltage is
#: de-energized and excluded from the ratio statistics (the convention of
#: :func:`injection_hse.analysis.scenario_features`).
ENERGIZED_FLOOR = 1e-6

#: A SOLVED harmonic order whose ``|V_h|`` p99 stays below this per-unit level carries no
#: information at all — nothing is injected there. Far below any physical distortion
#: (a measurable harmonic voltage is 1e-4 pu and up), so the check cannot mistake a
#: quiet order for a dead one.
DEAD_ORDER_PU = 1e-9
#: Ten times below the smallest distortion a real instrument resolves: not provably dead,
#: but not worth solving either.
NEAR_DEAD_ORDER_PU = 1e-6
#: Below this per-unit fundamental the grid is de-energized, not lightly loaded.
DE_ENERGIZED_PU = 0.1

#: Local hours in which the shipped activity presets are awake — the union of the
#: household morning ramp and evening peak, the office and industrial plateaus, the EV
#: arrival peak and the PV daylight bell (:mod:`injection_hse.data.scenarios.profiles`). A coherent
#: window lying entirely outside it is generated with every device roster dark.
LIVE_HOUR_BAND = (6.0, 23.0)

#: Ratio by which a realized median may miss a measured-population median before it warns,
#: and before it fails. An order-of-magnitude miss is the signature of a unit or reference
#: error, not of a differently-loaded feeder.
REFERENCE_WARN_FACTOR = 3.0
REFERENCE_FAIL_FACTOR = 10.0

#: The realized span of a declared-varied quantity, as a fraction of the span its
#: distribution declares, below which the dataset does not cover what it claims.
DECLARED_WARN_FRACTION = 0.1
DECLARED_FAIL_FRACTION = 0.01

#: An excursion smaller than this fraction of the train p1..p99 span still counts as
#: inside the training envelope (a split check should react to a systematic move, not to
#: bin noise) — the tolerance of the multi-grid distribution report.
ENVELOPE_TOLERANCE = 0.01

#: Scenarios read per dataset. The statistics are distributional, so a few hundred
#: scenarios resolve every quantile the report prints, while the reconstruction
#: ``I(h) = Y(h)*V(h)`` stays affordable on a corpus of many members.
DEFAULT_MAX_SAMPLES = 256

#: Realized per-device emission, read from the persisted ``<key>_mag`` sample columns.
PERSISTED_INJECTION_QUANTITY = "i_h_emission_pct"
#: Nodal net injection at the device rows, reconstructed as ``I(h) = Y(h)*V(h)``.
RECONSTRUCTED_INJECTION_QUANTITY = "i_h_ratio_pct"
#: Both injection quantities, the persisted (exact, per device) one first.
INJECTION_QUANTITIES = (
    PERSISTED_INJECTION_QUANTITY,
    RECONSTRUCTED_INJECTION_QUANTITY,
)

#: Where the injection audit takes its numbers from (see the module docstring).
InjectionSource = Literal["auto", "persisted", "reconstructed", "both"]

#: Sample-column suffixes that end in ``_mag`` but are NOT a realized injection: the
#: fingerprint mode bank holds the per-mode BASE magnitudes, before jitter and clamp.
_NOT_REALIZED_MAG = ("_mode_base_mag",)

Verdict = Literal["OK", "WARN", "FAIL"]
_SEVERITY = {"OK": 0, "WARN": 1, "FAIL": 2}


class DatasetValidationError(ValueError):
    """Raised by the validators when a report FAILS and ``strict=True``."""


# =============================================================================
# Report model
# =============================================================================
class SeriesStats(BaseModel):
    """Distribution of one measured quantity, optionally per harmonic order and phase."""

    model_config = ConfigDict(extra="forbid")

    quantity: str
    label: str
    unit: str = ""
    #: Harmonic order the series belongs to (``None`` for a per-sample quantity).
    order: Optional[int] = None
    #: Phase label ``"a"`` / ``"b"`` / ``"c"``, or ``"*"`` for a series pooled over rows.
    phase: str = "*"
    count: int = 0
    min: float = math.nan
    p1: float = math.nan
    p25: float = math.nan
    median: float = math.nan
    p75: float = math.nan
    p99: float = math.nan
    max: float = math.nan
    std: float = math.nan


class Check(BaseModel):
    """One verdict about one (quantity, order, phase), with the reason it was reached."""

    model_config = ConfigDict(extra="forbid")

    quantity: str
    verdict: Verdict = "OK"
    #: Which expectation produced it: ``hygiene`` / ``declared`` / ``norm`` /
    #: ``reference`` / ``cross`` (task-to-task consistency) / ``envelope`` (split shift).
    layer: str = "hygiene"
    order: Optional[int] = None
    phase: str = "*"
    reason: str = ""


class ValidationReport(BaseModel):
    """The result of validating one dataset, corpus or run — JSON-serializable.

    ``stats`` holds every measured series, ``checks`` every verdict, and ``members`` the
    per-dataset (or per-member) reports of a corpus / run. :attr:`verdict` is the worst
    verdict anywhere in the tree, so a caller gates on the top-level object alone.
    """

    model_config = ConfigDict(extra="forbid")

    #: ``dataset`` / ``corpus`` / ``run``.
    kind: str
    path: str
    #: Fingerprint of the generating configuration (``meta.json`` ``config_hash``), the
    #: identity a reuse gate must match; ``None`` for a dataset written before it existed.
    config_hash: Optional[str] = None
    config_type: Optional[str] = None
    schema_version: Optional[str] = None
    #: The recipe and device-library revisions the generator stamped
    #: (``injection_hse.data.scenarios.GENERATION_PROVENANCE``); ``None`` when it did not.
    se_preset_version: Optional[str] = None
    device_library_version: Optional[str] = None
    orders: tuple[int, ...] = ()
    n_scenarios: int = 0
    n_steps: int = 1
    n_failed: int = 0
    stats: list[SeriesStats] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    members: dict[str, "ValidationReport"] = Field(default_factory=dict)
    #: Expectations file the reference layer was read from (``None`` = layer not applied).
    reference: Optional[str] = None
    notes: list[str] = Field(default_factory=list)

    # -- aggregation ---------------------------------------------------------
    def all_checks(self) -> list[Check]:
        """Every check of this report AND of its members, members last."""
        out = list(self.checks)
        for member in self.members.values():
            out.extend(member.all_checks())
        return out

    @property
    def verdict(self) -> Verdict:
        """Worst verdict over this report and its members (``OK`` when there is none)."""
        worst = "OK"
        for check in self.all_checks():
            if _SEVERITY[check.verdict] > _SEVERITY[worst]:
                worst = check.verdict
        return worst  # type: ignore[return-value]

    @property
    def failed(self) -> bool:
        """Whether anything FAILED (the gate a driver aborts on)."""
        return self.verdict == "FAIL"

    @property
    def failures(self) -> list[Check]:
        """Every FAIL check, members included."""
        return [c for c in self.all_checks() if c.verdict == "FAIL"]

    @property
    def warnings(self) -> list[Check]:
        """Every WARN check, members included."""
        return [c for c in self.all_checks() if c.verdict == "WARN"]

    def series(self, quantity: str, order: Optional[int] = None, phase: str = "*"):
        """The :class:`SeriesStats` of one (quantity, order, phase), or ``None``."""
        for stat in self.stats:
            if (
                stat.quantity == quantity
                and stat.order == order
                and stat.phase == phase
            ):
                return stat
        return None

    def summary_line(self) -> str:
        """One-line verdict with the FAIL / WARN counts (what a driver prints)."""
        fails, warns = len(self.failures), len(self.warnings)
        return (
            f"{self.verdict}: {Path(self.path).name} ({self.kind}, "
            f"{self.n_scenarios} scenario(s)) — {fails} FAIL / {warns} WARN"
        )

    def table(self, *, limit: int = 20) -> str:
        """Compact printable verdict table: the failures first, then the warnings."""
        rows = self.failures + self.warnings
        if not rows:
            return f"{self.summary_line()}\n  all checks passed"
        lines = [self.summary_line()]
        for check in rows[:limit]:
            where = check.quantity
            if check.order is not None:
                where += f" h{check.order}"
            if check.phase != "*":
                where += f" [{check.phase}]"
            lines.append(f"  {check.verdict:<4} {where:<28} {check.reason}")
        if len(rows) > limit:
            lines.append(f"  ... and {len(rows) - limit} more")
        return "\n".join(lines)


ValidationReport.model_rebuild()


# =============================================================================
# Expected ranges (the measured-population layer)
# =============================================================================
@dataclass(frozen=True)
class ExpectedRange:
    """One measured-population expectation for a (quantity, order)."""

    p50: Optional[float] = None
    p95: Optional[float] = None
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None


#: ``{quantity: {order or None: ExpectedRange}}``.
ExpectedRanges = dict


def load_expected_ranges(path) -> ExpectedRanges:
    """Read an expectations YAML into ``{quantity: {order: ExpectedRange}}``.

    The file maps a quantity to its per-order entries, using the key ``all`` for a
    quantity that is not per-order::

        i_h_ratio_pct:
          3: {p50: 20.0, p95: 60.0, hard_max: 200.0}
        thd_pct:
          all: {p50: 3.2, p95: 5.8, hard_max: 8.0}

    Any further keys of an entry (``source``, ``note`` — the provenance of the number)
    are documentation and are ignored here. A missing bound simply skips its check, so a
    file may state only what it actually knows.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ranges: ExpectedRanges = {}
    for quantity, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        per_order: dict = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            order = None if str(key) == "all" else int(key)
            per_order[order] = ExpectedRange(
                p50=_opt_float(entry.get("p50")),
                p95=_opt_float(entry.get("p95")),
                hard_min=_opt_float(entry.get("hard_min")),
                hard_max=_opt_float(entry.get("hard_max")),
            )
        if per_order:
            ranges[str(quantity)] = per_order
    return ranges


def _opt_float(value) -> Optional[float]:
    return None if value is None else float(value)


# =============================================================================
# Measured series
# =============================================================================
@dataclass
class _Series:
    """One measured quantity before it is summarized (values are finite-filtered later)."""

    quantity: str
    label: str
    unit: str
    values: np.ndarray
    order: Optional[int] = None
    phase: str = "*"


def _summarize(series: _Series) -> Optional[SeriesStats]:
    """Quantile summary of one series (``None`` when it holds no finite value)."""
    values = np.asarray(series.values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    q = np.percentile(finite, [0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0])
    return SeriesStats(
        quantity=series.quantity,
        label=series.label,
        unit=series.unit,
        order=series.order,
        phase=series.phase,
        count=int(finite.size),
        min=float(q[0]),
        p1=float(q[1]),
        p25=float(q[2]),
        median=float(q[3]),
        p75=float(q[4]),
        p99=float(q[5]),
        max=float(q[6]),
        std=float(finite.std()),
    )


_PHASE_LABELS = ("a", "b", "c")


def _state_series(v, base_v, phase_codes, orders) -> list[_Series]:
    """Per-order voltage series: ``|V_h|`` in per unit and as a percentage of ``|V_1|``.

    ``v`` is complex ``[S, H, N]``, ``base_v`` float ``[N]``, ``phase_codes`` int ``[N]``.
    Neutral rows are excluded throughout (their near-zero voltage carries no ratio), and
    the percentage of the fundamental is formed only on energized rows — the definition
    DIN EN 50160 states its per-order limits in.
    """
    out: list[_Series] = []
    live = phase_codes != NEUTRAL_CODE
    if not live.any():
        return out
    magnitude = np.abs(v)  # [S, H, N]
    pu = magnitude / base_v[None, None, :]
    fundamental = orders.index(1) if 1 in orders else None
    energized = (
        pu[:, fundamental, :] > ENERGIZED_FLOOR if fundamental is not None else None
    )

    for position, order in enumerate(orders):
        for phase, label in [(None, "*"), *enumerate(_PHASE_LABELS)]:
            rows = live if phase is None else (live & (phase_codes == phase))
            if not rows.any():
                continue
            block = pu[:, position, rows]
            out.append(
                _Series(
                    "v_pu",
                    "|V| / V_base",
                    "pu",
                    block,
                    order=order,
                    phase=label,
                )
            )
        if fundamental is None or order == 1:
            continue
        ok = energized & live[None, :]
        ratio = np.where(
            ok,
            100.0
            * magnitude[:, position, :]
            / np.where(ok, magnitude[:, fundamental, :], 1.0),
            np.nan,
        )
        out.append(
            _Series(
                "v_h_ratio_pct",
                "|V_h| / |V_1|",
                "%",
                ratio,
                order=order,
            )
        )
    if fundamental is not None:
        worst = np.where(live[None, :], pu[:, fundamental, :], np.nan)
        out.append(
            _Series(
                "v_pu_min", "worst-row |V_1| / V_base", "pu", np.nanmin(worst, axis=1)
            )
        )
    return out


def _coverage_series(v, grid, orders, *, device: str = "cpu") -> list[_Series]:
    """Total harmonic distortion, unbalance and fundamental deviation, per sample.

    Delegates to :func:`injection_hse.analysis.scenario_features` so these three carry exactly the
    definitions the coverage report and the error-binning analysis use (max over the
    energized non-neutral rows of each sample).
    """
    from pgml.assembly import base_voltage_per_row, node_phase_index

    from injection_hse.analysis import scenario_features
    from injection_hse.evaluation import Prediction

    index = node_phase_index(grid)
    prediction = Prediction(
        target_v=v,
        pred_v=v,
        mask=np.ones((v.shape[0], v.shape[-1]), dtype=bool),
        orders=list(orders),
        base_v=base_voltage_per_row(grid).numpy().astype(np.float64),
        node_ids=index.node_ids.numpy().astype(np.int64),
        phase_codes=index.phase_codes.numpy().astype(np.int64),
        observed_rows=np.zeros(0, dtype=np.int64),
        sensor_names=[],
        grid=grid,
    )
    features = scenario_features(prediction, device=device)
    spec = (
        ("thd_max_pct", "thd_pct", "max voltage THD", "%"),
        ("vuf_max_pct", "vuf_pct", "max voltage unbalance", "%"),
        ("vdev_max_pct", "vdev_pct", "max |V_1| deviation", "%"),
    )
    return [
        _Series(quantity, label, unit, features[column].to_numpy())
        for column, quantity, label, unit in spec
        if column in features.columns
    ]


def _device_rows(grid) -> "np.ndarray":
    """Node-phase rows of every in-service injecting appliance, neutral rows dropped.

    These are the rows a device current is injected at, so the reconstructed
    ``I(h) = Y(h)*V(h)`` there is the population a device-emission expectation applies to.
    """
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import InjectionAppliance

    index = node_phase_index(grid)
    rows: set[int] = set()
    for appliance in grid.appliances:
        if not isinstance(appliance, InjectionAppliance) or not appliance.in_service:
            continue
        rows.update(
            int(r) for r in index.rows_for_terminal(appliance.node, appliance.phases)
        )
    codes = index.phase_codes.numpy()
    return np.asarray(
        sorted(r for r in rows if codes[r] != NEUTRAL_CODE), dtype=np.int64
    )


def _block_orders(config, key: str) -> Optional[list[int]]:
    """Harmonic orders on the order axis of the realized-injection block ``key``.

    The block's own columns carry no order labels — they live in the generating
    configuration: a randomized block is named after the ``h_mag`` parameter spec that
    wrote it (its ``orders``), a coherent one after the config itself (``<name>`` for the
    fingerprint, ``<name>_composed`` for the device composition, both over
    ``config.orders``). ``None`` when the block cannot be attributed, which is the signal
    to skip it rather than guess an order mapping.
    """
    for spec in getattr(config, "parameters", ()) or ():
        if getattr(spec, "name", None) == key:
            orders = getattr(spec, "orders", None)
            return [int(o) for o in orders] if orders else None
    name = getattr(config, "name", None)
    if name is not None and key in (str(name), f"{name}_composed"):
        orders = getattr(config, "orders", None)
        return [int(o) for o in orders] if orders else None
    return None


def _injection_blocks(loaded) -> list[tuple[str, "np.ndarray", list[int]]]:
    """``(key, magnitude[S, n_dev, n_ord], orders)`` per persisted realized-injection block.

    A block is the ``<key>_mag`` / ``<key>_phase`` pair the generating paths of
    :mod:`pgml.scenarios` write for every device whose spectrum they realize (the
    randomized ``h_mag`` specs, the coherent fingerprint, and the device composition).
    A sequence block ``[B, n_dev, n_ord, T]`` is folded to ``[B*T, n_dev, n_ord]``, so
    every ``(scenario, step)`` counts once — the sample axis of the state statistics.
    Blocks that carry no order attribution are skipped (see :func:`_block_orders`).
    """
    samples = getattr(loaded, "samples", None) or {}
    blocks: list[tuple[str, np.ndarray, list[int]]] = []
    for name in sorted(samples):
        if not name.endswith("_mag") or name.endswith(_NOT_REALIZED_MAG):
            continue
        key = name[: -len("_mag")]
        if f"{key}_phase" not in samples:
            continue
        magnitude = np.asarray(samples[name].detach().cpu().numpy(), dtype=np.float64)
        if magnitude.ndim == 4:  # [B, n_dev, n_ord, T] -> [B*T, n_dev, n_ord]
            magnitude = np.moveaxis(magnitude, 3, 1).reshape(
                -1, magnitude.shape[1], magnitude.shape[2]
            )
        if magnitude.ndim != 3:
            continue
        orders = _block_orders(loaded.config, key)
        if orders is None or len(orders) != magnitude.shape[2]:
            continue
        blocks.append((key, magnitude, orders))
    return blocks


def _persisted_injection_series(loaded, *, max_samples: int) -> list[_Series]:
    """Realized per-device emission ``|I_h| / |I_1|`` from the persisted sample columns.

    The generator records what it injected per device and order (in per unit of that
    device's own fundamental current), so this is the exact emission population — no
    reconstruction, no network model, and unaffected by how many devices share a node.
    Blocks are pooled per order over devices and scenarios.

    An exactly-zero magnitude is dropped: it means the device injected NOTHING there (an
    inactive composed aggregate, or an order this block does not excite), which is a
    statement about the roster rather than about what an emitting device emits. An order
    that is silent everywhere therefore yields no series at all, and the dead-order
    hygiene check on ``|V_h|`` is what reports it.
    """
    per_order: dict[int, list[np.ndarray]] = {}
    for _key, magnitude, orders in _injection_blocks(loaded):
        if max_samples and magnitude.shape[0] > max_samples:
            stride = int(math.ceil(magnitude.shape[0] / max_samples))
            magnitude = magnitude[::stride][:max_samples]
        for position, order in enumerate(orders):
            per_order.setdefault(int(order), []).append(
                magnitude[:, :, position].reshape(-1)
            )
    out: list[_Series] = []
    for order in sorted(per_order):
        values = np.concatenate(per_order[order])
        emitting = values[values > 0.0]
        if emitting.size == 0:
            continue
        out.append(
            _Series(
                PERSISTED_INJECTION_QUANTITY,
                "|I_h| / |I_1| realized per device",
                "%",
                100.0 * emitting,
                order=order,
            )
        )
    return out


def _injection_note(persisted: bool, reconstructed: bool, source: str) -> str:
    """One line stating which injection quantity a report carries, and why.

    The two are not interchangeable — a report that swapped them without saying so would
    silently move the reference thresholds from a node aggregate to a single device — so
    the source is always spelled out next to the statistics it produced.
    """
    both = (
        f"{PERSISTED_INJECTION_QUANTITY} = realized per-device emission from the "
        f"persisted sample columns; {RECONSTRUCTED_INJECTION_QUANTITY} = nodal net "
        "injection reconstructed as I(h) = Y(h)*V(h) at the device rows"
    )
    if persisted and reconstructed:
        return f"injection audit (source={source}): both quantities measured — {both}."
    if persisted:
        return (
            f"injection audit (source={source}): {PERSISTED_INJECTION_QUANTITY} = "
            "realized per-device emission read from the persisted sample columns; the "
            f"reconstruction {RECONSTRUCTED_INJECTION_QUANTITY} (nodal net injection, "
            "a different quantity) was not computed — use injection_source='both'."
        )
    if reconstructed:
        return (
            f"injection audit (source={source}): no realized-injection columns in this "
            f"dataset, so {RECONSTRUCTED_INJECTION_QUANTITY} = nodal net injection "
            "reconstructed as I(h) = Y(h)*V(h) at the device rows."
        )
    return (
        f"injection audit (source={source}): not measured — this dataset persists no "
        "realized-injection columns."
    )


def _injection_series(v, grid, orders, *, chunk_size: int = 64) -> list[_Series]:
    """Reconstructed nodal injection currents at the device rows, as ``|I_h| / |I_1|``.

    Recovered from the state through the network law: :class:`injection_hse.physics.NetworkSolver`
    holds exactly the per-order operator the data was solved with, and
    ``injection_targets`` applies it. The result is reported as a percentage of the row's
    own fundamental current, over the rows whose fundamental current is above a numerical
    floor. Unlike the persisted per-device emission
    (:func:`_persisted_injection_series`), this is the NET injection of a node-phase row:
    every device at that node plus the network's own shunt current.
    """
    import torch

    from injection_hse.physics import NetworkSolver

    rows = _device_rows(grid)
    if rows.size == 0 or 1 not in orders:
        return []
    solver = NetworkSolver.from_grid(
        grid, list(orders), dtype=torch.complex128, device=torch.device("cpu")
    )
    state = torch.as_tensor(v, dtype=torch.complex128)
    blocks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, state.shape[0], chunk_size):
            currents = solver.injection_targets(state[start : start + chunk_size])
            blocks.append(currents[..., rows].abs().numpy())
    magnitude = np.concatenate(blocks, axis=0)  # [S, H, D]
    fundamental = list(orders).index(1)
    i1 = magnitude[:, fundamental, :]
    floor = ENERGIZED_FLOOR * float(np.percentile(i1[np.isfinite(i1)], 99.0) or 0.0)
    carrying = i1 > max(floor, 0.0)
    out: list[_Series] = [
        _Series("i1_a", "|I_1| at device rows", "A", np.where(carrying, i1, np.nan))
    ]
    for position, order in enumerate(orders):
        if order == 1:
            continue
        ratio = np.where(
            carrying,
            100.0 * magnitude[:, position, :] / np.where(carrying, i1, 1.0),
            np.nan,
        )
        out.append(
            _Series(
                "i_h_ratio_pct", "|I_h| / |I_1| at device rows", "%", ratio, order=order
            )
        )
    return out


def _power_series(v, grid, orders) -> list[_Series]:
    """Realized fundamental infeed ``P`` / ``Q`` per sample, from the solved state.

    The apparent power flowing in at the slack rows, ``sum(V * conj(Y*V))`` over them at
    the fundamental — the aggregate operating point a sequence is supposed to move, and
    the quantity a near-zero activity band shows up in.
    """
    import torch

    from injection_hse.physics import NetworkSolver

    if 1 not in orders:
        return []
    solver = NetworkSolver.from_grid(
        grid, [1], dtype=torch.complex128, device=torch.device("cpu")
    )
    slack = solver.slack_rows.numpy()
    if slack.size == 0:
        return []
    position = list(orders).index(1)
    state = torch.as_tensor(v[:, position : position + 1, :], dtype=torch.complex128)
    with torch.no_grad():
        current = solver.injection_targets(state)[:, 0, :]
    power = (state[:, 0, :] * current.conj()).numpy()[:, slack].sum(axis=1)
    return [
        _Series("p_slack_w", "fundamental infeed P", "W", power.real),
        _Series("q_slack_var", "fundamental infeed Q", "var", power.imag),
    ]


def _time_axis(loaded, n_steps: int):
    """Absolute per-step epoch seconds ``[T]`` of a sequence dataset, or ``None``.

    A profiled dataset persists ``time_unix_s``; a composed one derives it from its
    config (``start_time`` + ``step_size_s``), exactly as the loader does. A fingerprint
    sequence is timeless and yields ``None``.
    """
    from .sources import _derive_time_unix_s

    stamped = loaded.samples.get("time_unix_s")
    if stamped is not None:
        return np.asarray(stamped, dtype=np.float64).reshape(-1)
    derived = _derive_time_unix_s(loaded.config, n_steps)
    return None if derived is None else np.asarray(derived, dtype=np.float64)


def _sequence_series(loaded, n_steps: int, power) -> list[_Series]:
    """Sequence-only series: the local hour of each step, and the idle-step fraction.

    ``power`` is the per-sample fundamental infeed (:func:`_power_series`) or ``None``;
    its idle fraction — steps drawing under 1 % of the window's peak — is the quantity
    that says whether a "coherent sequence" ever moves the operating point it exists to
    move.
    """
    if n_steps <= 1:
        return []
    out: list[_Series] = []
    axis = _time_axis(loaded, n_steps)
    if axis is not None and axis.size:
        out.append(
            _Series("hour_of_day", "step local hour", "h", (axis % 86400.0) / 3600.0)
        )
    idle = _idle_fraction(power)
    if idle is not None:
        out.append(
            _Series(
                "idle_step_fraction",
                "steps under 1 % of the peak infeed",
                "",
                np.array([idle]),
            )
        )
    return out


def _idle_fraction(power) -> Optional[float]:
    """Fraction of steps whose infeed is under 1 % of the window's peak (``None`` if n/a)."""
    if power is None:
        return None
    finite = np.abs(np.asarray(power, dtype=np.float64).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    peak = float(np.percentile(finite, 99.0))
    return None if peak <= 0.0 else float(np.mean(finite < 0.01 * peak))


# =============================================================================
# Checks
# =============================================================================
def _check(quantity, verdict, layer, reason, *, order=None, phase="*") -> Check:
    return Check(
        quantity=quantity,
        verdict=verdict,
        layer=layer,
        order=order,
        phase=phase,
        reason=reason,
    )


def _hygiene_checks(
    stats: dict, *, orders, meta: dict, n_scenarios: int, nonfinite: int, injected
) -> list[Check]:
    """Verdicts that need no reference: finiteness, convergence, and dead orders."""
    checks: list[Check] = []
    if nonfinite:
        checks.append(
            _check(
                "state",
                "FAIL",
                "hygiene",
                f"{nonfinite} non-finite value(s) in the stored voltages",
            )
        )
    failed = list(meta.get("failed_scenarios", ()))
    if failed and n_scenarios:
        fraction = len(failed) / n_scenarios
        checks.append(
            _check(
                "converged",
                "FAIL" if fraction > 0.1 else "WARN",
                "hygiene",
                f"{len(failed)}/{n_scenarios} scenario(s) did not converge "
                f"({fraction:.1%}); their voltages are best-effort iterates",
            )
        )

    for order in orders:
        for phase in ("*", *_PHASE_LABELS):
            entry = stats.get(("v_pu", order, phase))
            if entry is None:
                continue
            if order == 1:
                if entry.p99 < DE_ENERGIZED_PU:
                    checks.append(
                        _check(
                            "v_pu",
                            "FAIL",
                            "hygiene",
                            f"the fundamental is de-energized (p99 {entry.p99:.3g} pu)",
                            order=order,
                            phase=phase,
                        )
                    )
                continue
            if entry.p99 < DEAD_ORDER_PU:
                detail = (
                    f"the config injects orders {sorted(injected)}"
                    if injected
                    else "the config injects no harmonic order at all"
                )
                checks.append(
                    _check(
                        "v_pu",
                        "FAIL",
                        "hygiene",
                        f"DEAD ORDER h{order}: solved but |V_h| p99 is {entry.p99:.3g} pu "
                        f"— nothing is injected there ({detail})",
                        order=order,
                        phase=phase,
                    )
                )
            elif entry.p99 < NEAR_DEAD_ORDER_PU:
                checks.append(
                    _check(
                        "v_pu",
                        "WARN",
                        "hygiene",
                        f"order h{order} is nearly dead (|V_h| p99 {entry.p99:.3g} pu, "
                        "below any measurable distortion)",
                        order=order,
                        phase=phase,
                    )
                )
    return checks


def _sequence_checks(raw: dict, *, n_steps: int) -> list[Check]:
    """The coherent-window checks: is the sequence generated while anybody is awake?

    Two failure signatures of one bug. The time axis may sit outside the hours in which
    the activity presets are awake at all (:data:`LIVE_HOUR_BAND`), and the realized
    aggregate may sit at its floor for most of the window — both leave a "coherent
    sequence" that never moves the operating point it exists to move.
    """
    if n_steps <= 1:
        return []
    checks: list[Check] = []
    hours = raw.get("hour_of_day")
    if hours is not None and hours.size:
        low, high = LIVE_HOUR_BAND
        inside = float(np.mean((hours >= low) & (hours <= high)))
        window = f"{hours.min():.1f}-{hours.max():.1f} h"
        if inside <= 0.0:
            checks.append(
                _check(
                    "hour_of_day",
                    "FAIL",
                    "declared",
                    f"the whole {window} window lies outside the {low:.0f}:00-{high:.0f}:00 "
                    "activity band — every device roster is dark and the composed "
                    "fundamental never leaves its floor",
                )
            )
        elif inside < 0.5:
            checks.append(
                _check(
                    "hour_of_day",
                    "WARN",
                    "declared",
                    f"only {inside:.0%} of the {window} window lies in the "
                    f"{low:.0f}:00-{high:.0f}:00 activity band",
                )
            )

    idle = _idle_fraction(raw.get("p_slack_w"))
    if idle is not None and idle > 0.5:
        checks.append(
            _check(
                "idle_step_fraction",
                "FAIL" if idle > 0.9 else "WARN",
                "declared",
                f"{idle:.0%} of the steps draw under 1 % of the window's peak infeed"
                + (" — the aggregate never leaves its floor" if idle > 0.9 else ""),
            )
        )
    return checks


def _declared_checks(loaded, *, n_scenarios: int) -> list[Check]:
    """Does the dataset contain the variation its own configuration declares?

    Every parameter spec whose distribution has width must show that width in the
    persisted samples: a collapsed realized envelope means the draw never reached the
    solve (a selector matching nothing, a superseded field, a frozen seed).
    """
    config = loaded.config
    specs = getattr(config, "parameters", None)
    if not specs or n_scenarios < 2:
        return []
    checks: list[Check] = []
    for spec in specs:
        declared = _declared_span(spec.distribution)
        if declared is None or declared <= 0.0:
            continue
        values = loaded.samples.get(spec.name)
        if values is None:
            checks.append(
                _check(
                    spec.name,
                    "WARN",
                    "declared",
                    "declared varied but no realized sample column was persisted",
                )
            )
            continue
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size == 0:
            continue
        realized = float(np.percentile(array, 99.0) - np.percentile(array, 1.0))
        fraction = realized / declared
        if fraction < DECLARED_FAIL_FRACTION:
            checks.append(
                _check(
                    spec.name,
                    "FAIL",
                    "declared",
                    f"declared varied over a span of {declared:.3g} but the realized "
                    f"p1..p99 span is {realized:.3g} ({fraction:.1%} of it)",
                )
            )
        elif fraction < DECLARED_WARN_FRACTION:
            checks.append(
                _check(
                    spec.name,
                    "WARN",
                    "declared",
                    f"realized p1..p99 span {realized:.3g} is {fraction:.0%} of the "
                    f"declared {declared:.3g}",
                )
            )
    return checks


def _declared_span(distribution) -> Optional[float]:
    """Width of a distribution's nominal support (``None`` when it declares no width)."""
    kind = getattr(distribution, "kind", None)
    if kind == "uniform":
        return float(distribution.high - distribution.low)
    if kind == "loguniform":
        return float(distribution.high - distribution.low)
    if kind == "normal":
        return float(4.0 * distribution.scale)  # +/- 2 sigma
    if kind == "lognormal":
        return float(
            math.exp(distribution.loc + 2.0 * distribution.scale)
            - math.exp(distribution.loc - 2.0 * distribution.scale)
        )
    return None


#: Highest order DIN EN 50160 tabulates a harmonic voltage level for (Table 1).
EN50160_MAX_ORDER = 25

#: IEC 61000-2-2 compatibility levels for harmonic voltages in public LV networks [% of
#: the fundamental] at the orders the standard states explicitly; every other order
#: follows its formulas (:func:`iec61000_2_2_level_pct`). THD 8 %.
_IEC_61000_2_2_TABLE = {
    2: 2.0,
    3: 5.0,
    4: 1.0,
    5: 6.0,
    6: 0.5,
    7: 5.0,
    8: 0.5,
    9: 1.5,
    11: 3.5,
    13: 3.0,
    15: 0.4,
    21: 0.3,
}


def iec61000_2_2_level_pct(order: int) -> float:
    """IEC 61000-2-2 LV compatibility level of one harmonic order [% of the fundamental].

    The explicit table entries (h2-h9, h11, h13, h15, h21), else the standard's formulas:
    odd non-triplen ``2.27 * 17 / h - 0.27``, triplen above h21 ``0.2``, even
    ``0.25 * 10 / h + 0.25``.
    """
    h = int(order)
    if h in _IEC_61000_2_2_TABLE:
        return _IEC_61000_2_2_TABLE[h]
    if h % 2 == 0:
        return 0.25 * 10.0 / h + 0.25
    if h % 3 == 0:
        return 0.2
    return 2.27 * 17.0 / h - 0.27


@dataclass(frozen=True)
class HarmonicVoltageLimit:
    """The harmonic voltage levels one order is judged against [% of the fundamental].

    ``en50160_pct`` is the DIN EN 50160 level (amended table) up to
    :data:`EN50160_MAX_ORDER`; above it the standard states none and the value is the
    IEC 61000-2-2 level of the order, flagged by ``extrapolated``. ``iec61000_2_2_pct``
    is the IEC 61000-2-2 compatibility level.
    """

    order: int
    en50160_pct: float
    iec61000_2_2_pct: float
    extrapolated: bool

    @property
    def label(self) -> str:
        """``"EN 50160"``, or the extrapolation it stands for above h25."""
        return (
            "EN 50160 (extrapolated, IEC 61000-2-2 pattern)"
            if self.extrapolated
            else "EN 50160"
        )


def harmonic_voltage_limits(orders: Sequence[int]) -> dict[int, HarmonicVoltageLimit]:
    """Per-order harmonic voltage levels ``{order: HarmonicVoltageLimit}`` (orders >= 2).

    EN 50160 as tabulated up to h25 (:func:`pgml.scenarios.en50160_limits`), above h25 the
    IEC 61000-2-2 formulas (odd non-triplen ``2.27 * 17 / h - 0.27``, triplen ``0.2``,
    even ``0.25 * 10 / h + 0.25``) labelled ``extrapolated``. A harmonic analyser
    reports every integer order to h50 (IEC 61000-4-7), so every order has a level.
    Order 1 is skipped.
    """
    from pgml.scenarios import en50160_limits

    table = en50160_limits()
    out: dict[int, HarmonicVoltageLimit] = {}
    for order in orders:
        h = int(order)
        if h < 2:
            continue
        iec = iec61000_2_2_level_pct(h)
        tabulated = h <= EN50160_MAX_ORDER and h in table
        out[h] = HarmonicVoltageLimit(
            order=h,
            en50160_pct=100.0 * float(table[h]) if tabulated else iec,
            iec61000_2_2_pct=iec,
            extrapolated=not tabulated,
        )
    return out


def _norm_checks(stats: dict, *, orders) -> list[Check]:
    """DIN EN 50160: per-order voltage levels, THD, unbalance, fundamental deviation.

    A dataset is supposed to COVER the compatibility limits (the recipe deliberately
    reaches the measured envelope), so exceeding one at the p99 warns, while a MEDIAN
    beyond it fails — a feeder whose typical state violates the norm is not an LV feeder.
    Above h25, where EN 50160 states no level, the order is judged against the
    IEC 61000-2-2 pattern (:func:`harmonic_voltage_limits`) and the check says so.
    """
    checks: list[Check] = []
    aggregate = (
        ("thd_pct", 8.0, "total harmonic distortion"),
        ("vuf_pct", 2.0, "voltage unbalance"),
        ("vdev_pct", 10.0, "fundamental voltage deviation"),
    )
    for quantity, limit, label in aggregate:
        entry = stats.get((quantity, None, "*"))
        if entry is None:
            continue
        if entry.median > limit:
            checks.append(
                _check(
                    quantity,
                    "FAIL",
                    "norm",
                    f"median {label} {entry.median:.3g} % exceeds the EN 50160 "
                    f"{limit:g} % limit",
                )
            )
        elif entry.p99 > limit:
            checks.append(
                _check(
                    quantity,
                    "WARN",
                    "norm",
                    f"p99 {label} {entry.p99:.3g} % exceeds the EN 50160 {limit:g} % "
                    "limit (the recipe covers the envelope; check it is intended)",
                )
            )

    limits = harmonic_voltage_limits(orders)
    for order in orders:
        entry = stats.get(("v_h_ratio_pct", order, "*"))
        limit = limits.get(int(order))
        if entry is None or limit is None:
            continue
        limit_pct = limit.en50160_pct
        if entry.median > limit_pct:
            checks.append(
                _check(
                    "v_h_ratio_pct",
                    "FAIL",
                    "norm",
                    f"median {entry.median:.3g} % exceeds the {limit.label} h{order} "
                    f"compatibility level {limit_pct:.3g} %",
                    order=order,
                )
            )
        elif entry.p99 > limit_pct:
            checks.append(
                _check(
                    "v_h_ratio_pct",
                    "WARN",
                    "norm",
                    f"p99 {entry.p99:.3g} % exceeds the {limit.label} h{order} "
                    f"compatibility level {limit_pct:.3g} %",
                    order=order,
                )
            )
    return checks


def _reference_checks(stats: dict, ranges: ExpectedRanges) -> list[Check]:
    """Measured-population layer: is the realized median the right ORDER OF MAGNITUDE?

    The check that catches a mis-referenced emission model: a per-order median off by
    more than :data:`REFERENCE_WARN_FACTOR` warns and by more than
    :data:`REFERENCE_FAIL_FACTOR` fails, in either direction. Hard bounds, where given,
    fail on containment of the median.
    """
    checks: list[Check] = []
    for quantity, per_order in ranges.items():
        for order, expected in per_order.items():
            entry = stats.get((quantity, order, "*"))
            if entry is None or not math.isfinite(entry.median):
                continue
            label = quantity if order is None else f"{quantity} h{order}"
            if expected.p50 is not None and expected.p50 > 0.0:
                ratio = entry.median / expected.p50
                factor = max(ratio, 1.0 / ratio) if ratio > 0.0 else math.inf
                if factor >= REFERENCE_FAIL_FACTOR:
                    checks.append(
                        _check(
                            quantity,
                            "FAIL",
                            "reference",
                            f"median {entry.median:.3g} is {factor:.1f}x off the measured "
                            f"{label} median {expected.p50:.3g}",
                            order=order,
                        )
                    )
                    continue
                if factor >= REFERENCE_WARN_FACTOR:
                    checks.append(
                        _check(
                            quantity,
                            "WARN",
                            "reference",
                            f"median {entry.median:.3g} is {factor:.1f}x off the measured "
                            f"{label} median {expected.p50:.3g}",
                            order=order,
                        )
                    )
                    continue
            if expected.hard_min is not None and entry.median < expected.hard_min:
                checks.append(
                    _check(
                        quantity,
                        "FAIL",
                        "reference",
                        f"median {entry.median:.3g} is below the {label} hard minimum "
                        f"{expected.hard_min:.3g}",
                        order=order,
                    )
                )
            elif expected.hard_max is not None and entry.median > expected.hard_max:
                checks.append(
                    _check(
                        quantity,
                        "FAIL",
                        "reference",
                        f"median {entry.median:.3g} exceeds the {label} hard maximum "
                        f"{expected.hard_max:.3g}",
                        order=order,
                    )
                )
    return checks


def _cross_dataset_checks(
    base: ValidationReport, other: ValidationReport, *, base_name: str, other_name: str
) -> list[Check]:
    """Per-order median ratio between two datasets of one run — the drift detector.

    The two data-generating processes cover different envelopes by design, but not
    different ORDERS OF MAGNITUDE: a per-order harmonic median that differs by 10x means
    one of the two recipes is mis-scaled, which is exactly how an emission bug survives
    for months in the task nobody plots.
    """
    checks: list[Check] = []
    for quantity in ("v_h_ratio_pct", *INJECTION_QUANTITIES):
        for entry in base.stats:
            if entry.quantity != quantity or entry.phase != "*":
                continue
            counterpart = other.series(quantity, entry.order, "*")
            if counterpart is None:
                continue
            if not (
                math.isfinite(entry.median)
                and math.isfinite(counterpart.median)
                and entry.median > 0.0
                and counterpart.median > 0.0
            ):
                continue
            ratio = counterpart.median / entry.median
            factor = max(ratio, 1.0 / ratio)
            if factor < REFERENCE_WARN_FACTOR:
                continue
            checks.append(
                _check(
                    quantity,
                    "FAIL" if factor >= REFERENCE_FAIL_FACTOR else "WARN",
                    "cross",
                    f"{other_name} median {counterpart.median:.3g} is {factor:.1f}x "
                    f"{'above' if ratio > 1 else 'below'} {base_name} "
                    f"({entry.median:.3g}) at h{entry.order}",
                    order=entry.order,
                )
            )
    return checks


def _envelope_checks(
    train: dict, others: dict[str, dict], *, quantities: Sequence[tuple]
) -> list[Check]:
    """Do the val / test splits of a corpus lie inside the train p1..p99 envelope?

    The p1/p99 range rather than min/max: one outlier member should not decide whether a
    corpus is covered, while a systematic shift moves the percentiles. A held-out split
    that leaves the envelope is a stated distribution shift, not a broken dataset, so it
    WARNS.
    """
    checks: list[Check] = []
    for key in quantities:
        reference = train.get(key)
        if reference is None:
            continue
        span = max(abs(reference.p99 - reference.p1), 1e-12)
        quantity, order, phase = key
        for split, stats in others.items():
            entry = stats.get(key)
            if entry is None:
                continue
            below = (reference.p1 - entry.p1) / span
            above = (entry.p99 - reference.p99) / span
            if below <= ENVELOPE_TOLERANCE and above <= ENVELOPE_TOLERANCE:
                continue
            detail = (
                f"p1 {entry.p1:.4g} < train p1 {reference.p1:.4g} "
                f"(-{below:.0%} of the train span)"
                if below > above
                else f"p99 {entry.p99:.4g} > train p99 {reference.p99:.4g} "
                f"(+{above:.0%} of the train span)"
            )
            checks.append(
                _check(
                    quantity,
                    "WARN",
                    "envelope",
                    f"{split} split leaves the train envelope: {detail}",
                    order=order,
                    phase=phase,
                )
            )
    return checks


# =============================================================================
# Dataset validation
# =============================================================================
def _resolve_grid(root: Path, grid):
    """The grid a dataset was generated on: explicit, or ``grid.json`` at/above it.

    A dataset directory stores only the node-phase row layout, so the grid comes from
    beside it — the convention every generator here writes: a member corpus grid sits in
    the member directory, a workflow grid one level above its ``dataset_*`` directories.
    """
    if grid is not None:
        if isinstance(grid, (str, Path)):
            from pgml.schemas.grid_schema import Grid

            return Grid.model_validate_json(Path(grid).read_text(encoding="utf-8"))
        return grid
    from pgml.schemas.grid_schema import Grid

    for candidate in (root, root.parent, root.parent.parent):
        path = candidate / "grid.json"
        if path.is_file():
            return Grid.model_validate_json(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"no grid.json found at or above {root} — pass grid= explicitly (a dataset "
        "stores only its node-phase row layout, not the grid)."
    )


def _flatten_state(v, *, max_samples: int):
    """``[B, H, N]`` / ``[B, T, H, N]`` -> ``[S, H, N]``, subsampled to ``max_samples``.

    A sequence's ``T`` axis folds into the sample axis: every ``(scenario, step)`` is one
    state the model sees. The subsample is a deterministic stride, so a report is
    reproducible and covers the whole batch rather than its first rows.
    """
    array = np.asarray(v)
    if array.ndim == 4:
        array = array.reshape(-1, array.shape[-2], array.shape[-1])
    elif array.ndim == 3:
        pass
    elif array.ndim == 2:
        array = array.reshape(array.shape[0], 1, array.shape[1])
    else:
        raise ValueError(
            f"expected a [B, H, N] or [B, T, H, N] state, got {array.shape}"
        )
    if max_samples and array.shape[0] > max_samples:
        stride = int(math.ceil(array.shape[0] / max_samples))
        array = array[::stride][:max_samples]
    return array


def _injected_orders(config) -> list[int]:
    """The harmonic orders a configuration actually injects at (its own claim)."""
    orders: set[int] = set()
    for spec in getattr(config, "parameters", ()) or ():
        for order in getattr(spec, "orders", None) or ():
            if int(order) > 1:
                orders.add(int(order))
    for order in getattr(config, "orders", None) or ():
        if int(order) > 1:
            orders.add(int(order))
    return sorted(orders)


def _generation_stamp(meta: dict, key: str) -> Optional[str]:
    """A generator's version stamp from ``meta.json``.

    ``write_dataset(provenance=...)`` stores the caller's stamps under
    ``extra_provenance``; a dataset written before that key existed carried
    ``device_library_version`` at the top level. Either location counts.
    """
    extra = meta.get("extra_provenance") or {}
    value = extra.get(key, meta.get(key))
    return None if value is None else str(value)


def validate_dataset(
    root,
    *,
    grid=None,
    reference=None,
    strict: bool = False,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    write: bool = True,
    device: str = "cpu",
    injection_source: InjectionSource = "auto",
) -> ValidationReport:
    """Validate one written dataset directory; returns (and by default writes) its report.

    Parameters
    ----------
    root:
        A dataset directory in the :func:`pgml.scenarios.write_dataset` layout
        (``meta.json`` + ``voltages.parquet``), snapshots or coherent sequences.
    grid:
        The :class:`~pgml.schemas.grid_schema.Grid` the dataset was generated on, or a
        path to one. ``None`` resolves ``grid.json`` at, above, or two levels above the
        dataset — where every generator in this package writes it.
    reference:
        Optional path to an expectations YAML (see :func:`load_expected_ranges`); adds
        the measured-population layer on top of the norms.
    strict:
        Raise :class:`DatasetValidationError` when the report FAILS (the artifacts are
        still written first, so the failure is inspectable).
    max_samples:
        Scenarios (sequence steps included) read for the statistics.
    write:
        Write ``validation.json``, ``validation_summary.csv`` and the figure into
        ``root``.
    device:
        Torch device for the branch-current part of the coverage features.
    injection_source:
        Where the injected-current statistics come from. ``"auto"`` (default) reads the
        persisted realized-injection columns (``i_h_emission_pct``, per device) and falls
        back to the ``I(h) = Y(h)*V(h)`` reconstruction (``i_h_ratio_pct``, the nodal net
        injection) for a dataset written without them; ``"both"`` measures both — they are
        different quantities, so neither replaces the other silently; ``"persisted"`` /
        ``"reconstructed"`` pin one source. The choice is recorded in ``notes``.

    Returns
    -------
    ValidationReport
        Per-(quantity, order, phase) statistics plus the verdicts, ``verdict`` /
        ``failed`` / ``warnings`` aggregated.
    """
    if injection_source not in ("auto", "persisted", "reconstructed", "both"):
        raise ValueError(
            "injection_source must be 'auto', 'persisted', 'reconstructed' or 'both', "
            f"got {injection_source!r}."
        )
    from injection_hse.data.scenarios import read_dataset

    root = Path(root)
    loaded = read_dataset(root)
    grid = _resolve_grid(root, grid)
    meta = loaded.meta

    dims = meta.get("dims", {})
    n_scenarios = int(dims.get("B", loaded.v.shape[0]))
    n_steps = int(dims.get("T", 1))
    f0 = float(grid.base_frequency_hz)
    orders = (
        [int(round(float(f) / f0)) for f in loaded.frequencies_hz]
        if loaded.frequencies_hz is not None
        else [1]
    )

    from pgml.assembly import base_voltage_per_row, node_phase_index

    state = _flatten_state(loaded.v, max_samples=max_samples)
    nonfinite = int((~np.isfinite(state.real) | ~np.isfinite(state.imag)).sum())
    base_v = base_voltage_per_row(grid).numpy().astype(np.float64)
    phase_codes = node_phase_index(grid).phase_codes.numpy()

    series: list[_Series] = []
    series.extend(_state_series(state, base_v, phase_codes, orders))
    series.extend(_coverage_series(state, grid, orders, device=device))
    persisted = (
        _persisted_injection_series(loaded, max_samples=max_samples)
        if injection_source != "reconstructed"
        else []
    )
    series.extend(persisted)
    reconstruct = injection_source in ("reconstructed", "both") or (
        injection_source == "auto" and not persisted
    )
    if reconstruct:
        series.extend(_injection_series(state, grid, orders))
    notes = [_injection_note(bool(persisted), reconstruct, injection_source)]
    power = _power_series(state, grid, orders)
    series.extend(power)
    series.extend(_sequence_series(loaded, n_steps, power[0].values if power else None))

    stats_list = [s for s in (_summarize(item) for item in series) if s is not None]
    stats = {(s.quantity, s.order, s.phase): s for s in stats_list}
    raw = {
        item.quantity: np.asarray(item.values, dtype=np.float64).reshape(-1)
        for item in series
        if item.order is None and item.phase == "*"
    }

    injected = _injected_orders(loaded.config)
    checks = _hygiene_checks(
        stats,
        orders=orders,
        meta=meta,
        n_scenarios=n_scenarios,
        nonfinite=nonfinite,
        injected=injected,
    )
    checks += _declared_checks(loaded, n_scenarios=n_scenarios)
    checks += _sequence_checks(raw, n_steps=n_steps)
    checks += _norm_checks(stats, orders=orders)
    ranges = load_expected_ranges(reference) if reference is not None else {}
    if ranges:
        checks += _reference_checks(stats, ranges)

    report = ValidationReport(
        kind="dataset",
        path=str(root.resolve()),
        config_hash=meta.get("config_hash"),
        config_type=meta.get("config_type"),
        schema_version=meta.get("schema_version"),
        se_preset_version=_generation_stamp(meta, "se_preset_version"),
        device_library_version=_generation_stamp(meta, "device_library_version"),
        orders=tuple(orders),
        n_scenarios=n_scenarios,
        n_steps=n_steps,
        n_failed=len(meta.get("failed_scenarios", ())),
        stats=stats_list,
        checks=checks,
        reference=None if reference is None else str(reference),
        notes=notes,
    )
    if write:
        write_report(report, root)
    if strict and report.failed:
        raise DatasetValidationError(report.table())
    return report


# =============================================================================
# Corpus + run validation
# =============================================================================
def validate_corpus(
    root,
    *,
    reference=None,
    strict: bool = False,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_members: Optional[int] = None,
    write: bool = True,
    device: str = "cpu",
    injection_source: InjectionSource = "auto",
) -> ValidationReport:
    """Validate every member of a multi-grid corpus, then compare its splits.

    Walks the ``manifest.json`` members (each member's randomized dataset and, where it
    exists, its coherent one), validating each with the member's own persisted grid and
    writing that member's artifacts beside its data. The corpus report additionally
    compares the val / test split envelopes against train: the training set defines what
    the estimator saw, and a split leaving it is a distribution shift that must be
    stated rather than discovered after the fact.

    ``max_members`` caps the members read (``None`` = all); the artifacts are written
    beside ``manifest.json``. ``injection_source`` is passed to every member (see
    :func:`validate_dataset`).
    """
    from .multigrid import MultiGridManifest, read_grid, task_dataset_dir

    root = Path(root)
    manifest = MultiGridManifest.read(root)
    entries = list(manifest.grids)
    if max_members is not None:
        entries = entries[: int(max_members)]

    members: dict[str, ValidationReport] = {}
    per_split: dict[str, list[dict]] = {}
    for entry in entries:
        try:
            grid = read_grid(root, entry.key)
        except Exception as exc:  # noqa: BLE001 - a member must not kill the report
            _log.warning("validate_corpus: %s unreadable (%s)", entry.key, exc)
            continue
        tasks = ["random"] + (
            ["coherent"] if entry.n_coherent_scenarios is not None else []
        )
        for task in tasks:
            path = task_dataset_dir(root, entry.key, task)
            if not (path / "meta.json").is_file():
                continue
            name = entry.key if task == "random" else f"{entry.key}/{task}"
            try:
                member = validate_dataset(
                    path,
                    grid=grid,
                    reference=reference,
                    max_samples=max_samples,
                    write=write,
                    device=device,
                    injection_source=injection_source,
                )
            except Exception as exc:  # noqa: BLE001 - record, never abort the corpus
                _log.warning("validate_corpus: %s failed (%s)", name, exc)
                continue
            members[name] = member
            if task == "random":
                per_split.setdefault(entry.split, []).append(
                    {(s.quantity, s.order, s.phase): s for s in member.stats}
                )

    pooled = {split: _pool_stats(blocks) for split, blocks in per_split.items()}
    checks: list[Check] = []
    train = pooled.get("train")
    if train:
        # the pooled ROW aggregate per order; the per-phase series of a balanced corpus
        # would only repeat the same verdict three times
        quantities = [
            key
            for key in train
            if key[0] in ("v_pu", "v_h_ratio_pct", "thd_pct") and key[2] == "*"
        ]
        others = {
            split: stats
            for split, stats in pooled.items()
            if split != "train" and stats
        }
        checks += _envelope_checks(
            train, others, quantities=sorted(quantities, key=str)
        )

    report = ValidationReport(
        kind="corpus",
        path=str(root.resolve()),
        config_hash=manifest.config_hash,
        schema_version=manifest.schema_version,
        se_preset_version=manifest.se_preset_version,
        device_library_version=manifest.device_library_version,
        orders=tuple(int(o) for o in manifest.orders),
        n_scenarios=sum(e.n_scenarios for e in entries),
        stats=[stat for stats in pooled.values() for stat in stats.values()],
        checks=checks,
        members=members,
        reference=None if reference is None else str(reference),
        notes=[
            f"{len(members)} member dataset(s) over {len(entries)} manifest entries"
        ],
    )
    if write:
        write_report(report, root)
    if strict and report.failed:
        raise DatasetValidationError(report.table())
    return report


def _pool_stats(blocks: list[dict]) -> dict:
    """Pool several members' statistics into one per-split table (median of medians).

    Members differ in size, so the pooled entry keeps the extreme quantiles (the widest
    p1 / p99 over members — an envelope must contain every member) and the median of the
    members' medians as the central value.
    """
    if not blocks:
        return {}
    keys = {key for block in blocks for key in block}
    pooled: dict = {}
    for key in keys:
        entries = [block[key] for block in blocks if key in block]
        if not entries:
            continue
        first = entries[0]
        pooled[key] = SeriesStats(
            quantity=first.quantity,
            label=first.label,
            unit=first.unit,
            order=first.order,
            phase=first.phase,
            count=sum(e.count for e in entries),
            min=min(e.min for e in entries),
            p1=min(e.p1 for e in entries),
            p25=float(np.median([e.p25 for e in entries])),
            median=float(np.median([e.median for e in entries])),
            p75=float(np.median([e.p75 for e in entries])),
            p99=max(e.p99 for e in entries),
            max=max(e.max for e in entries),
            std=float(np.median([e.std for e in entries])),
        )
    return pooled


#: Dataset directories of a workflow run, in report order (the task they carry).
RUN_DATASETS = (
    ("dataset_random", "Randomized snapshots"),
    ("dataset_coherent", "Coherent sequences"),
    ("dataset_profile", "Profiled sequences"),
)


def validate_run(
    run_dir,
    *,
    grid=None,
    reference=None,
    strict: bool = False,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    write: bool = True,
    device: str = "cpu",
    injection_source: InjectionSource = "auto",
) -> ValidationReport:
    """Validate a workflow run's datasets AND compare them against each other.

    A run holds up to three datasets of the SAME grid drawn by different recipes
    (``dataset_random`` / ``dataset_coherent`` / ``dataset_profile``). Each is validated
    on its own, and then the per-order harmonic medians of the sequence tasks are
    compared against the randomized one: the two processes cover different envelopes by
    design, but an order-of-magnitude difference at a harmonic order means one recipe is
    mis-scaled.

    The per-dataset artifacts are written into each dataset directory, the run report
    into ``run_dir``.
    """
    run_dir = Path(run_dir)
    members: dict[str, ValidationReport] = {}
    for name, _label in RUN_DATASETS:
        path = run_dir / name
        if not (path / "meta.json").is_file():
            continue
        members[name] = validate_dataset(
            path,
            grid=grid,
            reference=reference,
            max_samples=max_samples,
            write=write,
            device=device,
            injection_source=injection_source,
        )

    checks: list[Check] = []
    base = members.get("dataset_random")
    if base is not None:
        for name, member in members.items():
            if name == "dataset_random":
                continue
            checks += _cross_dataset_checks(
                base, member, base_name="dataset_random", other_name=name
            )

    first = next(iter(members.values()), None)
    report = ValidationReport(
        kind="run",
        path=str(run_dir.resolve()),
        config_hash=None if first is None else first.config_hash,
        schema_version=None if first is None else first.schema_version,
        orders=() if first is None else first.orders,
        n_scenarios=sum(m.n_scenarios for m in members.values()),
        checks=checks,
        members=members,
        reference=None if reference is None else str(reference),
        notes=[f"datasets: {', '.join(members) or 'none found'}"],
    )
    if write:
        write_report(report, run_dir)
    if strict and report.failed:
        raise DatasetValidationError(report.table())
    return report


# =============================================================================
# Artifacts
# =============================================================================
def write_report(report: ValidationReport, out_dir) -> Path:
    """Write ``validation.json`` + ``validation_summary.csv`` + the figure; return the JSON.

    The figure is best-effort: a plotting failure is logged and never changes a verdict —
    the verdicts are computed before anything is drawn.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT_NAME
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _write_summary_csv(report, out / SUMMARY_NAME)
    try:
        plot_validation(report, out)
    except Exception as exc:  # noqa: BLE001 - a figure is a diagnostic, not a gate
        _log.warning("validation figure skipped (%s: %s)", type(exc).__name__, exc)
    return path


def _write_summary_csv(report: ValidationReport, path: Path) -> None:
    """One row per (dataset, quantity, order, phase): the statistics and their verdict."""
    import polars as pl

    rows: list[dict] = []

    def collect(name: str, current: ValidationReport) -> None:
        worst: dict[tuple, tuple[str, str]] = {}
        for check in current.checks:
            key = (check.quantity, check.order, check.phase)
            previous = worst.get(key)
            if previous is None or _SEVERITY[check.verdict] > _SEVERITY[previous[0]]:
                worst[key] = (check.verdict, check.reason)
        seen: set[tuple] = set()
        for stat in current.stats:
            key = (stat.quantity, stat.order, stat.phase)
            seen.add(key)
            verdict, reason = worst.get(key, ("OK", ""))
            rows.append(
                {
                    "dataset": name,
                    "quantity": stat.quantity,
                    "label": stat.label,
                    "unit": stat.unit,
                    "order": stat.order,
                    "phase": stat.phase,
                    **{key_: getattr(stat, key_) for key_ in STAT_KEYS},
                    "verdict": verdict,
                    "reason": reason,
                }
            )
        for key, (verdict, reason) in worst.items():
            if key in seen:
                continue
            quantity, order, phase = key
            rows.append(
                {
                    "dataset": name,
                    "quantity": quantity,
                    "label": "",
                    "unit": "",
                    "order": order,
                    "phase": phase,
                    **{key_: None for key_ in STAT_KEYS},
                    "verdict": verdict,
                    "reason": reason,
                }
            )
        for member_name, member in current.members.items():
            collect(f"{name}/{member_name}" if name else member_name, member)

    collect(Path(report.path).name, report)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(path)


def plot_validation(report: ValidationReport, out_dir) -> tuple[Path, Path]:
    """Four-panel validation figure -> ``(pdf, svg)`` paths.

    Per-order ``|V_h|`` and injected ``|I_h| / |I_1|`` distributions (median with an
    interquartile band, log scale — the axes on which a dead order and a decade-scale
    mis-drawn emission are both obvious), the EN 50160 markers, and a verdict panel. The
    injection panel draws whichever injection quantity a report carries
    (:data:`INJECTION_QUANTITIES`, the persisted per-device emission first). A run or
    corpus report overlays its members in the per-order panels: that overlay IS the
    snapshots vs sequences drift detector.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from injection_hse.analysis import SERIES_COLORS, figure_size, paper_style, save_paper_figure

    sources = report.members if report.members else {Path(report.path).name: report}
    sources = dict(list(sources.items())[:4])  # keep the legend readable

    with paper_style(columns=2):
        fig, ax = plt.subplots(2, 2, figsize=figure_size(columns=2, aspect=0.8))
        panels = (
            (ax[0, 0], ("v_pu",), "|V_h| / V_base [pu]", "Voltage per order"),
            (
                ax[0, 1],
                INJECTION_QUANTITIES,
                "|I_h| / |I_1| [%]",
                "Injected current per order",
            ),
        )
        for axis, quantities, ylabel, title in panels:
            drawn = False
            for i, (name, source) in enumerate(sources.items()):
                # a member reports ONE injection quantity (persisted or reconstructed);
                # draw whichever it carries, on the same axes.
                orders, median, low, high = ([], [], [], [])
                for quantity in quantities:
                    orders, median, low, high = _order_profile(source, quantity)
                    if orders:
                        break
                if not orders:
                    continue
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                axis.plot(orders, median, "o-", color=color, label=name, markersize=3)
                axis.fill_between(orders, low, high, color=color, alpha=0.18)
                drawn = True
            if drawn:
                axis.set_yscale("log")
                axis.legend(fontsize=5)
            else:
                axis.text(
                    0.5, 0.5, "n/a", ha="center", va="center", transform=axis.transAxes
                )
            axis.set_xlabel("harmonic order")
            axis.set_ylabel(ylabel)
            axis.set_title(title, fontsize=7, loc="left")

        axis = ax[1, 0]
        drawn = False
        for i, (name, source) in enumerate(sources.items()):
            orders, median, low, high = _order_profile(source, "v_h_ratio_pct")
            if not orders:
                continue
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            axis.plot(orders, median, "o-", color=color, label=name, markersize=3)
            axis.fill_between(orders, low, high, color=color, alpha=0.18)
            limits = harmonic_voltage_limits(orders)
            marks = [limits[o].en50160_pct for o in orders if o in limits]
            if marks:
                axis.plot(
                    [o for o in orders if o in limits],
                    marks,
                    "k_",
                    markersize=8,
                    label="EN 50160" if i == 0 else None,
                )
            drawn = True
        if drawn:
            axis.set_yscale("log")
            axis.legend(fontsize=5)
        else:
            axis.text(
                0.5, 0.5, "n/a", ha="center", va="center", transform=axis.transAxes
            )
        axis.set_xlabel("harmonic order")
        axis.set_ylabel("|V_h| / |V_1| [%]")
        axis.set_title("Harmonic voltage vs EN 50160", fontsize=7, loc="left")

        axis = ax[1, 1]
        axis.axis("off")
        axis.text(
            0.0,
            1.0,
            _verdict_text(report),
            ha="left",
            va="top",
            fontsize=5,
            family="monospace",
            transform=axis.transAxes,
        )
        try:
            return save_paper_figure(fig, out_dir, FIGURE_STEM)
        finally:
            # a corpus writes one figure per member; an unclosed figure per member
            # exhausts the pyplot figure pool long before the corpus is done.
            plt.close(fig)


def _order_profile(report: ValidationReport, quantity: str):
    """``(orders, median, p25, p75)`` of one per-order quantity, sorted by order.

    The INTERQUARTILE band, not p1..p99: the recipe deliberately draws emission down to
    zero, so a p1 whisker spans a dozen decades on a log axis and hides the profile the
    panel exists to show.
    """
    entries = [
        s
        for s in report.stats
        if s.quantity == quantity and s.phase == "*" and s.order is not None
    ]
    entries.sort(key=lambda s: s.order or 0)
    floor = 1e-18
    return (
        [int(s.order) for s in entries],
        [max(s.median, floor) for s in entries],
        [max(s.p25, floor) for s in entries],
        [max(s.p75, floor) for s in entries],
    )


def _verdict_text(report: ValidationReport, *, limit: int = 14) -> str:
    """The verdict panel's text: the headline plus the first failures and warnings."""
    lines = [report.summary_line()]
    for check in (report.failures + report.warnings)[:limit]:
        where = check.quantity + (f" h{check.order}" if check.order is not None else "")
        reason = check.reason if len(check.reason) < 78 else check.reason[:75] + "..."
        lines.append(f"{check.verdict:<4} {where:<18} {reason}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def detect_kind(path: Path) -> str:
    """Which layout ``path`` holds: ``"corpus"``, ``"run"`` or ``"dataset"``."""
    if (path / "manifest.json").is_file():
        try:
            if "grids" in json.loads((path / "manifest.json").read_text()):
                return "corpus"
        except (ValueError, OSError):
            pass
    if any((path / name / "meta.json").is_file() for name, _ in RUN_DATASETS):
        return "run"
    if (path / "meta.json").is_file():
        return "dataset"
    raise SystemExit(
        f"{path} is neither a dataset (meta.json), a corpus (manifest.json) nor a run "
        f"({' / '.join(name for name, _ in RUN_DATASETS)})."
    )


def validate_path(
    path,
    *,
    grid=None,
    reference=None,
    strict: bool = False,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    write: bool = True,
    device: str = "cpu",
    injection_source: InjectionSource = "auto",
) -> ValidationReport:
    """Validate whichever of the three layouts ``path`` holds (see :func:`detect_kind`)."""
    path = Path(path)
    kind = detect_kind(path)
    if kind == "corpus":
        return validate_corpus(
            path,
            reference=reference,
            strict=strict,
            max_samples=max_samples,
            write=write,
            device=device,
            injection_source=injection_source,
        )
    if kind == "run":
        return validate_run(
            path,
            grid=grid,
            reference=reference,
            strict=strict,
            max_samples=max_samples,
            write=write,
            device=device,
            injection_source=injection_source,
        )
    return validate_dataset(
        path,
        grid=grid,
        reference=reference,
        strict=strict,
        max_samples=max_samples,
        write=write,
        device=device,
        injection_source=injection_source,
    )


def _cli(argv=None) -> int:
    """``python -m injection_hse.data.validate`` — validate a dataset, corpus or run directory."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("matplotlib", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        prog="python -m injection_hse.data.validate",
        description="Validate a generated dataset, multi-grid corpus or workflow run.",
    )
    parser.add_argument("path", help="Dataset, corpus or run directory.")
    parser.add_argument(
        "--grid",
        default=None,
        help="grid.json (default: found at or above the dataset).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Expectations YAML (run/configs/expected_ranges.yaml) — the "
        "measured-population layer.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero when anything FAILS."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Scenarios read per dataset (sequence steps included).",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write the artifacts."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--injection-source",
        choices=("auto", "persisted", "reconstructed", "both"),
        default="auto",
        help="Injected-current statistics: the persisted per-device emission, the "
        "I(h)=Y(h)*V(h) reconstruction of the nodal net injection, or both "
        "(default: auto = persisted where available).",
    )
    args = parser.parse_args(argv)

    report = validate_path(
        args.path,
        grid=args.grid,
        reference=args.reference,
        strict=False,
        max_samples=args.max_samples,
        write=not args.no_write,
        device=args.device,
        injection_source=args.injection_source,
    )
    print(report.table(limit=40))
    for name, member in report.members.items():
        print(f"  [{name}] {member.summary_line()}")
    return 1 if (args.strict and report.failed) else 0


__all__ = [
    "STAT_KEYS",
    "DEAD_ORDER_PU",
    "LIVE_HOUR_BAND",
    "INJECTION_QUANTITIES",
    "PERSISTED_INJECTION_QUANTITY",
    "RECONSTRUCTED_INJECTION_QUANTITY",
    "REPORT_NAME",
    "SUMMARY_NAME",
    "DatasetValidationError",
    "SeriesStats",
    "Check",
    "ValidationReport",
    "ExpectedRange",
    "EN50160_MAX_ORDER",
    "HarmonicVoltageLimit",
    "harmonic_voltage_limits",
    "iec61000_2_2_level_pct",
    "load_expected_ranges",
    "validate_dataset",
    "validate_corpus",
    "validate_run",
    "validate_path",
    "detect_kind",
    "write_report",
    "plot_validation",
]


if __name__ == "__main__":  # `python -m injection_hse.data.validate <dir>`
    # Delegate to the PACKAGE module: ``python -m`` executes this file a second time under
    # the name ``__main__``, and the report types must be the ones the package exports (a
    # second copy would not compare or serialize as the same class).
    from injection_hse.data.validate import _cli as _package_cli

    raise SystemExit(_package_cli())
