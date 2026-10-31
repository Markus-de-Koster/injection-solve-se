"""Multi-grid dataset generation: many grids, one reproducible training set.

Every result injection_hse reports so far trains and evaluates on ONE grid. Cross-grid generalization
needs a training set whose members are whole GRIDS, and a split that holds topologies back
rather than scenarios. This module builds that set:

1. **Collect grids** from the configured sources — a ``pgg`` generation export (synthesized
   LV grids, cell-stratified), and the real published LV grids ``pghub`` loads (SimBench
   workbooks, ding0 districts). Each source has reproducible selection knobs (cap, stride,
   node-count window) so a named config plus its seed reproduces the member list.
2. **Generate per grid** with the unchanged ``pgml.scenarios`` pipeline: one scenario batch
   per grid from a SHARED sampling template (the same load-scale ranges everywhere, so the
   grids differ in topology, not in how they were excited), written by
   ``pgml.scenarios.write_dataset`` into ``<root>/grids/<key>/`` — with the grid itself
   persisted beside its dataset as ``grid.json``, because a parquet dataset stores only the
   node-phase row layout and a multi-grid consumer cannot be handed one grid explicitly.
   Per-grid seeds are children of the run seed (``numpy.random.SeedSequence``), so grids are
   independent and the whole set is reproducible from ``(config, seed)``.
   With a ``coherent`` section the same grid ADDITIONALLY gets a sequence dataset of coherent
   SEQUENCES (``injection_hse.data.scenarios.CoherentSpectrumConfig``) under ``<root>/grids/<key>/coherent/``
   — the two measurement-scarcity tasks of the state-estimation curriculum on one member list:
   The snapshots cover the operating space independently; the sequences move a device
   population through time. Both draw from the same member's seed stream, on separate words,
   so neither task's numbers depend on whether the other was generated.
3. **Record a manifest** at the root: every member with its split, size, provenance and
   layout fingerprint; every failure; and a LEAKAGE block — whether the ``pgg`` run that
   produced the synthetic grids was itself seeded from real hub grids, and which ding0
   districts were excluded. A cross-grid generalization number is only meaningful together
   with that block.
4. **Validate what was generated** (:func:`injection_hse.data.validate.validate_corpus`, on by
   default): every member against the norms and its own configuration's claims, plus the
   train/val/test split envelopes. The verdict is summarized in the manifest and written
   in full beside it.

The SPLIT is by grid: the test split holds REAL grids only (the hardest, most honest
question — does an estimator trained on synthesized topologies work on a measured one?), and
train/val split the generated grids by ARCHIVE CELL where the export provides one, so the
validation topologies come from behavior regions the training set never saw.

Boundary note: the loading path (:mod:`injection_hse.data.multigraph`) depends on ``pgml`` only. The
GENERATION path here additionally reads the ``pgg`` and ``pghub`` PUBLIC APIs to obtain
grids (installed by the ``power-grid-learn[multigrid]`` extra); both imports are lazy and
confined to the source resolvers, so importing :mod:`injection_hse.data` never pulls them in.

CLI::

    python -m injection_hse.data.multigrid --config multigrid.yaml [--workers 8] [--no-validate]
    python -m injection_hse.data.multigrid --json-schema | --example
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from injection_hse.data.scenarios import HIGH_ACTIVITY_START_TIME

_log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
GRID_FILE = "grid.json"
GRIDS_DIR = "grids"
#: Subdirectory of a member holding its COHERENT sequence dataset; the randomized
#: snapshots stay in the member directory itself.
COHERENT_DIR = "coherent"

#: Sources whose grids are REAL published networks — the held-out test split.
REAL_SOURCE_KINDS = ("simbench", "ding0")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Grid sources
# =============================================================================
class _SourceBase(_Base):
    """Shared selection knobs of a grid source (reproducible, order-stable)."""

    #: Maximum number of grids taken from this source (``None`` = no cap).
    max_grids: Optional[int] = Field(default=None, gt=0)
    #: Skip grids with more than this many NODES (``None`` = no limit) — the memory guard,
    #: since a dense solve grows with the node-phase row count. Applied before ``max_grids``,
    #: so the cap counts KEPT grids.
    max_nodes: Optional[int] = Field(default=None, gt=0)
    #: Skip grids with fewer than this many NODES (``None`` = no limit). The mirror of
    #: ``max_nodes``, also applied before ``max_grids``: a corpus that trains on synthesized
    #: feeders can exclude the degenerate few-node stubs while the REAL test sources keep
    #: theirs, so the training distribution is trimmed without narrowing the held-out question.
    min_nodes: Optional[int] = Field(default=None, gt=0)
    #: Take every ``stride``-th candidate of the source ENUMERATION, starting at ``offset``
    #: (a reproducible subsample, independent of what is kept; the pattern ``pgg``'s hub
    #: seeding uses to keep two slices of one dataset disjoint).
    stride: int = Field(default=1, ge=1)
    offset: int = Field(default=0, ge=0)


class PggExportSource(_SourceBase):
    """Grids synthesized by a ``pgg`` generation run, read from its dataset EXPORT.

    ``export_dir`` is a directory written by ``pgg.export.export_dataset`` (one genome JSON
    per grid plus a flat index carrying the behavior CELL each genome occupies). The cell is
    what the train/val split stratifies on, so validation topologies come from behavior
    regions training never saw.
    """

    kind: Literal["pgg"] = "pgg"
    export_dir: str
    #: Keep only feasible genomes (an export may carry flagged infeasible ones).
    feasible_only: bool = True


class SimbenchSource(_SourceBase):
    """Real SimBench LV grids, via the ``pghub`` archetype-workbook dataset.

    ``pghub`` ships the SimBench benchmark networks inside its ``european_archetype``
    workbook dataset (one family among several), so the source resolves to that dataset
    filtered to the requested ``families`` and the LV voltage level. ``families`` is matched
    case-insensitively as a substring of the record's family folder, so the default picks up
    whichever SimBench naming the local data root uses.
    """

    kind: Literal["simbench"] = "simbench"
    dataset: str = "european_archetype"
    families: tuple[str, ...] = ("simbench",)
    voltage_level: str = "lv"
    #: Override the ``pghub`` data root (``None`` = the ``PGHUB_DATA`` default).
    data_root: Optional[str] = None


class Ding0Source(_SourceBase):
    """Real ding0 LV grids, one district at a time.

    ``exclude_districts`` names the districts that must NOT contribute — the reference
    districts a generation study already compared against. Training on a grid that was used
    to shape or score the generator is leakage, and it is invisible after the fact, so the
    exclusion list is recorded in the manifest's leakage block.
    """

    kind: Literal["ding0"] = "ding0"
    #: District directory names to skip entirely (the generation study's reference set).
    exclude_districts: tuple[str, ...] = ()
    #: Only these district directory names contribute (``None`` = every district that is not
    #: excluded). A whitelist is how a TEST set names its districts precisely: enumerating
    #: thousands of districts loads every candidate grid, so selecting by exclusion alone
    #: does not scale to a handful of districts out of the full ding0 corpus.
    districts: Optional[tuple[str, ...]] = None
    #: Maximum LV grids taken from one district (breadth over districts beats depth in one).
    max_per_district: int = Field(default=4, gt=0)
    data_root: Optional[str] = None


class SyntheticFeederSource(_SourceBase):
    """Synthetic radial feeders of varying size (``pgml.grids.synthetic_feeder``).

    Needs no data root and no generation run: a self-contained source for smoke tests, CI
    and a first end-to-end pass of the multi-grid pipeline. The grids are schema-only radial
    MV feeders whose node count sweeps ``node_range`` in ``n_grids`` even steps (the sweep
    bounds are distinct from the inherited ``max_nodes`` LIMIT, which still applies).
    """

    kind: Literal["synthetic"] = "synthetic"
    n_grids: int = Field(default=6, gt=0)
    #: Inclusive ``(smallest, largest)`` node count of the generated feeders.
    node_range: tuple[int, int] = (5, 12)
    n_feeders: int = Field(default=2, ge=1)


GridSourceConfig = Annotated[
    Union[PggExportSource, SimbenchSource, Ding0Source, SyntheticFeederSource],
    Field(discriminator="kind"),
]


# =============================================================================
# Generation config
# =============================================================================
class CoherentSequenceConfig(_Base):
    """The COHERENT sequence dataset generated beside every member's snapshots.

    The randomized snapshots of :class:`MultiGridGenerationConfig` are the harmonic-flow
    surrogate's "cover the space" data: independent per-order draws, timeless. This
    section adds the second task of the state-estimation curriculum — coherent SEQUENCES
    (``injection_hse.data.scenarios.CoherentSpectrumConfig``, ``[B, T, H, N]``) in which a device population
    moves through time, so the estimator has to associate a pattern with a node rather than
    invert one snapshot.

    ``mode`` selects the temporal harmonic structure, the same two the single-grid workflow
    offers:

    - ``"composed"`` (default) — a statistical device-class COMPOSITION whose per-step device
      activity moves the fundamental power AND the injected spectrum jointly, stamped with an
      absolute time axis (the data behind the periodic time features) and per-class attribution
      labels. A load without a ``consumer_type`` falls back to the minimal base composition,
      so it applies to any member of a heterogeneous corpus.
    - ``"fingerprint"`` — the per-device mode bank: a stable signature that STICKS to a mode
      (Markov ``dwell``) and WANDERS (AR(1) jitter) over a per-scenario fundamental. Timeless;
      kept for ablations that isolate the joint power/spectrum coupling.

    The injected orders are the run's ``orders`` above 1 (the same rule the snapshots use), and
    the per-order magnitude reference is IEC 61000-3-2 — the appliance current-emission
    standard, which is the physically correct reference for a device fingerprint.

    ``mode_bank_seed`` seeds ONLY the per-device fingerprint bank. ``None`` (default) derives
    it per member from that member's own seed stream, so every grid carries its own device
    signatures and the whole set still reproduces from ``(config, seed)``. Setting it pins ONE
    bank across every member — the recipe for a second corpus whose fingerprints are known to
    differ from (or match) a first one.
    """

    #: Coherent sequences generated per grid (the ``B`` axis).
    n_scenarios: int = Field(default=64, gt=0)
    #: Steps per sequence (the ``T`` axis).
    n_steps: int = Field(default=8, gt=0)
    mode: Literal["composed", "fingerprint"] = "composed"
    #: Base spectra per device (the fingerprint mode bank).
    n_modes: int = Field(default=2, ge=1)
    #: Probability a device stays in its mode from one step to the next.
    dwell: float = Field(default=0.9, ge=0.0, le=1.0)
    #: Seconds between steps (900 = 15 min, a sub-hour window with a moving fundamental).
    step_size_s: float = Field(default=900.0, gt=0.0)
    #: Absolute ISO 8601 anchor of the composed sequences' daily / weekly / seasonal
    #: phase. The time axis is ONE shared window for every scenario, so it defaults to the
    #: late-afternoon high-activity band (households ramping, EV arrivals, PV still
    #: producing): anchored at midnight, a window of a few hours sits in the dead band of
    #: every activity preset and the whole corpus is generated with the rosters dark.
    start_time: str = HIGH_ACTIVITY_START_TIME
    #: Fingerprint (mode) bank seed; ``None`` derives one per member (see the class docstring).
    mode_bank_seed: Optional[int] = None
    #: Co-variation of the COMPOSED devices' activity, as the relative spread of a per-scenario
    #: shared latent (``behavioral`` for consumption, ``cloud`` for PV;
    #: :class:`injection_hse.data.scenarios.CompositionConfig`). In ``composed`` mode the composition
    #: SUPERSEDES the ``load_scale`` draw for every load it covers, so these — not
    #: ``load_correlation`` — are what decides whether a sequence's FUNDAMENTAL moves between
    #: scenarios. Left low, thousands of member devices average out and every coherent
    #: sequence sits at the same operating point, which makes the fundamental a near-constant
    #: an estimator never has to work for. Raise them to match the snapshot task's spread.
    behavioral_coupling: float = Field(default=0.3, ge=0.0)
    cloud_coupling: float = Field(default=0.5, ge=0.0)
    #: Duty-cycle multiplier on the composed devices' availability
    #: (:attr:`injection_hse.data.scenarios.CompositionConfig.activity_scale`). The composed aggregate
    #: otherwise sits far below the snapshot task's loading, so the sequences' FUNDAMENTAL
    #: barely moves and the critical, loaded states never appear in the sequences at all.
    activity_scale: float = Field(default=1.0, gt=0.0)


class MultiGridGenerationConfig(_Base):
    """One reproducible multi-grid dataset: sources, sampling template, split.

    ``(this config, seed) -> the dataset`` — the member list, the per-grid scenario seeds and
    the split all derive from it. The scenario TEMPLATE is deliberately shared across grids
    (one load-scale range, one spectrum range): the members must differ in topology, not in
    how hard they were driven, or a cross-grid result cannot be attributed to structure.

    ``orders`` defaults to the fundamental only (``[1]``) — the first multi-grid milestone
    is fundamental state estimation across topologies. Every order above 1 is an INJECTED
    order: requesting them adds the per-load current spectrum to the snapshot template and, with
    a ``coherent`` section, is the order axis of the sequences' device fingerprint.

    ``coherent`` (default ``None``) adds the second curriculum task: one coherent-sequence
    dataset per member beside its snapshots (:class:`CoherentSequenceConfig`). Without it the
    generated dataset is exactly what a snapshot-only run has always produced.
    """

    #: Where the dataset is written (``<root>/manifest.json`` + ``<root>/grids/<key>/``).
    output_root: str
    sources: list[GridSourceConfig] = Field(min_length=1)
    seed: int = 0
    #: Randomized snapshots generated per grid.
    n_scenarios: int = Field(default=256, gt=0)
    #: Solved harmonic orders (``[1]`` = fundamental only); orders > 1 are the INJECTED ones.
    orders: tuple[int, ...] = (1,)
    #: The COHERENT sequence dataset per member (``None`` = snapshots only).
    coherent: Optional[CoherentSequenceConfig] = None
    #: Per-load apparent-power scale range. Reaches down to no load: a real feeder spends
    #: much of its time near-idle, and covering that corner keeps the fundamental voltage
    #: deviation centred on nominal instead of several percent low.
    load_scale: tuple[float, float] = (0.0, 1.0)
    #: Rank correlation between the members' load draws (shared ``"demand"`` factor).
    #: Independent draws (``0``) leave the AGGREGATE demand of a feeder nearly constant
    #: however wide ``load_scale`` is — the mean of a few hundred loads concentrates as
    #: ``1/sqrt(N)`` — so the fundamental operating point barely moves between scenarios
    #: and a boundary current measurement alone reconstructs it. A positive correlation is
    #: what makes the total demand swing the way a real feeder's does over a day.
    load_correlation: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Rank correlation between the inverter (PV) draws (shared ``"solar"`` factor).
    #: ``1`` is one irradiance scale for the whole fleet, which makes every site's export
    #: a deterministic function of one number; below ``1`` the sites differ (orientation,
    #: shading, curtailment), so a feeder section can export while another imports — the
    #: local ambiguity a state estimator has to resolve and a boundary measurement cannot.
    #: ``None`` drops the shared factor and scales the whole fleet by ONE draw.
    pv_correlation: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    #: Per-load harmonic-current range as a fraction of the per-device IEC 61000-3-2
    #: emission limit (orders > 1). Reaching above 1 keeps the measured max-THD envelope
    #: while the zero end produces the near-clean loads that stop emission from being a
    #: deterministic function of the operating point.
    spectrum_fraction: tuple[float, float] = (0.0, 2.0)
    #: Fraction of the GENERATED grids held out for validation (split by archive cell where
    #: the export provides one, so validation topologies are structurally held out).
    val_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    sampling: Literal["sobol", "lhs", "independent"] = "sobol"
    dtype: Literal["complex64", "complex128"] = "complex128"
    layout: Literal["wide", "long"] = "wide"
    symmetry: Literal["auto", "symmetric", "asymmetric"] = "auto"
    #: Solve the scenario batch in slices of at most this many scenarios (memory guard on
    #: the larger members); ``None`` = one batch.
    chunk_size: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_coherent_orders(self) -> "MultiGridGenerationConfig":
        """A coherent section needs an injected order — its fingerprint IS the spectrum."""
        if self.coherent is not None and not [o for o in self.orders if int(o) > 1]:
            raise ValueError(
                "the coherent section needs at least one harmonic order above 1 "
                f"(orders={tuple(self.orders)}): its device fingerprint IS the injected "
                "spectrum. Add e.g. orders=(1, 3, 5) or drop the coherent section."
            )
        return self


# =============================================================================
# Manifest
# =============================================================================
class GridEntry(_Base):
    """One member grid of a multi-grid dataset."""

    key: str
    """Directory-safe unique key; the dataset lives at ``<root>/grids/<key>/``."""
    split: Literal["train", "val", "test"]
    source: str
    """Provenance string: ``pgg:<export>#cell<n>``, ``simbench:<code>``,
    ``ding0:<district>_<lv id>`` or ``synthetic:<n_nodes>``."""
    kind: Literal["pgg", "simbench", "ding0", "synthetic"]
    n_nodes: int
    """Nodes in the grid."""
    n_rows: int
    """Node-phase ROWS ``N`` (the state/tensor width of this member)."""
    n_scenarios: int
    """Randomized snapshots stored (the snapshot ``B`` axis)."""
    n_failed: int = 0
    """Snapshots whose solve did not converge (dropped by the loader by default)."""
    n_coherent_scenarios: Optional[int] = None
    """Coherent SEQUENCES stored under ``coherent/`` (``None`` = no sequence dataset)."""
    n_coherent_failed: int = 0
    """Coherent sequences whose solve did not converge."""
    n_steps: Optional[int] = None
    """Steps ``T`` per coherent sequence (``None`` = no sequence dataset)."""
    cell: Optional[int] = None
    """Behavior-archive cell (``pgg`` members only) — the stratification key."""
    layout_fingerprint: str
    """``pgml.topology.layout_fingerprint`` of the persisted grid (row-layout identity)."""


class GridFailure(_Base):
    """A grid that could not be generated (recorded, never fatal)."""

    key: str
    source: str
    error: str


class PggProvenance(_Base):
    """Where a ``pgg`` export came from, and whether that run saw real grids.

    A generation run may be SEEDED from real hub grids (``pgg`` ``SeedPopulationConfig
    .hub_dataset``). If it was, synthetic members can carry structure copied from real
    networks, and a test score on real grids is no longer fully held out. Recorded per
    export directory; ``"unknown"`` when the source run's config is not readable (the export
    only records the run directory, which may be gone).
    """

    export_dir: str
    source_run_dir: Optional[str] = None
    hub_seeding: Literal["on", "off", "unknown"] = "unknown"
    hub_dataset: Optional[str] = None
    hub_max_count: Optional[int] = None


class LeakageRecord(_Base):
    """What could make the held-out test split less held-out than it looks."""

    pgg: list[PggProvenance] = Field(default_factory=list)
    ding0_excluded_districts: tuple[str, ...] = ()


class MultiGridManifest(_Base):
    """The root record of a multi-grid dataset (``manifest.json``).

    Self-describing: the generation config that produced it, the member grids with their
    split and provenance, the failures, and the leakage block. The loader
    (:class:`injection_hse.data.MultiGridDataSource`) reads nothing else to assemble a run.

    ``version`` tracks the manifest layout: ``1`` records one randomized dataset per member,
    ``2`` additionally records the COHERENT sequence dataset each member may carry
    (``GridEntry.n_coherent_scenarios`` / ``n_steps``), ``3`` the generation provenance
    (``config_hash`` / ``provenance`` / ``device_library_version`` / ``standards``), ``4``
    the corpus ``validation`` summary, ``5`` the recipe revision
    (``se_preset_version``). An older manifest reads unchanged — the added fields default
    to "not recorded", which is what it describes.
    """

    version: int = 5
    created_utc: str
    schema_version: str
    seed: int
    orders: tuple[int, ...]
    config: MultiGridGenerationConfig
    grids: list[GridEntry] = Field(default_factory=list)
    failures: list[GridFailure] = Field(default_factory=list)
    leakage: LeakageRecord = Field(default_factory=LeakageRecord)
    #: Fingerprint of the generation config (``pgml.scenarios.config_hash``): what a
    #: corpus must be compared on before it is reused for a run configured differently.
    config_hash: Optional[str] = None
    #: The code that produced the corpus (``pgml.provenance.code_provenance``): commit,
    #: dirty flag, library versions. Two corpora built from one config and seed can still
    #: differ numerically if the code between them changed.
    provenance: dict = Field(default_factory=dict)
    #: Revision of the calibrated scenario recipe the members were drawn from
    #: (``injection_hse.data.scenarios.SE_PRESET_VERSION``). A recipe change moves the data without
    #: moving the config, so it is recorded beside the config fingerprint.
    se_preset_version: Optional[str] = None
    #: Revision of the shipped device-class library the composed sequences were drawn from
    #: (``injection_hse.data.scenarios.DEVICE_LIBRARY_VERSION``).
    device_library_version: Optional[str] = None
    #: The EN 50160 / IEC 61000-3-2 tables that were active, with their content hashes
    #: (an environment override replaces them silently).
    standards: dict = Field(default_factory=dict)
    #: Compact result of the corpus validation (:func:`injection_hse.data.validate.validate_corpus`):
    #: the overall verdict, the FAIL / WARN counts and the members that failed. The full
    #: report lives beside this file in ``validation.json``. Empty when the corpus was
    #: generated with validation switched off (or before it existed).
    validation: dict = Field(default_factory=dict)

    def split(self, name: str) -> list[GridEntry]:
        """Member entries of one split (``"train"`` / ``"val"`` / ``"test"``)."""
        return [entry for entry in self.grids if entry.split == name]

    @property
    def has_coherent(self) -> bool:
        """Whether every listed member carries a COHERENT dataset."""
        return bool(self.grids) and all(
            entry.n_coherent_scenarios is not None for entry in self.grids
        )

    def entry(self, key: str) -> GridEntry:
        """The member entry with this key (raises if absent)."""
        for candidate in self.grids:
            if candidate.key == key:
                return candidate
        raise KeyError(f"no grid {key!r} in the manifest.")

    def write(self, root) -> Path:
        """Write ``<root>/manifest.json``; returns the file path."""
        path = Path(root) / MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, root) -> "MultiGridManifest":
        """Read ``<root>/manifest.json`` (``root`` may be the file itself)."""
        path = Path(root)
        if path.is_dir():
            path = path / MANIFEST_NAME
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# =============================================================================
# Source resolution
# =============================================================================
class _Candidate:
    """One collected grid, before generation (grid + provenance)."""

    __slots__ = ("key", "kind", "source", "grid", "cell")

    def __init__(self, key: str, kind: str, source: str, grid, cell=None) -> None:
        self.key = key
        self.kind = kind
        self.source = source
        self.grid = grid
        self.cell = cell


_KEY_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def _key(*parts: object) -> str:
    """A directory-safe member key from provenance parts."""
    raw = "_".join(str(p) for p in parts if p is not None and str(p) != "")
    return _KEY_SAFE.sub("-", raw).strip("-_") or "grid"


def _n_rows(grid) -> int:
    """Node-phase row count ``N`` of a grid."""
    from pgml.assembly import node_phase_index

    return int(node_phase_index(grid).size)


def _take(factories: Iterable, source: _SourceBase) -> list[_Candidate]:
    """Select candidates from an ordered stream of ``() -> _Candidate | None`` factories.

    ``stride``/``offset`` subsample the SOURCE ENUMERATION (a reproducible slice of the
    dataset, independent of what is kept), then each surviving candidate is materialized and
    dropped if its node count falls outside ``[min_nodes, max_nodes]``, and ``max_grids`` caps
    the KEPT grids. Materializing lazily matters: building a grid means parsing a workbook or
    rebuilding a genome, and a capped source must not pay for the whole dataset.
    """
    kept: list[_Candidate] = []
    for i, make in enumerate(factories):
        if i % source.stride != source.offset % source.stride:
            continue
        candidate = make()
        if candidate is None or not _within_size(candidate.grid, source):
            continue
        kept.append(candidate)
        if source.max_grids is not None and len(kept) >= source.max_grids:
            break
    return kept


def _within_size(grid, source: _SourceBase) -> bool:
    """Whether a grid's node count lies in the source's ``[min_nodes, max_nodes]`` window."""
    n_nodes = len(grid.nodes)
    if source.max_nodes is not None and n_nodes > source.max_nodes:
        return False
    return source.min_nodes is None or n_nodes >= source.min_nodes


def _collect_pgg(source: PggExportSource) -> tuple[list[_Candidate], PggProvenance]:
    """Materialize the grids of a ``pgg`` dataset export (public ``pgg`` API only)."""
    from pgg import build, default_catalog
    from pgg.export import load_exported

    export_dir = Path(source.export_dir)
    catalog = default_catalog()

    def _make(ir, row):
        cell = row.get("cell")
        return _Candidate(
            key=_key("pgg", export_dir.name, row["grid_id"]),
            kind="pgg",
            source=f"pgg:{export_dir.name}#cell{cell}",
            grid=build(ir, catalog),
            cell=None if cell is None else int(cell),
        )

    factories = (
        (lambda ir=ir, row=row: _make(ir, row))
        for ir, row in load_exported(export_dir)
        if not source.feasible_only or row.get("feasible", True)
    )
    return _take(factories, source), _pgg_provenance(export_dir)


def _pgg_provenance(export_dir: Path) -> PggProvenance:
    """Read the export's source run and report whether that run used hub (real-grid) seeding."""
    provenance = PggProvenance(export_dir=str(export_dir))
    meta_path = export_dir / "meta.json"
    if not meta_path.is_file():
        return provenance
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return provenance
    run_dir = meta.get("source_run_dir")
    provenance = provenance.model_copy(update={"source_run_dir": run_dir})
    if not run_dir:
        return provenance
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.is_file():
        return provenance
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return provenance
    seeding = (raw.get("seed_population") or {}) if isinstance(raw, dict) else {}
    dataset = seeding.get("hub_dataset")
    count = seeding.get("hub_max_count")
    return provenance.model_copy(
        update={
            "hub_seeding": "on" if dataset and (count or 0) > 0 else "off",
            "hub_dataset": dataset,
            "hub_max_count": None if count is None else int(count),
        }
    )


def _hub_records(dataset: str, data_root: Optional[str], **kwargs):
    """``pghub`` records of a dataset, honoring a per-source data-root override."""
    import os

    import pghub

    previous = os.environ.get(pghub.registry.DATA_ROOT_ENV)
    if data_root is not None:
        os.environ[pghub.registry.DATA_ROOT_ENV] = str(data_root)
    try:
        return list(pghub.records(dataset, **kwargs))
    finally:
        if data_root is not None:
            if previous is None:
                os.environ.pop(pghub.registry.DATA_ROOT_ENV, None)
            else:
                os.environ[pghub.registry.DATA_ROOT_ENV] = previous


def _collect_simbench(source: SimbenchSource) -> list[_Candidate]:
    """Real SimBench LV grids via the ``pghub`` archetype dataset."""
    import pghub

    families = tuple(f.lower() for f in source.families)
    records = _hub_records(
        source.dataset, source.data_root, voltage_level=source.voltage_level
    )
    matching = [
        record
        for record in records
        if not families
        or any(f in str(record.metadata.get("family", "")).lower() for f in families)
    ]

    def _make(record):
        return _Candidate(
            key=_key("simbench", record.grid_id),
            kind="simbench",
            source=f"simbench:{record.grid_id}",
            grid=pghub.load(record).grid,
        )

    return _take(((lambda r=r: _make(r)) for r in matching), source)


def _collect_ding0(source: Ding0Source) -> list[_Candidate]:
    """Real ding0 LV grids, excluding the configured districts."""
    from pghub.loaders.ding0 import iter_lv_grids

    excluded = {str(d) for d in source.exclude_districts}
    included = (
        {str(d) for d in source.districts} if source.districts is not None else None
    )
    districts = [
        record
        for record in _hub_records("ding0", source.data_root)
        if record.grid_id not in excluded
        and (included is None or record.grid_id in included)
    ]

    def _factories():
        for record in districts:
            for taken, (lv_id, grid) in enumerate(iter_lv_grids(Path(record.source))):
                if taken >= source.max_per_district:
                    break
                district = record.grid_id
                yield lambda d=district, i=lv_id, g=grid: _Candidate(
                    key=_key("ding0", d, i),
                    kind="ding0",
                    source=f"ding0:{d}_{i}",
                    grid=g,
                )

    return _take(_factories(), source)


def _collect_synthetic(source: SyntheticFeederSource) -> list[_Candidate]:
    """Synthetic radial feeders sweeping ``node_range`` (needs no data root)."""
    from pgml.grids import synthetic_feeder

    low, high = (int(bound) for bound in source.node_range)
    if high < low or low < 2:
        raise ValueError(
            f"synthetic node_range must be an increasing pair of >= 2, got "
            f"{source.node_range}."
        )
    span = max(1, source.n_grids - 1)
    sizes = [
        low + round((high - low) * i / span) if source.n_grids > 1 else low
        for i in range(source.n_grids)
    ]
    factories = (
        lambda i=i, size=size: _Candidate(
            key=_key("synthetic", f"{i:03d}", f"n{size}"),
            kind="synthetic",
            source=f"synthetic:{size}",
            grid=synthetic_feeder(int(size), n_feeders=int(source.n_feeders)),
        )
        for i, size in enumerate(sizes)
    )
    return _take(factories, source)


def collect_grids(
    config: MultiGridGenerationConfig,
) -> tuple[list, LeakageRecord]:
    """Resolve every configured source into member grids (order-stable) + the leakage record.

    Sources are resolved in config order and each source's own selection knobs
    (``stride``/``offset``/``max_grids``/``max_nodes``) apply; duplicate keys are dropped
    with a warning so one grid cannot enter the set twice under two sources. A source that
    raises (missing data root, unreadable export) is logged and SKIPPED — a partial dataset
    is more useful than none, and the manifest records what was produced.

    Returns
    -------
    tuple[list, LeakageRecord]
        The candidate grids (opaque records consumed by :func:`generate_multigrid_dataset`)
        and the provenance/leakage block for the manifest.
    """
    candidates: list[_Candidate] = []
    leakage = LeakageRecord()
    excluded: list[str] = []
    for source in config.sources:
        try:
            if isinstance(source, PggExportSource):
                found, provenance = _collect_pgg(source)
                leakage.pgg.append(provenance)
            elif isinstance(source, SimbenchSource):
                found = _collect_simbench(source)
            elif isinstance(source, Ding0Source):
                found = _collect_ding0(source)
                excluded.extend(str(d) for d in source.exclude_districts)
            elif isinstance(source, SyntheticFeederSource):
                found = _collect_synthetic(source)
            else:  # pragma: no cover - the union is closed
                raise ValueError(f"unknown grid source {source!r}")
        except Exception as exc:  # noqa: BLE001 - a missing source must not be fatal
            _log.warning(
                "multigrid: source %r yielded no grids (%s: %s).",
                getattr(source, "kind", source),
                type(exc).__name__,
                exc,
            )
            continue
        _log.info(
            "multigrid: source %r contributed %d grid(s).", source.kind, len(found)
        )
        candidates.extend(found)

    seen: set[str] = set()
    unique: list[_Candidate] = []
    for candidate in candidates:
        if candidate.key in seen:
            _log.warning("multigrid: duplicate grid key %r dropped.", candidate.key)
            continue
        seen.add(candidate.key)
        unique.append(candidate)
    leakage = leakage.model_copy(
        update={"ding0_excluded_districts": tuple(dict.fromkeys(excluded))}
    )
    return unique, leakage


# =============================================================================
# Per-grid scenario template + generation
# =============================================================================
def scenario_config(grid, config: MultiGridGenerationConfig, seed: int):
    """The shared RANDOMIZED sampling template, instantiated for one grid.

    Every member is excited the SAME way — the calibrated state-estimation recipe of
    :func:`injection_hse.data.scenarios.se_random_scenario_config` (correlated load levels with a small
    per-phase unbalance, a slack-voltage draw, IEC 61000-3-2-referenced emission with
    per-order phase diversity, and the PV inverter signature) — so differences between
    members are differences of TOPOLOGY, not of how hard they were driven. The corpus
    config overrides only the knobs it declares: the load-scale range, the two
    correlations and the emission-fraction range.

    Correlated draws matter across a corpus in particular: independent loads make the
    aggregate demand of a feeder concentrate (``1/sqrt(N)`` over a few hundred sites), so
    every scenario sits at the same operating point, while ONE irradiance scale for the
    whole inverter fleet collapses the generation to a single degree of freedom that any
    boundary measurement reads off directly.

    Returns
    -------
    injection_hse.data.scenarios.EmissionLawScenarioConfig
        A config whose ``seed`` is this grid's child seed.
    """
    from injection_hse.data.scenarios import se_random_scenario_config

    return se_random_scenario_config(
        grid,
        orders=config.orders,
        n_samples=int(config.n_scenarios),
        seed=int(seed),
        method=config.sampling,
        load_scale=config.load_scale,
        load_correlation=config.load_correlation,
        spectrum_fraction=config.spectrum_fraction,
        pv_correlation=config.pv_correlation,
    )


def coherent_scenario_config(
    grid, config: MultiGridGenerationConfig, seed: int, *, mode_bank_seed: int
):
    """The shared COHERENT sequence template, instantiated for one grid.

    The sequence counterpart of :func:`scenario_config`, from the same calibrated recipe
    (:func:`injection_hse.data.scenarios.se_coherent_scenario_config`): the fundamental operating point
    is drawn once per sequence and held across the ``T`` steps, while the harmonic
    spectrum is owned by the coherent machinery — either the device-class COMPOSITION or
    the per-device fingerprint bank, selected by ``config.coherent.mode``. The injected
    orders are the run's ``orders`` above 1.

    Parameters
    ----------
    grid:
        The member grid (read only for its appliance mix).
    config:
        The generation config; its ``coherent`` section must be set.
    seed:
        This grid's coherent seed (the temporal Markov / AR(1) / operating-point streams).
    mode_bank_seed:
        This grid's fingerprint-bank seed — deliberately separate from ``seed`` so the device
        signatures and the temporal realization can be varied independently.

    Returns
    -------
    injection_hse.data.scenarios.CoherentSpectrumConfig
        A config reproducing this member's sequence dataset from ``(config, seed)``.
    """
    from injection_hse.data.scenarios import se_coherent_scenario_config

    coherent = config.coherent
    if coherent is None:
        raise ValueError(
            "coherent_scenario_config needs MultiGridGenerationConfig.coherent to be set."
        )
    return se_coherent_scenario_config(
        grid,
        orders=config.orders,
        n_scenarios=int(coherent.n_scenarios),
        n_steps=int(coherent.n_steps),
        seed=int(seed),
        mode=coherent.mode,
        step_size_s=float(coherent.step_size_s),
        start_time=coherent.start_time,
        n_modes=int(coherent.n_modes),
        dwell=float(coherent.dwell),
        mode_bank_seed=int(mode_bank_seed),
        activity_scale=float(coherent.activity_scale),
        behavioral_coupling=float(coherent.behavioral_coupling),
        cloud_coupling=float(coherent.cloud_coupling),
        load_scale=config.load_scale,
        load_correlation=config.load_correlation,
        pv_correlation=config.pv_correlation,
    )


def _child_seeds(seed: int, count: int, *, streams: int = 1) -> list[tuple[int, ...]]:
    """``count`` independent per-grid seed TUPLES of the run seed (``SeedSequence``).

    Each member gets one child sequence and the first ``streams`` words of it, so a member's
    tasks draw from independent streams that do not shift when another task is added: word 0
    is the randomized snapshots, word 1 the coherent sequences, word 2 the fingerprint bank.
    ``generate_state`` is prefix-stable (``generate_state(3)[0] == generate_state(1)[0]``),
    so a snapshot-only dataset keeps exactly the seeds it would have with the coherent task
    added. Another per-member stream must take the NEXT WORD of the same child; spawning
    more children or reordering them shifts every existing member's seed.
    """
    import numpy as np

    children = np.random.SeedSequence(int(seed)).spawn(int(count))
    return [
        tuple(int(word) % (2**31 - 1) for word in child.generate_state(int(streams)))
        for child in children
    ]


class _Task(BaseModel):
    """One grid's generation job — picklable, so it crosses a process-pool boundary."""

    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str
    source: str
    cell: Optional[int]
    #: Seed of the RANDOMIZED snapshots.
    seed: int
    #: Seed of the COHERENT sequences (an independent word of the same child).
    coherent_seed: int
    #: Seed of this member's fingerprint (mode) bank.
    mode_bank_seed: int
    grid_json: str
    out_dir: str
    config: MultiGridGenerationConfig


def _init_worker() -> None:
    """Pool-worker setup: ONE torch thread per worker (the pool provides the parallelism).

    Deliberately a pool initializer rather than part of :func:`_generate_one`, which also runs
    IN THE CALLER's process on the serial path — permanently shrinking that process's torch
    thread pool is an invisible side effect on everything the caller does afterwards, and it
    changes its numerics too (a BLAS reduction order depends on the thread count, so a
    tolerance-tight iterative solve can land on the other side of its convergence threshold).
    """
    import torch

    torch.set_num_threads(1)


def _generate_one(task: "_Task") -> dict:
    """Generate + persist one member grid; return a manifest row (or a failure record).

    Runs in a worker process (or, for a single worker, in the caller's): it re-parses the grid
    from JSON, writes ``grid.json`` beside the dataset, solves the scenario batch(es), and
    verifies that the persisted grid reloads to the SAME row layout (a dataset whose grid does
    not round-trip is unusable — the loader would refuse it later, far from the cause). With a
    coherent section the member's sequence dataset is solved on the SAME grid into
    ``coherent/``; a failure in EITHER dataset drops the whole member from the manifest, so
    no dataset is ever listed without its counterpart. Leaves no process-global state behind.
    """
    import torch

    from pgml.schemas.grid_schema import Grid
    from pgml.scenarios import write_dataset
    from pgml.topology import layout_fingerprint

    from .scenarios import GENERATION_PROVENANCE, run_scenarios

    try:
        grid = Grid.model_validate_json(task.grid_json)
        from injection_hse._switches import materialize_estimator_grid

        grid = materialize_estimator_grid(grid)
        out_dir = Path(task.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        grid_path = out_dir / GRID_FILE
        grid_path.write_text(grid.model_dump_json(), encoding="utf-8")
        reloaded = Grid.model_validate_json(grid_path.read_text(encoding="utf-8"))
        fingerprint = layout_fingerprint(grid)
        if layout_fingerprint(reloaded) != fingerprint:
            raise ValueError(
                "the persisted grid.json does not reload to the same node-phase row layout."
            )

        config = task.config
        orders = [int(o) for o in config.orders]
        solve = dict(
            calculation="harmonic",
            harmonic_orders=orders,
            symmetry=config.symmetry,
            dtype=getattr(torch, config.dtype),
            device=torch.device("cpu"),
            chunk_size=config.chunk_size,
        )
        result = run_scenarios(grid, scenario_config(grid, config, task.seed), **solve)
        write_dataset(
            result, out_dir, layout=config.layout, provenance=GENERATION_PROVENANCE
        )
        row = {
            "key": task.key,
            "split": "train",  # assigned after collection (grid-level split)
            "source": task.source,
            "kind": task.kind,
            "n_nodes": len(grid.nodes),
            "n_rows": _n_rows(grid),
            "n_scenarios": int(result.v.shape[0]),
            "n_failed": len(result.failed_states),
            "cell": task.cell,
            "layout_fingerprint": fingerprint,
        }
        if config.coherent is not None:
            coherent = run_scenarios(
                grid,
                coherent_scenario_config(
                    grid,
                    config,
                    task.coherent_seed,
                    mode_bank_seed=task.mode_bank_seed,
                ),
                **solve,
            )
            write_dataset(
                coherent,
                out_dir / COHERENT_DIR,
                layout=config.layout,
                provenance=GENERATION_PROVENANCE,
            )
            row.update(
                n_coherent_scenarios=int(coherent.v.shape[0]),
                n_coherent_failed=len(coherent.failed_states),
                n_steps=int(coherent.v.shape[1]),
            )
        return row
    except Exception as exc:  # noqa: BLE001 - one bad grid must not kill the run
        return {
            "failure": {
                "key": task.key,
                "source": task.source,
                "error": f"{type(exc).__name__}: {exc}",
            }
        }


def _assign_splits(rows: list[dict], *, val_fraction: float, seed: int) -> None:
    """Assign ``train``/``val``/``test`` per GRID, in place.

    Real grids (``simbench``/``ding0``) are the TEST split — the held-out question is whether
    an estimator trained on synthesized topologies transfers to measured ones. The generated
    grids split into train/val by ARCHIVE CELL when the export provides one: whole cells go
    to validation, so the validation topologies come from behavior regions absent from
    training (splitting grids individually would put near-duplicates of a training grid into
    validation and overstate generalization). Without cell information the split falls back
    to a seeded shuffle over grids.
    """
    import random

    for row in rows:
        if row["kind"] in REAL_SOURCE_KINDS:
            row["split"] = "test"

    generated = [row for row in rows if row["kind"] not in REAL_SOURCE_KINDS]
    if not generated or val_fraction <= 0.0:
        for row in generated:
            row["split"] = "train"
        return

    rng = random.Random(int(seed))
    cells = sorted({row["cell"] for row in generated if row["cell"] is not None})
    if cells and len(cells) > 1:
        shuffled = list(cells)
        rng.shuffle(shuffled)
        n_val = max(1, round(val_fraction * len(shuffled)))
        val_cells = set(shuffled[:n_val])
        for row in generated:
            in_val = row["cell"] is not None and row["cell"] in val_cells
            row["split"] = "val" if in_val else "train"
        return

    order = list(range(len(generated)))
    rng.shuffle(order)
    n_val = max(1, round(val_fraction * len(order))) if len(order) > 1 else 0
    val_rows = set(order[:n_val])
    for i, row in enumerate(generated):
        row["split"] = "val" if i in val_rows else "train"


def generate_multigrid_dataset(
    config: MultiGridGenerationConfig,
    *,
    workers: int = 1,
    validate: bool = True,
    reference=None,
) -> MultiGridManifest:
    """Build a complete multi-grid dataset from a configuration; return its manifest.

    Collects the member grids, generates one scenario dataset per grid (each in its own
    directory, with the grid persisted beside it) — plus, with a ``coherent`` section, that
    member's sequence dataset under ``coherent/`` — assigns the grid-level split, and
    writes ``manifest.json`` at ``config.output_root``. A grid that fails to generate is
    recorded in the manifest's ``failures`` and the run continues.

    Unless ``validate=False``, the finished corpus is then checked by
    :func:`injection_hse.data.validate.validate_corpus`: every member dataset against the norms and
    its own configuration's claims, plus the train/val/test split envelopes. The verdict is
    logged, summarized in the manifest's ``validation`` block, and written in full beside
    it — a corpus generated in a dead activity band or with a dead harmonic order says so
    on disk instead of surfacing days later as a training result nobody can explain.

    Parameters
    ----------
    config:
        The generation configuration (see :class:`MultiGridGenerationConfig`).
    workers:
        Worker PROCESSES for the per-grid generation (``1`` = in-process, the default). See
        :func:`_run_tasks` for the start method and why it is not a plain ``fork``.
    validate:
        Run (and record) the corpus validation after generation.
    reference:
        Optional expectations YAML for the validation's measured-population layer
        (``run/configs/expected_ranges.yaml``).

    Returns
    -------
    MultiGridManifest
        The written manifest (also on disk at ``<output_root>/manifest.json``).
    """
    from pgml.schemas import SCHEMA_VERSION
    from pgml.scenarios import config_hash, generation_provenance

    from .scenarios import GENERATION_PROVENANCE

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidates, leakage = collect_grids(config)
    if not candidates:
        _log.warning("multigrid: no grids collected; writing an empty manifest.")

    # three streams per member: the snapshots, the coherent sequences, the fingerprint bank
    seeds = _child_seeds(config.seed, max(1, len(candidates)), streams=3)
    pinned_bank = None if config.coherent is None else config.coherent.mode_bank_seed
    tasks = [
        _Task(
            key=candidate.key,
            kind=candidate.kind,
            source=candidate.source,
            cell=candidate.cell,
            seed=words[0],
            coherent_seed=words[1],
            mode_bank_seed=words[2] if pinned_bank is None else int(pinned_bank),
            grid_json=candidate.grid.model_dump_json(),
            out_dir=str(root / GRIDS_DIR / candidate.key),
            config=config,
        )
        for candidate, words in zip(candidates, seeds)
    ]

    results = _run_tasks(tasks, workers=workers)
    rows = [r for r in results if "failure" not in r]
    failures = [GridFailure(**r["failure"]) for r in results if "failure" in r]
    for failure in failures:
        _log.warning("multigrid: %s failed (%s).", failure.key, failure.error)

    _assign_splits(rows, val_fraction=config.val_fraction, seed=config.seed)
    manifest = MultiGridManifest(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        seed=int(config.seed),
        orders=tuple(int(o) for o in config.orders),
        config=config,
        grids=[GridEntry(**row) for row in rows],
        failures=failures,
        leakage=leakage,
        config_hash=config_hash(config),
        se_preset_version=GENERATION_PROVENANCE["se_preset_version"],
        device_library_version=GENERATION_PROVENANCE["device_library_version"],
        **generation_provenance(),
    )
    manifest.write(root)
    if validate:
        manifest = _validate_corpus(manifest, root, reference=reference)
    _log.info(
        "multigrid: %d grid(s) written to %s (train %d / val %d / test %d, %d failed)%s.",
        len(manifest.grids),
        root,
        len(manifest.split("train")),
        len(manifest.split("val")),
        len(manifest.split("test")),
        len(manifest.failures),
        (
            ""
            if config.coherent is None
            else f", each with {config.coherent.n_scenarios} coherent sequence(s) "
            f"of {config.coherent.n_steps} step(s)"
        ),
    )
    return manifest


def _validate_corpus(
    manifest: MultiGridManifest, root: Path, *, reference=None
) -> MultiGridManifest:
    """Validate the written corpus, record the summary in the manifest, rewrite it.

    Best-effort: a validation that cannot run (an unreadable member, a missing optional
    dependency) is logged and leaves the generated corpus untouched — the data is on disk
    either way, and the report is a diagnostic, not a generation step.
    """
    from .validate import REPORT_NAME, validate_corpus

    try:
        report = validate_corpus(root, reference=reference)
    except Exception as exc:  # noqa: BLE001 - never lose a generated corpus to its report
        _log.warning(
            "multigrid: corpus validation skipped (%s: %s).", type(exc).__name__, exc
        )
        return manifest
    failed_members = sorted(
        name for name, member in report.members.items() if member.failed
    )
    manifest = manifest.model_copy(
        update={
            "validation": {
                "verdict": report.verdict,
                "n_fail": len(report.failures),
                "n_warn": len(report.warnings),
                "failed_members": failed_members,
                "report": REPORT_NAME,
            }
        }
    )
    manifest.write(root)
    log = _log.warning if report.verdict != "OK" else _log.info
    log("multigrid: corpus validation %s", report.summary_line())
    for check in report.failures[:10]:
        _log.warning("multigrid:   FAIL %s %s", check.quantity, check.reason)
    return manifest


def _run_tasks(tasks: list["_Task"], *, workers: int) -> list[dict]:
    """Run the per-grid jobs serially or over a process pool.

    The pool uses the ``forkserver`` start method where the platform offers it. A plain
    ``fork`` inherits the parent's torch/OpenMP thread state and deadlocks a worker that
    allocates while a lock was held at fork time — which the parent has always done here,
    since it just built and serialized the grids; ``spawn`` avoids that but re-imports the
    caller's ``__main__`` in every worker (fatal under a test runner or any script with side
    effects). ``forkserver`` has neither problem: the server is launched fresh and forks the
    workers from a clean, single-threaded state. Each worker is initialized to ONE torch thread
    (:func:`_init_worker`); the serial path runs in the caller's process and leaves its thread
    settings alone.
    """
    if workers <= 1 or len(tasks) <= 1:
        return [_generate_one(task) for task in tasks]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    available = multiprocessing.get_all_start_methods()
    method = "forkserver" if "forkserver" in available else "spawn"
    context = multiprocessing.get_context(method)
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=context, initializer=_init_worker
    ) as pool:
        return list(pool.map(_generate_one, tasks))


def dataset_dir(root, key: str) -> Path:
    """Directory of one member's RANDOMIZED dataset (``<root>/grids/<key>``)."""
    return Path(root) / GRIDS_DIR / key


def coherent_dir(root, key: str) -> Path:
    """Directory of one member's COHERENT dataset (``<root>/grids/<key>/coherent``)."""
    return dataset_dir(root, key) / COHERENT_DIR


def task_dataset_dir(root, key: str, task: str = "random") -> Path:
    """Directory of one member's dataset for a TASK (``"random"`` | ``"coherent"``).

    The task switch every loader goes through: ``"random"`` is the member directory itself
    (the snapshots), ``"coherent"`` its ``coherent/`` subdirectory (the sequences).
    """
    if task == "random":
        return dataset_dir(root, key)
    if task == "coherent":
        return coherent_dir(root, key)
    raise ValueError(f"task must be 'random' or 'coherent'; got {task!r}.")


def read_grid(root, key: str):
    """Reload a member's persisted ``pgml.Grid`` (``<root>/grids/<key>/grid.json``)."""
    from pgml.schemas.grid_schema import Grid

    path = dataset_dir(root, key) / GRID_FILE
    return Grid.model_validate_json(path.read_text(encoding="utf-8"))


def _example() -> MultiGridGenerationConfig:
    """A representative configuration: synthesized grids to train on, real ones to test on.

    Both curriculum tasks over the harmonic orders 1..13 (odd): randomized snapshots per
    member plus a coherent device-composition sequence set. The generated TRAIN source drops
    the few-node stubs (``min_nodes``) while the real TEST sources keep theirs, so the training
    distribution is trimmed without narrowing the held-out question.
    """
    return MultiGridGenerationConfig(
        output_root="data/pgl/multigrid",
        n_scenarios=512,
        orders=(1, 3, 5, 7, 9, 11, 13),
        coherent=CoherentSequenceConfig(n_scenarios=128, n_steps=8),
        sources=[
            PggExportSource(
                export_dir="data/pgg/export",
                max_grids=64,
                min_nodes=10,
                max_nodes=200,
            ),
            SimbenchSource(max_grids=8, max_nodes=200),
            Ding0Source(max_grids=8, max_nodes=200, exclude_districts=("1690",)),
        ],
    )


__all__ = [
    "MultiGridGenerationConfig",
    "CoherentSequenceConfig",
    "GridSourceConfig",
    "PggExportSource",
    "SimbenchSource",
    "Ding0Source",
    "SyntheticFeederSource",
    "MultiGridManifest",
    "GridEntry",
    "GridFailure",
    "LeakageRecord",
    "PggProvenance",
    "collect_grids",
    "generate_multigrid_dataset",
    "scenario_config",
    "coherent_scenario_config",
    "dataset_dir",
    "coherent_dir",
    "task_dataset_dir",
    "read_grid",
    "MANIFEST_NAME",
    "GRID_FILE",
    "GRIDS_DIR",
    "COHERENT_DIR",
]


def _cli(argv=None) -> None:
    """``python -m injection_hse.data.multigrid`` — generate a dataset, or print the config contract."""
    import argparse

    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m injection_hse.data.multigrid",
        description="Generate a reproducible multi-grid state-estimation dataset.",
    )
    parser.add_argument("--config", help="Generation config (YAML or JSON).")
    parser.add_argument(
        "--json-schema", action="store_true", help="Print the config JSON Schema."
    )
    parser.add_argument(
        "--example", action="store_true", help="Print an example config as YAML."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for per-grid generation.",
    )
    parser.add_argument(
        "--output-root", default=None, help="Override the config's output_root."
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="Validate the finished corpus (default).",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip the corpus validation.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Expectations YAML for the validation's measured-population layer "
        "(run/configs/expected_ranges.yaml).",
    )
    args = parser.parse_args(argv)

    if args.json_schema:
        print(json.dumps(MultiGridGenerationConfig.model_json_schema(), indent=2))
        return
    if args.example:
        print(yaml.safe_dump(_example().model_dump(mode="json"), sort_keys=False))
        return
    if not args.config:
        parser.error("--config is required (or use --json-schema / --example).")

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = MultiGridGenerationConfig.model_validate(raw)
    if args.output_root:
        config = config.model_copy(update={"output_root": args.output_root})
    generate_multigrid_dataset(
        config,
        workers=args.workers,
        validate=args.validate,
        reference=args.reference,
    )


if __name__ == "__main__":  # `python -m injection_hse.data.multigrid --config <yaml>`
    # Delegate to the PACKAGE module: ``python -m`` executes this file a second time under
    # the name ``__main__``, and a worker process resolves the pickled job class by module
    # name — which must be ``injection_hse.data.multigrid``, not ``__main__``.
    from injection_hse.data.multigrid import _cli as _package_cli

    _package_cli()
