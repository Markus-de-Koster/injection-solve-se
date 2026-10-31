"""End-to-end harmonic state-estimation experiment: two datasets, two tasks, one model.

WHAT THIS DEMONSTRATES
----------------------
The complete state-estimation workflow on the three-phase CIGRE LV benchmark, from data
generation through a two-task measurement-scarcity curriculum to a full evaluation report:

1. Build the grid (3-phase CIGRE LV + synthesized Carson geometry + PV), or load a STORED
   one with ``--grid`` (a real network — its installed measurement devices then become the
   sensor placement, with their current channels).
2. Generate the self-describing parquet datasets:
   - RANDOMIZED (the snapshot pass, snapshots ``[B, H, N]``) via ``ScenarioConfig`` — independent
     per-order injection draws, the harmonic-flow surrogate's "cover the space" data
     (timeless);
   - COHERENT (the sequence pass, sequences ``[B, T, H, N]``) via ``CoherentSpectrumConfig`` — by default
     a statistical device-class COMPOSITION whose per-step device activity moves the
     fundamental power AND the injected harmonic spectrum jointly (the realistic device-pattern
     SE data, with an absolute time axis + per-class attribution labels). ``--coherent-mode
     fingerprint`` selects the earlier per-device mode-bank fingerprint instead (for ablations);
   - COMPOSED longer window (the profiled-sequence pass, optional via ``--n-profile-scenarios``) — the same
     composition over a wider, hourly-stepped diurnal window.
3. Train ONE temporal estimator across the datasets with WEIGHT CONTINUATION: the
   measurement-scarcity curriculum runs full -> sparse on the randomized data (the snapshot pass), then
   the same module (weights carried over) continues random -> the 3-sensor placement on the
   composed coherent data (the sequence pass), and optionally a longer-window placement finetune (the profiled-sequence pass).
   Periodic time features are enabled whenever composed data participates (timeless stages feed
   the all-zero time vector). A shared history collector spans all task fits. An optional final
   ``placement_anneal`` stage (``--anneal-epochs``) cosine-anneals the placement lr from a
   plateau.
4. Persist a manifest + epoch-indexed history so the evaluation runs standalone, then
   (``--evaluate``) write the full plot set — reporting BOTH the validation and the held-out
   TEST split metrics.

SWITCHABLE TOPOLOGY (``--topologies``)
--------------------------------------
A grid whose switches can be operated is several networks. ``--topologies all`` trains ONE
estimator across every switch combination that leaves the grid connected (``--switches``
names the switchable branches; a comma-separated subset trains on some configurations and
leaves the rest for a generalization test). Then: one dataset per configuration (the sizes
above are the run TOTAL, split across them, each with its own seed stream), each sample
decoded with ITS configuration's admittance and branch-current map, and the switch states fed
to the model (``--topology-conditioning``). The run is built on the UNION topology — every
switchable branch closed — so the graph, the placement and the metered half-edges are the
same whichever switches a configuration opens.

NETWORK-MODEL ERROR (``--grid-perturbation``)
---------------------------------------------
The estimator decodes with the grid it is given; a real network differs from its description.
``--grid-perturbation <yaml>`` (template ``run/configs/pgl_grid_perturbation.yaml``) generates
data on perturbed deep copies of the grid — line R / X / length, transformer R / X, source
impedance, a shared and a per-element factor each — while ``grid.json``, the graph and the
decode stay nominal (:mod:`injection_hse.data.scenarios.perturbation`). Each copy solves an equal block
of the SAME scenario draw; every sample records its copy (``grid_copy``) and ``meta.json``
each copy's factors. The regimes of a study:

- train nominal, test perturbed: ``--perturb-test heldout``;
- train on a perturbed ensemble, test on held-out perturbations: ``--perturb-train perturbed
  --perturb-test heldout`` (the test copies come from a separate seed pool);
- one systematic perturbation in both: ``--perturb-copies 1 --perturb-train perturbed
  --perturb-test train``.

The perturbed test set is its own dataset (``dataset_<eval kind>_perturbed_test``, a new
scenario seed, never read in training) and is scored per copy into
``evaluation/metrics_perturbed_test_per_copy.csv``; a perturbed training dataset's own test
split is scored per copy into ``metrics_test_per_copy.csv``. ``--load-shunt`` (with
``--load-shunt-basis``) sets the harmonic device shunt of the generation and of the decode
together.

RECIPE v4 (grid model and scenario recipe)
------------------------------------------
- ``--device-capacitance run/configs/pgl_device_capacitance.yaml``: one seeded wye capacitive
  shunt per load and PV generator, sized per kW of rated power (EMC capacitors, inverter
  filters; ranges and sources in the file). Part of ``grid.json`` and so of the estimator's
  ``Y(h)``.
- ``--source-impedance cigre-tb575``: the published benchmark supply (100 MVA, R/X = 1).
- ``--scenario-config run/configs/pgl_se_scenario_v4.yaml``: per-device emission
  persistence and normal emission angles (see the file's header, which also gives the
  ``--background-*`` example for the upstream distortion).

The two arms of the nominal experiment::

    # v4-consistent: the device shunt on the nameplate basis in data and decode
    pixi run python run/examples/injection_hse/train_se_full_workflow.py \
        --orders 1,3,5,7,9,11,13,15,17,19 \
        --scenario-config run/configs/pgl_se_scenario_v4.yaml \
        --device-capacitance run/configs/pgl_device_capacitance.yaml \
        --source-impedance cigre-tb575 --load-shunt opendss --load-shunt-basis nameplate

    # v4-realistic: operating-point shunt in the data, nameplate in the decode
    ... the same with --load-shunt-basis operating_point

RECIPE v5 (``run/configs/pgl_se_scenario_v5.yaml``) is v4 at every integer order 1..50, odd and
even, as a power-quality analyser resolves them: the same flags with
``--orders "$(seq -s, 1 50)"`` (the file's header gives the spectra rules, the reduced h9 and
the background example up to h49).

An operating-point shunt makes ``Y(h)`` scenario-dependent, so pgml assembles a
``[chunk, T, H, N, N]`` system per chunk; the driver caps the chunk to fit
``--chunk-memory-fraction`` of the device's free memory. Every setting (capacitance draws,
source impedance, the scenario file's SHA-256) is recorded in ``meta.json``,
``data_generation.yaml`` and the manifest.

By default ALL harmonic orders 1..20 (even AND odd) are solved and the injected orders are
2..20; ``--orders`` restricts both (e.g. ``--orders 1,3,5,7,9``) — the memory footprint and
the feature width scale with the order count.

REPRODUCIBILITY
- Everything flows from two ``injection_hse.ExperimentConfig`` objects (the snapshot pass / the sequence pass, sharing the
  seed, orders, model, measurement, optimizer) — both persisted as YAML next to the outputs.
- The simulations are serializable ``ScenarioConfig`` / ``CoherentSpectrumConfig`` (+ seed);
  the dataset directories round-trip via ``read_dataset``.
- The fitted normalizer + config travel with every stage checkpoint; the final placement
  checkpoint (recorded in the manifest) is the evaluation target.

RUN
---
::

    # tiny CPU smoke run (defaults are smoke-sized); evaluation runs automatically:
    pixi run -e cpu python run/examples/injection_hse/train_se_full_workflow.py

    # a real run on the cluster (GPU), larger — measurement-anchored injection decode at
    # FIXED per-stage weights, with half the state supervision on the predicted injections:
    pixi run python run/examples/injection_hse/train_se_full_workflow.py \
        --n-random 50000 --n-scenarios 4000 --n-steps 8 --epochs 200 \
        --batch-size 256 --device auto --measure-currents \
        --anchor-schedule 'full:0,random_50:1,random_25:5,placement:10' \
        --anchor-fixed --injection-loss-split 0.5

    # network-model error: train on 8 perturbed copies, test on 4 unseen ones, with the
    # device shunt on the nameplate basis in data and decode:
    pixi run python run/examples/injection_hse/train_se_full_workflow.py \
        --grid-perturbation run/configs/pgl_grid_perturbation.yaml \
        --perturb-train perturbed --perturb-test heldout --load-shunt opendss

    # a real grid across its switch configurations:
    pixi run python run/examples/injection_hse/train_se_full_workflow.py \
        --grid data/grids/my_grid/my_grid.json --topologies all \
        --orders 1,3,5,7,9 --measure-currents --n-random 200000 --n-scenarios 20000

    # (re)evaluate a finished run later — self-contained from its directory (reloads the
    # grid the run wrote), no retraining:
    pixi run python -m injection_hse.evaluation data/se_full_workflow
    # equivalently, via this script:
    pixi run python run/examples/injection_hse/train_se_full_workflow.py data/se_full_workflow --eval-only

PRODUCTION SIZES (vs the smoke defaults below)
- ``--n-random``     50k+ Sobol snapshots   (smoke: 32)
- ``--n-scenarios``  2k-8k coherent sequences (smoke: 16)
- ``--n-steps``      8-16 steps per sequence  (smoke: 3)
- ``--epochs``       100-300 max per stage    (smoke: 2)
- ``--batch-size``   256-1024                 (smoke: 16)

If data generation OOMs on the GPU at a large ``--n-random`` / ``--n-scenarios``, add
``--chunk-size`` (e.g. ``--chunk-size 20000``): the scenario batch is solved in slices and
concatenated, for an identical result that fits memory.

Outputs (datasets, checkpoints, history, manifest, evaluation/) + a summary are printed at
the end. Runs on any architecture (device auto-selected). Data generation and TRAINING have
independent precision: ``--dtype`` sets the generation precision (default ``complex128`` for
label accuracy) and ``--train-dtype`` the model precision (default ``complex64`` — a fast
float32 model on the high-precision labels; ``complex128`` for a full float64 study).
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

# Reduce CUDA allocator fragmentation BEFORE torch initialises the CUDA context: large-batch
# harmonic solves + training leave "reserved but unallocated" gaps that expandable segments
# can reuse, so a batch that just barely OOMs often fits. Harmless on CPU; ``setdefault`` keeps
# any value the user already exported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import yaml  # noqa: E402

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    EncodingConfig,
    ExperimentConfig,
    GNNConfig,
    LossConfig,
    HarmonicWeightConfig,
    MaskStageConfig,
    MeasurementConfig,
    InstrumentErrorConfig,
    NoiseConfig,
    NormalizationConfig,
    OptimConfig,
    PhysicsConfig,
    TemporalConfig,
    TrainConfig,
    PhaseCouplingConfig,
    DNNConfig,
    GraphormerConfig,
)
from injection_hse.curriculum import (
    anneal_stage,
    apply_anchor_schedule,
    apply_stage_weights,
    parse_anchor_schedule,
    parse_orders,
    parse_stage_weights,
    resume_stage_split,
    snapshot_stages,
    sequence_stages,
    profiled_sequence_stages,
)
from injection_hse.workflow import (
    PREFLIGHT_SAFE_FRACTION,
    TrainingPass,
    TrainingRunConfig,
    build_training_setup,
    launch_rank,
    preflight_training,
    run_training,
)
from injection_hse.data.scenarios import (
    DEVICE_LIBRARY_VERSION,
    HIGH_ACTIVITY_START_TIME,
    SE_PRESET_VERSION,
)

try:  # shared example plumbing (run as a script OR imported as a package)
    from run.examples.injection_hse._common import (
        DTYPES,
        configure_runtime,
        output_dir,
        parse_phases,
        resolve_device,
    )
except ModuleNotFoundError:  # script run: run/examples/injection_hse is on sys.path
    from _common import (
        DTYPES,
        configure_runtime,
        output_dir,
        parse_phases,
        resolve_device,
    )

# The 3 CIGRE LV sensor nodes, one per feeder (residential / commercial / industrial).
SENSOR_NODES = ["Bus R4", "Bus C5", "Bus I2"]

# Absolute anchor for the COMPOSED coherent sequences' diurnal / weekly / seasonal phases
# (the ``--profile-start`` default AND the fallback for callers that don't set it, e.g. the
# ablation driver's argparse namespace). The calibrated high-activity band: a sequence window
# of a few hours anchored at midnight sits in the dead band of every activity preset.
_DEFAULT_START_TIME = HIGH_ACTIVITY_START_TIME


# =============================================================================
# multi-GPU launch plumbing (SLURM srun / torchrun; single-process = rank 0)
# =============================================================================
def wait_for_sentinel(path: Path, *, description: str) -> None:
    """Block until the rank-0 process has written ``path`` (filesystem barrier).

    Non-zero ranks cannot use a ``torch.distributed`` barrier before the first
    ``Trainer.fit`` initializes the process group, so dataset generation is synchronized
    through a sentinel file on the (shared) run directory instead. Polls forever — if
    rank 0 dies before writing, the job's time limit ends the wait.
    """
    import time

    waited = 0.0
    while not path.exists():
        time.sleep(2.0)
        waited += 2.0
        if waited % 120.0 == 0.0:
            print(
                f"[rank {launch_rank()}] waiting for {description} "
                f"({int(waited)} s) ...",
                flush=True,
            )


def _resolve_config_path(path) -> Path:
    """Resolve a ``--scenario-config`` path robustly, tolerating the caller's CWD.

    Tries the path as given, then — for a relative path — relative to the repository root, so
    a run launched from anywhere (e.g. a SLURM job whose working directory is the run dir, not
    the checkout) still finds a repo-relative ``run/configs/...`` path. Raises a
    ``FileNotFoundError`` naming every location tried and where the templates live, so a typo
    (e.g. ``run/examples/config/`` for ``run/configs/``) is actionable rather than cryptic.
    """
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / p)
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"--scenario-config: no such file {path!r}. Tried:\n  {tried}\n"
        "Scenario-config templates live under run/configs/ (e.g. "
        "run/configs/pgl_se_scenario.yaml)."
    )


def load_scenario_config(path):
    """Load an optional scenario-DEVIATION YAML -> ``(random_cfg | None, coherent_cfg | None)``.

    The generators build from the calibrated preset (:mod:`injection_hse.data.scenarios.presets`); this
    file is how a run deviates from it deliberately. It may carry a ``random:`` section (a
    :class:`injection_hse.data.scenarios.EmissionLawScenarioConfig` for the randomized snapshot dataset) and/or a
    ``coherent:`` section (a :class:`injection_hse.data.scenarios.CoherentSpectrumConfig` for the sequence passes);
    a section present REPLACES the preset's shape — the distributions, per-spec
    ``correlation`` + shared ``factors``, ``symmetry``/``imbalance`` and
    ``harmonic_reference``. The SIZES (``n_samples``/``n_scenarios``/``seed``/``orders``)
    are injected from the CLI, so one file scales test -> full unchanged; any size/orders
    written in the YAML are placeholders. Example: ``run/configs/pgl_se_scenario.yaml``.
    """
    if path is None:
        return None, None
    import yaml

    from injection_hse.data.scenarios import CoherentSpectrumConfig, EmissionLawScenarioConfig

    doc = yaml.safe_load(_resolve_config_path(path).read_text())
    if not isinstance(doc, dict) or not ({"random", "coherent"} & set(doc)):
        raise SystemExit(
            f"--scenario-config {path}: expected a mapping with a 'random' and/or "
            "'coherent' section (see run/configs/pgl_se_scenario.yaml)."
        )
    rnd = (
        EmissionLawScenarioConfig.model_validate(doc["random"])
        if "random" in doc
        else None
    )
    coh = (
        CoherentSpectrumConfig.model_validate(doc["coherent"])
        if "coherent" in doc
        else None
    )
    return rnd, coh


def _emission_fraction_kw(args, name: str) -> dict:
    """``{name: (lo, hi)}`` from ``--emission-fraction lo,hi``, or ``{}`` for the preset's."""
    spec = getattr(args, "emission_fraction", None)
    if not spec:
        return {}
    lo, hi = (float(x) for x in str(spec).split(","))
    return {name: (lo, hi)}


def background_config(args):
    """The upstream harmonic background of ``--background-distortion``, or ``None``.

    ``"3:0.0025,5:0.007@-40"`` names per-order VOLTAGE distortion levels at the grid's
    source (per unit of the fundamental; ``@deg`` an order's phase in the sequence frame);
    ``--background-drift`` is the log-magnitude std of the
    slow shared drift and ``--background-sc-mva`` the source strength (a grid whose Source
    is near-ideal needs a strength well above ``V²/R_source`` for the level to appear at
    the coupling point). Applied to the randomized AND the coherent recipes, so every
    dataset of the run sits behind the same kind of supplying network
    (:class:`pgml.scenarios.BackgroundHarmonicConfig`).
    """
    spec = getattr(args, "background_distortion", None)
    if not spec:
        return None
    from pgml.scenarios import BackgroundHarmonicConfig

    levels, phases = {}, {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        order, level = item.split(":")
        if "@" in level:  # ``h:mag@deg`` carries the order's phase
            level, phase = level.split("@")
            phases[int(order)] = float(phase)
        levels[int(order)] = float(level)
    if not levels:
        return None
    return BackgroundHarmonicConfig(
        magnitude_pu=levels,
        phase_deg=phases,
        drift_std=float(getattr(args, "background_drift", 0.0) or 0.0),
        drift_phase_deg=float(getattr(args, "background_drift_phase_deg", 0.0) or 0.0),
        source_power_va=float(getattr(args, "background_sc_mva", 20.0) or 20.0) * 1e6,
    )


def scenario_recipe_identity(args) -> str:
    """One line naming the scenario recipe a run generates from.

    Either the built-in calibrated preset with its version, or the deviation file with a
    short content hash — the run log then says WHICH recipe produced the data, which the
    dataset's own ``meta.json`` config hash can be checked against later.
    """
    path = getattr(args, "scenario_config", None)
    if path is None:
        return (
            f"pgml.scenarios preset v{SE_PRESET_VERSION} "
            f"(device library v{DEVICE_LIBRARY_VERSION})"
        )
    import hashlib

    resolved = _resolve_config_path(path)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:12]
    return (
        f"{resolved} (sha256 {digest}) — a deviation from preset "
        f"v{SE_PRESET_VERSION}; device library v{DEVICE_LIBRARY_VERSION}"
    )


def _with_parameters(cfg, parameters):
    """``cfg`` with another parameter list (an emission-law config prunes its laws too)."""
    if hasattr(cfg, "with_parameters"):
        return cfg.with_parameters(parameters)
    return cfg.model_copy(update={"parameters": parameters})


def _check_harmonic_orders(cfg, injected, label):
    """Restrict every harmonic ParameterSpec to the SOLVED (injected) orders.

    A scenario template describes the WIDEST harmonic range the study covers (odd orders
    to h19, matching the measured device populations); a run restricted via ``--orders``
    injects the intersection. Specs are therefore pruned rather than rejected — one
    template scales across runs — and a per-order spec left with nothing to distort is
    dropped whole, which is what restricting the orders MEANS when the emission phase is
    specified per order. A template none of whose harmonic specs reaches the solved set
    is a typo rather than a restriction and raises. Returns the pruned config.
    """
    inj = set(injected)
    kept, notes, dropped = [], [], []
    harmonic, surviving = 0, 0
    for spec in cfg.parameters:
        if spec.is_harmonic and spec.orders is not None:
            harmonic += 1
            keep = [o for o in spec.orders if o in inj]
            extra = sorted(set(spec.orders) - inj)
            if not keep:
                dropped.append(f"{spec.name} ({extra})")
                continue
            surviving += 1
            if extra:
                notes.append(f"{spec.name}: dropped {extra}")
                spec = spec.model_copy(update={"orders": keep})
        kept.append(spec)
    if harmonic and not surviving:
        raise SystemExit(
            f"--scenario-config ({label}): no harmonic spec distorts any solved order "
            f"({sorted(inj)} from --orders) — the template addresses a different "
            "harmonic range entirely. Fix the specs or widen --orders."
        )
    if notes or dropped:
        detail = "; ".join([*notes, *(f"{d} dropped" for d in dropped)])
        print(
            f"  scenario-config ({label}): harmonic specs restricted to the solved set "
            f"{sorted(inj)} — {detail}"
        )
    return _with_parameters(cfg, kept)


# =============================================================================
# grid (reuses exactly what train_se_coherent.py uses)
# =============================================================================
def _add_pv_systems(grid, fraction: float) -> int:
    """Import ``add_pv_systems`` whether run as a script (``run/examples/`` on path) or a package."""
    try:
        from pgml.grids import add_pv_systems
    except ModuleNotFoundError:
        from pgml.grids import add_pv_systems  # pragma: no cover
    return add_pv_systems(grid, fraction=fraction)


def load_grid(path: Path):
    """Load a stored ``pgml.Grid`` JSON (a real network, e.g. a converted DSO plan).

    The alternative to the synthesized benchmark grid below: the run then trains on the grid
    exactly as stored — its own topology, its own installed measurement devices (which, unless
    ``--sensors`` overrides them, become the placement) and, with ``--topologies``, its
    switchable branches.
    """
    from pgml.schemas.grid_schema import Grid

    if not path.is_file():
        raise SystemExit(f"--grid: no such file {path}")
    return Grid.model_validate_json(path.read_text(encoding="utf-8"))


def build_grid(pv_fraction: float):
    """The 3-phase CIGRE LV + synthesized Carson geometry + PV (the SE benchmark grid)."""
    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.oracles import cigre_lv_full_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    synthesize_grid_geometry(grid)
    # Tag the aggregation-level consumer type by CIGRE feeder (R = residential,
    # C = commercial, I = industrial): the statistical device composition draws
    # each load's member-device roster from these types — untagged loads would
    # all fall back to the minimal base+electronics composition.
    from pgml.schemas.grid_schema import ConsumerType, Load

    feeder_type = {
        "R": ConsumerType.HOUSEHOLD,
        "C": ConsumerType.OFFICE,
        "I": ConsumerType.WORKSHOP,
    }
    for a in grid.appliances:
        if isinstance(a, Load) and a.consumer_type is None and a.name:
            tag = feeder_type.get(a.name.removeprefix("Load ")[:1])
            if tag is not None:
                a.consumer_type = tag
    n_pv = _add_pv_systems(grid, pv_fraction) if pv_fraction > 0 else 0
    return grid, n_pv


#: The TB 575 / pandapower CIGRE LV supply: short-circuit power and R/X of the MV source
#: (pandapower ``create_cigre_network_lv``: ``s_sc_max_mva`` 100, ``rx_max`` 1.0).
CIGRE_TB575_SOURCE = {"s_sc_va": 100e6, "r_over_x": 1.0}


def apply_source_impedance(grid, preset: str):
    """Set every in-service source's Thevenin impedance from ``--source-impedance``.

    ``"pgml"`` leaves the grid as built (pgml's import default). ``"cigre-tb575"`` sets the
    published benchmark supply, :data:`CIGRE_TB575_SOURCE`: ``|Z| = U_rated**2 / S_sc`` at
    the source node's rated voltage (4 ohm at 20 kV), split by R/X, as a diagonal per-phase
    R and L. Returns the record of what was set (``None`` for ``"pgml"``).
    """
    import math

    from pgml.schemas.grid_schema import Source

    if preset == "pgml":
        return None
    if preset != "cigre-tb575":
        raise SystemExit(f"--source-impedance: unknown preset {preset!r}.")
    nodes = {n.id: n for n in grid.nodes}
    f0 = float(grid.base_frequency_hz)
    s_sc, rx = CIGRE_TB575_SOURCE["s_sc_va"], CIGRE_TB575_SOURCE["r_over_x"]
    record = {"preset": preset, "s_sc_va": s_sc, "r_over_x": rx, "sources": []}
    for src in grid.appliances:
        if not isinstance(src, Source) or not src.in_service:
            continue
        u = float(nodes[src.node].u_rated_v)
        z = u * u / s_sc
        x = z / math.sqrt(1.0 + rx * rx)
        r = rx * x
        n = len(src.phases)
        src.resistance_ohm = [
            [r if i == j else 0.0 for j in range(n)] for i in range(n)
        ]
        src.inductance_h = [
            [x / (2.0 * math.pi * f0) if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        record["sources"].append(
            {"id": int(src.id), "u_rated_v": u, "r_ohm": r, "x_ohm": x}
        )
    return record


def load_device_capacitance(path) -> dict:
    """The ``device_capacitance:`` block of ``--device-capacitance`` (validated).

    Keys: ``seed`` (int), ``load_uf_per_kw`` (``{consumer_type | "default": [low, high]}``)
    and ``pv_uf_per_kw`` (``[low, high]``), densities in uF per kW of rated active power.
    Template: ``run/configs/pgl_device_capacitance.yaml``.
    """
    doc = yaml.safe_load(_resolve_config_path(path).read_text())
    block = doc.get("device_capacitance") if isinstance(doc, dict) else None
    if not isinstance(block, dict) or "load_uf_per_kw" not in block:
        raise SystemExit(
            f"--device-capacitance {path}: no 'device_capacitance:' block with "
            "load_uf_per_kw (see run/configs/pgl_device_capacitance.yaml)."
        )
    ranges = dict(block["load_uf_per_kw"])
    if "default" not in ranges:
        raise SystemExit(
            "--device-capacitance: load_uf_per_kw needs a 'default' range."
        )
    pv = block.get("pv_uf_per_kw")
    for name, rng in [*ranges.items(), ("pv_uf_per_kw", pv)]:
        if rng is None or len(rng) != 2 or not 0.0 < float(rng[0]) <= float(rng[1]):
            raise SystemExit(
                f"--device-capacitance: range {name} must be [low, high] with "
                f"0 < low <= high, got {rng!r}."
            )
    return {
        "seed": int(block.get("seed", 0)),
        "load_uf_per_kw": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "pv_uf_per_kw": [float(pv[0]), float(pv[1])],
    }


def add_device_capacitance(grid, config: dict) -> list[dict]:
    """Add one wye capacitive :class:`~pgml.schemas.grid_schema.ShuntAppliance` per device.

    Every in-service load and every PV generator (``consumer_type`` pv) gets a shunt at its
    node on its phases, ``C`` per phase = density x rated active power / phase count. The
    density [uF/kW] is drawn log-uniformly from the device's range
    (:func:`load_device_capacitance`) ONCE per device, from a stream keyed by the config
    seed and the device id, so it does not depend on the device order or on the other
    devices. Modifies ``grid`` in place (new appliance ids above every existing id) and
    returns one record per shunt: device, consumer type, rated kW, density, C per phase.
    A grid that already carries device-capacitance shunts is rejected, so a stored grid
    cannot receive them twice.
    """
    import numpy as np
    from pgml.schemas.grid_schema import Generator, Load, ShuntAppliance

    def _type(device):
        ctype = getattr(device, "consumer_type", None)
        return None if ctype is None else getattr(ctype, "value", str(ctype))

    if any(
        isinstance(a, ShuntAppliance) and "device_capacitance_of" in (a.tags or {})
        for a in grid.appliances
    ):
        raise SystemExit(
            "--device-capacitance: the grid already carries device capacitance (a "
            "generator's grid.json?); pass the grid without it."
        )
    next_id = 1 + max(
        [n.id for n in grid.nodes]
        + [b.id for b in grid.branches]
        + [a.id for a in grid.appliances]
    )
    records = []
    devices = sorted(
        (
            a
            for a in grid.appliances
            if a.in_service
            and (isinstance(a, Load) or (isinstance(a, Generator) and _type(a) == "pv"))
        ),
        key=lambda a: int(a.id),
    )
    for device in devices:
        kind = "load" if isinstance(device, Load) else "pv"
        ctype = _type(device)
        if kind == "pv":
            low, high = config["pv_uf_per_kw"]
        else:
            ranges = config["load_uf_per_kw"]
            low, high = ranges.get(ctype, ranges["default"])
        rng = np.random.default_rng(
            np.random.SeedSequence([int(config["seed"]), int(device.id)])
        )
        density = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        p_kw = abs(float(device.p_nom_w)) / 1e3
        n = len(device.phases)
        c_phase = density * p_kw / n * 1e-6
        grid.appliances.append(
            ShuntAppliance(
                id=next_id,
                name=f"device capacitance {device.name or device.id}",
                node=device.node,
                phases=device.phases,
                conductance_s=[0.0] * n,
                capacitance_f=[c_phase] * n,
                tags={"device_capacitance_of": str(device.id)},
            )
        )
        records.append(
            {
                "shunt_id": next_id,
                "device_id": int(device.id),
                "device": device.name,
                "kind": kind,
                "consumer_type": ctype,
                "p_rated_kw": p_kw,
                "uf_per_kw": density,
                "c_per_phase_uf": c_phase * 1e6,
            }
        )
        next_id += 1
    return records


def apply_grid_model(grid, args):
    """The grid-model options of the build (``--device-capacitance``, ``--source-impedance``).

    Applied to the grid before it is persisted, so ``grid.json`` — and with it the
    estimator's ``Y(h)`` — carries them. Returns their record (written to ``meta.json``,
    ``data_generation.yaml`` and the manifest), ``None`` when both are at their defaults.
    """
    record = {}
    source = apply_source_impedance(grid, getattr(args, "source_impedance", "pgml"))
    if source is not None:
        record["source_impedance"] = source
    path = getattr(args, "device_capacitance", None)
    if path:
        config = load_device_capacitance(path)
        shunts = add_device_capacitance(grid, config)
        record["device_capacitance"] = {
            "config": config,
            "config_file": file_fingerprint(path),
            "shunts": shunts,
        }
    return record or None


def file_fingerprint(path) -> dict:
    """``{"path", "sha256"}`` of a config file, so a run records the exact file it read."""
    import hashlib

    resolved = _resolve_config_path(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def add_time_axis_arguments(ap) -> None:
    """The per-sequence start-time and activity-calendar flags of the sequence passes."""
    ap.add_argument(
        "--start-time-range",
        default=None,
        help="START,END (ISO 8601, END exclusive): every COMPOSED sequence (the sequence "
        "pass and the profiled-sequence pass) starts at its own time drawn from this range "
        "instead of the one --profile-start window, so the data spans times of day, "
        "weekdays and seasons (e.g. 2024-01-01,2025-01-01). Default off.",
    )
    ap.add_argument(
        "--start-time-sampling",
        default="uniform",
        choices=["uniform", "stratified"],
        help="With --start-time-range: independent uniform starts, or one start per equal "
        "stratum of the range (even coverage at any sequence count).",
    )
    ap.add_argument(
        "--start-hours",
        default=None,
        help="With --start-time-range: admissible start hours LOW,HIGH of the day (UTC "
        "frame; LOW > HIGH wraps midnight).",
    )
    ap.add_argument(
        "--start-weekdays",
        default=None,
        help="With --start-time-range: admissible start weekdays, comma-separated "
        "(0 = Monday .. 6 = Sunday).",
    )
    ap.add_argument(
        "--activity-calendar",
        action="store_true",
        help="Modulate the composition's device activity by the weekly (weekend dip) and "
        "seasonal factors of injection_hse.data.scenarios.ActivityCalendar; without it the composed "
        "activity is purely diurnal.",
    )


def start_time_range(args):
    """The :class:`~injection_hse.data.scenarios.StartTimeRange` of the CLI flags, ``None`` when unset."""
    spec = getattr(args, "start_time_range", None)
    if not spec:
        if getattr(args, "start_hours", None) or getattr(args, "start_weekdays", None):
            raise SystemExit("--start-hours / --start-weekdays need --start-time-range")
        return None
    from injection_hse.data.scenarios import StartTimeRange

    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--start-time-range: START,END, got {spec!r}")
    hours = getattr(args, "start_hours", None)
    weekdays = getattr(args, "start_weekdays", None)
    return StartTimeRange(
        start=parts[0],
        end=parts[1],
        sampling=getattr(args, "start_time_sampling", "uniform"),
        hours=None if not hours else tuple(float(h) for h in hours.split(",")),
        weekdays=None if not weekdays else tuple(int(d) for d in weekdays.split(",")),
    )


def apply_time_axis_options(cfg, args, label: str):
    """``cfg`` with the CLI's per-sequence start times and activity calendar applied.

    ``--start-time-range`` replaces the config's single ``start_time`` (one window for the
    whole batch) by a start per sequence; ``--activity-calendar`` adds the weekly and
    seasonal modulation to its composition. Both apply to a timed (composed or profiled)
    config only; without either flag ``cfg`` is returned unchanged.
    """
    from injection_hse.data.scenarios import ActivityCalendar, CoherentSpectrumConfig

    rng = start_time_range(args)
    calendar = bool(getattr(args, "activity_calendar", False))
    if rng is None and not calendar:
        return cfg
    if not cfg.is_timed:
        raise SystemExit(
            f"{label}: --start-time-range / --activity-calendar need a time-anchored "
            "(composed or profiled) sequence; the fingerprint mode is timeless."
        )
    data = cfg.model_dump()
    if rng is not None:
        data.update(start_time=None, start_time_range=rng.model_dump())
    if calendar:
        if cfg.composition is None:
            raise SystemExit(f"{label}: --activity-calendar needs a composition.")
        if cfg.composition.calendar is None:
            data["composition"]["calendar"] = ActivityCalendar().model_dump()
    return CoherentSpectrumConfig.model_validate(data)


def time_axis_record(args) -> dict:
    """The per-sequence time settings of a run for the generation record (empty when off)."""
    rng = start_time_range(args)
    record = {}
    if rng is not None:
        record["start_time_range"] = rng.model_dump(mode="json")
    if getattr(args, "activity_calendar", False):
        record["activity_calendar"] = True
    return record


def prune_scenario_parameters(cfg, grid, label: str):
    """Drop parameter specs whose selector matches nothing in ``grid``.

    The scenario shapes are written for a grid population, not for one network: a spec
    varying PV output is meaningless on a feeder with no DER, and the sampler rejects an
    empty selector rather than sampling a dimension nobody reads. Dropping those specs (and
    saying so) is what lets ONE scenario configuration drive several grids.
    """
    parameters = list(getattr(cfg, "parameters", []) or [])
    if not parameters:
        return cfg
    keep = [spec for spec in parameters if spec.selector.resolve(grid)]
    dropped = [spec.name for spec in parameters if spec not in keep]
    if dropped:
        print(
            f"  {label}: dropped {dropped} — no matching component in this grid.",
            flush=True,
        )
    return _with_parameters(cfg, keep)


# =============================================================================
# switchable topology
# =============================================================================
def resolve_topology(grid, args):
    """The :class:`~injection_hse.config.TopologyConfig` this run trains across (``None`` = one topology).

    ``--topologies off`` (default) trains on the grid exactly as given. ``all`` enumerates
    every combination of the switchable branches that leaves the grid CONNECTED (a
    combination that de-energizes part of the grid has a singular harmonic system and is not
    a topology an estimator can be trained on); a comma-separated list selects a subset of
    those combinations by name (``c`` closed / ``o`` open per switch, in ``--switches``
    order — e.g. ``co,cc``), which is how a run trains on some topologies and is tested on
    the rest. ``--switches`` names the switchable branches (default: every switch in the
    grid).
    """
    from injection_hse.config import TopologyConfig, TopologyVariantConfig
    from injection_hse.data import enumerate_variants

    spec = (getattr(args, "topologies", "off") or "off").strip()
    if spec == "off":
        return None
    switches = _switch_ids(grid, getattr(args, "switches", "") or "")
    if not switches:
        raise SystemExit(
            "--topologies needs switchable branches: the grid carries no Switch "
            "(add one, or run with --topologies off)."
        )
    connected = enumerate_variants(grid, switches)
    dropped = 2 ** len(switches) - len(connected)
    if dropped:
        print(
            f"  {dropped} switch combination(s) skipped: they de-energize part of the grid.",
            flush=True,
        )
    if spec != "all":
        wanted = [t.strip() for t in spec.split(",") if t.strip()]
        known = {name for name, _ in connected}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise SystemExit(
                f"--topologies {unknown} are not connected switch combinations of this grid; "
                f"available: {sorted(known)}."
            )
        connected = [(n, o) for n, o in connected if n in wanted]
    return TopologyConfig(
        switch_ids=list(switches),
        variants=[
            TopologyVariantConfig(name=name, open_switches=open_ids)
            for name, open_ids in connected
        ],
        conditioning=getattr(args, "topology_conditioning", "edge_and_global"),
    )


def _switch_ids(grid, spec: str) -> list[int]:
    """Switchable branch ids from ``--switches`` (ids or names); empty spec = every switch."""
    from injection_hse.data import switch_branch_ids

    if not spec.strip():
        return switch_branch_ids(grid)
    by_name = {str(b.name): int(b.id) for b in grid.branches if b.name}
    ids = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token.isdigit():
            ids.append(int(token))
        elif token in by_name:
            ids.append(by_name[token])
        else:
            raise SystemExit(f"--switches: no branch named {token!r} in the grid.")
    return ids


def _first_variant(root, topology):
    """The first configuration's dataset directory (coverage + validation read one).

    Both readers characterise the OPERATING conditions (load level, distortion,
    unbalance) and the RECIPE behind them, which the scenario configuration — shared
    across the switch configurations — determines; one variant is representative, and a
    recipe fault (a dead order, a dark activity window) appears in every variant alike.
    """
    if topology is None:
        return root
    return Path(root) / topology.variants[0].name


def generate_topology_datasets(
    generate,
    grid,
    topology,
    orders,
    injected,
    args,
    out_dir,
    device,
    dtype,
    *,
    root_name,
) -> Path:
    """Generate ONE dataset per switch configuration under ``<out_dir>/<root_name>/<variant>``.

    ``generate`` is the per-dataset generator (randomized / coherent / profiled). Each
    configuration is solved on its own grid copy (its switches opened) and drawn from its own
    seed stream, so the configurations cover different operating points rather than the same
    ones re-solved. Returns the ROOT directory — what the run config records as
    ``data.dataset_path``.
    """
    from injection_hse.data import TopologySet

    topo = TopologySet.from_config(topology)
    root = Path(out_dir) / root_name
    for k, name in enumerate(topo.names):
        print(f"  topology {name} ...", flush=True)
        generate(
            topo.variant_grid(grid, k),
            orders,
            injected,
            args,
            out_dir,
            device,
            dtype,
            dataset_path=root / name,
            seed=int(args.seed) + _TOPOLOGY_SEED_STRIDE * k,
        )
    return root


#: Seed offset between switch configurations, so each draws its own operating points
#: (a shared seed would re-solve the SAME scenarios on every topology).
_TOPOLOGY_SEED_STRIDE = 7919


# =============================================================================
# datasets
# =============================================================================
def data_load_shunt(args) -> tuple:
    """``(model, basis)`` of the harmonic device shunt the datasets are generated with.

    ``(None, None)`` without ``--load-shunt``: pgml's documented defaults, the generation
    every run used before the option existed. With ``--load-shunt`` the basis is
    ``--load-shunt-basis``, ``nameplate`` when unset — the basis on which the estimator's
    decode, which always builds its shunt from the nameplate, sees the generator's ``Y(h)``.
    """
    model = getattr(args, "load_shunt", None)
    if model is None:
        return None, None
    return model, getattr(args, "load_shunt_basis", None) or "nameplate"


def load_shunt_record(args):
    """The data-side and decode-side shunt settings, ``None`` without ``--load-shunt``."""
    model, basis = data_load_shunt(args)
    if model is None:
        return None
    return {
        "data": {"model": model, "basis": basis},
        "decode": {"model": model, "basis": "nameplate"},
    }


def data_generation_record(args, plan=None) -> dict:
    """The generation settings beyond the scenario recipe; empty when all are default.

    Written to ``data_generation.yaml`` and the manifest, beside the experiment configs
    (which record the decode side, ``physics.load_shunt``, but not the data side).
    ``plan`` (:func:`perturbation_plan`) adds the grid-perturbation regime with each copy's
    magnitude; the per-element factors are in the perturbed datasets' ``meta.json``.
    """
    record = {}
    shunt = load_shunt_record(args)
    if shunt is not None:
        record["harmonic_load_shunt"] = shunt
    if getattr(args, "grid_model", None):
        record["grid_model"] = args.grid_model
    if getattr(args, "scenario_config", None):
        record["scenario_config_file"] = file_fingerprint(args.scenario_config)
    record.update(time_axis_record(args))
    if plan is not None:

        def copies(items):
            if items is None:
                return None
            return [
                {"pool": p.pool, "index": p.index, "magnitude_rms_log": p.magnitude}
                for p in items
            ]

        record["grid_perturbation"] = {
            "config": plan["config"].model_dump(mode="json"),
            "train": args.perturb_train,
            "test": args.perturb_test,
            "test_dataset": args.eval_dataset if plan["test"] is not None else None,
            "train_copies": copies(plan["train"]),
            "test_copies": copies(plan["test"]),
        }
    return record


#: Seed offset of the held-out perturbed test dataset's scenario draw: its operating points
#: are new as well as its networks.
_PERTURBED_TEST_SEED_OFFSET = 104729


def load_grid_perturbation(args):
    """The ``--grid-perturbation`` config with the CLI's copy counts; ``None`` when unset.

    The YAML holds a ``grid_perturbation:`` block (:class:`injection_hse.data.scenarios
    .GridPerturbationConfig`, template ``run/configs/pgl_grid_perturbation.yaml``);
    ``--perturb-copies`` / ``--perturb-test-copies`` override its pool sizes.
    """
    path = getattr(args, "grid_perturbation", None)
    if not path:
        return None
    from injection_hse.data.scenarios import GridPerturbationConfig

    doc = yaml.safe_load(_resolve_config_path(path).read_text())
    block = doc.get("grid_perturbation") if isinstance(doc, dict) else None
    if not isinstance(block, dict):
        raise SystemExit(
            f"--grid-perturbation {path}: no 'grid_perturbation:' block (see "
            "run/configs/pgl_grid_perturbation.yaml)."
        )
    if getattr(args, "perturb_copies", None) is not None:
        block = {**block, "n_copies": args.perturb_copies}
    if getattr(args, "perturb_test_copies", None) is not None:
        block = {**block, "n_test_copies": args.perturb_test_copies}
    return GridPerturbationConfig.model_validate(block)


def perturbation_plan(grid, args):
    """The perturbed grid copies of this run, ``None`` without ``--grid-perturbation``.

    ``{"config", "train", "test"}``: ``train`` the copies the training datasets are solved on
    (``None`` = the nominal grid, ``--perturb-train nominal``), ``test`` the copies of the
    held-out perturbed test dataset (``None`` = no such dataset). ``--perturb-test heldout``
    draws them from the ``"test"`` pool, which no training copy shares;
    ``--perturb-test train`` reuses the first training copies (a systematic perturbation
    present in training and test alike).
    """
    config = load_grid_perturbation(args)
    if config is None:
        return None
    from injection_hse.data.scenarios import draw_grid_perturbations

    train = (
        draw_grid_perturbations(grid, config, pool="train")
        if args.perturb_train == "perturbed"
        else None
    )
    test = None
    if args.perturb_test == "heldout":
        test = draw_grid_perturbations(grid, config, pool="test")
    elif args.perturb_test == "train":
        n_test = config.pool_size("test")
        if n_test > len(train):
            raise SystemExit(
                f"--perturb-test train evaluates on {n_test} training copies but the run "
                f"trains on {len(train)}; lower n_test_copies / --perturb-test-copies."
            )
        test = train[:n_test]
    return {"config": config, "train": train, "test": test}


def check_perturbation_flags(args) -> None:
    """Reject grid-perturbation flags that would select nothing or cannot be honoured."""
    regime = args.perturb_train != "nominal" or args.perturb_test != "none"
    if args.grid_perturbation is None:
        if regime or args.perturb_copies or args.perturb_test_copies:
            raise SystemExit(
                "--perturb-train / --perturb-test / --perturb-copies need "
                "--grid-perturbation (the perturbation spreads)."
            )
        return
    if not regime:
        raise SystemExit(
            "--grid-perturbation perturbs nothing on its own: choose --perturb-train "
            "perturbed and/or --perturb-test heldout|train."
        )
    if args.perturb_test == "train" and args.perturb_train != "perturbed":
        raise SystemExit(
            "--perturb-test train evaluates on the training copies; it needs "
            "--perturb-train perturbed."
        )
    if getattr(args, "topologies", "off") not in (None, "", "off"):
        raise SystemExit(
            "--grid-perturbation and --topologies cannot be combined: a perturbed copy "
            "is one network, a topology set several."
        )


def perturbed_test_size(args, n_copies: int) -> int:
    """Scenarios of the held-out perturbed test dataset.

    ``--perturb-test-scenarios`` when set; else the size of the evaluated dataset's own test
    split (``--test-fraction`` of it, the whole dataset at ``0``) — at least one scenario per
    copy.
    """
    if args.perturb_test_scenarios:
        return max(int(args.perturb_test_scenarios), n_copies)
    n = args.n_scenarios if args.eval_dataset == "coherent" else args.n_random
    frac = float(getattr(args, "test_fraction", 0.0) or 0.0)
    size = int(round(n * frac)) if frac > 0 else int(n)
    return max(size, n_copies)


#: How many ``[chunk, T, H, N, N]`` complex system tensors a scenario-dependent harmonic
#: solve holds at once (the assembled matrices, their factorization and a solve workspace)
#: when its chunk is sized against free memory. Measured on CPU (CIGRE LV, 132 rows, T = 8,
#: 9 harmonic orders, complex128): peak memory grows by 58.2 MiB per scenario, against
#: 57.4 MiB for three copies; at 49 harmonic orders (1..50) by 318.5 MiB, against
#: 312.6 MiB. Not measured on a GPU.
_SYSTEM_TENSOR_COPIES = 3


def _free_memory_bytes(device) -> int:
    """Free memory of ``device``: CUDA's free bytes, else the host's available RAM."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:  # pragma: no cover - psutil ships with the environment
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def scenario_dependent_chunk(cfg, orders, grid, args, device, dtype):
    """The chunk size of one dataset solve: ``--chunk-size``, capped to fit memory.

    A device shunt built from each scenario's OPERATING POINT (pgml's default basis, or
    ``--load-shunt-basis operating_point``) makes ``Y(h)`` scenario-dependent, so pgml
    assembles a ``[chunk, T, H, N, N]`` complex system per chunk (2000 sequences x 8 steps
    x 9 orders x 132 rows in complex128 is 37 GiB). The chunk is then capped so
    :data:`_SYSTEM_TENSOR_COPIES` such tensors take at most ``--chunk-memory-fraction`` of
    the device's free memory, and the cap is printed. A scenario-independent ``Y(h)``
    (nameplate basis, or no shunt) keeps one ``[H, N, N]`` and is not capped; neither is a
    batch that already fits, so a run that fit before is solved exactly as before.
    """
    from pgml import defaults as pgml_defaults
    from pgml.assembly import node_phase_index

    model, basis = data_load_shunt(args)
    model = model or pgml_defaults.get("appliance.harmonic_shunt.model")
    basis = basis or pgml_defaults.get("appliance.harmonic_shunt.basis")
    requested = getattr(args, "chunk_size", None)
    if model == "none" or basis != "operating_point":
        return requested
    n_scenarios = int(
        getattr(cfg, "n_samples", None) or getattr(cfg, "n_scenarios", None) or 0
    )
    n_steps = int(getattr(cfg, "n_steps", 1) or 1)
    n_harmonic = sum(1 for h in orders if h > 1)
    n_rows = int(node_phase_index(grid).size)
    itemsize = torch.empty((), dtype=dtype).element_size()
    per_scenario = _SYSTEM_TENSOR_COPIES * n_steps * n_harmonic * n_rows**2 * itemsize
    budget = float(getattr(args, "chunk_memory_fraction", 0.5)) * _free_memory_bytes(
        device
    )
    cap = max(1, int(budget // max(per_scenario, 1)))
    effective = min(requested or n_scenarios, n_scenarios)
    if n_scenarios and effective > cap:
        print(
            f"  operating-point shunt: solving in chunks of {cap} scenario(s) "
            f"({per_scenario / 2**20:.1f} MiB of system tensors per scenario, "
            f"{budget / 2**30:.1f} GiB budget)",
            flush=True,
        )
        return cap
    return requested


def solve_and_write(
    grid,
    cfg,
    orders,
    args,
    device,
    dtype,
    path,
    *,
    perturbations=None,
    perturbation_config=None,
) -> Path:
    """Solve one scenario template on ``grid`` and write the dataset to ``path``.

    The one generation path of every dataset this workflow writes: the harmonic solve of
    the template through :func:`injection_hse.data.scenarios.run_scenarios` under the run's device
    shunt (:func:`data_load_shunt`), with the shunt settings recorded in the dataset's
    ``meta.json`` (``extra_provenance``) when ``--load-shunt`` set them.

    ``perturbations`` (copies of :func:`injection_hse.data.scenarios.draw_grid_perturbations`) solves
    the batch across those perturbed grid copies instead
    (:func:`injection_hse.data.scenarios.run_perturbed_scenarios`): the scenarios are drawn on the
    nominal ``grid``, every scenario records its copy, and ``meta.json`` records each
    copy's factors under ``grid_perturbation``.

    The grid-model options (:func:`apply_grid_model`, ``args.grid_model``) and the
    ``--scenario-config`` file's fingerprint are recorded there as well, and a solve whose
    device shunt follows each scenario's operating point is chunked so its per-scenario
    system matrices fit the device (:func:`scenario_dependent_chunk`).
    """
    from pgml.scenarios import write_dataset
    from injection_hse.data.scenarios import (
        perturbation_provenance,
        run_perturbed_scenarios,
        run_scenarios,
    )

    model, basis = data_load_shunt(args)
    shunt = {} if model is None else {"load_shunt": model, "load_shunt_basis": basis}
    provenance = {}
    record = load_shunt_record(args)
    if record is not None:
        provenance["harmonic_load_shunt"] = record
    if getattr(args, "grid_model", None):
        provenance["grid_model"] = args.grid_model
    if getattr(args, "scenario_config", None):
        provenance["scenario_config_file"] = file_fingerprint(args.scenario_config)
    chunk_size = scenario_dependent_chunk(cfg, orders, grid, args, device, dtype)
    # data generation needs no gradients; collect results on the CPU so a large batch does
    # not accumulate the full [B, H, N] result in VRAM (chunk_size bounds only the per-solve
    # workspace, not the collected output).
    solve = dict(
        calculation="harmonic",
        harmonic_orders=orders,
        symmetry="auto",
        dtype=dtype,
        device=device,
        chunk_size=chunk_size,
        output_device="cpu",
        **shunt,
    )
    with torch.no_grad():
        if perturbations:
            result = run_perturbed_scenarios(grid, cfg, perturbations, **solve)
            provenance["grid_perturbation"] = perturbation_provenance(
                perturbation_config, perturbations, int(result.sampled.n_samples)
            )
        else:
            result = run_scenarios(grid, cfg, **solve)
    return write_dataset(result, Path(path), layout="wide", provenance=provenance)


def _background_update(override, args, section: str) -> dict:
    """``{"background": ...}`` for a ``--scenario-config`` section, or ``{}``.

    The CLI background (``--background-distortion``) applies to a deviation file's recipe as
    to the preset. A section that sets its own ``background`` and a CLI background at the
    same time are ambiguous and refused.
    """
    cli = background_config(args)
    if cli is None:
        return {}
    if getattr(override, "background", None) is not None:
        raise ValueError(
            f"--background-distortion and the '{section}' section of --scenario-config both "
            "set an upstream background; set it in one place."
        )
    return {"background": cli}


def random_scenario_config(grid, orders, injected, args, *, seed=None):
    """The snapshot sampling template this run will generate from.

    The calibrated preset (:func:`injection_hse.data.scenarios.se_random_scenario_config`) unless
    ``--scenario-config`` supplies a ``random:`` section, in which case that shape is
    used with the CLI's sizes, seed and solved orders — a deliberate deviation, recorded
    as such in the run banner. Building the config separately from solving it is what
    lets a driver compare the recipe it is ABOUT to generate against an existing
    dataset's, instead of re-solving to find out.
    """
    override = getattr(args, "scenario_random", None)
    if override is not None:
        # user-provided SHAPE; the CLI owns the sizes/seed so one config scales test -> full.
        override = _check_harmonic_orders(override, injected, "random")
        cfg = override.model_copy(
            update={
                "n_samples": args.n_random,
                "seed": seed if seed is not None else args.seed,
                **_background_update(override, args, "random"),
            }
        )
    else:
        from injection_hse.data.scenarios import se_random_scenario_config

        cfg = se_random_scenario_config(
            grid,
            orders=injected,
            n_samples=args.n_random,
            seed=seed if seed is not None else args.seed,
            emission_persistence=getattr(args, "emission_persistence", "scenario")
            or "scenario",
            background=background_config(args),
            **_emission_fraction_kw(args, "spectrum_fraction"),
        )
    return prune_scenario_parameters(cfg, grid, "random scenario")


def generate_random_dataset(
    grid,
    orders,
    injected,
    args,
    out_dir,
    device,
    dtype,
    *,
    dataset_path=None,
    seed=None,
    perturbations=None,
    perturbation_config=None,
) -> Path:
    """The RANDOMIZED dataset: Sobol snapshots, independent per-order draws.

    ``dataset_path`` / ``seed`` override the output directory and the sampling seed — the
    switchable-topology path generates one dataset per switch configuration, each under its
    variant's subdirectory and drawn from its own seed stream (so the configurations cover
    DIFFERENT operating points rather than the same ones re-solved). ``perturbations`` /
    ``perturbation_config`` solve the batch on perturbed grid copies (:func:`solve_and_write`);
    the coherent and profiled generators take them too.
    """
    cfg = random_scenario_config(grid, orders, injected, args, seed=seed)
    path = Path(dataset_path) if dataset_path else Path(out_dir) / "dataset_random"
    return solve_and_write(
        grid,
        cfg,
        orders,
        args,
        device,
        dtype,
        path,
        perturbations=perturbations,
        perturbation_config=perturbation_config,
    )


def coherent_scenario_config(grid, orders, injected, args, *, seed=None):
    """The sequence sequence template this run will generate from.

    The calibrated preset (:func:`injection_hse.data.scenarios.se_coherent_scenario_config`) at the
    CLI's sizes, seed, temporal mode and composition knobs, or the ``coherent:`` section
    of a ``--scenario-config`` as a deliberate deviation (its harmonic shape and
    fundamental specs are kept; the CLI still owns the sizes, the solved orders and the
    spectrum selector).
    """
    from pgml.scenarios import Selector
    from injection_hse.data.scenarios import (
        LoadProfileConfig,
        se_coherent_scenario_config,
    )

    # ``getattr`` defaults keep non-workflow callers (e.g. the ablation driver, whose argparse
    # namespace does not carry these knobs) working — they get the workflow default (composed).
    mode = getattr(args, "coherent_mode", "composed")
    composed = mode == "composed"
    step_size_s = float(getattr(args, "coherent_step_s", 900.0))
    start_time = getattr(args, "profile_start", None) or _DEFAULT_START_TIME
    override = getattr(args, "scenario_coherent", None)
    if override is None:
        cfg = se_coherent_scenario_config(
            grid,
            orders=injected,
            n_scenarios=args.n_scenarios,
            n_steps=args.n_steps,
            seed=seed if seed is not None else args.seed,
            background=background_config(args),
            **_emission_fraction_kw(args, "fingerprint_fraction"),
            mode=mode,
            step_size_s=step_size_s,
            start_time=start_time,
            n_modes=args.n_modes,
            dwell=args.dwell,
            activity_scale=float(getattr(args, "activity_scale", 1.0)),
            behavioral_coupling=float(getattr(args, "behavioral_coupling", 0.3)),
            cloud_coupling=float(getattr(args, "cloud_coupling", 0.5)),
        )
    else:
        # The COMPOSED path composes `Load` appliances by the composition's own selector, so
        # this selector addresses the devices the composition cannot reach: a PV modelled as a
        # `Generator` is never composed and would otherwise emit nothing at all, leaving it
        # indistinguishable from a co-located load whose fundamental cancels against it.
        # Pointing it at the generators gives each a persistent per-device mode-bank
        # fingerprint across the window, and the generator PROFILE moves their fundamental per
        # step so the harmonic injection moves with it. The FINGERPRINT path has no
        # composition, so there the loads are the spectrum source.
        update = {
            "selector": Selector(component="generator" if composed else "load"),
            "orders": injected,
            "n_steps": args.n_steps,
            "n_scenarios": args.n_scenarios,
            "n_modes": args.n_modes,
            "dwell": args.dwell,
            "seed": seed if seed is not None else args.seed,
            **_background_update(override, args, "coherent"),
        }
        if composed:
            update.update(
                {
                    "composition": override.composition or _composition(args),
                    "start_time": override.start_time or start_time,
                    "step_size_s": step_size_s,
                    "profile": override.profile
                    or LoadProfileConfig(selector=Selector(component="generator")),
                }
            )
        cfg = override.model_copy(update=update)
    cfg = apply_time_axis_options(cfg, args, "coherent scenario")
    return prune_scenario_parameters(cfg, grid, "coherent scenario")


def generate_coherent_dataset(
    grid,
    orders,
    injected,
    args,
    out_dir,
    device,
    dtype,
    *,
    dataset_path=None,
    seed=None,
    perturbations=None,
    perturbation_config=None,
) -> Path:
    """The COHERENT dataset: coherent sequences ``[B, T, H, N]``.

    Two modes select the temporal harmonic structure (``--coherent-mode``):

    - ``composed`` (default): a statistical device-class COMPOSITION
      (:class:`~injection_hse.data.scenarios.CompositionConfig`). Each covered aggregated load is a sum of
      member devices whose per-step activity moves BOTH the fundamental power AND the injected
      harmonic spectrum jointly — the realistic structure a state estimator must learn (device
      activity shifts power and spectrum together, not a sticky spectrum over a frozen
      fundamental). The sequences carry an absolute per-step ``time_unix_s`` (the data behind
      the periodic time features) and per-class attribution ground truth.
    - ``fingerprint``: the earlier per-device mode-bank fingerprint — a stable harmonic
      signature that STICKS to a mode (Markov dwell) and WANDERS (AR(1) jitter) over a
      per-scenario (constant-over-T) fundamental. Timeless (no ``start_time``). Kept for
      ablations that isolate the effect of the composed joint power/spectrum coupling.
    """
    cfg = coherent_scenario_config(grid, orders, injected, args, seed=seed)
    path = Path(dataset_path) if dataset_path else Path(out_dir) / "dataset_coherent"
    return solve_and_write(
        grid,
        cfg,
        orders,
        args,
        device,
        dtype,
        path,
        perturbations=perturbations,
        perturbation_config=perturbation_config,
    )


def _physics_inputs(args) -> list[str]:
    """``--physics-inputs`` as the ``PhysicsConfig.input_priors`` list (default none)."""
    raw = getattr(args, "physics_inputs", "") or ""
    return [token.strip() for token in raw.split(",") if token.strip()]


def _prior_anchor_weights(args) -> tuple[float, float]:
    """``--prior-anchor 'wv[/wi]'`` -> the prior solve's (voltage, current) anchor weights.

    An unset current side mirrors the voltage one (the two anchors carry commensurate
    units on the solve, as with ``--anchor-current``); unset entirely = the config default
    ``1.0/1.0``.
    """
    raw = getattr(args, "prior_anchor", None)
    if not raw:
        return 1.0, 1.0
    wv, _, wi = str(raw).partition("/")
    try:
        return float(wv), float(wi) if wi else float(wv)
    except ValueError:
        raise SystemExit(
            f"--prior-anchor must be 'wv' or 'wv/wi' with numeric weights, got {raw!r}."
        ) from None


def _anchor_current_weight(args) -> float:
    """``--anchor-current``, defaulting to ``--anchor-voltage`` when unset.

    Equal is the honest default: the solve multiplies the voltage residual by the median
    singular value of ``Y(h)`` squared, which puts BOTH anchor terms in the physics term's
    current units, so equal numbers mean equal trust rather than merely equal digits. An
    unset current weight silently anchoring nothing was the trap — a run asking for voltage
    anchoring got a solve that ignored its CTs.
    """
    current = getattr(args, "anchor_current", None)
    return (
        float(getattr(args, "anchor_voltage", 0.0) or 0.0)
        if current is None
        else float(current)
    )


def _composition(args):
    """The sequence device composition at the configured duty level and co-variation.

    The class activity presets are per-device duty cycles, so a composed aggregate sits far
    below the randomized task's loading and the sequences' FUNDAMENTAL barely moves — the sequence pass
    then never visits the loaded states worth estimating. ``--activity-scale`` lifts the whole
    population up its own diurnal curve; ``--behavioral-coupling`` decides how much of the
    remaining variation is SHARED, which is what keeps the aggregate from averaging out.
    """
    from injection_hse.data.scenarios import CompositionConfig

    return CompositionConfig(
        activity_scale=float(getattr(args, "activity_scale", 1.0)),
        behavioral_coupling=float(getattr(args, "behavioral_coupling", 0.3)),
        cloud_coupling=float(getattr(args, "cloud_coupling", 0.5)),
    )


def profile_scenario_config(grid, orders, injected, args, *, seed=None):
    """The profiled-sequence sequence template this run will generate from.

    The coherent preset over a longer, coarser step grid, WITHOUT the per-scenario
    slack-voltage draw: a constant ``[B]`` source scale does not broadcast against the
    ``[B, T]`` profiled solve, and the profile is what moves the fundamental here. A
    ``--scenario-config`` ``coherent:`` section again replaces the shape.
    """
    from pgml.scenarios import Selector
    from injection_hse.data.scenarios import (
        LoadProfileConfig,
        se_coherent_scenario_config,
    )

    override = getattr(args, "scenario_coherent", None)
    profile = LoadProfileConfig()
    if override is None:
        cfg = se_coherent_scenario_config(
            grid,
            orders=injected,
            n_scenarios=args.n_profile_scenarios,
            n_steps=args.profile_steps,
            seed=seed if seed is not None else args.seed,
            background=background_config(args),
            **_emission_fraction_kw(args, "fingerprint_fraction"),
            name="coherent_profile",
            step_size_s=args.profile_step_s,
            start_time=args.profile_start,
            n_modes=args.n_modes,
            dwell=args.dwell,
            activity_scale=float(getattr(args, "activity_scale", 1.0)),
            behavioral_coupling=float(getattr(args, "behavioral_coupling", 0.3)),
            cloud_coupling=float(getattr(args, "cloud_coupling", 0.5)),
            profile=profile,
            slack_voltage_std=0.0,
        )
        # The composition covers the loads; the profile drives every OTHER appliance over
        # the window, so both ground-truth accessors are populated.
        cfg = cfg.model_copy(update={"selector": Selector(component="load")})
    else:
        cfg = override.model_copy(
            update={
                "name": "coherent_profile",
                "selector": Selector(component="load"),
                "orders": injected,
                "n_steps": args.profile_steps,
                "n_scenarios": args.n_profile_scenarios,
                "n_modes": args.n_modes,
                "dwell": args.dwell,
                "seed": seed if seed is not None else args.seed,
                "step_size_s": args.profile_step_s,
                "profile": override.profile or profile,
                "composition": override.composition or _composition(args),
                "start_time": override.start_time or args.profile_start,
                # a per-scenario [B] source scale does not broadcast against the [B, T]
                # profiled solve, so a template's slack draw is dropped here
                "parameters": [
                    spec for spec in override.parameters if not spec.is_source_voltage
                ],
            }
        )
    cfg = apply_time_axis_options(cfg, args, "profiled scenario")
    return prune_scenario_parameters(cfg, grid, "profiled scenario")


def generate_profile_dataset(
    grid,
    orders,
    injected,
    args,
    out_dir,
    device,
    dtype,
    *,
    dataset_path=None,
    seed=None,
    perturbations=None,
    perturbation_config=None,
) -> Path:
    """The COMPOSED (the profiled-sequence pass) dataset: longer, hourly-stepped coherent sequences.

    A :class:`~injection_hse.data.scenarios.CoherentSpectrumConfig` over a longer, coarser step grid
    (``--profile-steps`` steps of ``--profile-step-s`` seconds, hourly by default) that spans a
    full diurnal / weekly window. The temporal structure is the statistical device-class
    COMPOSITION (:class:`~injection_hse.data.scenarios.CompositionConfig`), same as the composed the sequence pass but
    over the wider window: each covered load's per-step device activity moves the fundamental
    power AND the harmonic spectrum jointly, so the sequence carries a MOVING operating point
    stamped with an absolute per-step ``time_unix_s`` — the data behind the periodic time
    features — plus the per-class attribution ground truth.

    A :class:`~injection_hse.data.scenarios.LoadProfileConfig` is kept alongside the composition so the
    UNCOMPOSED appliances (the PV generators, which the load composition does not cover) still
    follow a multi-scale synthetic profile over the window (the composition supersedes the
    covered loads; the profile drives the rest). Both ground-truth accessors are then
    populated — ``composition_labels()`` (loads) and ``profile_factors()`` (all profiled
    devices).
    """
    cfg = profile_scenario_config(grid, orders, injected, args, seed=seed)
    path = Path(dataset_path) if dataset_path else Path(out_dir) / "dataset_profile"
    return solve_and_write(
        grid,
        cfg,
        orders,
        args,
        device,
        dtype,
        path,
        perturbations=perturbations,
        perturbation_config=perturbation_config,
    )


def report_split_metrics(eval_dir) -> None:
    """Print the headline unobserved pu-MAE / THD error for the VAL (and held-out TEST) split.

    Reads the ``metrics.json`` (validation) and ``metrics_test.json`` (the held-out test split,
    written by :func:`injection_hse.evaluation.evaluate_run` only when ``DataConfig.test_fraction > 0``)
    and prints one clearly-labeled line each, so the final generalization estimate is visible in
    the run log. The test split is never touched during training — it is a one-shot estimate.
    """
    eval_dir = Path(eval_dir)
    for label, stem in (("val", "metrics"), ("test", "metrics_test")):
        p = eval_dir / f"{stem}.json"
        if not p.exists():
            continue
        overall = json.loads(p.read_text()).get("overall", {})
        print(
            f"  [{label:>4}] pu-MAE(unobs) {overall.get('pu_mae_unobs', float('nan')):.4g} | "
            f"THD-MAE {overall.get('thd_mae_unobs', float('nan')):.4g} | "
            f"rel {overall.get('rel_unobs', float('nan')):.4g}",
            flush=True,
        )


def write_scenario_coverage(
    dataset_path, grid, out_dir, *, stem="scenario_coverage"
) -> None:
    """Write the scenario-coverage report for a generated dataset (best-effort, never fatal).

    Records what operating conditions the run actually trained on (``<stem>.{pdf,svg}`` +
    ``<stem>.csv``): the THD / VUF / voltage-deviation / load-scale distributions of the
    generated dataset. ``stem`` defaults to ``"scenario_coverage"`` (the snapshot / snapshot
    report); pass ``"scenario_coverage_taskb"`` for the coherent (sequence / sequence) dataset,
    whose ``T`` steps are flattened into the coverage batch. A diagnostic, so a failure here
    is logged but does NOT fail the run.
    """
    try:
        from run.examples.injection_hse.scenario_coverage import coverage_report
    except ModuleNotFoundError:  # script run: run/examples/injection_hse is on sys.path
        from scenario_coverage import coverage_report

    try:
        cov = coverage_report(dataset_path, grid, out_dir, device="cpu", stem=stem)
        print(
            f"  scenario coverage -> {cov.resolve()} (+ .svg, {stem}.csv)",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - a diagnostic, never break the run
        print(f"  (scenario coverage skipped: {type(exc).__name__}: {exc})", flush=True)


def validate_generated(dataset_path, grid, args, *, label: str) -> bool:
    """Validate a freshly generated dataset; print the verdicts. Returns "may train".

    The gate between generation and training: the dataset is measured against its own
    configuration's claims, DIN EN 50160 and — with ``--expected-ranges`` — measured
    device populations, and the verdicts are printed UNCONDITIONALLY (a dead harmonic
    order or a coherent window generated in the dead of night is invisible in the
    generation log otherwise). ``--strict-datagen`` turns a FAIL into an abort; without
    it the run continues and the verdict stands on record in the dataset's
    ``validation.json``.

    A validator that cannot run at all is reported and never blocks the run.
    """
    from injection_hse.data import validate_dataset

    try:
        report = validate_dataset(
            dataset_path,
            grid=grid,
            reference=getattr(args, "expected_ranges", None),
        )
    except Exception as exc:  # noqa: BLE001 - a broken report must not sink a run
        print(f"  (validation skipped: {type(exc).__name__}: {exc})", flush=True)
        return True
    print(f"  validation [{label}] {report.table(limit=12)}", flush=True)
    if report.failed and getattr(args, "strict_datagen", False):
        return False
    return True


# =============================================================================
# configs (the snapshot pass and the sequence pass share the model / measurement / optimizer / seed)
# =============================================================================
def _rel_sigma(text) -> float | list[float]:
    """``--prior-rel-sigma``: one value, or a comma-separated value per order."""
    values = [float(v) for v in str(text).split(",") if v.strip()]
    return values[0] if len(values) == 1 else values


def _harmonic_weight(args, orders) -> HarmonicWeightConfig:
    """The per-order loss priority: the shipped `physical` mode, `uniform`, or a spec.

    ``--harmonic-weight fundamental:W`` puts weight ``W`` on the fundamental and 1 on every
    other order, normalized to mean 1. The reason it exists: under the default weighting the
    fundamental carries a few percent of the blended loss on this benchmark — the harmonic
    orders' RELATIVE errors are one to two decades larger, so they dominate the gradient
    however the weight is nominally set — and a run can therefore drive its held-out loss
    down while the fundamental error rises.
    """
    spec = getattr(args, "harmonic_weight", "physical") or "physical"
    if spec in ("physical", "uniform"):
        return HarmonicWeightConfig(mode=spec)
    if spec.startswith("fundamental:"):
        w = float(spec.split(":", 1)[1])
        values = [w if h == 1 else 1.0 for h in orders]
        return HarmonicWeightConfig(mode="custom", values=values)
    raise SystemExit(
        f"--harmonic-weight {spec!r}: expected 'physical', 'uniform' or 'fundamental:<weight>'."
    )


def _snapshot_stages(args) -> list[MaskStageConfig]:
    """snapshot scarcity stages on the randomized data: full -> random 0.5 -> random 0.25.

    This driver's command line mapped onto :func:`injection_hse.curriculum.snapshot_stages` (the ladder
    and the target ramp live there; the ramp spans BOTH task passes).
    """
    return snapshot_stages(
        epochs=args.epochs,
        target_full=args.target_full,
        target_placement=args.target_placement,
        fundamental_fraction=_fundamental_fraction(args),
    )


def _fundamental_fraction(args) -> float | None:
    """The random-stage smart-meter demotion fraction (``None`` = homogeneous)."""
    frac = float(getattr(args, "fundamental_fraction", 0.0) or 0.0)
    return frac if frac > 0.0 else None


def _node_list(raw: str) -> list:
    """Comma-separated node names/ids -> a mixed list (all-digit tokens become IDS).

    Ids are the robust remote spelling — names with spaces do not survive every
    launcher's argument plumbing (the SLURM args file word-splits).
    """
    tokens = [t.strip() for t in (raw or "").split(",") if t.strip()]
    return [int(t) if t.isdigit() else t for t in tokens]


def _fundamental_nodes(args) -> list:
    """The fundamental-only placement sensors (``--fundamental-nodes``, names/ids)."""
    return _node_list(getattr(args, "fundamental_nodes", "") or "")


def _sensor_nodes(args) -> list:
    """The full-harmonic placement sensors (``--sensors``; default the 3-sensor set)."""
    return _node_list(getattr(args, "sensors", "") or "") or list(SENSOR_NODES)


def _sequence_stages(args) -> list[MaskStageConfig]:
    """sequence scarcity stages on the coherent data: random 0.10 -> the 3-sensor placement.

    The placement finetune optionally overrides the optimizer via ``--placement-lr`` /
    ``--placement-scheduler`` (per-stage :class:`MaskStageConfig` fields; unset = inherit
    the global ``--lr`` / ``--scheduler``). Continues the target ramp of
    :func:`injection_hse.curriculum.sequence_stages` where the snapshot pass left it.
    """
    return sequence_stages(
        epochs=args.epochs,
        target_full=args.target_full,
        target_placement=args.target_placement,
        fundamental_fraction=_fundamental_fraction(args),
        placement_lr=getattr(args, "placement_lr", None),
        placement_scheduler=getattr(args, "placement_scheduler", None),
    )


def _profiled_sequence_stages(args) -> list[MaskStageConfig]:
    """profiled-sequence stage on the PROFILED coherent data: the 3-sensor placement, time-aware.

    A single ``placement`` finetune continuing the sequence pass's weights, now on the profiled
    (time-stamped) sequences with the periodic time features enabled — the model conditions
    the placement estimate on the diurnal / weekly / seasonal profile structure. The optimizer
    inherits ``--placement-lr`` / ``--placement-scheduler`` like the sequence placement stage.
    """
    return profiled_sequence_stages(
        epochs=args.epochs,
        target_loss=args.target_placement,
        lr=getattr(args, "placement_lr", None),
        scheduler=getattr(args, "placement_scheduler", None),
    )


def _direct_stages(stages, args) -> list:
    """``--no-curriculum``: one pass collapsed to ONE stage with the pass's whole epoch budget.

    The scarcity schedule is the variable under test, the optimization budget and the dataset
    sequence are not; the snapshot pass still fits the normalizers and the operator bank, as it
    does for the ladder. The stage trains at the deployed placement, or on random sensor sets
    with ``--direct-random-fraction`` (:func:`_direct_random`).
    """
    stage = MaskStageConfig(
        name="placement",
        kind="placement",
        observed_fraction=None,
        target_loss=stages[-1].target_loss,
        max_epochs=sum(s.max_epochs for s in stages),
        lr=stages[-1].lr,
        scheduler=stages[-1].scheduler,
        anchor_voltage_weight=stages[-1].anchor_voltage_weight,
        anchor_current_weight=stages[-1].anchor_current_weight,
    )
    return [_direct_random(stage, args)]


def _with_stage_warmup(epochs: int, *passes):
    """Every stage after the run's first gets a ``epochs``-epoch learning-rate warmup.

    ``passes`` are the stage lists in training order (``None`` passes are kept as ``None``).
    A stage shorter than ``epochs + 1`` epochs warms up for ``max_epochs - 1``. Returns the
    passes with the stages replaced.
    """
    out, first = [], True
    for stages in passes:
        if stages is None:
            out.append(None)
            continue
        new = []
        for stage in stages:
            if first:
                first = False
                new.append(stage)
                continue
            n = min(int(epochs), int(stage.max_epochs) - 1)
            new.append(
                MaskStageConfig.model_validate(
                    {**stage.model_dump(), "warmup_epochs": max(n, 0)}
                )
            )
        out.append(new)
    return out


def _direct_random(stage, args):
    """``stage`` on RANDOM sensor sets when ``--direct-random-fraction`` is given.

    A placement-free network: every sample observes its own random ``fraction`` of the buses
    (the slack always), so no sensor set is privileged in training. The stage keeps its name,
    so ``--anchor-schedule`` keys and the checkpoint names are unchanged.
    """
    fraction = getattr(args, "direct_random_fraction", None)
    if fraction is None or stage.kind != "placement":
        return stage
    return stage.model_copy(
        update={"kind": "random", "observed_fraction": float(fraction)}
    )


def _anneal_stage(args) -> MaskStageConfig:
    """The final cosine-anneal placement finetune (appended when ``--anneal-epochs > 0``).

    See :func:`injection_hse.curriculum.anneal_stage` for why the anneal is a stage of its own.
    """
    return anneal_stage(
        epochs=args.anneal_epochs, lr=getattr(args, "placement_lr", None)
    )


def accumulation_steps(args) -> int:
    """Gradient-accumulation steps so the EFFECTIVE batch equals ``--effective-batch``.

    ``--effective-batch`` (when set) is the optimization batch held CONSTANT across an
    architecture comparison; each variant's ``--batch-size`` is only the memory-bound
    micro-batch. Under multi-GPU data parallelism every optimizer step already sees
    ``batch_size × devices`` samples, so the accumulation factor shrinks accordingly —
    the effective batch (and thus the optimization trajectory semantics) stays constant
    across GPU counts. Requires the effective batch to be a whole multiple of
    ``batch_size × devices``. ``None`` (default) means no accumulation.
    """
    eff = getattr(args, "effective_batch", None)
    if eff is None:
        return 1
    per_step = args.batch_size * max(1, int(getattr(args, "devices", 1)))
    if eff < per_step or eff % per_step:
        raise SystemExit(
            f"--effective-batch ({eff}) must be a whole multiple of --batch-size × "
            f"--devices ({args.batch_size} × {getattr(args, 'devices', 1)} = {per_step}) "
            "and at least as large."
        )
    return eff // per_step


def _noise_config(args) -> NoiseConfig:
    """Measurement noise from ``--noise-class``: ``off`` (default), an accuracy class or ``A``.

    An accuracy class becomes the fallback for every sensor without device metadata
    (config-driven placements); the per-class sigma table is the
    :class:`~injection_hse.config.NoiseConfig` default. ``A`` reads every voltage and current channel
    with the IEC 61000-4-30 class-A instrument model
    (:meth:`injection_hse.config.InstrumentErrorConfig.class_a`: a gain and phase error held per
    scenario, channel and order, plus a floor drawn per reading). Seeded off the run seed
    so a re-run is byte-identical.
    """
    cls = getattr(args, "noise_class", "off")
    if cls in (None, "off"):
        return NoiseConfig()
    draw_device = getattr(args, "noise_draw_device", "cpu")
    if cls == "A":
        return NoiseConfig(
            enabled=True,
            seed=args.seed,
            draw_device=draw_device,
            instrument=InstrumentErrorConfig.class_a(),
        )
    return NoiseConfig(
        enabled=True,
        default_class=cls,
        seed=args.seed,
        draw_device=draw_device,
    )


def _spatial_model(args):
    """The spatial model under the temporal wrapper (``--model``; GNN by default)."""
    kind = getattr(args, "model", "gnn") or "gnn"
    hidden = getattr(args, "hidden_dim", 128) or 128
    layers = getattr(args, "gnn_layers", 6) or 6
    if kind == "dnn":
        return DNNConfig(hidden_dims=[512, 512, 512])
    if kind == "graphormer":
        return GraphormerConfig(hidden_dim=hidden, num_layers=layers)
    return GNNConfig(
        hidden_dim=hidden,
        num_layers=layers,
        input_skip=getattr(args, "input_skip", False),
    )


def _base_config(
    args, dataset_path, orders, stages, out_dir, name, *, topology=None
) -> ExperimentConfig:
    """A full ``ExperimentConfig`` sharing the model / measurement / optimizer / trainer."""
    return ExperimentConfig(
        name=name,
        seed=args.seed,
        orders=orders,
        data=DataConfig(
            source="parquet",
            dataset_path=str(dataset_path),
            topology=topology,
            phase_coupling=PhaseCouplingConfig(
                enabled=bool(getattr(args, "phase_coupling", False))
            ),
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            test_fraction=getattr(args, "test_fraction", 0.0),
            keep_failed=getattr(args, "keep_failed", False),
            num_workers=args.num_workers,
        ),
        model=TemporalConfig(spatial=_spatial_model(args)),
        encoding=EncodingConfig(
            kind=getattr(args, "encoding", "trig"),
            time_features=getattr(args, "time_features", False),
            injection_kind=getattr(args, "injection_encoding", None),
        ),
        normalization=NormalizationConfig(per_unit=getattr(args, "per_unit", False)),
        physics=PhysicsConfig(
            decode=args.decode,
            input_priors=_physics_inputs(args),
            prior_anchor_voltage_weight=_prior_anchor_weights(args)[0],
            prior_anchor_current_weight=_prior_anchor_weights(args)[1],
            anchor_voltage_weight=getattr(args, "anchor_voltage", 0.0) or 0.0,
            anchor_current_weight=_anchor_current_weight(args),
            anchor_weight_draw=getattr(args, "anchor_draw", "uniform"),
            anchor_weighting=getattr(args, "anchor_weighting", "manual"),
            prior_injection_rel_sigma=_rel_sigma(getattr(args, "prior_rel_sigma", "1")),
            injection_prior_residual=bool(getattr(args, "prior_residual", False)),
            injection_prior_fit_states=int(getattr(args, "prior_fit_states", 4096)),
            exclude_tied_rows=not bool(getattr(args, "keep_tied_rows", False)),
            harmonic_slack=str(getattr(args, "harmonic_slack", "norton") or "norton"),
            load_shunt=getattr(args, "load_shunt", None),
        ),
        measurement=MeasurementConfig(
            voltage_nodes=_sensor_nodes(args),
            fundamental_nodes=_fundamental_nodes(args),
            measure_current=getattr(args, "measure_currents", False),
            current_channels=getattr(args, "current_channels", "incident"),
            measure_power=getattr(args, "measure_power", False),
            # the per-order observability channels are needed whenever ANY sensor (a
            # placement smart meter or a random-stage demotion) is fundamental-only
            heterogeneous_orders=bool(_fundamental_nodes(args))
            or _fundamental_fraction(args) is not None,
            slack_known=True,
        ),
        # localization head OFF (node-V SE only); the optional split moves part of the
        # state supervision onto the injection decode's predicted nodal currents.
        loss=LossConfig(
            harmonic_weight=_harmonic_weight(args, orders),
            injection_state_loss_split=getattr(args, "injection_loss_split", 0.0)
            or 0.0,
            injection_state_loss_normalize=not getattr(
                args, "raw_injection_loss", False
            ),
            plain_decode_weight=getattr(args, "plain_decode_weight", 0.0) or 0.0,
            plain_decode_observed_weight=getattr(
                args, "plain_decode_observed_weight", None
            ),
            prior_distillation_weight=getattr(args, "prior_distillation", 0.0) or 0.0,
            preconditioner=getattr(args, "precondition", "none") or "none",
            preconditioner_floor=getattr(args, "precondition_floor", 0.01) or 0.01,
            supervise_rows=getattr(args, "supervise_rows", "all") or "all",
        ),
        noise=_noise_config(args),
        curriculum=CurriculumConfig(stages=stages),
        optim=OptimConfig(
            name="adamw",
            lr=args.lr,
            scheduler=args.scheduler,
            eps=getattr(args, "adam_eps", 1e-8),
        ),
        train=TrainConfig(
            precision="32",
            dtype=getattr(args, "train_dtype", "complex64"),
            accelerator=args.accelerator,
            devices=args.devices,
            grad_clip=args.grad_clip,
            accumulate_grad_batches=accumulation_steps(args),
            early_stop_patience=getattr(args, "patience", 0),
            early_stop_min_delta=getattr(args, "min_delta", 0.0),
            carry_optimizer_state=bool(getattr(args, "carry_optimizer_state", False)),
            mlflow_tracking_uri=args.mlflow_uri,
            checkpoint_dir=str(Path(out_dir) / "checkpoints"),
            enable_progress_bar=not args.no_progress,
        ),
    )


# =============================================================================
# orchestration: ONE module, ONE StageManager pass per task (weights continue A -> B)
# =============================================================================
def training_run(
    cfg_a, cfg_b, cfg_c=None, *, grid_path=None, init_from=None, start_stage=None
) -> TrainingRunConfig:
    """This workflow's task passes as an EXECUTABLE :class:`injection_hse.workflow.TrainingRunConfig`.

    The run record dumped beside the per-task configs: one pass per enabled task, in
    training order, plus the grid the datasets were generated on and the resume pointers.
    ``python -m injection_hse.train --config <run dir>/training_run.yaml`` re-runs exactly this.
    """
    passes = [
        TrainingPass(name=f"task{label}", experiment=cfg)
        for label, cfg in (("A", cfg_a), ("B", cfg_b), ("C", cfg_c))
        if cfg is not None
    ]
    return TrainingRunConfig(
        name=cfg_a.name.removesuffix("-A"),
        passes=passes,
        grid_path=None if grid_path is None else str(grid_path),
        init_from=None if init_from is None else str(init_from),
        start_stage=start_stage,
    )


def build_curriculum(grid, cfg_a, cfg_b, orders, *, cfg_c=None):
    """Build the shared module, the datamodules, and the placement rows.

    The exact construction the multi-task training uses (:func:`injection_hse.workflow
    .build_training_setup`) — seeding, the data sources, the normalizer fit on the union of
    the two tasks' train splits, the injection solver when ``physics.decode == "injection"``
    — shared by :func:`run_curriculum` and :func:`preflight_curriculum` so a memory preflight
    probes the REAL training footprint. ``cfg_c`` (optional) is the PROFILED profiled-sequence config;
    its datamodule is built too (``dm_c is None`` when absent). Returns
    ``(module, dm_a, dm_b, dm_c, observed_rows, placement_extras)``.

    When the profiled-sequence pass is present, ``cfg_a`` carries ``encoding.time_features=True`` (so the ONE
    module gains the time-feature input width for every task); Tasks A/B — whose datasets
    carry no absolute timestamp — then feed the all-zero time vector.
    """
    if list(orders) != list(cfg_a.orders):
        raise SystemExit(
            f"the task configs declare orders {list(cfg_a.orders)} but the run builds "
            f"operators for {list(orders)}; they are one harmonic axis."
        )
    setup = build_training_setup(training_run(cfg_a, cfg_b, cfg_c), grid=grid)
    extras = {k: v for k, v in setup.stage_kwargs.items() if k != "observed_rows"}
    dms = setup.datamodules
    return (
        setup.module,
        dms[0],
        dms[1],
        dms[2] if cfg_c is not None else None,
        setup.stage_kwargs["observed_rows"],
        extras,
    )


def preflight_curriculum(
    grid, cfg_a, cfg_b, orders, *, device, allow_tight_memory=False
):
    """Build one configuration's module and probe ONE coherent training step.

    Returns the step's peak CUDA memory in bytes (``None`` on CPU, where no peak tracking
    exists) — see :func:`injection_hse.workflow.preflight_training`: any error the first real training
    step would hit (an oversized-batch OOM above all) is raised HERE, so a sweep can probe
    every planned configuration up front instead of discovering it hours in.
    """
    if list(orders) != list(cfg_a.orders):
        raise SystemExit(
            f"the task configs declare orders {list(cfg_a.orders)} but the preflight was "
            f"asked for {list(orders)}; they are one harmonic axis."
        )
    return preflight_training(
        training_run(cfg_a, cfg_b),
        grid=grid,
        device=device,
        allow_tight_memory=allow_tight_memory,
    )


def run_curriculum(
    grid,
    cfg_a,
    cfg_b,
    orders,
    out_dir,
    *,
    cfg_c=None,
    restore_best=True,
    device=None,
    extra_callbacks=None,
    allow_tight_memory=False,
    init_from=None,
    start_stage=None,
    grid_path=None,
):
    """Run this workflow's task passes over ONE module (:func:`injection_hse.workflow.run_training`).

    Returns ``(module, history_collector, ckpt_paths)``. The module's weights persist from
    the snapshot pass into the sequence pass into the optional profiled-sequence pass (the SAME object is
    reused); a shared :class:`~injection_hse.evaluation.HistoryCollector` accumulates the epoch-indexed
    history across all of them. ``extra_callbacks`` are attached to every stage of every pass
    — e.g. a :class:`injection_hse.analysis.GradientNormCollector`.

    ``init_from`` starts the run from an existing stage checkpoint (weights + fitted
    normalizers) — pass task configs whose stage lists hold only the stages still to train
    (:func:`injection_hse.curriculum.resume_stage_split`). A task pass whose stage list is empty is
    skipped. The resolved run config is written to ``<out_dir>/training_run.yaml``, so the
    run directory carries the re-entry point of what it just trained.
    """
    cfg = training_run(
        cfg_a,
        cfg_b,
        cfg_c,
        grid_path=grid_path,
        init_from=init_from,
        start_stage=start_stage,
    )
    result = run_training(
        cfg,
        grid=grid,
        device=device,
        out_dir=out_dir,
        restore_best=restore_best,
        extra_callbacks=extra_callbacks,
        allow_tight_memory=allow_tight_memory,
    )
    return result.module, result.history, result.checkpoints


# =============================================================================
# per-copy evaluation of perturbed datasets
# =============================================================================
def _per_copy_rows(prediction, copy_of_scenario, copies) -> list[str]:
    """CSV rows of one prediction: every copy, then all copies pooled; per order + ``all``."""
    import numpy as np

    from injection_hse.evaluation import prediction_metrics, prediction_metrics_by_group

    by_copy = prediction_metrics_by_group(prediction, copy_of_scenario)
    scenario_copy = np.asarray(copy_of_scenario)[prediction.scenario_index]
    meta = {int(c["index"]): c for c in copies}

    def rows(pool, copy_, magnitude, n_scen, n_samp, errors):
        lead = f"{pool},{copy_},{magnitude},{n_scen},{n_samp}"
        out = [f"{lead},all,{errors['mae_unobs']:.6g},{_cell(errors['rel_unobs'])}"]
        for o in prediction.orders:
            e = errors[o]
            out.append(f"{lead},{o},{e['mae_unobs']:.6g},{_cell(e['rel_unobs'])}")
        return out

    lines = []
    for k, errors in sorted(by_copy.items()):
        rec = meta.get(int(k), {})
        sel = scenario_copy == k
        lines += rows(
            rec.get("pool", ""),
            int(k),
            f"{rec.get('magnitude_rms_log', float('nan')):.6g}",
            len(set(prediction.scenario_index[sel].tolist())),
            int(sel.sum()),
            errors,
        )
    lines += rows(
        "all",
        "all",
        "",
        len(set(prediction.scenario_index.tolist())),
        prediction.n_samples,
        prediction_metrics(prediction),
    )
    return lines


def _cell(value: float) -> str:
    """A CSV number cell; ``nan`` (a silent order's relative error) is left empty."""
    return "" if value != value else f"{value:.6g}"


def evaluate_perturbation(out_dir, grid, eval_dir, *, device="cpu") -> list[Path]:
    """Per-copy metrics of the run's perturbed datasets -> CSVs beside ``metrics.json``.

    Reads the manifest. For the evaluated training dataset, when it was generated on
    perturbed copies and the run held out a test split, scores that split per copy
    (``metrics_test_per_copy.csv`` — copies seen in training, unseen operating points); for
    the held-out perturbed test dataset (``--perturb-test``), scores every scenario per copy
    (``metrics_perturbed_test_per_copy.csv``). Each CSV holds, per copy and pooled over all
    copies, the unobserved per-unit MAE and relative error per order and overall
    (``order=all``), with the copy's pool and perturbation magnitude (RMS log-factor).
    Returns the written paths.
    """
    from injection_hse.config import MeasurementConfig
    from injection_hse.data.scenarios import GRID_COPY_RECORD, read_dataset
    from injection_hse.evaluation import load_se_module, predict_dataset

    out_dir, eval_dir = Path(out_dir), Path(eval_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    datasets = manifest["datasets"]
    trained = datasets.get(
        "dataset_coherent"
        if manifest.get("eval_dataset", "coherent") == "coherent"
        else "dataset_random"
    )
    targets = [
        (trained, "test", "metrics_test_per_copy.csv"),
        (
            datasets.get("dataset_perturbed_test"),
            "full",
            "metrics_perturbed_test_per_copy.csv",
        ),
    ]
    measurement = MeasurementConfig(
        voltage_nodes=manifest["sensor_names"], slack_known=True
    )
    written = []
    header = (
        "pool,copy,magnitude_rms_log,n_scenarios,n_samples,order,pu_mae_unobs,rel_unobs"
    )
    for path, split, name in targets:
        if path is None:
            continue
        meta = json.loads((Path(path) / "meta.json").read_text())
        record = meta.get("extra_provenance", {}).get("grid_perturbation")
        if record is None:
            continue
        module = load_se_module(
            manifest["final_checkpoint"],
            grid=grid,
            dataset_path=str(path),
            device=device,
        )
        if split == "test" and module.cfg.data.test_fraction <= 0.0:
            continue
        prediction = predict_dataset(
            module,
            source_or_path=str(path),
            grid=grid,
            measurement=measurement,
            split=split,
            device=device,
        )
        copy_of_scenario = read_dataset(path).samples[GRID_COPY_RECORD].numpy()
        lines = _per_copy_rows(prediction, copy_of_scenario, record["copies"])
        target = eval_dir / name
        target.write_text("\n".join([header, *lines]) + "\n")
        written.append(target)
        pooled = next(line for line in lines if line.startswith("all,all,"))
        print(
            f"  [{name.removesuffix('_per_copy.csv')}] all copies: "
            f"pu-MAE(unobs) {pooled.split(',')[6]} | rel {pooled.split(',')[7]}"
            f" -> {target.name}",
            flush=True,
        )
    return written


# =============================================================================
# manifest
# =============================================================================
def write_manifest(
    out_dir,
    grid,
    cfg_a,
    cfg_b,
    orders,
    random_path,
    coherent_path,
    final_ckpt,
    *,
    cfg_c=None,
    profile_path=None,
    init_checkpoint=None,
    scenario_recipe=None,
    data_generation=None,
    perturbed_test_path=None,
    eval_dataset=None,
) -> Path:
    """Capture everything the evaluation needs to run standalone -> ``manifest.json``.

    ``data_generation`` (:func:`data_generation_record`) records the generation settings
    beyond the scenario recipe when any is set; ``perturbed_test_path`` the held-out
    perturbed test dataset (``datasets.dataset_perturbed_test``); ``eval_dataset`` the
    dataset kind the per-copy evaluation scores (written when given).

    ``scenario_recipe`` (see :func:`scenario_recipe_identity`) records WHICH scenario
    recipe generated the datasets, so a run directory says what it trained on without
    reopening the dataset sidecars.

    ``stages`` lists exactly the stages THIS run trained (a resumed run lists only the
    remaining ones, keeping the stage indices aligned with the history / gradient
    collector); ``init_checkpoint`` records the filename of the stage checkpoint the
    run was initialized from (copied into ``checkpoints/``), when resuming. When the optional
    PROFILED profiled-sequence pass ran, its stages are appended (task ``"C"``) and the profiled dataset
    is recorded under ``datasets.dataset_profile``.
    """
    from pgml.assembly import node_phase_index

    idx = node_phase_index(grid)
    # stage_index -> {task, stage_name, kind, observed_fraction}; A first, then B, then C
    tasks = [("A", cfg_a), ("B", cfg_b)]
    if cfg_c is not None:
        tasks.append(("C", cfg_c))
    stages: dict = {}
    si = 0
    for task, cfg in tasks:
        for st in cfg.curriculum.stages:
            stages[str(si)] = {
                "task": task,
                "stage_name": st.name,
                "kind": st.kind,
                "observed_fraction": st.observed_fraction,
            }
            si += 1
    switch_stage_index = len(cfg_a.curriculum.stages)  # first sequence stage index

    datasets = {
        "dataset_random": str(Path(random_path).resolve()),
        "dataset_coherent": str(Path(coherent_path).resolve()),
    }
    if profile_path is not None:
        datasets["dataset_profile"] = str(Path(profile_path).resolve())
    if perturbed_test_path is not None:
        datasets["dataset_perturbed_test"] = str(Path(perturbed_test_path).resolve())

    topology = cfg_a.data.topology
    manifest = {
        "stages": stages,
        "switch_stage_index": switch_stage_index,
        "sensor_names": list(cfg_b.measurement.voltage_nodes),
        "fundamental_sensor_names": list(cfg_b.measurement.fundamental_nodes),
        "measure_current": bool(cfg_b.measurement.measure_current),
        "measure_power": bool(cfg_b.measurement.measure_power),
        "orders": list(orders),
        "time_features": bool(cfg_a.encoding.time_features),
        "datasets": datasets,
        "final_checkpoint": str(Path(final_ckpt).resolve()),
        "grid": {"n_nodes": len(grid.nodes), "n_rows": int(idx.size)},
    }
    if topology is not None:
        manifest["topology"] = {
            "switch_ids": list(topology.switch_ids),
            "variants": {v.name: list(v.open_switches) for v in topology.variants},
            "conditioning": topology.conditioning,
        }
    if init_checkpoint is not None:
        manifest["init_checkpoint"] = str(init_checkpoint)
    if scenario_recipe is not None:
        manifest["scenario_recipe"] = str(scenario_recipe)
    if data_generation:
        manifest["data_generation"] = data_generation
    if eval_dataset is not None:
        manifest["eval_dataset"] = eval_dataset
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "out_dir",
        nargs="?",
        default=str(output_dir("se_full_workflow")),
        help="Output dir for the two datasets + checkpoints + mlruns + manifest "
        "(default: $PGML_EXPERIMENTS/se_full_workflow, else ./data/se_full_workflow).",
    )
    # simulation sizes (smoke-sized defaults; production sizes in the module docstring)
    ap.add_argument(
        "--n-random",
        type=int,
        default=32,
        help="Randomized snapshots (the snapshot pass, B).",
    )
    ap.add_argument(
        "--n-scenarios",
        type=int,
        default=16,
        help="Coherent sequences (the sequence pass, B).",
    )
    ap.add_argument(
        "--n-steps", type=int, default=3, help="Steps per coherent sequence (T)."
    )
    ap.add_argument(
        "--n-modes", type=int, default=2, help="Base spectra per device (coherent)."
    )
    ap.add_argument(
        "--dwell",
        type=float,
        default=0.9,
        help="Mode persistence per step (fingerprint).",
    )
    ap.add_argument(
        "--coherent-mode",
        default="composed",
        choices=["composed", "fingerprint"],
        help="Temporal harmonic structure of the coherent (the sequence passes) data. 'composed' "
        "(default): a statistical device-class COMPOSITION whose per-step device activity "
        "moves the fundamental power AND the harmonic spectrum jointly (the realistic "
        "structure; carries an absolute time axis for the periodic time features + per-class "
        "attribution labels). 'fingerprint': the earlier per-device mode-bank fingerprint "
        "(sticky spectrum over a per-scenario fundamental; timeless) — kept for ablations.",
    )
    ap.add_argument(
        "--coherent-step-s",
        type=float,
        default=900.0,
        help="Seconds between COMPOSED sequence steps (default 900 = 15 min), so the short "
        "sequence spans a sub-hour window with a moving fundamental. Ignored in "
        "--coherent-mode fingerprint (that data has no absolute time axis).",
    )
    ap.add_argument(
        "--pv-fraction", type=float, default=0.5, help="Load nodes with PV."
    )
    ap.add_argument(
        "--grid",
        default=None,
        help="Train on a STORED pgml.Grid JSON (a real network) instead of the synthesized "
        "CIGRE LV benchmark. Its installed "
        "measurement devices become the placement unless --sensors overrides them, and "
        "--pv-fraction is ignored (the grid is taken as it is).",
    )
    ap.add_argument(
        "--topologies",
        default="off",
        help="Train ONE estimator across several SWITCH CONFIGURATIONS of the grid: 'off' "
        "(default) = the grid as given; 'all' = every combination of the switchable branches "
        "that leaves the grid connected; or a comma-separated subset of those combination "
        "names ('c' closed / 'o' open per switch in --switches order, e.g. 'co,cc' — training "
        "on some topologies to test generalization to the rest). One dataset is generated per "
        "configuration (the sizes below are split across them), every sample is decoded with "
        "ITS configuration's physics, and the model is told which one it is "
        "(--topology-conditioning).",
    )
    ap.add_argument(
        "--switches",
        default="",
        help="With --topologies: the switchable branches, comma-separated ids or names "
        "(default: every switch in the grid).",
    )
    ap.add_argument(
        "--topology-conditioning",
        default="edge_and_global",
        choices=["edge_and_global", "edge", "global", "none"],
        help="How the model is TOLD which switch configuration a sample came from: as a "
        "per-half-edge conduction channel, as the switch-state vector broadcast to every "
        "node, both (default), or not at all ('none' — the model must infer it from the "
        "measurements; the physics decode still uses the true configuration).",
    )
    # optional PROFILED the profiled-sequence pass (time-varying fundamental + periodic time features); OFF by
    # default (--n-profile-scenarios 0) so the two-task run is byte-identical to before.
    ap.add_argument(
        "--n-profile-scenarios",
        type=int,
        default=0,
        help="COMPOSED longer-window coherent sequences (the profiled-sequence pass). > 0 enables the optional "
        "third task: a device-composition dataset over a wider (typically diurnal) window and "
        "a placement finetune of the SAME module. Time features (sin/cos time-of-day/week/year "
        "+ a validity flag) are on whenever composed data is in the curriculum. 0 (default) "
        "disables it.",
    )
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=24,
        help="Steps per COMPOSED profiled-sequence sequence (T). Longer than --n-steps so the window "
        "spans a diurnal window; with --profile-step-s 3600 the default is one day.",
    )
    ap.add_argument(
        "--profile-step-s",
        type=float,
        default=3600.0,
        help="Seconds between COMPOSED profiled-sequence steps. Default 3600 (hourly), so the time "
        "features sweep the day/week/season.",
    )
    ap.add_argument(
        "--profile-start",
        default=_DEFAULT_START_TIME,
        help="Absolute ISO 8601 start time anchoring the COMPOSED coherent sequences (the sequence pass "
        "and the profiled-sequence pass); sets the daily / weekly / seasonal phase of the device-activity model "
        "(interpreted as UTC when naive).",
    )
    add_time_axis_arguments(ap)
    ap.add_argument(
        "--dtype",
        default="complex128",
        choices=list(DTYPES),
        help="DATA-GENERATION solve precision (default complex128), independent of the "
        "training precision (--train-dtype). The harmonic Y-bus is ill-conditioned at higher "
        "orders, where complex64 loses accuracy in the small harmonic voltages the estimator "
        "must recover, so the LABELS are generated in double precision; the model itself is "
        "trained at --train-dtype (complex64 by default). Select complex64 here for a faster, "
        "lower-memory generation when the conditioning is acceptable.",
    )
    ap.add_argument(
        "--train-dtype",
        default="complex64",
        choices=list(DTYPES),
        help="TRAINING precision (default complex64), decoupled from the generated data's "
        "--dtype. complex64 casts the loaded complex128 labels to single precision at the "
        "module boundary and trains a float32 model (the headline case: high-precision labels, "
        "fast training); complex128 trains a float64 model for a full-precision study. The "
        "physics decode always solves in complex128 internally either way.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Solve the scenario batch in slices of at most this many scenarios when "
        "generating the datasets (streams the dense [B,...] system so a large --n-random / "
        "--n-scenarios fits GPU memory; the result is identical). Applies to BOTH datasets.",
    )
    # training
    ap.add_argument(
        "--epochs", type=int, default=2, help="Max epochs per curriculum stage."
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--effective-batch",
        type=int,
        default=None,
        help="Effective (optimization) batch size held constant via gradient "
        "accumulation; --batch-size is then only the memory-bound micro-batch. "
        "Must be a whole multiple of --batch-size. Default: no accumulation.",
    )
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient-norm clip (0 disables). Guards against the rare loss-spike that, "
        "without clipping, corrupts AdamW's moment estimates and pins the loss at a "
        "degenerate ~0.5 plateau for the rest of the stage.",
    )
    ap.add_argument(
        "--scheduler",
        default="none",
        choices=["none", "cosine", "plateau"],
        help="LR schedule (default none). 'cosine'/'plateau' help avoid late-stage spikes.",
    )
    ap.add_argument(
        "--placement-lr",
        type=float,
        default=None,
        help="Learning rate for the placement finetune stage only (default: inherit "
        "--lr). Stored on the stage's MaskStageConfig, so the run config stays the "
        "reproducible source of truth.",
    )
    ap.add_argument(
        "--placement-scheduler",
        default=None,
        choices=["none", "cosine", "plateau"],
        help="LR schedule for the placement finetune stage only (default: inherit "
        "--scheduler). 'cosine' anneals once over the stage's --epochs. Prefer a CONSTANT "
        "placement lr ('none') paired with --anneal-epochs (a separate final anneal stage) "
        "over cosine here: annealing across the whole placement stage drops the lr to ~0 "
        "early and can freeze progress while validation is still descending.",
    )
    ap.add_argument(
        "--anneal-epochs",
        type=int,
        default=0,
        help="Append a final 'placement_anneal' stage of this many epochs (0 = off, the "
        "default). The placement stage then runs at a CONSTANT lr (stopping on its target / "
        "plateau) and this stage polishes from that plateau with a cosine schedule from the "
        "placement lr down toward ~0 — instead of annealing across the whole placement stage. "
        "Recommended combination: --placement-scheduler none --anneal-epochs 150.",
    )
    ap.add_argument(
        "--decode",
        default="injection",
        choices=["voltage", "injection"],
        help="State-estimator decode (default: injection). 'injection' predicts nodal "
        "injection currents + reconstructs voltages via the differentiable network solve "
        "(physically consistent by construction); 'voltage' predicts the voltages directly.",
    )
    ap.add_argument(
        "--anchor-voltage",
        type=float,
        default=0.0,
        help="Injection decode only: MEASUREMENT-ANCHOR weight for the observed VOLTAGE rows "
        "(0 = off, the slack-only solve). The observed voltages become soft constraints so the "
        "reconstructed state is pulled toward every voltage sensor, not just the slack "
        "(over-determined least squares). Dimensionless relative trust (~1 = trust a voltage "
        "measurement like one nodal current balance).",
    )
    ap.add_argument(
        "--anchor-current",
        type=float,
        default=None,
        help="Injection decode only: MEASUREMENT-ANCHOR weight for the observed BRANCH CURRENTS. "
        "Adds the measured branch currents as soft constraints on the solve; requires "
        "--measure-currents. Unset MIRRORS --anchor-voltage: the solve scales the voltage "
        "residual onto the physics term's current units, so equal numbers mean equal trust, and "
        "both channel kinds carry the same accuracy class unless the sensor model says "
        "otherwise. Pass 0 to anchor voltages only.",
    )
    ap.add_argument(
        "--anchor-schedule",
        default=None,
        help="Per-stage anchoring weight SCHEDULE (injection decode; requires --measure-currents "
        "since it anchors currents too). Comma-separated 'stage:weight' pairs by stage name — "
        "'stage:w' sets BOTH anchors, 'stage:wv/wi' sets voltage/current separately (e.g. "
        "'placement:1/0.1' = noise-calibrated CT trust). Example: "
        "'full:0,random_50:0,random_25:1,random_10:5,placement:10,placement_anneal:20' (overriding "
        "--anchor-voltage/--anchor-current); unlisted stages inherit the global weights. The "
        "idea: low weight while many nodes are observed so the injection head keeps learning, "
        "ramping to measurement-dominant at the sparse placement so the decode leans on the few "
        "sensors it has. A 'lo~hi' side randomizes the weight per training batch; --anchor-fixed "
        "collapses such ranges to their midpoint.",
    )
    ap.add_argument(
        "--eval-anchor",
        default=None,
        help="Anchor weights 'wv' or 'wv/wi' that EVALUATION and INFERENCE deploy, "
        "independent of what training used. With an all-zero --anchor-schedule this is the "
        "model-first regime: the network must carry the state during training (the loss "
        "charges it for every injection error, with no solve to absorb them) and the anchored "
        "solve is applied only to the finished model. Recorded in the final checkpoint, so a "
        "reload deploys it.",
    )
    ap.add_argument(
        "--anchor-draw",
        choices=["uniform", "log_uniform"],
        default="uniform",
        help="How a randomized anchor range is drawn per batch. 'uniform' spreads the WEIGHT "
        "evenly, which spreads the informativeness very unevenly — the predicted injections "
        "only carry gradient where the anchor is weak. 'log_uniform' spreads the draw over "
        "decades of weight so weak-anchor batches are common enough to train on.",
    )
    ap.add_argument(
        "--anchor-weighting",
        choices=["manual", "noise_calibrated"],
        default="manual",
        help="Injection decode only: how the anchor weights are CONSTRUCTED. 'manual' (default) "
        "reads --anchor-voltage/--anchor-current/--anchor-schedule as dimensionless relative "
        "trusts (the voltage weight scaled by the operator, the current weight raw — one "
        "number trusts a noisy CT far more than a noisy voltage sensor). 'noise_calibrated' "
        "weights every anchored channel by its inverse variance, sigma_I^2 / sigma_meas^2: the "
        "injection prior sigma_I from --prior-rel-sigma times the training-set RMS injection, "
        "sigma_meas from the sensor's accuracy class at that order (the --noise-class model's own "
        "sigmas; a tiny relative sigma when noise is off), so voltage and CT anchors carry "
        "commensurate weights and the CT amplification disappears. The anchor weights then "
        "MULTIPLY the calibrated weight (1 = noise-optimal, 0 = off) — a schedule such as "
        "'random_25:1,placement:1' is the calibrated default.",
    )
    ap.add_argument(
        "--prior-rel-sigma",
        type=str,
        default="1",
        help="--anchor-weighting noise_calibrated only: the assumed prior error of the "
        "PREDICTED injection, relative to the training-set RMS injection of each order "
        "(PhysicsConfig.prior_injection_rel_sigma). 1 = a prior that knows nothing (the "
        "zero-injection prior); a network with a 10 %% injection error is 0.1. Larger values "
        "trust the measurements more. A comma-separated list gives one value per order.",
    )
    ap.add_argument(
        "--per-unit",
        action="store_true",
        help="Express the complex state in PER-UNIT before encoding and back after "
        "decoding (NormalizationConfig.per_unit): every feature, every normalizer fit and "
        "the currents the injection decode predicts live over the per-row base voltage / "
        "the per-unit current base, while the physics keeps solving and reporting in "
        "absolute units. A feeder's rows do not share one voltage class, so an absolute "
        "per-order statistic is set by that bimodal row population rather than by the "
        "variation between scenarios, and the fundamental's signal ends up a few percent "
        "of the feature scale. Default: the absolute-volt frame.",
    )
    ap.add_argument(
        "--harmonic-weight",
        default="physical",
        help="Per-order loss priority: 'physical' (default, weight proportional to the "
        "order's own magnitude scale), 'uniform', or 'fundamental:<weight>' to put an "
        "explicit weight on h=1 and 1 on the rest. The blended loss is dominated by the "
        "orders with the largest RELATIVE error, which on a distribution feeder are the "
        "harmonics, so the fundamental can carry a few percent of the gradient while being "
        "the quantity a deployment cares about.",
    )
    ap.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Train directly at the final placement scarcity, with the whole ladder's epoch "
        "budget in that one stage, instead of walking the observed-fraction ladder. The "
        "measurement-scarcity curriculum is a hypothesis about optimization, not a property "
        "of the estimator, and this is how it is tested at equal budget: the datasets, the "
        "normalizer fit and the deployed placement are unchanged.",
    )
    ap.add_argument(
        "--direct-random-fraction",
        type=float,
        default=None,
        help="With --no-curriculum: train every stage (and the anneal) on RANDOM sensor sets "
        "observing this fraction of the buses per sample, instead of the deployed placement "
        "— a network not tied to one placement. The deployed placement is still used to "
        "fit the linear prior and in the evaluation.",
    )
    ap.add_argument(
        "--encoding",
        choices=["trig", "cartesian"],
        default="trig",
        help="Split-real encoding of the STATE the model reads and the loss is computed on "
        "(EncodingConfig.kind). 'trig' (default) is magnitude plus a wrap-free direction, "
        "whose magnitude channel isolates the small per-scenario voltage deviations; "
        "'cartesian' is (re, im), in which the nodal law and the measurement model are both "
        "LINEAR, so a model has no polar-to-Cartesian map to learn before it can express "
        "them. The loss is computed on the Cartesian reconstruction either way.",
    )
    ap.add_argument(
        "--injection-encoding",
        choices=["trig", "cartesian"],
        default=None,
        help="Injection decode only: the PARAMETERISATION of the injection head "
        "(EncodingConfig.injection_kind), independent of the state encoding the model "
        "inputs use. 'cartesian' predicts (re, im) per order — the space in which the "
        "nodal law I = Y*V is linear and a zero-injection row is an ordinary point, "
        "rather than an exact zero of a log-magnitude channel. Default: the state "
        "encoding (unchanged behaviour).",
    )
    ap.add_argument(
        "--plain-decode-weight",
        type=float,
        default=0.0,
        help="Injection decode only: add this times the voltage loss of the PLAIN "
        "(slack-only) decode beside the anchored one while a stage anchors "
        "(LossConfig.plain_decode_weight). The anchored solve is flat in the predicted "
        "injections wherever the measurements pin the state; the plain term keeps "
        "supervising the network as a prior, and val_pu_mae_unobs_plain reports what the "
        "network alone achieves. 0 (default) = off; no effect while anchors are off "
        "(the plain decode IS the decode there).",
    )
    ap.add_argument(
        "--plain-decode-observed-weight",
        type=float,
        default=None,
        help="Row weights of the plain-decode term (LossConfig.plain_decode_observed_weight): "
        "every unobserved row weighs 1 and every observed row this value, so an injection "
        "error the meters would see (misplaced across a meter) costs more than one between "
        "electrically close unobserved nodes. Default: the rows of the voltage loss "
        "(--supervise-rows).",
    )
    ap.add_argument(
        "--prior-distillation",
        type=float,
        default=0.0,
        help="Injection decode only: add this times the injection loss of the predicted "
        "nodal injections against a fitted LINEAR INJECTION PRIOR's predictions "
        "(LossConfig.prior_distillation_weight). The prior is the closed-form least-squares "
        "map from this run's own measurements onto I(h)=Y(h)*V(h), fitted on the TRAIN "
        "split of each pass that distils, at the deployed placement, and carried in every "
        "checkpoint. "
        "It exists because neither other objective builds the injection map: the anchored "
        "solve is flat in the directions the measurements determine, and the TRUE injections "
        "are not identifiable from a sparse sensor set. The prior is a STEPPING STONE, not "
        "the target — it has an accuracy floor of its own and the deployed physics objective "
        "is what a run is judged on, so anneal the term away with "
        "--prior-distillation-schedule. The injection residual dwarfs the voltage one, so 1 "
        "already makes distillation the dominant term; read val_prior_distillation_loss "
        "rather than treating the weight as a share. 0 (default) = off, no prior fitted.",
    )
    ap.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="AdamW denominator term (OptimConfig.eps, default 1e-8 as in PyTorch). Under "
        "anchoring the per-parameter gradients can be of that order; a smaller value (e.g. "
        "1e-12) keeps such updates at the nominal learning rate.",
    )
    ap.add_argument(
        "--supervise-rows",
        default="all",
        choices=["all", "unobserved"],
        help="Rows the voltage loss supervises (LossConfig.supervise_rows): 'all' (default) "
        "or 'unobserved' — only the rows the batch's mask leaves unmeasured, the rows the "
        "estimator exists for (the anchored solve pins the measured ones).",
    )
    ap.add_argument(
        "--prior-distillation-schedule",
        default=None,
        help="Per-stage override of --prior-distillation, 'stage:weight,...' (e.g. "
        "'placement:1,placement_anneal:0' — build the injection map against the linear "
        "prior, then finish on the deployed physics objective alone). Stages not named "
        "inherit the global weight. Stamped into the serialized config, so the anneal is "
        "part of what reproduces the run.",
    )
    ap.add_argument(
        "--physics-inputs",
        default="",
        help="Comma-separated PHYSICS-DERIVED input priors appended to every node token "
        "(PhysicsConfig.input_priors; either decode): 'kcl' feeds each row the injection its "
        "observed neighbourhood implies through Y(h) (plus how complete that sum is), "
        "'linear_state' the measurement-anchored solve of the zero-injection prior — the best "
        "linear estimate the measurements and the known network give, so the model learns the "
        "nonlinear correction on top. Example: --physics-inputs kcl,linear_state. Default "
        "none (byte-identical).",
    )
    ap.add_argument(
        "--emission-persistence",
        choices=["scenario", "device", "class"],
        default="scenario",
        help="the snapshot pass only: 'scenario' (default) redraws every device's emission fraction, "
        "angle and load-dependence in every snapshot; 'device' draws them once per device "
        "for the whole dataset (pgml.scenarios se_random_scenario_config "
        "emission_persistence) — each device keeps its signature, so its harmonic is a "
        "stable function of its own loading; 'class' makes the law (floor, angle, slope) "
        "a constant of the device's consumer class — the same in every dataset — while "
        "the emission level stays a device property. Judge a 'device' run on a population "
        "drawn with another seed.",
    )
    ap.add_argument(
        "--emission-fraction",
        default=None,
        help="Range 'lo,hi' of the per-device IEC 61000-3-2 emission fraction (the level "
        "of every device's harmonic current; preset: 0,2 for snapshots, 0,1 for "
        "sequences). Lower it for a population whose devices are mostly linear. "
        "Default: the preset's range.",
    )
    ap.add_argument(
        "--background-distortion",
        default=None,
        help="Upstream harmonic background at the grid's source, per order and per unit "
        "of the fundamental voltage, e.g. '3:0.0025,5:0.007@-40,7:0.01' ('@deg' sets the "
        "order's phase in the sequence frame; pgml.scenarios BackgroundHarmonicConfig). "
        "Applied to every dataset of the run — the distortion "
        "a feeder inherits from the supplying network, shared by all of its devices. "
        "Default: none (byte-identical to before the option existed).",
    )
    ap.add_argument(
        "--background-drift",
        type=float,
        default=0.0,
        help="Log-magnitude std of the background's slow shared drift (0 = fixed level).",
    )
    ap.add_argument(
        "--background-drift-phase-deg",
        type=float,
        default=0.0,
        help="Angle std [deg] of the background's slow shared drift.",
    )
    ap.add_argument(
        "--background-sc-mva",
        type=float,
        default=20.0,
        help="Source strength [MVA] of the background's Thevenin source; a near-ideal "
        "grid Source needs a strength far above V^2/R_source for the level to appear.",
    )
    ap.add_argument(
        "--harmonic-slack",
        choices=["norton", "measured"],
        default="norton",
        help="How the metered slack enters the HARMONIC solves of the decode "
        "(PhysicsConfig.harmonic_slack): 'norton' (default) leaves the harmonics floating "
        "against an undistorted supply; 'measured' holds the slack rows at their MEASURED "
        "harmonic voltage at every order — the supply's own distortion enters through the "
        "measurement (a feeder whose harmonic voltage is dominated by its supply).",
    )
    ap.add_argument(
        "--load-shunt",
        choices=["none", "opendss", "motor"],
        default=None,
        help="Harmonic device shunt (pgml load_shunt) of the DATA generation AND of the "
        "estimator's decode Y(h) (PhysicsConfig.load_shunt), set together so both model "
        "the same network. The decode always builds the shunt from the nameplate. "
        "Default: unset — pgml's default model in both, the data on its default "
        "operating-point basis (unchanged behaviour).",
    )
    ap.add_argument(
        "--load-shunt-basis",
        choices=["nameplate", "operating_point"],
        default=None,
        help="Basis of the DATA generation's device shunt (needs --load-shunt; default "
        "nameplate, the basis on which decode and data Y(h) agree). operating_point "
        "follows each scenario's loading; the decode cannot (one Y(h) per batch), so its "
        "Y(h) then differs from the data's by the per-scenario operating-point shunt term.",
    )
    ap.add_argument(
        "--device-capacitance",
        default=None,
        help="Grid build: add device capacitance (EMC / X-capacitors, inverter filters) "
        "as one seeded wye ShuntAppliance per load and PV generator, sized per kW of "
        "rated power, from a YAML with a 'device_capacitance:' block (template "
        "run/configs/pgl_device_capacitance.yaml, ranges with sources). Written into "
        "grid.json, so the estimator's Y(h) contains it. Default: none.",
    )
    ap.add_argument(
        "--source-impedance",
        choices=["pgml", "cigre-tb575"],
        default="pgml",
        help="Grid build: the MV source impedance. 'pgml' (default) keeps pgml's import "
        "default; 'cigre-tb575' sets the published benchmark supply (pandapower "
        "create_cigre_network_lv: 100 MVA short-circuit power, R/X = 1, |Z| = 4 ohm at "
        "20 kV).",
    )
    ap.add_argument(
        "--chunk-memory-fraction",
        type=float,
        default=0.5,
        help="Share of the device's free memory the per-scenario system tensors of an "
        "operating-point-shunt solve may take; the generation chunk is capped to fit it "
        "(the solve assembles [chunk, T, H, N, N] matrices). Default 0.5.",
    )
    ap.add_argument(
        "--grid-perturbation",
        default=None,
        help="Network-model error study: a YAML with a 'grid_perturbation:' block "
        "(injection_hse.data.scenarios.GridPerturbationConfig; template "
        "run/configs/pgl_grid_perturbation.yaml) — relative spreads of line R / X / "
        "length, transformer R / X and source impedance. Data is generated on perturbed "
        "deep copies of the grid while the estimator keeps the nominal grid (grid.json). "
        "Select the regime with --perturb-train / --perturb-test. Default: off.",
    )
    ap.add_argument(
        "--perturb-train",
        choices=["nominal", "perturbed"],
        default="nominal",
        help="Training datasets on the nominal grid (default) or on the 'train' pool of "
        "perturbed copies (each copy solves an equal share of the SAME scenarios; every "
        "sample records its copy in the 'grid_copy' record).",
    )
    ap.add_argument(
        "--perturb-test",
        choices=["none", "heldout", "train"],
        default="none",
        help="Generate an extra perturbed TEST dataset of the --eval-dataset kind "
        "(dataset_<kind>_perturbed_test, new scenario seed, never read in training) on "
        "the 'test' pool of copies — perturbations no training copy shares — (heldout) "
        "or on the training copies (train: one systematic perturbation in both, with "
        "--perturb-copies 1). Scored per copy into "
        "evaluation/metrics_perturbed_test_per_copy.csv.",
    )
    ap.add_argument(
        "--perturb-copies",
        type=int,
        default=None,
        help="Override the config's n_copies (copies of the 'train' pool).",
    )
    ap.add_argument(
        "--perturb-test-copies",
        type=int,
        default=None,
        help="Override the config's n_test_copies (copies of the perturbed test dataset).",
    )
    ap.add_argument(
        "--perturb-test-scenarios",
        type=int,
        default=None,
        help="Scenarios of the perturbed test dataset (default: the size of the evaluated "
        "dataset's own test split, --test-fraction of it).",
    )
    ap.add_argument(
        "--keep-tied-rows",
        action="store_true",
        help="Injection decode: keep the rows tied to the slack through closed switches in "
        "the injection space (PhysicsConfig.exclude_tied_rows=False) — the previous "
        "behaviour, for a like-for-like comparison. By default those rows are excluded: no "
        "injection there can move a voltage, and their Y*V is the feeder's current.",
    )
    ap.add_argument(
        "--model",
        choices=["gnn", "dnn", "graphormer"],
        default="gnn",
        help="The spatial model under the temporal (GRU) wrapper: the message-passing GNN "
        "(default; --hidden-dim / --gnn-layers / --input-skip), the grid-wide DNN "
        "(512-512-512, every sensor reaches every row) or the Graphormer (full attention "
        "with the structural bias; --hidden-dim / --gnn-layers set its width and depth).",
    )
    ap.add_argument(
        "--phase-coupling",
        action="store_true",
        help="Add the network's phase-to-phase admittance couplings (mutual line "
        "impedance, the Dyn winding) as graph edges (DataConfig.phase_coupling): the "
        "series graph is three disconnected per-phase components on which two thirds of "
        "the sensor-to-device pairs are unreachable at any depth. Read off pgml's own "
        "admittance stamps; incompatible with --measure-currents. Default off.",
    )
    ap.add_argument(
        "--precondition",
        choices=["none", "plain", "placement"],
        default="none",
        help="Injection decode with --injection-encoding cartesian only: precondition the "
        "gradient of the loss at the injection head's output by the inverse of the loss's "
        "own curvature (LossConfig.preconditioner), read once before training from the "
        "deployed measurement-anchored decode at the placement ('placement') or from the "
        "slack-only decode ('plain'). Forward pass untouched; steep output directions take "
        "Gauss-Newton steps, flat ones (30-43 %% of them on CIGRE LV) are lifted by up to "
        "1/floor^2. Default none (byte-identical).",
    )
    ap.add_argument(
        "--precondition-floor",
        type=float,
        default=0.01,
        help="--precondition only: the spectral floor as a fraction of the largest singular "
        "value over all orders (LossConfig.preconditioner_floor). Default 0.01, the cutoff "
        "the loss-geometry analysis calls 'flat'.",
    )
    ap.add_argument(
        "--prior-residual",
        action="store_true",
        help="Injection decode only: parameterise the decode as the LINEAR INJECTION PRIOR "
        "plus a learned residual (PhysicsConfig.injection_prior_residual) — the prior is "
        "fitted on the TRAIN split of each pass at the deployed placement (the sequence "
        "pass refits it on the sequences), the network's output layer starts at zero, so "
        "training starts exactly at the prior's estimate and learns only what the linear "
        "map cannot represent. Inside a sparser curriculum stage the prior reads only the "
        "placement rows that stage observes.",
    )
    ap.add_argument(
        "--carry-optimizer-state",
        action="store_true",
        help="Carry the optimizer state (AdamW moments and step counts) from each stage into "
        "the next, across the snapshot -> sequence switch too, instead of a fresh optimizer "
        "per stage (injection_hse.config.TrainConfig.carry_optimizer_state). Off (default): fresh.",
    )
    ap.add_argument(
        "--stage-warmup",
        type=int,
        default=0,
        help="Linear learning-rate warmup, in epochs, at the start of every stage after the "
        "first (injection_hse.config.MaskStageConfig.warmup_epochs). 0 (default) = none.",
    )
    ap.add_argument(
        "--prior-fit-states",
        type=int,
        default=4096,
        help="TRAIN states the linear injection prior is fitted on, per pass "
        "(PhysicsConfig.injection_prior_fit_states; a sequence contributes every step). "
        "Default 4096. Memory grows linearly: the state sample plus one [states, "
        "features] float64 block.",
    )
    ap.add_argument(
        "--prior-anchor",
        default=None,
        help="--physics-inputs linear_state only: anchor weights 'wv' or 'wv/wi' of the "
        "prior's own zero-injection solve (PhysicsConfig.prior_anchor_voltage_weight / "
        "prior_anchor_current_weight; an unset '/wi' mirrors wv). Independent of the "
        "decode's --anchor-schedule; under --anchor-weighting noise_calibrated they multiply "
        "the calibrated weight at the zero-prior injection sigma. Default 1/1.",
    )
    ap.add_argument(
        "--activity-scale",
        type=float,
        default=1.0,
        help="Duty-cycle multiplier on the composed sequence devices' availability. The class "
        "presets put a composed aggregate far below the randomized task's loading, so the sequence pass "
        "never reaches the loaded states; raise this to place it in a comparable band.",
    )
    ap.add_argument(
        "--behavioral-coupling",
        type=float,
        default=0.3,
        help="Shared per-scenario activity latent of the composed consumption devices: how "
        "much of the population moves together, which is what keeps the AGGREGATE varying "
        "instead of averaging out over hundreds of devices.",
    )
    ap.add_argument(
        "--cloud-coupling", type=float, default=0.5, help="The PV counterpart."
    )
    ap.add_argument(
        "--anchor-fixed",
        action="store_true",
        help="Collapse every RANGE in --anchor-schedule to its midpoint, so each stage anchors "
        "at ONE fixed weight and nothing is drawn per training batch (the simple regime). Lets "
        "a ranged schedule string be reused verbatim; a schedule that is already fixed is "
        "unaffected.",
    )
    ap.add_argument(
        "--injection-loss-split",
        type=float,
        default=0.0,
        help="Injection decode only: fraction of the supervised loss placed DIRECTLY on the "
        "predicted nodal injection currents (vs the solved voltages): "
        "loss = (1-s)*voltage + s*injection, both in their normalized space with the same "
        "per-harmonic weighting. 0 (default) supervises the voltages only, so the injections "
        "are learned through the solve alone; 0.5 splits evenly. The injection term is "
        "rescaled onto the voltage term's running scale first (--raw-injection-loss opts "
        "out), so s is the fraction the injection term ACTUALLY contributes — verify it "
        "against the logged val_injection_share. Must be < 1.",
    )
    ap.add_argument(
        "--raw-injection-loss",
        action="store_true",
        help="Blend the two supervised terms UNSCALED, reproducing a run from before the "
        "injection term was normalized. The residuals are not commensurate (the injection "
        "one measured 82x-45000x above the voltage one, and the ratio itself moves ~500x "
        "across curriculum stages), so --injection-loss-split then behaves as an on/off "
        "switch rather than a dial: a nominal 0.1 realizes 90-99.98%% of the loss.",
    )
    ap.add_argument(
        "--noise-class",
        default="off",
        choices=["off", "0.1", "0.2", "0.5", "1", "A"],
        help="Measurement noise on every OBSERVED phasor (voltage sensors and, with "
        "--measure-currents, the CT channels — independent streams): the accuracy class "
        "keying the per-class magnitude/phase sigmas (see injection_hse.config.NoiseConfig), or 'A' "
        "for the IEC 61000-4-30 class-A instrument model (injection_hse.config.InstrumentErrorConfig: "
        "gain and phase error held per scenario, channel and order, floor drawn per "
        "reading). The supervised targets and metrics stay clean. 'off' (default) draws "
        "nothing and is byte-identical to the noiseless run.",
    )
    ap.add_argument(
        "--noise-draw-device",
        default="cpu",
        choices=["cpu", "input"],
        help="Where --noise-class draws its noise (injection_hse.config.NoiseConfig.draw_device): "
        "'cpu' (default) on a CPU generator, device-independent and reproducible across "
        "machines; 'input' on the phasors' device (the GPU in training), same distribution, "
        "a device-specific stream, no per-step host work.",
    )
    ap.add_argument(
        "--measure-currents",
        action="store_true",
        help="Feed measured BRANCH CURRENTS as additional model input: every measurement "
        "location (the slack included) meters all its connected branches; the currents are "
        "derived per half-edge from the state via the branch-current operator and enter the "
        "model as dynamic edge observations. The estimate stays voltage-only (currents are "
        "input, not a target). Off (default) = the voltage-only estimator.",
    )
    ap.add_argument(
        "--current-channels",
        default="incident",
        choices=["incident", "upstream"],
        help="With --measure-currents: the per-location CT budget. 'incident' (default) "
        "meters every branch connected to a measurement location; 'upstream' restricts "
        "each location to the single branch toward the slack (one CT per meter, per "
        "phase — the budget device), the slack keeping its incident feeder heads.",
    )
    ap.add_argument(
        "--measure-power",
        action="store_true",
        help="With --measure-currents: also feed the per-order branch-terminal apparent "
        "power S = V·I* as additional edge channels (derivable from V and I, so off by "
        "default; an ablation knob for realistic PQ-meter input sets).",
    )
    ap.add_argument(
        "--sensors",
        default="",
        help="Comma-separated node names/ids of the FULL-HARMONIC placement sensors "
        "(all-digit tokens resolve as node ids). Default: the 3-sensor CIGRE set "
        f"{SENSOR_NODES}. The metered slack is always additional.",
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="GNN hidden width of the temporal-GNN estimator (default 128).",
    )
    ap.add_argument(
        "--gnn-layers",
        type=int,
        default=6,
        help="GNN message-passing depth of the temporal-GNN estimator (default 6).",
    )
    ap.add_argument(
        "--input-skip",
        action="store_true",
        help="Long-range GNN skip: concatenate the encoder embedding into the head so a "
        "deep stack cannot oversmooth the input signal away (recommended with "
        "--gnn-layers >= 9; the per-layer residuals exist regardless).",
    )
    ap.add_argument(
        "--fundamental-nodes",
        default="",
        help="Comma-separated node names/ids carrying a FUNDAMENTAL-ONLY monitor (smart-"
        "meter class): their voltage (and current) is observed at h=1 only, alongside the "
        "full-harmonic placement sensors. Enables the heterogeneous per-order input "
        "channels. Example: --fundamental-nodes 'Bus R11,Bus C12'.",
    )
    ap.add_argument(
        "--fundamental-fraction",
        type=float,
        default=0.0,
        help="In every RANDOM curriculum stage, demote each observed location to a "
        "fundamental-only monitor with this probability (Bernoulli per sample+location), "
        "so the model trains on mixed sensor fleets before a heterogeneous placement "
        "finetune. 0 (default) keeps random stages full-harmonic.",
    )
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Held-out TEST fraction, carved off the seeded scenario permutation disjoint "
        "from train/val. NEVER read during training/early-stop/checkpoint selection — for a "
        "final one-shot generalization estimate only (evaluate with predict_dataset "
        "split='test'). 0 disables it (a two-way train/val split).",
    )
    ap.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep non-converged scenarios in the splits. By default a dataset's "
        "failed (non-converged) scenarios are dropped from train/val/test.",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live Lightning progress bar (per-stage train/val loss + pu-MAE in "
        "the terminal); metrics are still logged to MLflow + the history parquet.",
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="Resume from an existing stage checkpoint (stage_<name>.ckpt, possibly of "
        "another run): its weights + fitted normalizers initialize the model, the "
        "stages up to and including <name> are skipped, and training continues with "
        "the next stage — on THIS run's freshly generated datasets. The checkpoint "
        "must match the model architecture and --orders. It is copied into the new "
        "run's checkpoints/ and recorded in the manifest.",
    )
    ap.add_argument(
        "--start-stage",
        default=None,
        help="First stage to TRAIN (overrides the stage inferred from the --init-from "
        "filename); one of the ladder names, e.g. random_25.",
    )
    ap.add_argument(
        "--no-restore-best",
        action="store_true",
        help="Do NOT keep each stage's best-val_pu_mae weights. By default a stage checkpoints "
        "and continues from its best epoch (and halts on a NaN/inf loss), so a late-training "
        "collapse cannot overwrite the saved model or carry NaN into the next stage.",
    )
    ap.add_argument(
        "--target-full", type=float, default=1e-3, help="Full-obs pu-MAE target."
    )
    ap.add_argument(
        "--target-placement", type=float, default=1e-2, help="Placement pu-MAE target."
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Plateau early stopping: end a stage when the validation all-rows pu-MAE has "
        "not improved by >= --min-delta for this many consecutive validation epochs (0 "
        "disables; --epochs stays the hard cap). Pair with targets set below the achievable "
        "floor so a stage stops as soon as it stops learning.",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum val pu-MAE improvement counted as progress by --patience.",
    )
    ap.add_argument(
        "--allow-tight-memory",
        action="store_true",
        help=f"Proceed even when the coherent preflight peaks above "
        f"{PREFLIGHT_SAFE_FRACTION * 100:.0f}%% of GPU memory. By default that margin is treated as a "
        "FAILURE (a clean probe fits, but real training adds optimizer state + solve workspace "
        "+ allocator fragmentation on top and OOMs late); pair with "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if you override.",
    )
    # run
    ap.add_argument("--name", default="se-full-workflow", help="Experiment / run name.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", default="auto", help="'auto' (CUDA if available), 'cpu', 'cuda'."
    )
    ap.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Training processes/GPUs (data-parallel DDP). Values > 1 REQUIRE an external "
        "per-rank launcher with matching task count — SLURM: sbatch --gres=gpu:N "
        "--ntasks-per-node=N + srun; locally: torchrun --nproc_per_node=N. The effective "
        "batch is held constant (--effective-batch), so accumulation shrinks by the "
        "device count. Runs reproduce bitwise at fixed (seed, devices); different device "
        "counts are statistically equivalent, not bit-identical.",
    )
    ap.add_argument(
        "--orders",
        default=None,
        help="Comma-separated harmonic orders to simulate AND estimate, e.g. "
        "'1,3,5,7,9' (must include 1 and at least one order above 1). "
        "Default: 1..20. Fewer orders shrink the feature width H*C and the "
        "datasets — the memory footprint scales with the order count.",
    )
    ap.add_argument(
        "--scenario-config",
        default=None,
        help="DEVIATION from the calibrated scenario preset: a YAML with a 'random:' "
        "section (pgml.scenarios.ScenarioConfig) and/or a 'coherent:' section "
        "(CoherentSpectrumConfig). The section REPLACES the preset's shape wholesale "
        "(distributions, correlation, unbalance, harmonic reference); the sizes, the "
        "solved orders and the seed stay on the CLI. Example: "
        "run/configs/pgl_se_scenario.yaml. Default: the preset "
        "(injection_hse.data.scenarios.se_random_scenario_config / se_coherent_scenario_config).",
    )
    ap.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Do NOT write the scenario-coverage report (THD / VUF / voltage deviation / "
        "load-scale distributions of the generated randomized dataset). It is written by "
        "default so every run records what operating conditions it actually trained on.",
    )
    ap.add_argument(
        "--strict-datagen",
        action="store_true",
        help="ABORT before training when a generated dataset FAILS validation (a dead "
        "harmonic order, a de-energized or non-finite state, a coherent window outside "
        "the activity band, a declared-varied quantity that does not vary). The verdicts "
        "are printed either way; this only decides whether hours of training start on a "
        "dataset that is already known to be wrong.",
    )
    ap.add_argument(
        "--expected-ranges",
        default=None,
        help="Expectations YAML adding the measured-population layer to the dataset "
        "validation (run/configs/expected_ranges.yaml): a per-order median off by more "
        "than 3x warns, more than 10x fails.",
    )
    ap.add_argument(
        "--collect-gradients",
        action="store_true",
        help="Sample per-layer-group gradient norms during training "
        "(injection_hse.analysis.GradientNormCollector) and write gradient_norms.parquet "
        "next to the training history — feeds the analysis gradient-dynamics "
        "figure.",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do NOT write the evaluation plot set after training (evaluation runs by default).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip data generation + training; only (re)write the evaluation plot set for an "
        "existing run at out_dir. Equivalent to `python -m injection_hse.evaluation out_dir`.",
    )
    ap.add_argument(
        "--eval-dataset",
        default="coherent",
        choices=["coherent", "random"],
        help="Which dataset split to evaluate (default: coherent).",
    )
    ap.add_argument(
        "--heatmap-phases",
        default=None,
        help="Comma-separated phases for the per-phase error heatmaps (e.g. 'a,b,c'); "
        "default: every line phase present in the grid.",
    )
    args = ap.parse_args()
    heatmap_phases = parse_phases(args.heatmap_phases)

    if getattr(args, "prior_anchor", None) and "linear_state" not in _physics_inputs(
        args
    ):
        raise SystemExit(
            "--prior-anchor weights the linear-state prior's zero-injection solve; add "
            "linear_state to --physics-inputs (e.g. --physics-inputs linear_state)."
        )
    # measurement anchoring is a property of the injection decode's network solve
    if (
        args.anchor_voltage or args.anchor_current or args.anchor_schedule
    ) and args.decode != "injection":
        raise SystemExit(
            "--anchor-voltage / --anchor-current / --anchor-schedule require --decode injection "
            "(they constrain the injection decode's network solve; the voltage decode has no "
            "solve to anchor)."
        )
    if (
        args.prior_distillation or args.prior_distillation_schedule
    ) and args.decode != "injection":
        raise SystemExit(
            "--prior-distillation / --prior-distillation-schedule require --decode injection "
            "(the term supervises the injection decode's predicted nodal currents; the "
            "voltage decode produces none)."
        )
    if getattr(args, "prior_residual", False) and args.decode != "injection":
        raise SystemExit(
            "--prior-residual requires --decode injection (the prior supplies the nodal "
            "injections the decode solves from)."
        )
    if getattr(args, "precondition", "none") != "none":
        if args.decode != "injection":
            raise SystemExit(
                "--precondition requires --decode injection (it preconditions the "
                "injection head's output)."
            )
        if getattr(args, "injection_encoding", None) != "cartesian":
            raise SystemExit(
                "--precondition requires --injection-encoding cartesian (the decode chain "
                "is affine in the head's output only there)."
            )
    # the RESOLVED weight, not the raw flag: an unset --anchor-current mirrors the voltage
    # one, so a voltage-anchored run needs the CTs it is about to constrain against.
    if (
        _anchor_current_weight(args) or args.anchor_schedule
    ) and not args.measure_currents:
        raise SystemExit(
            "--anchor-current / --anchor-schedule require --measure-currents (the schedule "
            "anchors branch currents too; there are none to anchor to otherwise)."
        )
    # The DEPLOYED anchor obeys the same rules, and is validated HERE rather than where it
    # is applied: it takes effect after the last stage, so a bad value would otherwise end
    # a finished training run instead of never starting it.
    if args.eval_anchor:
        if args.decode != "injection":
            raise SystemExit(
                "--eval-anchor requires --decode injection (it deploys the injection "
                "decode's anchored solve; the voltage decode has no solve to anchor)."
            )
        wv, _, wi = args.eval_anchor.partition("/")
        try:
            weights = (float(wv), float(wi) if wi else float(wv))
        except ValueError:
            raise SystemExit(
                f"--eval-anchor must be 'wv' or 'wv/wi' with numeric weights, got "
                f"{args.eval_anchor!r}."
            ) from None
        if min(weights) < 0.0:
            raise SystemExit("--eval-anchor weights must be >= 0.")
        if weights[1] and not args.measure_currents:
            raise SystemExit(
                "--eval-anchor implies a CURRENT anchor weight (an unset '/wi' mirrors "
                "the voltage one) and needs --measure-currents; there are no branch "
                "currents to anchor to otherwise. Pass '<wv>/0' to deploy voltage "
                "anchoring alone."
            )
    check_perturbation_flags(args)
    if args.load_shunt_basis is not None and args.load_shunt is None:
        raise SystemExit(
            "--load-shunt-basis selects the basis of the --load-shunt model; name the "
            "model too (e.g. --load-shunt opendss --load-shunt-basis nameplate)."
        )
    if args.injection_loss_split and args.decode != "injection":
        raise SystemExit(
            "--injection-loss-split requires --decode injection (it supervises the "
            "injection decode's predicted nodal currents; the voltage decode predicts none)."
        )
    if not 0.0 <= args.injection_loss_split < 1.0:
        raise SystemExit("--injection-loss-split must be in [0, 1).")

    warnings.filterwarnings("ignore")  # the synthesized-GMR notice is expected here
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # re-evaluate a finished run without regenerating data or retraining
    if args.eval_only:
        from pgml.schemas.grid_schema import Grid

        from injection_hse.evaluation import evaluate_run

        print(f"Evaluating finished run at {out_dir.resolve()} ...", flush=True)
        eval_dir = evaluate_run(
            out_dir,
            dataset=args.eval_dataset,
            device="cpu",
            heatmap_phases=heatmap_phases,
        )
        evaluate_perturbation(
            out_dir,
            Grid.model_validate_json((out_dir / "grid.json").read_text()),
            eval_dir,
        )
        artifacts = sorted(p.name for p in eval_dir.iterdir())
        print(f"  evaluation -> {eval_dir.resolve()}\n  artifacts: {artifacts}")
        report_split_metrics(eval_dir)
        return
    device = resolve_device(args.device)
    configure_runtime(device)
    args.accelerator = "gpu" if device.type == "cuda" else "cpu"
    args.devices = max(1, int(args.devices))
    rank = launch_rank()
    if args.devices == 1 and rank != 0:
        raise SystemExit(
            "launched with multiple tasks (rank > 0) but --devices 1: pass "
            "--devices <ntasks> so training and the launcher agree."
        )
    args.mlflow_uri = f"file:{(out_dir / 'mlruns').resolve()}"
    dtype = DTYPES[args.dtype]

    # default: ALL orders 1..20 (even AND odd); the injected orders are those above 1
    orders = parse_orders(args.orders)
    injected = [h for h in orders if h > 1]
    # the profiled-sequence pass (COMPOSED longer window) is opt-in via --n-profile-scenarios. Periodic time
    # features are enabled whenever COMPOSED coherent data participates (composed the sequence pass, or
    # the profiled-sequence pass — both carry an absolute time axis); the ONE module then gains the time-feature
    # input for EVERY task (the timeless the snapshot pass — and a fingerprint the sequence pass — feed the all-zero
    # vector). With a fingerprint the sequence pass and no the profiled-sequence pass the data is timeless: features stay off.
    composed_sequences = args.coherent_mode == "composed"
    profiled_pass = args.n_profile_scenarios > 0
    args.time_features = composed_sequences or profiled_pass
    # optional user-provided scenario shape (distributions / correlation / unbalance / THD)
    args.scenario_random, args.scenario_coherent = load_scenario_config(
        args.scenario_config
    )

    if args.grid:
        grid = load_grid(Path(args.grid))
        label = grid.metadata.name or Path(args.grid).stem
        extra = f"{len(grid.measurement_devices)} installed meters"
    else:
        grid, n_pv = build_grid(args.pv_fraction)
        label = "CIGRE LV (3-ph)"
        extra = f"{n_pv} PV"
    from pgml.assembly import node_phase_index

    # grid-model options, applied before the grid is persisted (grid.json) so the
    # estimator's operators see them
    args.grid_model = apply_grid_model(grid, args)
    n_rows = node_phase_index(grid).size
    print(
        f"{label}: {len(grid.nodes)} nodes / {n_rows} node-phase rows, {extra} | "
        f"orders {orders[0]}..{orders[-1]} (injected {injected[0]}..{injected[-1]}) | "
        f"device {device.type} | {args.dtype}",
        flush=True,
    )
    print(f"  scenario recipe: {scenario_recipe_identity(args)}", flush=True)
    if args.grid_model is not None:
        if "source_impedance" in args.grid_model:
            print("  source impedance: CIGRE TB 575 (100 MVA, R/X = 1)", flush=True)
        if "device_capacitance" in args.grid_model:
            shunts = args.grid_model["device_capacitance"]["shunts"]
            total = sum(r["c_per_phase_uf"] for r in shunts)
            print(
                f"  device capacitance: {len(shunts)} shunts, "
                f"{total:.1f} uF per phase in total",
                flush=True,
            )
    shunt_record = load_shunt_record(args)
    if shunt_record is not None:
        data, decode = shunt_record["data"], shunt_record["decode"]
        print(
            f"  harmonic load shunt: data {data['model']}/{data['basis']}, decode "
            f"{decode['model']}/{decode['basis']}",
            flush=True,
        )
        if data["basis"] == "operating_point" and data["model"] != "none":
            print(
                "  NOTE: decode and data Y(h) differ by the per-scenario operating-point "
                "shunt term (the decode builds its shunt from the nameplate).",
                flush=True,
            )
    print(
        "  state feature frame: "
        + (
            "per-unit (per-row base voltage; currents over s(h)*V_ref^2/V_base)"
            if args.per_unit
            else "absolute (volts)"
        ),
        flush=True,
    )
    # the grid's own meters are the placement unless the CLI names the sensors explicitly
    if not args.sensors and grid.measurement_devices:
        from injection_hse.data import MeasurementModel

        args.sensors = ",".join(
            str(n) for n in MeasurementModel.from_grid(grid).voltage_nodes
        )
        print(f"  placement from the grid's meters: {args.sensors}", flush=True)
    # the switch configurations this run trains across (None = the grid as given)
    topology = resolve_topology(grid, args)
    if topology is not None:
        from injection_hse.data import TopologySet

        # the grid the run is built on is the UNION topology (every switchable branch
        # closed): it carries all branches, so the graph, the placement and the metered
        # half-edges are the same for every configuration. A variant only opens switches.
        grid = TopologySet.from_config(topology).union_grid(grid)
        print(
            f"  topologies ({len(topology.variants)}): "
            f"{TopologySet.from_config(topology).describe(grid)}",
            flush=True,
        )
    # persist the grid so evaluation is self-contained from the run dir
    # (`python -m injection_hse.evaluation out_dir`): the dataset stores only the row layout.
    if rank == 0:
        (out_dir / "grid.json").write_text(grid.model_dump_json())
    # data on perturbed copies of the grid (the estimator keeps the nominal grid.json)
    plan = perturbation_plan(grid, args)
    if plan is not None:
        for label, copies in (("train", plan["train"]), ("test", plan["test"])):
            if copies is None:
                continue
            mags = ", ".join(f"{p.pool}{p.index}:{p.magnitude:.3g}" for p in copies)
            print(
                f"  grid perturbation ({label} data, {plan['config'].distribution}): "
                f"{len(copies)} cop{'y' if len(copies) == 1 else 'ies'}, RMS log-factor "
                f"{mags}",
                flush=True,
            )
        sizes = [args.n_random, args.n_scenarios] + (
            [args.n_profile_scenarios] if args.n_profile_scenarios else []
        )
        if plan["train"] is not None and min(sizes) < len(plan["train"]):
            raise SystemExit(
                f"every training dataset needs at least one scenario per grid copy "
                f"({len(plan['train'])}); got sizes {sizes}."
            )
    train_perturbation = (
        {}
        if plan is None or plan["train"] is None
        else {"perturbations": plan["train"], "perturbation_config": plan["config"]}
    )

    # -- datasets: rank 0 generates, the other ranks wait on the sentinel (they cannot
    # use a torch.distributed barrier before the first fit initializes the group) --
    sentinel = out_dir / ".datasets_ready"
    if rank == 0:
        sentinel.unlink(missing_ok=True)
        # With a switchable topology the requested sizes are the run TOTAL, split evenly over
        # the configurations (each gets its own dataset directory + seed stream).
        n_variants = len(topology.variants) if topology is not None else 1
        if n_variants > 1:
            args.n_random = max(1, args.n_random // n_variants)
            args.n_scenarios = max(1, args.n_scenarios // n_variants)
            args.n_profile_scenarios = (
                max(1, args.n_profile_scenarios // n_variants)
                if args.n_profile_scenarios
                else 0
            )
            print(
                f"Per topology: {args.n_random} snapshots + {args.n_scenarios} coherent "
                f"sequences (x{n_variants} configurations).",
                flush=True,
            )

        def _generate(generator, root_name, **kwargs):
            if topology is None:
                return generator(
                    grid, orders, injected, args, out_dir, device, dtype, **kwargs
                )
            return generate_topology_datasets(
                generator,
                grid,
                topology,
                orders,
                injected,
                args,
                out_dir,
                device,
                dtype,
                root_name=root_name,
            )

        print(
            f"Generating RANDOMIZED dataset (the snapshot pass): {args.n_random} Sobol snapshots ...",
            flush=True,
        )
        random_path = _generate(
            generate_random_dataset, "dataset_random", **train_perturbation
        )
        print(f"  -> {random_path.resolve()}")
        if not args.skip_coverage:
            write_scenario_coverage(
                _first_variant(random_path, topology), grid, out_dir
            )
        # every generated dataset is validated before it is trained on; the verdicts are
        # printed here and recorded in the dataset's own validation.json
        clean = validate_generated(
            _first_variant(random_path, topology), grid, args, label="the snapshot pass"
        )
        # report the EFFECTIVE anchor: a --scenario-config `coherent.start_time` takes
        # precedence over --profile-start, and the banner is how a run records what it did
        coherent_override = getattr(args, "scenario_coherent", None)
        effective_start = (
            getattr(coherent_override, "start_time", None) or args.profile_start
        )
        if args.start_time_range:
            effective_start = f"per sequence in [{args.start_time_range}) ({args.start_time_sampling})"
        coh_detail = (
            f"composed step={args.coherent_step_s}s start={effective_start}"
            if composed_sequences
            else f"fingerprint modes={args.n_modes} dwell={args.dwell}"
        )
        print(
            f"Generating COHERENT dataset (the sequence pass): B={args.n_scenarios} T={args.n_steps} "
            f"{coh_detail} ...",
            flush=True,
        )
        coherent_path = _generate(
            generate_coherent_dataset, "dataset_coherent", **train_perturbation
        )
        print(f"  -> {coherent_path.resolve()}")
        if not args.skip_coverage:
            write_scenario_coverage(
                _first_variant(coherent_path, topology),
                grid,
                out_dir,
                stem="scenario_coverage_taskb",
            )
        clean &= validate_generated(
            _first_variant(coherent_path, topology),
            grid,
            args,
            label="the sequence pass",
        )
        profile_path = None
        if profiled_pass:
            print(
                f"Generating COMPOSED longer-window dataset (the profiled-sequence pass): "
                f"B={args.n_profile_scenarios} T={args.profile_steps} "
                f"step={args.profile_step_s}s start={effective_start} ...",
                flush=True,
            )
            profile_path = _generate(
                generate_profile_dataset, "dataset_profile", **train_perturbation
            )
            print(f"  -> {profile_path.resolve()}")
            clean &= validate_generated(
                _first_variant(profile_path, topology),
                grid,
                args,
                label="the profiled-sequence pass",
            )
        perturbed_test_path = None
        if plan is not None and plan["test"] is not None:
            n_test = perturbed_test_size(args, len(plan["test"]))
            kind = args.eval_dataset
            size_field = "n_scenarios" if kind == "coherent" else "n_random"
            print(
                f"Generating the PERTURBED TEST dataset ({kind}, {n_test} scenarios on "
                f"{len(plan['test'])} {plan['test'][0].pool}-pool copies) ...",
                flush=True,
            )
            perturbed_test_path = (
                generate_coherent_dataset
                if kind == "coherent"
                else generate_random_dataset
            )(
                grid,
                orders,
                injected,
                argparse.Namespace(**{**vars(args), size_field: n_test}),
                out_dir,
                device,
                dtype,
                dataset_path=out_dir / f"dataset_{kind}_perturbed_test",
                seed=int(args.seed) + _PERTURBED_TEST_SEED_OFFSET,
                perturbations=plan["test"],
                perturbation_config=plan["config"],
            )
            print(f"  -> {perturbed_test_path.resolve()}")
            clean &= validate_generated(
                perturbed_test_path, grid, args, label="the perturbed test set"
            )
        if not clean:
            raise SystemExit(
                "--strict-datagen: a generated dataset FAILED validation (see the "
                "verdicts above and validation.json in the dataset directories). "
                "Fix the generation recipe, or drop --strict-datagen to train on it "
                "anyway."
            )
        sentinel.write_text("ok")
    else:
        wait_for_sentinel(sentinel, description="rank-0 dataset generation")
        random_path = out_dir / "dataset_random"
        coherent_path = out_dir / "dataset_coherent"
        profile_path = (out_dir / "dataset_profile") if profiled_pass else None
        perturbed_test_path = (
            out_dir / f"dataset_{args.eval_dataset}_perturbed_test"
            if plan is not None and plan["test"] is not None
            else None
        )

    # -- configs (persist both, the reproducible source of truth) --
    stages_a, stages_b = _snapshot_stages(args), _sequence_stages(args)
    stages_c = _profiled_sequence_stages(args) if profiled_pass else None
    if getattr(args, "no_curriculum", False):
        stages_a, stages_b = (
            _direct_stages(stages_a, args),
            _direct_stages(stages_b, args),
        )
        if stages_c is not None:
            stages_c = _direct_stages(stages_c, args)
    # the final cosine-anneal placement finetune (--anneal-epochs > 0) is the LAST stage of the
    # whole curriculum — appended to the profiled-sequence pass when present, else the sequence pass.
    if args.anneal_epochs > 0:
        anneal = _direct_random(_anneal_stage(args), args)
        if stages_c is not None:
            stages_c = [*stages_c, anneal]
        else:
            stages_b = [*stages_b, anneal]
    # per-stage measurement-anchoring schedule (--anchor-schedule): stamp the per-stage weights
    # onto the matching stages of every task pass BEFORE building the configs (so the schedule
    # is part of the reproducible, serialized config). None = the global weights apply to all.
    anchor_schedule = parse_anchor_schedule(
        getattr(args, "anchor_schedule", None),
        fixed=getattr(args, "anchor_fixed", False),
    )
    if anchor_schedule:
        leftover: set[str] = set(anchor_schedule)
        stages_a, m = apply_anchor_schedule(stages_a, anchor_schedule)
        leftover &= m
        stages_b, m = apply_anchor_schedule(stages_b, anchor_schedule)
        leftover &= m
        if stages_c is not None:
            stages_c, m = apply_anchor_schedule(stages_c, anchor_schedule)
            leftover &= m
        if leftover:
            print(
                f"WARNING: --anchor-schedule names no stage matched anywhere: "
                f"{sorted(leftover)} (known stages: full, random_*, placement, "
                f"placement_anneal).",
                flush=True,
            )
    # per-stage prior-distillation schedule (--prior-distillation-schedule): the anneal of
    # the distillation term, stamped onto the matching stages of every task pass so it is
    # part of the reproducible config. None = the global weight applies to every stage.
    distill_schedule = parse_stage_weights(
        getattr(args, "prior_distillation_schedule", None),
        flag="--prior-distillation-schedule",
    )
    if distill_schedule:
        field = "prior_distillation_weight"
        leftover = set(distill_schedule)
        stages_a, m = apply_stage_weights(stages_a, distill_schedule, field=field)
        leftover &= m
        stages_b, m = apply_stage_weights(stages_b, distill_schedule, field=field)
        leftover &= m
        if stages_c is not None:
            stages_c, m = apply_stage_weights(stages_c, distill_schedule, field=field)
            leftover &= m
        if leftover:
            print(
                f"WARNING: --prior-distillation-schedule names no stage matched anywhere: "
                f"{sorted(leftover)} (known stages: full, random_*, placement, "
                f"placement_anneal).",
                flush=True,
            )

    # --stage-warmup: a linear learning-rate warmup at the start of every stage after the
    # first (the first starts from a fresh optimizer at the warmup's low end anyway when the
    # flag is combined with --carry-optimizer-state, which keeps the moments across borders)
    warmup = int(getattr(args, "stage_warmup", 0) or 0)
    if warmup:
        stages_a, stages_b, stages_c = _with_stage_warmup(
            warmup, stages_a, stages_b, stages_c
        )
    if args.init_from is not None or args.start_stage is not None:
        # resume: keep only the stages still to train; the target-loss ramp stays the
        # one those stages would have at their ladder positions in a full run. The
        # ladder spans EVERY enabled pass, so a checkpoint at the last sequence stage
        # resumes into the profiled-sequence pass rather than reading as "nothing left to train".
        ladders = [[s.name for s in stages_a], [s.name for s in stages_b]]
        if profiled_pass:
            ladders.append([s.name for s in stages_c])
        kept = resume_stage_split(
            *ladders, init_from=args.init_from, start_stage=args.start_stage
        )
        stages_a = [s for s in stages_a if s.name in kept[0]]
        stages_b = [s for s in stages_b if s.name in kept[1]]
        if profiled_pass:
            stages_c = [s for s in stages_c if s.name in kept[2]]
    cfg_a = _base_config(
        args,
        random_path,
        orders,
        stages_a,
        out_dir,
        f"{args.name}-A",
        topology=topology,
    )
    cfg_b = _base_config(
        args,
        coherent_path,
        orders,
        stages_b,
        out_dir,
        f"{args.name}-B",
        topology=topology,
    )
    cfg_c = None
    if profiled_pass:
        cfg_c = _base_config(
            args,
            profile_path,
            orders,
            stages_c,
            out_dir,
            f"{args.name}-C",
            topology=topology,
        )
    if rank == 0:
        (out_dir / "experiment_snapshots.yaml").write_text(
            yaml.safe_dump(cfg_a.model_dump())
        )
        (out_dir / "experiment_sequences.yaml").write_text(
            yaml.safe_dump(cfg_b.model_dump())
        )
        if cfg_c is not None:
            (out_dir / "experiment_profiled_sequences.yaml").write_text(
                yaml.safe_dump(cfg_c.model_dump())
            )
        generation = data_generation_record(args, plan)
        if generation:
            (out_dir / "data_generation.yaml").write_text(
                yaml.safe_dump(generation, sort_keys=False)
            )

    init_ckpt_name = None
    if args.init_from is not None:
        # keep the resumed run self-contained: the source checkpoint is copied into the
        # run's own checkpoint dir (it is also the weight-delta baseline of the first
        # trained stage in the analysis figures).
        import shutil

        src = Path(args.init_from)
        if not src.is_file():
            raise SystemExit(f"--init-from: no such checkpoint {src}")
        if rank == 0:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ckpt_dir / src.name)
        init_ckpt_name = src.name

    coherent_kind = "composed" if composed_sequences else "fingerprint"
    print(
        f"Training: model=temporal(gnn/gru) | the snapshot pass {len(cfg_a.curriculum.stages)} stages "
        f"(randomized) -> the sequence pass {len(cfg_b.curriculum.stages)} stages ({coherent_kind}, "
        f"placement {_sensor_nodes(args)}) | weights continue across the switch | "
        f"accelerator {args.accelerator}"
        + (
            f" -> the profiled-sequence pass {len(cfg_c.curriculum.stages)} stages "
            "(composed longer-window, time features)"
            if cfg_c is not None
            else ""
        )
        + (
            f" | +{args.anneal_epochs}-epoch cosine anneal finetune"
            if args.anneal_epochs > 0
            else ""
        )
        + (
            f" | physics input priors: {','.join(_physics_inputs(args))}"
            if _physics_inputs(args)
            else ""
        )
        + (f" | resumed from {init_ckpt_name}" if init_ckpt_name else ""),
        flush=True,
    )

    grad_collector = None
    if args.collect_gradients:
        from injection_hse.analysis import GradientNormCollector

        grad_collector = GradientNormCollector()

    module, history, ckpts = run_curriculum(
        grid,
        cfg_a,
        cfg_b,
        orders,
        out_dir,
        cfg_c=cfg_c,
        restore_best=not args.no_restore_best,
        device=device,
        extra_callbacks=[grad_collector] if grad_collector is not None else None,
        allow_tight_memory=args.allow_tight_memory,
        init_from=args.init_from,
        start_stage=args.start_stage,
        grid_path=(out_dir / "grid.json").resolve(),
    )

    if rank != 0:
        # training is finished on this rank (the final save_checkpoint barrier synced all
        # ranks); the artifacts + evaluation below are rank-0 work.
        print(f"[rank {rank}] training complete.", flush=True)
        return

    # -- history + manifest artifacts --
    hist_path = history.write_parquet(out_dir / "training_history.parquet")
    if grad_collector is not None:
        gn_path = grad_collector.write_parquet(out_dir / "gradient_norms.parquet")
        print(f"  gradient norms     -> {gn_path.resolve()}", flush=True)
    final_ckpt = ckpts[-1]
    manifest_path = write_manifest(
        out_dir,
        grid,
        cfg_a,
        cfg_b,
        orders,
        random_path,
        coherent_path,
        final_ckpt,
        cfg_c=cfg_c,
        profile_path=profile_path,
        init_checkpoint=init_ckpt_name,
        scenario_recipe=scenario_recipe_identity(args),
        data_generation=data_generation_record(args, plan),
        perturbed_test_path=perturbed_test_path,
        eval_dataset=args.eval_dataset if plan is not None else None,
    )

    ckpt_names = [Path(p).name for p in ckpts]
    profile_line = (
        f"  dataset (profile)  -> {Path(profile_path).resolve()}\n"
        if profile_path is not None
        else ""
    )
    print(
        "\nTRAINING DONE.\n"
        f"  dataset (random)   -> {random_path.resolve()}\n"
        f"  dataset (coherent) -> {coherent_path.resolve()}\n"
        + profile_line
        + f"  checkpoints        -> {(out_dir / 'checkpoints').resolve()}  ({ckpt_names})\n"
        f"  final checkpoint   -> {Path(final_ckpt).resolve()}\n"
        f"  history            -> {hist_path.resolve()}  ({len(history.records)} epochs)\n"
        f"  manifest           -> {manifest_path.resolve()}\n"
        f"  grid               -> {(out_dir / 'grid.json').resolve()}\n"
        f"  configs            -> experiment_snapshots.yaml + experiment_sequences.yaml\n"
        f"  run config         -> training_run.yaml (python -m injection_hse.train --config)\n"
        f"  mlflow             -> {args.mlflow_uri}",
        flush=True,
    )

    # DEPLOY-ONLY anchoring: train with the schedule (possibly zero throughout) and hand the
    # anchored solve to evaluation and inference instead. The checkpoint records the weights a
    # reload deploys (SEModule.from_checkpoint), so rewriting them there is what makes
    # evaluate_run and any later inference use this decode — and keeps the checkpoint honest
    # about what it deploys rather than carrying a training weight it was never meant to run at.
    if args.eval_anchor and ckpts:
        import torch as _torch

        wv, _, wi = args.eval_anchor.partition("/")
        wv, wi = float(wv), (float(wi) if wi else float(wv))  # 'wv' alone = equal trust
        module.set_stage_anchor(v_weight=wv, i_weight=wi)
        final = Path(ckpts[-1])
        state = _torch.load(final, map_location="cpu", weights_only=False)
        state["anchor_weights"] = [wv, wi]
        _torch.save(state, final)
        print(
            f"  deploy-only anchoring: eval + inference run at v={wv} i={wi} "
            f"(training used the --anchor-schedule weights); recorded in {final.name}",
            flush=True,
        )

    # evaluation runs by default (the plots are the point); --skip-eval opts out
    if not args.skip_eval:
        from injection_hse.evaluation import evaluate_run

        print(f"\nEvaluating ({args.eval_dataset} split) ...", flush=True)
        eval_dir = evaluate_run(
            out_dir,
            grid=grid,
            dataset=args.eval_dataset,
            device="cpu",
            heatmap_phases=heatmap_phases,
        )
        if plan is not None:
            evaluate_perturbation(out_dir, grid, eval_dir)
        artifacts = sorted(p.name for p in eval_dir.iterdir())
        print(f"  evaluation -> {eval_dir.resolve()}\n  artifacts: {artifacts}")
        report_split_metrics(eval_dir)


if __name__ == "__main__":
    main()
