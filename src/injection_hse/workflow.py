"""Executable training runs: the multi-pass contract and the runner that executes it.

An :class:`~injection_hse.config.ExperimentConfig` describes ONE curriculum pass over ONE dataset. A
real state-estimation run is several such passes over the SAME module — the scarcity
curriculum shifts the data distribution as it removes sensors:

- **Snapshots** — randomized operating points (the harmonic-flow surrogate, many nodes
  observed),
- **Sequences** — coherent device-population sequences down to the fixed sensor placement,
- **Profiled sequences** (optional) — the longer, time-stamped window, with the periodic
  time features.

:class:`TrainingRunConfig` is that run: an ordered list of :class:`TrainingPass` entries plus
the resume pointers. It is serializable pydantic like every other config in the suite, so a run
directory's ``training_run.yaml`` is not merely a record of what happened — it is the
RE-ENTRY POINT: ``python -m injection_hse.train --config <run.yaml>`` executes it again.

:func:`run_training` builds ONE module from the first pass's config, fits the normalizers
once, and runs one :class:`injection_hse.train.StageManager` per pass in order, so the weights and the
feature scale continue across the task switch (refitting between passes would rescale the
loss under the weights it is continuing). The linear injection prior is the exception: a
closed-form map of the measurements, it is refit on each pass's own train split (see
:func:`run_training`). A pass whose stage list is empty is skipped — that is how a resumed
run replays only the stages it still owes.

Training only: data GENERATION, evaluation and figures belong to the caller. The module
imports nothing from the example drivers, and the heavy training stack (lightning,
torch-geometric, mlflow) is imported lazily, so the config contract can be read and
validated without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ExperimentConfig
from .curriculum import apply_init_checkpoint

#: Leading passes whose train split feeds the shared normalizer fit (single-grid runs). The
#: randomized snapshots and the coherent sequences are two different DATA-GENERATING
#: PROCESSES covering different envelopes, so statistics from one leave the other's features
#: off-scale; a further pass (the profiled longer window) draws from the second process, and
#: its envelope is already represented.
_NORMALIZER_FIT_PASSES = 2

#: A clean single-step probe already using this fraction of the card is NOT a safe green
#: light. The real training peak sits ABOVE the probe by the optimizer state, the batched
#: ``torch.linalg.solve`` workspace, and — the usual late-run out-of-memory cause —
#: caching-allocator fragmentation that accrues over epochs. Leave ~10 % for those; below
#: that, a run that probes fine still fails hours in.
PREFLIGHT_SAFE_FRACTION = 0.9


# =============================================================================
# the run contract
# =============================================================================
class TrainingPass(BaseModel):
    """One curriculum pass: a name and the :class:`~injection_hse.config.ExperimentConfig` it runs.

    The name labels the pass in the run banner and in the tracked run name
    (``<experiment name>-<pass name>``); it carries no behaviour.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment: ExperimentConfig


class TrainingRunConfig(BaseModel):
    """A complete, re-runnable training run: the ordered passes over ONE module.

    Every pass trains the SAME module, so the fields the module DERIVES at construction —
    the seed, the harmonic orders, the model, the encoding, the dtypes, the loss weighting,
    the noise model, the decode — must agree across passes; the seed and the orders are
    checked here (the full set is checked against the built module in :func:`run_training`).
    What legitimately differs is the dataset a pass opens, its stage ladder, its batch size
    and its optimizer.

    ``init_from`` names a ``stage_<name>.ckpt`` whose weights and fitted normalizers the run
    starts from. The passes of a RESUMED run carry only the stages still to train
    (:func:`injection_hse.curriculum.resume_stage_split` trims the ladders), and ``start_stage``
    records where that trim starts — validated against the passes, so the resume pointers
    and the ladders cannot disagree. Both travel with the config, so a resumed run is
    reproducible from its own record.

    ``grid_path`` points at the serialized :class:`pgml.schemas.grid_schema.Grid` the
    datasets were generated on (a run directory's ``grid.json``): a parquet dataset stores
    only the node-phase row layout, not the topology the graph and the measurement model
    need. A multi-grid run leaves it unset — each member grid is persisted beside its own
    scenarios.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "training_run"
    passes: list[TrainingPass] = Field(min_length=1)
    grid_path: Optional[str] = None
    init_from: Optional[str] = None
    start_stage: Optional[str] = None

    @model_validator(mode="after")
    def _one_module(self) -> "TrainingRunConfig":
        names = [p.name for p in self.passes]
        if len(set(names)) != len(names):
            raise ValueError(f"training pass names must be unique, got {names}.")
        seeds = {p.experiment.seed for p in self.passes}
        if len(seeds) > 1:
            raise ValueError(
                f"the passes carry different seeds {sorted(seeds)}; one module is seeded "
                "once, so every pass of a run must declare the same seed."
            )
        orders = {tuple(p.experiment.orders) for p in self.passes}
        if len(orders) > 1:
            raise ValueError(
                f"the passes carry different harmonic orders {sorted(orders)}; the orders "
                "are the model's harmonic axis, fixed at construction for the whole run."
            )
        return self

    @model_validator(mode="after")
    def _resume_pointer(self) -> "TrainingRunConfig":
        if self.start_stage is None:
            return self
        names = [s.name for p in self.passes for s in p.experiment.curriculum.stages]
        if names[:1] != [self.start_stage]:
            raise ValueError(
                f"start_stage={self.start_stage!r} is not the first stage the passes "
                f"still train ({names[:1] or 'none'}); the ladders of a resumed run carry "
                "only the stages left to train."
            )
        return self

    @property
    def seed(self) -> int:
        """The run seed — one module, one seed (validated identical across the passes)."""
        return int(self.passes[0].experiment.seed)

    @property
    def orders(self) -> list[int]:
        """The harmonic orders the run trains on (identical across the passes)."""
        return list(self.passes[0].experiment.orders)

    @classmethod
    def from_experiment(
        cls, experiment: ExperimentConfig, *, pass_name: str = "train", **kwargs
    ) -> "TrainingRunConfig":
        """Wrap a single :class:`~injection_hse.config.ExperimentConfig` as a one-pass run."""
        return cls(
            name=experiment.name,
            passes=[TrainingPass(name=pass_name, experiment=experiment)],
            **kwargs,
        )

    @classmethod
    def load(cls, path) -> "TrainingRunConfig":
        """Read a run config from YAML/JSON; a bare experiment becomes a one-pass run.

        Accepting both spellings is what makes every config a run directory already holds
        executable: the dumped :class:`~injection_hse.config.ExperimentConfig` of a single pass runs
        as-is, and the dumped :class:`TrainingRunConfig` replays the whole curriculum.
        """
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a config mapping.")
        if "passes" in raw:
            return cls.model_validate(raw)
        return cls.from_experiment(ExperimentConfig.model_validate(raw))

    def write(self, path) -> Path:
        """Dump this run config as YAML (the run directory's re-entry point)."""
        import yaml

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        return out

    def with_output_dir(self, out_dir) -> "TrainingRunConfig":
        """A copy whose passes write their checkpoints and tracking store under ``out_dir``.

        Redirects every pass's ``train.checkpoint_dir`` to ``<out_dir>/checkpoints`` and its
        ``train.mlflow_tracking_uri`` to the run directory's own file store — the whole
        output relocation a re-run of a recorded config needs.
        """
        root = Path(out_dir)
        tracking = f"file:{(root / 'mlruns').resolve().as_posix()}"
        passes = [
            p.model_copy(
                update={
                    "experiment": p.experiment.model_copy(
                        update={
                            "train": p.experiment.train.model_copy(
                                update={
                                    "checkpoint_dir": str(root / "checkpoints"),
                                    "mlflow_tracking_uri": tracking,
                                }
                            )
                        }
                    )
                }
            )
            for p in self.passes
        ]
        return self.model_copy(update={"passes": passes})

    def identity(self) -> str:
        """A multi-line summary of WHAT this run trains — the banner of an executed config."""
        lines = [
            f"training run {self.name!r}: {len(self.passes)} pass(es), seed {self.seed}, "
            f"orders {self.orders}",
        ]
        for p in self.passes:
            exp = p.experiment
            stages = [s.name for s in exp.curriculum.stages]
            target = (
                "multi-grid corpus" if exp.data.multigrid is not None else "dataset"
            )
            lines.append(
                f"  {p.name}: {exp.name} | {target} {exp.data.dataset_path} | "
                f"model {exp.model.kind} | decode {exp.physics.decode} | "
                f"stages {stages}"
            )
        if self.grid_path:
            lines.append(f"  grid: {self.grid_path}")
        if self.init_from:
            lines.append(
                f"  resume: weights from {self.init_from}"
                + (
                    f", starting at stage {self.start_stage!r}"
                    if self.start_stage
                    else ""
                )
            )
        return "\n".join(lines)


# =============================================================================
# launcher plumbing
# =============================================================================
def launch_rank() -> int:
    """This process's global rank under an external distributed launcher, else 0.

    Multi-GPU training launches ONE process per GPU (``srun`` on SLURM, ``torchrun``
    locally); everything outside ``Trainer.fit`` — the memory preflight, the tracking-store
    creation, the artifact writes — must run on exactly one of them. The rank is read from
    the launcher's environment (``SLURM_PROCID`` / ``RANK`` / ``LOCAL_RANK``) because it is
    needed BEFORE Lightning initializes the process group.
    """
    import os

    for var in ("SLURM_PROCID", "RANK", "LOCAL_RANK"):
        if var in os.environ:
            return int(os.environ[var])
    return 0


# =============================================================================
# the built run
# =============================================================================
@dataclass(frozen=True)
class TrainingSetup:
    """Everything a training run needs, built once: ONE module and one datamodule per pass.

    ``sources`` are the passes' :class:`~injection_hse.data.DataSource` s (index-aligned with
    ``datamodules`` and with the config's passes) — the caller's handle for evaluation after
    training. ``stage_kwargs`` are the placement arguments every
    :class:`injection_hse.train.StageManager` of the run is constructed with (the sensor rows and the
    heterogeneous-fleet extras; empty on the multi-grid path, where the placement travels
    per member with each batch). ``prior_pass`` is the index of the pass whose train split
    the installed linear injection prior was fitted on (``None`` when no prior is
    installed); :func:`run_training` refits it for every other pass that needs it.
    """

    module: Any
    sources: list
    datamodules: list
    stage_kwargs: dict = field(default_factory=dict)
    prior_pass: Optional[int] = None


@dataclass(frozen=True)
class TrainingResult:
    """The outcome of :func:`run_training`: the trained module, its history, its checkpoints.

    ``history`` is the shared :class:`injection_hse.evaluation.HistoryCollector` spanning every stage
    of every pass (write it with ``history.write_parquet(...)``); ``checkpoints`` are the
    stage-boundary checkpoint paths in training order, so ``checkpoints[-1]`` is the final
    model. ``setup`` exposes the built module's data sources for the caller's evaluation.
    """

    module: Any
    history: Any
    checkpoints: list
    setup: TrainingSetup


# =============================================================================
# memory preflight
# =============================================================================
def preflight_verdict(
    peak, total, *, batch_size, allow_tight, safe_fraction=PREFLIGHT_SAFE_FRACTION
) -> str:
    """Turn a raw probe peak into a headroom verdict — pure, so it needs no device.

    :func:`injection_hse.train.probe_step_memory` proves ONE forward+backward FITS, not that the run
    SURVIVES: real training adds optimizer state, the batched-solve workspace, and allocator
    fragmentation on top of the probe peak. A probe already using ``>= safe_fraction`` of
    ``total`` is a red flag, not a pass. Returns a short utilization summary; raises
    :class:`SystemExit` (with the concrete remedies) when the fraction is unsafe and
    ``allow_tight`` is not set.
    """
    frac = peak / total
    summary = (
        f"peak {peak / 1e9:.2f} GB / {total / 1e9:.2f} GB ({frac * 100:.0f}%) "
        f"at batch {batch_size}"
    )
    if frac >= safe_fraction and not allow_tight:
        raise SystemExit(
            f"{summary} — too little headroom. A clean single-step probe fits, but real "
            "training adds optimizer state, the batched-solve workspace, and allocator "
            "fragmentation on top, so it will likely OOM late (as it just did). Fix one of:\n"
            "  - lower --batch-size;\n"
            f"  - keep the optimization batch via gradient accumulation, e.g. --batch-size "
            f"{max(1, batch_size // 4)} --effective-batch {batch_size};\n"
            "  - export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (cuts fragmentation);\n"
            "  - pass --allow-tight-memory to run anyway."
        )
    return summary


def check_preflight_headroom(peak, device, *, batch_size, allow_tight=False) -> str:
    """Device wrapper for :func:`preflight_verdict`: query the card's total, then verdict.

    Returns a summary string (a CPU note when ``peak is None`` — no peak tracking there).
    """
    if peak is None:
        return f"batch {batch_size}, CPU (no peak tracking)"
    import torch

    total = int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    return preflight_verdict(
        peak, total, batch_size=batch_size, allow_tight=allow_tight
    )


def preflight_training(
    cfg: TrainingRunConfig,
    *,
    grid=None,
    device,
    allow_tight_memory: bool = False,
    pass_index: int = -1,
):
    """Build this run's module and probe ONE training step of one pass.

    Returns the step's peak CUDA memory in bytes (``None`` on CPU, where no peak tracking
    exists). Any error the first real training step would hit — an oversized-batch
    out-of-memory above all — is raised HERE, so a sweep can probe every planned
    configuration up front and a misconfigured batch size surfaces before hours of training.
    A probe that FITS but leaves too little headroom (see :func:`preflight_verdict`) also
    raises, unless ``allow_tight_memory``.

    ``pass_index`` selects the pass to probe; the default (the LAST one) is the
    memory-heaviest — the coherent sequences carry a time axis the snapshots do not, and
    they train last.
    """
    from .train import probe_step_memory

    setup = build_training_setup(cfg, grid=grid)
    probed = cfg.passes[pass_index].experiment
    try:
        peak = probe_step_memory(
            setup.module, setup.datamodules[pass_index], device=device
        )
        check_preflight_headroom(
            peak,
            device,
            batch_size=probed.data.batch_size,
            allow_tight=allow_tight_memory,
        )
        return peak
    finally:
        del setup


# =============================================================================
# building the run
# =============================================================================
def _fit_sample(source, *, max_rows: int = 4096):
    """Up to ``max_rows`` train states of ``source`` as ``[Bt, H, N]`` (``None`` if empty).

    A SEQUENCE dataset's step axis is flattened into the batch, and its scenario budget
    scaled by the step count, so every task contributes a comparable number of ROWS to the
    normalizer fit whether it is snapshots or sequences.
    """
    import torch

    ds = source.train_dataset()
    if not len(ds):
        return None
    sample = ds[0]["v"]
    steps = int(sample.shape[0]) if sample.ndim == 3 else 1
    n = min(len(ds), max(1, max_rows // max(1, steps)))
    v = torch.stack([ds[i]["v"] for i in range(n)])
    return v.reshape(-1, *v.shape[-2:])


def _restore_normalizer(state: Optional[dict], key: str):
    """A fitted :class:`~injection_hse.normalization.HarmonicNormalizer` from a checkpoint payload."""
    if state is None or key not in state:
        return None
    from .normalization import HarmonicNormalizer

    restored = HarmonicNormalizer()
    restored.load_state_dict(state[key])
    return restored


def _placement_experiment(cfg: TrainingRunConfig) -> ExperimentConfig:
    """The pass whose measurement fleet defines the placement rows (the placement stage's)."""
    for p in cfg.passes:
        if any(s.kind == "placement" for s in p.experiment.curriculum.stages):
            return p.experiment
    return cfg.passes[0].experiment


def _build_single_grid(cfg: TrainingRunConfig, grid) -> TrainingSetup:
    """Build the shared module + one datamodule per pass for a SINGLE-grid run.

    The normalizers are fit on the UNION of the leading passes' train splits (see
    ``_NORMALIZER_FIT_PASSES``) and the physics operators — the branch-current map, the
    injection solver, the per-configuration topology bank — are built ONCE, sized for the
    LARGEST anchor weight any stage of any pass requests; each stage then dials its runtime
    scalar through the :class:`injection_hse.train.StageManager`.
    """
    import torch

    from .data import base_voltage_per_row, build_data_source, per_unit_state
    from .encoding import build_encoding, build_injection_encoding
    from .normalization import HarmonicNormalizer
    from .train import SEModule, anchor_build_weights, make_datamodule, training_dtypes

    first = cfg.passes[0].experiment
    orders = cfg.orders
    sources = [
        build_data_source(p.experiment.data, grid=grid, orders=orders)
        for p in cfg.passes
    ]

    # statistics from ONE task leave the other's features off-scale whenever the two
    # data-generating processes cover different envelopes (per-harmonic magnitudes
    # especially), and every stage after the task switch then trains out of distribution.
    encoding = build_encoding(first.encoding.kind)
    fit_v = [
        v
        for v in (_fit_sample(s) for s in sources[:_NORMALIZER_FIT_PASSES])
        if v is not None
    ]
    train_v = torch.cat(fit_v, dim=0)
    # THE FEATURE FRAME (``normalization.per_unit``): every normalizer is fit in the frame
    # the module encodes in. The per-row voltage base is known here; the per-unit CURRENT
    # base needs the solver's operator scale and is taken once that is built.
    base_v = base_voltage_per_row(grid)
    per_unit = bool(first.normalization.per_unit)
    train_feats = encoding.encode(
        per_unit_state(train_v, base_v) if per_unit else train_v
    ).transpose(-3, -2)  # [Bt, N, H, C]
    normalizer = HarmonicNormalizer().fit(
        train_feats, magnitude_transform=first.normalization.magnitude_transform
    )

    _, complex_dtype = training_dtypes(first.train.dtype)
    curricula = [p.experiment.curriculum for p in cfg.passes]
    build_v, build_i = anchor_build_weights(
        first.physics, curricula, measure_current=first.measurement.measure_current
    )

    # SWITCHABLE TOPOLOGY: one network solver + branch-current operator per switch
    # configuration, all on the union layout the graph is built from. The bank supplies the
    # module's representative operators too, so the single- and multi-topology paths build
    # exactly the same way.
    topology = None
    if first.data.topology is not None:
        from .data import TopologyBank, TopologySet

        topology = TopologyBank.build(
            TopologySet.from_config(first.data.topology),
            grid,
            list(orders),
            edge_branch_ids=sources[0].graph().edge_branch_ids,
            dtype=complex_dtype,
            measure_current=first.measurement.measure_current,
            decode=first.physics.decode,
            build_solvers=first.needs_solver,
            anchor_v_weight=build_v,
            anchor_i_weight=build_i,
            harmonic_slack=first.physics.harmonic_slack,
            load_shunt=first.physics.load_shunt,
        )
        print(
            f"  topologies: {TopologySet.from_config(first.data.topology).describe(grid)}",
            flush=True,
        )

    # measured branch currents (optional): the edge operator. Built BEFORE the solver so a
    # current-anchored injection decode can materialise its V->branch-current map from it.
    edge_op = None
    edge_normalizer = None
    power_normalizer = None
    if first.measurement.measure_current:
        from .data import BranchCurrentOperator

        edge_op = (
            topology.edge_ops[0]
            if topology is not None
            else BranchCurrentOperator(grid, list(orders), dtype=complex_dtype)
        )

    # the differentiable network solver + the injection normalizer: the physics-informed
    # decode and the physics-derived input priors (``physics.input_priors``, either decode)
    # both need them, and so does the per-unit CURRENT base of metered branch currents
    # (:attr:`~injection_hse.config.ExperimentConfig.needs_solver`). Measurement anchoring (the
    # PhysicsConfig weights or a per-stage schedule) makes the observed voltages / branch
    # currents soft constraints.
    solver = None
    inj_normalizer = None
    if first.needs_solver:
        from .physics import NetworkSolver

        # the solver is built at the TRAINING complex dtype (generation precision is
        # independent), so the injection normalizer is fit on decode-space targets at the
        # same precision; the internal linear solve stays complex128 regardless (the feeder
        # Y-bus is ill-conditioned).
        solver = (
            topology.solvers[0]
            if topology is not None
            else NetworkSolver.from_grid(
                grid,
                list(orders),
                dtype=complex_dtype,
                anchor_v_weight=build_v,
                anchor_i_weight=build_i,
                edge_op=edge_op,
                harmonic_slack=first.physics.harmonic_slack,
                load_shunt=first.physics.load_shunt,
            )
        )
    # the per-unit CURRENT base of the injection / branch-current features
    i_base = solver.i_base(base_v) if (per_unit and solver is not None) else None
    if edge_op is not None:
        # the edge (and power) normalizers on the SAME union fit sample as the state
        # normalizer, in the frame the module's edge channels are built in
        i_train = edge_op(train_v.to(complex_dtype))
        e_base = (
            None if i_base is None else i_base.index_select(-1, edge_op.edge_src_rows)
        )
        i_feat = i_train if e_base is None else i_train / e_base.to(i_train.real.dtype)
        edge_normalizer = HarmonicNormalizer().fit(
            encoding.encode(i_feat).transpose(-3, -2),
            magnitude_transform=first.normalization.magnitude_transform,
        )
        if first.measurement.measure_power:
            v_train = per_unit_state(train_v, base_v) if per_unit else train_v
            v_src = v_train.to(complex_dtype).index_select(-1, edge_op.edge_src_rows)
            power_normalizer = HarmonicNormalizer().fit(
                encoding.encode(v_src * i_feat.conj()).transpose(-3, -2),
                magnitude_transform=first.normalization.magnitude_transform,
            )
    if solver is not None:
        from .physics import fit_injection_normalizer

        inj_normalizer = fit_injection_normalizer(
            solver,
            train_v,
            build_injection_encoding(first.encoding),
            magnitude_transform=first.normalization.magnitude_transform,
            base=i_base,
            injection_spec=sources[0].injection_spec,
            exclude_tied=first.physics.exclude_tied_rows,
        )
        if first.physics.decode == "injection":
            _print_physics_banner(first, curricula, build_v, build_i)
        if first.physics.input_priors:
            print(
                f"  physics input priors: {', '.join(first.physics.input_priors)} "
                f"(prior anchors v={first.physics.prior_anchor_voltage_weight:g} "
                f"i={first.physics.prior_anchor_current_weight:g})",
                flush=True,
            )

    module = SEModule(
        first,
        n_nodes=sources[0].n_nodes,
        normalizer=normalizer,
        graph=sources[0].graph(),  # the passes share the grid -> the same graph
        base_v=base_v,
        injection_spec=None,
        solver=solver,
        inj_normalizer=inj_normalizer,
        grid=grid,  # checkpoints embed the row-layout fingerprint
        edge_op=edge_op,
        edge_normalizer=edge_normalizer,
        power_normalizer=power_normalizer,
        topology=topology,
    )

    # the placement rows + the heterogeneous placement extras (per-row order capability of
    # the sensor fleet; explicit metered edges stay None for a config-driven placement —
    # every observed location meters all incident branches)
    from .data import MeasurementModel

    placement = _placement_experiment(cfg)
    mm = MeasurementModel.resolve(
        grid,
        placement.measurement.voltage_nodes,
        fundamental_nodes=placement.measurement.fundamental_nodes,
        slack_known=placement.measurement.slack_known,
    )
    stage_kwargs = {
        "observed_rows": mm.observed_rows(grid),
        "order_capability": (
            mm.order_capability(grid, list(orders))
            if placement.measurement.heterogeneous_orders
            else None
        ),
        "observed_edge_rows": (
            mm.observed_edges(grid) if placement.measurement.measure_current else None
        ),
    }
    # the LINEAR INJECTION PRIOR (the distillation teacher / the prior-residual base) is
    # fitted on the train split of the pass that trains with it: here for the first such
    # pass (so the built module is complete, e.g. for the memory preflight), and refit by
    # ``run_training`` before every later pass that needs it. Not fitted when no pass asks.
    from .train import (
        attach_loss_preconditioner,
        preconditioner_requested,
    )

    prior_pass = _first_prior_pass(cfg)
    if prior_pass is not None:
        _attach_pass_prior(
            module,
            cfg.passes[prior_pass].experiment,
            sources[prior_pass],
            stage_kwargs["observed_rows"],
        )
    # the LOSS PRECONDITIONER (``loss.preconditioner``): the curvature of the deployed
    # decode, read once from the built module and installed on the head's gradient. Like
    # the prior, ONE per run — the first pass that asks for it defines it.
    needs_pre = next(
        (p.experiment for p in cfg.passes if preconditioner_requested(p.experiment)),
        None,
    )
    if needs_pre is not None:
        attach_loss_preconditioner(module, needs_pre, stage_kwargs["observed_rows"])

    datamodules = [
        make_datamodule(
            source,
            batch_size=p.experiment.data.batch_size,
            num_workers=p.experiment.data.num_workers,
        )
        for source, p in zip(sources, cfg.passes)
    ]
    return TrainingSetup(
        module=module,
        sources=sources,
        datamodules=datamodules,
        stage_kwargs=stage_kwargs,
        prior_pass=prior_pass,
    )


def _first_prior_pass(cfg: TrainingRunConfig) -> Optional[int]:
    """Index of the first pass that trains with the linear injection prior (``None``: none).

    A pass with an empty stage list (skipped by a resumed run) does not train, so the first
    pass that needs the prior AND trains is preferred; when every such pass is empty, the
    first that needs it is returned so the built module still carries a prior.
    """
    from .train import prior_requested

    needing = [k for k, p in enumerate(cfg.passes) if prior_requested(p.experiment)]
    training = [k for k in needing if cfg.passes[k].experiment.curriculum.stages]
    if training:
        return training[0]
    return needing[0] if needing else None


def _attach_pass_prior(module, experiment: ExperimentConfig, source, observed_rows):
    """Fit the linear injection prior on ``source``'s TRAIN states and install it.

    The first ``experiment.physics.injection_prior_fit_states`` train states of the pass's
    own dataset (:func:`injection_hse.train.prior_fit_states`), at the deployed placement
    ``observed_rows``, through :func:`injection_hse.train.attach_injection_prior`.
    """
    from .train import attach_injection_prior, prior_fit_states

    states = prior_fit_states(source, experiment.physics.injection_prior_fit_states)
    if states is None:
        raise ValueError(
            f"the linear injection prior of pass {experiment.name!r} is fitted on its "
            "TRAIN split, which is empty."
        )
    attach_injection_prior(module, experiment, states, observed_rows)


def _print_physics_banner(first, curricula, build_v, build_i) -> None:
    """Print the anchoring + state-supervision regime of an injection decode.

    A slack-only decode and a measurement-anchored one look identical in the config
    otherwise (the weights that matter are the per-stage ones the stage manager installs).
    """
    if build_v > 0.0 or build_i > 0.0:
        sched = sorted(
            {
                s.name: (s.anchor_voltage_weight, s.anchor_current_weight)
                for cur in curricula
                for s in cur.stages
                if s.anchor_voltage_weight is not None
                or s.anchor_current_weight is not None
            }.items()
        )
        detail = (
            f"per-stage schedule {dict(sched)}"
            if sched
            else f"fixed v={first.physics.anchor_voltage_weight} "
            f"i={first.physics.anchor_current_weight}"
        )
        randomized = any(
            s.anchor_voltage_range is not None or s.anchor_current_range is not None
            for cur in curricula
            for s in cur.stages
        )
        regime = "per-batch randomized" if randomized else "fixed weights"
        weighting = (
            "manual weights"
            if first.physics.anchor_weighting == "manual"
            else "noise-calibrated weights, prior rel sigma "
            f"{_format_rel_sigma(first.physics.prior_injection_rel_sigma)}; "
            "the weights are multipliers"
        )
        print(
            f"  measurement anchoring: ON ({detail}; {regime}; {weighting})", flush=True
        )
    else:
        print("  measurement anchoring: OFF (slack-only injection decode)", flush=True)
    split = first.loss.injection_state_loss_split
    scaling = (
        "scale-normalized"
        if first.loss.injection_state_loss_normalize
        else "RAW (the split is not a true fraction)"
    )
    plain = first.loss.plain_decode_weight
    print(
        "  state supervision: "
        + (
            f"{1 - split:.2f} solved voltages / {split:.2f} predicted injections "
            f"[{scaling}]"
            if split > 0.0
            else "solved voltages only"
        )
        + (
            f" + {plain:g} × the voltage loss on the PLAIN (slack-only) decode while "
            "anchors are on"
            + (
                f" (observed rows weighted {first.loss.plain_decode_observed_weight:g}, "
                "unobserved 1)"
                if first.loss.plain_decode_observed_weight is not None
                else ""
            )
            if plain > 0.0
            else ""
        )
        + (
            f"; injection head encoded as {first.encoding.injection_kind!r} "
            f"(voltage input {first.encoding.kind!r})"
            if first.encoding.injection_kind is not None
            else ""
        ),
        flush=True,
    )


def _format_rel_sigma(rho) -> str:
    """``ρ`` for the run log: one value, or one per order."""
    if isinstance(rho, list):
        return "per order [" + ", ".join(f"{float(r):g}" for r in rho) + "]"
    return f"{float(rho):g}"


def _build_multigrid(cfg: TrainingRunConfig) -> TrainingSetup:
    """Build the shared module + one datamodule per pass for a MULTI-GRID run.

    The passes open the same corpus through different per-member datasets (the task), so the
    member set must agree — and agree in ORDER: the bank is built from the first pass's
    source and a later pass's batch ``grid_index`` is resolved through ``bank.source``, so
    diverging member lists would silently cross the indices (checked below, raising). The
    bank depends on the grids only, never on the task dataset, which is what makes sharing
    it valid. The normalizers and the per-member operator bank are fit / built
    ONCE on the FIRST pass's source and carried. Two reasons: the feature scale must not move
    under the weights being continued, and the first pass is by construction the wider
    envelope — its randomized snapshots sweep the load-scale and emission ranges the coherent
    sequences move inside.

    Resuming (``cfg.init_from``) takes the normalizers from the checkpoint instead of
    refitting them: a refit sweeps every train member, and it only reproduces the
    checkpoint's scale as long as the corpus and config are untouched — reading them off the
    checkpoint keeps the continued run exact by construction.
    """
    from .data import build_data_source
    from .encoding import build_encoding, build_injection_encoding
    from .train import (
        SEModule,
        anchor_build_weights,
        fit_multigrid_normalizer,
        make_datamodule,
        training_dtypes,
    )

    first = cfg.passes[0].experiment
    orders = cfg.orders
    sources = [build_data_source(p.experiment.data, orders=orders) for p in cfg.passes]
    keys = [[bundle.key for bundle in source.grids] for source in sources]
    if any(k != keys[0] for k in keys[1:]):
        raise SystemExit(
            "the task sources resolve different member grids "
            f"({[len(k) for k in keys]}); every pass must read one corpus with one "
            "manifest and the same max_grids."
        )

    encoding = build_encoding(first.encoding.kind)
    state = None
    if cfg.init_from is not None:
        import torch

        state = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    normalizer = _restore_normalizer(state, "normalizer")
    if state is not None and normalizer is None:
        raise SystemExit(
            f"{cfg.init_from} carries no fitted normalizer to continue from."
        )
    if normalizer is None:
        normalizer = fit_multigrid_normalizer(sources[0], encoding, first)

    bank = None
    inj_normalizer = None
    edge_normalizer = None
    if first.physics.needs_solver or first.measurement.measure_current:
        from .physics import (
            MultiGridPhysicsBank,
            fit_multigrid_edge_normalizer,
            fit_multigrid_injection_normalizer,
        )

        _, complex_dtype = training_dtypes(first.train.dtype)
        build_v, build_i = anchor_build_weights(
            first.physics,
            [p.experiment.curriculum for p in cfg.passes],
            measure_current=first.measurement.measure_current,
        )
        bank = MultiGridPhysicsBank(
            sources[0],
            list(orders),
            dtype=complex_dtype,
            anchor_v_weight=build_v,
            anchor_i_weight=build_i,
            harmonic_slack=first.physics.harmonic_slack,
            load_shunt=first.physics.load_shunt,
        )
        if first.physics.needs_solver:
            inj_normalizer = _restore_normalizer(state, "inj_normalizer")
            if inj_normalizer is None:
                inj_normalizer = fit_multigrid_injection_normalizer(
                    bank, sources[0], build_injection_encoding(first.encoding), first
                )
        if first.measurement.measure_current:
            edge_normalizer = _restore_normalizer(state, "edge_normalizer")
            if edge_normalizer is None:
                edge_normalizer = fit_multigrid_edge_normalizer(
                    bank, sources[0], encoding, first
                )
        print(
            f"  physics: decode={first.physics.decode}, anchoring built for "
            f"v={build_v} i={build_i}, "
            f"state supervision {1 - first.loss.injection_state_loss_split:.2f} solved "
            f"voltages / {first.loss.injection_state_loss_split:.2f} predicted injections"
        )

    module = SEModule(
        first,
        normalizer=normalizer,
        grid_keys=keys[0],
        bank=bank,
        inj_normalizer=inj_normalizer,
        edge_normalizer=edge_normalizer,
    )
    datamodules = [
        make_datamodule(
            source,
            batch_size=p.experiment.data.batch_size,
            num_workers=p.experiment.data.num_workers,
        )
        for source, p in zip(sources, cfg.passes)
    ]
    # the placement of a multi-grid run is per MEMBER and travels with each batch, so the
    # stage manager carries the stage KIND only.
    return TrainingSetup(module=module, sources=sources, datamodules=datamodules)


def build_training_setup(cfg: TrainingRunConfig, *, grid=None) -> TrainingSetup:
    """Build ONE module and one datamodule per pass of ``cfg`` (no training yet).

    Seeds the run (``L.seed_everything``), opens each pass's data source, fits the
    normalizers, builds the physics operators the config asks for, and resolves the sensor
    placement. Separated from :func:`run_training` so a memory preflight probes the REAL
    training footprint (:func:`preflight_training`) and so a driver can inspect the sources
    before committing to a run.

    ``grid`` is the network the datasets were generated on — REQUIRED unless the run is
    multi-grid (each member grid is persisted beside its own scenarios) or the config names
    a ``grid_path``, which is loaded here.
    """
    import lightning as L

    L.seed_everything(cfg.seed, workers=True)
    if cfg.passes[0].experiment.data.multigrid is not None:
        return _build_multigrid(cfg)
    if grid is None:
        grid = load_grid(cfg.grid_path) if cfg.grid_path else None
    if grid is None:
        raise ValueError(
            "a single-grid training run needs the grid its datasets were generated on: "
            "pass grid=..., or set TrainingRunConfig.grid_path to a serialized Grid "
            "(the run directory's grid.json)."
        )
    return _build_single_grid(cfg, grid)


def load_grid(path):
    """Load a serialized :class:`pgml.schemas.grid_schema.Grid` (a run's ``grid.json``)."""
    from pgml.schemas.grid_schema import Grid

    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"no such grid file: {file}")
    return Grid.model_validate_json(file.read_text(encoding="utf-8"))


# =============================================================================
# running the run
# =============================================================================
def run_training(
    cfg: TrainingRunConfig,
    *,
    grid=None,
    device=None,
    out_dir=None,
    restore_best: bool = True,
    extra_callbacks: Optional[list] = None,
    allow_tight_memory: bool = False,
    logger=None,
    setup: Optional[TrainingSetup] = None,
) -> TrainingResult:
    """Execute a training run: one module, one stage-manager pass per configured pass.

    The module's weights persist from each pass into the next (the SAME object is reused) and
    a shared :class:`injection_hse.evaluation.HistoryCollector` accumulates the epoch-indexed history
    across all of them. A pass whose stage list is EMPTY is skipped, which is how a resumed
    run replays only the stages it still owes (:func:`injection_hse.curriculum.resume_stage_split`
    trims the ladders). Returns a :class:`TrainingResult`.

    The LINEAR INJECTION PRIOR (the prior-residual base, the distillation teacher) is the
    one fitted artefact that follows the pass instead of the run: every pass that trains
    with it starts from the map fitted on its OWN train split
    (``physics.injection_prior_fit_states`` states), so the sequence pass decodes with the
    map of the sequences rather than of the snapshots. The build fits it for the first such
    pass; a later one is refit before its first stage, and so is the first pass of a run
    started from ``init_from`` (whose checkpoint restored a prior of unknown data). Each
    stage checkpoint carries the prior its pass trained with.

    ``grid`` / ``setup`` — the network the datasets were generated on, or an already-built
    :class:`TrainingSetup` (from :func:`build_training_setup`) when the caller needs the data
    sources before training.

    ``device`` turns on the MEMORY PREFLIGHT: one real forward+backward of the LAST pass (the
    memory-heaviest — sequences carry a time axis snapshots do not, and they train last) is
    probed on rank 0 before the first fit, so an oversized batch fails in seconds instead of
    hours in. ``None`` skips it. ``allow_tight_memory`` accepts a probe that fits but leaves
    little headroom.

    ``restore_best`` (default ON here, unlike a bare :class:`injection_hse.train.StageManager`) hardens
    a long run: each stage checkpoints and continues from its lowest-``val_pu_mae`` weights
    and a stage that goes non-finite is halted, so a late collapse cannot poison the saved
    model. ``extra_callbacks`` are attached to every stage of every pass.

    ``logger`` is the tracking logger: ``None`` (default) builds one MLflow logger per pass,
    on rank 0 only — every rank constructing one races the file store's check-then-create of
    ``mlruns/*/meta.yaml`` on a shared filesystem, and Lightning logs from rank 0 anyway.
    Pass an explicit logger to record every pass as ONE tracked run, or ``False`` to disable
    tracking.

    ``out_dir`` (optional) is the run directory the resolved ``cfg`` is written into as
    ``training_run.yaml`` — the re-entry point ``python -m injection_hse.train --config`` consumes.
    """
    from .evaluation import HistoryCollector
    from .train import StageManager, probe_step_memory, prior_requested

    rank = launch_rank()
    _assert_passes_swappable(cfg)
    if out_dir is not None and rank == 0:
        cfg.write(Path(out_dir) / "training_run.yaml")

    if setup is None:
        setup = build_training_setup(cfg, grid=grid)
    module = setup.module
    if cfg.init_from is not None:
        apply_init_checkpoint(module, cfg.init_from)
        print(f"Initialized weights + normalizers from {cfg.init_from}", flush=True)

    loggers = [_pass_logger(p, rank) if logger is None else logger for p in cfg.passes]

    if device is not None and rank == 0:
        last = cfg.passes[-1].experiment
        peak = probe_step_memory(module, setup.datamodules[-1], device=device)
        summary = check_preflight_headroom(
            peak,
            device,
            batch_size=last.data.batch_size,
            allow_tight=allow_tight_memory,
        )
        print(f"Batch-size preflight OK ({last.name}, {summary}).", flush=True)

    history = HistoryCollector()
    callbacks = [history, *(extra_callbacks or [])]
    checkpoints: list = []
    # the pass the installed linear injection prior was fitted for; an init checkpoint
    # restores the prior it carried, whose data is not known here, so it is refit
    prior_pass = None if cfg.init_from is not None else setup.prior_pass
    # carried optimizer state (``train.carry_optimizer_state``): the init checkpoint's, then
    # each pass's last stage's into the next pass
    optimizer_state = _init_optimizer_state(cfg)
    for index, (training_pass, datamodule, pass_logger) in enumerate(
        zip(cfg.passes, setup.datamodules, loggers)
    ):
        experiment = training_pass.experiment
        if not experiment.curriculum.stages:
            continue
        if prior_requested(experiment) and prior_pass != index:
            print(
                f"Pass {training_pass.name!r}: fitting the linear injection prior on its "
                "train split.",
                flush=True,
            )
            _attach_pass_prior(
                module,
                experiment,
                setup.sources[index],
                setup.stage_kwargs["observed_rows"],
            )
            prior_pass = index
        manager = StageManager(
            experiment,
            restore_best=restore_best,
            **setup.stage_kwargs,
        )
        checkpoints += manager.run(
            module,
            datamodule,
            logger=pass_logger,
            extra_callbacks=callbacks,
            optimizer_state=optimizer_state,
        )
        optimizer_state = manager.last_optimizer_state
    return TrainingResult(
        module=module, history=history, checkpoints=checkpoints, setup=setup
    )


def _init_optimizer_state(cfg: TrainingRunConfig) -> Optional[dict]:
    """The optimizer state of ``cfg.init_from`` when the first pass carries optimizer state.

    ``None`` without ``init_from``, without ``train.carry_optimizer_state`` on the first
    pass that trains, or when the checkpoint holds no optimizer state (a stage checkpoint
    written by Lightning holds one).
    """
    if cfg.init_from is None:
        return None
    first = next((p.experiment for p in cfg.passes if p.experiment.curriculum.stages), None)
    if first is None or not first.train.carry_optimizer_state:
        return None
    import torch

    states = torch.load(cfg.init_from, map_location="cpu", weights_only=False).get(
        "optimizer_states"
    )
    if not states:
        print(
            f"{cfg.init_from} holds no optimizer state; the first stage starts fresh.",
            flush=True,
        )
        return None
    return states[0]


def _pass_logger(training_pass: TrainingPass, rank: int):
    """One MLflow logger for a pass, on rank 0 only (non-zero ranks train unlogged)."""
    if rank != 0:
        return False
    from .train.loop import mlflow_logger

    experiment = training_pass.experiment
    logger = mlflow_logger(
        experiment_name=experiment.train.mlflow_experiment,
        tracking_uri=experiment.train.mlflow_tracking_uri,
        run_name=f"{experiment.name}-{training_pass.name}",
    )
    logger.log_hyperparams(experiment.model_dump())
    return logger


def _assert_passes_swappable(cfg: TrainingRunConfig) -> None:
    """Reject a run whose passes disagree on a field the module DERIVES at construction.

    The stage manager checks this when it adopts a pass's config; doing it here too means a
    mismatch in the LAST pass costs seconds instead of the training time of every pass
    before it.
    """
    from .train import assert_cfg_swappable

    first = cfg.passes[0].experiment
    for training_pass in cfg.passes[1:]:
        assert_cfg_swappable(first, training_pass.experiment)


def _example() -> TrainingRunConfig:
    """A representative two-pass run: the snapshot / sequence scarcity curriculum on one grid."""
    from .config import (
        CurriculumConfig,
        DataConfig,
        GNNConfig,
        MeasurementConfig,
        PhysicsConfig,
        TemporalConfig,
        TrainConfig,
    )
    from .curriculum import snapshot_stages, sequence_stages

    def _pass(name: str, dataset: str, stages) -> TrainingPass:
        return TrainingPass(
            name=name,
            experiment=ExperimentConfig(
                name=f"se_curriculum-{name}",
                seed=0,
                orders=[1, 3, 5, 7, 9],
                data=DataConfig(source="parquet", dataset_path=dataset, batch_size=16),
                model=TemporalConfig(spatial=GNNConfig()),
                physics=PhysicsConfig(decode="injection"),
                measurement=MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"]),
                curriculum=CurriculumConfig(stages=stages),
                train=TrainConfig(checkpoint_dir="data/se_run/checkpoints"),
            ),
        )

    targets = {"target_full": 1e-4, "target_placement": 1e-3}
    return TrainingRunConfig(
        name="se_curriculum",
        grid_path="data/se_run/grid.json",
        passes=[
            _pass(
                "taskA",
                "data/se_run/dataset_random",
                snapshot_stages(epochs=20, **targets),
            ),
            _pass(
                "taskB",
                "data/se_run/dataset_coherent",
                sequence_stages(epochs=40, **targets),
            ),
        ],
    )


__all__ = [
    "PREFLIGHT_SAFE_FRACTION",
    "TrainingPass",
    "TrainingResult",
    "TrainingRunConfig",
    "TrainingSetup",
    "build_training_setup",
    "check_preflight_headroom",
    "launch_rank",
    "load_grid",
    "preflight_training",
    "preflight_verdict",
    "run_training",
]
