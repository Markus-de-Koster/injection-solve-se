"""Running a configured experiment: the data module, the memory probe, and the fit.

:func:`train_experiment` executes ONE :class:`~injection_hse.config.ExperimentConfig` end to end —
build the data source, fit the normalizer(s), assemble the physics operators, construct the
module, and run its curriculum through a :class:`~injection_hse.train.StageManager`. A full run is
several passes over one module (the task curriculum); that contract is :mod:`injection_hse.workflow`,
which reuses the pieces here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import lightning as L
import torch

from ..data import (
    MeasurementModel,
    StateCollate,
    base_voltage_per_row,
    build_data_source,
    per_unit_state,
)
from ..encoding import build_encoding, build_injection_encoding
from ..normalization import HarmonicNormalizer
from .anchoring import anchor_build_weights
from .dtypes import training_dtypes
from .layout import _to_nhc
from .module import SEModule
from .multigrid import fit_multigrid_normalizer
from .stages import StageManager

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from ..config import ExperimentConfig


class _SEDataModule(L.LightningDataModule):
    """Wraps a :class:`~injection_hse.data.DataSource` into Lightning train/val loaders."""

    def __init__(self, source, *, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._num_workers = num_workers
        # the source owns its batch layout: :class:`~injection_hse.data.StateCollate` stacks samples of
        # ONE grid, while a multi-grid source concatenates samples of DIFFERENT grids onto the
        # node axis. Falling back to StateCollate keeps a source written before
        # ``DataSource.collate`` existed working.
        collate = getattr(source, "collate", None)
        self._collate = StateCollate() if collate is None else collate()

    def _loader_kwargs(self) -> dict:
        """Worker/transfer settings shared by both loaders.

        Workers PERSIST across epochs: the dataset a worker holds is the whole in-memory
        source, so the per-epoch re-fork of that heap is pure overhead. Host memory is
        pinned when the training device is CUDA (an async H2D copy path).
        """
        kwargs: dict = {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "collate_fn": self._collate,
            "pin_memory": torch.cuda.is_available(),
        }
        if self._num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self._source.train_dataset(), shuffle=True, **self._loader_kwargs()
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self._source.val_dataset(), shuffle=False, **self._loader_kwargs()
        )


def make_datamodule(source, *, batch_size: int, num_workers: int = 0):
    """Wrap a :class:`~injection_hse.data.DataSource` into a Lightning ``DataModule`` (train/val).

    The public constructor for the train/val loaders the :class:`StageManager` consumes —
    used when driving the curriculum directly (e.g. continuing one module's weights across
    two datasets) instead of via :func:`train_experiment`. ``source`` supplies the train/val
    ``torch`` datasets AND its own ``collate_fn`` (:meth:`injection_hse.data.DataSource.collate`): the
    single-grid collate stacks the complex ``v`` (and any per-scenario sampled / injection /
    physics tensors) over the batch axis, while a multi-grid source concatenates items of
    DIFFERENT grids onto the node axis. ``batch_size`` counts scenario ITEMS either way — for a
    multi-grid source, member graphs.
    """
    return _SEDataModule(source, batch_size=batch_size, num_workers=num_workers)


def probe_step_memory(module: SEModule, datamodule, *, device) -> Optional[int]:
    """Run ONE training step (forward + backward) on a single batch to fail FAST on OOM.

    Surfaces an out-of-memory error at the chosen batch size BEFORE a long run — useful when a
    later, heavier stage would otherwise OOM only after hours on an earlier, lighter one (the
    node-coherent ``[B, T, H, N]`` data is the memory-heaviest because of its time axis, yet it
    trains last). Loads one batch from ``datamodule``'s train loader, moves the module + batch to
    ``device``, and runs the real forward + backward so the peak matches training. Returns the
    CUDA peak-allocated bytes (``None`` on CPU); raises a clear error naming the batch size on
    OOM. Leaves the module's weights, mode, AND mask RNG unchanged, so the subsequent run is
    byte-for-byte reproducible (the probe does not consume a mask draw).
    """
    import gc

    dev = torch.device(device)
    was_training = module.training
    rng_state = (
        module._mask_gen.get_state()
    )  # restore so the probe consumes no mask draw
    module.to(dev)
    module.train()
    batch = next(iter(datamodule.train_dataloader()))
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    try:
        loss = module._step(batch, train=True)[0]
        loss.backward()
    except RuntimeError as exc:  # pragma: no cover - exercised on a memory-limited GPU
        if "out of memory" in str(exc).lower():
            module.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            bs = int(batch["v"].shape[0])
            raise RuntimeError(
                f"batch_size={bs} does not fit on {dev} for this stage (forward+backward "
                f"out of memory). Reduce the batch size or the model width before the run. "
                f"Original error: {exc}"
            ) from exc
        raise
    finally:
        module._mask_gen.set_state(rng_state)
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    module.zero_grad(set_to_none=True)
    del loss, batch
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if not was_training:
        module.eval()
    return peak


def _train_multigrid(cfg: ExperimentConfig) -> SEModule:
    """Run a MULTI-GRID experiment (many grids, grid-level split).

    The counterpart of the single-grid setup in :func:`train_experiment`: one dataset-global
    voltage normalizer, and — when the config asks for the physics (the injection decode, the
    per-device localization head, or measured branch currents) — a
    :class:`~injection_hse.physics.MultiGridPhysicsBank` of per-member operators plus the pooled
    normalizers of the spaces it opens (injection currents, branch currents, terminal power),
    all in the per-unit current base that makes them comparable across voltage classes. The
    bank is ATTACHED to the source for the localization head, which is what makes the collate
    emit the per-device axes.
    """
    source = build_data_source(cfg.data)
    if list(source.orders) != list(cfg.orders):
        cfg = cfg.model_copy(update={"orders": list(source.orders)})
    encoding = build_encoding(cfg.encoding.kind)
    normalizer = fit_multigrid_normalizer(source, encoding, cfg)

    bank = None
    inj_normalizer = None
    edge_normalizer = None
    power_normalizer = None
    injection_orders = None
    if (
        cfg.physics.needs_solver
        or cfg.loss.predict_injection
        or cfg.measurement.measure_current
    ):
        from injection_hse.physics import (
            MultiGridPhysicsBank,
            fit_multigrid_edge_normalizer,
            fit_multigrid_injection_normalizer,
        )

        _, complex_dtype = training_dtypes(cfg.train.dtype)
        # size the anchoring capability for the LARGEST weight any stage requests (and the
        # linear-state prior's own solve); each stage then dials the runtime scalar
        # (SEModule.set_stage_anchor -> bank.set_anchor_weights).
        build_v, build_i = anchor_build_weights(
            cfg.physics,
            [cfg.curriculum],
            measure_current=cfg.measurement.measure_current,
        )
        bank = MultiGridPhysicsBank(
            source,
            list(cfg.orders),
            dtype=complex_dtype,
            anchor_v_weight=build_v,
            anchor_i_weight=build_i,
            harmonic_slack=cfg.physics.harmonic_slack,
            load_shunt=cfg.physics.load_shunt,
        )
        if cfg.loss.predict_injection:
            # the device axes are emitted only for a run that consumes them; attaching also
            # validates that the members share one injected-order axis.
            source.attach_physics_bank(bank)
            injection_orders = source.injection_orders
        if cfg.physics.needs_solver:
            # the injection space: the decode's head AND the ``kcl`` input prior are
            # normalized in it
            inj_normalizer = fit_multigrid_injection_normalizer(
                bank, source, build_injection_encoding(cfg.encoding), cfg
            )
        if cfg.measurement.measure_current:
            edge_normalizer = fit_multigrid_edge_normalizer(bank, source, encoding, cfg)
            if cfg.measurement.measure_power:
                power_normalizer = fit_multigrid_edge_normalizer(
                    bank, source, encoding, cfg, quantity="power"
                )

    module = SEModule(
        cfg,
        normalizer=normalizer,
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        inj_normalizer=inj_normalizer,
        edge_normalizer=edge_normalizer,
        power_normalizer=power_normalizer,
        injection_orders=injection_orders,
    )
    logger = mlflow_logger(
        experiment_name=cfg.train.mlflow_experiment,
        tracking_uri=cfg.train.mlflow_tracking_uri,
        run_name=cfg.name,
    )
    logger.log_hyperparams(cfg.model_dump())
    dm = make_datamodule(
        source, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    StageManager(cfg).run(module, dm, logger=logger)
    return module


def mlflow_logger(*, experiment_name: str, tracking_uri: str, run_name: str):
    """A Lightning ``MLFlowLogger`` for a run's tracking store.

    Runs track into a store of their own choosing (``TrainConfig.mlflow_tracking_uri``,
    by default a ``file:`` store next to the run's outputs so a run directory is
    self-contained and portable). MLflow treats file stores as a legacy backend and,
    from 3.11 on, refuses them unless ``MLFLOW_ALLOW_FILE_STORE`` opts in — this helper
    opts in for file-store URIs (an explicit user setting is left untouched) before the
    logger is created, so a database URI behaves exactly as before.
    """
    if tracking_uri.startswith("file:") or "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return L.pytorch.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )


def train_experiment(cfg: ExperimentConfig, *, grid: Optional[Grid] = None):
    """End-to-end entry point: build data source + model + normalizer, run the curriculum.

    Reproducible from ``cfg`` (+ ``cfg.seed``) alone; logs to MLflow and writes checkpoints.
    The ``grid`` the dataset was generated on is required (the parquet dataset stores only
    the node-phase row layout, not the topology / parameters the graph + measurement model
    need) — EXCEPT for a ``data.multigrid`` run, where each member grid is persisted beside
    its own scenarios and the normalizer is fit across all of them in per-unit. Returns the
    trained :class:`SEModule`.
    """
    if not cfg.curriculum.stages:
        raise ValueError(
            "cfg.curriculum.stages is empty — a from-scratch experiment needs at least "
            "one stage (an empty list is reserved for resumed runs whose task pass is "
            "already covered by the checkpoint)."
        )
    L.seed_everything(int(cfg.seed), workers=True)
    if cfg.data.multigrid is not None:
        return _train_multigrid(cfg)
    if grid is None:
        raise ValueError(
            "train_experiment requires the grid the dataset was generated on (only a "
            "data.multigrid run may omit it)."
        )

    # switchable topology: one dataset per switch configuration under ``data.dataset_path``,
    # and one set of physics operators per configuration (built after the normalizer fit).
    from ..data import topology_set

    topo_set = topology_set(cfg.data)
    source = build_data_source(cfg.data, grid=grid)
    orders = source.orders
    if list(orders) != list(cfg.orders):
        # the dataset's solved orders are authoritative; keep the loss/metrics aligned
        cfg = cfg.model_copy(update={"orders": list(orders)})

    # fit the normalizer on the TRAIN split only (persisted with the checkpoint).
    # cap the fit sample so very large datasets don't materialize fully in memory; the
    # per-harmonic statistics are stable well below the full train set.
    encoding = build_encoding(cfg.encoding.kind)
    train_ds = source.train_dataset()
    n_fit = min(len(train_ds), 8192)
    train_v = torch.stack([train_ds[i]["v"] for i in range(n_fit)])  # [Bt, H, N]
    # THE FEATURE FRAME (``normalization.per_unit``): every normalizer below is fit in the
    # frame the module encodes in, so the fitted statistics and the training features
    # describe the same quantity. The per-row voltage base is available immediately; the
    # per-unit CURRENT base needs the network solver's operator scale and is taken after the
    # solver is built.
    base_v = base_voltage_per_row(grid)  # per-row L-N base (the per-unit reference)
    per_unit = bool(cfg.normalization.per_unit)
    train_feats = _to_nhc(
        encoding.encode(per_unit_state(train_v, base_v) if per_unit else train_v)
    )  # [Bt, N, H, C]
    normalizer = HarmonicNormalizer().fit(
        train_feats,
        magnitude_transform=cfg.normalization.magnitude_transform,
        per_harmonic=cfg.normalization.per_harmonic,
    )

    _, complex_dtype = training_dtypes(cfg.train.dtype)
    # Size the anchoring capability for the LARGEST weight any stage requests (per-stage
    # overrides may exceed the global; the linear-state prior's solve has weights of its
    # own); each stage then sets its own runtime scalar.
    build_v, build_i = anchor_build_weights(
        cfg.physics,
        [cfg.curriculum],
        measure_current=cfg.measurement.measure_current,
    )

    # measured branch currents: precompute the edge operator. Built BEFORE the solver so a
    # current-anchored injection decode can materialise its V->current map from it.
    edge_op = None
    edge_normalizer = None
    power_normalizer = None
    topology = None
    if topo_set is not None:
        # every variant's operators at once (the union layout is shared); the normalizers
        # below are fit through the FIRST variant, whose currents span the same scale.
        from ..data.topology import TopologyBank

        topology = TopologyBank.build(
            topo_set,
            grid,
            list(orders),
            edge_branch_ids=source.graph().edge_branch_ids,
            dtype=complex_dtype,
            measure_current=cfg.measurement.measure_current,
            decode=cfg.physics.decode,
            build_solvers=cfg.needs_solver,
            anchor_v_weight=build_v,
            anchor_i_weight=build_i,
            harmonic_slack=cfg.physics.harmonic_slack,
            load_shunt=cfg.physics.load_shunt,
        )
    if cfg.measurement.measure_current:
        from ..data.currents import BranchCurrentOperator

        edge_op = (
            topology.edge_ops[0]
            if topology is not None
            else BranchCurrentOperator(grid, list(orders), dtype=complex_dtype)
        )

    # the differentiable network solver + the injection normalizer on the true nodal
    # injections: the physics-informed decode (``physics.decode="injection"``) and the
    # physics-derived input priors (``physics.input_priors``, with either decode) both need
    # them, and so does the per-unit CURRENT base of metered branch currents
    # (:attr:`~injection_hse.config.ExperimentConfig.needs_solver`). The optional measurement anchoring
    # turns the observed voltages / branch currents into soft constraints on the solve (the
    # current anchor uses the edge operator built above).
    solver = None
    inj_normalizer = None
    if cfg.needs_solver:
        from injection_hse.physics import NetworkSolver

        # build the solver at the TRAINING complex dtype so the injection normalizer is fit on
        # decode-space targets at the same precision (the internal solve stays complex128).
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
                harmonic_slack=cfg.physics.harmonic_slack,
                load_shunt=cfg.physics.load_shunt,
            )
        )
    # the per-unit CURRENT base of the injection / branch-current features
    i_base = solver.i_base(base_v) if (per_unit and solver is not None) else None
    if edge_op is not None:
        # the edge (and power) normalizer on the SAME train sample the state normalizer was
        # fit on, in the same frame the module's edge channels are built in
        i_train = edge_op(train_v.to(complex_dtype))  # [Bt, (T,) H, E]
        e_base = (
            None if i_base is None else i_base.index_select(-1, edge_op.edge_src_rows)
        )
        i_feat = i_train if e_base is None else i_train / e_base.to(i_train.real.dtype)
        edge_normalizer = HarmonicNormalizer().fit(
            _to_nhc(encoding.encode(i_feat)),
            magnitude_transform=cfg.normalization.magnitude_transform,
            per_harmonic=cfg.normalization.per_harmonic,
        )
        if cfg.measurement.measure_power:
            v_train = per_unit_state(train_v, base_v) if per_unit else train_v
            v_src = v_train.to(complex_dtype).index_select(
                -1, edge_op.edge_src_rows
            )  # [Bt, (T,) H, E]
            power_normalizer = HarmonicNormalizer().fit(
                _to_nhc(encoding.encode(v_src * i_feat.conj())),
                magnitude_transform=cfg.normalization.magnitude_transform,
                per_harmonic=cfg.normalization.per_harmonic,
            )
    if solver is not None:
        from injection_hse.physics import fit_injection_normalizer

        inj_normalizer = fit_injection_normalizer(
            solver,
            train_v,
            build_injection_encoding(cfg.encoding),
            magnitude_transform=cfg.normalization.magnitude_transform,
            base=i_base,
            injection_spec=source.injection_spec,
            exclude_tied=cfg.physics.exclude_tied_rows,
        )

    graph = source.graph()
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=normalizer,
        graph=graph,
        base_v=base_v,
        injection_spec=source.injection_spec,
        solver=solver,
        inj_normalizer=inj_normalizer,
        grid=grid,
        edge_op=edge_op,
        edge_normalizer=edge_normalizer,
        power_normalizer=power_normalizer,
        topology=topology,
    )

    observed_rows = None
    order_capability = None
    observed_edge_rows = None
    if any(s.kind == "placement" for s in cfg.curriculum.stages):
        mm = MeasurementModel.resolve(
            grid,
            cfg.measurement.voltage_nodes,
            fundamental_nodes=cfg.measurement.fundamental_nodes,
            slack_known=cfg.measurement.slack_known,
        )
        observed_rows = mm.observed_rows(grid)
        if cfg.measurement.heterogeneous_orders:
            order_capability = mm.order_capability(grid, list(orders))
        if cfg.measurement.measure_current:
            observed_edge_rows = mm.observed_edges(grid)

    # the LINEAR INJECTION PRIOR (the distillation teacher / the prior-residual base):
    # fitted before training on the first ``physics.injection_prior_fit_states`` TRAIN
    # states at the deployed placement, then carried in every checkpoint. Skipped when the
    # config asks for neither.
    from .distillation import attach_injection_prior, prior_fit_states, prior_requested
    from .preconditioner import attach_loss_preconditioner

    if prior_requested(cfg):
        attach_injection_prior(
            module,
            cfg,
            prior_fit_states(source, cfg.physics.injection_prior_fit_states),
            observed_rows,
        )
    attach_loss_preconditioner(module, cfg, observed_rows)

    logger = mlflow_logger(
        experiment_name=cfg.train.mlflow_experiment,
        tracking_uri=cfg.train.mlflow_tracking_uri,
        run_name=cfg.name,
    )
    logger.log_hyperparams(cfg.model_dump())

    dm = _SEDataModule(
        source, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    StageManager(
        cfg,
        observed_rows=observed_rows,
        order_capability=order_capability,
        observed_edge_rows=observed_edge_rows,
    ).run(module, dm, logger=logger)
    return module
