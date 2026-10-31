"""The Lightning module: the estimator, its two decode modes, and the training step.

Two decode modes are selected by :attr:`~injection_hse.config.PhysicsConfig.decode`:

- ``"voltage"`` (default) — the model predicts the normalized voltage features directly
  and the loss supervises against the true state.
- ``"injection"`` — the model predicts the nodal INJECTION currents; the differentiable
  network solve ``V(h) = solve(Y(h), I_pred)`` reconstructs the voltages so the output
  satisfies the network law ``Y(h)·V(h) = I(h)`` by construction.

Lightning does NOT touch the autograd graph; the physics-informed injection decode and any
``pgg`` generation losses call ``pgml`` and gradients flow as in raw PyTorch. Use
``precision="32"`` on the complex physics path (no AMP). The fitted normalizer and the config
are saved in the checkpoint (``on_save_checkpoint``) so a run reproduces exactly and
inference de-normalizes correctly.

A ``data.multigrid`` batch is served by the mirrored methods of
:class:`injection_hse.train.multigrid.MultiGridMixin`, which this module inherits.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import lightning as L
import torch
from torch import Tensor

from ..data import edge_feature_dim, encoding_dim, per_unit_state, physical_state
from ..data.graph import BASE_NODE_FEATURE_DIM
from ..encoding import build_encoding, build_injection_encoding
from ..io import build_model_input
from ..loss import StateEstimationLoss
from ..masking import build_masker
from ..metrics import state_errors
from ..models import InjectionHead, build_model
from ..normalization import HarmonicNormalizer
from ..time_features import TIME_FEATURE_DIM
from ..time_features import time_features as _time_features
from .anchoring import (
    _anchor_ranges,
    _anchor_weight_from,
    anchor_build_weights,
    install_anchor_calibration,
)
from .checkpoint import _check_grid_fingerprint, load_multigrid_checkpoint
from .dtypes import training_dtypes
from .layout import _flatten_features, _to_nhc, _unflatten_features
from .multigrid import MultiGridMixin
from .priors import (
    kcl_prior,
    linear_state_prior,
    prior_anchor_weights,
    prior_feature_width,
    prior_kinds,
)

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from ..config import ExperimentConfig


def _cancelling_injection_rows(grid) -> Tensor:
    """int64 rows of nodes carrying BOTH a consuming and an injecting appliance.

    These are the node-phase rows where the injection is not identifiable from voltages
    alone. A load and a generator on one node enter the network only through their SUM, so
    ``+20 kW`` beside ``−20 kW`` and ``+2 kW`` beside ``−2 kW`` produce the identical voltage
    profile while their individual injections differ by an order of magnitude: no voltage
    measurement, however dense, can separate them, and the direct injection supervision
    penalizes a split the state cannot reveal.

    What CAN separate them is spectral: the two devices distort differently, so their
    harmonic injections do not cancel even where their fundamentals do. Reporting the
    injection error on these rows PER ORDER is therefore the diagnostic for whether the
    estimator has learned that association — the fundamental is expected to stay ambiguous
    while the harmonic orders become identifiable.

    Empty (and the diagnostic silently inactive) for a grid with no such node, or when the
    module was built without one.
    """
    if grid is None:
        return torch.zeros(0, dtype=torch.int64)
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import InjectionAppliance, Load

    consuming: set[int] = set()
    injecting: set[int] = set()
    for appliance in grid.appliances:
        if not isinstance(appliance, InjectionAppliance) or not appliance.in_service:
            continue
        target = consuming if isinstance(appliance, Load) else injecting
        target.add(int(appliance.node))

    index = node_phase_index(grid)
    rows: list[int] = []
    for node_id in sorted(consuming & injecting):
        rows.extend(index.rows(node_id))
    return torch.tensor(sorted(set(rows)), dtype=torch.int64)


def _zero_output_layer(
    model: torch.nn.Module, n_out_features: int, n_nodes: int
) -> bool:
    """Zero the model's OUTPUT projection — the last ``Linear`` emitting the injection features.

    The prior-residual decode starts training exactly AT the linear prior: with the output
    layer at zero the network contributes nothing on the first step and learns only the
    residual. Every shipped estimator ends in one such layer — the GNN / Graphormer
    ``head`` emits ``n_out_features`` per node, the DNN's final MLP layer emits
    ``n_nodes · n_out_features`` for the whole grid at once; returns whether one was found.
    """
    widths = {int(n_out_features), int(n_nodes) * int(n_out_features)}
    target = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and int(mod.out_features) in widths:
            target = mod
    if target is None:
        return False
    with torch.no_grad():
        target.weight.zero_()
        if target.bias is not None:
            target.bias.zero_()
    return True


class SEModule(MultiGridMixin, L.LightningModule):
    """LightningModule wrapping a :class:`~injection_hse.models.base.StateEstimator`.

    Owns the encoding, the fitted normalizer, the loss, and the (per-stage) masker. The
    ``training_step`` implements the pipeline above; ``configure_optimizers`` builds the
    optimizer + scheduler from :class:`~injection_hse.config.OptimConfig`. The fitted normalizer and
    the config are saved in the checkpoint (``on_save_checkpoint``) so a run reproduces
    exactly and inference de-normalizes correctly.

    A ``data.multigrid`` run trains ONE estimator over many grids, whose batches carry a
    different layout (their sample axis folded into the node axis). Those mirrored methods
    come from :class:`~injection_hse.train.multigrid.MultiGridMixin`; everything below the layout —
    the encoding, the normalizer(s), the loss, the anchoring, the Lightning hooks — is
    shared with the single-grid path.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        n_nodes: Optional[int] = None,
        normalizer: HarmonicNormalizer,
        graph=None,
        base_v: Optional[Tensor] = None,
        injection_spec=None,
        solver=None,
        inj_normalizer=None,
        grid: Optional[Grid] = None,
        edge_op=None,
        edge_normalizer: Optional[HarmonicNormalizer] = None,
        power_normalizer: Optional[HarmonicNormalizer] = None,
        topology=None,
        grid_keys: Optional[list[str]] = None,
        bank=None,
        injection_orders: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # MULTI-GRID run (``data.multigrid``): one estimator over many grids. There is no
        # single graph, row count or per-unit base — each batch carries its own concatenated
        # graph and the member each row belongs to. ``grid_keys`` names the member grids
        # (index-aligned with ``MultiGridDataSource.grids``) so the per-grid validation metrics
        # get a stable name. The PHYSICS (the network solve, the branch-current operator, the
        # per-unit current base) comes from a :class:`~injection_hse.physics.MultiGridPhysicsBank`, which
        # replaces the single-grid ``solver`` / ``edge_op`` and is dispatched per member inside
        # the step; ``injection_orders`` sizes the localization head, whose device axis the
        # collate concatenates across members.
        self._multigrid = cfg.data.multigrid is not None
        self._grid_keys = list(grid_keys) if grid_keys is not None else None
        self._bank = bank if self._multigrid else None
        self._injection_orders = (
            None if injection_orders is None else [int(o) for o in injection_orders]
        )
        # THE FEATURE FRAME: whether the complex state (and the currents derived from it)
        # enters the encoder in PER-UNIT. A multi-grid run decides it with
        # ``data.multigrid.per_unit`` and applies it in its own step (the per-row base
        # travels with the batch); a single-grid run with ``normalization.per_unit`` applies
        # it at the seams below, against this module's own ``base_v``. Either way the physics
        # keeps solving, anchoring and reporting in absolute units.
        self._per_unit = (
            cfg.data.multigrid.per_unit
            if self._multigrid
            else cfg.normalization.per_unit
        )
        # The ANGLE counterpart of ``per_unit``: which frame the multi-grid state lives in
        # (see :class:`~injection_hse.config.MultiGridConfig`). Inputs, targets, the estimate and every
        # metric share it. The single-grid path has one angle convention and stays absolute.
        self._angle_reference = (
            cfg.data.multigrid.angle_reference if self._multigrid else "absolute"
        )
        if not self._multigrid and (graph is None or n_nodes is None):
            raise ValueError(
                "SEModule needs the grid's graph and node-phase row count; only a "
                "data.multigrid run may omit them (its batches carry their own graph)."
            )
        # SWITCHABLE TOPOLOGY (optional): a :class:`~injection_hse.data.topology.TopologyBank` holding
        # one network solver / branch-current operator per switch configuration, plus the
        # per-variant channels the model conditions on. Each sample carries its variant index
        # (batch key ``topology``), so a batch spanning several topologies dispatches to the
        # matching operators. The bank supplies the ``solver`` / ``edge_op`` below (its first
        # variant is the representative used for the layout checks — every variant shares the
        # union row / half-edge layout by construction).
        self._topology = topology
        if topology is not None:
            if topology.edge_ops:
                edge_op = topology.edge_ops[0]
            if topology.solvers:
                solver = topology.solvers[0]
        self.n_nodes = n_nodes
        self.orders = list(cfg.orders)
        # TRAINING precision, decoupled from the dataset's stored dtype: the loaded complex
        # state is cast to ``_complex_dtype`` at the module boundary (:meth:`_step`) and the
        # model parameters live in the matching real dtype. ``"complex64"`` (default) is
        # float32 params — byte-identical to a run written before ``TrainConfig.dtype`` existed.
        self._real_dtype, self._complex_dtype = training_dtypes(cfg.train.dtype)
        self.encoding = build_encoding(cfg.encoding.kind)
        # The INJECTION head's encoding (``encoding.injection_kind``, else ``kind``): the space
        # the predicted / target nodal injection currents are encoded and decoded in, and the
        # one the injection normalizer was fit on. Every voltage feature keeps ``encoding``.
        self.inj_encoding = build_injection_encoding(cfg.encoding)
        self.normalizer = normalizer
        self._graph = graph
        # per-row line-to-neutral base voltage [N] (the per-unit reference for the metrics).
        self.register_buffer(
            "base_v",
            None if base_v is None else base_v.to(torch.float64),
            persistent=False,
        )
        # Whether the model output is decoded as nodal INJECTION currents (the physics decode,
        # ``physics.decode="injection"`` served by a solver / the operator bank) — decided here
        # because it sizes the model's OUTPUT width.
        self._predict_injection_state = cfg.physics.decode == "injection" and (
            bank is not None if self._multigrid else solver is not None
        )
        # ``n_channels`` is the INPUT / state channel count (the voltage encoding, ``C_in``);
        # ``n_out_channels`` the channel count of the model's raw OUTPUT (``C_out``): the
        # injection encoding's under the injection decode, else the same ``C_in``.
        self.n_channels = self.encoding.n_features
        self.n_out_channels = (
            self.inj_encoding.n_features
            if self._predict_injection_state
            else self.n_channels
        )
        self.n_harmonics = len(self.orders)
        n_features = self.n_harmonics * self.n_channels
        n_out_features = self.n_harmonics * self.n_out_channels

        # The metered slack is a measurement like any other, but one that never
        # disappears: its node-phase rows (every slack/source node — the `is_slack`
        # graph feature marks them all) are forced OBSERVED in every mask stage,
        # including random ones. Kept a plain CPU tensor (masks are drawn on CPU).
        # A multi-grid batch carries its members' slack rows as global indices instead.
        self._forced_obs_rows: Optional[Tensor] = None
        if cfg.measurement.slack_known and graph is not None:
            self._forced_obs_rows = (
                (graph.x[:, 5] > 0.5).nonzero(as_tuple=False).flatten().cpu()
            )

        # width of the static node-feature block the models fold into their token: the graph's
        # own when there is one, else the base block plus the configured structural encoding
        # (every member grid of a multi-grid run is built with the SAME encoding).
        self._n_static_features = (
            int(graph.x.shape[1])
            if graph is not None
            else BASE_NODE_FEATURE_DIM + encoding_dim(cfg.data.structural_encoding)
        )
        # width of the static EDGE-feature block: the graph's own when there is one, else the
        # series block plus the flag columns a phase-coupling graph appends (every member of a
        # multi-grid run is built with the SAME edge set rule).
        self._n_edge_static_features = (
            int(graph.edge_attr.shape[1])
            if graph is not None
            else edge_feature_dim(cfg.data.phase_coupling)
        )

        # OPTIONAL per-step periodic time features (OFF by default). When enabled every model
        # node token gains the ``TIME_FEATURE_DIM``-wide vector (sin/cos time-of-day / week /
        # year + a validity flag); a timeless batch supplies the all-zero vector so one model
        # trains across timeless snapshots and timestamped profiled sequences. Fixed at
        # construction (the model input width depends on it); 0 = byte-identical.
        self._n_time_features = TIME_FEATURE_DIM if cfg.encoding.time_features else 0

        # How the model is TOLD which topology a sample came from (see
        # :class:`~injection_hse.config.TopologyConfig`): the switch-state vector broadcast to every
        # node token, the per-half-edge conduction channel, both, or neither. Fixed at
        # construction (they widen the model input); without a topology both are 0 =
        # byte-identical to a single-topology run.
        topo_cfg = cfg.data.topology
        conditioning = topo_cfg.conditioning if topo_cfg is not None else "none"
        self._n_topology_features = (
            topology.topology.n_switches
            if topology is not None and conditioning in ("edge_and_global", "global")
            else 0
        )
        self._n_edge_topology_features = (
            1
            if topology is not None and conditioning in ("edge_and_global", "edge")
            else 0
        )

        # position of the fundamental in the solved-order axis (the heterogeneous
        # smart-meter demotion and the injection-decode slack boundary both need it).
        self._fundamental_pos = self.orders.index(1) if 1 in self.orders else None

        # HETEROGENEOUS sensor fleet (OFF by default): the per-order observability mask is
        # threaded through the model input and every node/edge token carries H per-order
        # channels. Fixed at construction (input width); off = byte-identical.
        self._hetero = cfg.measurement.heterogeneous_orders

        # OPTIONAL measured branch currents as model input (OFF by default). The edge
        # operator derives every half-edge current from the state (the same linear map the
        # Y-bus stamps define); its output is encoded like the voltages, normalized by the
        # edge normalizer, masked to the observed edges, and fed as dynamic edge
        # observations. ``measure_power`` appends the terminal apparent power S = V·I* as
        # additional per-order edge channels. The state estimate stays voltage-only.
        self._measure_current = cfg.measurement.measure_current
        self.edge_op = edge_op
        self.edge_normalizer = edge_normalizer
        self.power_normalizer = power_normalizer
        # one-CT-budget masks of a MULTI-GRID run: static per member, built on first sight
        # (the single-grid path holds its one mask in the ``_edge_upstream`` buffer below).
        self._edge_upstream_by_grid: dict[int, Tensor] = {}
        n_edge_features = 0
        if self._measure_current:
            if edge_normalizer is None:
                raise ValueError(
                    "measurement.measure_current=True needs a fitted edge_normalizer."
                )
            if cfg.measurement.measure_power and power_normalizer is None:
                raise ValueError(
                    "measurement.measure_power=True needs a fitted power_normalizer."
                )
            n_channels_edge = self.encoding.n_features * (
                2 if cfg.measurement.measure_power else 1
            )
            n_edge_features = self.n_harmonics * n_channels_edge
        if self._measure_current and self._multigrid:
            # the half-edge layout is per member: the operators come from the bank and the
            # batch carries the concatenated ``edge_index`` the observations are gathered at.
            self._edge_upstream = None
            if bank is None:
                raise ValueError(
                    "measurement.measure_current=True on a data.multigrid run needs the "
                    "MultiGridPhysicsBank that supplies each member's BranchCurrentOperator "
                    "(bank=MultiGridPhysicsBank(source, ...))."
                )
        elif self._measure_current:
            if edge_op is None:
                raise ValueError(
                    "measurement.measure_current=True needs a BranchCurrentOperator "
                    "(edge_op) and a fitted edge_normalizer."
                )
            if edge_op.n_edges != int(graph.edge_index.shape[1]) or not torch.equal(
                edge_op.edge_src_rows.cpu(), graph.edge_index[0].cpu()
            ):
                raise ValueError(
                    "edge operator half-edge layout does not match the graph's "
                    "edge_index (build both from the same grid and orders)."
                )
            self.register_buffer(
                "_edge_src_rows",
                graph.edge_index[0].to(torch.int64),
                persistent=False,
            )
            # one-CT budget: keep only each metered location's half-edge toward the
            # slack (the boundary/slack location keeps its incident feeders). Static
            # per grid, so it composes with ANY node mask — fixed placements and the
            # random-masking curriculum alike.
            if cfg.measurement.current_channels == "upstream":
                if grid is None:
                    raise ValueError(
                        "measurement.current_channels='upstream' needs the grid to "
                        "derive the toward-slack half-edge per metered location."
                    )
                from injection_hse.data import upstream_edge_mask

                self.register_buffer(
                    "_edge_upstream",
                    upstream_edge_mask(grid, graph.edge_index),
                    persistent=False,
                )
            else:
                self._edge_upstream = None

        # PHYSICS-DERIVED INPUT PRIORS (``physics.input_priors``, OFF by default): the
        # injection the observed neighbourhood implies (``kcl``, in the injection encoding +
        # normalizer, plus the observed-neighbourhood fraction) and / or the
        # measurement-anchored zero-injection state (``linear_state``, in the state encoding +
        # normalizer), appended to every node token (:meth:`input_priors`). Fixed at
        # construction (they widen the model input); empty = byte-identical. Both need the
        # network solver (checked below, once it is known), with EITHER decode.
        self._input_priors = prior_kinds(cfg.physics.input_priors)
        self._n_prior_features = prior_feature_width(
            self._input_priors,
            n_orders=self.n_harmonics,
            n_state_channels=self.n_channels,
            n_injection_channels=self.inj_encoding.n_features,
        )

        self.model = build_model(
            cfg.model,
            n_features=n_features,
            n_nodes=n_nodes,
            n_time_features=self._n_time_features,
            n_topology_features=self._n_topology_features,
            n_order_mask_channels=self.n_harmonics if self._hetero else 0,
            n_prior_features=self._n_prior_features,
            n_edge_features=n_edge_features,
            n_edge_topology_features=self._n_edge_topology_features,
            n_static_features=self._n_static_features,
            n_edge_static_features=self._n_edge_static_features,
            n_out_features=n_out_features,
        )
        # PRIOR-RESIDUAL decode (``physics.injection_prior_residual``): the network's output
        # layer starts at ZERO, so the first estimate is exactly the linear prior's and the
        # network learns the residual on top (the prior itself is installed before training
        # by :func:`injection_hse.train.attach_injection_prior`). A checkpoint reload overwrites the
        # weights afterwards, so a trained residual is restored as trained.
        self._prior_residual = bool(cfg.physics.injection_prior_residual)
        #: the harmonic slack mode of the decode (see PhysicsConfig.harmonic_slack); read by
        #: _slack_voltage and by the deployment engine when it rebuilds the solver
        self._harmonic_slack = str(cfg.physics.harmonic_slack)
        #: the decode's harmonic device-shunt model (see PhysicsConfig.load_shunt); read by
        #: the deployment engine when it rebuilds the solver
        self._load_shunt = cfg.physics.load_shunt
        if self._prior_residual and not _zero_output_layer(
            self.model, n_out_features, n_nodes
        ):
            raise ValueError(
                "physics.injection_prior_residual needs a model whose output projection is "
                "a Linear layer emitting the injection features; none was found."
            )
        # Follow the training precision: complex128 -> float64 parameters. Guarded so the
        # default (float32) path never touches the freshly-built model — the parameters, and
        # hence the seeded initialization, stay byte-identical. Casting here (not only via the
        # Lightning precision plugin) keeps the OFF-trainer paths correct too: ``infer``,
        # ``probe_step_memory``, ``from_checkpoint`` reload, and direct ``_step`` calls all see
        # a model whose parameter dtype already matches the cast features.
        if self._real_dtype != torch.float32:
            self.model = self.model.to(self._real_dtype)
        self.loss_fn = StateEstimationLoss(
            cfg.loss,
            encoding=self.encoding,
            normalizer=self.normalizer,
            orders=self.orders,
        )

        # OPTIONAL per-device injection / localization head (OFF by default). Built only when
        # both the config enables it AND the data carries per-device injection samples;
        # otherwise node-V SE is byte-identical to a run without it. A MULTI-GRID run has no
        # single :class:`~injection_hse.data.InjectionSpec` — the head is sized by the source's validated
        # ``injection_orders`` and pools at the GLOBAL device rows the collate concatenates,
        # so one head serves every member. The head is constructed AFTER ``self.model`` on
        # purpose: building it consumes the global RNG, so building it first would shift the
        # seeded initialization of the state model. Keep this order.
        self._injection_spec = injection_spec
        self.injection_head = None
        if self._multigrid:
            n_inj_orders = len(self._injection_orders) if self._injection_orders else 0
        else:
            n_inj_orders = (
                0
                if injection_spec is None or injection_spec.is_empty
                else int(injection_spec.n_orders)
            )
        if cfg.loss.predict_injection and n_inj_orders:
            # the head pools the model's raw OUTPUT features, whose width is ``n_out_features``
            self.injection_head = InjectionHead(
                node_feature_dim=n_out_features,
                n_orders=n_inj_orders,
                hidden_dim=cfg.loss.injection_hidden_dim,
                n_channels=1,
                dropout=cfg.model.dropout,
            )
            # follow the training precision alongside the state model (guarded — float32
            # default is untouched, so a run with the head is byte-identical there too).
            if self._real_dtype != torch.float32:
                self.injection_head = self.injection_head.to(self._real_dtype)
            if not self._multigrid:
                self.register_buffer(
                    "_inj_rows", injection_spec.device_rows, persistent=False
                )
                self.register_buffer(
                    "_inj_rows_mask", injection_spec.device_rows_mask, persistent=False
                )
        elif cfg.loss.predict_injection and self._multigrid:
            raise ValueError(
                "loss.predict_injection=True on a data.multigrid run needs the members' "
                "injected orders (injection_orders=source.injection_orders); the head's "
                "order axis is shared by every member."
            )

        # PHYSICS-INFORMED DECODE (``physics.decode="injection"``): the model predicts the nodal
        # INJECTION currents and the differentiable network solve reconstructs the voltages
        # (``V(h) = solve(Y(h), I_pred)``) — so the output satisfies the network law by
        # construction, with no ground-truth injection on the RHS. The injection normalizer
        # puts the (wide-dynamic-range) per-order currents in a learnable feature range; the
        # measured slack voltage is the fundamental boundary. ``decode="voltage"`` (default) is
        # the original direct-voltage estimator (``solver`` unused, byte-identical). A
        # MULTI-GRID run decodes through the bank instead — one solver per member, dispatched
        # per batch segment (:meth:`_multigrid_decode`).
        self.solver = solver
        # The physics solve runs in complex128 internally regardless (the feeder Y-bus is
        # ill-conditioned); only its DECODE output dtype follows the training precision, so the
        # reconstructed voltages re-encode into the SAME real dtype as the supervised target.
        # (The solver was built once, possibly at the default complex64; the training dtype is
        # authoritative.)
        if solver is not None:
            solver._dtype = self._complex_dtype
        if bank is not None:
            bank.set_decode_dtype(self._complex_dtype)
        self.inj_normalizer = inj_normalizer
        # the switch-tied rows of the solver's grid (``physics.exclude_tied_rows``): no
        # estimable injection lives there, so the head's output at those rows is pinned to
        # the zero-injection features before the decode and the targets are zero there
        tied = (
            solver.tied_rows
            if (
                solver is not None
                and not self._multigrid
                and cfg.physics.exclude_tied_rows
            )
            else torch.zeros(0, dtype=torch.int64)
        )
        self.register_buffer("_tied_rows", tied.to(torch.int64).cpu(), persistent=False)
        # the measured slack rows of the solver's grid — the fundamental boundary of every
        # network solve this module runs (the injection decode, the zero-injection prior)
        if solver is not None and not self._multigrid:
            self.register_buffer(
                "_slack_rows", solver.slack_rows.to(torch.int64), persistent=False
            )
        self._check_input_priors(solver, bank)
        self._check_feature_frame()
        if self._multigrid and cfg.physics.decode == "injection" and bank is None:
            # a single-grid run may legitimately be built without a solver (evaluation of a
            # voltage decode); a multi-grid one may not silently fall back to it — the config
            # asked for the physics decode and there is nothing to decode with.
            raise ValueError(
                "physics.decode='injection' on a data.multigrid run needs the per-member "
                "operator bank (bank=MultiGridPhysicsBank(source, ...)) and the injection "
                "normalizer fit across its members "
                "(injection_hse.physics.fit_multigrid_injection_normalizer)."
            )
        if self._predict_injection_state:
            if not cfg.measurement.slack_known:
                raise ValueError(
                    "physics.decode='injection' requires a metered slack: the measured "
                    "fundamental slack voltage is the boundary of the reduced network "
                    "solve. Set measurement.slack_known=True (i.e. install a voltage "
                    "measurement at every slack node) or use physics.decode='voltage'."
                )
            if inj_normalizer is None:
                raise ValueError(
                    "physics.decode='injection' needs an injection normalizer (fit on the "
                    "true nodal injections Y(h)·V_true)."
                )
            if not self._multigrid:
                if solver.slack_rows.numel() == 0 and 1 in self.orders:
                    raise ValueError(
                        "physics.decode='injection' found no slack rows: the grid has no "
                        "in-service Source, so there is no slack measurement to anchor "
                        "the fundamental solve."
                    )
            self.register_buffer(
                "_cancelling_rows", _cancelling_injection_rows(grid), persistent=False
            )

        # DIRECT INJECTION SUPERVISION (``loss.injection_state_loss_split``): the same loss
        # structure as the voltage term but in the normalized INJECTION space, comparing the
        # decode's predicted nodal injection currents against the true ones Y(h)·V_true(h).
        # Built whenever the injection decode is active (it holds no parameters and costs
        # nothing until a positive split asks for the term, which is read per step so a
        # two-task curriculum can change it between passes).
        self.inj_loss_fn = (
            StateEstimationLoss(
                cfg.loss,
                encoding=self.inj_encoding,
                normalizer=self.inj_normalizer,
                orders=self.orders,
            )
            if self._predict_injection_state
            else None
        )
        if cfg.loss.plain_decode_weight > 0.0 and not self._predict_injection_state:
            warnings.warn(
                "loss.plain_decode_weight > 0 has no effect without "
                "physics.decode='injection': the plain-decode term supervises the "
                "slack-only solve of the predicted nodal injections, which the voltage "
                "decode does not produce. Ignoring it.",
                stacklevel=2,
            )
        if cfg.loss.injection_state_loss_split > 0.0 and self.inj_loss_fn is None:
            warnings.warn(
                "loss.injection_state_loss_split > 0 has no effect without "
                "physics.decode='injection': the direct injection term supervises the "
                "injection decode's predicted nodal currents, which the voltage decode "
                "does not produce. Ignoring it.",
                stacklevel=2,
            )
        if cfg.loss.prior_distillation_weight > 0.0 and self.inj_loss_fn is None:
            warnings.warn(
                "loss.prior_distillation_weight > 0 has no effect without "
                "physics.decode='injection': the distillation term supervises the injection "
                "decode's predicted nodal currents against the linear prior's, and the "
                "voltage decode produces no such currents. Ignoring it.",
                stacklevel=2,
            )
        # THE DISTILLATION TEACHER (``loss.prior_distillation_weight``): the fitted linear
        # injection prior, installed before training by :func:`injection_hse.train.attach_injection_prior`
        # and restored from the checkpoint on a reload, so the objective is reproducible.
        # ``None`` until installed — a positive weight without one is refused at the first step
        # rather than trained through as a silently different objective.
        self.injection_prior = None
        # THE LOSS PRECONDITIONER (``loss.preconditioner``): the backward-only curvature
        # normalisation of the injection head's output, built before training by
        # :func:`injection_hse.train.attach_loss_preconditioner` (``None`` = plain gradients).
        self.loss_preconditioner = None
        # per-stage distillation weight installed by the StageManager (``None`` = the config's
        # global weight), so a curriculum can distil early and finish on the physics objective
        self._stage_distillation: Optional[float] = None
        # The supervised loss split into its components (voltage / injection state), refreshed
        # by every :meth:`_step` and logged per step so the blend can be tuned from the curves.
        # The voltage part is reported for EVERY run, blend or not: the reported ``*_loss`` is
        # the blend, so without the split its curve is the only way to see the term that the
        # per-unit state metrics track.
        self._loss_components: dict[str, Tensor] = {}
        # The PLAIN (slack-only) decode's state estimate of the last VALIDATION step, detached
        # — set only while ``loss.plain_decode_weight`` adds the plain-decode term (the solver
        # anchors and the weight is positive), so the validation step can report what the
        # network itself achieves (``val_pu_mae_unobs_plain``) beside the anchored estimate.
        self._plain_pred_v: Optional[Tensor] = None
        # The fraction of the blended loss the injection term actually contributes — the
        # quantity ``injection_state_loss_split`` is meant to set. Logged separately from the
        # terms because it is the one number that says whether the split means what it says.
        self._loss_share: Optional[Tensor] = None
        # Per-order injection identifiability at the load+generator nodes, refreshed by every
        # VALIDATION step of an injection decode (empty otherwise).
        self._injection_diag: dict[str, float] = {}
        # Eval-only seam: when set, ``_step`` additionally stashes each supervised term as a
        # per-(row, order) map (:attr:`_loss_maps`), which
        # :func:`injection_hse.evaluation.loss_term_maps` accumulates into the loss heatmap. OFF during
        # training — the maps are the loss's own intermediate, but keeping them alive per
        # batch is pointless there.
        self._collect_loss_maps = False
        self._loss_maps: dict[str, Tensor] = {}

        # OPTIONAL learned grid-parameter calibration (Stage 1; injection decode only). A small,
        # shared correction on each configured grid parameter is trained JOINTLY with the
        # estimator and regularised toward 0 by its confidence — so a wrong line length /
        # resistance can be nudged to fit the measurements (beating the nominal-Y accuracy cap).
        self.grid_calibration = None
        if self._predict_injection_state and cfg.physics.calibrate:
            from injection_hse.calibration import GridCalibration

            if self._topology is not None:
                raise ValueError(
                    "physics.calibrate is not supported together with a switchable "
                    "topology: the calibration re-assembles ONE solver's operator per step, "
                    "while a topology run holds one solver per switch configuration."
                )
            self.grid_calibration = GridCalibration(
                solver._grid, list(cfg.physics.calibrate)
            )

        # count of training batches skipped for a non-finite loss (the batch guard in
        # :meth:`training_step`; a handful is a survivable data corner, a stream aborts)
        self._nonfinite_batches = 0
        # per-stage anchor state: fixed (validation) weights + optional per-batch
        # randomization ranges, installed by set_stage_anchor; the dedicated generator
        # keeps the batch draws reproducible and independent of the mask stream.
        self._stage_anchor = (0.0, 0.0)
        # how a stage's randomization range is sampled per batch (see PhysicsConfig)
        self._anchor_draw = cfg.physics.anchor_weight_draw
        self._anchor_ranges = None
        self._anchor_gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
        # the active masker (swapped per stage by the StageManager)
        self._masker = build_masker("full")
        # per-stage optimizer overrides (installed by the StageManager before each fit)
        self._stage_optim: dict = {}
        # a torch.Generator for per-sample masks, seeded from cfg (reproducibility)
        self._mask_gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
        self._mask_gen_rank_seeded = False

        # OPTIONAL measurement noise on the OBSERVED phasors (OFF by default). Perturbs the
        # model INPUT (voltage phasors) keyed by each sensor's accuracy class; the supervised
        # targets and the reported metrics stay clean. Built only when enabled, so a noise-off
        # run draws nothing and is byte-identical.
        self._noise = None
        # Independent branch-current (CT) noise stream, built only when noise is on AND branch
        # currents are metered — a current channel inherits its metering node's accuracy class
        # but draws from its own stream (see :meth:`SensorNoise.for_edges`). Applied to the
        # STATE-DERIVED edge currents only; real measured currents (``i_meas``) carry their own
        # physical error and are left untouched.
        self._edge_noise = None
        if cfg.noise.enabled:
            from ..data import SensorNoise
            from ..data.noise import _EDGE_NOISE_SEED_OFFSET, grid_current_range

            if self._multigrid:
                # No single row layout across members and no device metadata on a
                # generated corpus: uniform default-class sigmas, layout-independent.
                # The state is per-unit, so the voltage floor references nominal = 1;
                # the current channels carry reading/phase error only (their range has
                # no defined base on the per-unit path).
                self._noise = SensorNoise.uniform(
                    cfg.noise, nominal=1.0, orders=self.orders
                )
                if self._measure_current:
                    self._edge_noise = SensorNoise.uniform(
                        cfg.noise, nominal=None, seed_offset=_EDGE_NOISE_SEED_OFFSET
                    )
            elif cfg.noise.instrument is not None:
                # the datasheet instrument model (single grid): limits referred to each
                # row's nominal voltage and the grid's CT range
                from ..data import InstrumentNoise

                self._noise = InstrumentNoise.from_config(
                    cfg.noise, n_nodes=n_nodes, grid=grid, orders=self.orders
                )
            else:
                self._noise = SensorNoise.from_config(
                    cfg.noise, n_nodes=n_nodes, grid=grid, orders=self.orders
                )
            if not self._multigrid and self._measure_current:
                self._edge_noise = self._noise.for_edges(
                    self._edge_src_rows,
                    current_range_a=(
                        grid_current_range(grid) if grid is not None else None
                    ),
                )
            if cfg.noise.draw_device == "input":
                for model in (self._noise, self._edge_noise):
                    if model is not None:
                        model.draw_on_input_device()

        # Noise-calibrated anchor weights (``physics.anchor_weighting``): derived from the
        # fitted scales and the noise model just built, so the solver weights each anchored
        # channel by the inverse variance of the error it actually sees; re-derived
        # identically on a checkpoint reload (config + normalizers travel with it).
        install_anchor_calibration(self)

        # Factor-once operator caches (``physics.operator_cache_mb``): the network operators
        # are grid-constant while a run solves them tens of thousands of times, so the solve
        # reuses its factorizations under the configured byte budget. ``0`` = off (the
        # historical per-call factorization). One budget spans every solver this module
        # decodes with, whichever the batch builds first.
        cache_mb = int(cfg.physics.operator_cache_mb)
        if cache_mb > 0:
            if self._bank is not None:
                self._bank.enable_operator_cache(cache_mb)
            elif self._solvers:
                from ..physics import OperatorCacheBudget

                shared = OperatorCacheBudget(cache_mb * 2**20)
                for member in self._solvers:
                    member.enable_operator_cache(shared)

        # Grid identity carried into the checkpoint: the stable row-layout fingerprint plus
        # the node-id / phase-code lists (for a legible mismatch message). Recorded only when
        # the grid is supplied — a module built without it saves no fingerprint (an older
        # checkpoint of unknown provenance, which reloads with a warning rather than a crash).
        self._layout_fingerprint: Optional[str] = None
        self._layout_node_ids: Optional[list] = None
        self._layout_phase_codes: Optional[list] = None
        if grid is not None:
            from pgml.assembly import node_phase_index
            from pgml.topology import layout_fingerprint

            self._layout_fingerprint = layout_fingerprint(grid)
            idx = node_phase_index(grid)
            self._layout_node_ids = idx.node_ids.cpu().tolist()
            self._layout_phase_codes = idx.phase_codes.cpu().tolist()

        # running training-loss accumulator for the epoch MEAN (see :attr:`train_epoch_loss`).
        self._train_loss_sum: float = 0.0
        self._train_loss_count: int = 0
        # per-MEMBER validation error of a multi-grid run: a device-resident
        # ``(Σ|Δ|/base [G], element count [G])`` pair accumulated over the whole validation
        # epoch — a mean of per-batch means would weight a member by how often it happened
        # to be drawn, not by its size.
        self._val_grid_error: Optional[tuple[Tensor, Tensor]] = None
        self.save_hyperparameters({"config": cfg.model_dump()})

    def setup(self, stage: str) -> None:
        """Rank-decorrelate the mask stream under distributed (multi-GPU) training.

        ``seed_everything`` gives every rank the SAME global seed (required for identical
        weight init), so without an offset all ranks would draw the SAME mask sequence for
        DIFFERENT sample shards — correlated masks inside every global batch. Re-seed the
        mask generator once with a per-rank offset. Single-process training is untouched
        (the stream stays the continuous ``cfg.seed + 1`` sequence across all stages), so
        runs are bitwise reproducible at fixed (seed, world size); equivalence ACROSS world
        sizes is not a goal — no data-parallel framework provides it.
        """
        if (
            not self._mask_gen_rank_seeded
            and self._trainer is not None
            and self.trainer.world_size > 1
        ):
            rank = int(self.trainer.global_rank)
            self._mask_gen = torch.Generator().manual_seed(
                int(self.cfg.seed) + 1 + rank
            )
            # decorrelate the noise draws across ranks too (same rationale as the mask stream)
            if self._noise is not None:
                self._noise.reseed(int(self.cfg.noise.seed) + rank)
            if self._edge_noise is not None:
                from ..data.noise import _EDGE_NOISE_SEED_OFFSET

                self._edge_noise.reseed(
                    int(self.cfg.noise.seed) + _EDGE_NOISE_SEED_OFFSET + rank
                )
            self._mask_gen_rank_seeded = True

    # -- construction from a checkpoint ----------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        *,
        grid: Optional[Grid] = None,
        device="cpu",
        source=None,
    ) -> SEModule:
        """Rebuild a trained module from a stage checkpoint + the data it trained on.

        THE checkpoint-rebuild path: the inference engine, ``injection_hse.evaluation`` and the
        multi-grid report all come through here. The config, the trained weights, the fitted
        normalizer(s), the fitted linear injection prior of a distilling run and the anchoring
        weights active when the checkpoint was written travel with the file; what supplies the
        INPUT layout depends on the run:

        - ONE grid — the graph / row layout / per-unit base come from ``grid``, and (injection
          decode) the network solver is rebuilt from it, so no dataset is needed. Pass a
          ``source`` (a :class:`~injection_hse.data.DataSource` on the training dataset) to take the
          graph, orders and injection spec from the dataset instead — required when the run
          used the per-device localization head.
        - MANY grids (``data.multigrid``) — one grid cannot rebuild a many-grid input: every
          batch carries its own concatenated graph. ``source`` is then the
          :class:`~injection_hse.data.MultiGridDataSource` over the corpus and the rebuild is
          :func:`~injection_hse.train.load_multigrid_checkpoint`, which
          :func:`injection_hse.analysis.multigrid.load_multigrid_module` reaches through a run directory.

        The grid is validated against the stored ``grid_fingerprint``: a different row
        layout (relabeled or reordered nodes, a changed slack or base frequency) raises, and
        a checkpoint without a fingerprint loads with a warning. The solver is rebuilt
        whenever :attr:`~injection_hse.config.ExperimentConfig.needs_solver` holds, built for the
        larger of the recorded anchor weights and the ``linear_state`` prior's own, and the
        recorded ``anchor_weights`` (the final stage's resolved schedule value; the global
        :class:`~injection_hse.config.PhysicsConfig` weights when absent) are installed as the runtime
        scalars, so evaluation and inference reproduce the trained decode.
        """
        from pgml.assembly import node_phase_index

        from ..config import ExperimentConfig
        from ..data import base_voltage_per_row, build_graph

        dev = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        cfg = ExperimentConfig.model_validate(ckpt["pgl_config"])
        if cfg.data.multigrid is not None:
            if source is None:
                raise ValueError(
                    "this checkpoint was trained across MANY grids (data.multigrid), so one "
                    "grid cannot rebuild its input: every batch carries its own concatenated "
                    "graph. Pass the MultiGridDataSource it trained on "
                    "(source=build_data_source(cfg.data)), or rebuild the whole run from its "
                    "directory with injection_hse.analysis.multigrid.load_multigrid_module."
                )
            return load_multigrid_checkpoint(ckpt, source, device=device)
        if grid is None:
            raise ValueError(
                "from_checkpoint needs the grid the weights were trained on (only a "
                "data.multigrid checkpoint may omit it — that one takes the corpus' "
                "MultiGridDataSource as its source instead)."
            )
        # refuse a grid whose row layout differs from the one the weights were trained on
        # (an old checkpoint without the fingerprint reloads with a warning, not a crash).
        _check_grid_fingerprint(ckpt, grid)
        # keep the loss/metrics aligned with the dataset's actually-solved orders
        if source is not None and list(source.orders) != list(cfg.orders):
            cfg = cfg.model_copy(update={"orders": list(source.orders)})

        # restore the fitted normalizer BEFORE constructing the module: the loss
        # resolves its per-harmonic weights from the fitted Cartesian scale at
        # construction time (the default "physical" weight mode).
        normalizer = HarmonicNormalizer()
        normalizer.load_state_dict(ckpt["normalizer"])

        if source is not None:
            n_nodes = source.n_nodes
            graph = source.graph()
            injection_spec = source.injection_spec
        else:
            n_nodes = int(node_phase_index(grid).size)
            graph = build_graph(
                grid, orders=list(cfg.orders), coupling=cfg.data.phase_coupling
            )
            injection_spec = None

        # Deploy the anchoring weights ACTIVE when the checkpoint was written (the final
        # stage's schedule value) so eval/inference reproduce the trained decode exactly; a
        # legacy checkpoint without them falls back to the global PhysicsConfig weights. The
        # solver is BUILT with the larger of those and the weights the linear-state input
        # prior's own solve needs (its anchoring tensors are materialised at construction),
        # and the deployed weights are installed as the runtime scalars afterwards.
        anchor = [
            float(w)
            for w in ckpt.get(
                "anchor_weights",
                [cfg.physics.anchor_voltage_weight, cfg.physics.anchor_current_weight],
            )
        ]
        prior_v, prior_i = anchor_build_weights(
            cfg.physics, [], measure_current=cfg.measurement.measure_current
        )
        build_v, build_i = max(anchor[0], prior_v), max(anchor[1], prior_i)
        _, complex_dtype = training_dtypes(cfg.train.dtype)

        # measured branch currents: the edge operator is rebuilt from the grid (like the
        # network solver); the fitted edge/power normalizers travel with the checkpoint. Built
        # BEFORE the solver so a current-anchored decode can materialise its V->current map.
        edge_op = None
        edge_normalizer = None
        power_normalizer = None
        topology = None
        if cfg.measurement.measure_current:
            edge_normalizer = HarmonicNormalizer()
            edge_normalizer.load_state_dict(ckpt["edge_normalizer"])
            if cfg.measurement.measure_power:
                power_normalizer = HarmonicNormalizer()
                power_normalizer.load_state_dict(ckpt["power_normalizer"])

        solver = None
        inj_normalizer = None
        if cfg.needs_solver:
            # the injection space travels with every checkpoint that built the solver (the
            # decode's head, the ``kcl`` input prior, the noise-calibrated anchor weights)
            inj_normalizer = HarmonicNormalizer()
            inj_normalizer.load_state_dict(ckpt["inj_normalizer"])

        if cfg.data.topology is not None:
            # switchable topology: rebuild EVERY variant's operators (the config records the
            # switch set and the variants, so the deployed decode matches the trained one)
            from ..data.topology import TopologyBank, TopologySet

            topology = TopologyBank.build(
                TopologySet.from_config(cfg.data.topology),
                grid,
                list(cfg.orders),
                edge_branch_ids=graph.edge_branch_ids,
                dtype=complex_dtype,
                measure_current=cfg.measurement.measure_current,
                decode=cfg.physics.decode,
                build_solvers=cfg.needs_solver,
                anchor_v_weight=build_v,
                anchor_i_weight=build_i,
                load_shunt=cfg.physics.load_shunt,
            )
        else:
            from ..physics import NetworkSolver

            from ..data.currents import BranchCurrentOperator

            if cfg.measurement.measure_current:
                edge_op = BranchCurrentOperator(
                    grid, list(cfg.orders), dtype=complex_dtype
                )
            if cfg.needs_solver:
                solver = NetworkSolver.from_grid(
                    grid,
                    list(cfg.orders),
                    anchor_v_weight=build_v,
                    anchor_i_weight=build_i,
                    edge_op=edge_op,
                    harmonic_slack=cfg.physics.harmonic_slack,
                    load_shunt=cfg.physics.load_shunt,
                )

        module = cls(
            cfg,
            n_nodes=n_nodes,
            normalizer=normalizer,
            graph=graph,
            base_v=base_voltage_per_row(grid),
            injection_spec=injection_spec,
            solver=solver,
            inj_normalizer=inj_normalizer,
            grid=grid,
            edge_op=edge_op,
            edge_normalizer=edge_normalizer,
            power_normalizer=power_normalizer,
            topology=topology,
        )
        module.load_state_dict(ckpt["state_dict"])
        # the fitted distillation teacher, when the run trained with one: a reload then
        # continues against the same objective rather than a refitted approximation of it
        if "injection_prior" in ckpt:
            module.set_injection_prior(ckpt["injection_prior"])
        # the deployed runtime weights (the build capability above may exceed them)
        module.set_stage_anchor(v_weight=anchor[0], i_weight=anchor[1])
        module.to(dev)
        module.eval()
        return module

    # -- switchable topology --------------------------------------------------
    #: variant assumed for a batch that carries no ``topology`` tag — an attribution /
    #: memory probe, or a single-topology evaluation of a topology-trained model.
    _DEFAULT_TOPOLOGY_INDEX = 0

    @property
    def _solvers(self) -> tuple:
        """Every network solver this module decodes with (one per switch configuration)."""
        topology = getattr(self, "_topology", None)
        if topology is not None and topology.solvers:
            return topology.solvers
        return () if self.solver is None else (self.solver,)

    def _topology_index(self, batch, lead, device) -> Optional[Tensor]:
        """The per-sample variant index ``[B]`` (``None`` without a switchable topology)."""
        if self._topology is None:
            return None
        topo = batch.get("topology")
        if topo is None:
            return torch.full(
                (int(lead[0]),),
                self._DEFAULT_TOPOLOGY_INDEX,
                dtype=torch.int64,
                device=device,
            )
        return topo.reshape(-1).to(device=device, dtype=torch.int64)

    def topology_channels(self, n_samples: int, device):
        """The topology model channels for a PROBE batch: ``(node_topology, edge_topology)``.

        ``(None, None)`` without a switchable topology. Analysis probes assemble the model
        input themselves (they need gradient leaves), so they read the channels here instead
        of reconstructing them — on the default configuration, since a probe carries no
        per-sample tag.
        """
        if self._topology is None:
            return None, None
        lead = (int(n_samples),)
        topo = self._topology_index({}, lead, device)
        return (
            self._build_node_topology(topo, lead, device),
            self._build_edge_topology(topo, lead, device),
        )

    def _build_node_topology(self, topo: Optional[Tensor], lead, device):
        """The per-sample switch-state vector ``[*lead, S]`` broadcast to the model, or ``None``.

        ``1`` = closed, ``0`` = open, in the topology's switch order. Node-agnostic (the model
        broadcasts it to every node token, like the time features); constant over the step
        axis of a sequence, since a scenario keeps its switch configuration.
        """
        if not self._n_topology_features or topo is None:
            return None
        closed = self._topology.topology.closed.to(
            device=device, dtype=self._real_dtype
        )
        states = closed.index_select(0, topo)  # [B, S]
        if len(lead) == 2:
            states = states.unsqueeze(1).expand(lead[0], lead[1], states.shape[-1])
        return states

    def _build_edge_topology(self, topo: Optional[Tensor], lead, device):
        """The per-sample half-edge conduction channel ``[*lead, E, 1]``, or ``None``.

        ``1`` = the half-edge's branch conducts in this sample's configuration, ``0`` = its
        switch is open. Unlike the measured edge features this is NOT gated by the
        observation mask — the network structure is known everywhere.
        """
        if not self._n_edge_topology_features or topo is None:
            return None
        state = self._topology.edge_state.to(device=device, dtype=self._real_dtype)
        per_sample = state.index_select(0, topo)  # [B, E]
        if len(lead) == 2:
            per_sample = per_sample.unsqueeze(1).expand(
                lead[0], lead[1], per_sample.shape[-1]
            )
        return per_sample.unsqueeze(-1)

    # -- masker control (StageManager) ---------------------------------------
    def set_masker(self, masker) -> None:
        self._masker = masker

    def set_stage_optim(self, *, lr=None, scheduler=None, warmup_epochs=0) -> None:
        """Install per-stage optimizer overrides for the NEXT ``fit`` (StageManager).

        ``None`` values inherit :class:`~injection_hse.config.OptimConfig`; a fresh optimizer (and
        scheduler) is built at every stage's fit, so the override applies exactly to that
        stage — e.g. a lower, cosine-annealed learning rate on the placement finetune.
        ``warmup_epochs > 0`` prepends a linear warmup of that many epochs
        (:attr:`injection_hse.config.MaskStageConfig.warmup_epochs`).
        """
        self._stage_optim = {
            k: v for k, v in {"lr": lr, "scheduler": scheduler}.items() if v is not None
        }
        if warmup_epochs:
            self._stage_optim["warmup_epochs"] = int(warmup_epochs)

    def set_injection_prior(self, prior) -> None:
        """Install the fitted linear injection prior the distillation term supervises against.

        ``prior`` is a :class:`~injection_hse.analysis.LinearInjectionPrior` (or its ``state_dict``) —
        the closed-form least-squares map from this run's own measurements onto the nodal
        injections, fitted on the TRAIN split at the deployed placement. Fitted before a pass
        trains with it (:func:`injection_hse.train.attach_injection_prior`; a later pass of a multi-pass
        run installs its own refit) and restored on a reload, so the prior a checkpoint
        records is the one that trained it. Refused when the
        prior's grid / harmonic set / sensor layout is not this estimator's.
        """
        from .distillation import InjectionPriorTeacher

        if prior is None:
            self.injection_prior = None
            return
        if isinstance(prior, dict):
            from injection_hse.analysis.baselines import LinearInjectionPrior

            prior = LinearInjectionPrior.from_state_dict(prior)
        teacher = InjectionPriorTeacher(prior)
        teacher.check_compatible(self)
        self.injection_prior = teacher

    def set_loss_preconditioner(self, preconditioner) -> None:
        """Install (or remove, with ``None``) the loss preconditioner of the injection head.

        A :class:`~injection_hse.train.LossPreconditioner` built by
        :func:`injection_hse.train.build_loss_preconditioner` from THIS module's decode; refused
        when its row set is not this estimator's.
        """
        if preconditioner is None:
            self.loss_preconditioner = None
            return
        if int(preconditioner.matrices.shape[-1]) != 2 * int(
            preconditioner.rows.numel()
        ):
            raise ValueError(
                "the loss preconditioner's matrices do not match its rows."
            )
        if int(preconditioner.rows.max()) >= int(self.n_nodes):
            raise ValueError(
                "the loss preconditioner indexes rows this estimator does not have."
            )
        self.loss_preconditioner = preconditioner

    def set_stage_distillation(self, weight: Optional[float]) -> None:
        """Install this stage's prior-distillation weight (``None`` inherits the config's).

        The anneal seam of :attr:`~injection_hse.config.MaskStageConfig.prior_distillation_weight`: the
        prior is a stepping stone, not the target, so a curriculum can carry a large weight
        while the injection map is being built and drop it to ``0`` for the stage whose
        accuracy the run is judged on.
        """
        self._stage_distillation = None if weight is None else float(weight)

    def set_stage_anchor(
        self,
        *,
        v_weight: float,
        i_weight: float,
        v_range=None,
        i_range=None,
    ) -> None:
        """Install this stage's measurement-anchoring weights on the network solver.

        No-op unless the injection decode is active (``self.solver`` present). The
        grid-constant anchoring tensors (``_v_scale`` / ``_c_map``) are built ONCE at
        construction (sized for the largest weight any stage requests); this only flips the
        runtime scalars, so a stage can dial the anchor from off (``0``) to measurement-dominant
        without re-assembling anything. Raises if a stage asks for current anchoring on a solver
        built without the branch-current map (no ``edge_op`` / ``measure_current``).

        ``v_range``/``i_range`` (optional ``(low, high)``) enable per-BATCH weight
        randomization for this stage: every TRAINING batch draws one uniform sample
        and interpolates both ranges with it (the voltage/current trust ratio is
        preserved), so the model cannot co-adapt to a single solve strength.
        Validation/eval batches use the fixed ``v_weight``/``i_weight``.

        A MULTI-GRID run installs the same schedule on its operator BANK
        (:meth:`injection_hse.physics.MultiGridPhysicsBank.set_anchor_weights`), so members built later
        in the stage decode at the stage's strength too.
        """
        if self.solver is None and self._bank is None:
            return
        i_hi = float(i_range[1]) if i_range is not None else float(i_weight)
        if self._bank is not None:
            if i_hi > 0.0 and self._bank.anchor_i_weight <= 0.0:
                raise ValueError(
                    "a curriculum stage sets anchor_current_weight > 0 but the operator "
                    "bank was built without the branch-current map — enable "
                    "measurement.measure_current so the decode can build it."
                )
            self._stage_anchor = (float(v_weight), float(i_weight))
            self._anchor_ranges = _anchor_ranges(v_range, i_range)
            self._bank.set_anchor_weights(
                v_weight=float(v_weight), i_weight=float(i_weight)
            )
            return
        if i_hi > 0.0 and not self.solver.can_anchor_current:
            raise ValueError(
                "a curriculum stage sets anchor_current_weight > 0 but the solver has no "
                "branch-current map — enable measurement.measure_current so the decode can "
                "build it."
            )
        self._stage_anchor = (float(v_weight), float(i_weight))
        self._anchor_ranges = _anchor_ranges(v_range, i_range)
        # every switch configuration decodes with the same anchoring strength; the solver's
        # weight properties refresh its own anchoring state (injection_hse.physics.NetworkSolver).
        for solver in self._solvers:
            solver.anchor_v_weight = float(v_weight)
            solver.anchor_i_weight = float(i_weight)

    def _apply_batch_anchor(self, train: bool) -> None:
        """Set the solver's anchor weights for THIS batch (the randomization hook).

        Without configured ranges this is a no-op (the fixed stage weights stay
        installed). With ranges, a training batch draws one uniform sample shared
        by both ranges; an eval batch restores the fixed stage weights — the same
        solver scalars a training batch may have just perturbed. A multi-grid run
        dials its whole operator bank instead of one solver.
        """
        if (self.solver is None and self._bank is None) or self._anchor_ranges is None:
            return
        if train:
            u = float(torch.rand((), generator=self._anchor_gen))
            v_range, i_range = self._anchor_ranges
            # the draw law is resolved from PhysicsConfig once, at construction (this runs
            # per batch)
            draw = _anchor_weight_from(self._anchor_draw)
            wv = draw(v_range, u) if v_range is not None else self._stage_anchor[0]
            wi = draw(i_range, u) if i_range is not None else self._stage_anchor[1]
        else:
            wv, wi = self._stage_anchor
        if self._bank is not None:
            self._bank.set_anchor_weights(v_weight=wv, i_weight=wi)
            return
        for solver in self._solvers:
            solver.anchor_v_weight = wv
            solver.anchor_i_weight = wi

    def release_operator_caches(self) -> None:
        """Free the factor-once operator caches of every solver this module decodes with.

        The caches (:attr:`injection_hse.config.PhysicsConfig.operator_cache_mb`) pay off over the
        tens of thousands of solves of a TRAINING run; a post-training pass that must
        assemble operators for members never trained on — the test split of the final
        evaluation — needs the accelerator memory they hold more than the speedup they
        give a single scoring sweep. Caching stays OFF afterwards (every further solve
        takes the direct per-call path, on the members built later too). The caller
        empties the accelerator's allocator afterwards if the freed bytes must be
        visible to non-cache allocations (``torch.cuda.empty_cache()``).
        """
        if self._bank is not None:
            self._bank.enable_operator_cache(0)
        for solver in self._solvers:
            solver.enable_operator_cache(None)

    # -- the feature frame (normalization.per_unit) --------------------------
    def _to_feature_frame(self, v: Tensor) -> Tensor:
        """Complex state in VOLTS -> the frame the encoder and the normalizer live in.

        THE single-grid seam into the feature frame: under
        :attr:`~injection_hse.config.NormalizationConfig.per_unit` the state is divided by the per-row
        line-to-neutral base voltage (:attr:`base_v`), so the per-order statistics whiten the
        variation BETWEEN scenarios instead of the spread of a feeder's voltage classes. A
        multi-grid batch carries its own per-row base and converts inside
        :meth:`~injection_hse.train.multigrid.MultiGridMixin._multigrid_step`, so it passes through
        here; without ``per_unit`` this is the identity (byte-identical).
        """
        if self._multigrid or not self._per_unit:
            return v
        return per_unit_state(v, self.base_v)

    def _from_feature_frame(self, v_pu: Tensor) -> Tensor:
        """Inverse of :meth:`_to_feature_frame` — back to VOLTS.

        Every estimate leaves the module in volts, so the loss target, the metrics and
        :meth:`infer` keep the meaning they have without the frame.
        """
        if self._multigrid or not self._per_unit:
            return v_pu
        return physical_state(v_pu, self.base_v)

    def _injection_base(self) -> Optional[Tensor]:
        """float64 ``[H, N]`` per-unit CURRENT base of this module, or ``None``.

        ``i_base[h, n] = s(h)·V_ref² / V_base[n]`` (:meth:`injection_hse.physics.NetworkSolver.i_base`,
        a power base per row voltage) — the frame the model's INJECTION features live in under
        :attr:`~injection_hse.config.NormalizationConfig.per_unit`: the injection normalizer is fit on
        ``I / i_base``, a decoded injection is multiplied back before the solve, and the
        ``kcl`` input prior and the branch-current channels divide by it. ``None`` without
        the per-unit frame (the currents are amperes throughout, byte-identical) and on the
        multi-grid path (the bank serves a base per member).

        With a switchable topology the representative solver's operator scale is used for
        every variant: the base is a unit convention shared with the normalizer fit, not a
        per-variant physical quantity, and every variant shares the union row layout.
        """
        if self._multigrid or not self._per_unit or self.solver is None:
            return None
        # the solver is a plain object Lightning does not relocate, and its operator scale is
        # cached on its own device: follow the module's before building the base, so the
        # features that divide by it never pay a cross-device copy per step.
        self._sync_solver_device(self.base_v.device)
        return self.solver.i_base(self.base_v)

    def _to_injection_frame(self, i: Tensor) -> Tensor:
        """Nodal injection currents in AMPERES -> the injection feature frame."""
        base = self._injection_base()
        if base is None:
            return i
        return i / base.to(device=i.device, dtype=i.real.dtype)

    def _from_injection_frame(self, i_pu: Tensor) -> Tensor:
        """Inverse of :meth:`_to_injection_frame` — back to AMPERES (what the solve takes)."""
        base = self._injection_base()
        if base is None:
            return i_pu
        return i_pu * base.to(device=i_pu.device, dtype=i_pu.real.dtype)

    def _edge_current_base(self) -> Optional[Tensor]:
        """float64 ``[H, E]`` per-unit base of the half-edge currents, or ``None``.

        The injection base gathered at each directed half-edge's SOURCE row — the terminal
        the CT sits on — mirroring what the multi-grid bank does per member. ``None`` without
        the per-unit frame.
        """
        base = self._injection_base()
        if base is None:
            return None
        return base.index_select(-1, self._edge_src_rows.to(base.device))

    # -- pipeline ------------------------------------------------------------
    def _encode_normalize(self, v: Tensor) -> Tensor:
        """Complex ``[*lead, H, N]`` in VOLTS -> normalized features ``[*lead, N, H, C]``.

        Leading-dim-agnostic: ``*lead = [B]`` (snapshot) or ``[B, T]`` (sequence). The frame
        conversion happens here (:meth:`_to_feature_frame`), so no caller can encode a state
        in the wrong one.
        """
        feats = self.encoding.encode(self._to_feature_frame(v))  # [*lead, H, N, C]
        feats = _to_nhc(feats)  # [*lead, N, H, C]
        return self.normalizer.transform(feats)

    def _injection_target_features(self, v: Tensor, topo: Optional[Tensor]) -> Tensor:
        """True nodal injections of ``v [*lead, H, N]`` as normalized features ``[*lead, N, H, C]``.

        The label of the direct injection term: ``I(h) = Y(h)·V_true(h)`` (the same quantity
        the injection normalizer was fit on, :func:`injection_hse.physics.fit_injection_normalizer`),
        encoded and normalized into the space the model's raw output already lives in. With a
        switchable topology each sample is mapped through ITS configuration's admittance. Built
        entirely under ``no_grad``: a supervision target is a constant, never a graph (it would
        otherwise reach the grid parameters through a calibrated ``Y``). Under the per-unit
        feature frame the label is divided by the per-unit current base first.
        """
        with torch.no_grad():
            i_true = self._to_injection_frame(self._injection_targets(v, topo))
            feats = _to_nhc(self.inj_encoding.encode(i_true))  # [*lead, N, H, C_out]
            return self.inj_normalizer.transform(feats)

    def _zero_tied(self, i: Tensor) -> Tensor:
        """``i [*lead, H, N]`` with the switch-tied rows set to zero (identity without ties)."""
        if self._tied_rows.numel() == 0:
            return i
        return i.index_fill(-1, self._tied_rows.to(i.device), 0)

    def _pin_tied_features(self, feats: Tensor) -> Tensor:
        """Head features ``[*lead, N, H, C]`` with the switch-tied rows set to the features of
        a ZERO injection, so the decode yields exactly no current there and the injection-
        space terms see prediction and target agree at those rows."""
        if self._tied_rows.numel() == 0:
            return feats
        zero = self.inj_normalizer.transform(
            _to_nhc(
                self.inj_encoding.encode(
                    torch.zeros(
                        (1, self.n_harmonics, 1), dtype=feats.dtype, device=feats.device
                    ).to(self._complex_dtype)
                )
            )
        )  # [1, 1, H, C]
        out = feats.clone()
        rows = self._tied_rows.to(feats.device)
        out[..., rows, :, :] = zero.to(feats.dtype).expand(
            *feats.shape[:-3], int(rows.numel()), -1, -1
        )
        return out

    def _injection_features(self, i: Tensor) -> Tensor:
        """Nodal injections ``[*lead, H, N]`` (amperes) -> normalized head features
        ``[*lead, N, H, C_out]`` — the space the model's raw output lives in."""
        feats = _to_nhc(self.inj_encoding.encode(self._to_injection_frame(i)))
        return self.inj_normalizer.transform(feats)

    def _prior_residual_injection(self, v_in, i_edges, mask, edge_mask) -> Tensor:
        """The linear prior's injections the residual decode adds ``[*lead, H, N]``.

        Read from the SAME measured inputs the network sees, restricted to the placement
        rows (and metered edges) THIS batch observes — unobserved placement rows enter at
        their fitted mean — so the residual never reads more than the estimator does.
        """
        self._require_injection_prior()
        with torch.no_grad():
            return self.injection_prior.injections(
                self,
                v_in,
                i_edges,
                sample_mask=mask,
                sample_edge_mask=edge_mask,
            ).to(self._complex_dtype)

    def _prior_distillation_target_features(self, v_meas: Tensor, i_edges) -> Tensor:
        """The LINEAR PRIOR's injections as normalized features ``[*lead, N, H, C]``.

        The label of the distillation term: the fitted prior's estimate of ``I(h)`` from the
        SAME measurements the estimator reads (:meth:`injection_hse.train.InjectionPriorTeacher
        .injections`), encoded and normalized into the space the model's raw output already
        lives in — so the term is the direct injection term with the label changed from the
        true ``Y(h)·V_true(h)`` to the best linear estimate of it. Built entirely under
        ``no_grad``: a supervision target is a constant, never a graph.
        """
        with torch.no_grad():
            i_prior = self.injection_prior.injections(self, v_meas, i_edges)
            return self._injection_features(i_prior)  # [*lead, N, H, C_out]

    def _injection_targets(self, v: Tensor, topo: Optional[Tensor]) -> Tensor:
        """True nodal injections ``I(h) = Y(h)·V_true(h)`` ``[*lead, H, N]`` (no_grad)."""
        solvers = self._solvers
        with torch.no_grad():
            if topo is None or len(solvers) <= 1:
                return self._zero_tied(solvers[0].injection_targets(v))
            from ..data.topology import merge_groups, topology_groups

            groups = topology_groups(topo, len(solvers))
            parts = [
                solvers[k].injection_targets(v.index_select(0, sel))
                for k, sel in groups
            ]
            return self._zero_tied(merge_groups(parts, groups))

    def _stash_loss_maps(
        self,
        pred_feats_v: Tensor,
        target_feats: Tensor,
        pred_feats,
        target_inj,
        split: float,
        scale,
    ) -> None:
        """Store each supervised term as a per-``(row, order)`` map ``[B, N, H]`` (eval seam).

        The same weighted residual the loss reduces (:meth:`~injection_hse.loss.StateEstimationLoss
        .per_node_order`), kept per row and per order so a diagnostic can see WHERE the loss
        sits. ``total`` is the blend actually optimized, so the three maps answer "is this
        row's loss the voltage term or the injection term" directly.

        The injection map is stored SCALED — multiplied by the same factor the blend applies
        (:meth:`_injection_loss_scale`) — because that is what makes a voltage cell and an
        injection cell comparable at all. The raw terms differ by orders of magnitude, so an
        unscaled map would read as "the injection loss is negligible everywhere" for a reason
        that has nothing to do with the row. The plain-decode prior term
        (``loss.plain_decode_weight``) is a loss on ANOTHER decode of the same injections and
        the distillation term (``loss.prior_distillation_weight``) is the same injection loss
        against ANOTHER label; neither is mapped here — ``total`` is the voltage / injection
        blend.
        """
        with torch.no_grad():
            voltage = self.loss_fn.per_node_order(
                pred_feats_v.reshape(-1, *pred_feats_v.shape[-3:]),
                target_feats.reshape(-1, *target_feats.shape[-3:]),
            )
            maps = {"voltage": voltage}
            if pred_feats is None or target_inj is None:
                maps["total"] = voltage
            else:
                injection = (
                    self.inj_loss_fn.per_node_order(
                        pred_feats.reshape(-1, *pred_feats.shape[-3:]),
                        target_inj.reshape(-1, *target_inj.shape[-3:]),
                    )
                    * scale
                )
                maps["injection"] = injection
                maps["total"] = (1.0 - split) * voltage + split * injection
            self._loss_maps = {k: value.detach() for k, value in maps.items()}

    def _injection_loss_scale(self, voltage: Tensor, injection: Tensor) -> Tensor:
        """Factor bringing the injection term onto the voltage term's scale, THIS batch.

        ``loss.injection_state_loss_split`` is meant to read as "the fraction of the loss
        carried by the injection term", but the two residuals are not commensurate: measured
        across a curriculum the injection term runs 82x to 45000x above the voltage term, so
        a raw blend at a nominal ``0.1`` puts 90-99.98 % of the loss on the injections — the
        knob is a switch, not a dial.

        The factor is the batch's own ``voltage / injection`` ratio, DETACHED, so the scaled
        term equals the voltage term in magnitude and the blend
        ``(1-s)·V + s·(V/I)·I`` carries exactly ``s`` of its value on the injections — every
        batch, every stage, exactly. A running average was tried first and is wrong here: the
        voltage term falls by orders of magnitude WITHIN a stage, so an average lags it and
        the realized share overshoots (measured 0.8-0.999 for a nominal 0.5). Detaching keeps
        this a reweighting, not a gradient path: the injection term still trains the model
        through ``∇I``, scaled by a constant.

        The blend also stays the size of the voltage-only loss, so the learning rate and
        gradient clipping mean what they did at ``split=0``. Returns ``1`` when normalization
        is off (``injection_state_loss_normalize=False``), reproducing a pre-normalization
        run.
        """
        if not self.cfg.loss.injection_state_loss_normalize:
            return voltage.new_ones(())
        return voltage.detach() / injection.detach().clamp_min(1e-30)

    def _cancelling_row_errors(
        self,
        i_pred: Tensor,
        v: Tensor,
        topo: Optional[Tensor],
        mask: Tensor,
    ) -> dict[str, float]:
        """Per-order injection error where a load and a generator SHARE a node, vs elsewhere.

        Those rows (:func:`_cancelling_injection_rows`) are the ones the voltages cannot
        resolve: only the SUM of the two devices reaches the network, so at the fundamental
        the estimator has no way to recover the split and the direct injection supervision is
        asking for something the measurements do not contain. Their harmonic injections do
        NOT cancel — a load and an inverter distort differently — so if the estimator learns
        to associate a spectral signature with each device, the error on these rows falls at
        the harmonic orders while staying high at the fundamental.

        Reported per order as the residual against that order's own injection scale (RMS of
        ``|I_h|`` over the same rows), for the UNOBSERVED rows only, split into the
        cancelling rows and all the others: ``inj_rel_cancel_h{h}`` / ``inj_rel_other_h{h}``.
        Empty when the grid has no such node. Diagnostic only — never on the tape.
        """
        rows = getattr(self, "_cancelling_rows", None)
        if rows is None or rows.numel() == 0:
            return {}
        with torch.no_grad():
            i_true = self._injection_targets(v, topo)  # [*lead, H, N]
            err = (i_pred.detach() - i_true).abs()
            sig = i_true.abs()
            unobs = ~mask.reshape(-1, self.n_nodes)  # [B*, N]
            err = err.reshape(-1, len(self.orders), self.n_nodes)
            sig = sig.reshape(-1, len(self.orders), self.n_nodes)
            cancel = torch.zeros(self.n_nodes, dtype=torch.bool, device=err.device)
            cancel[rows.to(err.device)] = True

            out: dict[str, float] = {}
            for group, keep in (("cancel", cancel), ("other", ~cancel)):
                sel = (unobs & keep.unsqueeze(0)).to(err.dtype)  # [B*, N]
                count = float(sel.sum())
                if count <= 0.0:
                    continue  # nothing estimated in this group (a fully observed stage)
                for hi, order in enumerate(self.orders):
                    residual = (err[:, hi, :] * sel).sum() / count
                    scale = ((sig[:, hi, :] * sel).pow(2).sum() / count).sqrt()
                    if float(scale) <= 0.0:
                        continue  # the order carries no injection on these rows
                    out[f"inj_rel_{group}_h{int(order)}"] = float(residual / scale)
            return out

    def _draw_mask(
        self, lead: tuple[int, ...], device
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Observation masks ``([*lead, N], [*lead, N, H] | None)``, drawn on CPU then moved.

        The mask generator lives on CPU so the draw is identical regardless of the training
        device; one fresh mask is drawn per leading slot (per scenario AND per step when
        ``T>1``), then reshaped to ``[*lead, N]`` and moved to the batch device. The metered
        slack rows are forced observed AFTER the draw (the substation meter never
        disappears), which leaves the masker's random stream untouched.

        The second element is the per-ORDER observability of a HETEROGENEOUS fleet
        (``measurement.heterogeneous_orders``): the active masker's
        :meth:`~injection_hse.masking.Masker.order_mask` (a random smart-meter demotion, or the
        placement's device capability), with the metered slack forced full-harmonic.
        ``None`` for a homogeneous run/stage — every consumer then falls back to the
        location mask.
        """
        rows = 1
        for s in lead:
            rows *= int(s)
        mask = self._masker.mask(
            rows, self.n_nodes, device="cpu", generator=self._mask_gen
        )  # [rows, N]
        if self._forced_obs_rows is not None and self._forced_obs_rows.numel():
            mask[:, self._forced_obs_rows] = True
        order_mask = None
        if self._hetero:
            order_mask = self._masker.order_mask(
                mask,
                self.n_harmonics,
                fundamental_pos=self._fundamental_pos,
                generator=self._mask_gen,
            )  # [rows, N, H] or None
            if order_mask is not None:
                if self._forced_obs_rows is not None and self._forced_obs_rows.numel():
                    order_mask[:, self._forced_obs_rows, :] = True
                order_mask = order_mask.reshape(
                    *lead, self.n_nodes, self.n_harmonics
                ).to(device)
        return mask.reshape(*lead, self.n_nodes).to(device), order_mask

    def _build_node_time(self, batch, lead: tuple[int, ...], device):
        """Per-step time-feature vector ``[*lead, Ftime]`` (or ``None`` when disabled/timeless).

        Time features are OFF unless ``encoding.time_features`` is set. When on, an absolute
        per-step ``time_unix_s`` in the batch (a time-stamped coherent sequence) is turned into
        the periodic feature vector: a ``[T]`` axis shared by every scenario is broadcast to
        the leading ``[B, T]`` shape, a ``[B, T]`` axis (a start per scenario) is used per
        sample. A batch WITHOUT ``time_unix_s`` (a snapshot, or a timeless coherent sequence)
        returns ``None`` — the model then supplies the all-zero vector (``valid = 0``), the
        same input contract. Off = ``None`` always.
        """
        if not self._n_time_features:
            return None
        tu = batch.get("time_unix_s")
        if tu is None:
            return None  # timeless batch: the model fills zeros (valid flag = 0)
        # ``tu`` is the shared step axis ``[T]`` or the per-scenario ``[B, T]``; ``lead`` is
        # ``[B, T]`` for a sequence batch. ``time_features`` computes in float64 (epoch-second
        # precision); cast to the training real dtype at this boundary so the model input
        # matches its parameter dtype (the model would otherwise realign it — this makes the
        # contract explicit and dtype-agnostic).
        tfeat = _time_features(tu.to(device)).to(self._real_dtype)  # [(B,) T, Ftime]
        if tu.ndim == 2:
            return tfeat  # [B, T, Ftime]
        b = int(lead[0])
        return tfeat.unsqueeze(0).expand(b, *tfeat.shape)  # [B, T, Ftime]

    def _step(self, batch, *, train: bool):
        """Shared forward: ``(loss, mask, pred_v, target_v, inj_loss, phys_loss, order_mask)``.

        Handles both the snapshot (``v [B, H, N]``) and sequence (``v [B, T, H, N]``) cases:
        the encoded features carry the leading ``[*lead]`` dims, the model receives one
        :class:`~injection_hse.io.ModelInput` (``T`` lifted/threaded), and for ``T>1`` the loss/metrics
        are per-step (the mask broadcasts over time). ``pred_v``/``target_v`` are returned in
        the metric layout ``[*lead, H, N]``. ``inj_loss`` is the per-device injection MSE when
        the localization head is enabled (added on top), else ``None``. ``phys_loss`` is
        reserved in the return contract for future use and is always ``None`` — physics
        consistency is enforced STRUCTURALLY by the injection-decode solve
        (``physics.decode="injection"``), not as a soft residual blend. ``order_mask`` is the
        per-order observability ``[*lead, N, H]`` of a heterogeneous sensor fleet (``None``
        for a homogeneous run/stage — the metrics then split on the location mask alone).

        The supervised loss is the voltage term unless ``loss.injection_state_loss_split``
        blends in the direct injection term, plus ``loss.prior_distillation_weight`` times the
        injection loss against the LINEAR PRIOR's injections, plus
        ``loss.plain_decode_weight`` times the same voltage loss on the PLAIN (slack-only)
        decode while the solve anchors; ``_loss_components`` carries the parts (detached) for
        this step's logging — always the voltage term, plus the injection term whenever the
        blend is active, plus the ``prior_distillation`` and ``plain_decode`` terms whenever
        they are added.

        A MULTI-GRID batch (``data.multigrid``) is served by :meth:`_multigrid_step`, which
        returns the same tuple over the CONCATENATED node axis.
        """
        if self._multigrid:
            return self._multigrid_step(batch, train=train)
        v = batch["v"]  # complex [B, H, N] or [B, T, H, N]
        # the state as loaded (the dataset's precision) is what the simulated CTs read:
        # a branch current is a small difference of large terminal voltages (20 kV rows
        # against a few amperes), which complex64 cannot resolve
        v_loaded = v
        # Cast the loaded state to the TRAINING complex dtype at the module boundary, so every
        # real-valued feature derived from it (encoding, normalization, the masked model input,
        # the injection-decode re-encode) matches the model's parameter dtype. Guarded: a
        # dataset already at the training dtype is passed through UNTOUCHED — no autograd op,
        # so ``complex64`` data trained at ``complex64`` is byte-identical to before.
        if v.dtype != self._complex_dtype:
            v = v.to(self._complex_dtype)
        device = v.device
        has_time = v.dim() == 4
        lead = tuple(v.shape[:-2])  # [B] or [B, T]

        # The clean state is BOTH the loss target and the per-unit metric reference. When
        # sensor noise is enabled, the model INPUT is a noisy copy of the measured phasors
        # (``v_in``); the target stays clean. With noise off, ``v_in`` is ``v`` and the input
        # features ARE the target features (byte-identical to a run without noise).
        target_feats = self._encode_normalize(v)  # [*lead, N, H, C]
        if self._noise is not None:
            v_in = self._noise.apply(v)
            input_feats = self._encode_normalize(v_in)
        else:
            v_in = v
            input_feats = target_feats
        mask, order_mask = self._draw_mask(lead, device)  # [*lead, N] (+ [*lead, N, H])

        if order_mask is not None:
            # heterogeneous fleet: zero per (row, ORDER) — a fundamental-only smart meter
            # contributes its fundamental and nothing else (order_mask ⊆ mask row-wise).
            masked = _flatten_features(
                input_feats * order_mask.unsqueeze(-1).to(input_feats.dtype)
            )
        else:
            masked = _flatten_features(input_feats).clone()  # [*lead, N, F]
            masked = masked * mask.unsqueeze(-1).to(
                masked.dtype
            )  # zero unobserved rows
        i_meas = batch.get("i_meas")
        if i_meas is not None and i_meas.dtype != self._complex_dtype:
            i_meas = i_meas.to(self._complex_dtype)
        # the switch configuration each sample was generated on: it selects the per-variant
        # branch-current operator / network solver, and is fed to the model as the topology
        # channels (unless ``topology.conditioning="none"``).
        topo = self._topology_index(batch, lead, device)
        edge_obs, edge_mask, edge_order_mask, i_edges = self._edge_observations(
            v_in,
            mask,
            order_mask,
            lead,
            device,
            i_meas=i_meas,
            topo=topo,
            v_true=v_loaded,
        )
        node_time = self._build_node_time(batch, lead, device)
        # the physics-derived input priors read the SAME measured inputs the model sees (the
        # masked noisy state, the measured branch currents) — never the clean target
        node_prior = self.input_priors(
            v_in, mask, order_mask, i_edges, edge_mask, edge_order_mask, topo
        )
        model_input = build_model_input(
            self._graph,
            masked,
            mask,
            has_time=has_time,
            node_time=node_time,
            node_topology=self._build_node_topology(topo, lead, device),
            edge_topology=self._build_edge_topology(topo, lead, device),
            edge_obs=edge_obs,
            edge_mask=edge_mask,
            node_order_mask=order_mask,
            edge_order_mask=edge_order_mask,
            node_prior=node_prior,
        )
        pred_flat = self.model(model_input)  # [*lead, N, Fout] (T axis preserved)
        pred_flat = pred_flat.reshape(*lead, self.n_nodes, -1)
        pred_feats = _unflatten_features(
            pred_flat, self.n_harmonics, self.n_out_channels
        )  # [*lead, N, H, C_out]
        # LOSS PRECONDITIONER (``loss.preconditioner``): identity in the forward pass, the
        # inverse curvature of the loss on the gradient that reaches the head's output.
        if self.loss_preconditioner is not None:
            pred_feats = self.loss_preconditioner(pred_feats)

        # Decode to the complex state ``pred_v [*lead, H, N]`` and form the supervised loss.
        # Supervise ALL rows (the surrogate reconstructs the full state); the metric split
        # (observed vs unobserved) is reported separately. The loss collapses leading dims
        # internally, so the [*lead, N, H, C] layout works for T>1.
        if self._predict_injection_state:
            # PHYSICS DECODE: the model output is the nodal INJECTION current (normalized
            # features in the INJECTION encoding). Inverse-normalize + decode to physical
            # complex I, then the differentiable network solve reconstructs the voltages from
            # I + the measured fundamental slack voltage — the state satisfies Y(h)·V=I by
            # construction.
            # the per-unit current base is read off the solver's operator scale, so the
            # operators follow the batch device before the decode reads them
            self._sync_solver_device(pred_feats.device)
            # PRIOR-RESIDUAL decode (``physics.injection_prior_residual``): the network's
            # output is the residual on the linear prior's injections, added in the head's
            # own normalized feature space — so a zero output IS the prior (in amperes the
            # residual is ``z · std``, with no mean offset) — and every consumer of the
            # predicted injection features below (the direct term, distillation, the
            # diagnostics) sees the FULL injection, prior included.
            if self._prior_residual:
                pred_feats = pred_feats + self._injection_features(
                    self._prior_residual_injection(v_in, i_edges, mask, edge_mask)
                ).to(pred_feats.dtype)
            # switch-tied rows (``physics.exclude_tied_rows``): no injection is estimable
            # there, so the features are pinned to the zero injection before the decode
            pred_feats = self._pin_tied_features(pred_feats)
            i_pred = self._zero_tied(
                self._from_injection_frame(
                    self.inj_encoding.decode(
                        self.inj_normalizer.inverse_transform(pred_feats)
                    ).transpose(-2, -1)
                )
            )  # [*lead, N, H] -> [*lead, H, N], amperes (exactly zero at the ties)
            # learned calibration: write the corrected parameters onto the grid and re-assemble
            # Y(Δ) on the live tape, so the solve (and its gradient to Δ) sees the correction.
            if self.grid_calibration is not None:
                self.grid_calibration.apply_to_grid()
                self.solver.assemble()
            # the fundamental slack boundary is a MEASURED input, so it carries the sensor
            # noise too (``v_in`` == ``v`` when noise is off — byte-identical).
            # MEASUREMENT ANCHORING (optional): pull the reconstructed state toward every
            # observed measurement, not just the slack. The observed voltage rows come from the
            # (noisy) input state ``v_in`` under the per-sample mask; the observed branch
            # currents are the same ``i_edges`` fed to the model (state-derived or real
            # ``i_meas``). Off (both weights 0) -> the slack-only solve, byte-identical.
            self._apply_batch_anchor(train)
            pred_v = self._decode_state(
                i_pred,
                v_in=v_in,
                mask=mask,
                order_mask=order_mask,
                i_edges=i_edges,
                edge_mask=edge_mask,
                edge_order_mask=edge_order_mask,
                topo=topo,
            )  # [*lead, H, N]
            # supervise the SOLVED voltages, re-encoded to the V feature space, so the loss is
            # IDENTICAL to the direct-voltage estimator's (fair comparison, same metric).
            pred_feats_v = self._encode_normalize(pred_v)
            self._loss_rows = self._supervision_rows(mask)
            self._plain_rows = self._plain_decode_rows(mask)
            supervised = self.loss_fn(
                pred_feats_v.reshape(-1, *pred_feats_v.shape[-3:]),
                target_feats.reshape(-1, *target_feats.shape[-3:]),
                mask=self._loss_rows,
            )
            # OPTIONAL second supervised term ON THE PREDICTED INJECTIONS themselves
            # (``loss.injection_state_loss_split``): the voltage term reaches the injections
            # only through the solve, which is blind to any injection error the network law
            # maps onto (near-)zero voltage change. Comparing ``i_pred`` with the true
            # Y(h)·V_true(h) in the normalized injection space supervises them directly; the
            # per-harmonic weighting + Cartesian MSE structure is the voltage term's, but the
            # two RESIDUALS are not commensurate (the injection one runs orders of magnitude
            # higher), so the blend rescales it onto the voltage term's running scale — see
            # :meth:`_injection_loss_scale`. Read per step (the module adopts each task pass's
            # config); 0 = untouched, byte-identical.
            split = float(self.cfg.loss.injection_state_loss_split)
            if self.inj_loss_fn is not None and split > 0.0:
                target_inj = self._injection_target_features(v, topo)
                injection = self.inj_loss_fn(
                    pred_feats.reshape(-1, *pred_feats.shape[-3:]),
                    target_inj.reshape(-1, *target_inj.shape[-3:]),
                    mask=None,
                )
                self._loss_components = {
                    "voltage": supervised.detach(),
                    "injection_state": injection.detach(),
                }
                scale = self._injection_loss_scale(supervised, injection)
                scaled = injection * scale
                supervised = (1.0 - split) * supervised + split * scaled
                # what the split ACTUALLY bought, as a fraction of the blended loss
                self._loss_share = (
                    split * scaled.detach() / supervised.detach().clamp_min(1e-30)
                )
                if self._collect_loss_maps:
                    self._stash_loss_maps(
                        pred_feats_v, target_feats, pred_feats, target_inj, split, scale
                    )
            else:
                self._loss_components = {"voltage": supervised.detach()}
                self._loss_share = None
                if self._collect_loss_maps:
                    self._stash_loss_maps(
                        pred_feats_v, target_feats, None, None, 0.0, 1.0
                    )
            # OPTIONAL DISTILLATION term (``loss.prior_distillation_weight``, or a per-stage
            # override): supervise the predicted injections against the closed-form LINEAR
            # PRIOR's estimate of them from the same measurements. The true injections are
            # not identifiable from a sparse sensor set and the anchored solve is flat in the
            # directions that are, so neither the voltage term nor the true-injection label
            # builds the map; the prior's label is by construction reachable from the data.
            # 0 = untouched, byte-identical (no teacher is evaluated).
            if self._distillation_weight() > 0.0:
                self._require_injection_prior()
                supervised = self._add_prior_distillation_term(
                    supervised, pred_feats, v_in, i_edges
                )
            # OPTIONAL prior term on the PLAIN decode (``loss.plain_decode_weight``): while
            # the solve is measurement-anchored the voltage term above is flat in the
            # predicted injections wherever the measurements pin the state, so the same
            # voltage loss is additionally evaluated on the slack-only solve of the same
            # injections — the network is supervised as a prior whatever the anchored output
            # does. Without anchoring the plain decode IS ``pred_v`` (nothing is solved twice).
            self._plain_pred_v = None
            if self._plain_decode_active() and self._anchoring_active():
                pred_v_plain = self._decode_state(
                    i_pred,
                    v_in=v_in,
                    mask=mask,
                    order_mask=order_mask,
                    i_edges=i_edges,
                    edge_mask=edge_mask,
                    edge_order_mask=edge_order_mask,
                    topo=topo,
                    plain=True,
                )  # [*lead, H, N]
                supervised = self._add_plain_decode_term(
                    supervised,
                    self._encode_normalize(pred_v_plain),
                    target_feats,
                    pred_v_plain,
                    train=train,
                )
            if not train:
                self._injection_diag = self._cancelling_row_errors(
                    i_pred, v, topo, mask
                )
        else:
            supervised = self.loss_fn(
                pred_feats.reshape(-1, *pred_feats.shape[-3:]),
                target_feats.reshape(-1, *target_feats.shape[-3:]),
                mask=self._supervision_rows(mask),
            )
            self._loss_components = {"voltage": supervised.detach()}
            self._loss_share = None
            pred_v = self._from_feature_frame(
                self.encoding.decode(
                    self.normalizer.inverse_transform(pred_feats)
                ).transpose(-2, -1)
            )  # [*lead, N, H] -> [*lead, H, N], volts

        loss = supervised
        # ``phys_loss`` slot kept for the logging/return contract (now always None — physics is
        # STRUCTURAL via the solve, not a soft residual term).
        phys_loss = None

        # learned-calibration prior: pull unsupported grid corrections back toward 0 (weighted
        # by each knob's confidence). The supervised term provides the data fit.
        if self.grid_calibration is not None:
            loss = (
                loss
                + self.cfg.physics.calibration_weight
                * self.grid_calibration.regularizer().to(loss.dtype)
            )

        # OPTIONAL injection / localization head: predict per-device injected magnitudes from
        # the model's node-level representation (the predicted node features) and supervise
        # against the dataset's clean per-device target (added on top of the blend).
        inj_loss = None
        if self.injection_head is not None and "injection_target" in batch:
            pred_inj = self.injection_head(
                pred_flat, self._inj_rows, self._inj_rows_mask
            )  # [*lead, D, O, 1]
            target_inj = batch["injection_target"].to(pred_inj.dtype)  # [B, D, O, 1]
            inj_loss = torch.nn.functional.mse_loss(
                pred_inj.reshape(-1, *pred_inj.shape[-3:]),
                target_inj.reshape(-1, *target_inj.shape[-3:]),
            )
            loss = loss + self.cfg.loss.injection_loss_weight * inj_loss

        return loss, mask, pred_v, v, inj_loss, phys_loss, order_mask

    def _decode_state(
        self,
        i_pred: Tensor,
        *,
        v_in: Tensor,
        mask: Tensor,
        order_mask,
        i_edges,
        edge_mask,
        edge_order_mask,
        topo: Optional[Tensor],
        plain: bool = False,
    ) -> Tensor:
        """Reconstruct the voltages from the predicted injections (the physics decode).

        With a switchable topology the batch is SPLIT by switch configuration and each group
        decoded with its own network solver — the samples of one variant still solve as one
        batched linear solve (the number of groups is the number of topologies, not of
        samples), and the results are gathered back into the batch order differentiably.
        ``plain=True`` forces the slack-only solve whatever the solver's anchoring state (the
        plain-decode prior term, :attr:`~injection_hse.config.LossConfig.plain_decode_weight`).
        """
        solvers = self._solvers
        if topo is None or len(solvers) <= 1:
            return self._solve_group(
                solvers[0],
                i_pred,
                v_in,
                mask,
                order_mask,
                i_edges,
                edge_mask,
                edge_order_mask,
                plain=plain,
            )
        from ..data.topology import merge_groups, topology_groups

        groups = topology_groups(topo, len(solvers))
        if len(groups) == 1:
            k, _ = groups[0]
            return self._solve_group(
                solvers[k],
                i_pred,
                v_in,
                mask,
                order_mask,
                i_edges,
                edge_mask,
                edge_order_mask,
                plain=plain,
            )

        def _sel(t, sel):
            return None if t is None else t.index_select(0, sel)

        parts = [
            self._solve_group(
                solvers[k],
                i_pred.index_select(0, sel),
                v_in.index_select(0, sel),
                mask.index_select(0, sel),
                _sel(order_mask, sel),
                _sel(i_edges, sel),
                _sel(edge_mask, sel),
                _sel(edge_order_mask, sel),
                plain=plain,
            )
            for k, sel in groups
        ]
        return merge_groups(parts, groups)

    def _solve_group(
        self,
        solver,
        i_pred: Tensor,
        v_in: Tensor,
        mask: Tensor,
        order_mask,
        i_edges,
        edge_mask,
        edge_order_mask,
        *,
        plain: bool = False,
    ) -> Tensor:
        """One network solve: measurement-anchored when the stage asks for it, else slack-only.

        ``plain=True`` passes no measurements, so :meth:`injection_hse.physics.NetworkSolver.solve`
        takes its slack-only branch even on an anchoring solver.
        """
        if solver.anchoring and not plain:
            return solver.solve(
                i_pred,
                self._slack_voltage(v_in),
                v_meas=v_in,
                v_mask=mask,
                i_meas=i_edges,
                edge_mask=edge_mask,
                order_mask=order_mask,
                edge_order_mask=edge_order_mask,
            )
        return solver.solve(i_pred, self._slack_voltage(v_in))

    # -- the prior-distillation term (shared by both batch layouts) ----------
    def _distillation_weight(self) -> float:
        """The prior-distillation weight in force for THIS batch.

        The stage override (:meth:`set_stage_distillation`) when a
        :class:`~injection_hse.train.StageManager` installed one, else
        :attr:`~injection_hse.config.LossConfig.prior_distillation_weight`. Read per step, so a
        two-task curriculum can change it between passes. ``0`` without the injection decode
        (there are no predicted injections to distil into).
        """
        if self.inj_loss_fn is None:
            return 0.0
        weight = (
            self._stage_distillation
            if self._stage_distillation is not None
            else self.cfg.loss.prior_distillation_weight
        )
        return float(weight)

    def _add_prior_distillation_term(
        self, supervised: Tensor, pred_feats: Tensor, v_meas: Tensor, i_edges
    ) -> Tensor:
        """``supervised + prior_distillation_weight · L_distill`` — the distillation term.

        ``L_distill`` is the direct injection loss (:attr:`inj_loss_fn`) of the model's
        predicted injection features ``pred_feats`` ``[*lead, N, H, C_out]`` against the
        LINEAR PRIOR's estimate of the same quantity from the same measurements. Recorded as
        the ``prior_distillation`` loss component (logged as
        ``{train,val}_prior_distillation_loss``).

        The injection residual is not commensurate with the voltage term — it runs orders of
        magnitude above it — so a weight of ``1`` already makes this term the dominant part of
        the objective, which is the regime it was measured in. Read the logged component
        rather than assuming the weight is a share.
        """
        target = self._prior_distillation_target_features(v_meas, i_edges)
        distilled = self.inj_loss_fn(
            pred_feats.reshape(-1, *pred_feats.shape[-3:]),
            target.reshape(-1, *target.shape[-3:]),
            mask=None,
        )
        self._loss_components["prior_distillation"] = distilled.detach()
        return supervised + self._distillation_weight() * distilled

    def _require_injection_prior(self) -> None:
        """Refuse to train a distillation objective with no teacher installed."""
        if self.injection_prior is not None:
            return
        raise ValueError(
            "loss.prior_distillation_weight > 0 but no linear injection prior is installed: "
            "the distillation label cannot be formed. Fit one on the TRAIN split before "
            "training (injection_hse.train.attach_injection_prior, which injection_hse.workflow and "
            "injection_hse.train.train_experiment call for you), or install it directly with "
            "SEModule.set_injection_prior(...)."
        )

    # -- the plain-decode prior term (shared by both batch layouts) -----------
    def _supervision_rows(self, mask: Optional[Tensor]) -> Optional[Tensor]:
        """Row mask ``[B', N]`` of the voltage loss (``loss.supervise_rows``), or ``None``.

        ``None`` (every row) unless the config asks for the unobserved rows and the batch
        has some; the leading axes of ``mask`` ``[*lead, N]`` are flattened like the loss
        inputs.
        """
        if self.cfg.loss.supervise_rows != "unobserved" or mask is None:
            return None
        rows = ~mask.reshape(-1, mask.shape[-1])
        return rows if bool(rows.any()) else None

    def _plain_decode_rows(self, mask: Optional[Tensor]) -> Optional[Tensor]:
        """Rows ``[B', N]`` of the plain-decode term: a boolean selection or row weights.

        Without ``loss.plain_decode_observed_weight`` the term uses the voltage loss's rows
        (:meth:`_supervision_rows`). With a value ``λ`` every unobserved row weighs 1 and
        every observed row ``λ``; the loss's masked mean takes the float weights.
        """
        weight = self.cfg.loss.plain_decode_observed_weight
        if weight is None or mask is None:
            return self._supervision_rows(mask)
        observed = mask.reshape(-1, mask.shape[-1])
        return torch.where(observed, float(weight), 1.0).to(torch.float32)

    def _plain_decode_active(self) -> bool:
        """Whether ``loss.plain_decode_weight`` asks for the plain-decode term at all.

        Read per step (the module adopts each task pass's config). The term additionally
        needs the solve to anchor for THIS batch — otherwise the plain decode is the decode,
        its loss is the voltage term already in the blend, and nothing is solved twice.
        """
        return (
            self._predict_injection_state
            and float(self.cfg.loss.plain_decode_weight) > 0.0
        )

    def _anchoring_active(self) -> bool:
        """Whether the single-grid decode of the CURRENT batch is measurement-anchored.

        Every switch configuration's solver carries the same runtime weights
        (:meth:`set_stage_anchor` / :meth:`_apply_batch_anchor`), so any one of them answers
        for the batch. A multi-grid batch asks its member solvers instead
        (:meth:`~injection_hse.train.multigrid.MultiGridMixin._multigrid_anchoring`).
        """
        return any(solver.anchoring for solver in self._solvers)

    def _add_plain_decode_term(
        self,
        supervised: Tensor,
        plain_feats: Tensor,
        target_feats: Tensor,
        pred_v_plain: Tensor,
        *,
        train: bool,
    ) -> Tensor:
        """``supervised + plain_decode_weight · L_plain`` — the plain-decode prior term.

        ``plain_feats`` / ``target_feats`` ``[*lead, N, H, C]`` are the plain decode's
        re-encoded, normalized voltages and the target in the SAME feature space, so
        ``L_plain`` is the voltage loss (``self.loss_fn``) of the slack-only solve — on the
        tape, since it is what supervises the network where the anchored term is flat. The
        term is recorded as the ``plain_decode`` loss component (logged as
        ``{train,val}_plain_decode_loss``) and, for a validation step, the plain estimate
        ``pred_v_plain`` ``[*lead, H, N]`` is kept detached for the
        ``val_pu_mae_unobs_plain`` metrics.
        """
        plain = self.loss_fn(
            plain_feats.reshape(-1, *plain_feats.shape[-3:]),
            target_feats.reshape(-1, *target_feats.shape[-3:]),
            mask=getattr(self, "_plain_rows", None),
        )
        self._loss_components["plain_decode"] = plain.detach()
        if not train:
            self._plain_pred_v = pred_v_plain.detach()
        weight = float(self.cfg.loss.plain_decode_weight)
        return supervised + weight * plain

    def _check_feature_frame(self) -> None:
        """Refuse a per-unit module that cannot express one of its quantities in that frame.

        Called once at construction. The per-unit VOLTAGE frame needs the per-row base
        (:attr:`base_v`); the per-unit CURRENT frame of metered branch currents needs the
        network solver's operator scale, which is why
        :attr:`injection_hse.config.ExperimentConfig.needs_solver` builds the solver for such a run
        even under a direct-voltage decode. Missing either would otherwise surface as a
        silently mis-scaled feature rather than a failure.
        """
        if self._multigrid or not self._per_unit:
            return
        if self.base_v is None:
            raise ValueError(
                "normalization.per_unit=True needs the grid's per-row base voltage "
                "(base_v=injection_hse.data.base_voltage_per_row(grid)): the feature frame divides "
                "the state by it."
            )
        if self._measure_current and self.solver is None:
            raise ValueError(
                "normalization.per_unit=True with measurement.measure_current=True needs "
                "the network solver: the branch-current features are expressed over the "
                "per-unit current base s(h)·V_ref²/V_base, whose operator scale comes from "
                "the "
                "assembled network (see ExperimentConfig.needs_solver)."
            )

    # -- physics-derived input priors ------------------------------------------
    def _check_input_priors(self, solver, bank) -> None:
        """Refuse a prior the module cannot serve (no solver, no normalizer, no slack, no CTs).

        Called once at construction, after the solver / bank and the normalizers are known.
        The priors are inputs the model is BUILT for (they widen its token), so a missing
        operator would otherwise surface as a shape error in the first step.
        """
        if not self._input_priors:
            return
        physics = self.cfg.physics
        named = ", ".join(repr(k) for k in self._input_priors)
        if self._multigrid:
            if bank is None:
                raise ValueError(
                    f"physics.input_priors [{named}] on a data.multigrid run needs the "
                    "per-member operator bank (bank=MultiGridPhysicsBank(source, ...))."
                )
        elif solver is None:
            raise ValueError(
                f"physics.input_priors [{named}] needs the network solver "
                "(solver=NetworkSolver.from_grid(grid, orders, ...)) — the priors are "
                "linear functions of the measured state through Y(h), whatever the decode."
            )
        if "kcl" in self._input_priors and self.inj_normalizer is None:
            raise ValueError(
                "physics.input_priors 'kcl' needs the injection normalizer (fit on the true "
                "nodal injections Y(h)·V_true, injection_hse.physics.fit_injection_normalizer): the "
                "implied injections are encoded and normalized in the injection space."
            )
        if "linear_state" not in self._input_priors:
            return
        if not self.cfg.measurement.slack_known:
            raise ValueError(
                "physics.input_priors 'linear_state' requires measurement.slack_known: the "
                "zero-injection prior solve is reduced against the measured slack voltage."
            )
        if (
            not self._multigrid
            and solver.slack_rows.numel() == 0
            and self._fundamental_pos is not None
        ):
            raise ValueError(
                "physics.input_priors 'linear_state' found no slack rows: the grid has no "
                "in-service Source, so there is no slack measurement to bound the prior "
                "solve."
            )
        wants_currents = (
            self._measure_current and float(physics.prior_anchor_current_weight) > 0.0
        )
        if wants_currents and self._multigrid and bank.anchor_i_weight <= 0.0:
            raise ValueError(
                "physics.input_priors 'linear_state' anchors the prior solve to the "
                "measured branch currents (prior_anchor_current_weight > 0 with "
                "measurement.measure_current), but the operator bank was built without "
                "the branch-current map — build it with anchor_i_weight > 0 "
                "(injection_hse.train.anchor_build_weights(..., measure_current=True))."
            )
        if wants_currents and not self._multigrid:
            for member in self._solvers:
                if not member.can_anchor_current:
                    raise ValueError(
                        "physics.input_priors 'linear_state' anchors the prior solve to "
                        "the measured branch currents (prior_anchor_current_weight > 0 "
                        "with measurement.measure_current), but the network solver was "
                        "built without the BranchCurrentOperator (edge_op=...)."
                    )

    def input_priors(
        self,
        v_in: Tensor,
        mask: Tensor,
        order_mask: Optional[Tensor],
        i_edges: Optional[Tensor],
        edge_mask: Optional[Tensor],
        edge_order_mask: Optional[Tensor],
        topo: Optional[Tensor],
    ) -> Optional[Tensor]:
        """The ``node_prior`` block ``[*lead, N, Fp]`` of a single-grid batch, or ``None``.

        THE one place the physics-derived input priors (``physics.input_priors``) are
        assembled for a single-grid module: the training step, :meth:`infer` and every
        analysis probe that builds a :class:`~injection_hse.io.ModelInput` by hand call this, so the
        prior can never be missing at inference. ``None`` when the config lists no prior
        (byte-identical input); otherwise the canonical block order ``kcl`` (``H × C_inj``
        normalized injection channels + the observed-neighbourhood fraction) then
        ``linear_state`` (``H × C`` normalized state channels), in the training real dtype on
        ``v_in``'s device. Computed under ``no_grad`` — the priors are inputs, nothing in
        them is learned — from the same MEASURED inputs the model sees:

        - ``v_in`` complex ``[*lead, H, N]`` — the (noisy, when noise is on) input state;
        - ``mask`` bool ``[*lead, N]`` — the location mask;
        - ``order_mask`` bool ``[*lead, N, H]`` | None — a heterogeneous fleet's per-order
          observability (the ``kcl`` sum and the prior anchors are then per order);
        - ``i_edges`` complex ``[*lead, H, E]`` | None, ``edge_mask`` bool ``[*lead, E]`` |
          None, ``edge_order_mask`` bool ``[*lead, E, H]`` | None — the measured branch
          currents the ``linear_state`` solve anchors to when ``measurement.measure_current``;
        - ``topo`` int64 ``[B]`` | None — each sample's switch configuration under a
          switchable topology: the batch is split by variant and every group uses ITS
          configuration's operators (the dispatch the physics decode uses), merged back in
          sample order.

        The solver-level primitives are :func:`injection_hse.train.priors.kcl_prior` and
        :func:`injection_hse.train.priors.linear_state_prior`; a multi-grid batch is served per member
        by :meth:`~injection_hse.train.multigrid.MultiGridMixin._multigrid_input_priors`.
        """
        if not self._input_priors:
            return None
        with torch.no_grad():
            self._sync_solver_device(v_in.device)
            solvers = self._solvers
            if topo is None or len(solvers) <= 1:
                return self._prior_block(
                    solvers[0],
                    v_in,
                    mask,
                    order_mask,
                    i_edges,
                    edge_mask,
                    edge_order_mask,
                )
            from ..data.topology import merge_groups, topology_groups

            groups = topology_groups(topo, len(solvers))

            def _sel(t, sel):
                return None if t is None else t.index_select(0, sel)

            parts = [
                self._prior_block(
                    solvers[k],
                    v_in.index_select(0, sel),
                    mask.index_select(0, sel),
                    _sel(order_mask, sel),
                    _sel(i_edges, sel),
                    _sel(edge_mask, sel),
                    _sel(edge_order_mask, sel),
                )
                for k, sel in groups
            ]
            return merge_groups(parts, groups)

    def _prior_block(
        self,
        solver,
        v_in: Tensor,
        mask: Tensor,
        order_mask: Optional[Tensor],
        i_edges: Optional[Tensor],
        edge_mask: Optional[Tensor],
        edge_order_mask: Optional[Tensor],
    ) -> Tensor:
        """The prior block ``[*lead, N, Fp]`` through ONE solver (one switch configuration).

        ``kcl``: ``κ = Y·(mask·V_in)`` encoded with the INJECTION encoding and normalized by
        the injection normalizer (the space the injection decode's head lives in), plus the
        observed-neighbourhood fraction. ``linear_state``: the anchored zero-injection solve
        at the measured slack, encoded and normalized like the voltage INPUT. Both priors are
        computed in absolute units and converted at their frame's seam, so each lands in the
        space its normalizer was fit in.
        """
        blocks: list[Tensor] = []
        if "kcl" in self._input_priors:
            kappa, fraction = kcl_prior(solver, v_in, mask, order_mask)
            feats = self.inj_normalizer.transform(
                _to_nhc(self.inj_encoding.encode(self._to_injection_frame(kappa)))
            )  # [*lead, N, H, C_inj]
            blocks.append(_flatten_features(feats))
            blocks.append(fraction.unsqueeze(-1).to(feats.dtype))  # [*lead, N, 1]
        if "linear_state" in self._input_priors:
            wv, wi = prior_anchor_weights(
                self.cfg.physics,
                solver,
                currents=self._measure_current and i_edges is not None,
            )
            v_prior = linear_state_prior(
                solver,
                v_in,
                self._slack_voltage(v_in),
                mask,
                order_mask,
                i_edges,
                edge_mask,
                edge_order_mask,
                v_weight=wv,
                i_weight=wi,
            )  # [*lead, H, N]
            blocks.append(_flatten_features(self._encode_normalize(v_prior)))
        return torch.cat(blocks, dim=-1).to(self._real_dtype)

    def _edge_observations(
        self,
        v_in: Tensor,
        mask: Tensor,
        order_mask,
        lead,
        device,
        *,
        i_meas=None,
        topo: Optional[Tensor] = None,
        v_true: Optional[Tensor] = None,
    ):
        """Measured branch-current edge inputs ``(edge_obs, edge_mask, edge_order_mask, i_edges)``.

        Returns ``(None, None, None, None)`` unless ``measurement.measure_current``. ``i_edges``
        is the raw complex half-edge currents ``[*lead, H, E]`` (before encoding/normalization),
        exposed so the injection decode's measurement anchoring can constrain the solve with the
        same currents the model sees. The half-edge source-terminal currents come from
        ``i_meas`` (``[*lead, H, E]`` —
        REAL measured currents on the model's directed half-edges, the deployment path)
        when the batch carries them; otherwise they are derived from the TRUE state ``v_true``
        (``v_in`` when not given) by the edge operator — exactly the values a physical CT
        at the observed terminal would read, since the operator is the Y-bus-exact
        state→current map — and then carry the CT's own error only. A CT measures the
        branch current directly: deriving it from the noisy voltages would turn independent
        voltage errors at the two ends of a short branch into a current error far larger
        than the instrument's. Either way the currents are encoded like voltages, normalized
        by the fitted edge normalizer, and zeroed outside the observed edges, so the two
        paths produce identical model inputs wherever the measured values match the
        state-derived ones. ``measure_power`` appends the terminal apparent power
        ``S = V_src · I*`` channels (own normalizer) — the source-row voltage is read
        from ``v_in``, which at every observed edge is a measured row.

        THE FRAME: under ``normalization.per_unit`` the FEATURES are divided by the per-unit
        current base at each half-edge's source row (and the power channels by the product of
        that row's voltage and current bases), while the returned ``i_edges`` stays in
        amperes — the anchored solve constrains in absolute units.

        Edge observability: every measurement location meters ALL its incident branches by
        default — a half-edge is observed iff its source row is observed (a gather of the
        node mask at the half-edge source rows, so the metered slack's feeders are covered
        automatically and the random-masking curriculum carries edge observations
        consistently). A device-based placement (installed ``MeasurementDevice`` CT
        channels) overrides this with its explicit half-edge set. A heterogeneous fleet's
        per-order coverage transfers from the metering node to its edges (a
        fundamental-only meter reads fundamental currents only).
        """
        if not self._measure_current:
            return None, None, None, None
        if i_meas is not None:
            if int(i_meas.shape[-1]) != int(self._edge_src_rows.numel()):
                raise ValueError(
                    f"i_meas covers {int(i_meas.shape[-1])} half-edges, the model has "
                    f"{int(self._edge_src_rows.numel())}; build the measurement mapping "
                    "and the model from the same grid."
                )
            i_edges = i_meas  # [*lead, H, E]
        else:
            i_edges = self._derive_edge_currents(
                v_in if v_true is None else v_true, topo
            )  # [*lead, H, E]
            # independent CT error on the state-derived currents (each half-edge keyed to its
            # metering node's accuracy class, drawn from its own stream). Off / no-noise runs
            # leave ``_edge_noise`` None -> byte-identical. Real measured currents (the ``i_meas``
            # branch above) already carry physical error and are not perturbed.
            if self._edge_noise is not None:
                i_edges = self._edge_noise.apply(i_edges)
            # derived at the state's own precision (``v_true`` as loaded), then carried
            # at the training dtype like every other input
            i_edges = i_edges.to(v_in.dtype)
        # the FEATURES are the same currents in the run's frame: under
        # ``normalization.per_unit`` over the per-unit current base at each half-edge's
        # source row, while ``i_edges`` stays in amperes for the anchored solve.
        e_base = self._edge_current_base()
        i_feat = (
            i_edges
            if e_base is None
            else i_edges / e_base.to(device=i_edges.device, dtype=i_edges.real.dtype)
        )
        e_feats = self.edge_normalizer.transform(
            _to_nhc(self.encoding.encode(i_feat))
        )  # [*lead, E, H, C]
        if self.cfg.measurement.measure_power:
            # S = V_src·I* in the SAME frame as its two factors (per-unit volts times
            # per-unit amperes, i.e. over the base V_base[src]·i_base[src])
            v_src = self._to_feature_frame(v_in).index_select(
                -1, self._edge_src_rows
            )  # [*lead, H, E]
            s_feats = self.power_normalizer.transform(
                _to_nhc(self.encoding.encode(v_src * i_feat.conj()))
            )
            e_feats = torch.cat([e_feats, s_feats], dim=-1)  # [*lead, E, H, 2C]

        explicit = getattr(self._masker, "observed_edge_rows", None)
        if explicit is not None:
            e = int(self._edge_src_rows.numel())
            edge_mask = torch.zeros((*lead, e), dtype=torch.bool, device=device)
            if explicit.numel():
                edge_mask[..., explicit.to(device)] = True
        else:
            edge_mask = mask.index_select(-1, self._edge_src_rows)  # [*lead, E]
            if self._edge_upstream is not None:
                # one-CT budget: restrict each metered location to its toward-slack
                # half-edge (device-based explicit channel sets above are already exact)
                edge_mask = edge_mask & self._edge_upstream

        edge_order_mask = None
        if order_mask is not None:
            edge_order_mask = order_mask.index_select(
                -2, self._edge_src_rows
            ) & edge_mask.unsqueeze(-1)  # [*lead, E, H]
            e_feats = e_feats * edge_order_mask.unsqueeze(-1).to(e_feats.dtype)
        else:
            e_feats = e_feats * edge_mask.unsqueeze(-1).unsqueeze(-1).to(e_feats.dtype)
        edge_obs = e_feats.reshape(*e_feats.shape[:-2], -1)  # [*lead, E, H·C]
        return edge_obs, edge_mask, edge_order_mask, i_edges

    def _derive_edge_currents(self, v_in: Tensor, topo: Optional[Tensor]) -> Tensor:
        """Half-edge currents ``[*lead, H, E]`` the CTs read, from the input state.

        With a switchable topology each sample's currents come from ITS configuration's
        operator (an open switch carries none), so the batch is split by variant and
        gathered back — the same grouping the physics decode uses.
        """
        # the operators are plain objects (not nn.Modules) — Lightning does not relocate
        # their stored maps with the module, so follow the batch device here.
        ops = self._topology.edge_ops if self._topology is not None else ()
        if topo is None or len(ops) <= 1:
            if self.edge_op.device != v_in.device:
                self.edge_op.to(v_in.device)
            return self.edge_op(v_in)
        from ..data.topology import merge_groups, topology_groups

        groups = topology_groups(topo, len(ops))
        for k, _ in groups:
            if ops[k].device != v_in.device:
                ops[k].to(v_in.device)
        if len(groups) == 1:
            return ops[groups[0][0]](v_in)
        parts = [ops[k](v_in.index_select(0, sel)) for k, sel in groups]
        return merge_groups(parts, groups)

    def _slack_voltage(self, v: Tensor) -> Tensor:
        """The measured slack voltage — ``[*lead, K]`` at the fundamental, or ``[*lead, H, K]``
        at every order under ``physics.harmonic_slack="measured"`` (the network boundary).

        ``v`` is the TRUE state ``[*lead, H, N]``; the slack rows are metered and forced
        observed in every mask (``slack_known``), so reading them here uses exactly the
        values the model receives as input — a measurement read, not leakage. When the
        orders carry no fundamental, the harmonic Norton solve needs no slack and a zero
        placeholder is returned.
        """
        if getattr(self, "_harmonic_slack", "norton") == "measured":
            # every order's measured slack voltage: the boundary of every solve
            return v.index_select(-1, self._slack_rows)  # [*lead, H, K]
        if self._fundamental_pos is None:
            return v.new_zeros((*v.shape[:-2], int(self._slack_rows.numel())))
        return v.select(-2, self._fundamental_pos).index_select(-1, self._slack_rows)

    def _sync_solver_device(self, device) -> None:
        """Move the network solver's cached operators onto the model's device once.

        The solver is a plain object (not an ``nn.Module``), so Lightning does NOT relocate its
        cached ``Y(h)`` / slack tensors with the module. Check the SOLVER's OWN device, not the
        auto-moved ``_slack_rows`` buffer (Lightning already moved that, which would mask the
        mismatch and leave the solver's operators stranded on the CPU).

        A multi-grid run moves its whole operator BANK, which relocates every BUILT member
        (solver, current operator, cached per-unit base) and sends later ones straight there.
        """
        if self._bank is not None:
            self._bank.to(device)
            return
        for solver in self._solvers:
            if solver.slack_rows.device != device:
                solver.to(device)

    def _batch_items(self, batch) -> int:
        """Scenario items in a batch (the ``batch_size`` every logged metric is weighted by).

        A single-grid batch stacks its items on the leading axis of ``v``; a multi-grid batch
        folds them into the node axis, so the count is the number of member graphs it carries.
        """
        if self._multigrid:
            return int(batch["ptr"].numel()) - 1
        return int(batch["v"].shape[0])

    def on_train_epoch_start(self) -> None:
        """Reset the running training-loss accumulator for the new epoch (see
        :attr:`train_epoch_loss`)."""
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    @property
    def train_epoch_loss(self) -> float:
        """Sample-weighted MEAN training loss over the CURRENT epoch (``nan`` before any batch).

        Accumulated OFF the autograd graph inside :meth:`training_step` (the returned ``loss``
        is untouched for the backward pass). This is available and CORRECT at
        :meth:`on_validation_epoch_end` — validation runs after every training batch of the
        epoch — unlike Lightning's epoch-aggregated ``train_loss`` metric, which is only
        finalized at ``on_train_epoch_end`` (after validation) and is therefore one epoch stale
        when read at validation time. The training-history collector reads this so the recorded
        training loss is the epoch mean, not the last minibatch.
        """
        if not self._train_loss_count:
            return float("nan")
        return self._train_loss_sum / self._train_loss_count

    def training_step(self, batch, batch_idx):  # noqa: D401
        loss, _, _, _, inj_loss, phys_loss, _ = self._step(batch, train=True)
        # Non-finite guard: a single pathological batch (a data corner, an ill-posed
        # anchored solve) must not poison the weights — one NaN loss backpropagated
        # once turns every parameter NaN and silently kills all remaining stages.
        # The batch's contribution is replaced by an EXACT-ZERO gradient over all
        # parameters: Lightning forbids returning ``None`` from ``training_step``
        # under distributed training (every rank must run a synchronized backward),
        # so the surrogate keeps the backward + DDP all-reduce alive while the
        # optimizer receives no update from this batch. Logged via ``print`` (the
        # drivers suppress Python warnings) and aborted when it stops being rare —
        # a persistent stream of skips is a real divergence, not a corner.
        # one host read serves both the guard and the epoch-mean accumulator below
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            self._nonfinite_batches += 1
            msg = (
                f"non-finite training loss at global step {self.global_step} "
                f"(batch {batch_idx}, {self._nonfinite_batches} skipped so far) — "
                "zero-gradient surrogate applied."
            )
            print(f"WARNING: {msg}", flush=True)
            warnings.warn(msg, stacklevel=2)
            if self._nonfinite_batches > 50:
                raise RuntimeError(
                    "more than 50 training batches produced a non-finite loss — "
                    "persistent divergence, aborting instead of skipping further."
                )
            return sum((p * 0.0).sum() for p in self.parameters() if p.requires_grad)
        bs = self._batch_items(batch)
        # per-step value for the live progress bar and the non-finite guard (spikes show up
        # immediately); the epoch MEAN is tracked separately below and exposed as
        # ``train_epoch_loss`` for the history collector.
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        self._train_loss_sum += loss_value * bs
        self._train_loss_count += bs
        if inj_loss is not None:
            self.log("train_injection_loss", inj_loss, batch_size=bs)
        if phys_loss is not None:
            self.log("train_physics_loss", phys_loss, batch_size=bs)
        # the blended supervision's parts (``train_voltage_loss`` /
        # ``train_injection_state_loss`` / ``train_plain_decode_loss``), so the split can be
        # tuned from the curves
        for part, value in self._loss_components.items():
            self.log(f"train_{part}_loss", value, batch_size=bs)
        if self._loss_share is not None:
            self.log("train_injection_share", self._loss_share, batch_size=bs)
        return loss

    def on_after_backward(self) -> None:
        """Zero all gradients when ANY is non-finite (the backward-side batch guard).

        The loss can be finite while the backward produces NaN (an overflow in a
        discarded ``where`` branch, an ill-conditioned solve adjoint); by the time
        the next batch's loss reports NaN the weights are already poisoned. Under
        DDP the gradients are all-reduced before this hook, so every rank sees the
        same non-finite values and zeroes consistently. Shares the counter and the
        abort threshold with the loss-side guard in :meth:`training_step`.
        """
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        # ONE device-side reduction and ONE host read: per-tensor host reads would each
        # block on the CUDA pipeline, serializing every training step on the sync latency.
        finite = (
            torch.all(torch.stack([g.isfinite().all() for g in grads]))
            if grads
            else None
        )
        if finite is not None and not bool(finite):
            self._nonfinite_batches += 1
            msg = (
                f"non-finite gradients at global step {self.global_step} "
                f"({self._nonfinite_batches} skipped so far) — gradients zeroed."
            )
            print(f"WARNING: {msg}", flush=True)
            warnings.warn(msg, stacklevel=2)
            if self._nonfinite_batches > 50:
                raise RuntimeError(
                    "more than 50 training batches produced non-finite losses or "
                    "gradients — persistent divergence, aborting."
                )
            for g in grads:
                g.zero_()

    def on_validation_epoch_start(self) -> None:
        """Reset the per-member error accumulator of a multi-grid validation epoch."""
        self._val_grid_error = None

    def on_validation_epoch_end(self) -> None:
        """Publish the current epoch's MEAN training loss (``train_loss_epoch``) to the logger.

        Logged here — not in :meth:`training_step` — because the epoch mean is only complete
        once every training batch of the epoch has run (validation runs at that point). The
        sanity-check pass (before any training) and any validation with no preceding training
        batch are skipped. The training-history collector reads :attr:`train_epoch_loss`
        directly, so this logging is for MLflow / the progress bar.

        A multi-grid run additionally publishes its PER-MEMBER quality here, where the whole
        validation epoch has been seen (see :meth:`_log_grid_metrics`).
        """
        if self._multigrid:
            self._log_grid_metrics()
        if (
            getattr(self.trainer, "sanity_checking", False)
            or not self._train_loss_count
        ):
            return
        self.log("train_loss_epoch", self.train_epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self._multigrid:
            return self._multigrid_validation_step(batch)
        loss, mask, pred_v, target_v, inj_loss, phys_loss, order_mask = self._step(
            batch, train=False
        )
        bs = batch["v"].shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        if inj_loss is not None:
            self.log(
                "val_injection_loss",
                inj_loss,
                prog_bar=True,
                batch_size=bs,
                sync_dist=True,
            )
        if phys_loss is not None:
            self.log(
                "val_physics_loss",
                phys_loss,
                prog_bar=True,
                batch_size=bs,
                sync_dist=True,
            )
        for part, value in self._loss_components.items():
            self.log(f"val_{part}_loss", value, batch_size=bs, sync_dist=True)
        self._log_supervision_diagnostics(bs)
        # fold any time axis into the batch for the per-step metrics (mask broadcasts over T).
        # The metered slack rows are part of the mask itself (forced observed in
        # ``_draw_mask``), so the observed/unobserved metric split needs no
        # decode-specific correction — both decodes count the slack as observed.
        n = self.n_nodes
        metric_mask = mask.reshape(-1, n)
        # heterogeneous fleets pass the per-order observability through, so a smart-meter
        # row counts as observed at the fundamental and unobserved (estimated) at the
        # harmonic orders — the observed/unobserved split stays honest per order.
        metric_order_mask = (
            None if order_mask is None else order_mask.reshape(-1, n, len(self.orders))
        )
        errs = state_errors(
            pred_v.reshape(-1, len(self.orders), n),
            target_v.reshape(-1, len(self.orders), n),
            mask=metric_mask,
            orders=self.orders,
            base_v=self.base_v,
            order_mask=metric_order_mask,
        )
        sd = {"sync_dist": True, "batch_size": bs}
        self._log_state_metrics(errs, sd)
        if self._plain_pred_v is not None:
            # what the network itself achieves (the slack-only decode of its injections),
            # beside the anchored estimate the headline metrics report
            self._log_plain_metrics(
                state_errors(
                    self._plain_pred_v.reshape(-1, len(self.orders), n),
                    target_v.reshape(-1, len(self.orders), n),
                    mask=metric_mask,
                    orders=self.orders,
                    base_v=self.base_v,
                    order_mask=metric_order_mask,
                ),
                sd,
            )
        return loss

    def _log_plain_metrics(self, errs: dict, sd: dict) -> None:
        """Log the unobserved per-unit MAE of the PLAIN (slack-only) decode.

        ``val_pu_mae_unobs_plain`` (all orders) and ``val_pu_mae_unobs_plain_h{h}`` per order
        — the same definition as ``val_pu_mae_unobs`` / ``val_pu_mae_unobs_h{h}``, evaluated
        on the slack-only solve of the predicted injections instead of the anchored one.
        Reported only while the plain-decode prior term is active
        (:attr:`~injection_hse.config.LossConfig.plain_decode_weight` > 0 and the solve anchors), so
        the history shows what the network itself achieves beside the anchored estimate.
        """
        self.log("val_pu_mae_unobs_plain", errs["mae_unobs"], **sd)
        for order in self.orders:
            self.log(
                f"val_pu_mae_unobs_plain_h{int(order)}",
                errs[int(order)]["mae_unobs"],
                **sd,
            )

    def _log_state_metrics(self, errs: dict, sd: dict) -> None:
        """Log the state-quality metrics of one validation batch.

        Primary metrics are PER-UNIT when a base voltage is available. The curriculum
        early-stop monitors ``val_pu_mae`` (over ALL rows — well-defined for every stage,
        including a fully-observed one); ``val_pu_mae_unobs`` is the headline estimation
        quality. The physical-volt numbers are kept too.

        ``sync_dist``: under multi-GPU (DDP) every rank sees only its shard — the reduction
        makes the logged value (and thus ``callback_metrics``) identical on every rank, which
        the stopping / best-weight callbacks REQUIRE to stay in lockstep. A no-op on a single
        device.
        """
        self.log("val_pu_mae", errs["mae_all"], prog_bar=True, **sd)
        self.log("val_pu_mae_unobs", errs["mae_unobs"], prog_bar=True, **sd)
        self.log("val_pu_mae_obs", errs["mae_obs"], **sd)
        self.log("val_pu_rmse_unobs", errs["rmse_unobs"], **sd)
        self.log("val_mae_unobs_phys", errs["mae_unobs_phys"], **sd)
        self.log("val_mae_obs_phys", errs["mae_obs_phys"], **sd)
        self._log_per_order(errs, sd)

    def _log_supervision_diagnostics(self, bs: int) -> None:
        """Log what the supervision blend actually did this validation step.

        ``val_injection_share`` is the fraction of the blended loss the injection term
        carried — the quantity ``loss.injection_state_loss_split`` is meant to set, and the
        only honest check that it does. ``val_inj_rel_{cancel,other}_h{h}`` is the per-order
        injection identifiability at the load+generator nodes versus everywhere else
        (:meth:`_cancelling_row_errors`); absent for a grid with no such node.
        """
        sd = {"batch_size": bs, "sync_dist": True}
        if self._loss_share is not None:
            self.log("val_injection_share", self._loss_share, prog_bar=True, **sd)
        for name, value in getattr(self, "_injection_diag", {}).items():
            self.log(f"val_{name}", value, **sd)

    def _log_per_order(self, errs: dict, sd: dict) -> None:
        """Log the per-ORDER validation quality behind the aggregate ``val_pu_mae*``.

        The aggregates average over the solved orders, where the fundamental's per-unit error
        is orders of magnitude larger than any harmonic's — a regression confined to one band
        is invisible in them. Per order: ``val_pu_mae_h{h}`` (all rows) /
        ``val_pu_mae_unobs_h{h}`` (estimated rows) and ``val_rel_unobs_h{h}``, the residual
        against the order's OWN signal scale (a small order estimated badly reads ``0.5``
        there and ``~0`` in per-unit). The two band aggregates ``val_pu_mae_unobs_fund`` (the
        fundamental) and ``val_pu_mae_unobs_harm`` (the mean over the harmonic orders) are the
        compact "fundamental vs harmonics" pair the training-history parquet and the live
        progress view read.
        """
        harm: list[float] = []
        for order in self.orders:
            entry = errs[int(order)]
            self.log(f"val_pu_mae_h{int(order)}", entry["mae_all"], **sd)
            self.log(f"val_pu_mae_unobs_h{int(order)}", entry["mae_unobs"], **sd)
            self.log(f"val_rel_unobs_h{int(order)}", entry["rel_unobs"], **sd)
            if int(order) == 1:
                self.log("val_pu_mae_unobs_fund", entry["mae_unobs"], **sd)
            else:
                harm.append(float(entry["mae_unobs"]))
        if harm:
            self.log("val_pu_mae_unobs_harm", sum(harm) / len(harm), **sd)

    def configure_optimizers(self):
        """Build the optimizer and, optionally, a learning-rate scheduler.

        Reads :attr:`injection_hse.config.OptimConfig` from ``self.cfg.optim``, with the current
        stage's :class:`~injection_hse.config.MaskStageConfig` ``lr``/``scheduler`` overrides
        applied when set (:meth:`set_stage_optim` — a fresh optimizer is built at every
        stage's fit). Optimizer choices: ``"adam"``, ``"sgd"``, or ``"adamw"`` (default).
        Scheduler choices: ``"none"`` (returns the optimizer directly),
        ``"cosine"`` (CosineAnnealingLR annealed ONCE over the stage's ``max_epochs``), or
        ``"plateau"`` (ReduceLROnPlateau monitoring ``val_loss``). A stage warmup
        (``warmup_epochs``) ramps the rate linearly from ``lr / (warmup + 1)`` to ``lr``
        first; the cosine then anneals over the remaining epochs.
        """
        oc = self.cfg.optim
        lr = float(self._stage_optim.get("lr", oc.lr))
        scheduler = self._stage_optim.get("scheduler", oc.scheduler)
        # include the injection head's parameters when localization is enabled
        params = list(self.model.parameters())
        if self.injection_head is not None:
            params += list(self.injection_head.parameters())
        # include the learned grid-calibration corrections (trained jointly)
        if self.grid_calibration is not None:
            params += list(self.grid_calibration.parameters())
        if oc.name == "adam":
            opt = torch.optim.Adam(
                params, lr=lr, weight_decay=oc.weight_decay, eps=oc.eps
            )
        elif oc.name == "sgd":
            opt = torch.optim.SGD(params, lr=lr, weight_decay=oc.weight_decay)
        else:
            opt = torch.optim.AdamW(
                params, lr=lr, weight_decay=oc.weight_decay, eps=oc.eps
            )

        warmup = int(self._stage_optim.get("warmup_epochs", 0))
        if warmup:
            # linear ramp from lr / (warmup + 1) to lr, then the stage's own schedule over
            # the remaining epochs (stepped per epoch, like the cosine below)
            ramp = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1.0 / (warmup + 1), end_factor=1.0, total_iters=warmup
            )
            if scheduler == "plateau":
                raise ValueError("a learning-rate warmup cannot precede 'plateau'.")
            if scheduler == "cosine":
                t_max = int(getattr(self.trainer, "max_epochs", None) or 50) - warmup
                main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max))
            else:
                main = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[ramp, main], milestones=[warmup]
            )
            return {"optimizer": opt, "lr_scheduler": sched}
        if scheduler == "none":
            return opt
        if scheduler == "cosine":
            # Anneal ONCE over the stage's epoch budget. A fixed T_max (e.g. 50) shorter than
            # max_epochs makes the cosine PERIODIC: the LR returns to its hot maximum mid-stage
            # (at 2·T_max, 4·T_max, ...), and a hot LR on an already-converged model in a sharp
            # minimum is exactly when a single step destabilises AdamW and the loss collapses to
            # the constant-prediction floor. Tying T_max to max_epochs makes it monotonic → tiny
            # LR late in training, where it belongs.
            t_max = int(getattr(self.trainer, "max_epochs", None) or 50)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max))
            return {"optimizer": opt, "lr_scheduler": sched}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }

    # -- persistence: the normalizer(s) travel with the checkpoint -----------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add the fitted state a reload needs to the Lightning checkpoint.

        Beside the weights (``state_dict``) a checkpoint carries:

        - ``pgl_config`` — the :class:`~injection_hse.config.ExperimentConfig` (``model_dump``);
        - ``normalizer`` and, when built, ``inj_normalizer`` / ``edge_normalizer`` /
          ``power_normalizer`` — the fitted :class:`~injection_hse.normalization.HarmonicNormalizer`
          states;
        - ``pgml_provenance`` — :func:`pgml.provenance.code_provenance` of the writing code;
        - ``injection_prior`` — the fitted linear injection prior, when one is installed;
        - ``loss_preconditioner`` — ``{mode, floor, sigma_max, n_flat}``, when installed;
        - ``anchor_weights`` — ``[v, i]`` active when written (solver or operator bank);
        - ``grid_fingerprint`` + ``grid_node_ids`` / ``grid_phase_codes`` — the row-layout
          identity, when the module was built with a ``grid``.
        """
        from pgml.provenance import code_provenance

        checkpoint["normalizer"] = self.normalizer.state_dict()
        checkpoint["pgl_config"] = self.cfg.model_dump()
        # WHICH CODE produced these weights: the commit + dirty flag + library versions
        # (:func:`pgml.provenance.code_provenance`, a JSON-ready dict). Config and seed alone
        # do not identify a run — the code between two identical configs may have changed.
        # Reading it back is optional everywhere: a checkpoint written before this existed
        # simply carries no such key.
        checkpoint["pgml_provenance"] = code_provenance()
        if self.inj_normalizer is not None:
            checkpoint["inj_normalizer"] = self.inj_normalizer.state_dict()
        if self.edge_normalizer is not None:
            checkpoint["edge_normalizer"] = self.edge_normalizer.state_dict()
        if self.power_normalizer is not None:
            checkpoint["power_normalizer"] = self.power_normalizer.state_dict()
        # The distillation teacher (``loss.prior_distillation_weight``): a FITTED artefact
        # like the normalizers, so it travels with the checkpoint — a reload continues (or
        # reproduces) the exact objective that trained these weights, without refitting.
        if self.injection_prior is not None:
            checkpoint["injection_prior"] = self.injection_prior.prior_state()
        # The loss preconditioner leaves the forward pass untouched, so a reload needs
        # nothing to infer; its identity is recorded so a resumed run can rebuild and
        # check the same one (``injection_hse.train.attach_loss_preconditioner``).
        if self.loss_preconditioner is not None:
            checkpoint["loss_preconditioner"] = {
                "mode": self.loss_preconditioner.mode,
                "floor": self.loss_preconditioner.floor,
                "sigma_max": self.loss_preconditioner.sigma_max,
                "n_flat": self.loss_preconditioner.n_flat,
            }
        # Measurement-anchoring weights ACTIVE when this checkpoint was written (the stage's
        # resolved schedule value). A reload deploys these — so the final placement/anneal
        # checkpoint carries the measurement-dominant weight the model was tuned for, and
        # inference reproduces that exact decode.
        if self.solver is not None:
            checkpoint["anchor_weights"] = [
                float(self.solver.anchor_v_weight),
                float(self.solver.anchor_i_weight),
            ]
        elif self._bank is not None:
            checkpoint["anchor_weights"] = [float(w) for w in self._stage_anchor]
        # Grid identity: a stable fingerprint of the row-layout contract the weights were
        # trained against, so a reload against a differently-laid-out grid is refused rather
        # than silently mislabeling the state rows (:meth:`from_checkpoint`).
        if self._layout_fingerprint is not None:
            checkpoint["grid_fingerprint"] = self._layout_fingerprint
            checkpoint["grid_node_ids"] = self._layout_node_ids
            checkpoint["grid_phase_codes"] = self._layout_phase_codes

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "normalizer" in checkpoint:
            self.normalizer.load_state_dict(checkpoint["normalizer"])
        if self.inj_normalizer is not None and "inj_normalizer" in checkpoint:
            self.inj_normalizer.load_state_dict(checkpoint["inj_normalizer"])
        if self.edge_normalizer is not None and "edge_normalizer" in checkpoint:
            self.edge_normalizer.load_state_dict(checkpoint["edge_normalizer"])
        if self.power_normalizer is not None and "power_normalizer" in checkpoint:
            self.power_normalizer.load_state_dict(checkpoint["power_normalizer"])
        if "injection_prior" in checkpoint:
            self.set_injection_prior(checkpoint["injection_prior"])
        # The noise-calibrated anchor weights are derived FROM the normalizer scales, so a
        # checkpoint restore that swaps the normalizer state (init_from onto freshly
        # generated data, a Lightning ckpt_path resume) must re-derive them — otherwise the
        # solver keeps weights from the pre-restore fit while the loss uses the restored
        # statistics. Idempotent; a no-op under manual weighting.
        from .anchoring import install_anchor_calibration

        install_anchor_calibration(self)

    # -- inference seam (evaluation) -----------------------------------------
    @torch.no_grad()
    def infer(self, batch) -> dict:
        """Decode the estimated complex state for one batch (eval, no autograd).

        Runs the shared forward in ``eval`` mode under ``torch.no_grad()`` (so dropout /
        batch-norm are inactive and nothing is taped) and returns the de-normalized complex
        prediction alongside the target and the per-sample observation mask:

        - ``pred_v``   complex ``[*lead, H, N]`` — the estimated state in physical units;
        - ``target_v`` complex ``[*lead, H, N]`` — the ground-truth state;
        - ``mask``     bool ``[*lead, N]`` (``True`` = observed at that sample/step);
        - ``order_mask`` bool ``[*lead, N, H]`` — the per-order observability of a
          heterogeneous fleet, or ``None`` (homogeneous).

        ``*lead`` is ``[B]`` for a snapshot batch or ``[B, T]`` for a sequence batch (the
        evaluation collapses ``T`` into the sample axis downstream). Tensors stay on the
        module's device; the caller detaches to numpy for plotting (off the tape).

        MULTI-GRID: ``pred_v`` and ``target_v`` come back in the frame the module trains in
        (``data.multigrid.angle_reference``) — with anything but ``"absolute"`` both are
        rotated into each member's own reference frame, so they are directly comparable to
        each other but angle-shifted against the raw solver state the dataset stores.
        Magnitudes, and hence the per-unit residual every reported metric is built from, are
        unaffected.
        """
        was_training = self.training
        self.eval()
        try:
            _, mask, pred_v, target_v, _, _, order_mask = self._step(batch, train=False)
        finally:
            if was_training:
                self.train()
        return {
            "pred_v": pred_v,
            "target_v": target_v,
            "mask": mask.bool(),
            "order_mask": None if order_mask is None else order_mask.bool(),
        }
