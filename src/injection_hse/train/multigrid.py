"""The multi-grid path: ONE estimator over many grids.

A multi-grid batch (:class:`injection_hse.data.MultiGridCollate`) folds its sample axis into the node
axis, so the batch state is ``[H, N*]`` over the concatenated node-phase rows of its member
grids — a different layout from the single-grid ``[B, H, N]``, with its own masking, its own
per-member physics dispatch and its own per-member metrics. :class:`MultiGridMixin` holds
that path as the mirror of the single-grid methods on :class:`injection_hse.train.SEModule`, which
inherits it: ``_multigrid_step`` mirrors ``_step``, ``_multigrid_mask`` mirrors
``_draw_mask``, ``_multigrid_edge_observations`` mirrors ``_edge_observations``. The two are
deliberately NOT merged — the batch layouts differ for physical reasons — but they share
every layer below the layout (the encoding, the normalizer, the loss, the anchoring).

Beside the mixin: the dataset-global normalizer fit (:func:`fit_multigrid_normalizer`) and
the split scoring (:func:`evaluate_multigrid`), the two module-level operations a multi-grid
run needs that the single-grid path solves differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from ..data import (
    per_unit_state,
    physical_state,
    placement_mask,
    segment_random_mask,
    slack_frame_rotation,
)
from ..io import build_model_input
from ..masking import PlacementMasker, RandomMasker
from ..metrics import state_errors
from ..normalization import HarmonicNormalizer
from ..time_features import time_features as _time_features
from .layout import (
    _BatchGraph,
    _flatten_features,
    _gather_segments,
    _scatter_segments,
    _to_nhc,
    _unflatten_features,
)
from .priors import kcl_prior, linear_state_prior, prior_anchor_weights

if TYPE_CHECKING:
    from .module import SEModule


def _member_slack_voltage(
    solver, v_abs: Tensor, node_rows: Tensor, *, fundamental_pos
) -> Tensor:
    """One member's measured slack voltage — the boundary of its solves.

    Gathered from the batch's ABSOLUTE-frame state ``v_abs [*lead, H, N*]`` at the member
    ``solver``'s slack rows of each of its ``k`` items (``node_rows [k, N_g]`` global):
    the fundamental alone, ``[k, *lead, K]``, for a Norton harmonic slack, or EVERY order,
    ``[k, *lead, H, K]``, when the member's solver holds the metered slack's harmonic
    voltage fixed (``PhysicsConfig.harmonic_slack="measured"``) — the supply's own
    distortion then enters the member through its boundary rows, as on the single-grid
    path. Without a fundamental among the solved orders (``fundamental_pos`` None) or a
    member without slack rows the Norton harmonic solve needs no boundary, and a zero
    placeholder keeps the solver's call signature uniform.
    """
    slack_rows = solver.slack_rows.to(node_rows.device)
    if slack_rows.numel() and getattr(solver, "measured_harmonic_slack", False):
        return _gather_segments(
            v_abs, node_rows.index_select(1, slack_rows)
        )  # [k, *lead, H, K]
    if fundamental_pos is None or slack_rows.numel() == 0:
        k = int(node_rows.shape[0])
        return v_abs.new_zeros((k, *v_abs.shape[:-2], int(slack_rows.numel())))
    return _gather_segments(
        v_abs.select(-2, fundamental_pos), node_rows.index_select(1, slack_rows)
    )  # [k, *lead, K]


class MultiGridMixin:
    """The ``[H, N*]`` multi-grid counterparts of the single-grid pipeline methods.

    Mixed into :class:`injection_hse.train.SEModule`; every method here runs only when the module was
    built for a ``data.multigrid`` run (``self._multigrid``), and reads the same module state
    the single-grid methods do — the encoding, the normalizer(s), the loss, the masker and
    the operator bank.
    """

    def _multigrid_step(self, batch, *, train: bool):
        """Forward for a batch of DIFFERENT grids; same return tuple as :meth:`_step`.

        The sample axis is folded into the node axis, so the batch's state is ``[H, N*]``
        (``[T, H, N*]`` for sequences) over the concatenated node-phase rows of its member
        grids and the leading ``*lead`` is empty (resp. ``[T]``). The pipeline mirrors
        :meth:`_step`: per-unit -> encode -> normalize -> per-graph mask -> model -> decode ->
        loss, with every physics object dispatched PER MEMBER out of the operator bank.

        PER-UNIT FIRST (``data.multigrid.per_unit``): features enter the encoder divided by
        each row's line-to-neutral base voltage, so ONE dataset-global normalizer spans the
        members' voltage classes; the returned estimate is converted back to volts, leaving the
        metrics (and hence ``pu_mae``) exactly the quantity the single-grid path reports. Its
        current analogue is the per-unit CURRENT base
        (:meth:`injection_hse.physics.MultiGridPhysicsBank.i_base`), which the injection decode divides
        by / multiplies back exactly the same way.

        THE ANGLE FRAME (``data.multigrid.angle_reference``): the state is rotated into each
        member's own reference frame ONCE, here, before anything reads it — so the model's
        input, its supervision target, the returned estimate and every metric live in the same
        frame, and no member can be off by the constant angle its data source happened to solve
        in. The reference row of each row comes from the batch (``reference_rows``, built by
        the source for the configured frame: the member's slack, or its LV busbar). The
        rotation is applied in the training dtype and preserves magnitudes exactly.

        THE FRAME AND THE PHYSICS. That rotation is a row-dependent diagonal ``D`` (order ``h``
        by ``h·phi``), and ``D`` does NOT commute with ``Y``: every physics operation — the
        decode solve, the injection targets, the branch currents, the anchoring residuals —
        therefore runs in the ABSOLUTE frame, on the RAW volts state the batch carries, and only
        the quantities the MODEL sees or predicts are rotated. Concretely, the injection decode
        runs model features (slack frame, per-unit) -> inverse-normalize -> unrotate to absolute
        -> × ``i_base`` (amperes) -> per-member solve against that member's absolute measured
        slack -> volts (absolute) -> rotate into the slack frame -> per-unit -> encode -> the
        IDENTICAL voltage loss the direct decode uses. Currents transform with the same ``D`` as
        voltages, which is what keeps the residual pairing ``i' - (D Y D^-1) v'`` a rotation of
        ``i - Y v``. A BRANCH current is rotated by the rotation of its half-edge's SOURCE row
        (the terminal a CT sits on); that is a convention rather than an identity — mutual
        coupling mixes phases with different reference angles, so ``C(h)·(D v)`` differs from
        ``D_edge·(C(h) v)`` — which is why the branch-current operator, too, is applied in the
        absolute frame. The terminal apparent power ``S = V_src·I*`` is frame-invariant (the
        two unit rotations cancel) and is never rotated.

        ``order_mask`` is always ``None`` (heterogeneous fleets are rejected on this path);
        ``phys_loss`` is ``None`` as in :meth:`_step` (physics is structural, not a soft term).
        """
        v_abs = batch["v"]  # complex [H, N*] or [T, H, N*], ABSOLUTE frame, volts
        # the simulated CTs read the state as loaded (see SEModule._step: complex64 cannot
        # resolve a branch current between two 20 kV terminals)
        v_loaded = v_abs
        if v_abs.dtype != self._complex_dtype:
            v_abs = v_abs.to(self._complex_dtype)
        rotation = (
            None  # the frame's unit phasors [*lead, H, N*] (None = absolute frame)
        )
        if self._angle_reference != "absolute":
            rows = batch.get("reference_rows")
            if rows is None:
                raise ValueError(
                    f"the {self._angle_reference!r} angle frame needs the batch's per-row "
                    "reference (batch['reference_rows'], emitted by MultiGridCollate); set "
                    "data.multigrid.angle_reference='absolute' to train in the raw frame."
                )
            rotation = slack_frame_rotation(v_abs, rows, orders=self.orders)
            v = v_abs * rotation
        else:
            v = v_abs
        device = v.device
        base_v = batch["base_voltage"].to(device)
        n = int(v.shape[-1])
        lead = tuple(v.shape[:-2])  # () snapshot | (T,) sequence
        has_time = bool(lead)

        # The clean state is the loss target, the injection label and the metric reference.
        # Everything MEASURED is noisy: one draw per sample feeds BOTH the model's input
        # features and the anchored solve's constraints (the observed voltages and the
        # measured slack). One meter yields one reading — drawing twice would let the solve
        # fit a measurement the model never saw, which no deployment can produce, and would
        # leave the anchored floor artificially exact. With noise off, the input features ARE
        # the target features (byte-identical to a run without noise).
        v_feat = per_unit_state(v, base_v) if self._per_unit else v
        target_feats = self._encode_normalize(v_feat)  # [*lead, N*, H, C]
        v_abs_meas = v_abs
        if self._noise is not None:
            # the per-unit state is the noise model's reference (nominal = 1), so the draw
            # happens AFTER the per-unit conversion, unlike the single-grid path's volts.
            v_feat_meas = self._noise.apply(v_feat)
            input_feats = self._encode_normalize(v_feat_meas)
            # back to the ABSOLUTE frame the solve constrains in: the noise is a magnitude
            # scale and a phase shift, so it commutes with both the per-unit scaling and the
            # slack rotation and the same realization survives the round trip.
            v_abs_meas = (
                physical_state(v_feat_meas, base_v) if self._per_unit else v_feat_meas
            )
            if rotation is not None:
                v_abs_meas = v_abs_meas * rotation.conj()
        else:
            input_feats = target_feats
        mask = self._multigrid_mask(batch, n, device).expand(*lead, n)  # [*lead, N*]
        masked = _flatten_features(input_feats) * mask.unsqueeze(-1).to(
            input_feats.dtype
        )

        # the per-member operators follow the batch device (plain objects, not moved by
        # Lightning); done before the branch currents / the decode read them. Both the
        # dispatch table and the current base are built only for a step that solves or meters
        # (a localization-head-only run reads neither).
        segments = None
        i_base = None
        if self._bank is not None and (
            self._predict_injection_state or self._measure_current or self._input_priors
        ):
            self._sync_solver_device(device)
            segments = self._multigrid_segments(batch)
            i_base = self._multigrid_i_base(batch).to(v_abs.real.dtype)  # [H, N*]
        edge_obs, edge_mask, i_edges = self._multigrid_edge_observations(
            v_abs, rotation, mask, batch, segments, i_base, device, v_loaded=v_loaded
        )
        # the physics-derived input priors read the MEASURED state (absolute frame) and
        # the measured branch currents the model sees, per member through the bank
        node_prior = self._multigrid_input_priors(
            v_abs_meas, rotation, mask, batch, segments, i_base, i_edges, edge_mask
        )

        model_input = build_model_input(
            _BatchGraph(
                batch["node_static"], batch["edge_index"], batch["edge_static"]
            ),
            masked,
            mask,
            has_time=has_time,
            node_time=self._multigrid_node_time(batch, has_time, device),
            edge_obs=edge_obs,
            edge_mask=edge_mask,
            node_prior=node_prior,
            node_batch=batch["batch"],
            node_ptr=batch["ptr"],
            edge_batch=batch["edge_batch"],
        )
        pred_flat = self.model(model_input).reshape(*lead, n, -1)  # [*lead, N*, Fout]
        pred_feats = _unflatten_features(
            pred_flat, self.n_harmonics, self.n_out_channels
        )  # [*lead, N*, H, C_out]

        if self._predict_injection_state:
            # model features (slack frame, per-unit currents, the INJECTION encoding) ->
            # absolute amperes
            i_pred = self.inj_encoding.decode(
                self.inj_normalizer.inverse_transform(pred_feats)
            ).transpose(-2, -1)  # [*lead, N*, H] -> [*lead, H, N*]
            if rotation is not None:
                i_pred = i_pred * rotation.conj()
            i_pred = i_pred * i_base
            self._apply_batch_anchor(train)
            pred_v_abs = self._multigrid_decode(
                i_pred,
                v_abs_meas,  # MEASURED state: the slack boundary + the anchor constraints
                mask,
                batch,
                segments=segments,
                i_edges=i_edges,
                edge_mask=edge_mask,
            )  # [*lead, H, N*] volts, ABSOLUTE frame
            pred_v = pred_v_abs if rotation is None else pred_v_abs * rotation
            pred_feats_v = self._encode_normalize(
                per_unit_state(pred_v, base_v) if self._per_unit else pred_v
            )
            supervised = self.loss_fn(
                pred_feats_v.reshape(-1, *pred_feats_v.shape[-3:]),
                target_feats.reshape(-1, *target_feats.shape[-3:]),
                mask=None,
            )
            # The direct injection term, blended exactly as on the single-grid path: the two
            # residuals are not commensurate (the injection one runs orders of magnitude
            # higher), so ``loss.injection_state_loss_normalize`` rescales it onto the voltage
            # term's scale before the blend — otherwise a nominal split of 0.1 realizes most of
            # the loss on the injections. ``_loss_share`` reports what the split actually bought.
            split = float(self.cfg.loss.injection_state_loss_split)
            if self.inj_loss_fn is not None and split > 0.0:
                target_inj = self._multigrid_injection_target_features(
                    v_abs, rotation, batch, segments, i_base
                )
                injection = self.inj_loss_fn(
                    pred_feats.reshape(-1, *pred_feats.shape[-3:]),
                    target_inj.reshape(-1, *target_inj.shape[-3:]),
                    mask=None,
                )
                self._loss_components = {
                    "voltage": supervised.detach(),
                    "injection_state": injection.detach(),
                }
                scaled = injection * self._injection_loss_scale(supervised, injection)
                supervised = (1.0 - split) * supervised + split * scaled
                self._loss_share = (
                    split * scaled.detach() / supervised.detach().clamp_min(1e-30)
                )
            else:
                self._loss_components = {"voltage": supervised.detach()}
                self._loss_share = None
            # The plain-decode prior term, exactly as on the single-grid path: the same
            # voltage loss on the slack-only solve of the same injections, per member through
            # the bank, while the members' solvers anchor (``loss.plain_decode_weight``).
            self._plain_pred_v = None
            if self._plain_decode_active() and self._multigrid_anchoring(segments):
                plain_abs = self._multigrid_decode(
                    i_pred,
                    v_abs_meas,
                    mask,
                    batch,
                    segments=segments,
                    plain=True,
                )  # [*lead, H, N*] volts, ABSOLUTE frame
                pred_v_plain = plain_abs if rotation is None else plain_abs * rotation
                supervised = self._add_plain_decode_term(
                    supervised,
                    self._encode_normalize(
                        per_unit_state(pred_v_plain, base_v)
                        if self._per_unit
                        else pred_v_plain
                    ),
                    target_feats,
                    pred_v_plain,
                    train=train,
                )
        else:
            supervised = self.loss_fn(
                pred_feats.reshape(-1, *pred_feats.shape[-3:]),
                target_feats.reshape(-1, *target_feats.shape[-3:]),
                mask=None,
            )
            self._loss_components = {"voltage": supervised.detach()}
            self._loss_share = None
            pred_v = self.encoding.decode(
                self.normalizer.inverse_transform(pred_feats)
            ).transpose(-2, -1)  # [*lead, N*, H] -> [*lead, H, N*]
            if self._per_unit:
                pred_v = physical_state(pred_v, base_v)

        loss = supervised
        # per-DEVICE localization over the batch's concatenated device axis: the head pools the
        # predicted node features at the GLOBAL device rows and is supervised against the
        # collate's target, which is already dimensionless (the per-unit current base).
        inj_loss = None
        if self.injection_head is not None and "injection_target" in batch:
            pred_inj = self.injection_head(
                pred_flat, batch["device_rows"], batch["device_rows_mask"]
            )  # [*lead, D*, O, 1]
            if pred_inj.numel():
                target_inj = batch["injection_target"].to(pred_inj.dtype)
                inj_loss = torch.nn.functional.mse_loss(
                    pred_inj.reshape(-1, *pred_inj.shape[-3:]),
                    target_inj.reshape(-1, *target_inj.shape[-3:]),
                )
                loss = loss + self.cfg.loss.injection_loss_weight * inj_loss
            else:
                # a batch drawn only from injection-free members has nothing to localize; the
                # head stays on the graph with an exactly-zero contribution, so a distributed
                # backward still sees its parameters (and no misleading 0 is logged).
                loss = loss + pred_inj.sum().to(loss.dtype) * 0.0
        return loss, mask, pred_v, v, inj_loss, None, None

    # -- multi-grid per-member dispatch ---------------------------------------
    def _multigrid_segments(self, batch) -> list[tuple[int, Tensor, Tensor]]:
        """Group a batch by member grid: ``(grid index, node rows [k, N_g], edge rows [k, E_g])``.

        The dispatch table of every per-member physics operation. ``grid_index`` / ``ptr`` /
        ``edge_ptr`` are int64 batch METADATA (never on the autograd tape), so grouping them in
        Python is a bookkeeping loop over the DISTINCT members of the batch — at most one
        iteration per item — while the tensors it produces keep the node/half-edge axes fully
        vectorized inside each member's solve. Members appear in first-sight order, and each
        member's rows are listed in item order, so the concatenation of all node rows is a
        permutation of ``range(N*)`` (:func:`_scatter_segments` inverts it).
        """
        device = batch["ptr"].device
        ptr = batch["ptr"].tolist()
        edge_ptr = batch["edge_ptr"].tolist()
        grouped: dict[int, list[int]] = {}
        for item, index in enumerate(batch["grid_index"].tolist()):
            grouped.setdefault(int(index), []).append(item)
        segments: list[tuple[int, Tensor, Tensor]] = []
        for index, items in grouped.items():
            n_g = ptr[items[0] + 1] - ptr[items[0]]
            e_g = edge_ptr[items[0] + 1] - edge_ptr[items[0]]
            node_rows = torch.tensor(
                [ptr[b] for b in items], dtype=torch.int64, device=device
            ).unsqueeze(-1) + torch.arange(n_g, dtype=torch.int64, device=device)
            edge_rows = torch.tensor(
                [edge_ptr[b] for b in items], dtype=torch.int64, device=device
            ).unsqueeze(-1) + torch.arange(e_g, dtype=torch.int64, device=device)
            segments.append((index, node_rows, edge_rows))
        return segments

    def _multigrid_i_base(self, batch) -> Tensor:
        """float64 ``[H, N*]`` per-unit CURRENT base of a batch (off the tape).

        Each item contributes its member's :meth:`injection_hse.physics.MultiGridPhysicsBank.i_base`, so
        the concatenation follows the batch's node axis exactly. Grid constants: the injection
        features are divided by this and the decoded amperes multiplied back, and neither
        direction back-propagates into ``Y`` through the unit convention.
        """
        return torch.cat(
            [self._bank.i_base(int(g)) for g in batch["grid_index"].tolist()], dim=-1
        )

    def _multigrid_anchoring(self, segments) -> bool:
        """Whether the members' solvers anchor the decode of THIS batch.

        The bank installs one runtime anchoring strength on every member
        (:meth:`injection_hse.physics.MultiGridPhysicsBank.set_anchor_weights`), so the batch's
        members — built on first access, as the decode builds them anyway — answer for it.
        """
        return any(self._bank.solver(index).anchoring for index, _, _ in segments)

    def _multigrid_decode(
        self,
        i_pred: Tensor,
        v_abs: Tensor,
        mask: Tensor,
        batch,
        *,
        segments=None,
        i_edges=None,
        edge_mask=None,
        plain: bool = False,
    ) -> Tensor:
        """Per-member physics decode of a multi-grid batch (the ABSOLUTE frame throughout).

        ``i_pred`` ``[*lead, H, N*]`` are the predicted nodal injection currents in AMPERES,
        ``v_abs`` the batch's raw state (volts, absolute) supplying every member's measured
        slack boundary and — when the stage anchors — its observed voltages. The batch is
        grouped by member (:meth:`_multigrid_segments`), each member's item segments gathered
        into ``[k, *lead, H, N_g]`` and solved in ONE batched
        :meth:`injection_hse.physics.NetworkSolver.solve` call, and the results scattered back into row
        order out-of-place. The python loop runs over the DISTINCT member grids of the batch,
        never over nodes, harmonics or samples. ``plain=True`` forces every member's
        slack-only solve whatever its anchoring state (the plain-decode prior term,
        :attr:`~injection_hse.config.LossConfig.plain_decode_weight`).

        Returns the reconstructed voltages ``[*lead, H, N*]`` in volts, ABSOLUTE frame.
        """
        if segments is None:
            segments = self._multigrid_segments(batch)
        parts: list[Tensor] = []
        rows: list[Tensor] = []
        for index, node_rows, edge_rows in segments:
            solver = self._bank.solver(index)
            # the member's switch-tied rows carry no estimable injection
            i_seg = solver.zero_tied(
                _gather_segments(i_pred, node_rows)
            )  # [k, *lead, H, N_g]
            v_slack = _member_slack_voltage(
                solver, v_abs, node_rows, fundamental_pos=self._fundamental_pos
            )
            if solver.anchoring and not plain:
                parts.append(
                    solver.solve(
                        i_seg,
                        v_slack,
                        v_meas=_gather_segments(v_abs, node_rows),
                        v_mask=_gather_segments(mask, node_rows),
                        i_meas=(
                            None
                            if i_edges is None
                            else _gather_segments(i_edges, edge_rows)
                        ),
                        edge_mask=(
                            None
                            if edge_mask is None
                            else _gather_segments(edge_mask, edge_rows)
                        ),
                    )
                )
            else:
                parts.append(solver.solve(i_seg, v_slack))
            rows.append(node_rows)
        return _scatter_segments(parts, rows, int(v_abs.shape[-1]))

    def _multigrid_input_priors(
        self,
        v_abs: Tensor,
        rotation,
        mask: Tensor,
        batch,
        segments,
        i_base,
        i_edges,
        edge_mask,
    ) -> Optional[Tensor]:
        """The ``node_prior`` block ``[*lead, N*, Fp]`` of a multi-grid batch, or ``None``.

        The per-member mirror of :meth:`injection_hse.train.SEModule.input_priors`: ``v_abs`` is the
        MEASURED state in volts, ABSOLUTE frame (the noisy one when the noise model is on —
        the priors are functions of the measurements the model sees), ``mask`` the per-graph
        location mask ``[*lead, N*]``, ``i_edges`` / ``edge_mask`` the measured half-edge
        currents (absolute amperes) and their observability. The batch is grouped by member
        (``segments``), each member's items run through ITS operators
        (:func:`injection_hse.train.priors.kcl_prior` / :func:`~injection_hse.train.priors.linear_state_prior`
        on the bank's solver, in that member's absolute units) and the results are scattered
        back onto the batch's node axis — then converted into the space the model reads:
        the implied injections divided by the per-unit CURRENT base and rotated into the
        run's angle frame (the injection normalizer's space, exactly as the injection labels
        of :meth:`_multigrid_injection_target_features`), the prior state rotated and made
        per-unit (the voltage features' space). ``None`` when the config lists no prior.
        Computed under ``no_grad``; in the training real dtype on the batch device.
        """
        if not self._input_priors:
            return None
        with torch.no_grad():
            if segments is None:
                segments = self._multigrid_segments(batch)
            n = int(v_abs.shape[-1])
            rows = [node_rows for _, node_rows, _ in segments]
            kappa_parts: list[Tensor] = []
            fraction_parts: list[Tensor] = []
            state_parts: list[Tensor] = []
            for index, node_rows, edge_rows in segments:
                solver = self._bank.solver(index)
                v_seg = _gather_segments(v_abs, node_rows)  # [k, *lead, H, N_g]
                m_seg = _gather_segments(mask, node_rows)  # [k, *lead, N_g]
                if "kcl" in self._input_priors:
                    kappa, fraction = kcl_prior(solver, v_seg, m_seg, None)
                    kappa_parts.append(kappa)
                    fraction_parts.append(fraction)
                if "linear_state" in self._input_priors:
                    currents = self._measure_current and i_edges is not None
                    wv, wi = prior_anchor_weights(
                        self.cfg.physics, solver, currents=currents
                    )
                    state_parts.append(
                        linear_state_prior(
                            solver,
                            v_seg,
                            _member_slack_voltage(
                                solver,
                                v_abs,
                                node_rows,
                                fundamental_pos=self._fundamental_pos,
                            ),
                            m_seg,
                            None,
                            _gather_segments(i_edges, edge_rows) if currents else None,
                            _gather_segments(edge_mask, edge_rows)
                            if currents
                            else None,
                            None,
                            v_weight=wv,
                            i_weight=wi,
                        )
                    )
            blocks: list[Tensor] = []
            if kappa_parts:
                kappa = (
                    _scatter_segments(kappa_parts, rows, n) / i_base
                )  # [*lead, H, N*]
                if rotation is not None:
                    kappa = kappa * rotation
                feats = self.inj_normalizer.transform(
                    _to_nhc(self.inj_encoding.encode(kappa))
                )  # [*lead, N*, H, C_inj]
                blocks.append(_flatten_features(feats))
                fraction = _scatter_segments(fraction_parts, rows, n)  # [*lead, N*]
                blocks.append(fraction.unsqueeze(-1).to(feats.dtype))
            if state_parts:
                v_prior = _scatter_segments(
                    state_parts, rows, n
                )  # [*lead, H, N*] volts
                if rotation is not None:
                    v_prior = v_prior * rotation
                if self._per_unit:
                    v_prior = per_unit_state(
                        v_prior, batch["base_voltage"].to(v_abs.device)
                    )
                blocks.append(_flatten_features(self._encode_normalize(v_prior)))
            return torch.cat(blocks, dim=-1).to(self._real_dtype)

    def _multigrid_injection_target_features(
        self, v_abs: Tensor, rotation, batch, segments, i_base: Tensor
    ) -> Tensor:
        """The direct injection term's LABEL for a multi-grid batch: ``[*lead, N*, H, C]``.

        Per member, the true nodal injections ``I(h) = Y_g(h)·V_true(h)`` of its own segments
        (:meth:`injection_hse.physics.NetworkSolver.injection_targets`, formed in the ABSOLUTE frame
        because ``Y`` does not commute with the slack rotation), divided by the per-unit current
        base and rotated into the slack frame — the space
        :func:`injection_hse.physics.fit_multigrid_injection_normalizer` fit the injection normalizer on,
        and the space the model's raw output already lives in. Built entirely under
        ``no_grad``: a supervision target is a constant, never a graph.
        """
        if segments is None:
            segments = self._multigrid_segments(batch)
        with torch.no_grad():
            parts = [
                self._bank.solver(index).zero_tied(
                    self._bank.solver(index).injection_targets(
                        _gather_segments(v_abs, node_rows)
                    )
                )
                for index, node_rows, _ in segments
            ]
            i_true = _scatter_segments(
                parts, [node_rows for _, node_rows, _ in segments], int(v_abs.shape[-1])
            )
            i_true = i_true / i_base
            if rotation is not None:
                i_true = i_true * rotation
            feats = _to_nhc(self.inj_encoding.encode(i_true))  # [*lead, N*, H, C_out]
            return self.inj_normalizer.transform(feats)

    def _multigrid_edge_observations(
        self,
        v_abs: Tensor,
        rotation,
        mask: Tensor,
        batch,
        segments,
        i_base,
        device,
        v_loaded: Optional[Tensor] = None,
    ):
        """Measured branch-current edge inputs of a multi-grid batch.

        Returns ``(edge_obs, edge_mask, i_edges)`` — ``(None, None, None)`` unless
        ``measurement.measure_current``. ``i_edges`` ``[*lead, H, E*]`` are the raw half-edge
        currents in the ABSOLUTE frame (the frame the anchored solve constrains in), derived
        per member by that member's :class:`~injection_hse.data.BranchCurrentOperator` and scattered back
        onto the batch's half-edge axis.

        The model's FEATURES are the same currents made cross-grid comparable: divided by the
        per-unit current base at each half-edge's SOURCE row (the terminal the CT sits on) and
        rotated into the slack frame with the rotation of that row — currents transform with the
        same ``D`` as the voltages. ``measure_power`` appends the terminal apparent power
        ``S = V_src·I*`` over the base ``V_base[src]·i_base[src]``; ``S`` is invariant under the
        frame (the two unit rotations cancel), so it carries no rotation.

        Edge observability is the all-incident default: a half-edge is observed iff its source
        row is (a gather of the node mask at ``edge_index[0]``), optionally restricted to the
        toward-slack half-edges of each member (``measurement.current_channels="upstream"``).

        A batch carrying ``i_meas`` ``[*lead, H, E*]`` (a deployment's measured half-edge
        currents, :func:`injection_hse.inference.deployment.deployment_batch`) uses those readings in
        place of the state-derived currents and restricts the observability to its
        ``edge_mask`` — the metered half-edges — on top of the rule above.
        """
        if not self._measure_current:
            return None, None, None
        src = batch["edge_index"][0].to(device=device, dtype=torch.int64)  # [E*]
        measured = batch.get("i_meas")
        if measured is not None:
            # a DEPLOYMENT batch carries the CTs' own readings (absolute amperes on the
            # model's half-edges, ``injection_hse.inference``): they replace the state-derived
            # currents of training, and carry no synthetic noise — the instrument's error
            # is already in them
            i_edges = measured.to(device=device, dtype=v_abs.dtype)  # [*lead, H, E*]
        else:
            v_ct = v_abs if v_loaded is None else v_loaded
            parts = [
                self._bank.current_operator(index)(_gather_segments(v_ct, node_rows))
                for index, node_rows, _ in segments
            ]
            i_edges = _scatter_segments(
                parts, [edge_rows for _, _, edge_rows in segments], int(src.numel())
            ).to(v_abs.dtype)  # [*lead, H, E*]
        e_base = i_base.index_select(-1, src)  # [H, E*]
        # The CT reads the TRUE branch current with its own error, and that ONE reading is
        # both the model's feature and the anchored solve's constraint — so the draw happens
        # once here, on the per-unit current the multi-grid noise model references, and the
        # returned ``i_edges`` carries it back to the decode in amperes.
        i_feat = i_edges / e_base
        if self._edge_noise is not None and measured is None:
            i_feat = self._edge_noise.apply(i_feat)
            i_edges = i_feat * e_base
        if rotation is not None:
            i_feat = i_feat * rotation.index_select(-1, src)
        e_feats = self.edge_normalizer.transform(
            _to_nhc(self.encoding.encode(i_feat))
        )  # [*lead, E*, H, C]
        if self.cfg.measurement.measure_power:
            v_src = v_abs.index_select(-1, src)  # [*lead, H, E*]
            base_v = batch["base_voltage"].to(device=device, dtype=v_abs.real.dtype)
            base_v = torch.where(base_v > 0, base_v, torch.ones_like(base_v))
            s_base = base_v.index_select(-1, src) * e_base
            s_feats = self.power_normalizer.transform(
                _to_nhc(self.encoding.encode(v_src * i_edges.conj() / s_base))
            )
            e_feats = torch.cat([e_feats, s_feats], dim=-1)  # [*lead, E*, H, 2C]

        edge_mask = mask.index_select(-1, src)  # [*lead, E*]
        if self.cfg.measurement.current_channels == "upstream":
            edge_mask = edge_mask & self._multigrid_upstream(batch, device)
        if measured is not None:
            # only a half-edge the deployment actually meters is an observation; the
            # engine has already checked that every half-edge the placement requires is
            # among them (an unmetered required terminal is refused, not zero-filled)
            edge_mask = edge_mask & batch["edge_mask"].to(
                device=device, dtype=torch.bool
            )
        e_feats = e_feats * edge_mask.unsqueeze(-1).unsqueeze(-1).to(e_feats.dtype)
        edge_obs = e_feats.reshape(*e_feats.shape[:-2], -1)  # [*lead, E*, H·C]
        return edge_obs, edge_mask, i_edges

    def _multigrid_upstream(self, batch, device) -> Tensor:
        """bool ``[E*]`` one-CT-budget half-edge mask of a batch (per member, cached).

        The multi-grid counterpart of the ``_edge_upstream`` buffer: each member's toward-slack
        half-edges (:func:`injection_hse.data.upstream_edge_mask`) are static per grid, computed on first
        sight and concatenated in item order.
        """
        from injection_hse.data import upstream_edge_mask

        blocks: list[Tensor] = []
        for index in batch["grid_index"].tolist():
            cached = self._edge_upstream_by_grid.get(int(index))
            if cached is None:
                bundle = self._bank.source.bundle(int(index))
                cached = upstream_edge_mask(bundle.grid, bundle.graph.edge_index)
                self._edge_upstream_by_grid[int(index)] = cached
            blocks.append(cached)
        return torch.cat(blocks).to(device)

    def _multigrid_mask(self, batch, n: int, device) -> Tensor:
        """Per-GRAPH observation mask ``[N*]`` of a multi-grid batch (``True`` = observed).

        The fraction of a ``random`` stage applies PER MEMBER (a global top-k would spend the
        whole budget on the largest grid in the batch), and the members' metered slack rows —
        carried by the collate as GLOBAL indices — are forced observed on top, the same rule
        the single-grid path applies to its own slack. A ``placement`` stage uses the fixed
        per-member placement the source resolved (installed measurement devices, else
        :func:`injection_hse.data.central_placement` under ``multigrid.placement_fraction``). A batch that
        already carries a ``node_mask`` (a collate given its own masker) wins over both.

        Drawn on the CPU mask stream and moved to the batch device, so the draw does not depend
        on the training device. One draw per BATCH: the sample axis already lives on the node
        axis, so every item still gets its own independent per-graph mask — but the steps of a
        SEQUENCE member share it (the single-grid path redraws per step).
        """
        supplied = batch.get("node_mask")
        if supplied is not None:
            return supplied.to(device=device, dtype=torch.bool)
        slack = batch.get("slack_rows") if self.cfg.measurement.slack_known else None
        if slack is not None:
            slack = slack.cpu()
        masker = self._masker
        if isinstance(masker, PlacementMasker):
            rows = batch.get("observed_rows")
            if rows is None:
                raise ValueError(
                    "a 'placement' stage over a multi-grid dataset needs each member's fixed "
                    "sensor rows in the batch (batch['observed_rows']): set "
                    "data.multigrid.placement_fraction, or attach measurement devices to the "
                    "member grids."
                )
            mask = placement_mask(rows.cpu(), n)
            if slack is not None and slack.numel():
                mask[slack] = True
        elif isinstance(masker, RandomMasker):
            if self.cfg.data.multigrid.random_mask_unit == "bus":
                row_bus = batch["row_bus"].cpu()
                forced_buses = row_bus[slack] if slack is not None else None
                bus_mask = segment_random_mask(
                    batch["bus_ptr"].cpu(),
                    fraction=masker.observed_fraction,
                    forced_rows=forced_buses,
                    generator=self._mask_gen,
                )
                mask = bus_mask[row_bus]
            else:
                mask = segment_random_mask(
                    batch["ptr"].cpu(),
                    fraction=masker.observed_fraction,
                    forced_rows=slack,
                    generator=self._mask_gen,
                )
        else:  # FullMasker
            mask = torch.ones(n, dtype=torch.bool)
        return mask.to(device)

    def _multigrid_node_time(self, batch, has_time: bool, device):
        """Per-step time features ``[T, Ftime]`` of a multi-grid batch (``None`` when off).

        The step axis of a multi-grid batch IS the leading axis (the sample axis lives on the
        node axis), so the per-step feature vector needs no batch broadcast — unlike
        :meth:`_build_node_time`, whose leading shape is ``[B, T]``.
        """
        if not self._n_time_features or not has_time:
            return None
        tu = batch.get("time_unix_s")
        if tu is None:
            return None  # timeless batch: the model fills zeros (valid flag = 0)
        return _time_features(tu.to(device)).to(self._real_dtype)

    # -- multi-grid validation + per-member metrics ------------------------
    def _multigrid_validation_step(self, batch):
        """Validation of a multi-grid batch: pooled metrics + the per-member accumulation.

        The pooled metrics are the single-grid ones computed over the CONCATENATED node axis,
        so ``val_pu_mae`` weights a member by its node count; the per-member breakdown is
        accumulated here and published at the end of the epoch (:meth:`_log_grid_metrics`).
        """
        loss, mask, pred_v, target_v, inj_loss, _, _ = self._step(batch, train=False)
        bs = self._batch_items(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        if inj_loss is not None:
            self.log(
                "val_injection_loss",
                inj_loss,
                prog_bar=True,
                batch_size=bs,
                sync_dist=True,
            )
        for part, value in self._loss_components.items():
            self.log(f"val_{part}_loss", value, batch_size=bs, sync_dist=True)
        self._log_supervision_diagnostics(bs)
        n = int(pred_v.shape[-1])
        errs = state_errors(
            pred_v.reshape(-1, len(self.orders), n),
            target_v.reshape(-1, len(self.orders), n),
            mask=mask.reshape(-1, n),
            orders=self.orders,
            base_v=batch["base_voltage"],
        )
        sd = {"sync_dist": True, "batch_size": bs}
        self._log_state_metrics(errs, sd)
        if self._plain_pred_v is not None:
            self._log_plain_metrics(
                state_errors(
                    self._plain_pred_v.reshape(-1, len(self.orders), n),
                    target_v.reshape(-1, len(self.orders), n),
                    mask=mask.reshape(-1, n),
                    orders=self.orders,
                    base_v=batch["base_voltage"],
                ),
                sd,
            )
        self._accumulate_grid_errors(batch, pred_v, target_v)
        return loss

    def _log_grid_metrics(self) -> None:
        """Log the per-MEMBER validation pu-MAE of a multi-grid epoch, plus the worst member.

        ``val_pu_mae_grid_<key>`` is one member grid's all-rows per-unit MAE (the per-member
        reading of the pooled ``val_pu_mae``, which weights a member by its node count), and
        ``val_pu_mae_worst_grid`` is the largest of them — the number that says whether the
        estimator generalizes across topologies or only serves the easy ones. Rank-local under
        multi-GPU training: a member is only scored on the shard that drew it.
        """
        sanity = self._trainer is not None and self.trainer.sanity_checking
        if sanity or self._val_grid_error is None:
            return
        # ONE device->host transfer for the whole epoch's accumulation
        total, count = (t.cpu() for t in self._val_grid_error)
        seen = count > 0
        if not bool(seen.any()):
            return
        values = total[seen] / count[seen]
        for index, value in zip(seen.nonzero(as_tuple=False).squeeze(-1), values):
            self.log(f"val_pu_mae_grid_{self._grid_name(int(index))}", float(value))
        self.log("val_pu_mae_worst_grid", float(values.max()), prog_bar=True)

    def _grid_name(self, index: int) -> str:
        """The stable metric-name suffix of a member grid (its manifest key, else its index)."""
        if self._grid_keys is not None and 0 <= index < len(self._grid_keys):
            return self._grid_keys[index]
        return str(int(index))

    def _accumulate_grid_errors(self, batch, pred_v: Tensor, target_v: Tensor) -> None:
        """Add this batch's per-member ``Σ|Δ|/base`` and element count to the epoch totals.

        The per-unit residual is the SAME quantity ``injection_hse.metrics.state_errors`` averages into
        ``mae_all``; accumulating the sum and the count (instead of per-batch means) makes the
        epoch value independent of how the members were distributed over the batches.
        """
        base = (
            batch["base_voltage"]
            .to(device=pred_v.device, dtype=pred_v.real.dtype)
            .clamp_min(1e-12)
        )
        residual = (pred_v - target_v).abs() / base  # [*lead, H, N*] per-unit
        # Everything stays on the device — per-item reads here would each block on the
        # accelerator pipeline; the accumulator is transferred ONCE at epoch end
        # (:meth:`_log_grid_metrics`).
        n_rows = residual.shape[-1]
        lead_numel = residual.numel() // n_rows
        row_sum = residual.reshape(-1, n_rows).sum(dim=0).to(torch.float64)  # [N*]
        grid_index = batch["grid_index"]
        ptr = batch["ptr"]
        rows_per_item = ptr[1:] - ptr[:-1]  # [n_items]
        item_index = torch.repeat_interleave(
            torch.arange(grid_index.numel(), device=row_sum.device), rows_per_item
        )
        item_sum = torch.zeros(
            grid_index.numel(), dtype=torch.float64, device=row_sum.device
        ).index_add_(0, item_index, row_sum)
        if self._val_grid_error is None:
            n_members = (
                len(self._grid_keys)
                if self._grid_keys is not None
                else int(grid_index.max()) + 1
            )
            self._val_grid_error = (
                torch.zeros(n_members, dtype=torch.float64, device=row_sum.device),
                torch.zeros(n_members, dtype=torch.float64, device=row_sum.device),
            )
        total, count = self._val_grid_error
        # only the keyless fallback can meet an index beyond its first-batch sizing (the
        # keyed accumulator spans every member by construction — no per-batch host read)
        if self._grid_keys is None and total.numel() <= int(grid_index.max()):
            grow = int(grid_index.max()) + 1 - total.numel()
            pad = torch.zeros(grow, dtype=torch.float64, device=total.device)
            self._val_grid_error = (
                torch.cat([total, pad]),
                torch.cat([count, pad.clone()]),
            )
            total, count = self._val_grid_error
        total.index_add_(0, grid_index, item_sum)
        count.index_add_(0, grid_index, (rows_per_item * lead_numel).to(torch.float64))


# =============================================================================
# The dataset-global fit and the split scoring
# =============================================================================
def fit_multigrid_normalizer(source, encoding, cfg) -> HarmonicNormalizer:
    """Fit ONE dataset-global :class:`~injection_hse.normalization.HarmonicNormalizer` over many grids.

    The statistics are estimated on the PER-UNIT train states of every member
    (:meth:`injection_hse.data.MultiGridDataSource.per_unit_states`, capped per member so the fit sees
    every topology rather than mostly the largest dataset) and pooled over the node axis, which
    is what makes a single normalizer valid across voltage classes: in volts a 400 V feeder and
    a 20 kV feeder would make each per-harmonic statistic a mixture of two scales and cost the
    small harmonic orders their visibility.

    The states are taken in the run's ANGLE FRAME (``data.multigrid.angle_reference``), the
    same one :meth:`SEModule._multigrid_step` puts every batch in: the magnitude statistics
    are rotation-invariant, but the Cartesian loss scale the normalizer also carries is not.
    Asking the source for a frame it was not built with raises rather than fitting against
    the wrong reference bus.

    Returns the fitted normalizer (as with the single-grid fit, it travels with the checkpoint).
    """
    multigrid = cfg.data.multigrid
    states = source.per_unit_states(
        split="train",
        max_scenarios_per_grid=8,
        angle_reference="slack" if multigrid is None else multigrid.angle_reference,
    )
    if not states:
        raise ValueError("the multi-grid manifest lists no train grids to fit on.")
    parts = []
    for state in states:
        feats = _to_nhc(encoding.encode(state))  # [S, (T,) N_g, H, C]
        parts.append(feats.reshape(-1, 1, feats.shape[-2], feats.shape[-1]))
    return HarmonicNormalizer().fit(
        torch.cat(parts),  # [M, 1, H, C] — every (sample, row) of every member
        magnitude_transform=cfg.normalization.magnitude_transform,
        per_harmonic=cfg.normalization.per_harmonic,
    )


@torch.no_grad()
def evaluate_multigrid(
    module: SEModule,
    source,
    *,
    split: str = "test",
    batch_size: int = 8,
    device="cpu",
    max_batches: Optional[int] = None,
) -> dict:
    """Score a trained multi-grid estimator on one split, pooled AND per member grid.

    The generalization reading of a multi-grid run: ``split="test"`` is the manifest's held-out
    set (the REAL grids — topologies never trained on), ``"val"`` the held-out generated ones.
    Every member is evaluated under the module's CURRENT masking policy, so the numbers are
    comparable to the ``val_pu_mae*`` logged during training.

    The reported per-unit MAE is measured in the module's ANGLE FRAME
    (``data.multigrid.angle_reference``; :meth:`SEModule.infer` returns estimate and target in
    the same one). In a referenced frame (``"slack"`` / ``"lv_busbar"``) a member's residual is
    therefore free of the constant angle its data source was solved in — which is the point: a
    real test grid with a different angle convention is scored on the state it actually got
    wrong.

    Parameters
    ----------
    module:
        A multi-grid :class:`SEModule` (``data.multigrid`` set).
    source:
        The :class:`injection_hse.data.MultiGridDataSource` the module trained on (its ``collate`` and
        member keys define the batch layout and the metric names).
    split:
        ``"test"`` (default), ``"val"`` or ``"train"``.
    batch_size:
        Member graphs per batch.
    device:
        Where to run; the module is moved there.
    max_batches:
        Stop after this many batches (smoke runs); ``None`` = the whole split.

    Returns
    -------
    dict
        ``{"split", "n_items", "pu_mae", "pu_mae_unobs", "worst_grid", "per_grid":
        {key: {"pu_mae", "pu_mae_unobs", "n_items", "n_nodes"}}}`` — all-rows and
        unobserved-row per-unit MAE, pooled over the split and per member grid. Empty
        (``n_items=0``) when the split holds no grids.
    """
    from torch.utils.data import DataLoader

    dataset = {
        "train": source.train_dataset,
        "val": source.val_dataset,
        "test": source.test_dataset,
    }[split]()
    dev = torch.device(device)
    module.to(dev)
    out: dict = {
        "split": split,
        "n_items": 0,
        "pu_mae": float("nan"),
        "pu_mae_unobs": float("nan"),
        "worst_grid": None,
        "per_grid": {},
    }
    if not len(dataset):
        return out

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=source.collate()
    )
    # per member: [Σ|Δ|/base over ALL rows, its element count, the same over the UNOBSERVED
    # rows, its count, scenario items]. Sums + counts (not per-batch means) so the split value
    # is independent of how the members were distributed over the batches.
    totals: dict[int, list[float]] = {}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        estimate = module.infer(batch)
        base = (
            batch["base_voltage"].to(device=dev, dtype=estimate["pred_v"].real.dtype)
        ).clamp_min(1e-12)
        residual = (estimate["pred_v"] - estimate["target_v"]).abs() / base
        unobserved = ~estimate["mask"]  # [*lead, N*]
        ptr = batch["ptr"].tolist()
        for item, grid_index in enumerate(batch["grid_index"].tolist()):
            lo, hi = ptr[item], ptr[item + 1]
            block = residual[..., lo:hi]
            hidden = unobserved[..., lo:hi].unsqueeze(-2).expand_as(block)
            entry = totals.setdefault(int(grid_index), [0.0, 0.0, 0.0, 0.0, 0.0])
            entry[0] += float(block.sum())
            entry[1] += float(block.numel())
            entry[2] += float(block[hidden].sum())
            entry[3] += float(hidden.sum())
            entry[4] += 1.0

    pooled = [sum(e[i] for e in totals.values()) for i in range(5)]
    out["n_items"] = int(pooled[4])
    out["pu_mae"] = pooled[0] / pooled[1] if pooled[1] else float("nan")
    out["pu_mae_unobs"] = pooled[2] / pooled[3] if pooled[3] else float("nan")
    worst, worst_key = -1.0, None
    for grid_index, entry in sorted(totals.items()):
        bundle = source.bundle(grid_index)
        value = entry[0] / entry[1] if entry[1] else float("nan")
        out["per_grid"][bundle.key] = {
            "pu_mae": value,
            "pu_mae_unobs": entry[2] / entry[3] if entry[3] else float("nan"),
            "n_items": int(entry[4]),
            "n_nodes": bundle.n_nodes,
        }
        if value == value and value > worst:  # skip a NaN member
            worst, worst_key = value, bundle.key
    out["worst_grid"] = worst_key
    return out
