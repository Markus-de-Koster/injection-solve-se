"""The curriculum: per-stage maskers, the stage callbacks, and the config-swap guard.

:class:`StageManager` drives one ``Trainer.fit`` per masking stage over ONE module, so the
weights continue across the ladder (``full`` -> ``random`` -> ``placement``) and a checkpoint
is forced at every stage boundary. A stage ends at whichever comes first: its ``target_loss``
(:class:`TargetLossEarlyStop`), a plateau, or ``max_epochs``.

A two-task run reuses one module across several managers, each with its own config, so the
checkpoints record the config that actually trained them. That is only honest for fields read
per step: :func:`assert_cfg_swappable` rejects a swap that touches anything the module DERIVED
at construction (:data:`CFG_BEHAVIOUR_FIELDS`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import lightning as L
import torch
from torch import Tensor

from ..masking import build_masker
from .dtypes import _lightning_precision

if TYPE_CHECKING:
    from ..config import ExperimentConfig, MaskStageConfig
    from .module import SEModule


class TargetLossEarlyStop(L.Callback):
    """Stop a stage's ``fit`` once the monitored validation metric reaches its target.

    Monitors a logged scalar (default ``val_pu_mae`` — the per-unit MAE over ALL node-phase
    rows) and sets ``trainer.should_stop`` when it falls to ``<= target`` after a validation
    epoch. The all-rows metric is the deliberate default: it is well-defined for EVERY stage,
    including a fully-observed (``full``) one whose unobserved set is empty (where the
    unobserved MAE is vacuously 0 and would early-stop the stage after a single epoch). The
    unobserved-only error (``val_pu_mae_unobs``) remains the headline test-time quality.
    Complements the ``max_epochs`` guard: a stage ends at whichever comes first. The early-stop
    reason is recorded on the callback (``stopped_on_target``) so the driver can log how each
    stage ended.
    """

    def __init__(self, target: float, *, monitor: str = "val_pu_mae") -> None:
        super().__init__()
        self.target = float(target)
        self.monitor = monitor
        self.stopped_on_target = False
        self.best: float = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # the sanity-check validation runs BEFORE any training; stopping there would end the
        # stage with zero trained epochs (e.g. a "full" observability stage whose unobserved
        # metric is vacuously 0). Only act on a real post-training validation epoch.
        if getattr(trainer, "sanity_checking", False):
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        v = float(value)
        self.best = min(self.best, v)
        # ``trainer.should_stop`` is NOT synchronized by the fit loop: a rank that stops
        # alone leaves the others hanging at their next collective. Reduce the decision
        # (any-rank-True) so every rank exits the stage together; the monitored metric is
        # logged with ``sync_dist`` so ranks normally agree anyway. No-op on one device.
        stop = bool(
            trainer.strategy.reduce_boolean_decision(v <= self.target, all=False)
        )
        if stop:
            self.stopped_on_target = True
            trainer.should_stop = True


class _OptimizerStateLoader(L.Callback):
    """Load a carried optimizer state into a stage's fresh optimizer at the start of training.

    The moments and step counts come from ``state`` (an ``Optimizer.state_dict()``); the
    learning rate the stage's own schedule installed (``lr``, ``initial_lr`` per parameter
    group) is kept, so the stage's rate and warmup still apply.
    """

    def __init__(self, state: dict) -> None:
        self.state = state

    def on_train_start(self, trainer, pl_module) -> None:
        opt = trainer.optimizers[0]
        keep = [
            {k: group[k] for k in ("lr", "initial_lr") if k in group}
            for group in opt.param_groups
        ]
        opt.load_state_dict(self.state)
        for group, values in zip(opt.param_groups, keep):
            group.update(values)


class _BestWeightKeeper(L.Callback):
    """Keep a stage's best (lowest-``monitor``) weights and halt the stage on a non-finite loss.

    Two safeguards against a late-training instability (an AdamW step spike that pins the loss
    at the degenerate constant-prediction floor and then blows the weights up to ``inf``/NaN):

    - **best-weight snapshot** — at every (post-sanity) validation epoch the lowest finite
      ``monitor`` (the all-rows pu-MAE) seen so far is recorded together with a CPU copy of the
      module weights, so the stage can CONTINUE from its best state instead of a collapsed one;
    - **non-finite halt** — if ``monitor`` or the train/val loss becomes NaN/``inf`` the stage
      is stopped immediately. NaN weights are terminal and, because the curriculum reuses the
      SAME module, would otherwise carry into the next stage (every later epoch stays NaN).
    """

    def __init__(self, monitor: str = "val_pu_mae") -> None:
        super().__init__()
        self.monitor = monitor
        self.best: float = float("inf")
        self.best_state: Optional[dict] = None
        self.halted_nonfinite = False

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        import math

        if getattr(trainer, "sanity_checking", False):
            return
        metrics = trainer.callback_metrics
        nonfinite = any(
            metrics.get(key) is not None and not math.isfinite(float(metrics[key]))
            for key in (self.monitor, "val_loss", "train_loss")
        )
        # reduced (any-rank-True) so a non-finite blow-up on ONE rank halts every rank
        # together instead of deadlocking the collective. No-op on one device.
        if bool(trainer.strategy.reduce_boolean_decision(nonfinite, all=False)):
            self.halted_nonfinite = True
            trainer.should_stop = True
            return
        value = metrics.get(self.monitor)
        if value is None:
            return
        v = float(value)
        if v < self.best:
            self.best = v
            self.best_state = {
                k: t.detach().cpu().clone() for k, t in pl_module.state_dict().items()
            }


#: The config fields :class:`SEModule` DERIVES state from at construction — the model and its
#: input widths, the encoding, the numeric dtypes, the physics decode and its anchoring draw,
#: the loss weighting resolved into :class:`~injection_hse.loss.StateEstimationLoss`, the sensor-noise
#: model, the per-unit / angle frame, and the seeds of the mask and anchor streams. A config
#: SWAP (the two-task curriculum) may change run behaviour that is read per step, never one of
#: these: the derived state would keep describing the old config while the checkpoint records
#: the new one.
CFG_BEHAVIOUR_FIELDS: tuple[str, ...] = (
    "seed",
    "orders",
    "train.dtype",
    "model",
    "encoding.kind",
    "encoding.injection_kind",
    "encoding.time_features",
    # the multi-grid MAGNITUDE + ANGLE frames (the task a pass opens is a per-pass decision,
    # and a config with no multigrid section reads both as ``None`` — so a switch between the
    # single-grid and the multi-grid path is caught here too)
    "data.multigrid.per_unit",
    "data.multigrid.angle_reference",
    # the single-grid FEATURE frame: derived at construction (the normalizers were fit in it,
    # the seams convert against it) — a pass may not switch frames mid-run
    "normalization.per_unit",
    "data.topology",
    "data.structural_encoding",
    "measurement.slack_known",
    "measurement.heterogeneous_orders",
    "measurement.measure_current",
    "measurement.measure_power",
    "measurement.current_channels",
    "loss.harmonic_weight",
    "loss.magnitude_term_weight",
    "loss.predict_injection",
    "loss.injection_hidden_dim",
    "physics.decode",
    "physics.calibrate",
    "physics.anchor_weight_draw",
    # the decode's Y(h) is assembled once, with its device shunt, when the solver is built
    "physics.load_shunt",
    # the noise calibration of the anchor weights is installed on the solver at construction
    "physics.anchor_weighting",
    "physics.prior_injection_rel_sigma",
    "physics.clean_sensor_rel_sigma",
    # the physics-derived input priors widen the model token and size the solver's build
    # anchoring; both are fixed at construction
    "physics.input_priors",
    "physics.prior_anchor_voltage_weight",
    "physics.prior_anchor_current_weight",
    "noise",
)


def _config_value(cfg, path: str):
    """The value at a dotted config path (``None`` when a parent on the way is ``None``)."""
    value = cfg
    for part in path.split("."):
        if value is None:
            return None
        value = getattr(value, part)
    return value


def assert_cfg_swappable(current: ExperimentConfig, incoming: ExperimentConfig) -> None:
    """Raise unless ``incoming`` can replace ``current`` on an already-built module.

    A two-task curriculum continues ONE module across two :class:`StageManager` passes, and
    the module adopts each pass's config so its checkpoints record what actually trained. That
    is only honest for fields read per step; a field in :data:`CFG_BEHAVIOUR_FIELDS` was
    already baked into the module at construction (the model, the loss weights, the noise
    model, the decode), so swapping it would leave the module behaving as before while
    reporting the new config. Names the offending field rather than silently disagreeing.
    """
    differing = [
        path
        for path in CFG_BEHAVIOUR_FIELDS
        if _config_value(current, path) != _config_value(incoming, path)
    ]
    if differing:
        raise ValueError(
            "this StageManager's config cannot be adopted by the module it is running: "
            + ", ".join(differing)
            + " differ(s) from the config the module was built with, and the module derives "
            "state from it at construction (architecture, encoding, dtypes, loss weighting, "
            "noise, decode). Build a new module for such a change; only run-behaviour fields "
            "(learning rate, loss blend, stages, checkpoint dir) may differ between passes."
        )


#: Pre-package spellings of the guard and its field set, kept so existing call sites (and
#: their tests) resolve; :func:`assert_cfg_swappable` / :data:`CFG_BEHAVIOUR_FIELDS` are the
#: names to use.
_assert_cfg_swappable = assert_cfg_swappable
_CFG_BEHAVIOUR_FIELDS = CFG_BEHAVIOUR_FIELDS


class StageManager:
    """Drives the masking curriculum: per-stage masker + early-stop-on-target + checkpoints.

    Runs the ordered :class:`~injection_hse.config.MaskStageConfig` stages (``full`` -> ``random``
    ladder -> ``placement`` finetune), CONTINUING the model weights across stages (no
    re-init). For each stage it swaps the per-sample masker (:mod:`injection_hse.masking`), installs a
    :class:`TargetLossEarlyStop` on the validation per-unit all-rows MAE (+ the stage's
    ``max_epochs`` guard, and — when ``train.early_stop_patience > 0`` — a Lightning
    ``EarlyStopping`` plateau guard on the same metric), ``fit``\\ s, then forces a
    stage-boundary checkpoint
    ``stage_<name>.ckpt`` carrying the normalizer + config. The checkpoint written BEFORE the
    ``placement`` finetune stage is always preserved (so the pre-placement model is never
    lost). ``placement`` stages use the :class:`~injection_hse.data.MeasurementModel` ``observed_rows``.

    ``restore_best`` (default off) hardens a long run against a late-training instability: each
    stage then checkpoints and continues from its lowest-``val_pu_mae`` weights instead of its
    last epoch, and a stage that goes non-finite (NaN/``inf``) is halted — so a collapse cannot
    overwrite the saved model or carry NaN weights into the next stage.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        observed_rows: Optional[Tensor] = None,
        restore_best: bool = False,
        order_capability: Optional[Tensor] = None,
        observed_edge_rows: Optional[Tensor] = None,
    ) -> None:
        self.cfg = cfg
        self.observed_rows = observed_rows
        # heterogeneous placement extras (both optional): the per-row order coverage of the
        # sensor fleet (``MeasurementModel.order_capability``) and, for device-based
        # instrumentation, the explicit metered half-edges (``MeasurementModel
        # .observed_edges``; None = every incident branch of an observed row).
        self.order_capability = order_capability
        self.observed_edge_rows = observed_edge_rows
        # when True, each stage CONTINUES (and checkpoints) from its lowest-``val_pu_mae``
        # weights rather than its last epoch, and a stage that goes non-finite (NaN/inf) is
        # halted — so a late-training collapse cannot poison the saved model or the next stage.
        self.restore_best = restore_best
        #: optimizer state after the last ``run`` (``None`` unless ``carry_optimizer_state``)
        self.last_optimizer_state: Optional[dict] = None

    def _masker_for(self, stage: MaskStageConfig):
        observed_rows = self.observed_rows
        if stage.kind == "placement" and observed_rows is None:
            if self.cfg.data.multigrid is None:
                raise ValueError(
                    "a 'placement' stage requires sensor rows; set "
                    "MeasurementConfig.voltage_nodes (resolved to observed_rows)."
                )
            # multi-grid: the placement is per MEMBER and travels with each batch
            # (``batch["observed_rows"]``), so the masker only carries the stage KIND.
            observed_rows = torch.zeros(0, dtype=torch.int64)
        return build_masker(
            stage.kind,
            observed_fraction=stage.observed_fraction,
            observed_rows=observed_rows,
            fundamental_fraction=stage.fundamental_fraction,
            order_capability=self.order_capability,
            observed_edge_rows=self.observed_edge_rows,
        )

    def run(
        self,
        module: SEModule,
        datamodule,
        *,
        logger=None,
        extra_callbacks: Optional[list] = None,
        optimizer_state: Optional[dict] = None,
    ) -> list[Path]:
        """Run every stage sequentially; return the boundary checkpoint paths (in order).

        With :attr:`injection_hse.config.TrainConfig.carry_optimizer_state` each stage's optimizer
        starts from the previous stage's moments (the first stage from ``optimizer_state``
        when given, e.g. the state of the checkpoint a run resumes from); the state after
        the last stage is left in :attr:`last_optimizer_state` for a following pass. Without
        the flag every stage starts from a fresh optimizer and ``optimizer_state`` is unused.

        ``extra_callbacks`` are appended to every stage's callback list (after the
        per-stage :class:`TargetLossEarlyStop`), so a STATEFUL callback shared across calls
        — e.g. a history collector spanning the snapshots → sequences dataset switch — accumulates
        over all stages of every ``run``. Backward compatible (default: none).
        """
        ckpt_dir = Path(self.cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Determinism is a process-wide property, not a per-trainer one: TF32 matmuls (the
        # runtime enables them on CUDA for speed) round differently run-to-run under
        # accumulation-order changes, so they are turned off BEFORE any stage builds a
        # trainer. The kernel-level switch itself is Lightning's (`deterministic=` below).
        if self.cfg.train.deterministic:
            torch.set_float32_matmul_precision("highest")
        # The module adopts THIS manager's config: a two-task curriculum reuses one module
        # across two StageManager passes (snapshots then sequences), and the config drives runtime
        # training behaviour (optimizer, loss blend, stages) AND the serialized checkpoint
        # provenance (``on_save_checkpoint`` dumps ``module.cfg``). Without this the sequence
        # checkpoints would embed the snapshot pass's config. Everything the module DERIVED at
        # construction must already agree — checked here, so a mismatch names the field
        # instead of producing a checkpoint whose config is not the one that trained it.
        # Single-task runs pass the same object the module was built with (a no-op).
        assert_cfg_swappable(module.cfg, self.cfg)
        module.cfg = self.cfg
        extra = list(extra_callbacks or [])
        written: list[Path] = []
        carry = bool(self.cfg.train.carry_optimizer_state)
        state = optimizer_state if carry else None
        for stage in self.cfg.curriculum.stages:
            module.set_masker(self._masker_for(stage))
            # per-stage optimizer overrides (None = inherit OptimConfig); the fresh
            # Trainer below rebuilds the optimizer, so they apply exactly to this stage.
            module.set_stage_optim(
                lr=stage.lr, scheduler=stage.scheduler, warmup_epochs=stage.warmup_epochs
            )
            # per-stage prior-distillation weight (None = inherit LossConfig): the anneal
            # that lets a run build the injection map against the linear prior and then
            # finish on the deployed physics objective. No-op without the injection decode.
            module.set_stage_distillation(
                stage.prior_distillation_weight
                if stage.prior_distillation_weight is not None
                else self.cfg.loss.prior_distillation_weight
            )
            # per-stage measurement-anchoring weights (None = inherit PhysicsConfig): the
            # weight schedule ramps the anchor from injection-learning (low, dense stages) to
            # measurement-dominant (high, sparse placement). No-op without an injection solver.
            module.set_stage_anchor(
                v_weight=(
                    stage.anchor_voltage_weight
                    if stage.anchor_voltage_weight is not None
                    else self.cfg.physics.anchor_voltage_weight
                ),
                i_weight=(
                    stage.anchor_current_weight
                    if stage.anchor_current_weight is not None
                    else self.cfg.physics.anchor_current_weight
                ),
                v_range=stage.anchor_voltage_range,
                i_range=stage.anchor_current_range,
            )
            early = TargetLossEarlyStop(stage.target_loss)
            keeper = _BestWeightKeeper() if self.restore_best else None
            # optional plateau stop: end the stage once the monitored validation metric stops
            # improving (Lightning's EarlyStopping — patience exhausted). Same monitor as the
            # target early-stop (the all-rows pu-MAE, defined for every stage). Non-finite
            # halting is left to _BestWeightKeeper / the target stop, so disable check_finite.
            if self.cfg.train.early_stop_patience > 0:
                from lightning.pytorch.callbacks import EarlyStopping

                plateau = EarlyStopping(
                    monitor="val_pu_mae",
                    mode="min",
                    patience=self.cfg.train.early_stop_patience,
                    min_delta=self.cfg.train.early_stop_min_delta,
                    check_finite=False,
                )
            else:
                plateau = None
            trainer = L.Trainer(
                max_epochs=stage.max_epochs,
                accelerator=self.cfg.train.accelerator,
                devices=self.cfg.train.devices,
                # complex128 training -> Lightning double precision (float64 params/inputs);
                # complex64 keeps the configured precision (default "32"). See
                # :func:`_lightning_precision`.
                precision=_lightning_precision(self.cfg),
                logger=logger if logger is not None else True,
                callbacks=[
                    *([_OptimizerStateLoader(state)] if state is not None else []),
                    early,
                    *([plateau] if plateau else []),
                    *([keeper] if keeper else []),
                    *extra,
                ],
                accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,
                gradient_clip_val=(self.cfg.train.grad_clip or None),
                # False -> None: Lightning treats an explicit False as "force-disable",
                # while None leaves torch's global state untouched (today's behaviour).
                deterministic=(self.cfg.train.deterministic or None),
                enable_progress_bar=self.cfg.train.enable_progress_bar,
                enable_model_summary=False,
                # stage-boundary checkpoints are written explicitly below into
                # ``cfg.train.checkpoint_dir``; disable Lightning's automatic ModelCheckpoint
                # so it does not also litter a ``checkpoints/`` directory in the working dir.
                enable_checkpointing=False,
            )
            # continue from the prior stage's weights (the module is reused; Lightning's
            # fit resets only the trainer/optimizer state, not the model parameters).
            trainer.fit(module, datamodule=datamodule)
            if carry:
                state = trainer.optimizers[0].state_dict()
            # restore this stage's BEST weights before checkpointing + continuing, so a late
            # collapse (or a non-finite blow-up) does not become the saved / carried-forward
            # model. Loads the CPU snapshot back onto the module's device in place.
            if keeper is not None and keeper.best_state is not None:
                module.load_state_dict(keeper.best_state)
            # force a checkpoint at the stage boundary (the pre-placement model is preserved)
            path = ckpt_dir / f"stage_{stage.name}.ckpt"
            trainer.save_checkpoint(str(path))
            written.append(path)
        self.last_optimizer_state = state
        return written
