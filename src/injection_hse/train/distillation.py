"""The LINEAR-PRIOR distillation objective: teach the injection head the optimal linear map.

Behind :attr:`injection_hse.config.LossConfig.prior_distillation_weight`. An injection-decode estimator
is supervised through a measurement-anchored network solve, and that solve is FLAT in exactly
the injection directions the measurements already determine — the objective carries no signal
there, so the predicted injections drift while the reported voltage loss keeps falling. The
closed-form linear injection prior (:func:`injection_hse.analysis.fit_linear_injection_prior`) is the
conditional-mean injection under a linear model of the estimator's own measurements, and it
does not suffer from that: it is fitted directly, in closed form, on the injections.

Distillation supervises the model's predicted nodal injections against the PRIOR's
predictions, in the model's own normalized injection feature space — the same loss structure
as the direct injection term (:attr:`~injection_hse.config.LossConfig.injection_state_loss_split`) with
the LABEL changed from the true ``Y(h)·V(h)`` to the prior's estimate of it. Supervising the
TRUE injections instead is a different, worse objective: the individual nodal injections are
not identifiable from a sparse sensor set, so the raw label asks for information the data does
not contain, while the prior's label is by construction reachable from the measurements.

The prior is a means, not the target — a per-stage weight
(:attr:`injection_hse.config.MaskStageConfig.prior_distillation_weight`) anneals the term away so a run
can finish on the physics objective it is judged by.

Two pieces:

- :class:`InjectionPriorTeacher` — the fitted map as an on-device, batched teacher: the
  prior's own FIXED placement is read out of the batch's measured state (and metered branch
  currents), pushed through the affine map in torch on the module's device, and returned as
  nodal injection currents in amperes. Kept as a child module of
  :class:`~injection_hse.train.SEModule` so it follows the device; its buffers are non-persistent (the
  fitted map is not a trained parameter — it travels with the checkpoint as its own payload).
- :func:`attach_injection_prior` — the one fit both single-grid training entry points call
  before a pass trains with the prior: fit on that pass's TRAIN split
  (:func:`prior_fit_states`), at the deployed placement, and install.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from injection_hse.analysis.baselines import LinearInjectionPrior
    from injection_hse.config import ExperimentConfig
    from injection_hse.data import DataSource

    from .module import SEModule

__all__ = [
    "InjectionPriorTeacher",
    "attach_injection_prior",
    "distillation_requested",
    "prior_fit_states",
    "prior_requested",
]


def distillation_requested(cfg: "ExperimentConfig") -> bool:
    """Whether ``cfg`` asks for the prior-distillation term anywhere in its curriculum.

    The global weight (:attr:`injection_hse.config.LossConfig.prior_distillation_weight`) or ANY stage
    override (:attr:`injection_hse.config.MaskStageConfig.prior_distillation_weight`) — a run that
    distils only in one stage still needs the prior fitted before training starts.
    """
    if float(cfg.loss.prior_distillation_weight) > 0.0:
        return True
    return any(
        stage.prior_distillation_weight is not None
        and float(stage.prior_distillation_weight) > 0.0
        for stage in cfg.curriculum.stages
    )


def prior_requested(cfg: "ExperimentConfig") -> bool:
    """Whether ``cfg`` needs the fitted linear injection prior installed before training.

    Either as the distillation teacher (:func:`distillation_requested`) or as the base of
    the prior-residual decode (:attr:`injection_hse.config.PhysicsConfig.injection_prior_residual`).
    """
    return distillation_requested(cfg) or bool(cfg.physics.injection_prior_residual)


class InjectionPriorTeacher(torch.nn.Module):
    """The fitted linear injection prior, evaluated on a training batch on the model's device.

    Holds :class:`~injection_hse.analysis.LinearInjectionPrior`'s affine coefficients as non-persistent
    float64 buffers plus the FIXED placement the fit read, and turns a batch of measured
    states into the prior's predicted nodal injection currents (:meth:`injections`). A child
    module of :class:`~injection_hse.train.SEModule`, so Lightning moves it with the estimator; the
    fitted map itself is saved and restored as a checkpoint payload
    (:meth:`prior_state`), not as part of the trained ``state_dict``.

    The teacher always reads its OWN placement — the sensors the map was fitted on — rather
    than whatever mask the current curriculum stage drew. A distillation label must be the
    same function of the data in every stage, and the deployed placement is the setting the
    prior is defined in; a stage that observes more rows than the teacher simply gives the
    student more input than its teacher had.
    """

    def __init__(self, prior: "LinearInjectionPrior") -> None:
        super().__init__()
        if prior.observed_rows is None:
            raise ValueError(
                "this linear injection prior carries no record of the sensor placement it "
                "was fitted at, so the distillation label cannot be rebuilt from it. Refit "
                "it with injection_hse.analysis.fit_linear_injection_prior (which records the "
                "placement) instead of loading a bundle written before it did."
            )
        self.orders = tuple(int(o) for o in prior.orders)
        self.n_rows = int(prior.n_rows)
        self.noisy = bool(prior.noisy)
        self.n_train = int(prior.n_train)
        self._prior_state = {k: v for k, v in prior.state_dict().items()}
        f64 = torch.float64
        self.register_buffer(
            "feature_mean", torch.as_tensor(prior.feature_mean, dtype=f64), False
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(prior.feature_scale, dtype=f64), False
        )
        self.register_buffer(
            "target_mean", torch.as_tensor(prior.target_mean, dtype=f64), False
        )
        self.register_buffer("coef", torch.as_tensor(prior.coef, dtype=f64), False)
        self.register_buffer(
            "target_rows", torch.as_tensor(prior.rows, dtype=torch.int64), False
        )
        # the fitted placement as boolean masks with a single leading slot: the feature
        # construction reads only the observed COLUMNS off them, so one row serves any batch
        node_mask = torch.zeros((1, self.n_rows), dtype=torch.bool)
        node_mask[0, torch.as_tensor(prior.observed_rows, dtype=torch.int64)] = True
        self.register_buffer("node_mask", node_mask, False)
        edge_mask = None
        if prior.observed_edge_rows is not None:
            n_edges = int(prior.observed_edge_rows.max()) + 1
            edge_mask = torch.zeros((1, n_edges), dtype=torch.bool)
            edge_mask[
                0, torch.as_tensor(prior.observed_edge_rows, dtype=torch.int64)
            ] = True
        self.register_buffer("edge_mask", edge_mask, False)

    def prior_state(self) -> dict:
        """The fitted map as plain arrays — what a checkpoint carries, so a reload distils
        against exactly the same teacher (:meth:`injection_hse.analysis.LinearInjectionPrior.state_dict`).
        """
        return dict(self._prior_state)

    @property
    def prior(self) -> "LinearInjectionPrior":
        """The fitted :class:`~injection_hse.analysis.LinearInjectionPrior` this teacher evaluates."""
        from injection_hse.analysis.baselines import LinearInjectionPrior

        return LinearInjectionPrior.from_state_dict(self._prior_state)

    def check_compatible(self, module: "SEModule") -> None:
        """Refuse a prior that was not fitted on THIS module's grid, orders or placement.

        The map's input and output layouts are the estimator's own row layout and harmonic
        set; a mismatch would supervise the injection head against a label indexed
        differently, which no metric would reveal.
        """
        if tuple(int(o) for o in module.orders) != self.orders:
            raise ValueError(
                f"the linear injection prior was fitted on orders {list(self.orders)}, the "
                f"estimator solves {[int(o) for o in module.orders]}."
            )
        if int(module.n_nodes) != self.n_rows:
            raise ValueError(
                f"the linear injection prior was fitted on a grid with {self.n_rows} "
                f"node-phase rows, the estimator has {int(module.n_nodes)}."
            )
        if self.edge_mask is not None and not module._measure_current:
            raise ValueError(
                "the linear injection prior reads metered branch currents, but this run "
                "does not measure them (measurement.measure_current is off)."
            )

    def feature_availability(
        self, sample_mask: Tensor, sample_edge_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Which of the fitted map's features a batch actually observes, ``[S, F]`` bool.

        The map's feature layout is that of :func:`injection_hse.analysis.linear_prior_features`:
        order-major, per order the real then the imaginary parts of the placement rows
        (ascending row index), then — for a current-reading fit — the same for the
        metered half-edges. ``sample_mask`` ``[S, N]`` is the batch's own location mask;
        a placement row it does not observe is unavailable. ``sample_edge_mask``
        ``[S, E]`` plays the same role for the edges (``None`` = every fitted edge is
        available).
        """
        rows = torch.nonzero(self.node_mask[0], as_tuple=False).flatten()
        node_avail = sample_mask.to(torch.bool).index_select(-1, rows)  # [S, K]
        blocks = [node_avail, node_avail]
        if self.edge_mask is not None:
            edges = torch.nonzero(self.edge_mask[0], as_tuple=False).flatten()
            if sample_edge_mask is None:
                edge_avail = torch.ones(
                    (node_avail.shape[0], edges.numel()),
                    dtype=torch.bool,
                    device=node_avail.device,
                )
            else:
                edge_avail = sample_edge_mask.to(torch.bool).index_select(-1, edges)
            blocks += [edge_avail, edge_avail]
        per_order = torch.cat(blocks, dim=-1)  # [S, F / H]
        return per_order.repeat(1, len(self.orders))

    @torch.no_grad()
    def injections(
        self,
        module: "SEModule",
        v_meas: Tensor,
        i_edges=None,
        *,
        sample_mask: Optional[Tensor] = None,
        sample_edge_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """The prior's nodal injection currents ``[*lead, H, N]`` (amperes) for one batch.

        ``v_meas`` is the MEASURED state ``[*lead, H, N]`` the estimator itself reads (noisy
        when the run's sensor noise is on) and ``i_edges`` the metered half-edge currents
        ``[*lead, H, E]`` in amperes, both exactly as the training step derived them — the
        teacher and the student then read one realization of the measurements. Rows the fit
        does not predict (the slack) stay zero, which is what the network solve expects there.

        By default the map reads its OWN fitted placement whatever the batch observes (the
        distillation label). With ``sample_mask`` ``[*lead, N]`` (and ``sample_edge_mask``
        ``[*lead, E]``) the standardised features of the placement rows a sample does NOT
        observe are set to zero — their fitted mean — so the prediction never reads a row
        the estimator cannot see (the prior-residual decode inside a sparser stage).

        Runs in float64 on the batch's device, under ``no_grad``: a supervision target is a
        constant, never a graph.
        """
        from injection_hse.analysis.baselines import linear_prior_features

        lead = tuple(v_meas.shape[:-2])
        n_h, n = int(v_meas.shape[-2]), int(v_meas.shape[-1])
        flat_v = v_meas.reshape(-1, n_h, n)
        flat_i = (
            None
            if i_edges is None
            else i_edges.reshape(-1, int(i_edges.shape[-2]), int(i_edges.shape[-1]))
        )
        feats = linear_prior_features(
            module,
            flat_v,
            self.node_mask,
            i_edges=flat_i,
            edge_mask=self.edge_mask,
        )  # [S, F] float64
        if int(feats.shape[-1]) != int(self.coef.shape[1]):
            raise ValueError(
                f"the measurement features of this run are {int(feats.shape[-1])} wide, the "
                f"linear injection prior was fitted on {int(self.coef.shape[1])}; the prior "
                "and the estimator read different sensors."
            )
        z = (feats - self.feature_mean) / self.feature_scale
        if sample_mask is not None:
            avail = self.feature_availability(
                sample_mask.reshape(-1, n),
                None
                if sample_edge_mask is None
                else sample_edge_mask.reshape(-1, int(sample_edge_mask.shape[-1])),
            )
            z = z * avail.to(z.dtype)
        flat = self.target_mean + z @ self.coef.T  # [S, K]
        parts = flat.reshape(-1, len(self.orders), int(self.target_rows.numel()), 2)
        out = torch.zeros(
            (parts.shape[0], len(self.orders), self.n_rows),
            dtype=torch.complex128,
            device=feats.device,
        )
        out.index_copy_(
            -1,
            self.target_rows,
            torch.complex(parts[..., 0], parts[..., 1]),
        )
        return out.reshape(*lead, len(self.orders), self.n_rows).to(
            module._complex_dtype
        )


def prior_fit_states(source: "DataSource", n_states: int) -> Optional[Tensor]:
    """The first ``n_states`` TRAIN states of ``source`` as ``[S, H, N]`` (``None`` if empty).

    The deterministic, split-seeded sample the linear injection prior is fitted on: the
    train split's scenarios in order, a SEQUENCE dataset contributing every step of each
    (its ``[T, H, N]`` scenarios are flattened into the state axis, and only the scenarios
    needed to reach ``n_states`` are read). ``S = min(n_states, train states)``; the dtype
    is the dataset's.
    """
    ds = source.train_dataset()
    if not len(ds):
        return None
    first = ds[0]["v"]
    steps = int(first.shape[0]) if first.ndim == 3 else 1
    n_scenarios = min(len(ds), max(1, math.ceil(int(n_states) / steps)))
    v = torch.stack([ds[i]["v"] for i in range(n_scenarios)])
    return v.reshape(-1, *v.shape[-2:])[: int(n_states)]


def attach_injection_prior(
    module: "SEModule",
    cfg: "ExperimentConfig",
    train_v: Tensor,
    observed_rows: Optional[Sequence[int]],
    *,
    device: str = "cpu",
) -> Optional["LinearInjectionPrior"]:
    """Fit the linear injection prior on ``cfg``'s TRAIN data and install it on ``module``.

    Called by both single-grid training entry points before a pass trains with the prior
    (the distillation teacher, the prior-residual base), so the objective is fixed for the
    whole pass and the fitted map travels with every checkpoint of it. A multi-pass run
    (:func:`injection_hse.workflow.run_training`) calls it again before every later pass that needs
    the prior, on that pass's own data, replacing the installed map. A no-op (returning
    ``None``) unless the config asks for the prior (:func:`prior_requested`).

    ``train_v`` is a TRAIN-split state sample of ``cfg``'s dataset (``[S, H, N]``, or with a
    step axis ``[B, T, H, N]``) — a deterministic, split-seeded draw such as
    :func:`prior_fit_states`, so re-running the config refits the same map. Any leading
    axes are flattened into the state axis (a sequence contributes its steps as states)
    and the first ``cfg.physics.injection_prior_fit_states`` are used. The fitted bundle
    records ``"<cfg.data.dataset_path>:train"`` as its ``fit_dataset``.
    ``observed_rows`` is the DEPLOYED placement — the sensor set the prior is defined at and
    the run is judged on. The map is fitted on the run's own noise regime, so the teacher and
    the student read the same kind of measurement, and on the same construction of what this
    run's sensors report as the shipped linear-prior baseline
    (:func:`injection_hse.analysis.measured_inputs`), so the two are directly comparable.

    Returns the fitted :class:`~injection_hse.analysis.LinearInjectionPrior` (already installed).
    """
    if not prior_requested(cfg):
        return None
    if cfg.physics.decode != "injection":
        raise ValueError(
            "loss.prior_distillation_weight > 0 / physics.injection_prior_residual need "
            "physics.decode='injection': both act on the decode's predicted nodal "
            "injection currents, which the voltage decode does not produce."
        )
    if observed_rows is None or len(observed_rows) == 0:
        raise ValueError(
            "the linear injection prior is fitted at a FIXED sensor placement, so a "
            "distilling or prior-residual run needs one: give the curriculum a 'placement' "
            "stage (or set measurement.voltage_nodes) so the deployed sensor rows are "
            "defined."
        )
    from injection_hse.analysis.baselines import fit_linear_injection_prior

    rows = torch.as_tensor(list(observed_rows), dtype=torch.int64)
    n_fit = int(cfg.physics.injection_prior_fit_states)
    states = train_v.reshape(-1, *train_v.shape[-2:])[:n_fit]
    mask = torch.zeros((int(states.shape[0]), int(module.n_nodes)), dtype=torch.bool)
    mask[:, rows] = True
    # the fit evaluates the module, so restore whatever mode it was handed in: an
    # eval-mode estimator entering ``Trainer.fit`` would train with dropout / batch-norm off
    was_training = module.training
    try:
        prior = fit_linear_injection_prior(
            module,
            states,
            mask,
            noisy=bool(cfg.noise.enabled),
            device=device,
        )
    finally:
        module.train(was_training)
    dataset = cfg.data.dataset_path
    prior = prior.with_fit_dataset(None if dataset is None else f"{dataset}:train")
    module.set_injection_prior(prior)
    roles = []
    if distillation_requested(cfg):
        roles.append("distillation teacher")
    if cfg.physics.injection_prior_residual:
        roles.append("residual base (the output layer starts at zero)")
    source = "" if dataset is None else f" of {Path(dataset).name}"
    print(
        f"  linear injection prior as {' + '.join(roles)}: fitted on {prior.n_train} "
        f"TRAIN states{source} ({int(rows.numel())} observed rows"
        f"{', noisy features' if prior.noisy else ''})",
        flush=True,
    )
    return prior
