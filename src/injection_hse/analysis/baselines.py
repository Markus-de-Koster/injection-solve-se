"""Non-learned reference predictors for the harmonic state estimator.

Four baselines put the learned estimator's numbers in context — a model that beats none of
them is not learning anything a closed-form reference could not. Three are trivial: they
train nothing, hold no parameters, and use plain tensor ops:

- **zero-harmonic** (``zero_harmonic_prediction``) — a flat fundamental profile at the observed
  slack voltage (nominal line-to-neutral where a slack phase is unavailable) and ZERO for every
  harmonic order ``h > 1``. The "assume a clean nominal grid" reference: it is exactly right for
  the fundamental of a lightly loaded feeder and maximally wrong about the distortion the
  estimator exists to recover.
- **training-mean** (``training_mean_prediction``) — the per-node-phase-row, per-order MEAN of
  the training targets, broadcast to every evaluated sample. The "predict the average state"
  reference: it captures the typical operating point but nothing sample-specific.
- **persistence** (``persistence_prediction``) — the previous time step's true state, for
  COHERENT sequence data only. The "nothing changed since last tick" reference for the temporal
  model; undefined (and skipped) for snapshot data.

The fourth is the demanding one, and it applies to an INJECTION-decode estimator:

- **linear injection prior** (:class:`LinearInjectionPrior`,
  :func:`fit_linear_injection_prior`, :func:`linear_injection_prediction`) — a closed-form
  ridge regression from the estimator's OWN measurement features onto the true nodal
  injections ``I(h) = Y(h)·V(h)``, decoded through the estimator's own measurement-anchored
  network solve. It is the conditional-mean injection under a linear model; since that solve
  is affine in the injection at fixed measurements, its state estimate is the corresponding
  conditional mean — the accuracy an estimator reaches by exploiting nothing but a linear
  function of what it is given. It holds parameters and it is fitted, but on a seeded split,
  in closed form, from a few hundred snapshots, so it sits with the references rather than
  with the learned variants. A learned estimator that loses to it is not short of
  information; it is short of using the information it already has.

Each returns a :class:`~injection_hse.evaluation.Prediction` (the model-free counterpart of
:func:`injection_hse.evaluation.predict_dataset`), so the SAME metrics and splits score them and their
rows drop straight into the ablation / report tables next to the learned results. The linear
prior additionally appears per harmonic order in
:func:`injection_hse.analysis.estimator_band_ablation`, beside the ``anchors_only`` (zero-injection
prior) row it is the direct upgrade of.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    import polars as pl
    from torch import Tensor

    from injection_hse.config import MeasurementConfig
    from injection_hse.data import DataSource
    from injection_hse.evaluation import Prediction
    from injection_hse.train import SEModule

#: Offset applied to a run's noise seed when the prior is fitted on NOISY features, so the
#: fit's draws are reproducible and independent of the draws any scoring pass makes.
_LINEAR_PRIOR_NOISE_OFFSET = 2000


# =============================================================================
# Scoring (shared with the ablation table)
# =============================================================================
def score_prediction(prediction: "Prediction") -> dict:
    """The common per-run metric row from a :class:`~injection_hse.evaluation.Prediction`.

    Scores the prediction with :func:`injection_hse.metrics.state_errors` on the UNOBSERVED rows (the
    headline estimation quality) and returns the columns shared by the learned and the baseline
    rows: aggregate pu ``mae`` / ``rmse``, the fundamental / harmonic-band pu ``mae_fund`` /
    ``mae_harm``, the mean per-harmonic RELATIVE error ``rel_harm`` (residual over the order's
    own signal scale, so small orders are not hidden), and the voltage-THD error ``thd_mae``.
    """
    import torch

    from injection_hse.metrics import state_errors

    errors = state_errors(
        torch.as_tensor(prediction.pred_v),
        torch.as_tensor(prediction.target_v),
        mask=torch.as_tensor(prediction.mask),
        orders=prediction.orders,
        base_v=torch.as_tensor(prediction.base_v),
    )
    harm = [o for o in prediction.orders if o != 1]
    return {
        "mae": float(errors["mae_unobs"]),
        "rmse": float(errors["rmse_unobs"]),
        "mae_fund": (
            float(errors[1]["mae_unobs"]) if 1 in prediction.orders else float("nan")
        ),
        "mae_harm": (
            float(np.mean([errors[o]["mae_unobs"] for o in harm]))
            if harm
            else float("nan")
        ),
        "rel_harm": (
            float(np.nanmean([errors[o]["rel_unobs"] for o in harm]))
            if harm
            else float("nan")
        ),
        "thd_mae": float(errors["thd_mae_unobs"]),
    }


# =============================================================================
# The trivial baselines (Prediction -> Prediction, pred_v replaced)
# =============================================================================
def _with_pred(reference: "Prediction", pred_v: np.ndarray) -> "Prediction":
    """A copy of ``reference`` with ``pred_v`` replaced (everything else identical)."""
    import dataclasses

    return dataclasses.replace(reference, pred_v=pred_v.astype(reference.pred_v.dtype))


def zero_harmonic_prediction(
    reference: "Prediction", *, slack_rows: Optional[np.ndarray] = None
) -> "Prediction":
    """Flat fundamental at the observed slack voltage, zero harmonics.

    The fundamental of each node-phase row is set to the observed slack voltage of the SAME
    phase (averaged over the slack rows of that phase when several slacks are metered), falling
    back to the nominal line-to-neutral base voltage for a phase with no metered slack; every
    harmonic order ``h > 1`` is predicted as exactly zero. ``slack_rows`` are the metered slack
    rows (from :meth:`injection_hse.data.MeasurementModel.slack_rows`); when omitted or empty the flat
    profile uses the nominal base voltage.
    """
    target = reference.target_v  # [S, H, N] complex
    pred = np.zeros_like(target)
    h1 = reference.orders.index(1) if 1 in reference.orders else 0
    phase_codes = reference.phase_codes  # [N]
    base_v = reference.base_v  # [N]

    fundamental = np.tile(base_v.astype(target.dtype)[None, :], (target.shape[0], 1))
    if slack_rows is not None and len(slack_rows):
        slack_rows = np.asarray(slack_rows, dtype=np.int64)
        slack_phase = phase_codes[slack_rows]
        for p in np.unique(phase_codes):
            rows_p = np.flatnonzero(phase_codes == p)
            sr_p = slack_rows[slack_phase == p]
            if sr_p.size:
                # per-sample mean slack fundamental of this phase -> broadcast to its rows
                val = target[:, h1, :][:, sr_p].mean(axis=1)  # [S] complex
                fundamental[:, rows_p] = val[:, None]
    pred[:, h1, :] = fundamental
    return _with_pred(reference, pred)


def training_mean_prediction(
    reference: "Prediction", train_target: np.ndarray
) -> "Prediction":
    """The per-row, per-order MEAN of the training targets, broadcast to every sample.

    ``train_target`` is the complex training-split state ``[S_train, H, N]`` (the ground-truth
    voltages, NOT any model output). Its mean over the sample axis is the constant estimate for
    every evaluated sample.
    """
    mean_v = train_target.mean(axis=0)  # [H, N] complex
    pred = np.broadcast_to(mean_v[None, ...], reference.target_v.shape).copy()
    return _with_pred(reference, pred)


def persistence_prediction(
    seq_target: np.ndarray, reference: "Prediction"
) -> "Prediction":
    """Previous-step persistence for coherent sequence data.

    ``seq_target`` is the scenario-major sequence state ``[B, T, H, N]``; the estimate at step
    ``t`` is the true state at step ``t-1`` (the first step copies itself — a naive forecast has
    no earlier state). The result is folded back to ``[S = B*T, H, N]`` matching the sample
    order of :func:`injection_hse.evaluation.predict_dataset` (scenario-major, step-minor).
    """
    if seq_target.ndim != 4:
        raise ValueError(
            f"persistence needs a [B, T, H, N] sequence, got shape {seq_target.shape}."
        )
    shifted = np.empty_like(seq_target)
    shifted[:, 0] = seq_target[:, 0]  # no earlier step -> copy the first
    shifted[:, 1:] = seq_target[:, :-1]  # step t <- truth at t-1
    b, t, h, n = seq_target.shape
    pred = shifted.reshape(b * t, h, n)
    return _with_pred(reference, pred)


# =============================================================================
# The linear injection prior (a least-squares reference for the injection decode)
# =============================================================================
#: Diagonal loading of the ridge fit, in units of the (standardized) feature variance.
#: The feature covariance of a scarce placement is RANK-DEFICIENT — the observed rows span
#: far fewer directions than the injection space has — so a strictly positive loading is
#: what makes the closed-form solve well posed at all. The fitted estimate is insensitive to
#: its exact value over some ten decades (``1e-14`` … ``1e-4`` land within 1 % of each other
#: on the CIGRE LV benchmark); this default sits in the middle of that plateau, large enough
#: to regularize the unobserved null space and small enough to leave every direction the
#: measurements do resolve untouched.
LINEAR_PRIOR_RIDGE = 1e-9


def _optional_rows(state, key: str) -> Optional[np.ndarray]:
    """An optional int64 row index from a persisted state (``None`` when absent/empty)."""
    value = state.get(key) if hasattr(state, "get") else None
    if value is None:
        return None
    rows = np.asarray(value, dtype=np.int64).reshape(-1)
    return rows if rows.size else None


def _optional_text(state, key: str) -> Optional[str]:
    """An optional string entry of a persisted state (``None`` when absent)."""
    value = state.get(key) if hasattr(state, "get") else None
    if value is None:
        return None
    return str(np.asarray(value).item())


@dataclass(frozen=True)
class LinearInjectionPrior:
    """The fitted least-squares map from the observed measurements to the nodal injections.

    The coefficients of :func:`fit_linear_injection_prior`: one affine map from the
    standardized real measurement features to the real/imaginary parts of the nodal
    injection currents ``I(h) = Y(h)·V(h)`` at every non-slack row, over all harmonic
    orders at once. Pushing its prediction through the estimator's own decode
    (:func:`linear_injection_prediction`) yields the state estimate.

    ``rows`` are the predicted node-phase rows: every NON-slack row, since a slack row's
    entry of ``Y·V`` is the free boundary current the network solve never reads.
    ``feature_mean``/``feature_scale`` standardize the features the map is applied to (a
    feature that never varies keeps scale ``1`` and contributes nothing), and ``coef`` lives
    in that standardized space, so the map is applied as
    ``target_mean + ((features - feature_mean) / feature_scale) @ coef.T``. The bundle is
    plain arrays with no torch state, so :meth:`save` / :meth:`load` records it beside a run.

    ``observed_rows`` / ``observed_edge_rows`` record the FIXED placement the fit read — the
    observed node-phase rows and, for a current-measuring run, the metered half-edges. The
    map's input layout is that placement, so recording it makes the bundle self-describing:
    a consumer can rebuild the feature vector without being handed the fit's mask again
    (:func:`injection_hse.train.prior_injection_features`). ``observed_edge_rows`` is ``None`` for a
    run without measured currents; both are ``None`` for a bundle written before they were
    recorded.

    ``fit_dataset`` names the data the fit drew its states from (the dataset path and split,
    e.g. ``"<dataset>:train"``) when the fitting code recorded it; ``None`` otherwise.
    """

    orders: tuple  # the harmonic orders, in the module's order
    rows: np.ndarray  # int64 [R] (the predicted, non-slack node-phase rows)
    n_rows: int  # the grid's node-phase row count N
    feature_mean: np.ndarray  # float64 [F]
    feature_scale: np.ndarray  # float64 [F]
    target_mean: np.ndarray  # float64 [K], K = len(orders)*R*2
    coef: np.ndarray  # float64 [K, F] (in the standardized feature space)
    ridge: float  # the diagonal loading the fit used
    n_train: int  # training samples the fit consumed
    noisy: bool = False  # features drawn through the module's sensor-noise model
    observed_rows: Optional[np.ndarray] = None  # int64 [K] (the fitted placement)
    observed_edge_rows: Optional[np.ndarray] = None  # int64 [M] (metered half-edges)
    fit_dataset: Optional[str] = (
        None  # the data the fit drew from ("<dataset>:<split>")
    )

    @property
    def n_features(self) -> int:
        """Measurement features the map reads."""
        return int(self.coef.shape[1])

    @property
    def n_targets(self) -> int:
        """Real injection components the map predicts (``len(orders)·R·2``)."""
        return int(self.coef.shape[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Measurement features ``[S, F]`` -> nodal injection currents ``[S, H, N]`` (amperes).

        Rows the fit does not predict (the slack) stay exactly zero, which is what the
        network solve expects there: the slack voltage is a hard boundary condition, so its
        injection entry is never read.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(
                f"features must be [S, {self.n_features}], got {features.shape}."
            )
        z = (features - self.feature_mean) / self.feature_scale
        flat = self.target_mean + z @ self.coef.T  # [S, K]
        s = flat.shape[0]
        h, r = len(self.orders), int(self.rows.size)
        parts = flat.reshape(s, h, r, 2)
        out = np.zeros((s, h, self.n_rows), dtype=np.complex128)
        out[:, :, self.rows] = parts[..., 0] + 1j * parts[..., 1]
        return out

    def state_dict(self) -> dict:
        """The fitted map as plain arrays / scalars (the persisted form)."""
        state = {
            "orders": np.asarray(self.orders, dtype=np.int64),
            "rows": self.rows,
            "n_rows": np.int64(self.n_rows),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "target_mean": self.target_mean,
            "coef": self.coef,
            "ridge": np.float64(self.ridge),
            "n_train": np.int64(self.n_train),
            "noisy": np.bool_(self.noisy),
        }
        # optional (absent in a bundle fitted before the placement was recorded); ``.npz``
        # has no representation for ``None``, so an unset entry is simply not written
        if self.observed_rows is not None:
            state["observed_rows"] = self.observed_rows
        if self.observed_edge_rows is not None:
            state["observed_edge_rows"] = self.observed_edge_rows
        if self.fit_dataset is not None:
            state["fit_dataset"] = str(self.fit_dataset)
        return state

    @classmethod
    def from_state_dict(cls, state) -> "LinearInjectionPrior":
        """Rebuild from :meth:`state_dict` (or a loaded ``.npz`` archive)."""
        return cls(
            orders=tuple(int(o) for o in np.asarray(state["orders"]).tolist()),
            rows=np.asarray(state["rows"], dtype=np.int64),
            n_rows=int(state["n_rows"]),
            feature_mean=np.asarray(state["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(state["feature_scale"], dtype=np.float64),
            target_mean=np.asarray(state["target_mean"], dtype=np.float64),
            coef=np.asarray(state["coef"], dtype=np.float64),
            ridge=float(state["ridge"]),
            n_train=int(state["n_train"]),
            noisy=bool(state["noisy"]),
            observed_rows=_optional_rows(state, "observed_rows"),
            observed_edge_rows=_optional_rows(state, "observed_edge_rows"),
            fit_dataset=_optional_text(state, "fit_dataset"),
        )

    def with_fit_dataset(self, fit_dataset: Optional[str]) -> "LinearInjectionPrior":
        """The same map, recorded as fitted on ``fit_dataset`` (``"<dataset>:<split>"``)."""
        return replace(self, fit_dataset=fit_dataset)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the fitted map to a compressed ``.npz`` archive; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.state_dict())
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearInjectionPrior":
        """Read a fitted map back from :meth:`save`."""
        with np.load(Path(path)) as archive:
            return cls.from_state_dict({k: archive[k] for k in archive.files})


def _fixed_placement(mask: "Tensor") -> "Tensor":
    """The observed columns of a mask that is the SAME in every sample.

    The linear prior is a map with a fixed input layout, so it is defined for a fixed
    sensor placement (the deployed reading); a per-sample random mask has no such layout
    and is rejected rather than silently averaged over.
    """
    import torch

    always = mask.all(dim=0)
    if not bool(torch.equal(always, mask.any(dim=0))):
        raise ValueError(
            "the linear injection prior needs a FIXED sensor placement (the same observed "
            "rows in every sample); the given mask varies per sample."
        )
    return torch.nonzero(always, as_tuple=False).flatten()


def _reads_edges(module: "SEModule") -> bool:
    """Whether this run's measurement features include metered half-edge currents."""
    return getattr(module, "edge_op", None) is not None and bool(
        getattr(module.cfg.measurement, "measure_current", False)
    )


def measured_inputs(
    module: "SEModule", v: "Tensor", mask: "Tensor", *, noisy: bool = False
) -> "tuple[Tensor, Optional[Tensor], Optional[Tensor]]":
    """What this run's sensors report for ``(v [S, H, N], mask [S, N])``.

    Returns ``(v_meas [S, H, N], i_edges [S, H, E] | None, edge_mask [S, E] | None)``: the
    measured state (the true one, or a draw through the module's sensor-noise model when
    ``noisy``) and, for a current-measuring module, the metered half-edge currents in
    amperes with their observability mask — derived exactly as the training step derives
    them (from the true state, plus the CT error when ``noisy``), the one-CT budget of
    ``measurement.current_channels="upstream"`` included. This is
    the single construction of the estimator's measured inputs the linear prior is fitted
    and scored on, so its features and the anchors it decodes through read one realization.
    """
    from .attribution import _edge_features

    v_true = v.to(module._complex_dtype)
    v_meas = v_true
    if noisy:
        noise = getattr(module, "_noise", None)
        if noise is None:
            raise ValueError(
                "noisy measurements need a module with an enabled sensor-noise model "
                "(measurement noise is off in this run's config)."
            )
        v_meas = noise.apply(v_true)
    # a CT reads the TRUE branch current with its own error: derive from the true state,
    # never from the noisy voltages (that would amplify the voltage error across short
    # branches), then draw the CT error
    _, edge_mask, i_edges = _edge_features(module, v, mask)  # at v's own precision
    if noisy and i_edges is not None:
        edge_noise = getattr(module, "_edge_noise", None)
        if edge_noise is not None:
            i_edges = edge_noise.apply(i_edges)
    return v_meas, i_edges, edge_mask


def linear_prior_features(
    module: "SEModule",
    v_meas: "Tensor",
    mask: "Tensor",
    *,
    i_edges: Optional["Tensor"] = None,
    edge_mask: Optional["Tensor"] = None,
) -> "Tensor":
    """The estimator's own measurements as real regression features ``[S, F]`` (float64).

    Exactly the information the network reads at the observed locations, resolved through
    the module's own seams so the baseline can never see more than the estimator does: the
    observed voltage rows of the MEASURED state ``v_meas`` in the run's feature frame
    (per-unit under :attr:`~injection_hse.config.NormalizationConfig.per_unit`) and — for a
    current-measuring module — the metered half-edge currents over their per-unit base, each
    split into real and imaginary parts, order-major. Pass the ``i_edges`` / ``edge_mask``
    of :func:`measured_inputs` so the features and the anchors read the same realization;
    without them the currents are re-derived from ``v_meas`` noise-free.

    The parameterization is Cartesian rather than the model's trig/log encoding: the encoding
    is an invertible per-row reparameterization of the same measurements, and the Cartesian
    one makes the baseline a property of the SETTING (grid, placement, anchoring) instead of
    a property of :class:`~injection_hse.config.EncodingConfig`. The physics-derived input priors
    (:attr:`~injection_hse.config.PhysicsConfig.input_priors`) are deliberately omitted — they are
    deterministic functions of these same measurements, so they add nothing a linear map
    could use.
    """
    import torch

    rows = _fixed_placement(mask)
    v_meas = v_meas.to(module._complex_dtype)
    blocks = [module._to_feature_frame(v_meas).index_select(-1, rows)]  # [S, H, K]
    if _reads_edges(module):
        if i_edges is None or edge_mask is None:
            _, i_edges, edge_mask = measured_inputs(module, v_meas, mask)
        base = module._edge_current_base()
        if base is not None:
            i_edges = i_edges / base.to(device=i_edges.device, dtype=i_edges.real.dtype)
        blocks.append(i_edges.index_select(-1, _fixed_placement(edge_mask)))

    # order-major real layout: Re/Im of every block at order 1, then order 2, ...
    parts = [
        torch.cat([b.select(-2, k).real, b.select(-2, k).imag], dim=-1)
        for k in range(module.n_harmonics)
        for b in blocks
    ]
    return torch.cat(parts, dim=-1).to(torch.float64)


def fit_linear_injection_prior(
    module: "SEModule",
    v: "Tensor",
    mask: "Tensor",
    *,
    ridge: float = LINEAR_PRIOR_RIDGE,
    noisy: bool = False,
    device: str = "cpu",
    batch_size: int = 256,
    cross_order: bool = True,
) -> LinearInjectionPrior:
    """Fit the least-squares injection prior on ``(v [S, H, N], mask [S, N])``.

    The closed-form ridge regression from the estimator's own measurement features
    (:func:`linear_prior_features`) onto the true nodal injections ``I(h) = Y(h)·V(h)`` at
    every non-slack row, all harmonic orders at once. It is the conditional-mean injection
    under a linear model, and — because the measurement-anchored network solve is affine in
    the injection at fixed measurements — pushing it through that solve gives the
    corresponding conditional-mean STATE. That makes it the honest reference for an
    injection-decode estimator: a network that does not beat it has learned nothing a
    closed-form regression on its own inputs could not.

    Every non-slack row is predicted, not only the rows an appliance sits on: a row that
    carries no device has a target that is zero to the solve's own roundoff, so the fit gives
    it a proportionally negligible prediction on its own. The device-row restriction is
    implied by the data rather than imposed, and no appliance bookkeeping is needed.

    Deterministic (no sampling, no iteration); with ``noisy`` the module's noise streams are
    reseeded from the run's own seed first, so a fit reproduces. The measurement features
    are held once as a standardized ``[S, F]`` float64 block and the target cross moments
    accumulate in float64 over chunks of ``batch_size`` samples, so beyond ``v`` itself the
    peak memory is ``S·F`` plus the ``[K, F]`` coefficient block. Fit on the TRAIN split — scoring the prior on the
    samples it was fitted on reports its training error, not its accuracy. Requires the
    injection decode, a single-grid module and a FIXED placement (the same observed rows in
    every sample — the map has a fixed input layout).

    The features are standardized, so ``ridge`` (default :data:`LINEAR_PRIOR_RIDGE`) is a
    diagonal loading in units of feature variance. The feature covariance of a scarce
    placement is rank-deficient; a positive loading is what makes the solve well posed.

    ``cross_order=False`` restricts the map to SAME-ORDER features: the injections of order
    ``h`` are regressed on the order-``h`` measurements only (the cross-order blocks of
    ``coef`` are exactly zero). It is the per-order regression a model without any
    cross-order coupling could reach; comparing it with the default (every order's
    features feed every order's injections) measures how much the data's cross-order
    structure is worth to a linear map. The diagonal loading is taken from the whole
    feature set in both cases, so the two fits differ only in the coupling.
    """
    import torch

    if not getattr(module, "_predict_injection_state", False):
        raise ValueError(
            "the linear injection prior needs the injection decode "
            "(physics.decode='injection')."
        )
    if getattr(module, "_multigrid", False):
        raise ValueError(
            "the linear injection prior is a single-grid baseline; a multi-grid module "
            "serves one operator set per member."
        )
    dev = torch.device(device)
    module.to(dev).eval()
    v, mask = v.to(dev), mask.to(dev)
    n_samples = int(v.shape[0])
    if n_samples == 0:
        raise ValueError(
            "the linear injection prior needs at least one training sample."
        )
    if noisy:
        seed = int(module.cfg.noise.seed)
        module._noise.reseed(seed + _LINEAR_PRIOR_NOISE_OFFSET)
        if getattr(module, "_edge_noise", None) is not None:
            module._edge_noise.reseed(seed + _LINEAR_PRIOR_NOISE_OFFSET + 7)

    module._sync_solver_device(dev)
    # the FIXED placement the map's input layout is: recorded on the bundle so a consumer
    # can rebuild the feature vector from the placement alone (the distillation objective
    # reads the prior's own sensors, whatever mask a curriculum stage happens to draw)
    obs_rows = _fixed_placement(mask)
    solver = module.solver
    keep = torch.ones(int(module.n_nodes), dtype=torch.bool, device=dev)
    if solver.slack_rows.numel():
        keep = keep.index_fill(0, solver.slack_rows.to(dev), False)
    tied = getattr(module, "_tied_rows", None)
    if tied is not None and tied.numel():
        # electrically the slack (physics.exclude_tied_rows): the map predicts nothing there
        keep = keep.index_fill(0, tied.to(dev), False)
    rows = torch.nonzero(keep, as_tuple=False).flatten()

    with torch.no_grad():
        chunks = []
        obs_edge_rows = None
        for a in range(0, n_samples, batch_size):
            vb, mb = v[a : a + batch_size], mask[a : a + batch_size]
            v_meas, i_edges, edge_mask = measured_inputs(module, vb, mb, noisy=noisy)
            if edge_mask is not None and _reads_edges(module):
                obs_edge_rows = _fixed_placement(edge_mask)
            chunks.append(
                linear_prior_features(
                    module, v_meas, mb, i_edges=i_edges, edge_mask=edge_mask
                )
            )
        feats = torch.cat(chunks)  # [S, F]
        del chunks  # one [S, F] block alive from here on (the fit's peak is S-linear)
        mean = feats.mean(dim=0)
        scale = feats.std(dim=0, unbiased=False)
        # a feature that never varies carries no information; dividing by its (numerically
        # noisy) zero spread would manufacture some, so it keeps unit scale and stays ~0.
        floor = 1e-12 * scale.max().clamp_min(1e-300)
        scale = torch.where(scale > floor, scale, torch.ones_like(scale))
        z = feats.sub_(mean).div_(scale)  # standardized in place
        cyy = (z.T @ z) / n_samples  # [F, F]

        n_features = int(z.shape[1])
        n_targets = module.n_harmonics * int(rows.numel()) * 2
        x_sum = torch.zeros(n_targets, dtype=torch.float64, device=dev)
        sxy = torch.zeros((n_targets, n_features), dtype=torch.float64, device=dev)
        for a in range(0, n_samples, batch_size):
            zb = z[a : a + batch_size]
            i_true = (
                solver.injection_targets(v[a : a + batch_size])
                .to(torch.complex128)
                .index_select(-1, rows)
            )  # [b, H, R]
            xb = torch.stack([i_true.real, i_true.imag], dim=-1).reshape(
                zb.shape[0], n_targets
            )
            x_sum += xb.sum(dim=0)
            sxy += xb.T @ zb
        target_mean = x_sum / n_samples
        # the features are centered, so the raw cross moment IS the cross covariance
        cxy = sxy / n_samples  # [K, F]
        loading = float(ridge) * float(torch.diagonal(cyy).mean())
        if not cross_order:
            # block-diagonal normal equations: features and targets are order-major, so
            # order k owns one contiguous block of each; zeroing the cross-order blocks
            # solves every order's regression on its own features in one system
            n_orders = module.n_harmonics
            f_order = torch.arange(n_features, device=dev) // (n_features // n_orders)
            k_order = torch.arange(n_targets, device=dev) // (n_targets // n_orders)
            cyy = cyy * (f_order[:, None] == f_order[None, :])
            cxy = cxy * (k_order[:, None] == f_order[None, :])
        cyy = cyy + loading * torch.eye(n_features, dtype=cyy.dtype, device=dev)
        coef = torch.linalg.solve(cyy, cxy.T).T  # [K, F]

    return LinearInjectionPrior(
        orders=tuple(int(o) for o in module.orders),
        rows=rows.cpu().numpy().astype(np.int64),
        n_rows=int(module.n_nodes),
        feature_mean=mean.cpu().numpy(),
        feature_scale=scale.cpu().numpy(),
        target_mean=target_mean.cpu().numpy(),
        coef=coef.cpu().numpy(),
        ridge=float(ridge),
        n_train=n_samples,
        noisy=bool(noisy),
        observed_rows=obs_rows.cpu().numpy().astype(np.int64),
        observed_edge_rows=(
            None
            if obs_edge_rows is None
            else obs_edge_rows.cpu().numpy().astype(np.int64)
        ),
    )


def linear_prior_injections(
    module: "SEModule",
    prior: LinearInjectionPrior,
    v_meas: "Tensor",
    mask: "Tensor",
    *,
    i_edges: Optional["Tensor"] = None,
    edge_mask: Optional["Tensor"] = None,
) -> "Tensor":
    """The prior's nodal injection currents ``[S, H, N]`` for one batch of measurements.

    The prior's own feature construction on the MEASURED inputs
    (:func:`measured_inputs`), then its affine map, cast to the module's complex training
    dtype — the injection the decode is fed. Slack rows stay exactly zero.
    """
    import torch

    feats = linear_prior_features(
        module, v_meas, mask, i_edges=i_edges, edge_mask=edge_mask
    )
    i_pred = prior.predict(feats.detach().cpu().numpy())
    return torch.as_tensor(i_pred, device=v_meas.device).to(module._complex_dtype)


def linear_injection_prediction(
    module: "SEModule",
    prior: LinearInjectionPrior,
    reference: "Prediction",
    *,
    device: str = "cpu",
    batch_size: int = 8,
) -> "Prediction":
    """Score a fitted prior through the estimator's DEPLOYED decode -> a ``Prediction``.

    The predicted injections are pushed through the module's own network solve at the
    measured slack, with the deployed measurement anchoring and (for a current-measuring
    module) the metered branch currents — the identical path the learned estimate takes, so
    the only difference between this row and the model's is where the injections came from.
    ``reference`` supplies the evaluated samples, their placement mask and the metric
    metadata; as for the other baselines only ``pred_v`` is replaced.
    """
    import torch

    from injection_hse.physics import solve_injection

    dev = torch.device(device)
    module.to(dev).eval()
    v = torch.as_tensor(reference.target_v).to(dev)
    mask = torch.as_tensor(reference.mask).to(dev)
    preds = []
    with torch.no_grad():
        for a in range(0, v.shape[0], batch_size):
            vb, mb = v[a : a + batch_size], mask[a : a + batch_size]
            v_meas, i_edges, edge_mask = measured_inputs(
                module, vb, mb, noisy=prior.noisy
            )
            i_pred = linear_prior_injections(
                module, prior, v_meas, mb, i_edges=i_edges, edge_mask=edge_mask
            )
            preds.append(
                solve_injection(
                    module,
                    i_pred,
                    v_meas,
                    mask=mb,
                    edge_mask=edge_mask,
                    i_edges=i_edges,
                )
                .cpu()
                .numpy()
            )
    return _with_pred(reference, np.concatenate(preds, axis=0))


# =============================================================================
# Driver: score every baseline on the same split as the model
# =============================================================================
def collect_baseline_metrics(
    source_or_path: Union[str, Path, "DataSource"],
    *,
    grid: "Grid",
    measurement: "MeasurementConfig",
    split: str = "val",
    training_split: str = "train",
    device: str = "cpu",
    baselines: Sequence[str] = (
        "zero_harmonic",
        "training_mean",
        "persistence",
        "linear_injection",
    ),
    module: Optional["SEModule"] = None,
    ridge: float = LINEAR_PRIOR_RIDGE,
    n_prior_samples: Optional[int] = None,
) -> "pl.DataFrame":
    """Score the non-learned baselines on a dataset split -> one metric row per baseline.

    Loads the target state of ``split`` under the sensor placement (the SAME fixed observed rows
    a placement-finetuned model is evaluated on) and evaluates each requested baseline with
    :func:`score_prediction`, so the rows share the ablation/report columns (``variant`` /
    ``n_params`` / ``epochs`` / ``mae`` / ``rmse`` / ``mae_fund`` / ``mae_harm`` / ``rel_harm``
    / ``thd_mae``). Non-learned rows carry ``n_params = 0`` and ``epochs = 0``. The variant
    names are prefixed ``baseline_`` so they never collide with a learned variant.

    A baseline whose requirements this dataset or run does not meet is skipped without a row:
    persistence on a snapshot dataset (it needs the time axis), and the linear injection prior
    without a ``module`` or on a module that does not use the injection decode. ``module`` is
    a trained (or merely built) :class:`~injection_hse.train.SEModule` — the linear prior reads only its
    measurement seams, its network solver and its deployed anchor weights, never its weights,
    so any module of the run defines the decode the prior is scored through. It is fitted on
    ``training_split`` (``n_prior_samples`` snapshots, default all of it) with diagonal loading
    ``ridge``.
    """
    import polars as pl

    from injection_hse.data import MeasurementModel, base_voltage_per_row, build_data_source
    from injection_hse.evaluation import Prediction

    from pgml.assembly import node_phase_index

    if isinstance(source_or_path, (str, Path)):
        from injection_hse.config import DataConfig

        source = build_data_source(
            DataConfig(source="parquet", dataset_path=str(source_or_path)), grid=grid
        )
    else:
        source = source_or_path

    mm = MeasurementModel.resolve(
        grid,
        measurement.voltage_nodes,
        fundamental_nodes=measurement.fundamental_nodes,
        slack_known=measurement.slack_known,
    )
    observed_rows = mm.observed_rows(grid)
    slack_rows = mm.slack_rows(grid).cpu().numpy().astype(np.int64)

    idx = node_phase_index(grid)
    base_v = base_voltage_per_row(grid).detach().cpu().numpy().astype(np.float64)

    # -- load the split target (fold time into the sample axis, keep the seq for persistence) --
    seq = _load_sequence(source, split)  # [B, T, H, N] or None
    target = _load_split_target(source, split)  # [S, H, N]
    n = target.shape[-1]
    mask = np.zeros((target.shape[0], n), dtype=bool)
    mask[:, observed_rows.cpu().numpy().astype(np.int64)] = True

    reference = Prediction(
        target_v=target,
        pred_v=np.zeros_like(target),
        mask=mask,
        orders=list(source.orders),
        base_v=base_v,
        node_ids=idx.node_ids.cpu().numpy().astype(np.int64),
        phase_codes=idx.phase_codes.cpu().numpy().astype(np.int64),
        observed_rows=observed_rows.cpu().numpy().astype(np.int64),
        sensor_names=list(measurement.voltage_nodes),
        grid=grid,
    )

    records: list[dict] = []
    for name in baselines:
        if name == "zero_harmonic":
            pred = zero_harmonic_prediction(reference, slack_rows=slack_rows)
        elif name == "training_mean":
            train_target = _load_split_target(source, training_split)
            pred = training_mean_prediction(reference, train_target)
        elif name == "persistence":
            if seq is None:
                continue  # snapshot data has no previous step
            pred = persistence_prediction(seq, reference)
        elif name == "linear_injection":
            if module is None or not getattr(module, "_predict_injection_state", False):
                continue  # no injection decode to push a predicted injection through
            import torch

            train_target = _load_split_target(source, training_split)
            if n_prior_samples is not None:
                train_target = train_target[: int(n_prior_samples)]
            train_v = torch.as_tensor(train_target)
            train_mask = torch.zeros(
                (train_v.shape[0], n), dtype=torch.bool, device=train_v.device
            )
            train_mask[:, observed_rows.cpu()] = True
            prior = fit_linear_injection_prior(
                module, train_v, train_mask, ridge=ridge, device=device
            )
            pred = linear_injection_prediction(module, prior, reference, device=device)
        else:
            raise ValueError(f"unknown baseline {name!r}.")
        row = {"variant": f"baseline_{name}", "n_params": 0, "epochs": 0}
        row.update(score_prediction(pred))
        records.append(row)
    return pl.DataFrame(records)


def _load_split_target(source: "DataSource", split: str) -> np.ndarray:
    """Complex target state of a split, folded to ``[S, H, N]`` (scenario-major, step-minor)."""
    import torch

    ds = _split_dataset(source, split)
    vs = [torch.as_tensor(ds[i]["v"]) for i in range(len(ds))]  # each [H,N] or [T,H,N]
    if not vs:
        raise ValueError(f"split {split!r} is empty (no scenarios).")
    stacked = torch.stack(vs)  # [B, H, N] or [B, T, H, N]
    if stacked.dim() == 4:  # [B, T, H, N] -> [B*T, H, N]
        b, t, h, nn = stacked.shape
        stacked = stacked.reshape(b * t, h, nn)
    return stacked.detach().cpu().numpy()


def _load_sequence(source: "DataSource", split: str) -> Optional[np.ndarray]:
    """The scenario-major sequence ``[B, T, H, N]`` of a split, or ``None`` for snapshot data."""
    import torch

    if source.n_steps <= 1:
        return None
    ds = _split_dataset(source, split)
    vs = [torch.as_tensor(ds[i]["v"]) for i in range(len(ds))]  # each [T, H, N]
    if not vs:
        return None
    return torch.stack(vs).detach().cpu().numpy()  # [B, T, H, N]


def _split_dataset(source: "DataSource", split: str):
    if split == "train":
        return source.train_dataset()
    if split == "val":
        return source.val_dataset()
    if split == "test":
        return source.test_dataset()
    raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}.")


__all__ = [
    "LINEAR_PRIOR_RIDGE",
    "LinearInjectionPrior",
    "score_prediction",
    "zero_harmonic_prediction",
    "training_mean_prediction",
    "persistence_prediction",
    "measured_inputs",
    "linear_prior_features",
    "fit_linear_injection_prior",
    "linear_prior_injections",
    "linear_injection_prediction",
    "collect_baseline_metrics",
]
