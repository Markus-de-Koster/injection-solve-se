"""Experiment configuration for injection_hse — the reproducible source of truth.

One :class:`ExperimentConfig` (+ its ``seed``) fully determines a training run: the data
source, model, state encoding, normalization, loss/weighting, measurement model, masking
curriculum, optimizer, and trainer. It is plain pydantic (serializes to YAML/JSON), logged
with the run and embedded with checkpoints — mirroring ``pgml.scenarios.ScenarioConfig`` so
"one config + seed reproduces the experiment" holds across the dev box and the cluster.

The rationale for every choice here is in ``docs/pgl/index.md`` (read it first).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# State encoding (swappable: trigonometric default, Cartesian alternative)
# =============================================================================
class EncodingConfig(_Base):
    """Split-real encoding of the complex node-phase-harmonic state.

    ``"trig"`` -> ``(m, cosθ, sinθ)`` (3 channels): physical magnitude + wrap-free
    direction (default). ``"cartesian"`` -> ``(re, im)`` (2 channels). The loss is always
    computed on the Cartesian reconstruction (so a near-zero harmonic's meaningless angle is
    not penalized); see :mod:`injection_hse.encoding`.

    ``injection_kind`` (default ``None``) gives the INJECTION head of the physics decode
    (``PhysicsConfig.decode="injection"``) an encoding of its own; ``None`` means the head
    shares ``kind``. The head predicts the nodal injection currents ``I(h) = Y(h)·V(h)``, a
    map that is LINEAR in Cartesian coordinates and strongly nonlinear in the polar
    ``(|I|, cosθ, sinθ)`` parameterisation with its ``log1p``-standardised magnitude — where a
    zero-injection row additionally needs an exact zero magnitude.
    ``injection_kind="cartesian"`` lets the head emit per-order standardised ``(re, im)``
    currents, so the map the network must represent is linear, while the voltage INPUT keeps
    ``kind`` (whose
    magnitude channel isolates the small voltage deviations). The injection encoding is used
    wherever predicted or target injections are encoded or decoded — the injection normalizer
    fit, the decode of the model output, the direct injection term, the checkpoint rebuild —
    and the model's OUTPUT width follows it (``H × n_features`` of the injection encoding)
    while its input width follows ``kind``. ``None`` is byte-identical; under the voltage
    decode the field is inert.

    ``time_features`` (default OFF) adds a per-step PERIODIC time feature vector (sin/cos of
    time-of-day, day-of-week, day-of-year, plus a validity flag — :mod:`injection_hse.time_features`)
    to every node's model input, so the estimator can condition on the diurnal / weekly /
    seasonal structure of a PROFILED coherent dataset (the absolute ``time_unix_s`` stamped
    by :mod:`injection_hse.data.scenarios.profiles`). A dataset WITHOUT timestamps (the randomized snapshot
    stage, or a non-profiled coherent sequence) supplies the ALL-ZERO vector with
    ``valid = 0`` — the SAME input contract — so one model trains across timeless snapshots
    and timestamped sequences. OFF is byte-identical (no extra input channels).
    """

    kind: Literal["trig", "cartesian"] = "trig"
    injection_kind: Optional[Literal["trig", "cartesian"]] = None
    time_features: bool = False


# =============================================================================
# Structural (graph) node encoding — the cross-grid positional signal
# =============================================================================
class StructuralEncodingConfig(_Base):
    """Structural node features of the network graph (:mod:`injection_hse.data.encodings`).

    A message-passing estimator distinguishes two rows only by what their features and their
    neighbourhood say; on ONE grid the base features (:func:`injection_hse.data.build_graph`) suffice,
    but across grids of different size and voltage class the grid-ABSOLUTE ones (rated
    voltage, slack distance in km) invite memorization instead of generalization. This config
    controls both halves of the fix.

    ``base`` selects the base block: ``"absolute"`` (default) is today's layout — the slack
    distance in kilometres, byte-identical for single-grid runs; ``"normalized"`` divides
    that depth by the grid's own maximum, so column ``[6]`` reads "how far down the feeder"
    in ``[0, 1]`` on every grid.

    ``kind`` selects the structural encoding appended to the base block:

    - ``"none"`` (default) — no extra columns.
    - ``"rwse"`` — the ``K``-step RANDOM-WALK structural encoding: the return probabilities
      ``diag((D^-1 A)^k)`` for ``k = 1 … rwse_steps``. A sign-free, purely topological
      positional signal (how enclosed a row is at each scale), cheap on the sparse
      node-phase graph.
    - ``"lappe"`` — ``lap_pe_dim`` low-frequency non-trivial eigenpairs of the symmetric
      normalized graph Laplacian.  Each pair contributes ``(eigenvalue, eigenvector entry)``;
      a Graphormer with matching ``spectral_pe_dim`` learns the SAN-style positional code.
    - ``"electrical"`` — the effective impedance to the slack, split into
      ``(R_eff, X_eff)`` at the base frequency, per-unit-normalized by the grid's own
      maximum. Effective resistance IS electrical distance on a grid: it collapses parallel
      paths the way the network does, and it is the quantity a voltage drop responds to.

    ``rwse_steps`` is the walk length ``K``. ``lap_pe_dim`` is the number of Laplacian
    eigenpairs (two columns each). ``normalize`` (electrical only) divides each column by
    the grid's maximum finite value so the feature is comparable across voltage classes and
    grid sizes.
    """

    kind: Literal["none", "rwse", "lappe", "electrical"] = "none"
    base: Literal["absolute", "normalized"] = "absolute"
    rwse_steps: int = Field(default=16, ge=1, le=64)
    lap_pe_dim: int = Field(default=16, ge=1, le=64)
    normalize: bool = True


# =============================================================================
# Phase coupling — the mutual-admittance edges of the message-passing graph
# =============================================================================
class PhaseCouplingConfig(_Base):
    """Whether the graph carries the network's PHASE-TO-PHASE admittance couplings.

    :func:`injection_hse.data.build_graph` draws one edge per branch and phase INDEX — phase ``a`` of
    the from-node to phase ``a`` of the to-node, ``b`` to ``b``, ``c`` to ``c``. The physics
    the estimator must invert does not factor that way: a line's series impedance is a full
    per-phase MATRIX (the conductors' mutual inductance and resistance), its shunt
    capacitance likewise, and a Dyn transformer's winding incidence mixes the phases across
    the winding. The resulting ``Y(h)`` therefore carries off-diagonal phase blocks that the
    per-phase-index edge set has no path for, and on a three-phase feeder with no
    phase-mixing appliance the message-passing graph falls apart into one disconnected
    component per phase — a cross-phase dependency is then unreachable at ANY depth.

    ``enabled`` (default ``False``) adds an edge for every such coupling, taken from the
    SAME primitive admittance stamps the Y-bus assembly scatters
    (``pgml.assembly.branch_stamp_blocks``), so the edge set is the sparsity pattern of the
    passive network admittance rather than a heuristic: the same-node phase-to-phase entries
    of every incident branch (the mutual impedance seen at a terminal, the delta winding of a
    transformer) and the cross-node cross-phase entries along a branch (the mutual impedance
    along the conductor bundle, the phase mixing across a Dyn winding). The added half-edges
    follow the series ones, carry their branch's provenance, and are flagged in two extra
    ``edge_attr`` columns so a model can learn a different message for a coupling than for a
    series connection. ``False`` is byte-identical to a graph built before the field existed.

    ``tolerance`` is the RELATIVE magnitude below which a stamp entry counts as structurally
    absent: an entry ``Y[p, q]`` is drawn when ``|Y[p, q]| > tolerance * sqrt(|Y[p, p]| *
    |Y[q, q]|)`` within its branch's primitive block. The default is small enough to keep
    every genuinely stamped coupling and large enough to reject float round-off; raise it to
    keep only the strong couplings (a real coupling on a distribution line sits at ``1e-1``).

    The edge-feature WIDTH follows this config, not the grid: a grid whose impedance matrices
    happen to be diagonal still yields the wider ``edge_attr`` (all-zero in the two extra
    columns), so every member of a multi-grid corpus and every checkpoint of a run agree on
    the model's input width.
    """

    enabled: bool = False
    tolerance: float = Field(default=1e-9, gt=0.0, lt=1.0)


# =============================================================================
# Normalization (per-harmonic, magnitude channel) + per-harmonic loss weighting
# =============================================================================
class NormalizationConfig(_Base):
    """Per-harmonic normalization, fit on the train split and persisted with the model.

    Standardizes (z-score) per harmonic order so every order is numerically perceptible
    regardless of absolute scale. ``magnitude_transform="log1p"`` first log-compresses the
    heavy-tailed magnitudes. For the trig encoding this applies to the magnitude channel;
    ``cosθ/sinθ`` are passed through. The same fitted per-harmonic statistics also
    standardize the Cartesian residual in the loss so harmonics stay comparable / learnable.

    ``per_unit`` (default OFF) is the FEATURE FRAME of a single-grid run: the complex state
    is divided by the per-row line-to-neutral base voltage
    (:func:`injection_hse.data.base_voltage_per_row`) before encoding and multiplied back after
    decoding, and the currents the injection decode predicts are expressed over the matching
    per-unit current base ``i_base(h, r) = s(h)·V_ref² / V_base(r)``
    (:meth:`injection_hse.physics.NetworkSolver.i_base` — a power base per row voltage, so the current
    base scales INVERSELY with the voltage class). Only the frame changes — the physics still
    solves, anchors and reports in absolute units, and the validation metrics stay the
    per-unit MAE they always were.

    Why it matters: a feeder's rows do not share one voltage class (an LV population beside
    the MV rows above the transformer), so a per-order statistic fit in ABSOLUTE volts is set
    by the bimodal ROW population rather than by the variation between scenarios. The
    fundamental's per-scenario signal then occupies a few percent of the feature channel's
    scale, which makes it log-slow to learn and lets a direct-voltage decode collapse onto
    the population mean. In per-unit every row sits near 1 and the standardization sees the
    scenario variation it is meant to whiten. The multi-grid path already lives in this frame
    (:attr:`MultiGridConfig.per_unit`), which is why ``per_unit=True`` is rejected there.

    ``False`` (default) is byte-identical to a run written before the field existed.
    """

    per_harmonic: bool = True
    per_unit: bool = False
    magnitude_transform: Literal["log1p", "none"] = "log1p"


class HarmonicWeightConfig(_Base):
    """Per-harmonic LOSS weight — the priority knob (independent of normalization).

    Normalization makes every order *visible*; this weight sets each order's *priority* in
    the loss. ``"physical"`` (default): ``w_h ∝`` the order's typical magnitude/variance from
    train stats, so the fundamental dominates and high orders are down-weighted but NOT
    zeroed (still learnable when diagnostic). ``"uniform"``: equal weight. ``"custom"``:
    explicit ``values`` aligned to the solved harmonic ``orders``.
    """

    mode: Literal["physical", "uniform", "custom"] = "physical"
    values: Optional[list[float]] = None


class LossConfig(_Base):
    """SE loss: per-harmonic-weighted MSE on the per-harmonic-standardized Cartesian state.

    ``magnitude_term_weight`` optionally adds an explicit ``|V|`` term so drops/swells are
    learned crisply (0 disables it).

    ``predict_injection`` (default OFF) enables the per-device injection / localization head:
    an auxiliary output predicting the clean, non-superposed per-device injected
    harmonic-magnitude spectrum (the "where did the disturbance originate" target). When ON,
    ``injection_loss_weight`` scales its MSE term added to the node-V supervised loss; the
    injection magnitudes are EN 50160-relative (~``[0, 1]``) so the term is already
    well-scaled. ``injection_hidden_dim`` sizes the shared head's MLP. With the head OFF the
    node-V state estimation is byte-identical (the head is not constructed).

    ``injection_state_loss_split`` supervises the NODAL INJECTION STATE — a different
    quantity from the localization head above. The injection decode
    (``PhysicsConfig.decode="injection"``) predicts a nodal injection current per node-phase
    ROW and the network solve turns it into voltages; by default only those solved voltages
    are supervised, so the injection is learned indirectly through the solve. This split adds
    a SECOND supervised term directly on the predicted injections against the true ones
    ``I(h) = Y(h)·V_true(h)``, in the normalized injection space with the same per-harmonic
    weighting and Cartesian MSE structure as the voltage term (both are dimensionless
    relative-error means, so the blend is scale-meaningful)::

        loss = (1 - split)·L_voltage + split·L_injection

    ``0.0`` (default) is the voltage-only supervision, byte-identical. The term needs the
    injection decode (there is no predicted injection otherwise) and is ignored with a
    warning under ``decode="voltage"``. Distinguish the two knobs: ``injection_loss_weight``
    weights the per-DEVICE localization head, ``injection_state_loss_split`` splits the state
    supervision between voltages and the per-ROW injection currents.

    ``injection_state_loss_normalize`` (default ON) is what makes that split MEAN what it
    says. The two terms are dimensionless but not commensurate: the injection residual runs
    orders of magnitude above the voltage residual (measured: 82x to 45000x, and the ratio
    itself moves by ~500x across curriculum stages), so a raw blend at a nominal ``0.1``
    realizes a 90-99.98 % weight on the injection term — the split behaves as an on/off
    switch rather than a dial. With normalization the injection term is rescaled onto the
    voltage term's running scale (the ratio of their exponential moving averages, detached
    and re-seeded per stage) before blending, so ``s`` is the fraction of the loss the
    injection term actually contributes and the voltage term keeps its absolute scale — the
    learning rate and gradient clipping mean the same as at ``split=0``. Set to ``False`` to
    reproduce a pre-normalization run. The realized fraction is logged as
    ``train/val_injection_share`` either way; read it rather than trusting ``s``.

    ``plain_decode_weight`` (default ``0.0``) supervises the UNANCHORED decode beside the
    anchored one. With measurement anchoring on (``PhysicsConfig.anchor_voltage_weight`` /
    ``anchor_current_weight``, or a per-stage schedule) the voltage loss is evaluated on the
    anchored solve, which is flat in the predicted injections wherever the measurements pin
    the state: the network receives no gradient there and its injections drift while the run
    reports the solve's accuracy. A positive weight adds the SAME voltage loss on the PLAIN
    (slack-only) decode of the same predicted injections — ``‖Z(Î − I_true)‖²``, the
    observability-weighted injection error — so the network is supervised as a prior whatever
    the anchored output does; the deployed and monitored estimate stays the anchored one::

        loss = blended + plain_decode_weight·L_plain

    (``blended`` is the voltage term after the optional injection-state split above.) The
    term only exists while the solver anchors: without anchoring the plain decode IS the
    decode, its loss is the voltage term already in ``blended``, and no second solve is run.
    ``0.0`` is byte-identical (no extra solve, no extra graph). Needs the injection decode.
    The plain term is logged as ``train/val_plain_decode_loss`` beside the other components,
    and validation additionally reports ``val_pu_mae_unobs_plain`` (+ ``_h<order>``), the
    unobserved per-unit MAE of the plain decode — what the network itself achieves beside the
    anchored estimate.

    ``prior_distillation_weight`` (default ``0.0``) supervises the predicted nodal injections
    against a fitted LINEAR INJECTION PRIOR's predictions instead of against the truth — the
    closed-form least-squares map from this run's own measurements onto ``I(h) = Y(h)·V(h)``
    (:func:`injection_hse.analysis.fit_linear_injection_prior`), fitted on the TRAIN split of the pass
    before it trains (``PhysicsConfig.injection_prior_fit_states`` states) at the deployed
    placement and carried in the checkpoint, so a reload distils against the same teacher::

        loss = blended + prior_distillation_weight·L_distill

    The term exists because neither of the other two objectives builds the injection map: the
    measurement-anchored solve is FLAT in the injection directions the measurements already
    determine (so the voltage term carries no gradient there), while the true injections of
    ``injection_state_loss_split`` are not identifiable from a sparse sensor set (so that
    label asks for information the data does not contain). The prior's label is by
    construction a function of the measurements, and it is a strong one — on the CIGRE LV
    benchmark the prior itself estimates the unobserved fundamental state 2.4x better than
    the zero-injection solve. It is a STEPPING STONE, not the target: the prior has its own
    floor, and the physics objective is what a deployment is judged on, so
    :attr:`MaskStageConfig.prior_distillation_weight` anneals the term away per stage (a
    large weight while the map is being built, ``0`` for the stage that finishes the run).

    The term shares the injection loss's structure and scale, and that residual is orders of
    magnitude above the voltage one, so a weight of ``1`` already makes distillation the
    dominant part of the objective — read the logged ``{train,val}_prior_distillation_loss``
    component rather than treating the weight as a share. ``0.0`` is byte-identical (no prior
    is fitted and no teacher is evaluated). Needs the injection decode and a single grid (the
    prior is one grid's map; a multi-grid run rejects a non-default value).

    ``preconditioner`` (default ``"none"``) preconditions the GRADIENT of the loss with
    respect to the injection head's output by the inverse of the loss's own curvature
    (:mod:`injection_hse.train.preconditioner`). At fixed measurements the loss is an exact quadratic
    in that output whose Hessian ``CᵀGᵀWᵀWGC`` is grid-constant and badly conditioned
    (measured on CIGRE LV: κ ≈ 100 at the 1 % cutoff in every order, 30–43 % of the
    directions flat), so descent resolves the few steep directions and leaves the rest,
    however long it runs. The map is built once before training by finite differences on
    the module's own decode — ``"placement"`` from the deployed measurement-anchored decode
    at the placement (the curvature the run is judged under), ``"plain"`` from the
    slack-only decode — SVD'd, and installed as a backward-only transform: forward
    identity, backward ``∂L/∂z ← σ_max²·V·diag(1/(σ²+ε²))·Vᵀ·∂L/∂z`` with ``ε =
    preconditioner_floor · σ_max`` (``σ_max`` over all orders). Steep directions take
    Gauss-Newton steps, flat ones are lifted by at most ``1/floor²``. The forward pass, and
    hence every checkpoint's inference, is untouched; the preconditioner is rebuilt
    deterministically whenever training resumes. Needs the injection decode with the
    Cartesian injection encoding (``EncodingConfig.injection_kind="cartesian"``, where the
    decode chain is affine in the head's output) and a single grid.

    The physics consistency of the estimate is enforced STRUCTURALLY by the injection-decode
    estimator (:class:`PhysicsConfig`), not by a soft residual term on the loss.
    """

    harmonic_weight: HarmonicWeightConfig = Field(default_factory=HarmonicWeightConfig)
    magnitude_term_weight: float = Field(default=0.0, ge=0.0)
    predict_injection: bool = False
    injection_loss_weight: float = Field(default=1.0, ge=0.0)
    injection_hidden_dim: int = Field(default=128, gt=0)
    injection_state_loss_split: float = Field(default=0.0, ge=0.0, lt=1.0)
    injection_state_loss_normalize: bool = True
    plain_decode_weight: float = Field(default=0.0, ge=0.0)
    #: Row weights of the plain-decode term (``plain_decode_weight``). ``None`` (default)
    #: evaluates it on the voltage loss's rows (``supervise_rows``). A value ``λ`` weights
    #: every unobserved row 1 and every observed row ``λ``. The plain term is the injection
    #: error seen through the network impedances, ``‖Z(Î − I_true)‖²``: moving injection
    #: between electrically close nodes changes no voltage and costs almost nothing, while
    #: misplacing it across a meter (the meter on the path between the predicted and the true
    #: location) changes the metered voltage, and ``λ > 1`` makes that error the expensive
    #: one.
    plain_decode_observed_weight: Optional[float] = Field(default=None, ge=0.0)
    prior_distillation_weight: float = Field(default=0.0, ge=0.0)
    #: Which rows the voltage loss supervises. ``"all"`` (default) averages over every
    #: node-phase row. ``"unobserved"`` averages over the rows the batch's mask leaves
    #: unobserved only — the rows the estimator exists for; under measurement anchoring the
    #: observed rows are pinned by the solve and contribute residuals the network cannot
    #: change. A batch without unobserved rows (a full-observability stage) falls back to
    #: all rows. Applies to the anchored, the plain-decode and the direct voltage loss.
    supervise_rows: Literal["all", "unobserved"] = "all"
    preconditioner: Literal["none", "plain", "placement"] = "none"
    preconditioner_floor: float = Field(default=0.01, gt=0.0, le=1.0)


class CalibrationKnobConfig(_Base):
    """One calibratable grid parameter + the user's confidence in its recorded value.

    ``target``/``id`` select the grid element (``"line"``/``"transformer"`` by ``id``);
    ``field`` is the physical attribute (e.g. ``"length_m"``,
    ``"series_resistance_ohm_per_m"``). ``tolerance`` is the RELATIVE confidence — ``0.2`` =
    "known to ±20%" (a soft prior std on the multiplicative correction); ``0.0`` FIXES it.
    """

    target: Literal["line", "transformer", "branch"] = "line"
    id: int
    field: str
    tolerance: float = Field(default=0.1, ge=0.0)


class PhysicsConfig(_Base):
    """Physics-informed decode: predict nodal injection currents, SOLVE for the voltages.

    ``decode="injection"``: the estimator predicts the nodal INJECTION currents ``I(h)`` (the
    net current injected at each node — not a branch flow) and the differentiable network solve
    reconstructs the voltage state ``V(h) = solve(Y(h), I_pred)`` — the fundamental via an
    ideal-slack reduced solve held at the measured slack voltage, the harmonics via the
    Norton-shunted system. The output satisfies the nodal network law by construction (no
    ground-truth injection on the RHS → no measurement leakage), and because ``Y(h)`` is a
    differentiable function of the grid description the same path yields grid-parameter
    sensitivities and a learned grid calibration. ``decode="voltage"`` (default) is the original
    estimator that predicts the voltages directly.

    ``calibrate`` (a list of :class:`CalibrationKnobConfig`, empty by default) turns on the
    learned grid-parameter calibration: a small, shared correction on each listed parameter is
    trained JOINTLY with the estimator and regularised toward 0 by each knob's confidence
    (``calibration_weight`` scales the prior). Only used with ``decode="injection"``.

    ``anchor_voltage_weight`` / ``anchor_current_weight`` turn the injection decode from a
    slack-only-anchored solve into a MEASUREMENT-ANCHORED (over-determined) one: on top of the
    predicted nodal injections, the OBSERVED voltage rows and OBSERVED branch-current channels
    become soft constraints, so the reconstructed state is pulled toward every measurement the
    model sees, not only the slack. The solve minimises ``||Y·V - I_pred||² +
    w_v·||V_obs - V_meas||² + w_i·||C·V - I_meas||²`` per order (least squares — the slack stays
    hard-fixed as the network boundary). Both weights default ``0.0`` = OFF (the slack-only
    solve, byte-identical). Only used with ``decode="injection"``. What a positive weight MEANS
    is set by ``anchor_weighting``:

    - ``"manual"`` (default): DIMENSIONLESS relative-trust knobs (measurement vs. the
      injection prediction) — the current term shares the physics term's units and takes the
      weight directly, the voltage term is auto-scaled by the per-order operator admittance
      scale so ``1.0`` means "trust a voltage measurement like one nodal current balance".
      One number per order trusts every channel equally, and — the voltage weight being
      scaled and the current weight raw — trusts a noisy CT far more than a noisy voltage
      sensor.
    - ``"noise_calibrated"``: the least-squares solve is read as the Gaussian MAP estimate it
      is, whose correct weights are INVERSE VARIANCES. With the physics block at weight 1,
      every anchored channel gets ``w = σ_I² / σ_meas²`` — ``σ_I(h) =
      prior_injection_rel_sigma · s_I(h)`` the assumed error of the predicted injection
      (``s_I`` the training-set RMS injection magnitude of that order over the rows that
      CARRY A DEVICE — the fitted injection normalizer's ``aux_scale``; the slack row's
      ``Y·V`` entry is the total feeder current and a passive junction's is zero, so either
      one pooled in would misstate how large a device injection is) and ``σ_meas`` the
      per-real-component error of THAT channel at
      that order under the configured sensor model: ``½(σ_mag² + σ_phase²)·typ(h)² +
      (floor·range(h))²`` from the channel's accuracy class (the voltage rows with the state
      normalizer's RMS ``|V(h)|`` and the metered node's nominal range, the CT channels with
      the edge normalizer's RMS ``|I(h)|`` and the CT range — exactly the sigmas the
      :class:`NoiseConfig` model draws from), or ``clean_sensor_rel_sigma · typ(h)`` when
      the noise model is off. Voltage and current channels then carry commensurate
      weights, the CT amplification of a raw current weight disappears, and
      ``anchor_voltage_weight`` / ``anchor_current_weight`` (global, per stage, per-batch
      range) become dimensionless MULTIPLIERS on the calibrated weight: ``1`` = the
      noise-optimal weight, ``0`` = off. Installed once at module construction from the
      fitted normalizers and the noise model, and re-derived identically on a checkpoint
      reload (both travel with the checkpoint).

    ``injection_prior_residual`` (default OFF) parameterises the injection decode as the
    LINEAR INJECTION PRIOR plus a learned residual: ``I_pred = I_prior(measurements) +
    I_network``, with the prior the closed-form least-squares map from this run's own
    measurements onto ``Y(h)·V(h)`` (:func:`injection_hse.analysis.fit_linear_injection_prior`, fitted
    on the TRAIN split of the pass that trains with it at the deployed placement, carried in
    every checkpoint) and the residual added in the head's own normalized injection feature space and the network's
    output layer initialised to ZERO, so training starts exactly at the prior —
    the estimate a few hundred snapshots already identify and that a network trained from
    scratch has not reached (measured 2× behind it) — and learns only what the linear map
    cannot represent. Inside a curriculum stage that observes fewer sensors than the
    fitted placement, the prior's standardised features of the unobserved placement rows
    are set to their mean (zero), so the residual never reads a row the network cannot see;
    at the placement the prior reads exactly its own sensors. The injection-space terms
    (``LossConfig.injection_state_loss_split``, ``prior_distillation_weight``) see the FULL
    predicted injection, prior included. Injection decode, single grid.

    ``injection_prior_fit_states`` (default ``4096``) is the number of TRAIN states the
    linear injection prior is fitted on, whenever a pass needs it (the residual base or the
    distillation teacher, ``LossConfig.prior_distillation_weight``): the first states of
    the pass's own train split, a sequence contributing every step. In a multi-pass run
    (:class:`injection_hse.workflow.TrainingRunConfig`) each pass that needs the prior refits it on its
    own train split before it trains, so the snapshot pass and the sequence pass each start
    from the map of the data they train on, and every checkpoint carries the prior its pass
    used. The fit is a closed-form ridge; its peak memory is the state sample (``S·H·N``
    complex values) plus one ``[S, F]`` float64 feature block (``F = 2·H·`` the observed rows
    and metered half-edges) plus the ``[K, F]`` coefficient block (``K = 2·H·`` the
    predicted rows), so a larger sample costs memory linearly in ``S``.

    ``exclude_tied_rows`` (default ON) removes the rows tied to the slack through CLOSED
    switches from the injection space: an injection there leaves through the tie into the
    slack and moves no voltage, so the head's output at those rows is fixed to zero
    injection, the injection normalizer and the linear prior skip them, and their ``Y·V``
    entry — the whole feeder's current, like a slack row's — never becomes a target
    (:func:`injection_hse.physics.tied_rows_of`). On CIGRE LV the three MV buses behind the feeder
    switches are 7 % of the rows and carry 73 % of the fundamental injection energy in the
    head's own frame, so every injection-space objective and diagnostic (the direct term,
    distillation, copy quality) was dominated by rows that cannot affect any voltage. The
    rows stay in the STATE: their voltage is solved and reported as before, the row layout
    is unchanged, and a switch-state inference from the estimated voltages is untouched.
    The decoded state is the same either way (the excluded directions are exactly flat);
    ``False`` restores the previous injection space for a like-for-like comparison.

    ``input_priors`` (default empty) adds PHYSICS-DERIVED per-row INPUT channels to every
    node token, with EITHER decode. A message-passing estimator sees the measurements of its
    few-hop neighbourhood, while the fundamental voltage at an unobserved row is a GLOBAL
    function of the measurements (the drop along the whole feeder, which a handful of layers
    cannot carry), and Kirchhoff's current law ``I = Y·V`` needs a neighbour voltage
    MULTIPLIED by an admittance, which additive edge messages do not represent exactly. Both
    are cheap linear functions of the measured state and the known network, so they are
    supplied as inputs and the network learns the nonlinear correction on top of the best
    linear estimate (the structure of learned-iterative reconstruction). The channels are
    per row and per-unit-scaled, so they carry over to a multi-grid run unchanged.

    - ``"kcl"`` — the nodal injection the OBSERVED neighbourhood implies, ``κ_r(h) =
      Σ_j Y(h)_rj · (mask_j · V_in,j(h))`` (the masked, noisy-when-noise-is-on input state
      through the grid-constant operators; each sample's own switch configuration under a
      switchable topology): exact where the whole neighbourhood is observed, a partial sum
      otherwise. Encoded with the INJECTION encoding and normalizer (``H × C_inj``
      channels), plus ONE channel ``f_r`` — the fraction of row ``r``'s coupled rows
      (itself included) observed under the location mask — so the model knows how complete
      each sum is.
    - ``"linear_state"`` — the measurement-anchored solve of the ZERO injection prior: the
      MAP state under "no knowledge of the injections" at every row, anchored to the observed
      voltages (and the measured branch currents when ``measure_current``) with the weights
      ``prior_anchor_voltage_weight`` / ``prior_anchor_current_weight``. Encoded with the
      STATE encoding and normalizer (``H × C`` channels). Requires ``measurement.slack_known``
      (the fundamental boundary of the solve).

    Both are computed under ``no_grad`` (inputs, nothing learned) and need the network
    solver, which is therefore built for a voltage decode too whenever the list is
    non-empty. An empty list is byte-identical (no extra channels, no solver on the voltage
    decode).

    ``prior_anchor_voltage_weight`` / ``prior_anchor_current_weight`` (default ``1.0``) are
    the anchor weights of the ``"linear_state"`` prior's own solve, installed on the solver
    for that solve only and restored afterwards — they are independent of the decode's
    anchoring schedule. Under ``anchor_weighting="manual"`` they are the usual dimensionless
    voltage trust / raw current weight; under ``"noise_calibrated"`` they are multipliers on
    the calibrated weight, and the prior solve uses the ZERO-prior injection sigma (``ρ =
    1``, the order's RMS injection — the prior knows nothing about the injections), which is
    realised by scaling the multiplier by ``1 / prior_injection_rel_sigma²`` relative to the
    calibration installed for the decode. The current weight is used only when currents are
    measured.
    """

    decode: Literal["voltage", "injection"] = "voltage"
    #: Physics-derived INPUT channels (see the class docstring); empty = byte-identical.
    #: Entries are unique; the block order is ``kcl`` then ``linear_state`` whatever the
    #: list order.
    input_priors: list[Literal["kcl", "linear_state"]] = Field(default_factory=list)
    #: Anchor weights of the ``"linear_state"`` prior's own zero-injection solve (see the
    #: class docstring). The current weight is read only when ``measurement.measure_current``.
    prior_anchor_voltage_weight: float = Field(default=1.0, ge=0.0)
    prior_anchor_current_weight: float = Field(default=1.0, ge=0.0)
    calibrate: list[CalibrationKnobConfig] = Field(default_factory=list)
    calibration_weight: float = Field(default=1e-6, ge=0.0)
    #: Trust in the observed voltage rows / branch-current channels of the anchored solve;
    #: ``0`` = off. A dimensionless relative trust under ``anchor_weighting="manual"``, a
    #: dimensionless MULTIPLIER on the noise-optimal weight under ``"noise_calibrated"``.
    anchor_voltage_weight: float = Field(default=0.0, ge=0.0)
    anchor_current_weight: float = Field(default=0.0, ge=0.0)
    #: How the anchor weights are constructed (see the class docstring). ``"manual"`` is
    #: byte-identical to a config written before the field existed.
    anchor_weighting: Literal["manual", "noise_calibrated"] = "manual"
    #: ``ρ`` — the assumed prior error of the PREDICTED injection at each order, RELATIVE to
    #: that order's RMS injection magnitude over the training rows (``σ_I(h) = ρ · s_I(h)``).
    #: ``1`` is a prior that knows nothing (the zero-injection prior); a trained network with
    #: a 10 % injection error is ``0.1``. Larger ``ρ`` trusts the measurements more. Read only
    #: under ``anchor_weighting="noise_calibrated"``. A list gives one ``ρ`` per solved order
    #: (aligned with ``ExperimentConfig.orders``): an estimator's injection error differs
    #: between orders, and an estimator computed from the same readings the solve anchors to
    #: is best decoded with its own (small) error, which approaches the unanchored decode.
    prior_injection_rel_sigma: Union[float, list[float]] = 1.0

    @field_validator("prior_injection_rel_sigma")
    @classmethod
    def _positive_rel_sigma(cls, value):
        values = value if isinstance(value, list) else [value]
        if not values or any(float(v) <= 0.0 for v in values):
            raise ValueError("prior_injection_rel_sigma must be positive (every order).")
        return value
    injection_prior_residual: bool = False
    #: TRAIN states the linear injection prior is fitted on (see the class docstring).
    injection_prior_fit_states: int = Field(default=4096, ge=1)
    exclude_tied_rows: bool = True
    #: How the metered slack enters the HARMONIC solves of the decode. ``"norton"`` (default,
    #: byte-identical to before the field existed): the source's own shunt admittance with a
    #: zero harmonic emf — the harmonics float against an undistorted supply, and any
    #: distortion the supply carries has to be produced by injections. ``"measured"``: the
    #: slack rows' MEASURED harmonic voltage is the Dirichlet boundary at every order, as the
    #: fundamental already is — the upstream network's distortion enters through the
    #: measurement, which is the right model for a feeder whose harmonic voltage is dominated
    #: by its supply. Needs a metered slack (``measurement.slack_known``).
    harmonic_slack: Literal["norton", "measured"] = "norton"
    #: The harmonic DEVICE SHUNT of the decode's ``Y(h)`` (pgml's ``load_shunt``:
    #: ``"none"`` / ``"opendss"`` / ``"motor"``), always built on the NAMEPLATE basis: the
    #: decode solves a whole batch with one ``Y(h)`` per order, so the shunt cannot follow
    #: each scenario's operating point. ``None`` (default) passes no shunt setting, so the
    #: operator is assembled as before the field existed: pgml's defaults with no
    #: fundamental solution, i.e. the default shunt model at the devices' stored P, Q and
    #: rated voltage.
    #: Set it to the model the training data was generated with — on the ``nameplate``
    #: basis there (``run_scenarios(load_shunt=..., load_shunt_basis="nameplate")``) — and
    #: the decode's ``Y(h)`` equals the generator's to rounding; data generated on the
    #: ``operating_point`` basis differs from any decode ``Y(h)`` by each scenario's
    #: operating-point shunt term.
    load_shunt: Optional[Literal["none", "opendss", "motor"]] = None
    #: The relative per-real-component measurement sigma assumed for a channel when the
    #: noise model is OFF (``noise.enabled=False``): a clean sensor is still not infinitely
    #: trusted — it caps the weight and keeps the anchored correction system's conditioning
    #: (which grows with the weight) finite. Read only under
    #: ``anchor_weighting="noise_calibrated"``.
    clean_sensor_rel_sigma: float = Field(default=1e-4, gt=0.0)
    #: Byte budget (MiB) of the FACTOR-ONCE operator caches of the network solve; ``0``
    #: (default) = off, byte-identical to the per-call factorization the solve always did.
    #: The operators ``Y(h)`` are grid-constant while a training run solves them tens of
    #: thousands of times, so with a budget the solver caches, per order, the anchored
    #: solve's ``Z = Y_ff^{-1}`` image (each batch then answered through the anchored-rows
    #: correction — ``O(R²F)`` for ``R`` anchored channels instead of ``O(F³)``) and the
    #: plain solve's LU factors. Results are algebraically identical (machine-rounding
    #: differences only); a ``Y`` on the autograd tape (``calibrate``) is never cached. The
    #: budget is shared across all member grids / switch configurations, first-come; size
    #: it to the accelerator's free memory (the cache per grid and order is roughly two
    #: complex128 ``[F, F]`` images).
    operator_cache_mb: int = Field(default=0, ge=0)
    #: How a stage's per-batch anchor randomization draws inside its range.
    #: ``"uniform"`` spreads the WEIGHT evenly, which spreads the informativeness very
    #: unevenly: the injection prediction only carries gradient where the anchor is weak, so
    #: a uniform draw over ``0~20`` leaves ~95 % of batches unable to teach the injection
    #: head anything. ``"log_uniform"`` spreads the draw over DECADES of weight instead
    #: (:data:`ANCHOR_LOG_DECADES` below the range maximum), so weak-anchor batches are
    #: common enough to train on. Only the randomization changes; fixed weights are untouched.
    anchor_weight_draw: Literal["uniform", "log_uniform"] = "uniform"

    @model_validator(mode="after")
    def _unique_priors(self) -> "PhysicsConfig":
        if len(set(self.input_priors)) != len(self.input_priors):
            raise ValueError(
                f"physics.input_priors lists a prior twice ({self.input_priors}); each "
                "prior adds its channel block once."
            )
        return self

    @property
    def needs_solver(self) -> bool:
        """Whether the run needs the differentiable network solver at all.

        The injection decode solves every step; the physics-derived input priors need the
        same grid-constant operators (``Y(h)``, the anchored solve) under either decode. The
        training entry points, the checkpoint rebuild and the multi-grid operator bank all
        build the solver on this condition.
        """
        return self.decode == "injection" or bool(self.input_priors)


# =============================================================================
# Measurement model (which nodes/quantities are observed)
# =============================================================================
class MeasurementConfig(_Base):
    """The physical sensor set.

    ``voltage_nodes`` are node names or ids carrying a full-harmonic measurement device
    (voltage observed at every solved order). ``fundamental_nodes`` are additional
    FUNDAMENTAL-ONLY monitors (smart-meter class): their rows are observed at ``h=1`` only —
    requires ``heterogeneous_orders``. ``slack_known`` marks the slack bus(es)
    as metered: the slack measurement is an ordinary (full-harmonic) measurement — its
    voltage is a model input, exempt from random masking (the substation meter does not
    disappear) — and the physics-informed decode additionally requires it as the
    fundamental network boundary (``physics.decode="injection"`` raises without it).

    ``measure_current`` adds the measured BRANCH CURRENTS as model input: every measurement
    location (the slack included) meters the current of ALL its connected branches by
    default, entering the model as dynamic edge observations on the directed half-edges
    whose source terminal sits at an observed row (a per-device override — restricted CT
    channels, no current capability — comes from the grid's installed
    :class:`~pgml.schemas.grid_schema.MeasurementDevice` list). ``current_channels``
    selects the per-location CT budget: ``"incident"`` (default) meters every connected
    branch, ``"upstream"`` restricts each metered location to the single branch toward
    the slack (the one-CT budget meter; per phase) — the slack location itself keeps its
    incident feeders, it is the network boundary meter. ``measure_power`` appends
    the per-order terminal apparent power ``S = V · I*`` as additional edge channels
    (requires ``measure_current``; derivable from V and I, so off by default). The
    state estimate itself remains voltage-only — currents are INPUT, the full state
    (and thus every branch flow) is reconstructed from the estimated voltages.

    ``heterogeneous_orders`` switches the model input to a HETEROGENEOUS sensor fleet:
    every node (and edge) token carries per-order observability channels, so
    fundamental-only and full-harmonic devices can coexist. It must be set explicitly
    (not inferred) because it changes the model input width — every curriculum pass of
    one module must agree on it. Off (default) is byte-identical to before.

    Grids with attached :class:`~pgml.schemas.grid_schema.MeasurementDevice`
    instrumentation can derive the sensor set via ``MeasurementModel.from_grid``.
    """

    voltage_nodes: list[Union[int, str]] = Field(default_factory=list)
    fundamental_nodes: list[Union[int, str]] = Field(default_factory=list)
    measure_current: bool = False
    current_channels: Literal["incident", "upstream"] = "incident"
    measure_power: bool = False
    heterogeneous_orders: bool = False
    slack_known: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> "MeasurementConfig":
        if self.measure_power and not self.measure_current:
            raise ValueError(
                "measurement.measure_power requires measure_current (power is a "
                "branch-terminal quantity of the metered current channels)."
            )
        if self.fundamental_nodes and not self.heterogeneous_orders:
            raise ValueError(
                "measurement.fundamental_nodes requires heterogeneous_orders=True "
                "(fundamental-only monitors need the per-order observability channels)."
            )
        overlap = {str(n) for n in self.voltage_nodes} & {
            str(n) for n in self.fundamental_nodes
        }
        if overlap:
            raise ValueError(
                f"nodes {sorted(overlap)} appear in BOTH voltage_nodes and "
                "fundamental_nodes; a location is either full-harmonic or "
                "fundamental-only."
            )
        return self


# =============================================================================
# Sensor noise (opt-in; keyed by the device accuracy class)
# =============================================================================
class AccuracyClassNoise(_Base):
    """The per-measurement noise magnitude of one instrument accuracy class.

    ``magnitude_pct`` is the standard deviation of the MULTIPLICATIVE magnitude error
    (percent of the reading) and ``phase_deg`` the standard deviation of the ADDITIVE
    phase error (degrees), applied as independent zero-mean Gaussians to each measured
    phasor. ``floor_pct`` is the standard deviation of an ADDITIVE range-referenced
    error (percent of the instrument RANGE, per real component): an instrument's
    resolution is set by its range, not by the reading, so a harmonic bin at millivolts
    against a 230 V range sees the floor, not the reading error — a purely
    multiplicative model shrinks with the reading and flatters exactly the small
    harmonic channels that identifiability analyses show are binding
    (``injection_hse.analysis.feasibility``). The class-A-style default pairs 0.05 % of range
    with the 0.1 reading class, in the spirit of the IEC 61000-4-7 small-signal limit.
    The defaults are a pragmatic reading of common revenue/PQ-meter accuracy classes —
    NOT an implementation of IEC 61869 instrument-transformer error limits.
    """

    magnitude_pct: float = Field(default=0.5, ge=0.0)
    phase_deg: float = Field(default=0.2, ge=0.0)
    floor_pct: float = Field(default=0.25, ge=0.0)


def _default_accuracy_classes() -> dict[str, AccuracyClassNoise]:
    """The default class -> noise map: tighter magnitude/phase/floor for finer classes."""
    return {
        "0.1": AccuracyClassNoise(magnitude_pct=0.1, phase_deg=0.05, floor_pct=0.05),
        "0.2": AccuracyClassNoise(magnitude_pct=0.2, phase_deg=0.1, floor_pct=0.1),
        "0.5": AccuracyClassNoise(magnitude_pct=0.5, phase_deg=0.2, floor_pct=0.25),
        "1": AccuracyClassNoise(magnitude_pct=1.0, phase_deg=0.5, floor_pct=0.5),
    }


class InstrumentBand(_Base):
    """Datasheet error limits of one quantity over one band of harmonic orders.

    The limits are MAXIMUM errors as a datasheet states them; :class:`InstrumentErrorConfig`
    turns them into standard deviations (``limit / coverage``). ``threshold_pct`` and
    ``floor_pct`` are percent of the band's RANGE, ``fsr_over_nominal`` times the channel's
    nominal value (``1`` = the nominal value itself; a datasheet that states a band against
    the full-scale range of an input whose nominal value is smaller gives the ratio here). A
    reading whose magnitude is at or above the threshold is in the PROPORTIONAL regime:
    ``reading_pct`` (percent of the reading) bounds its magnitude error and ``phase_deg``
    its phase error. ``floor_pct`` is the magnitude limit below the threshold; the datasheet
    gives no phase limit there, so the floor is an ISOTROPIC complex error of that size (its
    phase error is the one the floor implies). The floor is present at every level, above
    the threshold too (see :class:`InstrumentErrorConfig`).
    """

    reading_pct: float = Field(ge=0.0)
    phase_deg: float = Field(ge=0.0)
    floor_pct: float = Field(ge=0.0)
    threshold_pct: float = Field(ge=0.0)
    fsr_over_nominal: float = Field(default=1.0, gt=0.0)


class InstrumentErrorConfig(_Base):
    """Error model of an IEC 61000-4-30 class-A power-quality instrument, stable / random.

    The defaults are the class-A limits: the magnitude limits of IEC 61000-4-30 class A,
    with the harmonic bands of IEC 61000-4-7 class I, and the phase limits and the current
    fundamental band of a class-A instrument with a 5 A current input of 10 A full scale.
    ``U_nom`` is the channel's nominal line-to-neutral voltage, ``I_nom`` the CT range of
    the current channels (:func:`injection_hse.data.grid_current_range`), and the full-scale range of
    the current input is ``FSR = 2·I_nom``:

    ==========================  ============================================  ==================
    quantity                    magnitude                                     phase
    ==========================  ============================================  ==================
    voltage, fundamental        ±0.1 % of ``U_nom``                           ±0.02° (≥ 10 %)
    voltage, harmonics 2..50    ±5 % of reading (≥ 1 % ``U_nom``), else       ±0.5° (≥ 1 %)
                                ±0.05 % of ``U_nom``
    current, fundamental        ±0.1 % of reading (≥ 10 % FSR), else          ±0.1° (≥ 10 %)
                                ±0.01 % of FSR
    current, harmonics 2..50    ±5 % of reading (≥ 3 % ``I_nom``), else       ±0.5° (≥ 3 %)
                                ±0.15 % of ``I_nom``
    ==========================  ============================================  ==================

    The voltage fundamental's limit of 0.1 % of ``U_nom`` is modelled as 0.1 % of the
    READING (with a floor of 0.01 % of ``U_nom`` below 10 % ``U_nom``, the same absolute
    error at the threshold), which differs by at most the operating deviation of the voltage
    from nominal (about ±10 %) and keeps the phase error at 0.02°.

    A limit is read as ``coverage`` standard deviations (``2`` ≈ 95 % for a Gaussian). Each
    reading's regime follows its own magnitude, and its error has two parts:

    - the PROPORTIONAL part (above the threshold): a gain error of standard deviation
      ``reading_pct / coverage`` and a phase error of ``phase_deg / coverage``. A fraction
      ``stable_fraction`` of its variance is STABLE — a gain and phase error drawn once per
      scenario (or per site key, :attr:`injection_hse.data.InstrumentNoise.site`), channel and order
      and held over the scenario's time steps, the calibration error of a real instrument,
      which cancels in a reading CHANGE — and the rest is drawn anew for every reading.
      ``1.0`` holds it entirely, ``0.0`` makes it entirely random.
    - the FLOOR (every level): an isotropic complex error of standard deviation
      ``floor_pct / coverage`` of the band's range per real component, drawn anew for every
      reading. Above the threshold it is the instrument's resolution, which a held gain does
      not remove; without it a held gain would make reading changes error-free.

    Orders above ``max_order`` have no stated limit; a noise model built for them is
    refused. :meth:`class_a` returns these defaults; a run config states the bands as
    fields (its ``model_dump``).
    """

    voltage_fundamental: InstrumentBand = Field(
        default_factory=lambda: InstrumentBand(
            reading_pct=0.1, phase_deg=0.02, floor_pct=0.01, threshold_pct=10.0
        )
    )
    voltage_harmonic: InstrumentBand = Field(
        default_factory=lambda: InstrumentBand(
            reading_pct=5.0, phase_deg=0.5, floor_pct=0.05, threshold_pct=1.0
        )
    )
    current_fundamental: InstrumentBand = Field(
        default_factory=lambda: InstrumentBand(
            reading_pct=0.1,
            phase_deg=0.1,
            floor_pct=0.01,
            threshold_pct=10.0,
            fsr_over_nominal=2.0,
        )
    )
    current_harmonic: InstrumentBand = Field(
        default_factory=lambda: InstrumentBand(
            reading_pct=5.0, phase_deg=0.5, floor_pct=0.15, threshold_pct=3.0
        )
    )
    #: Number of standard deviations a datasheet limit stands for.
    coverage: float = Field(default=2.0, gt=0.0)
    #: Share of the proportional error VARIANCE held per scenario, channel and order.
    stable_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    #: Highest harmonic order with a stated limit.
    max_order: int = Field(default=50, ge=1)

    @classmethod
    def class_a(cls, **overrides: Any) -> "InstrumentErrorConfig":
        """The IEC 61000-4-30 class-A instrument (the field defaults); ``overrides``
        replace fields (e.g. ``stable_fraction``)."""
        return cls(**overrides)


class NoiseConfig(_Base):
    """Opt-in measurement noise on the OBSERVED phasors (never the supervised targets).

    When ``enabled``, each measured voltage phasor — and each measured branch-current (CT)
    phasor, when currents are metered — that feeds the model is perturbed by a multiplicative
    magnitude error and an additive phase error, drawn per class from ``classes`` and keyed by
    each sensor's :class:`~pgml.schemas.grid_schema.MeasurementDevice` ``accuracy_class``. A
    sensor whose device carries no matching class (or a config-driven placement with no device
    metadata) falls back to ``default_class``. A current channel inherits its metering node's
    class but is drawn from an INDEPENDENT stream, so the voltage and current errors are not
    perfectly correlated. The perturbation is applied in the training AND the
    evaluation input path; the ground-truth state used for the loss and the reported metrics
    is UNTOUCHED. Deterministic given ``seed``; ``enabled=False`` is byte-identical to a run
    with no noise (nothing is drawn).

    The class -> sigma mapping (``classes``) is a pragmatic default, documented on
    :class:`AccuracyClassNoise` — not an IEC 61869 instrument-transformer error model.

    On the MULTI-GRID path (``data.multigrid``) the sigmas are uniform: every channel
    draws the ``default_class`` values (members differ in size and a generated corpus
    carries no device metadata, so there is nothing to key per-row classes on), the
    voltage floor references the per-unit nominal, and the current channels carry
    reading/phase error only.

    ``instrument`` (single-grid path) replaces the accuracy classes by a datasheet error
    model with a stable and a random part (:class:`InstrumentErrorConfig`,
    :class:`injection_hse.data.InstrumentNoise`): every voltage and current channel is read by that
    instrument, and ``default_class``, ``classes``, ``harmonic_range`` and the
    magnitude-only meters do not apply.
    """

    enabled: bool = False
    seed: int = 0
    default_class: str = "0.1"
    classes: dict[str, AccuracyClassNoise] = Field(
        default_factory=_default_accuracy_classes
    )
    #: Instrument range of a HARMONIC subchannel, as a fraction of the fundamental range —
    #: the scale the additive floor (``floor_pct``) refers to at orders above the
    #: fundamental. A floor referenced to the fundamental for every order is the wrong
    #: instrument model: one absolute floor is 0.05 % of a 230 V fundamental but ~17 % of a
    #: 0.7 V nineteenth, so the orders an estimator exists to recover arrive buried in a
    #: noise that a real analyser's harmonic subchannel does not have. ``1.0`` restores the
    #: fundamental-referenced floor.
    harmonic_range: float = Field(default=0.1, gt=0.0, le=1.0)
    #: Fraction of the voltage rows that are MAGNITUDE-ONLY meters (multi-grid path only):
    #: per sample, each row independently becomes such a meter with this probability. Its
    #: reading is the true RMS magnitude with the ``angle_fill_class`` reading error, at
    #: the NOMINAL angle — the phasor a deployment builds from an RMS-only instrument by
    #: filling the phase's nominal angle. In the slack-referenced frame the nominal angle
    #: is zero (a multiple of 30° behind a vector-group transformer), so the reading takes
    #: the true angle rounded to the nearest 30°. ``0.0`` (default) draws nothing extra
    #: (byte-identical).
    angle_fill_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Accuracy class (a key of ``classes``) of the magnitude-only meters; ``None`` = the
    #: ``default_class``. Its ``magnitude_pct`` is the RMS reading error; its phase and
    #: floor terms do not apply (the angle is filled, not measured).
    angle_fill_class: Optional[str] = None
    #: Where the noise draws are generated. ``"cpu"`` (default) draws on a CPU generator and
    #: moves the noise to the phasors' device: device-independent and reproducible across
    #: machines. ``"input"`` draws on the phasors' own device with a generator seeded from the
    #: same seed: the same distribution, a different (device-specific) stream, and no
    #: per-step host work — on a GPU the CPU draw of a sequence batch can dominate the step.
    draw_device: Literal["cpu", "input"] = "cpu"
    #: Datasheet instrument error model used INSTEAD of the accuracy classes; ``None``
    #: (default) keeps the accuracy-class model and its draws unchanged.
    instrument: Optional[InstrumentErrorConfig] = None


# =============================================================================
# Masking curriculum (per-sample masks, target-loss-driven stages)
# =============================================================================
class MaskStageConfig(_Base):
    """One curriculum stage.

    ``kind``: ``"full"`` (all nodes observed), ``"random"`` (per-sample random mask keeping
    ``observed_fraction`` of nodes), or ``"placement"`` (the fixed sensor set in
    :class:`MeasurementConfig` — the finetune target). Training advances when the monitored
    loss reaches ``target_loss`` (early-stop-on-target) or ``max_epochs`` is hit. A
    checkpoint is forced at every stage boundary.

    ``lr`` and ``scheduler`` optionally override :class:`OptimConfig` FOR THIS STAGE
    (``None`` = inherit), and ``warmup_epochs`` ramps the rate up at the stage's start: each
    stage builds a fresh optimizer (its moments carried over only with
    :attr:`TrainConfig.carry_optimizer_state`), so e.g. the ``placement``
    finetune can anneal a lower learning rate (``scheduler="cosine"`` anneals once over
    the stage's ``max_epochs``) while the earlier stages keep the global setting. The
    per-stage optimization is part of the serialized config — one config + seed still
    reproduces the run.

    ``fundamental_fraction`` (``random`` stages; requires
    ``measurement.heterogeneous_orders``) demotes that fraction of each sample's observed
    rows to FUNDAMENTAL-ONLY monitors (observed at ``h=1`` only), so the curriculum
    presents mixed sensor fleets before a heterogeneous ``placement`` finetune. ``None``
    (default) keeps every observed row full-harmonic.

    ``prior_distillation_weight`` optionally overrides :attr:`LossConfig
    .prior_distillation_weight` FOR THIS STAGE (``None`` = inherit). This is the term's
    ANNEAL: the linear injection prior is a means of building the injection map, not the
    accuracy the run is judged on, so a curriculum carries a large weight while the map is
    being learned and drops it to ``0`` for the final stage, which then optimizes the
    deployed physics objective alone. A stage weight of ``0`` costs nothing — no teacher is
    evaluated — but the prior is still fitted before training whenever ANY stage asks for it.

    ``anchor_voltage_weight`` / ``anchor_current_weight`` optionally override
    :class:`PhysicsConfig`'s measurement-anchoring weights FOR THIS STAGE (``None`` = inherit;
    injection decode only). This expresses a per-stage anchoring SCHEDULE: keep the weight low
    while many nodes are observed (a strong anchor there over-determines the state and starves
    the injection head of gradient — the observed rows are pinned to the measurement regardless
    of the prediction), then raise it as observability drops so the sparse-placement decode is
    dominated by the measurements it does have. The final (placement / anneal) stage's weights
    are the ones a reloaded checkpoint deploys at inference.
    """

    name: str
    kind: Literal["full", "random", "placement"] = "random"
    observed_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fundamental_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_loss: float = Field(gt=0.0)
    max_epochs: int = Field(gt=0)
    lr: Optional[float] = Field(default=None, gt=0.0)
    scheduler: Optional[Literal["none", "cosine", "plateau"]] = None
    #: Linear learning-rate warmup at the start of this stage, in epochs (``0`` = off): the
    #: rate rises from ``lr / (warmup_epochs + 1)`` to ``lr`` over these epochs, then the
    #: stage's scheduler takes over (a ``"cosine"`` anneal over the remaining epochs).
    #: Refused with ``"plateau"``.
    warmup_epochs: int = Field(default=0, ge=0)
    prior_distillation_weight: Optional[float] = Field(default=None, ge=0.0)
    anchor_voltage_weight: Optional[float] = Field(default=None, ge=0.0)
    anchor_current_weight: Optional[float] = Field(default=None, ge=0.0)
    #: Optional per-BATCH anchor-weight randomization ranges ``(low, high)`` for this
    #: stage: every training batch draws one uniform sample and scales both ranges with
    #: it (the voltage/current ratio is preserved), so the model cannot co-adapt to a
    #: single solve strength — it must emit injections that are correct at every anchor
    #: weight. Validation and checkpointing use the FIXED ``anchor_*_weight`` (set it to
    #: the range midpoint), so early stopping judges a stationary target.
    anchor_voltage_range: Optional[tuple[float, float]] = None
    anchor_current_range: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _warmup_valid(self) -> "MaskStageConfig":
        if self.warmup_epochs and self.scheduler == "plateau":
            raise ValueError("warmup_epochs cannot precede a 'plateau' scheduler.")
        if self.warmup_epochs >= self.max_epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be below max_epochs "
                f"({self.max_epochs})."
            )
        return self

    @model_validator(mode="after")
    def _anchor_ranges_valid(self) -> "MaskStageConfig":
        for name in ("anchor_voltage_range", "anchor_current_range"):
            rng = getattr(self, name)
            if rng is not None and not (0.0 <= rng[0] <= rng[1]):
                raise ValueError(f"{name} must satisfy 0 <= low <= high, got {rng}.")
        return self


class CurriculumConfig(_Base):
    """The ordered stages (full observability → random masking ladder → placement finetune).

    The masker draws a fresh mask PER SAMPLE (per-batch masking would bias each step toward
    one placement); see :mod:`injection_hse.masking`.

    An EMPTY stage list is valid and means "nothing left to train" — the case of a task
    pass fully covered by a resume checkpoint (the config artifact then records exactly
    what the run trained). :func:`injection_hse.train.train_experiment` rejects it (a from-scratch
    experiment with no stages is a configuration error).
    """

    stages: list[MaskStageConfig] = Field(default_factory=list)


# =============================================================================
# Models (DNN baseline, GNN, Graphormer) — discriminated union
# =============================================================================
class DNNConfig(_Base):
    """Fixed-topology, fixed-input MLP baseline."""

    kind: Literal["dnn"] = "dnn"
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 512, 512])
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class GNNConfig(_Base):
    """PyG message-passing over the node-phase graph.

    The message-passing layers are node-shared with no fixed-``N`` assumption, so the same
    weights ACCEPT a grid of different size/topology (an architectural property). Cross-grid
    transfer performance is not validated — results train and evaluate on one grid.

    ``input_skip`` (default off) adds a LONG-RANGE skip: the encoder's per-node embedding is
    concatenated to the last message-passing layer's output before the head. Every layer
    already carries a residual + LayerNorm (gradient flow), but a deep stack can still
    OVERSMOOTH — node embeddings converge and the input signal washes out; the skip hands
    the head direct access to the pre-propagation features. Intended for deep stacks
    (``num_layers`` ≳ 9); off is byte-identical to the previous architecture.
    """

    kind: Literal["gnn"] = "gnn"
    hidden_dim: int = 128
    num_layers: int = Field(default=6, gt=0)
    conv: Literal["gcn", "sage", "gin", "gat", "gine"] = "gine"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    input_skip: bool = False


class GraphormerConfig(_Base):
    """Transformer / Graphormer with structural + edge encodings (topology-aware).

    ``attention_bias`` selects what the learned per-head attention bias is keyed by — the
    structural prior that makes the attention topology-aware:

    - ``"spd"`` (default) — the unweighted HOP distance between node-phase rows, bucketed.
      Purely topological, and the classic Graphormer choice.
    - ``"electrical"`` — the pairwise EFFECTIVE RESISTANCE (the electrical distance of the
      network, per-unit-normalized per grid), log-bucketed. Two rows one hop apart across a
      1 m switch are electrically the same bus, two rows one hop apart across a 500 m cable
      are not; only the electrical distance separates them, which matters most when one model
      spans grids whose hop counts mean different things.
    - ``"both"`` — the two learned biases are summed, so attention can use hop structure and
      electrical proximity together.
    - ``"none"`` — no pairwise distance bias (the degree embedding remains), to isolate
      the effect of global attention from the effect of structural distance coding.

    ``"spd"`` builds no extra parameters and is byte-identical to a model configured before
    the field existed.
    """

    kind: Literal["graphormer"] = "graphormer"
    hidden_dim: int = 128
    num_layers: int = Field(default=6, gt=0)
    num_heads: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    attention_bias: Literal["none", "spd", "electrical", "both"] = "spd"
    # Number of ``(eigenvalue, eigenvector-entry)`` pairs at the END of node_static to
    # transform with a SAN-style spectral encoder. Zero keeps the earlier Graphormer exact.
    spectral_pe_dim: int = Field(default=0, ge=0, le=64)


#: A purely SPATIAL model (no time mixing); the per-step estimator a temporal wrapper drives.
SpatialModelConfig = Annotated[
    Union[DNNConfig, GNNConfig, GraphormerConfig], Field(discriminator="kind")
]


class TemporalConfig(_Base):
    """A pluggable TEMPORAL wrapper around a spatial model (exploits the ``T`` axis).

    Wraps a ``spatial`` :data:`SpatialModelConfig` and mixes across time CAUSALLY so the
    estimate at step ``t`` uses only steps ``<= t`` (plus a running memory) — required for the
    live/streaming use case. ``kind`` selects the temporal mechanism (swappable):
    ``"gru"`` (a per-node recurrent memory — the simplest causal-with-memory, implemented) or
    ``"causal_attention"`` (causal-masked temporal attention — a planned variant). The
    spatial model is applied PER STEP on a temporally-enriched observation, so any
    DNN/GNN/Graphormer slots in unchanged. See :mod:`injection_hse.models.temporal`.
    """

    kind: Literal["temporal"] = "temporal"
    spatial: SpatialModelConfig = Field(default_factory=GNNConfig)
    temporal: Literal["gru", "causal_attention"] = "gru"
    hidden_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


ModelConfig = Annotated[
    Union[DNNConfig, GNNConfig, GraphormerConfig, TemporalConfig],
    Field(discriminator="kind"),
]


# =============================================================================
# Switchable topology (one estimator across several switch configurations)
# =============================================================================
class TopologyVariantConfig(_Base):
    """One switch configuration: which of the switchable branches are OPEN.

    ``name`` labels the variant and names its dataset subdirectory under
    :attr:`DataConfig.dataset_path`; every switch of
    :attr:`TopologyConfig.switch_ids` not listed in ``open_switches`` is closed.
    """

    name: str = Field(min_length=1)
    open_switches: list[int] = Field(default_factory=list)


class TopologyConfig(_Base):
    """The discrete switch configurations ONE estimator is trained across.

    A grid whose switches can be operated has several topologies; each is a distinct
    network, so the training data is generated per variant (one dataset directory each,
    named by the variant) and the physics decode / branch-current operators are built per
    variant. Every variant shares the grid's node-phase row layout and the graph edge set —
    the graph is built from the grid as given, with every switchable branch CLOSED (the
    union topology), and a variant merely opens some of them. That keeps the model input
    layout, the sensor placement and the metered half-edges identical across variants.

    ``conditioning`` selects how the model is TOLD which topology a sample came from:

    - ``"edge_and_global"`` (default) — a per-sample switch-state channel on every
      half-edge AND the switch-state vector broadcast to every node token;
    - ``"edge"`` / ``"global"`` — only one of the two;
    - ``"none"`` — the model is not told (it must infer the topology from the
      measurements); the physics decode still uses the sample's true topology.
    """

    switch_ids: list[int] = Field(min_length=1)
    variants: list[TopologyVariantConfig] = Field(min_length=1)
    conditioning: Literal["edge_and_global", "edge", "global", "none"] = (
        "edge_and_global"
    )

    @model_validator(mode="after")
    def _consistent(self) -> "TopologyConfig":
        if len(set(self.switch_ids)) != len(self.switch_ids):
            raise ValueError("topology.switch_ids must be unique.")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"topology variant names must be unique, got {names}.")
        known = set(self.switch_ids)
        for v in self.variants:
            unknown = sorted(set(v.open_switches) - known)
            if unknown:
                raise ValueError(
                    f"topology variant {v.name!r} opens branches {unknown} that are not "
                    f"in switch_ids {self.switch_ids}."
                )
        return self


class MultiGridConfig(_Base):
    """Training across MANY grids (heterogeneous topologies), not one.

    ``DataConfig.dataset_path`` then points at a multi-grid dataset ROOT: a ``manifest.json``
    naming one dataset directory per grid (each written by ``pgml.scenarios.write_dataset``,
    with the grid itself persisted beside it), produced by :mod:`injection_hse.data.multigrid`. The
    train/val/test split is by GRID — validation and test measure held-out TOPOLOGIES, which
    is the generalization claim — so a batch mixes grids of different node count and the
    sample axis folds into the node axis (:class:`injection_hse.data.MultiGridDataSource`).

    ``per_unit`` is the multi-grid NORMALIZATION decision: the state enters encoding and
    normalization divided by each row's line-to-neutral base voltage, so ONE dataset-global
    :class:`~injection_hse.normalization.HarmonicNormalizer` stays valid across voltage classes
    (a 400 V feeder and a 20 kV feeder differ by a factor 50 in volts and not at all in
    per-unit). Set it ``False`` only for a single-voltage-class study.

    ``placement_fraction`` (default ``None``) selects the fixed-placement rule for grids
    that carry no measurement devices: the fraction of nodes metered by
    :func:`injection_hse.data.central_placement` (slack + per-feeder central nodes). ``None`` leaves
    the placement to the random-masking curriculum (the slack rows stay forced-observed).

    ``angle_reference`` is the multi-grid ANGLE decision, the counterpart of ``per_unit``
    for the phase. Voltage magnitudes are comparable across grids once per-unit, but angles
    are not — each data source carries its own arbitrary global reference (the convention of
    the source EMF it was solved against, the vector-group clock a converter folded in),
    which a model cannot infer from the topology. Three frames:

    - ``"slack"`` (default) expresses every grid's state in ITS OWN slack-referenced frame:
      each row is rotated by the fundamental angle of its same-phase slack row, order ``h``
      by ``exp(-1j*h*phi)`` (:func:`injection_hse.data.slack_reference_rows` +
      :func:`injection_hse.data.slack_referenced_state`). It removes exactly the per-source offset and
      keeps the structure that IS explainable — including the clock of a transformer between
      the slack and a node, which the estimator must then reproduce.
    - ``"lv_busbar"`` references each grid at its LOCAL reference bus instead: the
      lower-voltage-side node of the in-service transformer nearest the slack
      (:func:`injection_hse.data.reference_bus_rows`), i.e. the LV busbar of the feeding substation.
      That is where a real measurement campaign puts its reference meter, so the frame is the
      one the field data arrives in, and it is independent of how the transformer above the
      busbar is modelled (its vector group no longer sits between the reference and every
      node of the grid). Choose it for a deployment-facing run, or whenever the members'
      transformer modelling is less trustworthy than their LV networks. A member with no
      in-service transformer has no such busbar and keeps the slack frame — on an
      all-LV corpus the two frames coincide.
    - ``"absolute"`` keeps the raw solver frame. Use it for a single-source study, where
      every member shares one angle convention.

    The choice must reach the data source as well (``injection_hse.data.build_multigrid_source`` passes
    it): the per-row reference table a batch carries is built for THIS frame.

    ``task`` selects WHICH per-member dataset the run trains on, the multi-grid form of the
    two-task measurement-scarcity curriculum: ``"random"`` (default) the randomized snapshots
    every member carries, ``"coherent"`` the sequence dataset generated beside them
    (``injection_hse.data.multigrid.CoherentSequenceConfig``, present only when the corpus was generated with
    a coherent section). The two tasks share the member list, the grids, the graphs and the
    grid-level split, so a two-pass curriculum is two ``ExperimentConfig``\\ s differing in this
    field and their stage ladder, continuing ONE module's weights.

    ``max_grids`` caps the number of member grids loaded per split (smoke runs / debugging;
    ``None`` = every grid the manifest lists).

    ``random_mask_unit="bus"`` samples physical buses and observes all their present
    phases together. The default ``"row"`` preserves the earlier independent node-phase
    sampling for existing runs.
    """

    per_unit: bool = True
    task: Literal["random", "coherent"] = "random"
    random_mask_unit: Literal["row", "bus"] = "row"
    angle_reference: Literal["slack", "lv_busbar", "absolute"] = "slack"
    placement_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    #: Per-feeder floor of the structural placement (``injection_hse.data.central_placement``).
    #: The default 1 meters EVERY feeder regardless of the fraction, which inflates the
    #: effective coverage far above ``placement_fraction`` on corpora of small multi-feeder
    #: members (a nominal 2 % resolved to a mean 12.8 % observed on the 555-member corpus).
    #: 0 = strict budget: ``ceil(fraction · non-slack nodes)`` meters in total (never less
    #: than one) plus the always-observed slack — the honest sparse-deployment mode. Always
    #: read the achieved coverage off the report's per-member ``observed_fraction``.
    placement_min_per_feeder: int = Field(default=1, ge=0)
    max_grids: Optional[int] = Field(default=None, gt=0)


# =============================================================================
# Data source (offline parquet / online scenario generation)
# =============================================================================
class DataConfig(_Base):
    """Where training samples come from.

    ``source="parquet"`` reads a dataset written by ``pgml.scenarios.write_dataset``
    (``dataset_path``). ``source="online"`` generates fresh batches via
    ``pgml.scenarios.run_scenarios`` from ``scenario_config_path`` (the hook for future
    hard-example augmentation). See :mod:`injection_hse.data`.

    ``split_seed`` seeds the train/val/test partition. It is DELIBERATELY separate from
    the experiment-level ``seed`` so replicate runs (different model init / mask
    draws) share one fixed partition, and so an evaluation that reloads the
    checkpointed config reproduces the exact training split. Change it to draw a
    different fold.

    ``test_fraction`` (default ``0.0``) carves a THIRD, held-out split off the seeded
    scenario permutation, disjoint by scenario from train and val. It is NEVER read by
    training, early stopping, checkpoint selection, or the curriculum — it exists only
    for a final, one-shot generalization estimate (:meth:`injection_hse.data.DataSource.test_dataset`,
    ``predict_dataset(split="test")``). At ``0.0`` the train/val partition is byte-identical
    to a config written before three-way splitting existed.

    ``topology`` turns the run into a SWITCHABLE-topology one: ``dataset_path`` is then the
    parent of one dataset directory per switch configuration (named by the variant), every
    sample is tagged with the configuration it was generated on, and the training step decodes
    it with that configuration's physics. See :class:`TopologyConfig`.

    ``keep_failed`` (default ``False``) governs non-converged scenarios: by default a
    dataset's ``failed_scenarios`` (solves that did not converge — their stored voltages
    are best-effort iterates, not solutions) are DROPPED from every split; set it ``True``
    to keep them (e.g. to study the failures). A dataset with no failures is unaffected.

    ``multigrid`` turns the run into a MULTI-GRID one: ``dataset_path`` is then the root of a
    multi-grid dataset (a ``manifest.json`` over one per-grid dataset directory each,
    :mod:`injection_hse.data.multigrid`), the split is by GRID rather than by scenario, and batches mix
    grids of different size. See :class:`MultiGridConfig`.

    ``structural_encoding`` selects the graph's node-feature variant + optional structural
    encoding (:class:`StructuralEncodingConfig`); the default leaves the graph
    byte-identical.

    ``phase_coupling`` adds the network's phase-to-phase admittance couplings to the graph's
    edge set (:class:`PhaseCouplingConfig`) — the mutual impedance of a line and the winding
    incidence of a transformer, which the per-phase-index series edges leave out. The default
    (off) leaves the graph byte-identical.
    """

    source: Literal["parquet", "online"] = "parquet"
    dataset_path: Optional[str] = None
    #: switchable topology (``None`` = one fixed topology, the grid as given). With it,
    #: ``dataset_path`` is the PARENT of one dataset directory per switch configuration.
    topology: Optional[TopologyConfig] = None
    #: multi-grid training (``None`` = one grid). With it, ``dataset_path`` is the ROOT of a
    #: multi-grid dataset (``manifest.json`` + one dataset directory per grid).
    multigrid: Optional["MultiGridConfig"] = None
    structural_encoding: StructuralEncodingConfig = Field(
        default_factory=StructuralEncodingConfig
    )
    phase_coupling: PhaseCouplingConfig = Field(default_factory=PhaseCouplingConfig)
    scenario_config_path: Optional[str] = None
    batch_size: int = Field(default=256, gt=0)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    num_workers: int = Field(default=4, ge=0)
    split_seed: int = 0
    keep_failed: bool = False


# =============================================================================
# Optimization + trainer / deployment
# =============================================================================
class OptimConfig(_Base):
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    #: Adam/AdamW denominator term (PyTorch's default 1e-8). Adam is invariant to the scale
    #: of the loss except through this term: with gradients near or below it, the update is
    #: damped by about ``|g| / (|g| + eps)``. Under measurement anchoring the per-parameter
    #: gradients of the injection decode can be of order 1e-8 or smaller, so a smaller
    #: value lets them move the weights at the nominal learning rate.
    eps: float = Field(default=1e-8, gt=0.0)
    scheduler: Literal["none", "cosine", "plateau"] = "plateau"


class TrainConfig(_Base):
    """Lightning trainer + MLflow + checkpointing (portable across dev box and cluster).

    ``precision="32"`` for the complex physics path (AMP/autocast is fragile on complex
    tensors; use ``"64"`` only for gradient verification). ``mlflow_tracking_uri`` is a local
    dir for local development or a shared/remote URI on the cluster.

    ``dtype`` is the TRAINING numeric precision of the state-estimation pipeline, DECOUPLED
    from the dataset's stored precision. Data is generated at high precision (``complex128``)
    because a feeder's harmonic Y-bus is ill-conditioned, so the small harmonic voltages the
    labels carry are only trustworthy in double precision; training that model, however, is
    usually best (and much cheaper) in single precision. ``dtype="complex64"`` (default,
    today's behaviour) casts the loaded complex state to ``complex64`` at the module boundary
    and runs a float32 model; ``dtype="complex128"`` keeps everything in double
    (float64 parameters, via Lightning double precision) for a full-precision study. Both
    combinations work against a ``complex128`` dataset. The physics decode
    (:class:`PhysicsConfig`) always solves in ``complex128`` internally regardless — only its
    inputs/outputs follow ``dtype``. A ``complex64`` dataset trained at ``complex64`` is
    byte-identical to a run written before this field existed (no cast occurs).

    ``deterministic`` is OFF (``False``) by default: a run is seeded but not
    kernel-deterministic, and CUDA's TF32 matmul path stays on wherever the runtime
    enables it — a seeded run is not on its own a bitwise-reproducible one, since TF32 and
    several common kernels round differently between runs. Set it to ``True`` to disable
    TF32 and request deterministic kernels for the whole process before any stage builds
    its ``Trainer``; an operation with no deterministic implementation then RAISES, naming
    itself, rather than silently falling back to a non-deterministic one. ``"warn"``
    downgrades that raise to a warning (best-effort determinism). Bitwise repeatability at
    a fixed ``(seed, N)`` holds only under ``True`` or a ``"warn"`` run that never actually
    hits a non-deterministic kernel.
    """

    precision: Literal["32", "64", "bf16-mixed", "16-mixed"] = "32"
    dtype: Literal["complex64", "complex128"] = "complex64"
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: Union[int, str] = "auto"
    grad_clip: float = Field(default=0.0, ge=0.0)
    # Accumulate gradients over this many micro-batches before an optimizer step, so the
    # EFFECTIVE batch is ``data.batch_size * accumulate_grad_batches``. Holding the effective
    # batch constant while the micro-batch adapts to each model's memory keeps the optimization
    # regime identical across an architecture comparison (a fair ablation); the accumulated
    # gradient equals the single-large-batch gradient (no batch-coupled op — LayerNorm only).
    accumulate_grad_batches: int = Field(default=1, ge=1)
    # Plateau early stopping (per stage), on top of the target-loss and ``max_epochs`` guards:
    # end a stage when the validation all-rows pu-MAE fails to improve by at least
    # ``early_stop_min_delta`` for ``early_stop_patience`` consecutive validation epochs.
    # ``0`` disables it (the stage then runs until the target loss or ``max_epochs``). Useful
    # when the target is set below the achievable floor so ``max_epochs`` is the only other
    # bound — the stage then stops as soon as it stops learning instead of burning the budget.
    early_stop_patience: int = Field(default=0, ge=0)
    early_stop_min_delta: float = Field(default=0.0, ge=0.0)
    #: Carry the optimizer state (AdamW moments and step counts) from one stage into the
    #: next, and from a run's ``init_from`` checkpoint into its first stage, instead of
    #: starting every stage from fresh moments. The new stage's learning rate and schedule
    #: still apply. ``False`` (default) keeps the fresh optimizer per stage.
    carry_optimizer_state: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "pgl-state-estimation"
    checkpoint_dir: str = "checkpoints"
    # show Lightning's live progress bar (per-stage train/val loss + pu-MAE in the terminal).
    # Off by default so library/test runs stay quiet; the example scripts enable it.
    enable_progress_bar: bool = False
    # Deterministic training. ``True`` asks torch for deterministic kernels
    # (``torch.use_deterministic_algorithms``, via Lightning) and disables the TF32 matmul
    # path the example runtime otherwise enables on CUDA — an op with no deterministic
    # implementation then RAISES, naming itself, so a run that claims reproducibility
    # cannot silently contain a non-deterministic kernel. ``"warn"`` downgrades that raise
    # to a warning (best-effort determinism). ``False`` (default) is today's behaviour:
    # seeded but not kernel-deterministic, TF32 on where the runtime enables it.
    deterministic: Union[bool, Literal["warn"]] = False


# =============================================================================
# Top-level experiment
# =============================================================================
class ExperimentConfig(_Base):
    """A full, reproducible training experiment (model + data + curriculum + trainer)."""

    name: str = "experiment"
    seed: int = 0
    orders: list[int] = Field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13])
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=GNNConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    curriculum: CurriculumConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _hetero_stages(self) -> "ExperimentConfig":
        # a heterogeneous random stage needs the per-order observability channels, and the
        # channel count is fixed at model construction — so the flag must be explicit.
        if not self.measurement.heterogeneous_orders and any(
            s.fundamental_fraction for s in self.curriculum.stages
        ):
            raise ValueError(
                "a curriculum stage sets fundamental_fraction but "
                "measurement.heterogeneous_orders is False; enable it so the model "
                "carries the per-order observability channels."
            )
        return self

    @model_validator(mode="after")
    def _per_order_rel_sigma(self) -> "ExperimentConfig":
        rho = self.physics.prior_injection_rel_sigma
        if isinstance(rho, list) and len(rho) != len(self.orders):
            raise ValueError(
                f"physics.prior_injection_rel_sigma has {len(rho)} values; give one per "
                f"solved order ({len(self.orders)}) or a single value."
            )
        return self

    @model_validator(mode="after")
    def _prior_boundary(self) -> "ExperimentConfig":
        # the linear-state prior is the zero-injection solve anchored to the measurements;
        # its fundamental boundary is the MEASURED slack, exactly as for the injection decode.
        if (
            "linear_state" in self.physics.input_priors
            and not self.measurement.slack_known
        ):
            raise ValueError(
                "physics.input_priors 'linear_state' requires measurement.slack_known: the "
                "zero-injection prior solve is reduced against the measured slack voltage."
            )
        return self

    @model_validator(mode="after")
    def _loss_geometry(self) -> "ExperimentConfig":
        # the preconditioner reads the injection head's output frame, which is affine in
        # the head only under the Cartesian injection encoding; the prior residual needs
        # the injection decode and the fixed placement the prior is fitted at.
        if self.loss.preconditioner != "none":
            if self.physics.decode != "injection":
                raise ValueError(
                    "loss.preconditioner needs physics.decode='injection' (it "
                    "preconditions the injection head's output)."
                )
            kind = self.encoding.injection_kind or self.encoding.kind
            if kind != "cartesian":
                raise ValueError(
                    "loss.preconditioner needs the Cartesian injection encoding "
                    "(encoding.injection_kind='cartesian'): the decode chain is affine "
                    "in the head's output only there."
                )
        if self.physics.injection_prior_residual:
            if self.physics.decode != "injection":
                raise ValueError(
                    "physics.injection_prior_residual needs physics.decode='injection' "
                    "(the prior supplies nodal injections the voltage decode has no use "
                    "for)."
                )
            if not any(s.kind == "placement" for s in self.curriculum.stages) and not (
                self.measurement.voltage_nodes
            ):
                raise ValueError(
                    "physics.injection_prior_residual needs the deployed placement the "
                    "prior is fitted at: give the curriculum a 'placement' stage or set "
                    "measurement.voltage_nodes."
                )
        return self

    @model_validator(mode="after")
    def _angle_fill_frame(self) -> "ExperimentConfig":
        # the nominal angle is zero only in the slack-referenced frame of the multi-grid
        # path, and a filled angle is a fundamental-only notion (an RMS meter has no
        # harmonic phasors to fill)
        noise = self.noise
        if noise.angle_fill_fraction > 0.0:
            multigrid = self.data.multigrid
            if multigrid is None or multigrid.angle_reference != "slack":
                raise ValueError(
                    "noise.angle_fill_fraction needs the multi-grid path in the slack "
                    "angle frame (data.multigrid.angle_reference='slack'), where the "
                    "nominal angle of every row is zero."
                )
            if list(self.orders) != [1]:
                raise ValueError(
                    "noise.angle_fill_fraction models fundamental RMS meters; it needs "
                    f"orders == [1], got {list(self.orders)}."
                )
        if noise.angle_fill_class is not None and noise.angle_fill_class not in noise.classes:
            raise ValueError(
                f"noise.angle_fill_class {noise.angle_fill_class!r} is not in noise.classes "
                f"{sorted(noise.classes)}."
            )
        return self

    @model_validator(mode="after")
    def _phase_coupling_edges(self) -> "ExperimentConfig":
        # the measured-current input is indexed by the SERIES half-edge layout (one channel
        # per branch terminal, from injection_hse.data.BranchCurrentOperator); a phase-coupling graph
        # carries extra half-edges that are not branch terminals and hold no measurement, so
        # the two edge axes would disagree. Reject rather than pad a meaningless channel.
        if self.data.phase_coupling.enabled and self.measurement.measure_current:
            raise ValueError(
                "data.phase_coupling.enabled with measurement.measure_current: the measured "
                "branch-current channels are indexed by the SERIES half-edges, while the "
                "phase-coupling graph adds mutual-admittance half-edges that carry no "
                "terminal current. Turn one of the two off."
            )
        return self

    @property
    def needs_solver(self) -> bool:
        """Whether the RUN builds the differentiable network solver (and its injection space).

        :attr:`PhysicsConfig.needs_solver` is the physics' own need — the injection decode
        solves every step, the physics-derived input priors need the same grid-constant
        operators. The PER-UNIT feature frame (:attr:`NormalizationConfig.per_unit`) adds
        one: metered branch currents are then expressed over the per-unit current base
        ``i_base(h, r) = s(h)·V_ref² / V_base(r)``
        (:meth:`injection_hse.physics.NetworkSolver.i_base`), whose operator scale ``s(h)`` only the
        assembled network carries.

        Every training entry point, the checkpoint rebuild and the topology bank build the
        solver on THIS condition — and with it fit and persist the injection normalizer, so
        the injection space travels with every checkpoint that built a solver.
        """
        return self.physics.needs_solver or (
            self.normalization.per_unit and self.measurement.measure_current
        )

    @model_validator(mode="after")
    def _multigrid_supported(self) -> "ExperimentConfig":
        """Hold the multi-grid path to its capability contract.

        A multi-grid batch folds the sample axis into the node axis and mixes grids of
        different size, so anything keyed to ONE grid's row layout without a per-member
        dispatch has no batch-wide meaning. Every field of this config is therefore
        classified as either :data:`MULTIGRID_CONSUMED` (demonstrably read on that path) or
        :data:`MULTIGRID_IGNORED` (not read, with the reason); a field of the second group
        holding a NON-DEFAULT value is rejected here, naming the field and the reason. A
        knob that changes nothing must fail loudly rather than be silently accepted.

        The model architecture is checked structurally on top: a fixed-input DNN has one
        grid's node count baked into its width. The PHYSICS features (the injection decode,
        the per-device localization head, the branch-current operator) ARE served per member —
        the operators come from a :class:`injection_hse.physics.MultiGridPhysicsBank` and the device /
        half-edge axes are concatenated by :class:`injection_hse.data.MultiGridCollate` the way the node
        axis is — with one condition of their own: the decode's fundamental boundary is the
        MEASURED slack.
        """
        if self.data.multigrid is None:
            return self
        ignored = [
            f"{path} ({reason})"
            for path, reason in MULTIGRID_IGNORED.items()
            if _holds_non_default(self, path.split("."))
        ]
        if ignored:
            raise ValueError(
                "data.multigrid trains one estimator across grids of different size; the "
                "following are set but NOT read on that path: "
                + "; ".join(ignored)
                + "."
            )
        spatial = getattr(self.model, "spatial", self.model)
        if isinstance(spatial, DNNConfig):
            raise ValueError(
                "data.multigrid does not support model.kind='dnn': its input width is one "
                "grid's node count."
            )
        if self.physics.decode == "injection" and not self.measurement.slack_known:
            raise ValueError(
                "data.multigrid with physics.decode='injection' requires "
                "measurement.slack_known: every member's fundamental solve is reduced "
                "against its own MEASURED slack voltage, and the batch carries the slack "
                "rows precisely because they are observed on every grid."
            )
        return self


# =============================================================================
# The multi-grid capability contract (what that path reads, and what it does not)
# =============================================================================
#: Dotted :class:`ExperimentConfig` paths the MULTI-GRID training path demonstrably reads
#: (verified against :meth:`injection_hse.train.SEModule._multigrid_step` and its helpers,
#: :func:`injection_hse.train.loop._train_multigrid`, :func:`injection_hse.data.build_multigrid_source`,
#: :mod:`injection_hse.data.encodings`, :mod:`injection_hse.data.noise` and the models). A path naming a
#: sub-config covers its whole subtree.
MULTIGRID_CONSUMED: frozenset[str] = frozenset(
    {
        "name",  # the MLflow run name
        "seed",  # model init + the mask / anchor draw streams
        "orders",  # the harmonic axis of the model, the loss and the metrics
        # -- data ---------------------------------------------------------------
        "data.dataset_path",  # the corpus root
        "data.multigrid.per_unit",
        "data.multigrid.task",
        "data.multigrid.random_mask_unit",
        "data.multigrid.angle_reference",
        "data.multigrid.placement_fraction",
        "data.multigrid.placement_min_per_feeder",
        "data.multigrid.max_grids",
        "data.structural_encoding.kind",
        "data.structural_encoding.base",
        "data.structural_encoding.rwse_steps",
        "data.structural_encoding.lap_pe_dim",
        "data.structural_encoding.normalize",
        "data.phase_coupling.enabled",
        "data.phase_coupling.tolerance",
        "data.batch_size",
        "data.num_workers",
        "data.keep_failed",
        # -- the model architecture (built by injection_hse.models.build_model either way) --
        "model",
        # -- encoding / normalization -------------------------------------------
        "encoding.kind",
        "encoding.injection_kind",
        "encoding.time_features",
        "normalization.per_harmonic",
        "normalization.magnitude_transform",
        # -- loss ----------------------------------------------------------------
        "loss.harmonic_weight.mode",
        "loss.harmonic_weight.values",
        "loss.magnitude_term_weight",
        "loss.predict_injection",
        "loss.injection_loss_weight",
        "loss.injection_hidden_dim",
        "loss.injection_state_loss_split",
        "loss.injection_state_loss_normalize",
        "loss.plain_decode_weight",
        # -- physics -------------------------------------------------------------
        "physics.decode",
        "physics.anchor_voltage_weight",
        "physics.anchor_current_weight",
        "physics.anchor_weight_draw",
        # the noise-calibrated anchor weights are installed on the operator bank from the
        # corpus-wide per-unit scales and the uniform default-class sigmas
        # (MultiGridPhysicsBank.set_anchor_calibration)
        "physics.anchor_weighting",
        "physics.prior_injection_rel_sigma",
        "physics.clean_sensor_rel_sigma",
        "physics.operator_cache_mb",
        # every member's solver is built with the run's harmonic-slack mode
        # (MultiGridPhysicsBank.harmonic_slack) and the decode hands it the member's
        # measured slack voltage at every order under "measured" (_member_slack_voltage)
        "physics.harmonic_slack",
        # every member's Y(h) is assembled with the run's decode shunt model
        # (MultiGridPhysicsBank.load_shunt)
        "physics.load_shunt",
        # the physics-derived input priors are computed per member through the operator
        # bank (each member's own Y(h) / anchored zero-injection solve, in the per-unit
        # current base and the run's angle frame — SEModule._multigrid_input_priors)
        "physics.input_priors",
        "physics.prior_anchor_voltage_weight",
        "physics.prior_anchor_current_weight",
        # -- measurement ---------------------------------------------------------
        "measurement.measure_current",
        "measurement.current_channels",
        "measurement.measure_power",
        "measurement.slack_known",
        # -- noise (the uniform default-class model) ------------------------------
        "noise.enabled",
        "noise.seed",
        "noise.default_class",
        "noise.classes",
        "noise.harmonic_range",
        # the device of the noise draws applies to the uniform model as to the per-row one
        # (SEModule builds both and switches them alike)
        "noise.draw_device",
        # the magnitude-only (nominal-angle-filled) meter class, drawn per row and sample
        # in the slack-referenced per-unit frame (SensorNoise.uniform / apply)
        "noise.angle_fill_fraction",
        "noise.angle_fill_class",
        # -- curriculum ----------------------------------------------------------
        "curriculum.stages.name",
        "curriculum.stages.kind",
        "curriculum.stages.observed_fraction",
        "curriculum.stages.target_loss",
        "curriculum.stages.max_epochs",
        "curriculum.stages.lr",
        "curriculum.stages.scheduler",
        "curriculum.stages.warmup_epochs",
        "curriculum.stages.anchor_voltage_weight",
        "curriculum.stages.anchor_current_weight",
        "curriculum.stages.anchor_voltage_range",
        "curriculum.stages.anchor_current_range",
        # -- optimizer / trainer --------------------------------------------------
        "optim.name",
        "optim.lr",
        "optim.weight_decay",
        "optim.eps",
        "optim.scheduler",
        "train.precision",
        "train.dtype",
        "train.accelerator",
        "train.devices",
        "train.grad_clip",
        "train.accumulate_grad_batches",
        "train.early_stop_patience",
        "train.early_stop_min_delta",
        "train.carry_optimizer_state",
        "train.mlflow_tracking_uri",
        "train.mlflow_experiment",
        "train.checkpoint_dir",
        "train.enable_progress_bar",
        "train.deterministic",
    }
)

#: Dotted :class:`ExperimentConfig` paths the multi-grid path does NOT read, each with the
#: reason. A non-default value is rejected by :meth:`ExperimentConfig._multigrid_supported`
#: — the alternative (silent acceptance) is what makes a knob look applied when it is not.
MULTIGRID_IGNORED: dict[str, str] = {
    "data.source": (
        "a multi-grid run always loads the corpus root through "
        "injection_hse.data.build_multigrid_source; the parquet/online switch is never consulted"
    ),
    "data.scenario_config_path": (
        "online scenario generation has no multi-grid form; a corpus is generated ahead "
        "of the run by injection_hse.data.multigrid"
    ),
    "data.topology": (
        "a switchable topology reads dataset_path as its own per-variant layout, and its "
        "operators are keyed to ONE grid's row layout"
    ),
    "data.val_fraction": (
        "the split is by GRID and comes from the corpus manifest, not from a per-scenario "
        "fraction — every scenario of a train member is training data"
    ),
    "data.test_fraction": (
        "the held-out split is the corpus manifest's test group (the real grids), not a "
        "scenario fraction carved out of each member"
    ),
    "data.split_seed": (
        "the grid-level split is fixed when the corpus is generated; there is no scenario "
        "permutation left to reseed"
    ),
    "measurement.voltage_nodes": (
        "the sensor set is per member — each member's installed measurement devices, else "
        "data.multigrid.placement_fraction — so a node name has no cross-grid meaning"
    ),
    "measurement.fundamental_nodes": (
        "fundamental-only monitors need the per-order observability channels, which are "
        "drawn per grid segment and not modelled by the multi-grid masker"
    ),
    "measurement.heterogeneous_orders": (
        "per-order observability is drawn per grid segment, which the multi-grid masker "
        "does not model"
    ),
    "physics.calibrate": (
        "a calibration knob corrects ONE grid's parameters and re-assembles that grid's "
        "operator, which has no meaning across a bank of member grids; calibrate on a "
        "single-grid run"
    ),
    "physics.calibration_weight": (
        "it weights the learned grid calibration's prior, which the multi-grid path never "
        "builds (see physics.calibrate)"
    ),
    "loss.supervise_rows": (
        "the multi-grid step supervises every row of the batch's member graphs (its voltage "
        "loss is called without a row mask); restricting it to the unobserved rows is not "
        "wired into that step"
    ),
    "loss.plain_decode_observed_weight": (
        "the multi-grid step evaluates the plain-decode term on every row of the batch's "
        "member graphs without row weights; weighting the observed rows is not wired into "
        "that step"
    ),
    "loss.prior_distillation_weight": (
        "the distillation teacher is the LINEAR INJECTION PRIOR, a closed-form map from ONE "
        "grid's fixed sensor placement onto that grid's node-phase injection rows "
        "(injection_hse.analysis.fit_linear_injection_prior refuses a multi-grid module); a batch "
        "mixing grids of different size has no such map"
    ),
    "curriculum.stages.prior_distillation_weight": (
        "it anneals the prior-distillation term, which the multi-grid path never forms "
        "(see loss.prior_distillation_weight)"
    ),
    "loss.preconditioner": (
        "the preconditioner is built from ONE grid's decode map W·G·C (a fixed row layout "
        "and one fixed placement); a batch mixing grids of different size has no such map"
    ),
    "loss.preconditioner_floor": (
        "it floors the preconditioner's spectrum, which the multi-grid path never builds "
        "(see loss.preconditioner)"
    ),
    "physics.exclude_tied_rows": (
        "the multi-grid decode always zeroes each member's switch-tied rows (the bank's "
        "solvers carry them); the False escape hatch is a single-grid comparison aid"
    ),
    "physics.injection_prior_residual": (
        "the residual's prior is the LINEAR INJECTION PRIOR of ONE grid's fixed placement "
        "(see loss.prior_distillation_weight); the multi-grid path has none"
    ),
    "physics.injection_prior_fit_states": (
        "it sizes the fit of the LINEAR INJECTION PRIOR, which the multi-grid path never "
        "builds (see loss.prior_distillation_weight)"
    ),
    "curriculum.stages.fundamental_fraction": (
        "demoting observed rows to fundamental-only needs the per-order observability "
        "channels, which the multi-grid masker does not model"
    ),
    "noise.instrument": (
        "the datasheet error model refers each reading to its channel's nominal voltage "
        "and CT range in physical units; the per-unit multi-grid batch has no CT range "
        "and draws the uniform accuracy-class model"
    ),
    "normalization.per_unit": (
        "the multi-grid path is always per-unit (data.multigrid.per_unit); its per-row base "
        "voltage travels with the batch and the frame is applied in the multi-grid step"
    ),
}


def _model_types(annotation: Any) -> list[type[BaseModel]]:
    """Every pydantic model reachable through a field annotation.

    Unwraps ``Optional`` / ``Union`` / ``list`` / ``Annotated`` so a discriminated union or
    a list of sub-configs is walked like a plain sub-config. Returns an empty list for a
    scalar (leaf) annotation.
    """
    found: list[type[BaseModel]] = []
    stack = [annotation]
    while stack:
        ann = stack.pop()
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            found.append(ann)
            continue
        stack.extend(get_args(ann))
    return found


def unclassified_multigrid_fields(
    model_cls: type[BaseModel] = ExperimentConfig, prefix: str = ""
) -> list[str]:
    """Config fields in neither :data:`MULTIGRID_CONSUMED` nor :data:`MULTIGRID_IGNORED`.

    Walks ``model_cls`` recursively (through sub-configs, unions and lists of sub-configs)
    and returns the dotted paths of every LEAF whose multi-grid behaviour is undeclared —
    empty for a complete contract. A classified path stops the descent, so naming a
    sub-config covers its whole subtree.
    """
    missing: list[str] = []
    for name, field in model_cls.model_fields.items():
        path = f"{prefix}{name}"
        if path in MULTIGRID_CONSUMED or path in MULTIGRID_IGNORED:
            continue
        subs = _model_types(field.annotation)
        if subs:
            for sub in subs:
                missing.extend(unclassified_multigrid_fields(sub, f"{path}."))
        else:
            missing.append(path)
    return sorted(set(missing))


def _holds_non_default(obj: BaseModel, parts: list[str]) -> bool:
    """Whether the dotted ``parts`` path of ``obj`` holds a value other than its default.

    A ``None`` sub-config short-circuits to ``False`` (there is nothing set below it); a
    list of sub-configs is non-default when ANY element's field is.
    """
    name, rest = parts[0], parts[1:]
    field = type(obj).model_fields.get(name)
    if field is None:  # pragma: no cover - guarded by the completeness test
        return False
    value = getattr(obj, name)
    if not rest:
        return value != field.get_default(call_default_factory=True)
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_holds_non_default(v, rest) for v in value)
    return _holds_non_default(value, rest)


# =============================================================================
# Default CIGRE LV curriculum + measurement (the benchmark scarcity ladder)
# =============================================================================
def cigre_lv_measurement() -> MeasurementConfig:
    """Default 2-sensor placement for the full 3-phase CIGRE LV benchmark.

    Junction nodes "Bus R4" (residential feeder) + "Bus C5" (commercial feeder) carry a PQ
    voltage device; the MV/LV slack voltage is a known boundary condition. (The CIGRE LV
    nodes are named "Bus R*/C*/I*"; pass ids or other names to override.)
    """
    return MeasurementConfig(voltage_nodes=["Bus R4", "Bus C5"], slack_known=True)


def cigre_lv_curriculum(
    *,
    target_full: float = 1e-3,
    target_placement: float = 1e-2,
    max_epochs: int = 200,
) -> CurriculumConfig:
    """The measurement-scarcity ladder for the ~132-row CIGRE LV: full -> sparse -> placement.

    Stages: full observability, then per-sample random masking at observed fractions
    0.5 -> 0.25 -> 0.1, then the fixed 2-sensor ``placement`` finetune
    (:func:`cigre_lv_measurement`). Per-stage ``target_loss`` is the validation per-unit
    UNOBSERVED MAE early-stop target, ramping from a tight full-observability target up to the
    relaxed final placement target (fewer observed nodes -> a higher achievable floor).
    Defaults are the production ladder; relax ``target_*`` / ``max_epochs`` for smoke tests.
    """
    # geometric ramp of the per-stage early-stop target from full -> placement
    ladder = [
        ("full", "full", None),
        ("frac_50", "random", 0.5),
        ("frac_25", "random", 0.25),
        ("frac_10", "random", 0.1),
        ("placement", "placement", None),
    ]
    n = len(ladder)
    stages = []
    for i, (name, kind, frac) in enumerate(ladder):
        # interpolate target in log-space from target_full to target_placement
        t = i / (n - 1)
        target = target_full * (target_placement / target_full) ** t
        stages.append(
            MaskStageConfig(
                name=name,
                kind=kind,
                observed_fraction=frac,
                target_loss=target,
                max_epochs=max_epochs,
            )
        )
    return CurriculumConfig(stages=stages)


__all__ = [
    "EncodingConfig",
    "StructuralEncodingConfig",
    "PhaseCouplingConfig",
    "NormalizationConfig",
    "HarmonicWeightConfig",
    "LossConfig",
    "PhysicsConfig",
    "CalibrationKnobConfig",
    "MeasurementConfig",
    "AccuracyClassNoise",
    "InstrumentBand",
    "InstrumentErrorConfig",
    "NoiseConfig",
    "MaskStageConfig",
    "CurriculumConfig",
    "DNNConfig",
    "GNNConfig",
    "GraphormerConfig",
    "TemporalConfig",
    "SpatialModelConfig",
    "ModelConfig",
    "DataConfig",
    "TopologyVariantConfig",
    "TopologyConfig",
    "MultiGridConfig",
    "OptimConfig",
    "TrainConfig",
    "ExperimentConfig",
    "MULTIGRID_CONSUMED",
    "MULTIGRID_IGNORED",
    "unclassified_multigrid_fields",
    "cigre_lv_measurement",
    "cigre_lv_curriculum",
]


def _example() -> ExperimentConfig:
    """A representative, valid experiment: a temporal GNN on the CIGRE LV scarcity ladder."""
    return ExperimentConfig(
        name="example",
        model=TemporalConfig(spatial=GNNConfig()),
        data=DataConfig(source="parquet", dataset_path="data/se_coherent/dataset"),
        measurement=cigre_lv_measurement(),
        curriculum=cigre_lv_curriculum(),
    )


if __name__ == "__main__":  # `python -m injection_hse.config --json-schema | --example`
    import argparse
    import json
    import sys

    import yaml

    ap = argparse.ArgumentParser(
        prog="python -m injection_hse.config",
        description="Inspect the injection_hse ExperimentConfig: its JSON Schema or an example YAML.",
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--json-schema", action="store_true", help="Print the config JSON Schema."
    )
    grp.add_argument(
        "--example", action="store_true", help="Print a valid example config as YAML."
    )
    ns = ap.parse_args()
    if ns.json_schema:
        json.dump(ExperimentConfig.model_json_schema(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(yaml.safe_dump(_example().model_dump(), sort_keys=False))
