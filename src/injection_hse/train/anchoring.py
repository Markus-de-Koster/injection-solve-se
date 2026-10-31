"""Measurement-anchoring weights: what the solver is BUILT for, the per-batch draw, and the
noise calibration.

The anchored injection decode turns the observed voltages / branch currents into soft
constraints on the network solve. The grid-constant anchoring tensors are built ONCE, sized
by the largest weight any curriculum stage will request (:func:`anchor_build_weights`);
each stage then dials its own runtime scalar, optionally randomized per training batch
(:func:`_anchor_weight_from`). Under ``PhysicsConfig.anchor_weighting="noise_calibrated"``
those scalars multiply per-channel inverse-variance weights derived from the fitted scales
and the sensor-noise model (:func:`install_anchor_calibration`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .module import SEModule


def anchor_build_weights(
    physics, curricula, *, measure_current: bool = False
) -> tuple[float, float]:
    """The (voltage, current) anchoring weights the SOLVER must be BUILT for.

    Per-stage overrides can raise the anchor above the global :class:`PhysicsConfig` value, and
    the grid-constant anchoring tensors (``NetworkSolver._v_scale`` / ``_c_map``) are built once
    at construction — sized by whether ANY stage of ANY task pass will anchor. Returns the max
    over the global weights and every stage override in ``curricula`` (a list of
    :class:`CurriculumConfig`), so the solver is built with the capability the schedule needs;
    each stage then dials its own runtime scalar via ``SEModule.set_stage_anchor``.

    The ``"linear_state"`` input prior (:attr:`~injection_hse.config.PhysicsConfig.input_priors`)
    anchors its own zero-injection solve with ``prior_anchor_voltage_weight`` /
    ``prior_anchor_current_weight`` whatever the schedule does, so those are folded into the
    maximum too — the current one only when ``measure_current`` (there is no branch-current
    map to build otherwise, and a solver asked to anchor currents without one raises).
    """
    v = float(physics.anchor_voltage_weight)
    i = float(physics.anchor_current_weight)
    if "linear_state" in physics.input_priors:
        v = max(v, float(physics.prior_anchor_voltage_weight))
        if measure_current:
            i = max(i, float(physics.prior_anchor_current_weight))
    for cur in curricula:
        for stage in cur.stages:
            if stage.anchor_voltage_weight is not None:
                v = max(v, float(stage.anchor_voltage_weight))
            if stage.anchor_current_weight is not None:
                i = max(i, float(stage.anchor_current_weight))
            if stage.anchor_voltage_range is not None:
                v = max(v, float(stage.anchor_voltage_range[1]))
            if stage.anchor_current_range is not None:
                i = max(i, float(stage.anchor_current_range[1]))
    return v, i


#: Decades a ``log_uniform`` anchor draw spans below its range maximum. Three decades put
#: roughly half the draws below a tenth of the maximum weight, which is the regime where the
#: predicted injections still move the solved state and can therefore be learned.
ANCHOR_LOG_DECADES = 3.0


def _anchor_weight_from(kind: str):
    """The map ``(range, u in [0,1]) -> weight`` for a stage's per-batch randomization.

    ``uniform`` interpolates the weight linearly. ``log_uniform`` interpolates its LOGARITHM
    between the range maximum and :data:`ANCHOR_LOG_DECADES` decades below it, so the weak
    anchor settings — the only ones whose gradient reaches the injection prediction — are
    drawn as often as the strong ones. A range that already starts above zero keeps its own
    lower bound; ``low = 0`` cannot be reached on a log scale and becomes that floor.
    """
    if kind == "uniform":
        return lambda r, u: r[0] + (r[1] - r[0]) * u
    if kind != "log_uniform":
        raise ValueError(f"unknown anchor_weight_draw {kind!r}.")

    def _log_uniform(r, u):
        high = float(r[1])
        if high <= 0.0:
            return high
        low = float(r[0]) or high * 10.0**-ANCHOR_LOG_DECADES
        return float(math.exp(math.log(low) + u * (math.log(high) - math.log(low))))

    return _log_uniform


def _anchor_ranges(v_range, i_range):
    """Normalize a stage's optional anchor randomization ranges (``None`` = fixed weights)."""
    if v_range is None and i_range is None:
        return None
    return (
        tuple(map(float, v_range)) if v_range is not None else None,
        tuple(map(float, i_range)) if i_range is not None else None,
    )


# =============================================================================
# Noise-calibrated anchor weights: from the fitted scales + the noise model to the solver
# =============================================================================
def _channel_sigma(
    noise, typical: Tensor, clean_rel_sigma: float, *, base: Optional[Tensor] = None
) -> Tensor:
    """float64 ``[H, R]`` per-real-component measurement sigma of one channel class.

    From the :class:`~injection_hse.data.SensorNoise` model's own sigmas when the model exists
    (:func:`injection_hse.physics.anchor_measurement_sigma`), else the clean-sensor floor
    (:func:`injection_hse.physics.clean_sensor_sigma`, ``[H, 1]``).

    ``base`` (broadcastable against ``[H, R]``) is the per-channel unit base when ``typical``
    is a FEATURE-frame scale while the sensor model measures in PHYSICAL units — the
    single-grid per-unit frame (:attr:`injection_hse.config.NormalizationConfig.per_unit`), whose noise
    model perturbs volts and amperes. The reading errors are RELATIVE and scale with the
    base, the instrument floor is ABSOLUTE and does not, so the floor is converted into the
    feature frame before the combination and the whole sigma back into physical units after
    it. ``None`` = the scale and the noise model already share a frame.

    An :class:`~injection_hse.data.InstrumentNoise` model decides the regime by the typical magnitude
    in its own (physical) units, so the scale is converted into them first and the sigma is
    its marginal per-reading value (:meth:`~injection_hse.data.InstrumentNoise.reading_sigma`).
    """
    from ..data.instrument import InstrumentNoise
    from ..physics import anchor_measurement_sigma, clean_sensor_sigma

    if isinstance(noise, InstrumentNoise):
        typ = torch.as_tensor(typical, dtype=torch.float64)
        typ = typ.reshape(-1, 1) if typ.ndim < 2 else typ  # [H, 1] | [H, R]
        if base is not None:
            typ = typ * torch.as_tensor(base, dtype=torch.float64)
        return noise.reading_sigma(typ)
    if noise is None:
        sigma = clean_sensor_sigma(typical, clean_rel_sigma)  # [H, 1]
    else:
        floor = noise.floor_sigma
        if floor is not None and base is not None:
            floor = torch.as_tensor(floor, dtype=torch.float64) / base
        sigma = anchor_measurement_sigma(
            typical,
            mag_sigma=noise.mag_sigma,
            phase_sigma_rad=noise.phase_sigma_rad,
            floor_sigma=floor,
            order_floor_scale=noise.order_floor_scale,
        )
    return sigma if base is None else sigma * base


def anchor_calibration_sigmas(
    physics,
    *,
    state_scale: Tensor,
    injection_scale: Tensor,
    edge_scale: Optional[Tensor] = None,
    voltage_noise=None,
    current_noise=None,
    injection_base: Optional[Tensor] = None,
    voltage_base: Optional[Tensor] = None,
    current_base: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """The three sigma tensors of the noise-calibrated anchor weights.

    Returns ``(sigma_injection [H] | [H, N], sigma_voltage [H, N] | [H, 1], sigma_current
    [H, E] | [H, 1] | None)``, float64 on the CPU, in the units of the scales given:

    - ``sigma_injection(h) = ρ(h) · injection_scale(h)`` — the prior error of the predicted
      injection (``ρ`` = ``prior_injection_rel_sigma``, one value or one per order;
      ``injection_scale`` = the injection normalizer's ``cartesian_scale``, the training-set
      RMS ``|Y(h)·V_true(h)|``);
    - ``sigma_voltage`` from ``voltage_noise`` (a :class:`~injection_hse.data.SensorNoise` over the
      rows) at the typical magnitude ``state_scale`` — ``[H]`` pooled over the rows, or
      ``[H, N]`` each row's own (:func:`channel_scale`) — or
      ``clean_sensor_rel_sigma · state_scale`` without a noise model;
    - ``sigma_current`` the same for the CT channels with ``current_noise`` and
      ``edge_scale`` (the edge normalizer's ``cartesian_scale``); ``None`` when no
      ``edge_scale`` is given (no metered currents).

    The three ``*_base`` arguments (``[H, N]`` / ``[N]`` / ``[H, E]``, or ``None``) carry the
    per-unit frame: the scales are then FEATURE-frame quantities and the returned sigmas are
    converted into the physical units the solver anchors in, the reading errors scaling with
    the base and the instrument floor not (see :func:`_channel_sigma`). All ``None`` = the
    scales and the noise model already live in the solver's units.

    ``physics`` is the :class:`~injection_hse.config.PhysicsConfig`.
    """
    clean = float(physics.clean_sensor_rel_sigma)
    s_inj = torch.as_tensor(injection_scale, dtype=torch.float64).cpu().reshape(-1)
    rho = torch.as_tensor(
        physics.prior_injection_rel_sigma, dtype=torch.float64
    ).reshape(-1)
    if rho.numel() not in (1, s_inj.numel()):
        raise ValueError(
            f"prior_injection_rel_sigma has {rho.numel()} values for {s_inj.numel()} orders."
        )
    s_v = _scale_tensor(state_scale)
    sigma_injection = rho * s_inj
    if injection_base is not None:
        sigma_injection = sigma_injection.unsqueeze(-1) * injection_base  # [H, N]
    sigma_voltage = _channel_sigma(voltage_noise, s_v, clean, base=voltage_base)
    sigma_current = None
    if edge_scale is not None:
        s_e = _scale_tensor(edge_scale)
        sigma_current = _channel_sigma(current_noise, s_e, clean, base=current_base)
    return sigma_injection, sigma_voltage, sigma_current


def _scale_tensor(scale) -> Tensor:
    """A typical-magnitude scale as float64 on the CPU: ``[H]`` pooled or ``[H, R]``."""
    s = torch.as_tensor(scale, dtype=torch.float64).cpu()
    return s.reshape(-1) if s.ndim < 2 else s


#: A channel's own typical magnitude is floored at this fraction of the pooled one, so a
#: channel that is (nearly) zero over the training data cannot become an infinitely trusted
#: anchor.
CHANNEL_SCALE_FLOOR = 1e-3


def channel_scale(normalizer) -> Tensor:
    """The typical reading magnitude the anchor calibration evaluates the sensor error at.

    ``[H, R]`` — each row's (half-edge's) own RMS magnitude per order
    (:attr:`injection_hse.normalization.HarmonicNormalizer.row_scale`), floored at
    :data:`CHANNEL_SCALE_FLOOR` times the pooled scale — when the normalizer recorded it,
    else the pooled ``cartesian_scale`` ``[H]`` (checkpoints written before the per-row
    scale existed keep the calibration they were trained with). One pooled magnitude per
    order mixes rows of different voltage classes and loadings: at the fundamental the
    0.4 kV voltage readings of a feeder below a 20 kV slack were assumed an order of
    magnitude noisier than they are.
    """
    pooled = normalizer.cartesian_scale.detach().to(torch.float64).cpu().reshape(-1)
    rows = getattr(normalizer, "row_scale", None)
    if rows is None:
        return pooled
    rows = rows.detach().to(torch.float64).cpu()
    return torch.maximum(rows, CHANNEL_SCALE_FLOOR * pooled.unsqueeze(-1))


def _per_unit_calibration_bases(module: "SEModule") -> Optional[dict]:
    """The ``*_base`` arguments of :func:`anchor_calibration_sigmas` for a per-unit module.

    ``None`` unless the single-grid per-unit feature frame is active
    (:attr:`injection_hse.config.NormalizationConfig.per_unit`), in which case the fitted scales are
    dimensionless and each needs its own base to reach the solver's units: the per-unit
    current base ``i_base[h, r]`` for the injection prior, the per-row voltage base for the
    measured voltage rows, and ``i_base`` at each CT's source row for the current channels.
    """
    i_base = module._injection_base()
    if i_base is None:
        return None
    v_base = module.base_v.to(torch.float64).cpu()
    # a row with no rated voltage keeps base 1, exactly as injection_hse.data.per_unit_state does (a
    # zero base would be an infinitely trusted anchor channel)
    v_base = torch.where(v_base > 0, v_base, torch.ones_like(v_base))
    bases = {"injection_base": i_base.cpu(), "voltage_base": v_base}
    if module.edge_normalizer is not None and module._measure_current:
        bases["current_base"] = module._edge_current_base().cpu()
    return bases


def install_anchor_calibration(module: "SEModule") -> bool:
    """Install the noise-calibrated anchor weights on a module's network solver(s).

    Reads ``module.cfg.physics`` and, under ``anchor_weighting="noise_calibrated"``, builds
    the calibration (:func:`anchor_calibration_sigmas`) from the module's fitted normalizers
    (state / injection / edge ``cartesian_scale``) and its sensor-noise models (``_noise`` for
    the voltage rows, ``_edge_noise`` for the CT channels; ``None`` = clean sensors), then
    installs it on every solver the module decodes with — each switch configuration's
    solver, or the multi-grid operator bank
    (:meth:`injection_hse.physics.MultiGridPhysicsBank.set_anchor_calibration`, in the per-unit
    feature frame). A single-grid run in the per-unit frame
    (:attr:`injection_hse.config.NormalizationConfig.per_unit`) converts the feature-frame scales into
    the solver's units here (:func:`_per_unit_calibration_bases`), which is the same
    conversion the bank performs per member. Called once at module construction, AFTER the
    noise model exists; because
    the config, the normalizers and the noise config travel with every checkpoint, a reload
    re-derives the identical calibration. Returns whether a calibration was installed
    (``False`` for manual weighting or a module without the injection decode).
    """
    physics = module.cfg.physics
    if physics.anchor_weighting != "noise_calibrated":
        return False
    solvers = tuple(module._solvers)
    bank = module._bank
    if not solvers and bank is None:
        return False
    if module.inj_normalizer is None:
        raise ValueError(
            "physics.anchor_weighting='noise_calibrated' needs the fitted injection "
            "normalizer (its per-order scale sets the injection prior sigma)."
        )
    # the typical reading magnitudes: each channel's own on a single grid; the multi-grid
    # bank's noise model is uniform over members whose rows differ, so it keeps the pool
    if bank is None:
        state_scale = channel_scale(module.normalizer)
        edge_scale = (
            channel_scale(module.edge_normalizer)
            if module.edge_normalizer is not None
            else None
        )
    else:
        state_scale = module.normalizer.cartesian_scale
        edge_scale = (
            module.edge_normalizer.cartesian_scale
            if module.edge_normalizer is not None
            else None
        )
    # The prior sigma reads the DEVICE-ROW injection RMS (``aux_scale``) where the fit
    # recorded one (:func:`injection_hse.physics._prior_scale_rows`): the slack row's Y·V entry is the
    # total feeder current, and a passive junction's is zero, so pooling either into the
    # statistic misstates how large a device injection is. The recorded value is used as fit —
    # never recomputed — so a checkpoint keeps the scale its own run was calibrated with;
    # older checkpoints without it fall back to the pooled feature scale.
    inj_scale = (
        module.inj_normalizer.aux_scale
        if module.inj_normalizer.aux_scale is not None
        else module.inj_normalizer.cartesian_scale
    )
    # THE FRAME. The fitted scales live in the run's FEATURE frame while the solve anchors in
    # volts and amperes. A single-grid per-unit run (``normalization.per_unit``) therefore
    # converts each scale with this grid's own bases here — the per-unit current base for the
    # injection prior and the CT channels, the per-row voltage base for the voltage rows. What
    # survives in the weight `(σ_I/σ_meas)²` is the RATIO of the two bases at each row, which
    # is the grid's operator scale on a single-voltage-class grid (the frame then leaves the
    # anchoring untouched) and otherwise carries each row's voltage class the way a per-unit
    # weighting must (:meth:`injection_hse.physics.NetworkSolver.i_base`). A multi-grid run hands the
    # feature-frame sigmas to the bank, which converts per member.
    per_unit_bases = _per_unit_calibration_bases(module) if bank is None else None
    sigma_injection, sigma_voltage, sigma_current = anchor_calibration_sigmas(
        physics,
        state_scale=state_scale,
        injection_scale=inj_scale,
        edge_scale=edge_scale,
        voltage_noise=module._noise,
        current_noise=module._edge_noise,
        **(per_unit_bases or {}),
    )
    if bank is not None:
        # the multi-grid noise model is uniform (size-1 sigmas), so the per-order sigmas
        # are ``[H, 1]``; the bank scales them per member with that member's bases.
        bank.set_anchor_calibration(
            sigma_injection=sigma_injection,
            sigma_voltage=sigma_voltage.reshape(-1),
            sigma_current=(
                None if sigma_current is None else sigma_current.reshape(-1)
            ),
            voltage_per_unit=bool(module._per_unit),
        )
        return True
    for solver in solvers:
        solver.set_anchor_calibration(
            sigma_injection=sigma_injection,
            sigma_voltage=sigma_voltage,
            sigma_current=sigma_current,
        )
    return True
