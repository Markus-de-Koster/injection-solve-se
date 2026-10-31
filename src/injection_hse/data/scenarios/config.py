"""Scenario composition configs: how a training batch's devices and profiles are drawn.

These are the serializable recipes on top of :mod:`pgml.scenarios`. pgml declares what a
solver input can be — which quantity varies, over what distribution, at which component —
and this module declares the POPULATION behind a state-estimation training set: what a
device class emits as a function of its loading, how many of each class sit behind an
aggregated meter point, and how their activity moves over a day.

Three configs, in the order a sequence is built:

- :class:`LoadProfileConfig` — the multi-scale fundamental recurrence (seasonal, weekly,
  diurnal, short-term) that a profiled device's P/Q follows.
- :class:`CompositionConfig` over :class:`DeviceClassSpec` / :class:`ConsumerComposition` —
  an aggregated load as a statistical roster of member devices whose per-step activity
  drives BOTH its fundamental power and its injected spectrum.
- :class:`CoherentSpectrumConfig` — the per-device harmonic fingerprint over a step
  sequence, which ties the two together and is what
  :func:`~pgml.scenarios.run_scenarios` solves.

The measured numbers in the class bands below come from the emission study this package
exists for; they are calibration, not standards. The standards themselves (IEC 61000-3-2,
EN 50160) stay in pgml (:func:`pgml.scenarios.iec61000_3_2_fraction`), continued to every
order by :mod:`injection_hse.data.scenarios.emission_caps`, so a composed aggregate and a randomized
draw are bounded by the same physical envelope, and both follow the one load-dependent
emission law of :mod:`injection_hse.data.scenarios.emission`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar, Literal, Optional

from pgml.scenarios import (
    BackgroundHarmonicConfig,
    Distribution,
    LatentFactor,
    ParameterSpec,
    Selector,
    Uniform,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

if TYPE_CHECKING:  # the samplers import this module, so the batch type is a forward ref
    from pgml.scenarios import SampledScenarios


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Optional fields left out of the serialized form while they are ``None``. A field
    #: added to a config after datasets were written with it is listed here, so a config
    #: that does not use it serializes, and therefore hashes (``config_hash``), unchanged.
    _OMIT_WHEN_NONE: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_unset_optional(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if isinstance(data, dict):
            for key in self._OMIT_WHEN_NONE:
                if getattr(self, key) is None:
                    data.pop(key, None)
        return data


# =============================================================================
# Time-varying fundamental load profiles (multi-scale synthetic recurrence)
# =============================================================================
class LoadProfileConfig(_Base):
    """Multi-scale synthetic load-profile generator for a coherent step sequence.

    Turns the constant per-scenario fundamental of a
    :class:`CoherentSpectrumConfig` into a per-STEP time series: every matched
    device gets a multiplicative factor composed on four time scales,

        ``factor(t) = f_seasonal(t) * f_weekly(t) * f_daily(t) * f_short(t)``,

    applied to the device's per-scenario base ``P`` and ``Q`` TOGETHER (constant
    power factor, the ``field="pq"`` semantics). The base is the device's sampled
    per-scenario operating point (from ``CoherentSpectrumConfig.parameters``) if it
    has one, else its nominal ``p_nom_w`` / ``q_nom_var``. Gradients still flow to a
    tensor-valued base (the factor is a plain, off-tape multiplier).

    The daily SHAPE is class-aware, keyed by each device's ``consumer_type``
    (household evening peak, office/commercial business hours, EV evening charging,
    a flat industrial plateau, and a neutral default). ``pv`` is special: a solar
    bell that is ZERO at night, with a daylight window and amplitude that widen in
    summer (the seasonal modulation folds into the daily bell, not ``f_seasonal``).
    The concrete shapes live in :mod:`injection_hse.data.scenarios.profiles`; this config sets
    their amplitudes and the stochastic ranges.

    Correlation model (per-scenario co-variation)
    ---------------------------------------------
    Two per-scenario SHARED latents make devices co-vary within a scenario:

    - a ``behavioral`` latent scales the DAILY AMPLITUDE of every non-``pv`` device
      (a busy day lifts everyone's daily swing together), coupling strength
      ``behavioral_coupling``;
    - a ``cloudiness`` latent scales every ``pv`` device's output together
      (an overcast day dims all panels), coupling strength ``cloud_coupling``.

    On top of the shared latents each device draws IDIOSYNCRATIC per-scenario values:
    an overall level (``level_min`` .. ``level_max``), an amplitude multiplier
    (``amplitude_jitter_min`` .. ``amplitude_jitter_max``), a daily phase offset
    (``+/- phase_offset_hours``, so devices do not all peak at the same instant), and
    a per-step AR(1) short-term term (``short_rho`` stickiness, ``short_sigma``
    std). The composed factor is clamped to be non-negative.

    The generator draws on an RNG stream DERIVED from ``CoherentSpectrumConfig.seed``
    with its own offset, DISTINCT from the fingerprint / Markov / AR(1) jitter and the
    operating-point cube — so enabling a profile leaves the harmonic fingerprint and
    the raw parameter draws byte-identical.
    """

    # Devices to profile: None = every in-service Load and Generator in the grid.
    selector: Optional[Selector] = None

    # Overall daily-shape depth (per-class base depths live in ``profiles``).
    daily_amplitude: float = Field(default=1.0, ge=0.0)

    # Per-scenario, per-device idiosyncratic draws.
    level_min: float = Field(default=0.85, gt=0.0)
    level_max: float = Field(default=1.15, gt=0.0)
    amplitude_jitter_min: float = Field(default=0.8, ge=0.0)
    amplitude_jitter_max: float = Field(default=1.2, ge=0.0)
    phase_offset_hours: float = Field(default=1.0, ge=0.0)

    # Per-scenario SHARED latents (co-variation across devices).
    behavioral_coupling: float = Field(default=0.3, ge=0.0)  # non-pv daily amplitude
    cloud_coupling: float = Field(default=0.5, ge=0.0)  # pv output

    # Weekly weekday/weekend contrast (consumption; pv is unaffected).
    weekend_contrast: float = Field(default=0.15)

    # Seasonal modulation (annual sinusoid). ``*_peak_doy`` is a day-of-year phase.
    seasonal_amplitude: float = Field(default=0.15, ge=0.0)  # consumption swing
    seasonal_peak_doy: float = Field(default=15.0)  # consumption peaks in winter
    pv_seasonal_amplitude: float = Field(default=0.4, ge=0.0)  # pv output swing
    pv_seasonal_peak_doy: float = Field(default=172.0)  # summer solstice
    pv_daylight_hours: float = Field(default=12.0, gt=0.0)  # mean day length
    pv_daylight_swing: float = Field(default=4.0, ge=0.0)  # +/- seasonal day-length

    # Short-term stochastic term (AR(1), per device per step).
    short_rho: float = Field(default=0.9, ge=0.0, le=1.0)
    short_sigma: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "LoadProfileConfig":
        if self.level_max < self.level_min:
            raise ValueError("level_max must be >= level_min.")
        if self.amplitude_jitter_max < self.amplitude_jitter_min:
            raise ValueError("amplitude_jitter_max must be >= amplitude_jitter_min.")
        return self


# =============================================================================
# Statistical device-class composition of aggregated loads
# =============================================================================
#: Diurnal activity-rate presets a device class may follow. They REFERENCE the
#: class-aware daily shapes in :mod:`injection_hse.data.scenarios.profiles` (household evening,
#: office hours, EV evening, restaurant lunch/dinner, industrial plateau, the ``pv``
#: solar bell) plus a ``"flat"`` constant availability.
ActivityPreset = Literal[
    "household", "office", "ev", "restaurant", "industrial", "pv", "flat"
]


class ActivityCalendar(_Base):
    """Weekly and seasonal modulation of a device composition's activity rates.

    The composition's activity presets are DIURNAL shapes only: a member's availability
    depends on the hour of day (and, for ``pv``, on the season through the daylight
    window), never on the weekday or on the season's consumption level. Set on
    :class:`CompositionConfig` ``calendar``, this multiplies each member's availability rate
    by the same weekly and seasonal factors a :class:`LoadProfileConfig` applies to a
    profiled fundamental (the defaults here are that config's), so composed and profiled
    devices share one calendar:

    - every consumption preset by ``f_seasonal(t) = 1 + a * cos(2 pi (doy -
      seasonal_peak_doy) / 365.25)``, an annual sinusoid peaking in winter, with ``a`` =
      ``preset_seasonal_amplitude[preset]`` if listed, else ``seasonal_amplitude``;
    - every consumption preset except ``"flat"`` additionally by a smooth weekend dip
      ``f_weekly(t)`` of depth ``weekend_contrast`` (Saturday/Sunday); ``"flat"`` classes
      (heating, constant base load) do not follow the weekday;
    - ``pv`` by ``1 + pv_seasonal_amplitude * cos(2 pi (doy - pv_seasonal_peak_doy) /
      365.25)`` on top of its seasonally widening daylight bell (stronger summer
      irradiance).

    The modulated rate is clamped back to a probability. Days of week and year are read
    on the UTC epoch frame of the sequence's time axis.
    """

    weekend_contrast: float = Field(default=0.15)
    seasonal_amplitude: float = Field(default=0.15, ge=0.0)
    #: Per-preset seasonal amplitude overriding ``seasonal_amplitude`` (e.g. a larger
    #: winter swing of the ``"flat"`` heating classes).
    preset_seasonal_amplitude: dict[ActivityPreset, float] = Field(default_factory=dict)
    seasonal_peak_doy: float = Field(default=15.0)
    pv_seasonal_amplitude: float = Field(default=0.4, ge=0.0)
    pv_seasonal_peak_doy: float = Field(default=172.0)

    @model_validator(mode="after")
    def _check(self) -> "ActivityCalendar":
        if any(a < 0.0 for a in self.preset_seasonal_amplitude.values()):
            raise ValueError("ActivityCalendar seasonal amplitudes must be >= 0.")
        if "pv" in self.preset_seasonal_amplitude:
            raise ValueError(
                "ActivityCalendar: the pv season is pv_seasonal_amplitude, not a "
                "preset_seasonal_amplitude entry."
            )
        return self


class DeviceState(_Base):
    """One operating state of a MULTI-STATE device class.

    Captures the washing-machine / white-goods pattern where the SAME appliance draws
    very different current AND injects a very different spectrum depending on its cycle
    phase — e.g. a high-power, near-linear resistive HEATING state versus a low-power,
    harmonic-rich INVERTER-driven spinning state. The device jumps between its states on
    a Markov chain (dwell ``DeviceClassSpec.state_dwell``); the stationary probability of
    a state is proportional to its ``weight``.

    - ``power_fraction`` — the state's loading as a fraction of the device's rated power
      (drives BOTH the drawn power and, through the ``gamma`` law, the harmonic
      magnitude).
    - ``spectrum_scale`` — an ADDITIONAL multiplier on the device's per-order harmonic
      magnitudes in this state, capturing the qualitative spectral change that is NOT a
      function of power magnitude alone (heating ≈ clean, inverter ≈ rich).
    """

    name: str = "state"
    power_fraction: float = Field(gt=0.0, le=1.0)
    spectrum_scale: float = Field(default=1.0, ge=0.0)
    weight: float = Field(default=1.0, gt=0.0)


class DeviceClassSpec(_Base):
    """A STATISTICAL device class — a distribution over one appliance's behaviour.

    The goal is NOT an accurate appliance model but plausible statistical coverage: an
    aggregated load is composed of several members drawn from these classes (see
    :class:`ConsumerComposition`), and a state estimator learns to attribute an observed
    aggregate spectrum to a class mix. Every range below is drawn PER MEMBER (once, at
    roster build); the temporal draws (activity, loading, state) then evolve per step.

    The per-order harmonic magnitude ranges are FRACTIONS of the device's OWN fundamental
    current at rated load and are only LOOSELY bounded by the IEC 61000-3-2 emission
    shape (odd-dominated, decreasing with order) — they are a modelling convenience for
    generating diverse training data, NOT the standard's per-appliance limits.

    Load dependence (the measured reality that a device's harmonic signature depends on
    how hard it is driven):

    - magnitude ``mag_h(lam) = mag_h_rated * lam ** gamma_h`` with ``gamma_h`` drawn per
      member per order from ``gamma`` (``gamma = 0`` → constant ratio; ``gamma < 0`` →
      the THD FRACTION falls as load rises while the absolute harmonic current still
      grows — the measured EV-charger / PV-inverter behaviour);
    - phase ``ang_h(lam) = ang_h0 + s_h * (lam - 1)`` with ``s_h`` [deg] drawn per member
      per order from ``phase_slope_deg``.

    Activity: ``activity_preset`` is a diurnal availability shape (see
    :data:`ActivityPreset`). ``discrete_activity`` selects how it is realised — a
    switching appliance (``True``, the default) follows a two-state on/off Markov chain
    whose target occupancy tracks the diurnal rate (stickiness drawn from
    ``on_off_dwell``); a continuously-modulated device (``False``, e.g. a PV inverter
    tracking irradiance, a background base load tracking occupancy) uses the diurnal rate
    itself as a fractional availability in ``[0, 1]``.
    """

    name: str
    #: ``+1`` consuming (Load-like), ``-1`` injecting (PV-like).
    sign: Literal[1, -1] = 1
    #: Rated active power per instance ``[low, high]`` in watts (magnitude).
    rated_power_w: tuple[float, float]
    #: Fundamental displacement power factor ``lambda`` (``Q = P * tan(acos(pf))``).
    power_factor: float = Field(default=1.0, gt=0.0, le=1.0)

    #: Per-order harmonic magnitude ranges at RATED load, as a FRACTION of the device's
    #: own fundamental current: ``{order: [low, high]}`` (orders >= 2; absent = no
    #: emission at that order). Every drawn member ratio is additionally CAPPED at the
    #: member's IEC 61000-3-2 emission fraction (evaluated at the member's effective
    #: rated power after ``scale_to_nominal``), so an aggregated roster can never emit
    #: beyond what its individual appliances are permitted to inject — the same physical
    #: envelope the randomized ``h_mag`` sampling references.
    harmonic_magnitude: dict[int, tuple[float, float]] = Field(default_factory=dict)
    #: IEC 61000-3-2 equipment class for the member emission cap: ``"A"``/``"B"``/
    #: ``"C"``/``"D"``, or ``None`` (default) to resolve automatically from the member's
    #: effective per-phase power (Class D inside its 75-600 W window, else Class A).
    emission_class: Optional[Literal["A", "B", "C", "D"]] = None
    #: Per-order harmonic phase ranges [deg] at rated load: ``{order: [low, high]}``.
    harmonic_phase_deg: dict[int, tuple[float, float]] = Field(default_factory=dict)
    #: Range for the per-order magnitude load-dependence exponent ``gamma_h``.
    gamma: tuple[float, float] = (0.0, 0.0)
    #: Range for the per-order phase load-dependence slope ``s_h`` [deg per unit load].
    phase_slope_deg: tuple[float, float] = (0.0, 0.0)
    #: Range for the per-order LOAD-INDEPENDENT share of the rated harmonic phasor.
    #:
    #: Measured devices do not emit a harmonic current proportional to their fundamental.
    #: A part of the emission is present whenever the device is on and does not scale with
    #: loading, so the emission is affine, ``I_h(lam) = A_h + B_h * lam``, and the RATIO to
    #: the fundamental therefore grows as the device unloads. ``emission_floor`` is
    #: ``|A_h|`` as a share of ``|A_h| + |B_h|``: ``0.0`` (default) restores the purely
    #: proportional law ``I_h ~ lam`` exactly, ``0.61`` reproduces the ratio inflation
    #: measured across certified inverters (``ratio(lam) = ratio_rated * (0.39 + 0.61/lam)``).
    #: The rated (``lam = 1``) emission is unchanged at any value, so this re-shapes the
    #: load dependence without moving the emission envelope.
    emission_floor: tuple[float, float] = (0.0, 0.0)
    #: Range for ``arg(A_h) - arg(B_h)`` [deg], the angle between the load-independent and
    #: the load-proportional part. Non-zero makes the emission ANGLE move with loading —
    #: measured devices rotate their harmonic phase as they unload, which decides whether
    #: two devices at different operating points add or cancel. Ignored when
    #: ``emission_floor`` is zero.
    emission_floor_phase_deg: tuple[float, float] = (0.0, 0.0)

    #: Diurnal availability shape.
    activity_preset: ActivityPreset = "flat"
    #: ``True`` → a switching appliance (on/off Markov); ``False`` → continuous
    #: availability equal to the diurnal rate (PV irradiance, background base load).
    discrete_activity: bool = True
    #: Range for the on/off Markov stickiness (per-step probability of persisting).
    on_off_dwell: tuple[float, float] = (0.85, 0.98)

    #: Loading floor ``lam_min`` (the per-step loading is clamped to ``[lam_min, 1]``).
    loading_min: float = Field(default=0.2, gt=0.0, le=1.0)
    #: Range for the per-member mean loading (single-state classes).
    loading_mean: tuple[float, float] = (0.6, 1.0)
    #: Fractional std of the AR(1) per-step loading jitter.
    loading_jitter: float = Field(default=0.05, ge=0.0)
    #: AR(1) stickiness of the per-step loading jitter.
    loading_rho: float = Field(default=0.85, ge=0.0, le=1.0)

    #: Optional multi-state operation (empty = single-state, continuous loading).
    states: list[DeviceState] = Field(default_factory=list)
    #: Range for the multi-state Markov dwell (per-step probability of staying in state).
    state_dwell: tuple[float, float] = (0.85, 0.97)

    @model_validator(mode="after")
    def _check(self) -> "DeviceClassSpec":
        lo, hi = self.rated_power_w
        if lo <= 0.0 or hi < lo:
            raise ValueError(
                f"class {self.name!r} rated_power_w must be 0 < low <= high."
            )
        for field_name in ("harmonic_magnitude", "harmonic_phase_deg"):
            for order in getattr(self, field_name):
                if order < 2:
                    raise ValueError(
                        f"class {self.name!r} {field_name} orders must be >= 2."
                    )
        if not (
            self.loading_min <= self.loading_mean[0] <= self.loading_mean[1] <= 1.0
        ):
            raise ValueError(
                f"class {self.name!r} requires loading_min <= loading_mean[0] "
                "<= loading_mean[1] <= 1."
            )
        for lo_, hi_ in (
            self.gamma,
            self.phase_slope_deg,
            self.on_off_dwell,
            self.state_dwell,
        ):
            if hi_ < lo_:
                raise ValueError(f"class {self.name!r} has a range with high < low.")
        return self


class ClassCount(_Base):
    """How many instances of a device class an aggregated load contains.

    ``count`` is an inclusive ``[min, max]`` integer range; ``power_share`` weights this
    class's instances when :class:`CompositionConfig` scales the roster to the load's
    nominal power (a larger share claims a larger slice of the nameplate).
    """

    class_name: str
    count: tuple[int, int] = (1, 1)
    power_share: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "ClassCount":
        lo, hi = self.count
        if lo < 0 or hi < lo:
            raise ValueError(
                f"ClassCount {self.class_name!r} count must be 0 <= min <= max."
            )
        return self


class ConsumerComposition(_Base):
    """A composition rule: the device-class roster of one consumer character.

    A rule matches an aggregated load by explicit ``load_ids`` (a per-appliance override
    — e.g. a public charging station is just an EV charger) if given, else by
    ``consumer_type`` (matched against the load's own ``consumer_type``); a rule with
    both ``load_ids`` and ``consumer_type`` unset is the fallback for any load no other
    rule claims. ``classes`` lists the device classes and their instance counts.
    """

    consumer_type: Optional[str] = None
    load_ids: Optional[list[int]] = None
    classes: list[ClassCount] = Field(min_length=1)


#: Version of the built-in device-class library and its composition rules. Bumped
#: whenever a range changes the drawn population, and recorded in the metadata of every
#: dataset generated from it — two datasets written from identical configs and seeds can
#: otherwise differ numerically with nothing to show why. Revision 4 continued the bands
#: from h19 to every order up to h50 (:func:`continue_emission_bands`); a run at the odd
#: orders up to h19 draws what revision 3 drew.
DEVICE_LIBRARY_VERSION = "4"

#: Highest harmonic order the built-in library emits at: the range a power-quality
#: analyser resolves (IEC 61000-4-7 groups every integer order up to h50).
EMISSION_MAX_ORDER = 50
#: Even-order emission of a half-wave-SYMMETRIC converter (bridge rectifier, SMPS, active
#: front end, PV / EV / heat-pump inverter) relative to the mean of its two neighbouring
#: odd-order bands. A symmetric waveform has no even harmonics; what remains comes from
#: component and supply asymmetry. No measured even-order device spectra back a value;
#: it is bounded by the even-to-neighbouring-odd ratio of the IEC 61000-3-2 Class A
#: current limits (0.38-0.58 at h2-h10, what the standard permits) and of the public-LV
#: voltage compatibility levels (IEC 61000-2-2 / EN 50160: 0.10-0.17 at h6-h14), and set
#: below both so that the even-order voltage the library produces on the CIGRE LV
#: benchmark stays near the EN 50160 even level of 0.5 % at its 95th percentile.
EVEN_ORDER_RATIO = 0.05
#: Even-order ratio of a half-wave-ASYMMETRIC load (phase-angle or half-wave controlled
#: motors and heaters of white goods), four times the symmetric one: the asymmetry the
#: standard's even-order current limits are written for.
ASYMMETRIC_EVEN_ORDER_RATIO = 0.2


def continue_emission_bands(
    cls: DeviceClassSpec,
    *,
    max_order: int = EMISSION_MAX_ORDER,
    even_ratio: float = EVEN_ORDER_RATIO,
) -> DeviceClassSpec:
    """A device class whose emission bands cover every order up to ``max_order``.

    Tabulated orders are kept as they are. An untabulated ODD order ``h`` takes the band
    of the nearest tabulated odd order ``h0`` below it times ``h0 / h``: the ``1/h``
    decline of the IEC 61000-3-2 odd-order limits (Class A ``0.15 * 15 / h`` A for
    15 <= h <= 39, Class D ``3.85 / h`` mA/W for 13 <= h <= 39), extrapolated above h39,
    where the standard sets no limit. An untabulated EVEN order takes ``even_ratio`` times
    the mean of its neighbouring odd bands (``h - 1`` and ``h + 1``; h2 from h3 alone),
    low and high ends separately. The phase range of an order that gains a band is that of
    the nearest tabulated order (the lower one on a tie). A class without any harmonic
    band is returned unchanged.

    Parameters
    ----------
    cls:
        The class to continue.
    max_order:
        Highest order to fill in.
    even_ratio:
        Even-to-neighbouring-odd band ratio (:data:`EVEN_ORDER_RATIO` for a symmetric
        converter, :data:`ASYMMETRIC_EVEN_ORDER_RATIO` for a half-wave-asymmetric load).

    Returns
    -------
    DeviceClassSpec
        A copy with the continued ``harmonic_magnitude`` / ``harmonic_phase_deg``.
    """
    mags = {
        int(o): (float(lo), float(hi)) for o, (lo, hi) in cls.harmonic_magnitude.items()
    }
    if not mags:
        return cls
    odd_tab = sorted(o for o in mags if o % 2)
    odd: dict[int, tuple[float, float]] = {}
    for h in range(3, int(max_order) + 2, 2):  # one past max_order for the last even
        if h in mags:
            odd[h] = mags[h]
            continue
        below = [o for o in odd_tab if o < h]
        if below:
            h0 = below[-1]
            odd[h] = (mags[h0][0] * h0 / h, mags[h0][1] * h0 / h)
    out = dict(mags)
    for h in range(2, int(max_order) + 1):
        if h in out:
            continue
        if h % 2:
            if h in odd:
                out[h] = odd[h]
            continue
        sides = [odd[o] for o in (h - 1, h + 1) if o >= 3 and o in odd]
        if sides:
            out[h] = (
                even_ratio * sum(lo for lo, _ in sides) / len(sides),
                even_ratio * sum(hi for _, hi in sides) / len(sides),
            )
    phases = {int(o): tuple(r) for o, r in cls.harmonic_phase_deg.items()}
    if phases:
        tabulated = sorted(phases)
        for h in out:
            if h not in phases:
                nearest = min(tabulated, key=lambda o: (abs(o - h), o))
                phases[h] = phases[nearest]
    return cls.model_copy(
        update={
            "harmonic_magnitude": dict(sorted(out.items())),
            "harmonic_phase_deg": dict(sorted(phases.items())),
        }
    )


def default_device_classes() -> list[DeviceClassSpec]:
    """The built-in statistical device-class library (every range user-overridable).

    Seven plausible LV device characters: a harmonic-free linear base load, an
    SMPS-electronics class (3rd/5th-dominated), an EV charger and a PV inverter (both
    with the measured falling-THD-fraction-with-load behaviour through
    ``emission_floor``; PV injecting), a multi-state inverter drive (white goods: a near-linear heating state vs
    a harmonic-rich inverter state), a kW-scale inverter heat pump, and a thermostatic
    resistive heater. Magnitudes are loosely IEC 61000-3-2-shaped plausibility, NOT
    appliance models.

    The roster is balanced so that an observed aggregate spectrum is INFORMATIVE about
    the drawn power: the power share sits on the kW-scale nonlinear classes (EV charging,
    inverter heat pumps, drives) rather than on spectrally silent linear ones, the EV
    load-dependence exponent is flat enough that its ABSOLUTE emission tracks its
    loading, and the class power factors are spread so the composed fundamental-current
    angles scatter like a real device population.

    Per-order magnitudes up to h11 follow the measured band of that class; h13 and above
    continue with the declining envelope of the certification-workbook populations, and
    the emission phase saturates at the full circle from h13, where measured populations
    show no preferred angle. Up to h19 the bands are tabulated; every further order up to
    :data:`EMISSION_MAX_ORDER`, odd and even, follows :func:`continue_emission_bands`
    (the inverter drive, a half-wave-asymmetric white-goods load, at
    :data:`ASYMMETRIC_EVEN_ORDER_RATIO`). A member draws only the requested orders, so
    the draws at h2-h19 odd are those of the library before the continuation.

    The five nonlinear classes carry the load dependence through ``emission_floor``
    (0.43-0.73, the measured load-independent share of the rated emission) rather than
    through a ``gamma`` exponent. Certification sweeps and lab racks show an emission
    FLOOR — the absolute harmonic current grows only 1.2-1.9x while the fundamental grows
    10x — which makes the ratio to the fundamental inflate 6-10x between rated and 10 %
    loading and puts a cancellation null inside the operating range. A power law cannot
    reproduce that: its elasticity ``d ln|I_h| / d ln lam`` is the constant ``1 + gamma``,
    where the measured elasticity is 0.04-0.08 at 10 % loading and 0.54-0.82 at rated.
    The mean loadings reach further down than the emission-envelope draw alone would
    need, because the whole effect lives below ``lam ~ 0.3``.
    """
    classes = [
        DeviceClassSpec(
            name="base_linear",
            sign=1,
            rated_power_w=(150.0, 1500.0),
            power_factor=0.97,
            activity_preset="household",
            discrete_activity=False,
            loading_min=0.3,
            loading_mean=(0.6, 1.0),
        ),
        DeviceClassSpec(
            name="electronics_smps",
            sign=1,
            rated_power_w=(20.0, 400.0),
            power_factor=0.97,
            emission_class="D",
            harmonic_magnitude={
                3: (0.5, 0.85),
                5: (0.25, 0.6),
                7: (0.1, 0.4),
                9: (0.05, 0.25),
                11: (0.03, 0.15),
                13: (0.02, 0.1),
                15: (0.015, 0.08),
                17: (0.01, 0.06),
                19: (0.008, 0.05),
            },
            harmonic_phase_deg={
                3: (-30.0, 30.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
                13: (-180.0, 180.0),
                15: (-180.0, 180.0),
                17: (-180.0, 180.0),
                19: (-180.0, 180.0),
            },
            gamma=(0.0, 0.0),
            phase_slope_deg=(-20.0, 20.0),
            emission_floor=(0.43, 0.73),
            emission_floor_phase_deg=(100.0, 150.0),
            activity_preset="household",
            on_off_dwell=(0.5, 0.8),
            loading_min=0.1,
            loading_mean=(0.15, 0.9),
        ),
        DeviceClassSpec(
            name="ev_charger",
            sign=1,
            rated_power_w=(3700.0, 11000.0),
            power_factor=0.98,
            harmonic_magnitude={
                3: (0.02, 0.08),
                5: (0.03, 0.1),
                7: (0.02, 0.06),
                9: (0.01, 0.04),
                11: (0.005, 0.025),
                13: (0.005, 0.025),
                15: (0.003, 0.016),
                17: (0.003, 0.015),
                19: (0.003, 0.014),
            },
            harmonic_phase_deg={
                3: (-40.0, 40.0),
                5: (-40.0, 40.0),
                7: (-60.0, 60.0),
                9: (-90.0, 90.0),
                11: (-120.0, 120.0),
                13: (-150.0, 150.0),
                15: (-180.0, 180.0),
                17: (-180.0, 180.0),
                19: (-180.0, 180.0),
            },
            # Near-flat load dependence: the ABSOLUTE harmonic current of a charging
            # session tracks the drawn power, so the emission carries fundamental
            # information. A steeply negative exponent makes the harmonic current almost
            # loading-independent and the spectrum uninformative about P.
            gamma=(0.0, 0.0),
            phase_slope_deg=(-15.0, 15.0),
            emission_floor=(0.43, 0.73),
            emission_floor_phase_deg=(100.0, 150.0),
            activity_preset="ev",
            on_off_dwell=(0.6, 0.82),
            loading_min=0.2,
            loading_mean=(0.25, 1.0),
        ),
        DeviceClassSpec(
            name="pv_inverter",
            sign=-1,
            rated_power_w=(1000.0, 8000.0),
            harmonic_magnitude={
                3: (0.01, 0.04),
                5: (0.02, 0.07),
                7: (0.01, 0.05),
                9: (0.005, 0.03),
                11: (0.002, 0.045),
                13: (0.002, 0.035),
                15: (0.001, 0.03),
                17: (0.001, 0.06),
                19: (0.001, 0.025),
            },
            harmonic_phase_deg={
                3: (-60.0, 60.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
                13: (-150.0, 150.0),
                15: (-180.0, 180.0),
                17: (-180.0, 180.0),
                19: (-180.0, 180.0),
            },
            gamma=(0.0, 0.0),
            phase_slope_deg=(-25.0, 25.0),
            emission_floor=(0.43, 0.73),
            emission_floor_phase_deg=(100.0, 150.0),
            activity_preset="pv",
            discrete_activity=False,
            loading_min=0.05,
            loading_mean=(0.2, 1.0),
        ),
        DeviceClassSpec(
            name="inverter_drive",
            sign=1,
            rated_power_w=(500.0, 3000.0),
            power_factor=0.92,
            harmonic_magnitude={
                3: (0.08, 0.3),
                5: (0.15, 0.5),
                7: (0.08, 0.35),
                9: (0.03, 0.18),
                11: (0.02, 0.12),
                13: (0.015, 0.08),
                15: (0.01, 0.05),
                17: (0.008, 0.04),
                19: (0.005, 0.03),
            },
            harmonic_phase_deg={
                3: (-45.0, 45.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
                13: (-180.0, 180.0),
                15: (-180.0, 180.0),
                17: (-180.0, 180.0),
                19: (-180.0, 180.0),
            },
            gamma=(0.0, 0.0),
            phase_slope_deg=(-30.0, 30.0),
            emission_floor=(0.43, 0.73),
            emission_floor_phase_deg=(100.0, 150.0),
            activity_preset="household",
            on_off_dwell=(0.6, 0.85),
            loading_min=0.15,
            loading_mean=(0.2, 1.0),
            states=[
                DeviceState(
                    name="heating", power_fraction=0.95, spectrum_scale=0.2, weight=0.5
                ),
                DeviceState(
                    name="inverter", power_fraction=0.35, spectrum_scale=1.6, weight=0.5
                ),
            ],
            state_dwell=(0.85, 0.97),
        ),
        DeviceClassSpec(
            # kW-scale inverter-driven compressor: a LARGE consumer whose spectrum is
            # both distinctive (h5/h7-dominant) and power-tracking — the class that lets
            # a model infer significant fundamental power from an observed pattern.
            name="heat_pump_inverter",
            sign=1,
            rated_power_w=(2000.0, 12000.0),
            power_factor=0.95,
            harmonic_magnitude={
                3: (0.05, 0.15),
                5: (0.1, 0.3),
                7: (0.05, 0.2),
                9: (0.02, 0.08),
                11: (0.01, 0.05),
                13: (0.008, 0.035),
                15: (0.005, 0.025),
                17: (0.004, 0.02),
                19: (0.003, 0.015),
            },
            harmonic_phase_deg={
                3: (-45.0, 45.0),
                5: (-60.0, 60.0),
                7: (-90.0, 90.0),
                9: (-120.0, 120.0),
                11: (-150.0, 150.0),
                13: (-180.0, 180.0),
                15: (-180.0, 180.0),
                17: (-180.0, 180.0),
                19: (-180.0, 180.0),
            },
            gamma=(0.0, 0.0),
            phase_slope_deg=(-20.0, 20.0),
            emission_floor=(0.43, 0.73),
            emission_floor_phase_deg=(100.0, 150.0),
            activity_preset="flat",
            on_off_dwell=(0.9, 0.98),
            loading_min=0.15,
            loading_mean=(0.2, 0.95),
            loading_jitter=0.08,
        ),
        DeviceClassSpec(
            name="resistive_heating",
            sign=1,
            rated_power_w=(500.0, 3000.0),
            activity_preset="flat",
            on_off_dwell=(0.5, 0.75),
            loading_min=0.9,
            loading_mean=(0.95, 1.0),
            loading_jitter=0.02,
        ),
    ]
    return [
        continue_emission_bands(
            c,
            even_ratio=ASYMMETRIC_EVEN_ORDER_RATIO
            if c.name == "inverter_drive"
            else EVEN_ORDER_RATIO,
        )
        for c in classes
    ]


def default_compositions() -> list[ConsumerComposition]:
    """The built-in per-``consumer_type`` composition rules (over the default library).

    The power shares put the nameplate on the classes that EMIT — a roster whose power
    sits on the spectrally silent linear classes produces an aggregate spectrum that
    reveals device activity but almost nothing about the drawn power, which is the
    quantity a state estimator has to recover.

    Every count / share is user-overridable. The rule with no ``consumer_type`` /
    ``load_ids`` is the fallback for any unmatched load.
    """

    def cc(name: str, lo: int, hi: int, share: float = 1.0) -> ClassCount:
        return ClassCount(class_name=name, count=(lo, hi), power_share=share)

    return [
        ConsumerComposition(
            consumer_type="household",
            classes=[
                cc("base_linear", 1, 1, 1.5),
                cc("electronics_smps", 1, 3, 1.0),
                cc("resistive_heating", 0, 1, 1.0),
                cc("ev_charger", 0, 1, 2.0),
                cc("heat_pump_inverter", 0, 1, 2.0),
                cc("pv_inverter", 0, 1, 1.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="office",
            classes=[
                cc("base_linear", 1, 1, 2.0),
                cc("electronics_smps", 3, 10, 2.0),
                cc("inverter_drive", 1, 3, 2.5),
                cc("heat_pump_inverter", 0, 2, 1.5),
            ],
        ),
        ConsumerComposition(
            consumer_type="restaurant",
            classes=[
                cc("base_linear", 1, 1, 2.0),
                cc("inverter_drive", 1, 3, 2.5),
                cc("electronics_smps", 1, 4, 1.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="heat_pump",
            classes=[
                cc("base_linear", 1, 1, 1.0),
                cc("heat_pump_inverter", 1, 2, 3.0),
            ],
        ),
        ConsumerComposition(
            consumer_type="workshop",
            classes=[
                cc("base_linear", 1, 1, 1.5),
                cc("inverter_drive", 2, 4, 3.0),
                cc("electronics_smps", 1, 4, 1.0),
                cc("resistive_heating", 0, 1, 0.5),
            ],
        ),
        ConsumerComposition(
            consumer_type="ev_charging",
            classes=[cc("base_linear", 0, 1, 0.5), cc("ev_charger", 1, 2, 1.0)],
        ),
        ConsumerComposition(
            consumer_type="pv",
            classes=[cc("pv_inverter", 1, 1, 1.0)],
        ),
        ConsumerComposition(
            classes=[
                cc("base_linear", 1, 1, 2.0),
                cc("electronics_smps", 1, 2, 1.0),
                cc("inverter_drive", 0, 1, 1.0),
            ],
        ),
    ]


def composition_silent_orders(
    classes: list[DeviceClassSpec], orders: list[int]
) -> list[int]:
    """Requested orders NO class in a device roster emits at (sorted, fundamental excluded).

    A roster that stops short of the solved harmonic range makes those orders silent:
    the true harmonic voltage there is zero, relative metrics turn into NaN and the
    corresponding feature width is dead. Checking the roster against the requested orders
    is the one-line guard that catches it before hours of generation.
    """
    emitting = {
        int(order)
        for cls in classes
        for order, (_low, high) in cls.harmonic_magnitude.items()
        if float(high) > 0.0
    }
    return [o for o in sorted({int(o) for o in orders}) if o > 1 and o not in emitting]


# =============================================================================
# Composing an aggregated load out of its member device classes
# =============================================================================
class CompositionConfig(_Base):
    """Statistical device-class composition of aggregated loads.

    Set on :class:`CoherentSpectrumConfig`. When present it SUPERSEDES the mode-bank
    fingerprint machinery for the loads it covers: each such load becomes a sum of
    statistical member devices (drawn from ``classes`` per the matching
    :class:`ConsumerComposition` in ``compositions``), whose per-step activity drives
    BOTH the drawn fundamental power AND the injected harmonic spectrum — so the dataset
    carries a consistent load-to-spectrum mapping a model can learn from (and, in the
    best case, use to attribute an observed spectrum to a device mix / an error source).

    A load covered by the composition draws its fundamental P/Q and its harmonic
    injection ENTIRELY from the composition; any ``parameters`` / ``profile`` targeting
    the same load is superseded. Loads NOT covered (no matching rule, or outside
    ``selector``) keep the fingerprint / profile behaviour.

    Roster: per covered load a device roster is drawn once (seeded, persisted). With
    ``scale_to_nominal`` the members' rated powers are rescaled so their share-weighted
    installed capacity equals the load's ``p_nom_w`` (keeping the nameplate meaningful
    across grids); otherwise the class rated ranges are absolute. Cross-device
    correlation reuses the profile latents: one ``behavioral`` latent scales every
    consumption activity together, one ``cloud`` latent scales every PV together.

    Guard: near a net-zero aggregate fundamental (e.g. PV cancelling load) the RELATIVE
    harmonic magnitude explodes (a physically real residual-THD effect); it is capped at
    ``max_injection_pu`` and the binding is recorded in the samples.
    """

    #: Loads to compose. ``None`` = every in-service Load. A load covered here but
    #: without a matching rule in ``compositions`` falls through to the fingerprint.
    selector: Optional[Selector] = None
    #: The device-class library.
    classes: list[DeviceClassSpec] = Field(default_factory=default_device_classes)
    #: The per-consumer-type composition rules.
    compositions: list[ConsumerComposition] = Field(
        default_factory=default_compositions
    )
    #: Rescale each roster's share-weighted installed capacity to the load's ``p_nom_w``.
    scale_to_nominal: bool = True
    #: Cap on the aggregate harmonic magnitude [pu of the aggregate fundamental current].
    max_injection_pu: float = Field(default=3.0, gt=0.0)
    #: Cross-device correlation strengths (shared per-scenario latents).
    behavioral_coupling: float = Field(default=0.3, ge=0.0)
    cloud_coupling: float = Field(default=0.5, ge=0.0)
    #: Multiplier on every member's diurnal availability RATE (clamped back to a
    #: probability). The class presets are per-device duty cycles, so a composed aggregate
    #: sits at a small fraction of installed capacity — a fair average hour, but one that
    #: never reaches the loaded states where the voltage profile actually moves and an
    #: estimator has something to recover. Raise it to place the population in a loaded
    #: band; the diurnal shape, the rosters and every other draw are unchanged.
    activity_scale: float = Field(default=1.0, gt=0.0)
    #: Optional distinct seed for the roster + temporal draws (a held-out composition
    #: bank). ``None`` derives both streams from ``CoherentSpectrumConfig.seed``.
    roster_seed: Optional[int] = None
    #: Weekly and seasonal modulation of the activity rates (:class:`ActivityCalendar`).
    #: ``None`` keeps the activity purely diurnal.
    calendar: Optional[ActivityCalendar] = None

    _OMIT_WHEN_NONE: ClassVar[tuple[str, ...]] = ("calendar",)

    def class_names(self) -> list[str]:
        """Ordered device-class names (the ``n_class`` axis of the recorded samples)."""
        return [c.name for c in self.classes]

    @model_validator(mode="after")
    def _check(self) -> "CompositionConfig":
        names = [c.name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("CompositionConfig class names must be unique.")
        known = set(names)
        for rule in self.compositions:
            for cc in rule.classes:
                if cc.class_name not in known:
                    raise ValueError(
                        f"ConsumerComposition references unknown class "
                        f"{cc.class_name!r} (not in classes)."
                    )
        return self


# =============================================================================
# Per-sequence start times
# =============================================================================
def _epoch_s(stamp: str) -> float:
    """UTC epoch seconds of an ISO 8601 timestamp (a naive one is read as UTC)."""
    dt = datetime.fromisoformat(stamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class StartTimeRange(_Base):
    """Distribution of the per-sequence start time of a coherent batch.

    Set on :class:`CoherentSpectrumConfig` ``start_time_range`` (instead of the single
    ``start_time``), each of the ``n_scenarios`` sequences starts at its own time drawn
    from this range, so a dataset covers many times of day, weekdays and seasons instead
    of one window. The candidates are the instants ``start + k * resolution_s`` before
    ``end`` whose hour of day lies in ``hours`` and whose weekday is in ``weekdays``
    (both on the UTC frame of the time axis; a naive timestamp is read as UTC).

    ``sampling="uniform"`` draws every start independently and uniformly over the
    candidates; ``"stratified"`` cuts the ordered candidates into ``n_scenarios`` equal
    strata and draws one start uniformly inside each (an even coverage of the range at any
    batch size), then shuffles them over the sequences. Both draw on their own seeded
    stream derived from ``CoherentSpectrumConfig.seed``, so the start times leave every
    other draw of the batch unchanged.
    """

    #: First admissible start (ISO 8601, inclusive).
    start: str
    #: End of the range (ISO 8601, exclusive).
    end: str
    sampling: Literal["uniform", "stratified"] = "uniform"
    #: Admissible start hours ``[low, high)`` of the day; ``low > high`` wraps midnight
    #: (``(22, 6)`` = night starts). ``None`` admits every hour.
    hours: Optional[tuple[float, float]] = None
    #: Admissible start weekdays (Monday 0 .. Sunday 6). ``None`` admits every day.
    weekdays: Optional[tuple[int, ...]] = None
    #: Spacing of the candidate starts [s]. ``None`` uses the sequence's ``step_size_s``,
    #: so every start lies on the step grid of the range start.
    resolution_s: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "StartTimeRange":
        for label, stamp in (("start", self.start), ("end", self.end)):
            try:
                datetime.fromisoformat(stamp)
            except ValueError as exc:
                raise ValueError(
                    f"StartTimeRange.{label} {stamp!r} is not a valid ISO 8601 timestamp."
                ) from exc
        if _epoch_s(self.end) <= _epoch_s(self.start):
            raise ValueError("StartTimeRange.end must lie after start.")
        if self.hours is not None and not all(0.0 <= h <= 24.0 for h in self.hours):
            raise ValueError("StartTimeRange.hours must lie in [0, 24].")
        if self.weekdays is not None and (
            not self.weekdays or not all(0 <= d <= 6 for d in self.weekdays)
        ):
            raise ValueError(
                "StartTimeRange.weekdays must be a non-empty set of 0 (Monday) .. 6."
            )
        return self


# =============================================================================
# Node-coherent harmonic "fingerprint" sampling (temporal sequences)
# =============================================================================
class CoherentSpectrumConfig(_Base):
    """Node-coherent harmonic sampling: a stable per-device fingerprint over a sequence.

    Each matched device draws a small set of base spectra (``n_modes`` "modes" — e.g.
    appliance operating states like a washing machine heating vs spinning), drawn once
    (or per scenario). Over ``n_steps`` consecutive steps it STICKS to a mode (Markov
    dwell ``dwell``) and WANDERS around it (AR(1) jitter with stickiness ``ar1_rho``),
    clamped to the per-order emission reference (``harmonic_reference``). This yields a
    ``[B, T]`` batch of harmonic injections in which each node keeps a recognisable
    signature that varies realistically — so a state estimator can attribute the pattern
    to the node.

    The result voltages are ``[B, T, H, N]`` (B = ``n_scenarios`` sequences, T = steps);
    per-step timestamps are recorded as ``samples["time_s"]``.

    ``harmonic_reference`` selects the per-order magnitude reference (and the upper
    clamp): the default ``"iec61000-3-2"`` is the IEC 61000-3-2 appliance harmonic
    CURRENT-emission standard — the physically correct per-device reference for a device
    current fingerprint, keyed by ``emission_class`` (``"auto"`` resolves per device from
    its ``consumer_type`` and nominal power). ``"en50160"`` instead SHAPES the fingerprint
    by the DIN EN 50160 supply-VOLTAGE compatibility levels (kept for
    background-distortion-shaped experiments and backward compatibility; it is NOT an
    appliance emission model). ``None`` treats magnitudes as absolute pu (clamped to 1.0).

    Held-out test set recipe: reproducibility hangs on both ``seed`` (the temporal
    Markov + AR(1) stream) and ``mode_bank_seed`` (the per-device fingerprint bank). For
    an unseen-FINGERPRINT test set, give a DISTINCT ``mode_bank_seed`` (a different
    device signature bank while every other setting is shared); combine it with a
    distinct ``seed`` for an entirely independent temporal realization too. Leaving
    ``mode_bank_seed=None`` draws the bank from the ``seed`` stream (the default,
    byte-identical to a config with no ``mode_bank_seed`` set).

    ``parameters`` / ``factors`` add a FUNDAMENTAL operating-point variation on top of the
    harmonic fingerprint: the same :class:`ParameterSpec` / :class:`LatentFactor` machinery
    as :class:`ScenarioConfig`, but drawn ONCE PER SCENARIO (shape ``[B]``, held constant
    across the ``T`` steps and broadcast in the solve). So a coherent sequence may vary the
    load level, PV output, and the slack voltage (``field="u_ref"``) per scenario while the
    per-step spectrum keeps its device fingerprint. Harmonic ``ParameterSpec`` fields
    (``h_mag`` / ``h_phase``) are REJECTED here — the fingerprint machinery owns the
    harmonics; ``parameters`` shapes only the fundamental. The draws are sampled on a unit
    cube seeded from a stream DISTINCT from the fingerprint RNG, so the realized
    ``harmonic_injection`` is byte-identical with and without ``parameters``. When
    ``parameters`` is empty (the default) the fundamental P/Q stays nominal and the source
    at ``u_ref_v`` — the original fingerprint-only behavior.

    ``profile`` (a :class:`LoadProfileConfig`) makes the fundamental P/Q TIME-VARYING
    across the ``T`` steps instead of constant: each device's per-scenario base P/Q (from
    ``parameters`` if set, else nominal) is multiplied by a multi-scale synthetic
    profile factor (seasonal / weekly / daily / short-term, class-aware by
    ``consumer_type``). The operating point then carries the step axis (``[B, T]``),
    aligned with the ``[B, T]`` harmonic injection, so the solve yields ``[B, T, H, N]``
    with a moving fundamental. ``start_time`` (ISO 8601) is REQUIRED when ``profile`` is
    set — the daily / weekly / seasonal phases need an absolute anchor. Because the
    harmonic injection magnitude is RELATIVE to the device's fundamental current, a
    profile-scaled fundamental already scales the absolute harmonic current; no extra
    coupling is applied. ``profile=None`` (the default) leaves the fundamental constant
    over the sequence — byte-identical to the fingerprint-only behavior. The profile is
    drawn on an RNG stream distinct from the fingerprint, Markov path, AR(1) jitter, and
    the operating-point cube, so enabling it leaves the harmonic fingerprint unchanged.

    ``start_time`` places every sequence of the batch in the SAME absolute window.
    ``start_time_range`` (a :class:`StartTimeRange`, exclusive with ``start_time``) instead
    draws one start per sequence, so the batch spans times of day, weekdays and seasons:
    the profile, the composition's activity rates and the recorded ``time_unix_s`` then
    follow a ``[B, T]`` time axis. The start draw has its own derived stream, so the
    fingerprint, the operating-point cube and every other draw are unchanged by it.
    """

    name: str = "harmonics"
    selector: Selector
    orders: list[int] = Field(min_length=1)
    n_steps: int = Field(gt=0)  # T
    n_scenarios: int = Field(default=1, gt=0)  # B
    n_modes: int = Field(default=2, ge=1)
    seed: int = 0
    mag_distribution: Distribution = Field(
        default_factory=lambda: Uniform(low=0.0, high=1.0)
    )
    harmonic_reference: Optional[Literal["en50160", "iec61000-3-2"]] = "iec61000-3-2"
    emission_class: Literal["A", "B", "C", "D", "auto"] = "auto"
    phase_distribution: Distribution = Field(
        default_factory=lambda: Uniform(low=-180.0, high=180.0)
    )
    jitter_mag: float = Field(default=0.05, ge=0.0)  # AR(1) fractional std on magnitude
    jitter_phase_deg: float = Field(default=5.0, ge=0.0)  # AR(1) std on phase (deg)
    ar1_rho: float = Field(default=0.8, ge=0.0, le=1.0)  # temporal stickiness of jitter
    dwell: float = Field(default=0.9, ge=0.0, le=1.0)  # P(stay in mode) per step
    step_size_s: float = Field(default=1.0, gt=0.0)
    resample_modes_per_scenario: bool = False
    # Seeds ONLY the per-device fingerprint (mode) bank; None draws it from `seed`.
    mode_bank_seed: Optional[int] = None
    # Per-scenario fundamental operating-point variation (constant across the T steps).
    parameters: list[ParameterSpec] = Field(default_factory=list)
    factors: list[LatentFactor] = Field(default_factory=list)
    # Time-varying fundamental profile (per-step P/Q). Requires ``start_time``.
    profile: Optional[LoadProfileConfig] = None
    # Statistical device-class composition of aggregated loads (supersedes the
    # fingerprint + fundamental for the loads it covers). Requires ``start_time``.
    composition: Optional[CompositionConfig] = None
    #: Upstream harmonic background at the source, shared by every device on the feeder.
    background: Optional[BackgroundHarmonicConfig] = None
    # Absolute anchor for the profile's daily / weekly / seasonal phases (ISO 8601).
    # A naive (timezone-less) timestamp is interpreted as UTC.
    start_time: Optional[str] = None
    #: Orders deliberately left WITHOUT emission — the escape hatch of the silent-order
    #: guard below (e.g. the even orders of an odd-only device roster). Declaring them
    #: makes the intent explicit; leaving an order silent by accident raises.
    allow_silent_orders: tuple[int, ...] = ()
    #: Per-sequence start times drawn from a range (:class:`StartTimeRange`) instead of
    #: one ``start_time`` shared by every sequence; the two are exclusive. The time axis
    #: is then ``[B, T]`` and ``samples["time_unix_s"]`` a per-sequence record.
    start_time_range: Optional[StartTimeRange] = None
    #: Orders the FINGERPRINT devices (the ``selector`` devices no composition covers) do
    #: not emit at: their magnitude is zero there, e.g. the triplen orders of a
    #: three-phase three-wire inverter, which has no path for zero-sequence current. The
    #: bank and jitter are drawn as without it, so every other order is unchanged.
    #: ``None`` (default) leaves every order emitting.
    fingerprint_silent_orders: Optional[tuple[int, ...]] = None

    _OMIT_WHEN_NONE: ClassVar[tuple[str, ...]] = (
        "start_time_range",
        "fingerprint_silent_orders",
    )

    @property
    def is_timed(self) -> bool:
        """Whether the sequences are anchored to absolute time (``start_time`` or a range)."""
        return self.start_time is not None or self.start_time_range is not None

    @property
    def harmonic_orders(self) -> list[int]:
        """The solved order set a fingerprint sequence implies: fundamental + ``orders``."""
        return [1, *self.orders]

    def sample(self, grid) -> "SampledScenarios":
        """Draw the sequences (see :func:`~injection_hse.data.scenarios.sample_coherent_spectra`).

        This is what makes the config a :class:`pgml.scenarios.ScenarioSpec`, so
        ``run_scenarios(grid, config)`` solves a sequence batch without the engine
        knowing this recipe.
        """
        from .coherent import sample_coherent_spectra

        return sample_coherent_spectra(grid, self)

    @model_validator(mode="after")
    def _check(self) -> "CoherentSpectrumConfig":
        if any(o < 2 for o in self.orders):
            raise ValueError("harmonic `orders` must all be >= 2 (1 = fundamental).")
        self._check_emission_covers_orders()
        if self.emission_class != "auto" and self.harmonic_reference != "iec61000-3-2":
            raise ValueError(
                "emission_class is only valid with harmonic_reference='iec61000-3-2'."
            )
        for spec in self.parameters:
            if spec.is_harmonic:
                raise ValueError(
                    f"CoherentSpectrumConfig.parameters spec {spec.name!r} is harmonic "
                    f"(field={spec.field!r}); the fingerprint owns the harmonic spectrum. "
                    "parameters may vary only the fundamental (p/q/pq/u_ref)."
                )
        if self.start_time is not None and self.start_time_range is not None:
            raise ValueError(
                "CoherentSpectrumConfig: set either start_time (one window for every "
                "sequence) or start_time_range (a start per sequence), not both."
            )
        if self.profile is not None and not self.is_timed:
            raise ValueError(
                "CoherentSpectrumConfig.profile requires start_time (ISO 8601) or "
                "start_time_range: the daily / weekly / seasonal phases need an absolute "
                "anchor."
            )
        if self.composition is not None and not self.is_timed:
            raise ValueError(
                "CoherentSpectrumConfig.composition requires start_time (ISO 8601) or "
                "start_time_range: the device activity model is temporal (diurnal "
                "activity rates need an absolute anchor)."
            )
        if self.start_time is not None:
            try:
                datetime.fromisoformat(self.start_time)
            except ValueError as exc:
                raise ValueError(
                    f"start_time {self.start_time!r} is not a valid ISO 8601 "
                    "timestamp (e.g. '2024-06-21T00:00:00')."
                ) from exc
        return self

    def _check_emission_covers_orders(self) -> None:
        """Reject a spectrum source that is silent at a requested order.

        With a ``composition`` the source is the device roster: an order no class emits
        at carries no injection at all. Without one the source is the fingerprint bank,
        whose magnitudes are fractions of the ``harmonic_reference`` limit, so an order
        the EN 50160 table does not list is equally silent (the IEC 61000-3-2 reference
        caps every order >= 2, :mod:`injection_hse.data.scenarios.emission_caps`). Both cases
        produce a dataset whose top orders hold nothing but numerical noise;
        ``allow_silent_orders`` declares the ones that are meant to stay quiet.
        """
        allowed = {int(o) for o in self.allow_silent_orders}
        if self.composition is not None:
            silent = [
                o
                for o in composition_silent_orders(
                    self.composition.classes, list(self.orders)
                )
                if o not in allowed
            ]
            if silent:
                raise ValueError(
                    f"no device class emits at order(s) {silent}: the composed loads "
                    "would inject nothing there while the run solves for them. Extend "
                    "the classes' harmonic_magnitude, drop the orders, or list them in "
                    "allow_silent_orders to declare the silence deliberate."
                )
            return
        if self.harmonic_reference != "en50160":
            return  # no reference (absolute pu) or IEC 61000-3-2, which caps every order
        from pgml.scenarios import en50160_limits

        referenced = {o for o, limit in en50160_limits().items() if limit > 0.0}
        silent = [
            int(o)
            for o in sorted(self.orders)
            if o not in referenced and o not in allowed
        ]
        if silent:
            raise ValueError(
                f"the {self.harmonic_reference} reference has no limit at order(s) "
                f"{silent}, so the fingerprint would inject nothing there. Drop the "
                "orders, choose another harmonic_reference, or list them in "
                "allow_silent_orders."
            )


__all__ = [
    "LoadProfileConfig",
    "ActivityCalendar",
    "StartTimeRange",
    "ActivityPreset",
    "DeviceState",
    "DeviceClassSpec",
    "ClassCount",
    "ConsumerComposition",
    "CompositionConfig",
    "CoherentSpectrumConfig",
    "DEVICE_LIBRARY_VERSION",
    "EMISSION_MAX_ORDER",
    "EVEN_ORDER_RATIO",
    "ASYMMETRIC_EVEN_ORDER_RATIO",
    "continue_emission_bands",
    "default_device_classes",
    "default_compositions",
    "composition_silent_orders",
]
