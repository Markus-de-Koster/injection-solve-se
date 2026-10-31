"""Calibrated scenario recipes for harmonic state-estimation training data.

One place defines HOW a state-estimation dataset is excited. Every generator — the
single-grid workflow, the multi-grid corpus builder and the benchmark helper
:func:`se_benchmark_scenario_config` (at the order set of :mod:`pgml.grids`) — builds
its :class:`EmissionLawScenarioConfig` / :class:`CoherentSpectrumConfig` here, so a
recipe fix reaches all of them and two datasets carrying the same preset version were
drawn the same way.

The recipe, and why each part of it is there:

- Load level: ``U(0, 1)`` of nameplate, coupled through a shared ``demand`` latent
  (rank correlation 0.5). Independent per-load draws leave the AGGREGATE of a few
  hundred sites nearly constant (the mean concentrates as ``1/sqrt(N)``), so the feeder
  never reaches a system-wide peak or valley; the zero end covers the light-load corner
  a real LV feeder spends much of its time in.
- Per-phase unbalance: a small perturbation of fractional standard deviation 0.15
  around the balanced base (``symmetry="small_imbalance"``), which composes with the
  correlation — fully independent phases have no component-level base to correlate.
- Slack voltage: ``Normal(1.0, 0.0333)`` on the source reference, so the estimator is
  not fitted to one boundary condition; +/-10 % is reached only at three sigma.
- Harmonic emission: a fraction of the per-device IEC 61000-3-2 CURRENT-emission
  limit — the appliance emission standard, which is the physically correct reference for
  a device current fingerprint. The DIN EN 50160 supply-VOLTAGE compatibility levels are
  never used as current fractions here: read as per-device emission they are roughly an
  order of magnitude too small. The fraction spans ``[0, 2]``: the zero end produces
  near-clean loads (otherwise emission is fully entangled with the operating point and an
  estimator shortcuts every harmonic from the load level), the upper end keeps the
  measured max-THD envelope (median ~3.2 %, p99 ~5.8 % on CIGRE LV).
- Emission phase diversity: per (device, order), widening with order (+/-30 deg at
  h3 to the full circle from h13). Without it every device injects at 0 deg, all
  injections add coherently and the harmonic voltage field carries no cancellation.
- Load-dependent emission: the drawn fraction is the RATED ratio, and the device's
  actual ratio follows the complex affine law ``I_h(lam) = A_h + B_h * lam``
  (:mod:`injection_hse.data.scenarios.emission`) against the loading its own load draw realised:
  a load-independent floor of 43-73 % of the rated phasor (``emission_floor``) at
  100-150 deg to the proportional part (``emission_floor_phase_deg``), plus an explicit
  phase slope of +/-25 deg per unit loading (``phase_slope_deg``). In the measured device
  population behind these ranges (``docs/pgl/harmonic-emission.md``) the ratio to the
  fundamental at 10 % load is several times the rated one, the harmonic angle rotates as
  a device unloads and the magnitude shows a cancellation null inside the operating
  range; a proportional draw has none of it, and a state estimator trained on it can
  never learn how a harmonic follows the fundamental. The three ranges are the ones the
  composed device library carries, so the randomized and coherent recipes below share
  one law; ``(0, 0)`` for all three reproduces the proportional recipe bit-for-bit.
- PV inverter emission: its own h5-dominant per-order shape (a load's class-D
  spectrum is h3-dominant) — the contrast that separates a generation/consumption pair
  whose fundamentals cancel at a shared bus. Spans are calibrated to the measured
  populations (certification workbooks plus lab racks, including the measured h17 bump).
- Coherent sequences: anchored at 16:00 local summer time, the high-activity band
  (PV still producing, households ramping, EV arrivals). The time axis is one shared
  window for all scenarios, so a midnight anchor leaves most device rosters dark.

Sizes (``n_samples``/``n_scenarios``/``seed``) and the harmonic ``orders`` are always
arguments: one recipe scales from a smoke test to a production run, and the requested
order set flows into the generated spectrum rather than being configured twice.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pgml.errors import InputError
from pgml.scenarios import (
    BackgroundHarmonicConfig,
    Constant,
    Correlation,
    LatentFactor,
    Normal,
    ParameterSpec,
    Selector,
    Uniform,
)

from .config import CoherentSpectrumConfig, CompositionConfig, LoadProfileConfig
from .emission import EmissionLaw, EmissionLawScenarioConfig

__all__ = [
    "SE_PRESET_VERSION",
    "HIGH_ACTIVITY_START_TIME",
    "EMISSION_FLOOR",
    "EMISSION_FLOOR_PHASE_DEG",
    "EMISSION_PHASE_SLOPE_DEG",
    "LOAD_PHASE_SPAN_DEG",
    "PV_EMISSION_HIGH",
    "PV_PHASE_SPAN_DEG",
    "se_random_scenario_config",
    "se_coherent_scenario_config",
    "se_benchmark_scenario_config",
]

#: Version of the calibrated recipe below. Bumped whenever a default changes the drawn
#: population; recorded beside a generated dataset so two datasets can be compared.
SE_PRESET_VERSION = "3"

#: Absolute anchor of the coherent sequences' diurnal phase: the late-afternoon band in
#: which households ramp, EVs arrive and PV still produces.
HIGH_ACTIVITY_START_TIME = "2024-06-21T16:00:00"

#: Per-order emission PHASE half-width [deg] of a load's harmonic current. Measured
#: device populations show a preferred angle at the low orders (a rectifier's conduction
#: angle pins h3) and none from h13 on, where the scatter saturates at the full circle.
LOAD_PHASE_SPAN_DEG: dict[int, float] = {3: 30.0, 5: 60.0, 7: 90.0, 9: 120.0, 11: 150.0}
#: Phase half-width of any order not tabulated above (the saturated full circle).
FULL_CIRCLE_DEG = 180.0
#: Range of the load-INDEPENDENT share of a device's rated harmonic phasor (the affine
#: emission law's ``|A_h| / (|A_h| + |B_h|)``), fitted to the loading sweeps of the
#: measured device population (certified PV inverters and laboratory racks,
#: ``docs/pgl/harmonic-emission.md``). The same range the composed device library
#: carries, so both recipes draw one law.
EMISSION_FLOOR: tuple[float, float] = (0.43, 0.73)
#: Range of the angle between the floor and the load-proportional part [deg]; near
#: anti-phase, which is what produces the measured cancellation null inside the
#: operating range and the rotation of the emission angle with loading.
EMISSION_FLOOR_PHASE_DEG: tuple[float, float] = (100.0, 150.0)
#: Range of the explicit emission phase slope [deg per unit loading], on top of the
#: affine law's own rotation (the composed library spans +/-15 to +/-30 by class).
EMISSION_PHASE_SLOPE_DEG: tuple[float, float] = (-25.0, 25.0)

#: Per-order upper end of a PV inverter's emission, as a fraction of the device's OWN
#: fundamental current. h5-dominant, unlike the h3-dominant class-D load spectrum. The
#: spans reach above an inverter's rated-output distortion because that FRACTION rises
#: steeply as the output falls, and a randomized snapshot has no loading state to key it
#: on. h11 and above are calibrated to the measured populations (certification workbooks
#: plus lab racks; the h17 bump is measured — lab-rack median 4.8 % of the fundamental).
PV_EMISSION_HIGH: dict[int, float] = {
    3: 0.10,
    5: 0.15,
    7: 0.08,
    9: 0.05,
    11: 0.045,
    13: 0.035,
    15: 0.03,
    17: 0.06,
    19: 0.025,
}
#: Emission phase half-width [deg] per order for a PV inverter. Wider than a load's at
#: h3 (+/-60): an inverter's third-harmonic angle is not pinned by a conduction angle.
PV_PHASE_SPAN_DEG: dict[int, float] = {
    3: 60.0,
    5: 60.0,
    7: 90.0,
    9: 120.0,
    11: 150.0,
}

#: Envelope continuing :data:`PV_EMISSION_HIGH` at an order it does not tabulate:
#: ``_PV_ENVELOPE / order``, anchored on the h3 entry (a ``1/h`` decline, the shape of
#: the standard's own odd-order limits), times :data:`_PV_EVEN_RATIO` at an even order.
_PV_ENVELOPE = 3.0 * PV_EMISSION_HIGH[3]
#: Even-to-neighbouring-odd emission ratio of a symmetric converter, read off the
#: IEC 61000-3-2 Class A limits (h2/h3 = 0.47, h4/h5 = 0.38, h6/h7 = 0.39, h8/h9 = 0.58).
_PV_EVEN_RATIO = 0.4


def _injected_orders(orders: Sequence[int]) -> list[int]:
    """The injected orders of a solved set: sorted, unique, order 1 (fundamental) removed.

    An empty result is legitimate — a fundamental-only run injects no spectrum at all.
    Every order above 1 has an IEC 61000-3-2 emission reference: the standard's table
    where it lists the order, its envelope elsewhere
    (:mod:`injection_hse.data.scenarios.emission_caps`).
    """
    return sorted({int(o) for o in orders if int(o) > 1})


def _pv_emission_high(order: int) -> float:
    """Upper end of the PV emission fraction at ``order`` (tabulated or continued)."""
    tabulated = PV_EMISSION_HIGH.get(int(order))
    if tabulated is not None:
        return tabulated
    envelope = _PV_ENVELOPE / float(order)
    return envelope if int(order) % 2 else envelope * _PV_EVEN_RATIO


def _phase_specs(
    orders: Sequence[int],
    spans: dict[int, float],
    *,
    name: str,
    selector: Selector,
    per: str = "each",
) -> list[ParameterSpec]:
    """Per-order emission-phase specs; the full-circle orders share ONE spec.

    Orders with a tabulated half-width get their own ``<name>_h<order>`` spec; every
    remaining order draws from the full circle and is collected into ``<name>_high``,
    because a spec is one sampling dimension per (device, order) either way and one
    grouped spec keeps the parameter list readable at twenty orders.
    """
    specs: list[ParameterSpec] = []
    for order in orders:
        span = spans.get(int(order))
        if span is None:
            continue
        specs.append(
            ParameterSpec(
                name=f"{name}_h{int(order)}",
                selector=selector,
                distribution=Uniform(low=-span, high=span),
                field="h_phase",
                mode="absolute",
                per=per,
                orders=[int(order)],
            )
        )
    saturated = [int(o) for o in orders if int(o) not in spans]
    if saturated:
        specs.append(
            ParameterSpec(
                name=f"{name}_high",
                selector=selector,
                distribution=Uniform(low=-FULL_CIRCLE_DEG, high=FULL_CIRCLE_DEG),
                field="h_phase",
                mode="absolute",
                per=per,
                orders=saturated,
            )
        )
    return specs


def _emission_law_specs(
    orders: Sequence[int],
    *,
    name: str,
    selector: Selector,
    floor: tuple[float, float],
    floor_phase_deg: tuple[float, float],
    slope_deg: tuple[float, float],
    per: str = "each",
) -> tuple[list[ParameterSpec], Optional[EmissionLaw]]:
    """The emission law of one device group: its specs and the law that names them.

    One ``h_param`` spec per law parameter — ``<name>_floor`` (the affine law's
    load-independent share), ``<name>_floor_phase`` (its angle) and ``<name>_slope`` (the
    explicit phase slope) — each drawn per device and order like the emission itself. A
    range that is ``(0, 0)`` emits NO spec: the law is inert at zero anyway, and not
    consuming a sampling dimension keeps every other draw of the recipe bit-identical, so
    the proportional recipe is exactly recoverable. The law is ``None`` when no spec is
    left.
    """
    specs: list[ParameterSpec] = []
    roles: dict[str, str] = {}
    for role, rng in (
        ("floor", floor),
        ("floor_phase", floor_phase_deg),
        ("slope", slope_deg),
    ):
        lo, hi = float(rng[0]), float(rng[1])
        if lo == 0.0 and hi == 0.0:
            continue
        roles[role] = f"{name}_{role}"
        specs.append(
            ParameterSpec(
                name=roles[role],
                selector=selector,
                distribution=Constant(value=lo)
                if lo == hi
                else Uniform(low=lo, high=hi),
                field="h_param",
                mode="absolute",
                per=per,
                orders=[int(o) for o in orders],
            )
        )
    return specs, (EmissionLaw(**roles) if roles else None)


def _law_groups(grid, component: str, name: str, per_device: str, persistence: str):
    """``(spec name, selector, per)`` of the emission-law specs for one component kind.

    One group for the whole kind unless the persistence is ``"class"``, which emits one
    group per consumer class present in the grid (``Selector(consumer_type=...)``, the
    untyped devices by id) with ``per="class"`` — the law is then a constant of the class.
    """
    from pgml.schemas.grid_schema import Generator, Load

    if persistence != "class":
        return [(name, Selector(component=component), per_device)]
    cls = {"load": Load, "generator": Generator}[component]
    by_type: dict = {}
    untyped: list[int] = []
    for a in grid.appliances:
        if not isinstance(a, cls) or not a.in_service:
            continue
        ctype = getattr(a, "consumer_type", None)
        if ctype is None:
            untyped.append(int(a.id))
        else:
            by_type.setdefault(str(getattr(ctype, "value", ctype)), None)
    groups = [
        (f"{name}_{ctype}", Selector(component=component, consumer_type=ctype), "class")
        for ctype in sorted(by_type)
    ]
    if untyped:
        groups.append(
            (f"{name}_untyped", Selector(component=component, ids=untyped), "class")
        )
    return groups


def _has_generator(grid) -> bool:
    from pgml.schemas.grid_schema import Generator

    return any(isinstance(a, Generator) for a in grid.appliances)


def _has_pv(grid) -> bool:
    from pgml.schemas.grid_schema import ConsumerType, Generator

    return any(
        isinstance(a, Generator) and a.consumer_type == ConsumerType.PV
        for a in grid.appliances
    )


def _fundamental_specs(
    grid,
    *,
    load_scale: tuple[float, float],
    load_correlation: Optional[float],
    imbalance: float,
    pv_scale: tuple[float, float],
    pv_correlation: Optional[float],
    slack_voltage_std: float,
) -> tuple[list[ParameterSpec], list[LatentFactor]]:
    """The fundamental operating point: load level, inverter output, slack voltage.

    Returns ``(specs, factors)``. A ``load_correlation`` / ``pv_correlation`` of ``None``
    means "no shared latent": the loads then vary independently and the inverters all
    follow ONE scale (``per="shared"``, the single-irradiance case of a small grid).
    """
    load = ParameterSpec(
        name="load_scale",
        selector=Selector(component="load"),
        distribution=Uniform(low=load_scale[0], high=load_scale[1]),
        field="pq",
        mode="scale",
        per="each",
        symmetry="small_imbalance" if imbalance > 0.0 else "balanced",
        imbalance=float(imbalance),
        correlation=(
            None
            if load_correlation is None
            else Correlation(factor="demand", rho=float(load_correlation))
        ),
    )
    specs = [load]
    factors = [] if load_correlation is None else [LatentFactor(name="demand")]
    if _has_generator(grid):
        specs.append(
            ParameterSpec(
                name="pv_scale",
                selector=Selector(component="generator"),
                distribution=Uniform(low=pv_scale[0], high=pv_scale[1]),
                field="pq",
                mode="scale",
                per="each" if pv_correlation is not None else "shared",
                correlation=(
                    None
                    if pv_correlation is None
                    else Correlation(factor="solar", rho=float(pv_correlation))
                ),
            )
        )
        if pv_correlation is not None:
            factors.append(LatentFactor(name="solar"))
    if slack_voltage_std > 0.0:
        specs.append(
            ParameterSpec(
                name="source_scale",
                selector=Selector(component="source"),
                distribution=Normal(loc=1.0, scale=float(slack_voltage_std)),
                field="u_ref",
                mode="scale",
                per="shared",
            )
        )
    return specs, factors


def se_random_scenario_config(
    grid,
    *,
    orders: Sequence[int],
    n_samples: int,
    seed: int,
    method: str = "sobol",
    load_scale: tuple[float, float] = (0.0, 1.0),
    load_correlation: Optional[float] = 0.5,
    imbalance: float = 0.15,
    spectrum_fraction: tuple[float, float] = (0.0, 2.0),
    pv_scale: tuple[float, float] = (0.0, 1.0),
    pv_correlation: Optional[float] = None,
    slack_voltage_std: float = 0.0333,
    emission_floor: tuple[float, float] = EMISSION_FLOOR,
    emission_floor_phase_deg: tuple[float, float] = EMISSION_FLOOR_PHASE_DEG,
    phase_slope_deg: tuple[float, float] = EMISSION_PHASE_SLOPE_DEG,
    emission_persistence: Literal["scenario", "device", "class"] = "scenario",
    background: Optional[BackgroundHarmonicConfig] = None,
) -> EmissionLawScenarioConfig:
    """The randomized-snapshot recipe: independent operating points, redrawn per scenario.

    ``background`` places the upstream harmonic background of
    :class:`~pgml.scenarios.BackgroundHarmonicConfig` behind every snapshot — the
    distortion a feeder inherits from the supplying network, shared by all of its devices.
    Each snapshot draws its own level (a snapshot has no step axis for the drift to walk).

    Parameters
    ----------
    grid:
        The grid to excite. Read only for its appliance mix: the inverter specs are
        emitted when it carries a :class:`~pgml.schemas.grid_schema.Generator`, the PV
        emission specs when one is tagged ``consumer_type="pv"``.
    orders:
        The SOLVED harmonic orders. Order 1 is the fundamental and is not injected; every
        order above it gets a load-emission draw, an emission-phase draw and — on a PV
        grid — an inverter emission and phase draw, so the requested order set decides
        what is injected instead of being configured a second time. A fundamental-only
        set (``[1]``) yields the operating-point specs alone.
    n_samples, seed, method:
        Batch size, sampling seed and the unit-cube sampler (``"sobol"`` / ``"lhs"`` /
        ``"independent"``). ``(config, seed)`` reproduces the batch.
    load_scale, load_correlation, imbalance:
        Load apparent-power scale range, rank correlation through the shared ``demand``
        latent (``None`` = independent), and the per-phase unbalance std.
    spectrum_fraction:
        Range of the per-device IEC 61000-3-2 emission fraction (see the module
        docstring for why it reaches 0 and above 1).
    pv_scale, pv_correlation:
        Inverter output-scale range and its rank correlation through a shared ``solar``
        latent; ``None`` (default) gives the whole fleet ONE irradiance scale.
    slack_voltage_std:
        Standard deviation of the ``Normal(1.0, .)`` slack-reference scale; ``0``
        disables the slack draw.
    emission_floor, emission_floor_phase_deg, phase_slope_deg:
        The load-dependent emission law, drawn per device and order for loads and PV
        inverters alike (see the module docstring): the affine law's load-independent
        share and its angle, and the explicit phase slope. ``(0, 0)`` for a range emits
        no spec for it; all three at ``(0, 0)`` is the proportional recipe, bit-for-bit.
    emission_persistence:
        ``"scenario"`` (default) redraws every device's emission fraction, phase and law
        parameters in every scenario — a fresh device population per snapshot, so across
        the dataset the fundamental predicts a harmonic only through the law's mean.
        ``"device"`` draws them ONCE per device for the whole dataset (``per="fixed"``):
        each device keeps its signature, so its harmonic is a stable function of its own
        loading — the relation a learner can exploit, and one tied to this population's
        signatures (judge such a model on a population drawn with another seed).
        ``"class"`` makes the LAW — floor, floor angle, slope — a constant of the device's
        consumer class (one spec per class present in the grid, ``per="class"``: the same
        draw in every dataset, whatever the seed), while the emission fraction and phase
        stay one draw per device (``per="fixed"``, population-specific): the SHAPE of a
        device's harmonic response is then a class property that transfers across
        populations, its LEVEL a device property that does not. The operating point stays
        a fresh draw per scenario in every mode.

    Returns
    -------
    EmissionLawScenarioConfig
        The sampling template (a :class:`pgml.scenarios.ScenarioConfig` that applies the
        emission laws in its ``sample``); feed it to
        :func:`injection_hse.data.scenarios.run_scenarios`.
    """
    injected = _injected_orders(orders)
    laws: list[EmissionLaw] = []
    per_device = "each" if emission_persistence == "scenario" else "fixed"
    specs, factors = _fundamental_specs(
        grid,
        load_scale=load_scale,
        load_correlation=load_correlation,
        imbalance=imbalance,
        pv_scale=pv_scale,
        pv_correlation=pv_correlation,
        slack_voltage_std=slack_voltage_std,
    )
    load_selector = Selector(component="load")
    if injected:
        specs.append(
            ParameterSpec(
                name="load_spectrum",
                selector=load_selector,
                distribution=Uniform(
                    low=spectrum_fraction[0], high=spectrum_fraction[1]
                ),
                field="h_mag",
                per=per_device,
                orders=injected,
                harmonic_reference="iec61000-3-2",
            )
        )
        specs.extend(
            _phase_specs(
                injected,
                LOAD_PHASE_SPAN_DEG,
                name="spectrum_phase",
                selector=load_selector,
                per=per_device,
            )
        )
        for name, selector, per in _law_groups(
            grid, "load", "load_emission", per_device, emission_persistence
        ):
            law_specs, law = _emission_law_specs(
                injected,
                name=name,
                selector=selector,
                floor=emission_floor,
                floor_phase_deg=emission_floor_phase_deg,
                slope_deg=phase_slope_deg,
                per=per,
            )
            specs.extend(law_specs)
            laws.extend([law] if law is not None else [])
    if injected and _has_pv(grid):
        pv_selector = Selector(component="generator", consumer_type="pv")
        specs.extend(
            ParameterSpec(
                name=f"pv_spectrum_h{order}",
                selector=pv_selector,
                distribution=Uniform(low=0.0, high=_pv_emission_high(order)),
                field="h_mag",
                mode="absolute",
                per=per_device,
                orders=[order],
            )
            for order in injected
        )
        specs.extend(
            _phase_specs(
                injected,
                PV_PHASE_SPAN_DEG,
                name="pv_phase",
                selector=pv_selector,
                per=per_device,
            )
        )
        law_specs, law = _emission_law_specs(
            injected,
            name="pv_emission",
            selector=pv_selector,
            floor=emission_floor,
            floor_phase_deg=emission_floor_phase_deg,
            slope_deg=phase_slope_deg,
            per="class" if emission_persistence == "class" else per_device,
        )
        specs.extend(law_specs)
        laws.extend([law] if law is not None else [])
    return EmissionLawScenarioConfig(
        emission_laws=laws,
        n_samples=int(n_samples),
        seed=int(seed),
        method=method,
        parameters=specs,
        factors=factors,
        background=background,
    )


def se_coherent_scenario_config(
    grid,
    *,
    orders: Sequence[int],
    n_scenarios: int,
    n_steps: int,
    seed: int,
    mode: str = "composed",
    name: str = "coherent",
    step_size_s: float = 900.0,
    start_time: Optional[str] = HIGH_ACTIVITY_START_TIME,
    n_modes: int = 2,
    dwell: float = 0.9,
    mode_bank_seed: Optional[int] = None,
    fingerprint_fraction: tuple[float, float] = (0.0, 1.0),
    activity_scale: float = 1.0,
    behavioral_coupling: float = 0.3,
    cloud_coupling: float = 0.5,
    composition: Optional[CompositionConfig] = None,
    profile: Optional[LoadProfileConfig] = None,
    load_scale: tuple[float, float] = (0.0, 1.0),
    load_correlation: Optional[float] = 0.5,
    imbalance: float = 0.15,
    pv_scale: tuple[float, float] = (0.0, 1.0),
    pv_correlation: Optional[float] = None,
    slack_voltage_std: float = 0.0333,
    background: Optional[BackgroundHarmonicConfig] = None,
) -> CoherentSpectrumConfig:
    """The coherent-sequence recipe: a device population moving through time.

    Two temporal structures, selected by ``mode``:

    - ``"composed"`` (default) — a statistical device-class COMPOSITION whose per-step
      member activity moves the fundamental power AND the injected spectrum together, the
      coupling a state estimator has to learn. The composition covers the LOADS, so the
      per-device fingerprint machinery is pointed at the generators it cannot reach (a PV
      would otherwise emit nothing and be indistinguishable from a co-located load whose
      fundamental cancels against it), and those generators additionally follow a
      time-varying profile so their emission moves with their output.
    - ``"fingerprint"`` — the per-device mode bank alone: a stable signature that sticks
      to a mode and wanders around it over a per-scenario constant fundamental. Timeless;
      the ablation baseline that isolates the composed power/spectrum coupling.

    Parameters
    ----------
    grid:
        The grid to excite (read for its appliance mix, as in
        :func:`se_random_scenario_config`).
    orders:
        The SOLVED harmonic orders; order 1 is dropped (the fundamental is not injected).
    n_scenarios, n_steps, seed:
        Sequences ``B``, steps ``T`` per sequence, and the temporal seed.
    mode, name, step_size_s, start_time:
        Temporal structure, config name (the prefix of the recorded attribution samples),
        seconds per step, and the absolute anchor of the diurnal phase (composed mode
        only — the fingerprint is timeless).
    n_modes, dwell, mode_bank_seed, fingerprint_fraction:
        The fingerprint bank: base spectra per device, the probability of staying in a
        mode, the bank's own seed (``None`` = derived from ``seed``), and the range of
        the per-order emission fraction drawn for a mode.
    activity_scale, behavioral_coupling, cloud_coupling:
        Composition knobs: the duty-cycle multiplier lifting the population up its own
        diurnal curve (the class presets are per-device duty cycles, so an aggregate sits
        far below installed capacity), and how strongly consumption / PV co-vary.
    composition, profile:
        Explicit overrides. ``composition=None`` builds the default library at the knobs
        above; ``profile=None`` profiles the devices the composition does not cover.
    load_scale, load_correlation, imbalance, pv_scale, pv_correlation, slack_voltage_std:
        The per-scenario fundamental operating point (drawn once per sequence and held
        across the ``T`` steps), identical to the randomized recipe. A composed load's
        fundamental comes from the composition instead; the draw still shapes every
        uncomposed device.

    Returns
    -------
    CoherentSpectrumConfig
        The sequence template; feed it to :func:`injection_hse.data.scenarios.run_scenarios`.
    """
    if mode not in ("composed", "fingerprint"):
        raise InputError(
            f"coherent mode must be 'composed' or 'fingerprint', got {mode!r}."
        )
    injected = _injected_orders(orders)
    if not injected:
        raise InputError(
            f"a coherent sequence needs at least one harmonic order above 1, got "
            f"{list(orders)}: its device fingerprint IS the injected spectrum."
        )
    specs, factors = _fundamental_specs(
        grid,
        load_scale=load_scale,
        load_correlation=load_correlation,
        imbalance=imbalance,
        pv_scale=pv_scale,
        pv_correlation=pv_correlation,
        slack_voltage_std=slack_voltage_std,
    )
    composed = mode == "composed"
    # The composition supersedes the loads it covers, so in composed mode the fingerprint
    # addresses the generators instead. A grid without one keeps the load selector: a
    # selector matching no device yields zero-width sample columns the dataset cannot
    # store, and the composition covers those loads anyway.
    fingerprint_generators = composed and _has_generator(grid)
    return CoherentSpectrumConfig(
        name=name,
        selector=Selector(component="generator" if fingerprint_generators else "load"),
        orders=injected,
        n_steps=int(n_steps),
        n_scenarios=int(n_scenarios),
        n_modes=int(n_modes),
        dwell=float(dwell),
        seed=int(seed),
        mode_bank_seed=None if mode_bank_seed is None else int(mode_bank_seed),
        mag_distribution=Uniform(
            low=fingerprint_fraction[0], high=fingerprint_fraction[1]
        ),
        harmonic_reference="iec61000-3-2",
        step_size_s=float(step_size_s),
        parameters=specs,
        factors=factors,
        composition=(
            (
                composition
                if composition is not None
                else CompositionConfig(
                    activity_scale=float(activity_scale),
                    behavioral_coupling=float(behavioral_coupling),
                    cloud_coupling=float(cloud_coupling),
                )
            )
            if composed
            else None
        ),
        profile=(
            profile
            if profile is not None
            else (
                LoadProfileConfig(selector=Selector(component="generator"))
                if fingerprint_generators
                else None
            )
        ),
        start_time=start_time if (composed or profile is not None) else None,
        background=background,
    )


def se_benchmark_scenario_config(grid, *, n_samples: int, seed: int, orders=None):
    """The canonical randomized benchmark batch for a reference grid.

    A thin front door onto :func:`se_random_scenario_config` with the benchmark's harmonic
    order set, so the published benchmark, the workflow datasets and the multi-grid corpus
    cannot drift apart.

    Parameters
    ----------
    grid:
        The benchmark grid (read for its appliance mix).
    n_samples, seed:
        Batch size and sampling seed.
    orders:
        Injected harmonic orders; ``None`` uses ``pgml.grids.LOAD_HARMONIC_ORDERS``.

    Returns
    -------
    EmissionLawScenarioConfig
        The sampling template.
    """
    from pgml.grids import LOAD_HARMONIC_ORDERS

    return se_random_scenario_config(
        grid,
        orders=LOAD_HARMONIC_ORDERS if orders is None else orders,
        n_samples=n_samples,
        seed=seed,
    )
