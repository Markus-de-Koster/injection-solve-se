"""injection_hse.data.scenarios — how a state-estimation training set's scenarios are composed.

pgml owns the batch contract: what a solver input can be, how a batch of them is solved,
and how the result is persisted. This package owns the recipe — which device population
sits behind each meter point, what it emits as a function of its loading, and how that
moves over a day — because those are calibrated modeling choices of this study, not
properties of the engine.

Entry points, in the order a dataset is produced:

- :func:`se_random_scenario_config` — the calibrated RANDOMIZED snapshot recipe
  (correlated load levels, per-phase unbalance, a slack-voltage draw, IEC 61000-3-2
  referenced emission with the load-dependent emission law). Returns an
  :class:`EmissionLawScenarioConfig`.
- :func:`se_benchmark_scenario_config` — the same recipe at the published benchmark's
  harmonic order set, the one front door the benchmark and the corpus share.
- :func:`se_coherent_scenario_config` — the calibrated COHERENT sequence recipe (a stable
  per-device harmonic fingerprint over a step sequence, optionally over a statistical
  device composition and a time-varying fundamental). Returns a
  :class:`CoherentSpectrumConfig`.
- Both go straight into :func:`run_scenarios`: each config's own ``sample(grid)`` makes
  it a scenario spec, so the engine needs no knowledge of this package.
- :mod:`injection_hse.data.scenarios.emission` — the load-dependent emission law both recipes
  share (:func:`affine_emission_correction`, :func:`phase_slope_shift`,
  :data:`LOADING_FLOOR`) and its application to a randomized batch
  (:class:`EmissionLaw`, :func:`apply_emission_laws`).
- :mod:`injection_hse.data.scenarios.emission_caps` — the IEC 61000-3-2 emission cap of every
  order (:func:`iec61000_3_2_cap`): the standard's table where it lists the order, its
  envelope elsewhere, so a referenced device is never silent at an analyser order.
- :mod:`injection_hse.data.scenarios.discrete` — large discrete devices (EV charger, heat pump,
  drive) added to the loads of a randomized batch, each in one operating state per
  snapshot, so a load's injection is a mixture of a few spectra
  (:class:`DiscreteEmitterConfig`, :func:`apply_discrete_emitters`).
- :func:`read_dataset` — the front door onto :func:`pgml.scenarios.read_dataset` that knows
  how to rebuild the configs defined here.
- :mod:`injection_hse.data.scenarios.perturbation` — network-model error: the same scenario batch
  solved on perturbed copies of the grid (:class:`GridPerturbationConfig`,
  :func:`draw_grid_perturbations`, :func:`run_perturbed_scenarios`) while the estimator keeps
  the nominal grid.
- :data:`GENERATION_PROVENANCE` — the stamps to pass to
  ``pgml.scenarios.write_dataset(provenance=...)`` so a dataset records the recipe and
  device-library revisions its config does not capture, and the ideal-switch model the
  front door solved with (:data:`RECIPE_PROVENANCE` holds the revisions alone).
"""

from __future__ import annotations

from pgml.scenarios import read_dataset as _read_dataset
from pgml.scenarios import run_scenarios as _run_scenarios

from injection_hse._switches import ESTIMATOR_IDEAL_SWITCH_OHM, materialize_estimator_grid

from .coherent import sample_coherent_spectra
from .composition import (
    CompositionDraw,
    resolve_composed_ids,
    sample_device_composition,
)
from .config import (
    ASYMMETRIC_EVEN_ORDER_RATIO,
    DEVICE_LIBRARY_VERSION,
    EMISSION_MAX_ORDER,
    EVEN_ORDER_RATIO,
    ActivityCalendar,
    ActivityPreset,
    ClassCount,
    CoherentSpectrumConfig,
    CompositionConfig,
    ConsumerComposition,
    DeviceClassSpec,
    DeviceState,
    LoadProfileConfig,
    StartTimeRange,
    composition_silent_orders,
    continue_emission_bands,
    default_compositions,
    default_device_classes,
)
from .discrete import (
    DiscreteEmitterConfig,
    EmitterCount,
    EmitterRule,
    apply_discrete_emitters,
)
from .emission import (
    LOADING_FLOOR,
    EmissionLaw,
    EmissionLawScenarioConfig,
    StoredScenarioConfig,
    affine_emission_correction,
    apply_emission_laws,
    phase_slope_shift,
)
from .emission_caps import (
    IEC61000_3_2_MAX_ORDER,
    iec61000_3_2_cap,
    iec61000_3_2_caps,
)
from .presets import (
    EMISSION_FLOOR,
    EMISSION_FLOOR_PHASE_DEG,
    EMISSION_PHASE_SLOPE_DEG,
    HIGH_ACTIVITY_START_TIME,
    SE_PRESET_VERSION,
    se_benchmark_scenario_config,
    se_coherent_scenario_config,
    se_random_scenario_config,
)
from .perturbation import (
    GRID_COPY_RECORD,
    PERTURBATION_KINDS,
    PERTURBATION_POOLS,
    GridPerturbation,
    GridPerturbationConfig,
    PerturbationSpread,
    copy_blocks,
    draw_grid_perturbations,
    perturb_grid,
    perturbation_provenance,
    run_perturbed_scenarios,
)
from .profiles import (
    ProfileDraw,
    apply_load_profiles,
    load_profile_factors,
    sequence_start_times,
    sequence_time_axis,
)

#: Config classes defined here, keyed by the ``config_type`` string
#: :func:`pgml.scenarios.write_dataset` records. Passed to
#: :func:`pgml.scenarios.read_dataset` so a dataset generated by this package reads back
#: as a typed config rather than a raw dict.
SCENARIO_CONFIG_TYPES = {
    "CoherentSpectrumConfig": CoherentSpectrumConfig,
    "EmissionLawScenarioConfig": EmissionLawScenarioConfig,
}

#: Version stamps of the recipe inputs that change generated data WITHOUT changing the
#: config: the calibrated preset revision and the shipped device-class library revision.
RECIPE_PROVENANCE = {
    "se_preset_version": SE_PRESET_VERSION,
    "device_library_version": DEVICE_LIBRARY_VERSION,
}

#: What a dataset generated through :func:`run_scenarios` should record: the recipe
#: stamps plus the switch model, since this front door solves an exactly ideal switch as a
#: contact of ``ESTIMATOR_IDEAL_SWITCH_OHM`` ohm where pgml on its own fuses its terminals
#: (``ideal_switch_ohm`` of ``0.0``). Hand to
#: ``pgml.scenarios.write_dataset(provenance=GENERATION_PROVENANCE)``. A generator that
#: solves a recipe through pgml directly stamps :data:`RECIPE_PROVENANCE` and
#: ``ideal_switch_ohm: 0.0`` instead.
GENERATION_PROVENANCE = {
    **RECIPE_PROVENANCE,
    "ideal_switch_ohm": ESTIMATOR_IDEAL_SWITCH_OHM,
}

# Canonical __module__ for the public re-exports, so documentation and error messages
# name the package rather than the private submodule.
for _name in [
    "LoadProfileConfig",
    "ActivityCalendar",
    "StartTimeRange",
    "DeviceState",
    "DeviceClassSpec",
    "ClassCount",
    "ConsumerComposition",
    "CompositionConfig",
    "CoherentSpectrumConfig",
    "EmissionLaw",
    "EmissionLawScenarioConfig",
    "GridPerturbationConfig",
    "PerturbationSpread",
]:
    _obj = locals().get(_name)
    if _obj is not None:
        _obj.__module__ = __name__
CompositionDraw.__module__ = __name__
ProfileDraw.__module__ = __name__
GridPerturbation.__module__ = __name__


def read_dataset(path, *, config_types=None):
    """Reload a dataset, reconstructing the configs this package defines.

    Thin front door onto :func:`pgml.scenarios.read_dataset` that supplies
    :data:`SCENARIO_CONFIG_TYPES`, so a coherent dataset comes back with a real
    :class:`CoherentSpectrumConfig` instead of a dict. A randomized dataset written with
    pgml 0.5.0, whose ``ScenarioConfig`` declared the emission law through sampler fields
    the engine has dropped, comes back as an :class:`EmissionLawScenarioConfig` that draws
    the same batch. ``config_types`` adds further classes for a caller with its own.

    Parameters
    ----------
    path:
        Dataset directory written by :func:`pgml.scenarios.write_dataset`.
    config_types:
        Extra ``{class_name: class}`` entries, merged over
        :data:`SCENARIO_CONFIG_TYPES`.

    Returns
    -------
    pgml.scenarios.LoadedDataset
        The reloaded dataset.
    """
    return _read_dataset(
        path,
        config_types={
            "ScenarioConfig": StoredScenarioConfig,
            **SCENARIO_CONFIG_TYPES,
            **(config_types or {}),
        },
    )


def run_scenarios(grid, config, *, load_shunt=None, load_shunt_basis=None, **kwargs):
    """Run a PGL dataset recipe on the finite full-row estimator grid.

    This is the PGL data-generation front door onto :func:`pgml.scenarios.run_scenarios`.
    It preserves the engine's result and keyword contract while ensuring that generated
    voltages use the same ideal-switch approximation as PGL's graph, network solver, and
    branch-current operator.

    ``load_shunt`` (``"none"`` / ``"opendss"`` / ``"motor"``) and ``load_shunt_basis``
    (``"operating_point"`` / ``"nameplate"``) select the harmonic device shunt of the
    generating solve, forwarded unchanged; ``None`` is pgml's documented default for
    either. A dataset meant for the injection decode is consistent with it when generated
    on the ``"nameplate"`` basis with the model the estimator's
    :attr:`injection_hse.config.PhysicsConfig.load_shunt` names: the decode solves a batch with one
    ``Y(h)`` per order and cannot follow a per-scenario operating-point shunt.
    """
    return _run_scenarios(
        materialize_estimator_grid(grid),
        config,
        load_shunt=load_shunt,
        load_shunt_basis=load_shunt_basis,
        **kwargs,
    )


__all__ = [
    "LoadProfileConfig",
    "ActivityCalendar",
    "StartTimeRange",
    "run_scenarios",
    "ActivityPreset",
    "DeviceState",
    "DeviceClassSpec",
    "ClassCount",
    "ConsumerComposition",
    "CompositionConfig",
    "CompositionDraw",
    "CoherentSpectrumConfig",
    "ProfileDraw",
    "DEVICE_LIBRARY_VERSION",
    "SE_PRESET_VERSION",
    "HIGH_ACTIVITY_START_TIME",
    "EMISSION_FLOOR",
    "EMISSION_FLOOR_PHASE_DEG",
    "EMISSION_PHASE_SLOPE_DEG",
    "LOADING_FLOOR",
    "EmissionLaw",
    "EmissionLawScenarioConfig",
    "affine_emission_correction",
    "phase_slope_shift",
    "apply_emission_laws",
    "DiscreteEmitterConfig",
    "EmitterCount",
    "EmitterRule",
    "apply_discrete_emitters",
    "GENERATION_PROVENANCE",
    "RECIPE_PROVENANCE",
    "SCENARIO_CONFIG_TYPES",
    "composition_silent_orders",
    "EMISSION_MAX_ORDER",
    "EVEN_ORDER_RATIO",
    "ASYMMETRIC_EVEN_ORDER_RATIO",
    "continue_emission_bands",
    "IEC61000_3_2_MAX_ORDER",
    "iec61000_3_2_cap",
    "iec61000_3_2_caps",
    "default_device_classes",
    "default_compositions",
    "se_random_scenario_config",
    "se_coherent_scenario_config",
    "se_benchmark_scenario_config",
    "sample_coherent_spectra",
    "sample_device_composition",
    "resolve_composed_ids",
    "load_profile_factors",
    "apply_load_profiles",
    "sequence_start_times",
    "sequence_time_axis",
    "read_dataset",
    "GRID_COPY_RECORD",
    "PERTURBATION_KINDS",
    "PERTURBATION_POOLS",
    "GridPerturbation",
    "GridPerturbationConfig",
    "PerturbationSpread",
    "copy_blocks",
    "draw_grid_perturbations",
    "perturb_grid",
    "perturbation_provenance",
    "run_perturbed_scenarios",
]
