"""Node-coherent harmonic "fingerprint" sampling over a step sequence.

Builds a ``[B, T]`` batch of harmonic injections in which every device keeps a stable,
recognisable per-node signature that varies realistically step to step — the training
signal a harmonic state estimator needs to attribute a pattern to a node. Each device
draws ``n_modes`` base spectra (operating "states"); over ``T`` steps it sticks to a
mode (Markov dwell) and wanders around it (AR(1) jitter), clamped to the per-order
emission reference (IEC 61000-3-2 by default; DIN EN 50160 or none optionally).

The output is a :class:`~pgml.scenarios.SampledScenarios` whose ``harmonic_injection``
carries ``[B, T]``-shaped per-(device, order) tensors; feeding it through
:func:`~pgml.scenarios.run_scenarios` (calculation="harmonic") yields node voltages
``[B, T, H, N]`` (the explicit time axis). Per-step timestamps and the ground-truth active
mode per device/step are recorded in ``samples`` for ML labels.

Sampling uses seeded RNG (not the QMC cube): the Markov mode path and the AR(1) jitter
are inherently sequential. Deterministic for a fixed seed; all torch ops, no
``.item()``/``.detach()``, honors broadcasting so the realized tensors feed the solver.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pgml.errors import InputError
from pgml.scenarios import (
    SampledScenarios,
    ScenarioConfig,
    broadcast_operating_point,
    build_background_sources,
    en50160_limit,
    sample,
)
from pgml.schemas.grid_schema import Grid

from pgml.scenarios import ar1_noise
from .emission_caps import iec61000_3_2_caps
from .composition import resolve_composed_ids, sample_device_composition
from .config import CoherentSpectrumConfig
from .profiles import apply_load_profiles, sequence_time_axis

_F64 = torch.float64

# The coherent operating-point unit cube is drawn on a stream DISTINCT from the
# fingerprint RNG (which is seeded from ``config.seed``): a fixed derived offset keeps
# the two independent, so the realized ``harmonic_injection`` is byte-identical with and
# without ``config.parameters`` (reproducible reconstruction from config + seed).
_OP_CUBE_SEED_OFFSET = 0x9E3779B9  # 2654435769; golden-ratio mix


def _split_shared(samples: dict, name: str, resampled: bool) -> dict:
    """Pop the records that carry NO leading scenario axis out of ``samples``.

    A sequence batch records a step-time vector, the device-id columns of its attribution
    blocks, the member roster and (unless it is redrawn per scenario) the fingerprint bank
    once for the whole batch. Declaring them keeps the persistence layer from having to
    guess from a leading dimension that may coincide with the scenario count. The absolute
    axis ``time_unix_s`` is shared only when it is one ``[T]`` window; a ``[B, T]`` axis
    (a start per sequence) stays a per-scenario record.
    """
    keys = [
        "time_s",
        f"{name}_device_ids",
        f"{name}_profile_device_ids",
        f"{name}_agg_ids",
        f"{name}_roster_p_rated",
    ]
    if not resampled:
        keys.append(f"{name}_mode_base_mag")
    time_unix_s = samples.get("time_unix_s")
    if time_unix_s is not None and time_unix_s.ndim == 1:
        keys.insert(1, "time_unix_s")
    return {key: samples.pop(key) for key in keys if key in samples}


def _markov_path(
    b: int, n_dev: int, t: int, n_modes: int, dwell: float, gen: torch.Generator
) -> Tensor:
    """``[B, n_dev, T]`` int64 mode path: stay w.p. ``dwell``, else switch uniformly."""
    path = torch.empty((b, n_dev, t), dtype=torch.long)
    cur = torch.randint(0, n_modes, (b, n_dev), generator=gen)
    path[..., 0] = cur
    for step in range(1, t):
        if n_modes > 1:
            stay = torch.rand((b, n_dev), generator=gen) < dwell
            offset = torch.randint(1, n_modes, (b, n_dev), generator=gen)
            cur = torch.where(stay, cur, (cur + offset) % n_modes)
        path[..., step] = cur
    return path


def _sample_operating_specs(
    grid: Grid, config: CoherentSpectrumConfig, b: int
) -> tuple[dict, dict, dict]:
    """Draw the coherent config's fundamental operating-point specs, once per scenario.

    Reuses the :class:`ScenarioConfig` sampler (Sobol unit cube, copula correlation,
    per-phase symmetry, and the source ``u_ref`` scale) with ``n_samples = b`` on a seed
    DERIVED from ``config.seed`` (a stream distinct from the fingerprint RNG, so the
    harmonic injection is unchanged). Returns ``(operating_point, samples)`` — the ``[B]``
    per-scenario operating point (constant across the ``T`` steps; the harmonic solve
    aligns the fundamental voltage against the ``[B, T]`` injection) and the raw ``[B, ...]``
    draws recorded in ``samples`` exactly as :class:`ScenarioConfig` records them. Empty
    when the config has no ``parameters``.
    """
    if not config.parameters:
        return {}, {}, {}
    op_seed = (int(config.seed) + _OP_CUBE_SEED_OFFSET) & 0x7FFFFFFF
    inner = ScenarioConfig(
        n_samples=b,
        seed=op_seed,
        method="sobol",
        parameters=list(config.parameters),
        factors=list(config.factors),
    )
    drawn = sample(grid, inner)
    return drawn.operating_point, drawn.samples, drawn.shared_samples


def sample_coherent_spectra(
    grid: Grid, config: CoherentSpectrumConfig
) -> SampledScenarios:
    """Sample node-coherent harmonic injection sequences (reproducible).

    Implements the Markov-mode + AR(1) jitter fingerprint model described in
    :class:`CoherentSpectrumConfig`.  The call is deterministic for a
    fixed ``config.seed``.

    Parameters
    ----------
    grid : Grid
        The reference grid.  Only the appliances matched by ``config.selector`` are
        varied; all others keep their nominal spectra.
    config : CoherentSpectrumConfig
        Fully specified sampling config (mode count, temporal structure, emission
        reference + clamping, etc.).  Validated on construction.

    Returns
    -------
    SampledScenarios
        ``operating_point`` is empty when ``config.parameters`` is empty (fundamental P/Q
        stays nominal, source at ``u_ref_v``); otherwise it carries the per-scenario
        fundamental draws as ``[B]`` tensors (constant across the ``T`` steps — the
        harmonic solve broadcasts the ``[B]`` fundamental against the ``[B, T]`` injection),
        incl. a per-source ``u_ref_scale`` for the slack. When ``config.profile`` is set
        the P/Q totals instead carry the step axis (``[B, T]``), a TIME-VARYING
        fundamental aligned with the ``[B, T]`` injection.
        ``harmonic_injection`` maps
        ``{device_id: {order: (mag[B, T], phase[B, T])}}`` — pass directly to
        ``solve_harmonic_flow``.
        ``samples`` contains the following keys (``name`` = ``config.name``,
        default ``"harmonics"``):

        - ``"<name>_mode"`` ``[B, n_dev, T]`` — active mode index per device per
          step; the ground-truth attribution label for ML training.
        - ``"<name>_mag"`` / ``"<name>_phase"`` ``[B, n_dev, n_ord, T]`` — the REALIZED
          injection per device and order after AR(1) jitter and the emission clamp:
          magnitude in per unit of the device's own fundamental current, phase in
          degrees. The device axis is ``"<name>_device_ids"``.
        - ``"<name>_mode_base_mag"`` — per-device mode fingerprint magnitudes
          (before jitter; shape ``[n_dev, n_ord, n_modes]`` or
          ``[B, n_dev, n_ord, n_modes]`` when ``resample_modes_per_scenario=True``).
        - ``"<name>_device_ids"`` ``[n_dev]`` — matched device IDs in selector order.
        - ``"time_s"`` ``[T]`` — step timestamps in seconds (relative to the start).

        When ``config.profile`` is set the fundamental P/Q is TIME-VARYING (per step),
        so ``operating_point`` carries the step axis (``[B, T]``) and ``samples`` also
        records the ground-truth profile:

        - ``"<name>_profile_factor"`` ``[B, n_dev, T]`` — the realized multiplicative
          profile factor per profiled device per step (ML ground truth).
        - ``"<name>_profile_device_ids"`` ``[n_dev]`` — the profiled device IDs.
        - ``"time_unix_s"`` — absolute per-step timestamps (epoch seconds): ``[T]``
          (``config.start_time`` + ``k * step_size_s``, shared by every sequence), or
          ``[B, T]`` with a ``config.start_time_range`` (each sequence from its own start).
          A composed batch records it too.

        When ``config.composition`` is set the covered loads leave the fingerprint device
        set, and their class attribution plus their REALIZED aggregate spectrum
        (``"<name>_composed_mag"`` / ``"<name>_composed_phase"`` ``[B, n_agg, n_ord, T]``
        on the ``"<name>_agg_ids"`` device axis, same units as ``"<name>_mag"``) are
        recorded instead — see
        :func:`~injection_hse.data.scenarios.composition.sample_device_composition`.

        Pass this :class:`SampledScenarios` to :func:`~pgml.scenarios.run_scenarios`
        (or directly to ``solve_harmonic_flow``) to obtain ``v[B, T, H, N]``.

    Raises
    ------
    ValueError
        If the selector matches no in-service components.

    Notes
    -----
    All intermediate tensors use ``float64``.  No ``.item()`` / ``.detach()`` calls
    are made; the realized tensors can feed the solver's autograd tape.
    """
    ids = config.selector.resolve(grid)
    nm = config.name
    n_ord = len(config.orders)
    n_modes, b, t = config.n_modes, config.n_scenarios, config.n_steps

    # A statistical device composition (if set) SUPERSEDES the mode-bank fingerprint for
    # the loads it covers: those ids are dropped from the fingerprint device set and
    # filled from the composition instead. Composition-free runs keep `fp_ids == ids`
    # (byte-identical).
    comp = config.composition
    composed_ids = resolve_composed_ids(grid, comp) if comp is not None else []
    composed_set = set(composed_ids)
    fp_ids = [i for i in ids if i not in composed_set]
    if not fp_ids and not composed_ids:
        raise InputError(
            "CoherentSpectrumConfig selector matched no in-service components."
        )

    samples: dict = {"time_s": torch.arange(t, dtype=_F64) * config.step_size_s}  # [T]
    harmonic_injection: dict = {}

    if fp_ids:
        n_dev = len(fp_ids)
        gen = torch.Generator().manual_seed(config.seed)
        # The per-device fingerprint (mode) bank draws from its own generator when
        # `mode_bank_seed` is set (a distinct signature bank for a held-out test set);
        # `None` draws it from `gen`, byte-identical to a config without the field -- the
        # bank is the first consumption of the `seed` stream (the Markov path + AR(1)
        # jitter follow). With `mode_bank_seed` set, the path + jitter no longer consume
        # the bank draws, so they are a distinct (statistically identical) realization.
        bank_gen = (
            gen
            if config.mode_bank_seed is None
            else torch.Generator().manual_seed(int(config.mode_bank_seed))
        )

        # 1. per-order emission caps (also the upper clamp). IEC 61000-3-2 is PER DEVICE
        #    (each device's nominal P + node voltage): shape [n_dev, n_ord, 1]. EN 50160
        #    is a global per-order voltage-compatibility level; None is absolute pu (cap
        #    1.0): shape [n_ord, 1]. The trailing 1 broadcasts over the mode axis (base
        #    multiply) and the T axis (clamp).
        ref = config.harmonic_reference
        clamp_to_cap = ref is not None
        if ref == "iec61000-3-2":
            # every order >= 2 has a cap: the standard's envelope where its table lists
            # none (injection_hse.data.scenarios.emission_caps)
            cap_map = iec61000_3_2_caps(
                grid, fp_ids, config.orders, emission_class=config.emission_class
            )
            caps = torch.tensor(
                [[cap_map[cid][o] for o in config.orders] for cid in fp_ids],
                dtype=_F64,
            ).reshape(n_dev, n_ord, 1)
        else:
            caps = torch.tensor(
                [en50160_limit(o) if ref == "en50160" else 1.0 for o in config.orders],
                dtype=_F64,
            ).reshape(n_ord, 1)

        # 2. base mode spectra (the fingerprints): magnitude (fraction of cap) + phase.
        mode_shape = (
            (b, n_dev, n_ord, n_modes)
            if config.resample_modes_per_scenario
            else (n_dev, n_ord, n_modes)
        )
        base_mag = (
            config.mag_distribution.icdf(
                torch.rand(mode_shape, generator=bank_gen, dtype=_F64)
            )
            * caps
        )  # [..., n_ord, n_modes], cap broadcasts over modes (+ devices for [n_ord, 1])
        base_phase = config.phase_distribution.icdf(
            torch.rand(mode_shape, generator=bank_gen, dtype=_F64)
        )

        # 3. Markov mode path + gather each step's base spectrum -> [B, n_dev, n_ord, T].
        path = _markov_path(b, n_dev, t, n_modes, config.dwell, gen)
        idx = path.unsqueeze(2).expand(b, n_dev, n_ord, t)  # [B, n_dev, n_ord, T]
        bm = base_mag if config.resample_modes_per_scenario else base_mag.unsqueeze(0)
        bp = (
            base_phase
            if config.resample_modes_per_scenario
            else base_phase.unsqueeze(0)
        )
        bm = bm.expand(b, n_dev, n_ord, n_modes).contiguous()
        bp = bp.expand(b, n_dev, n_ord, n_modes).contiguous()
        sel_mag = torch.gather(bm, -1, idx)  # [B, n_dev, n_ord, T]
        sel_phase = torch.gather(bp, -1, idx)

        # 4. AR(1) jitter around the selected base, clamped to [0, cap] / wrapped phase.
        e_mag = ar1_noise((b, n_dev, n_ord, t), config.ar1_rho, gen)
        e_phase = ar1_noise((b, n_dev, n_ord, t), config.ar1_rho, gen)
        mag = (sel_mag * (1.0 + config.jitter_mag * e_mag)).clamp(min=0.0)
        if clamp_to_cap:
            mag = torch.minimum(mag, caps)  # caps' trailing 1 broadcasts over T
        if config.fingerprint_silent_orders:
            silent = {int(o) for o in config.fingerprint_silent_orders}
            keep = torch.tensor(
                [0.0 if int(o) in silent else 1.0 for o in config.orders], dtype=_F64
            ).reshape(n_ord, 1)
            mag = mag * keep  # [B, n_dev, n_ord, T]; the draws above are unchanged
        phase = sel_phase + config.jitter_phase_deg * e_phase  # degrees

        # 5. assemble harmonic_injection {id: {order: (mag[B,T], phase[B,T])}}.
        for d, cid in enumerate(fp_ids):
            harmonic_injection[cid] = {
                order: (mag[:, d, o, :], phase[:, d, o, :])
                for o, order in enumerate(config.orders)
            }

        samples.update(
            {
                f"{nm}_mode": path,  # [B, n_dev, T]
                f"{nm}_mag": mag,  # [B, n_dev, n_ord, T]
                f"{nm}_phase": phase,  # [B, n_dev, n_ord, T]
                f"{nm}_mode_base_mag": base_mag,  # fingerprints
                f"{nm}_device_ids": torch.tensor(fp_ids, dtype=torch.long),
            }
        )

    # Optional per-scenario fundamental operating point (load / PV / slack-voltage specs).
    # Drawn on a separate stream (above) so the harmonic fingerprint is untouched.
    operating_point, op_samples, op_shared = _sample_operating_specs(grid, config, b)
    samples.update(op_samples)

    # Optional TIME-VARYING fundamental profile: lift the per-scenario operating point
    # to a per-step [B, T] one (seasonal / weekly / daily / short-term, class-aware).
    # Drawn on yet another distinct stream, so the harmonic fingerprint + the raw
    # parameter draws stay byte-identical to a run without a profile. The harmonic
    # injection magnitude is RELATIVE to each device's fundamental current, so a
    # profile-scaled fundamental already scales the absolute harmonic current at the
    # solve — no extra coupling here.
    if config.profile is not None:
        operating_point, profile_samples = apply_load_profiles(
            grid, config, operating_point
        )
        samples.update(profile_samples)

    # Statistical device composition: OVERRIDE the composed loads' fundamental (a per-step
    # [B, T] aggregate) and harmonic injection (the summed member spectrum). Drawn on
    # streams distinct from the fingerprint / op-cube / profile. The mixed [B]/[B, T]
    # operating point is unified to [B, T] so every device shares one leading batch.
    if comp is not None:
        draw = sample_device_composition(grid, config)
        if draw.operating_point:
            operating_point = broadcast_operating_point(operating_point, b, t)
            operating_point.update(draw.operating_point)
            harmonic_injection.update(draw.harmonic_injection)
            samples.update(draw.samples)
        if "time_unix_s" not in samples:
            # A composed sequence is anchored to absolute time (activity presets are
            # clock-driven); stamp the step axis like the profile path does, so the
            # dataset carries its own time axis instead of consumers re-deriving it.
            samples["time_unix_s"] = sequence_time_axis(config)

    node_sources = (
        build_background_sources(
            grid,
            config.background,
            (b, t),
            torch.Generator().manual_seed(config.seed + 8117),
        )
        if config.background is not None
        else []
    )
    shared = _split_shared(samples, nm, config.resample_modes_per_scenario)
    shared.update(op_shared)
    return SampledScenarios(
        operating_point=operating_point,
        samples=samples,
        n_samples=b,
        config=config,
        harmonic_injection=harmonic_injection,
        node_sources=node_sources,
        n_steps=t,
        shared_samples=shared,
    )


__all__ = ["sample_coherent_spectra"]
