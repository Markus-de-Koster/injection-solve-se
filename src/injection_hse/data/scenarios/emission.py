"""The load-dependent harmonic emission law and its place in the randomized recipe.

A proportional emission model states a device's harmonic current as a fixed ratio of its
fundamental, ``I_h = r_h * I_1``. The measured device population this package is
calibrated on (type-test tables of certified PV inverters and laboratory rack sweeps, see
``docs/pgl/harmonic-emission.md``) does not behave like that: part of the emission is
present whenever the device is on and does not scale with how hard it is driven. The law
that describes it is COMPLEX AFFINE in the loading::

    I_h(lam) = A_h + B_h * lam        lam = |I_1| / |I_1 at rating|

with ``A_h`` the load-independent floor and ``B_h`` the load-proportional part. The ratio
to the fundamental therefore grows as the device unloads, the emission ANGLE rotates with
loading (from ``arg A_h`` at low load toward ``arg B_h`` at rating), and ``|I_h|(lam)``
has an interior cancellation null where the two parts are near anti-phase. A separate,
explicit phase slope models the rotation a device shows beyond the affine geometry.

The functions express the law as a CORRECTION on the proportional emission the generators
draw, so the rated operating point is untouched and a zero floor reproduces the
proportional law bit-for-bit. Two generators apply the same functions:

- the randomized recipe, through :class:`EmissionLawScenarioConfig`. pgml draws the law's
  parameters as free per-device, per-order values (``ParameterSpec(field="h_param")``) in
  the batch's own cube; :func:`apply_emission_laws` folds them into the sampled injection
  against the loading each device's own power draw realised;
- the device composition (:mod:`injection_hse.data.scenarios.composition`), per roster member and
  step.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Optional

import torch
from pgml.scenarios import (
    ParameterSpec,
    SampledScenarios,
    ScenarioConfig,
    nominal_power,
)
from pgml.scenarios import sample as _sample_scenarios
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from torch import Tensor

from .discrete import DiscreteEmitterConfig, apply_discrete_emitters

__all__ = [
    "LOADING_FLOOR",
    "EMISSION_LAW_ROLES",
    "EmissionLaw",
    "EmissionLawScenarioConfig",
    "StoredScenarioConfig",
    "affine_emission_correction",
    "phase_slope_shift",
    "apply_emission_laws",
]

#: The loading below which the randomized recipe evaluates the affine law at this value
#: instead of the drawn loading. The correction is expressed as a RATIO to the device's
#: fundamental current, which the solver then multiplies by the actual ``|I_1|``; the ratio
#: grows as ``1 / lam`` and the fundamental shrinks as ``lam``, so the harmonic current
#: itself stays finite — but at ``lam = 0`` the product is ``0 * inf``. Below the floor the
#: harmonic current therefore falls linearly with the fundamental to zero (a device that
#: draws nothing emits nothing), instead of holding the floor current of a device that is
#: on but idle.
LOADING_FLOOR = 0.05

#: The law's parameters, in the order they are declared on an :class:`EmissionLaw`.
EMISSION_LAW_ROLES: tuple[str, ...] = ("floor", "floor_phase", "slope")

#: ``ParameterSpec.field`` values pgml 0.5.0 used for the law's parameters, by role. A
#: config written with them is migrated on load (see :class:`EmissionLawScenarioConfig`).
_LEGACY_FIELDS = {
    "h_floor": "floor",
    "h_floor_phase": "floor_phase",
    "h_slope": "slope",
}


def affine_emission_correction(lam: Tensor, floor: Tensor, delta_deg: Tensor) -> Tensor:
    """The affine emission law as a complex correction on the proportional phasor.

    Returns ``z(lam) / (lam * z(1))`` with ``z(lam) = floor * exp(j * delta) +
    (1 - floor) * lam``: multiplying the proportional harmonic phasor ``r_h * I_1(lam)`` by
    it gives the affine one, normalised so that the RATED (``lam = 1``) emission is
    unchanged for any floor. ``floor`` is ``|A_h|`` as a share of ``|A_h| + |B_h|`` and
    ``delta_deg`` is ``arg(A_h) - arg(B_h)``. At ``delta_deg = 0`` the magnitude is
    ``floor / lam + (1 - floor)``, so a floor of ``0.61`` inflates the ratio to the
    fundamental 6.5-fold at 10 % loading.

    At ``floor = 0`` the result is exactly ``1 + 0j`` (``lam / lam`` is ``1.0`` in IEEE
    arithmetic for any finite non-zero ``lam``), so a configuration that does not ask for the
    law is unaffected bit-for-bit. Broadcasts over any leading axes.
    """
    a = floor * torch.exp(1j * torch.deg2rad(delta_deg))
    z = a + (1.0 - floor) * lam
    z_rated = a + (1.0 - floor)
    return z / (lam.clamp(min=1e-9) * z_rated)


def phase_slope_shift(slope_deg: Tensor, lam: Tensor) -> Tensor:
    """The explicit loading-dependent phase shift ``s_h * (lam - 1)`` [deg].

    Zero at rating by construction, so the rated emission angle stays the drawn one; a
    negative slope rotates the phasor one way as the device unloads, a positive one the
    other way. Independent of the affine law's own rotation (which is set by the floor
    angle) and added to it.
    """
    return slope_deg * (lam - 1.0)


class EmissionLaw(BaseModel):
    """One load-dependent emission law, as the names of the specs that draw it.

    Each member names a ``field="h_param"`` :class:`~pgml.scenarios.ParameterSpec` of the
    same config, whose selector and ``orders`` decide which devices and orders the law
    covers and whose ``per`` decides how long a draw persists (per scenario, per device,
    per class). A member left ``None`` is inert: no floor is the proportional law, no
    floor phase puts the floor in phase with the proportional part, no slope adds no
    rotation.

    - ``floor``: the load-INDEPENDENT share ``|A_h| / (|A_h| + |B_h|)`` of the rated
      phasor, drawn from ``[0, 1]``. The drawn ``h_mag`` is read as the RATED ratio and
      inflated by ``|z(lam)| / lam`` as the device unloads
      (:func:`affine_emission_correction`).
    - ``floor_phase``: ``arg(A_h) - arg(B_h)`` in degrees, which makes the emission
      phase ROTATE with loading and produces the cancellation null. Inert without a floor.
    - ``slope``: the explicit phase slope ``s_h`` [deg per unit loading]; the phase is
      shifted by ``s_h * (lam - 1)`` (:func:`phase_slope_shift`).
    """

    model_config = ConfigDict(extra="forbid")

    floor: Optional[str] = None
    floor_phase: Optional[str] = None
    slope: Optional[str] = None

    def spec_names(self) -> dict[str, str]:
        """``{role: spec name}`` of the members that are set."""
        return {
            role: name
            for role in EMISSION_LAW_ROLES
            if (name := getattr(self, role)) is not None
        }


class EmissionLawScenarioConfig(ScenarioConfig):
    """A randomized batch whose device emission follows its own drawn loading.

    A :class:`pgml.scenarios.ScenarioConfig` plus ``emission_laws``. The batch is drawn by
    pgml exactly as the base config would draw it, with the law's parameters occupying
    their own columns of the cube as ``h_param`` specs. :meth:`sample` then folds each
    law into the sampled ``harmonic_injection`` (:func:`apply_emission_laws`). Draw it
    through ``config.sample(grid)`` or a ``run_scenarios`` front door;
    ``pgml.scenarios.sample(grid, config)`` returns the batch BEFORE the law.

    A config written for pgml 0.5.0, which declared the law's parameters with the
    ``ParameterSpec`` fields ``h_floor`` / ``h_floor_phase`` / ``h_slope``, validates
    unchanged: each such spec becomes an ``h_param`` spec and one :class:`EmissionLaw`
    naming it, which draws and applies the same values.

    ``discrete_emitters`` (default ``None``) adds large discrete devices to the loads
    after the laws (:class:`~injection_hse.data.scenarios.DiscreteEmitterConfig`,
    :func:`~injection_hse.data.scenarios.apply_discrete_emitters`). Unset, it is left out of the
    serialized config, so the config's hash and every draw are those of a config written
    before the field existed.
    """

    emission_laws: list[EmissionLaw] = Field(default_factory=list)
    discrete_emitters: Optional[DiscreteEmitterConfig] = None

    @model_serializer(mode="wrap")
    def _omit_unset_emitters(self, handler):
        data = handler(self)
        if isinstance(data, dict) and data.get("discrete_emitters") is None:
            data.pop("discrete_emitters", None)
        return data

    def injected_orders(self) -> list[int]:
        """The orders the config's ``h_mag`` / ``h_phase`` specs inject at (sorted)."""
        return sorted(
            {
                int(order)
                for spec in self.parameters
                if spec.field in ("h_mag", "h_phase") and spec.orders
                for order in spec.orders
            }
        )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data):
        if not isinstance(data, dict):
            return data
        parameters = data.get("parameters")
        if not isinstance(parameters, list):
            return data
        migrated, laws = [], []
        for spec in parameters:
            role = (
                _LEGACY_FIELDS.get(spec.get("field"))
                if isinstance(spec, dict)
                else None
            )
            if role is not None:
                spec = {**spec, "field": "h_param"}
                laws.append({role: spec.get("name")})
            migrated.append(spec)
        if not laws:
            return data
        return {
            **data,
            "parameters": migrated,
            "emission_laws": [*(data.get("emission_laws") or []), *laws],
        }

    @model_validator(mode="after")
    def _check_emission_laws(self) -> "EmissionLawScenarioConfig":
        by_name = {spec.name: spec for spec in self.parameters}
        for law in self.emission_laws:
            for role, name in law.spec_names().items():
                spec = by_name.get(name)
                if spec is None or not spec.is_free_parameter:
                    raise ValueError(
                        f"emission law {role}={name!r} must name a field='h_param' "
                        "parameter of this config."
                    )
                if role == "floor":
                    _check_floor_bounds(spec)
        if self.discrete_emitters is not None and not self.injected_orders():
            raise ValueError(
                "discrete_emitters inject at the orders the config's h_mag / h_phase "
                "specs cover; this config has none."
            )
        return self

    def with_parameters(self, parameters) -> "EmissionLawScenarioConfig":
        """A copy with another parameter list, its laws restricted to the specs kept.

        A law member whose spec is gone is cleared and a law left without a member is
        dropped, so pruning a template (to a grid's population, to the solved orders)
        leaves a config that still validates.
        """
        names = {spec.name for spec in parameters}
        laws = [
            EmissionLaw(**{r: n for r, n in law.spec_names().items() if n in names})
            for law in self.emission_laws
        ]
        return self.model_copy(
            update={
                "parameters": list(parameters),
                "emission_laws": [law for law in laws if law.spec_names()],
            }
        )

    def sample(self, grid) -> SampledScenarios:
        """Draw the batch, apply the emission laws, then add the discrete emitters.

        An IEC 61000-3-2 referenced ``h_mag`` draw at an order the standard's table does
        not list for the device's class is referenced to the standard's envelope there
        (:func:`injection_hse.data.scenarios.emission_caps.iec61000_3_2_cap`) instead of pgml's
        silent zero; at every tabulated order the batch is pgml's, unchanged.
        """
        sampled = _cap_unlisted_orders(grid, _sample_scenarios(grid, self), self)
        sampled = apply_emission_laws(grid, sampled, self)
        return apply_discrete_emitters(grid, sampled, self)


class StoredScenarioConfig:
    """Rebuilds a stored ``ScenarioConfig``, whichever release wrote it.

    A randomized dataset written with pgml 0.5.0 records ``config_type="ScenarioConfig"``
    and declares the law through ``h_floor`` / ``h_floor_phase`` / ``h_slope`` specs, which
    the engine's class no longer accepts. Handed to ``read_dataset(config_types=...)``
    under that name, this returns an :class:`EmissionLawScenarioConfig` for such a config
    and a plain :class:`pgml.scenarios.ScenarioConfig` for every other one.
    """

    @staticmethod
    def model_validate_json(text: str) -> ScenarioConfig:
        doc = json.loads(text)
        legacy = any(
            isinstance(spec, dict) and spec.get("field") in _LEGACY_FIELDS
            for spec in doc.get("parameters") or []
        )
        return (EmissionLawScenarioConfig if legacy else ScenarioConfig).model_validate(
            doc
        )


def _cap_unlisted_orders(
    grid, sampled: SampledScenarios, config: ScenarioConfig
) -> SampledScenarios:
    """Reference IEC-referenced draws at orders pgml's table leaves without a limit.

    pgml multiplies an ``harmonic_reference="iec61000-3-2"`` draw by the device's table
    limit, which is ``0`` at an order the class table does not list. Here the raw draw is
    multiplied by the envelope cap of :mod:`injection_hse.data.scenarios.emission_caps` instead, for
    exactly the (device, order) slots whose final writer is such a spec and whose cap
    differs from pgml's. Returns ``sampled`` itself when no slot qualifies.
    """
    from pgml.scenarios import iec61000_3_2_device_caps

    from .emission_caps import iec61000_3_2_caps

    writer: dict[tuple[int, int], tuple[int, int]] = {}
    for spec in config.parameters:  # declaration order: a later spec wins a shared slot
        if spec.field != "h_mag" or not spec.orders:
            continue
        ids_key = f"{spec.name}_device_ids"
        if ids_key not in sampled.shared_samples:
            continue
        for j, cid in enumerate(sampled.shared_samples[ids_key].tolist()):
            for order in spec.orders:
                writer[(int(cid), int(order))] = (id(spec), j)
    updates: dict[int, dict[int, tuple]] = {}
    for spec in config.parameters:
        if spec.field != "h_mag" or spec.harmonic_reference != "iec61000-3-2":
            continue
        ids = sampled.shared_samples[f"{spec.name}_device_ids"].tolist()
        table = iec61000_3_2_device_caps(
            grid, ids, spec.orders, emission_class=spec.emission_class
        )
        total = iec61000_3_2_caps(
            grid, ids, spec.orders, emission_class=spec.emission_class
        )
        draws = sampled.samples[spec.name]  # [B, n_eff, n_orders]
        one_draw = spec.per in ("shared", "class")
        for j, cid in enumerate(ids):
            for k, order in enumerate(spec.orders):
                cap = total[cid][order]
                if cap == table[cid][order] or writer.get((cid, order)) != (
                    id(spec),
                    j,
                ):
                    continue
                _, phase = sampled.harmonic_injection[cid][order]
                draw = draws[:, 0 if one_draw else j, k].to(torch.float64)
                updates.setdefault(cid, {})[order] = (draw * cap, phase)
    if not updates:
        return sampled
    injection = {cid: dict(per) for cid, per in sampled.harmonic_injection.items()}
    for cid, per_order in updates.items():
        injection[cid].update(per_order)
    return replace(sampled, harmonic_injection=injection)


def _check_floor_bounds(spec: ParameterSpec) -> None:
    bounds = [
        b
        for b in (
            getattr(spec.distribution, "low", None),
            getattr(spec.distribution, "high", None),
            getattr(spec.distribution, "value", None),
        )
        if b is not None
    ]
    if any(b < 0.0 or b > 1.0 for b in bounds):
        raise ValueError(
            f"emission law floor {spec.name!r} is the load-independent SHARE of the "
            "emission and must be drawn from [0, 1]."
        )


def _column(value, b: int, like: Tensor) -> Tensor:
    """One injection coefficient as a ``[B]`` real column in ``like``'s dtype.

    A coefficient is a per-scenario tensor (a spec drew it) or a scalar carried over from
    the device's stored spectrum, which broadcasts across the batch.
    """
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=like.dtype, device=like.device)
    return tensor.expand(b) if tensor.ndim == 0 else tensor


def _device_loading(cid: int, operating_point: dict, nominal: dict) -> Optional[Tensor]:
    """The loading ``lam [B]`` a device's operating-point draw realised, or ``None``.

    The ratio of the drawn active power to the nameplate: the balanced total for a
    scalar entry, the mean of the per-phase ratios for a per-phase one (each phase over
    its own nameplate share, so an unbalanced draw reads as its average loading).
    ``None`` when no power spec varied the device or its nameplate is zero — the law then
    reads the device as rated.
    """
    entry = operating_point.get(cid)
    rec = nominal.get(cid)
    if entry is None or rec is None:
        return None
    if "p_per_phase_w" in entry:
        ratios = [
            torch.as_tensor(p).abs() / abs(float(pn))
            for p, pn in zip(entry["p_per_phase_w"], rec.p_pp)
            if float(pn) != 0.0
        ]
        if not ratios:
            return None
        return torch.stack(ratios, dim=-1).mean(dim=-1)
    if "p_w" in entry and float(rec.p_total) != 0.0:
        return torch.as_tensor(entry["p_w"]).abs() / abs(float(rec.p_total))
    return None


def _drawn_laws(grid, config: EmissionLawScenarioConfig, samples: dict) -> dict:
    """``{device id: {order: {role: draw[B]}}}`` from the recorded ``h_param`` draws."""
    roles = {
        name: role
        for law in config.emission_laws
        for role, name in law.spec_names().items()
    }
    laws: dict[int, dict[int, dict[str, Tensor]]] = {}
    for spec in config.parameters:  # declaration order: a later spec wins a shared slot
        role = roles.get(spec.name)
        if role is None:
            continue
        draws = samples[spec.name]  # [B, n_eff, n_orders]
        one_draw = spec.per in ("shared", "class")
        for j, cid in enumerate(spec.selector.resolve(grid)):
            for k, order in enumerate(spec.orders):
                laws.setdefault(cid, {}).setdefault(order, {})[role] = draws[
                    :, 0 if one_draw else j, k
                ]
    return laws


def apply_emission_laws(
    grid, sampled: SampledScenarios, config: EmissionLawScenarioConfig
) -> SampledScenarios:
    """Fold the drawn emission laws into a sampled batch's harmonic injection.

    Per device the loading is what its own power draw realised (drawn active power over
    the nameplate, ``1`` for a device no power spec varies, floored at
    :data:`LOADING_FLOOR`); per order the magnitude is scaled by the affine law's ratio
    inflation and the phase shifted by the law's own rotation plus the explicit slope. A
    device without a law is untouched, and a zero floor and zero slope leave the values
    bit-identical.

    The realized-injection records of every ``h_mag`` spec (``<spec>_mag`` /
    ``<spec>_phase``) are rewritten to the post-law values the solver receives, and
    ``<spec>_loading`` ``[B, n_dev]`` records the loading the law read for each device
    (``1`` where no law covers it), so the magnitude-to-loading relation a dataset
    carries is auditable without re-running the recipe.

    Parameters
    ----------
    grid:
        The grid the batch was drawn for (resolves the law specs' selectors).
    sampled:
        The batch ``pgml.scenarios.sample(grid, config)`` returned.
    config:
        The config that declares the laws.

    Returns
    -------
    pgml.scenarios.SampledScenarios
        A new batch; ``sampled`` is not modified.
    """
    b = int(sampled.n_samples)
    laws = _drawn_laws(grid, config, sampled.samples)
    nominal = nominal_power(grid) if laws else {}
    injection = {cid: dict(per) for cid, per in sampled.harmonic_injection.items()}
    loading: dict[int, Tensor] = {}
    for cid, per_order in laws.items():
        lam = _device_loading(cid, sampled.operating_point, nominal)
        lam = (
            torch.ones(b, dtype=torch.float64)
            if lam is None
            else lam.to(torch.float64).reshape(-1).expand(b).clone()
        )
        lam_eff = lam.clamp(min=LOADING_FLOOR)
        loading[cid] = lam_eff
        dev = injection.setdefault(cid, {})
        zeros = torch.zeros(b, dtype=torch.float64)
        for order, params in per_order.items():
            mag0, ph0 = dev.get(order, (0.0, 0.0))
            floor = params.get("floor", zeros).to(torch.float64)
            delta = params.get("floor_phase", zeros).to(torch.float64)
            slope = params.get("slope", zeros).to(torch.float64)
            corr = affine_emission_correction(lam_eff, floor, delta)  # complex [B]
            mag = _column(mag0, b, zeros)
            ph = _column(ph0, b, zeros)
            dev[order] = (
                mag * corr.abs(),
                ph
                + torch.rad2deg(torch.angle(corr))
                + phase_slope_shift(slope, lam_eff),
            )

    # Re-record what was injected, in the key order the sampler wrote it, with the
    # loading directly behind each spec's phase record.
    h_mag = {spec.name: spec for spec in config.parameters if spec.field == "h_mag"}
    samples: dict = {}
    like = torch.zeros((), dtype=torch.float64)
    for key, value in sampled.samples.items():
        name = key[: -len("_phase")] if key.endswith("_phase") else None
        spec = h_mag.get(key[: -len("_mag")]) if key.endswith("_mag") else None
        if spec is not None and f"{spec.name}_device_ids" in sampled.shared_samples:
            ids = sampled.shared_samples[f"{spec.name}_device_ids"].tolist()
            samples[key] = _realized(injection, ids, spec.orders, 0, b, like)
            continue
        if name in h_mag and f"{name}_device_ids" in sampled.shared_samples:
            ids = sampled.shared_samples[f"{name}_device_ids"].tolist()
            samples[key] = _realized(injection, ids, h_mag[name].orders, 1, b, like)
            samples[f"{name}_loading"] = torch.stack(
                [_column(loading.get(cid, 1.0), b, like) for cid in ids], dim=1
            )
            continue
        samples[key] = value
    return replace(sampled, harmonic_injection=injection, samples=samples)


def _realized(
    injection: dict, ids: list, orders: list, part: int, b: int, like: Tensor
) -> Tensor:
    """``[B, n_dev, n_ord]`` magnitude (``part=0``) or phase (``part=1``) records."""
    return torch.stack(
        [
            torch.stack(
                [_column(injection[cid][o][part], b, like) for o in orders], dim=-1
            )
            for cid in ids
        ],
        dim=1,
    )
