"""Network-model error: generate data on perturbed copies of a grid.

A state estimator decodes with the grid description it is given; a real network differs
from that description (cable lengths off by some percent, a resistance from a data sheet, a
transformer impedance from its type plate, a supply stiffer or weaker than assumed). This
module produces training or test data on PERTURBED copies of a grid while the estimator
keeps the nominal one, so the effect of that mismatch can be measured.

pgml solves one network per call (a per-scenario batch of branch parameters is not
assembled), so a perturbation is a set of ``K`` deep-copied grids. The scenario batch is
drawn ONCE on the nominal grid — every copy solves the same kind of operating points, and a
perturbed dataset carries exactly the operating points of the nominal dataset with the same
seed — and split into ``K`` contiguous blocks, block ``k`` solved on copy ``k``. The result
concatenates the blocks in copy order and records each scenario's copy index in the
per-scenario sample record :data:`GRID_COPY_RECORD`.

Copies are drawn from a named POOL (``"train"`` / ``"test"``), each with its own seed stream,
so a test set can be generated on perturbation draws no training copy shares.

Perturbation kinds (:data:`PERTURBATION_KINDS`), each a relative factor per element:

- ``line_r``: the series resistance of every line — the resistance matrix of an explicit
  R/L/C line, the DC resistance of every conductor of a line with conductor geometry;
- ``line_x``: the series reactance — the inductance matrix of an explicit line; on a
  geometry line the phase conductors' GMR is moved so the fundamental positive-sequence
  reactance (the self reactance of a single-phase line) scales by the factor, exact for the
  ``gmr`` internal-inductance model;
- ``line_length``: the length, which scales the line's R, L and C together;
- ``transformer_r`` / ``transformer_x``: a transformer's series resistance / leakage
  inductance;
- ``source_impedance``: the Thevenin resistance and inductance matrices of every source
  (the short-circuit power scales by the inverse factor).

An element's factor is the product of a factor SHARED by every element of its kind (a
systematic error) and an independent per-element factor, each drawn with its own spread
(:class:`PerturbationSpread`) from a log-normal (default, ``exp(spread·z)``) or a uniform
(``1 + spread·u``, ``u ~ U(-1, 1)``) distribution.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid
    from pgml.scenarios import SampledScenarios, ScenarioResult

#: The perturbation kinds, in the order their streams are seeded and recorded.
PERTURBATION_KINDS = (
    "line_r",
    "line_x",
    "line_length",
    "transformer_r",
    "transformer_x",
    "source_impedance",
)

#: The per-scenario sample record holding each scenario's copy index (int64 ``[B]``).
GRID_COPY_RECORD = "grid_copy"

#: Seed-stream word of each copy pool: the pools draw disjoint perturbations.
PERTURBATION_POOLS = {"train": 0, "test": 1}

_MU0 = 4e-7 * math.pi


class PerturbationSpread(BaseModel):
    """Spreads of one perturbation kind: the shared factor and the per-element factor.

    ``shared`` is the spread of the ONE factor every element of the kind is multiplied by
    (a systematic error of the whole description), ``element`` that of each element's
    independent factor. Log-normal: the std of ``log(factor)``; uniform: the half-width of
    ``factor - 1``. ``0`` leaves that part at ``1``.
    """

    model_config = ConfigDict(extra="forbid")

    shared: float = Field(default=0.0, ge=0.0)
    element: float = Field(default=0.0, ge=0.0)

    @property
    def active(self) -> bool:
        """Whether this kind is perturbed at all."""
        return self.shared > 0.0 or self.element > 0.0


class GridPerturbationConfig(BaseModel):
    """How the generation grid is perturbed (see the module docstring).

    ``n_copies`` is the number of perturbed copies a pool generates (``K``); every copy
    solves an equal contiguous share of the scenario batch. ``n_test_copies`` sizes the
    ``"test"`` pool instead when set. ``seed`` seeds every draw; a copy's factors depend only
    on ``(seed, pool, copy index, kind)``, so changing one kind's spread leaves the other
    kinds' factors unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    n_copies: int = Field(default=1, ge=1)
    n_test_copies: Optional[int] = Field(default=None, ge=1)
    distribution: Literal["lognormal", "uniform"] = "lognormal"
    seed: int = 0
    line_r: PerturbationSpread = Field(default_factory=PerturbationSpread)
    line_x: PerturbationSpread = Field(default_factory=PerturbationSpread)
    line_length: PerturbationSpread = Field(default_factory=PerturbationSpread)
    transformer_r: PerturbationSpread = Field(default_factory=PerturbationSpread)
    transformer_x: PerturbationSpread = Field(default_factory=PerturbationSpread)
    source_impedance: PerturbationSpread = Field(default_factory=PerturbationSpread)

    @model_validator(mode="after")
    def _uniform_keeps_factors_positive(self) -> "GridPerturbationConfig":
        if self.distribution == "uniform":
            wide = [
                kind
                for kind in PERTURBATION_KINDS
                if max(getattr(self, kind).shared, getattr(self, kind).element) >= 1.0
            ]
            if wide:
                raise ValueError(
                    f"uniform spreads must be < 1 so every factor stays positive; {wide} "
                    "reach 1."
                )
        return self

    def spread(self, kind: str) -> PerturbationSpread:
        """The spreads of one kind of :data:`PERTURBATION_KINDS`."""
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind {kind!r}.")
        return getattr(self, kind)

    def pool_size(self, pool: str) -> int:
        """The number of copies ``pool`` generates."""
        _check_pool(pool)
        if pool == "test" and self.n_test_copies is not None:
            return int(self.n_test_copies)
        return int(self.n_copies)


@dataclass(frozen=True)
class GridPerturbation:
    """One perturbed copy of a grid and the factors that made it.

    ``factors`` maps each perturbed kind to ``{element id: factor}`` (only kinds with a
    non-zero spread appear), ``shared`` maps it to the factor all its elements share.
    ``grid`` is the perturbed deep copy; the nominal grid is never modified.
    """

    pool: str
    index: int
    factors: dict
    shared: dict
    grid: "Grid"

    @property
    def magnitude(self) -> float:
        """RMS of ``log(factor)`` over every perturbed element (``0`` for an unperturbed copy).

        One number per copy that orders the copies by how far their network is from the
        nominal description, the axis per-perturbation metrics are binned on.
        """
        logs = [
            math.log(f) for per_kind in self.factors.values() for f in per_kind.values()
        ]
        if not logs:
            return 0.0
        return math.sqrt(sum(x * x for x in logs) / len(logs))

    def record(self) -> dict:
        """JSON-ready record of the copy (``meta.json``): pool, index, magnitude, factors."""
        return {
            "pool": self.pool,
            "index": self.index,
            "magnitude_rms_log": self.magnitude,
            "shared": dict(self.shared),
            "factors": {
                kind: {str(eid): f for eid, f in per_kind.items()}
                for kind, per_kind in self.factors.items()
            },
        }


def _check_pool(pool: str) -> None:
    if pool not in PERTURBATION_POOLS:
        raise ValueError(
            f"pool must be one of {sorted(PERTURBATION_POOLS)}, got {pool!r}."
        )


def _elements(grid: "Grid", kind: str) -> list:
    """The elements a kind perturbs, sorted by id (the order of their draws)."""
    from pgml.schemas.grid_schema import Line, Source, Transformer

    if kind.startswith("line_"):
        items = [b for b in grid.branches if isinstance(b, Line)]
    elif kind.startswith("transformer_"):
        items = [b for b in grid.branches if isinstance(b, Transformer)]
    else:
        items = [a for a in grid.appliances if isinstance(a, Source)]
    return sorted(items, key=lambda e: int(e.id))


def _draw(rng: np.random.Generator, spread: float, size: int, distribution: str):
    """``size`` factors of one part (shared or element) of a kind."""
    if distribution == "lognormal":
        return np.exp(spread * rng.standard_normal(size))
    return 1.0 + spread * rng.uniform(-1.0, 1.0, size)


def draw_grid_perturbations(
    grid: "Grid", config: GridPerturbationConfig, *, pool: str = "train"
) -> list[GridPerturbation]:
    """Draw the copies of ``pool`` and build each perturbed grid.

    Returns :meth:`GridPerturbationConfig.pool_size` copies, ``index`` ``0..K-1``. Each copy
    is a deep copy of ``grid`` with every perturbed kind's elements scaled
    (:func:`perturb_grid`); ``grid`` itself is not modified. Deterministic in
    ``(config, pool)``. A kind whose elements the grid lacks is skipped.
    """
    _check_pool(pool)
    copies = []
    for index in range(config.pool_size(pool)):
        factors: dict = {}
        shared: dict = {}
        for k, kind in enumerate(PERTURBATION_KINDS):
            spread = config.spread(kind)
            elements = _elements(grid, kind)
            if not spread.active or not elements:
                continue
            seq = np.random.SeedSequence(
                [int(config.seed), PERTURBATION_POOLS[pool], index, k]
            )
            rng = np.random.default_rng(seq)
            common = float(_draw(rng, spread.shared, 1, config.distribution)[0])
            own = _draw(rng, spread.element, len(elements), config.distribution)
            shared[kind] = common
            factors[kind] = {
                int(e.id): common * float(f) for e, f in zip(elements, own)
            }
        copies.append(
            GridPerturbation(
                pool=pool,
                index=index,
                factors=factors,
                shared=shared,
                grid=perturb_grid(grid, factors),
            )
        )
    return copies


def _scaled(value, factor: float):
    """``value * factor`` for a float, a tensor or a nested list of floats."""
    if isinstance(value, (list, tuple)):
        return type(value)(_scaled(v, factor) for v in value)
    return value * factor


def _require(element, field: str):
    value = getattr(element, field)
    if value is None:
        raise ValueError(
            f"{type(element).__name__} {element.id} has no {field}; resolve its type "
            "before perturbing it."
        )
    return value


def _geometry_reference_reactance(line, f0: float) -> float:
    """Fundamental positive-sequence (single-phase: self) reactance [ohm/m] of a geometry line."""
    from pgml.geometry import line_constants

    geo = line.conductor_geometry
    phase = [
        next(c for c in geo.conductors if not c.is_neutral and c.phase == ph)
        for ph in line.from_phases
    ]
    ordered = phase + [c for c in geo.conductors if c.is_neutral]

    def column(name):
        return torch.tensor(
            [float(getattr(c, name)) for c in ordered], dtype=torch.float64
        )

    z, _ = line_constants(
        column("x_m"),
        column("y_m"),
        column("gmr_m"),
        column("r_dc_ohm_per_m"),
        column("radius_m"),
        torch.tensor(float(geo.earth_resistivity_ohm_m), dtype=torch.float64),
        torch.tensor([f0], dtype=torch.float64),
        len(line.from_phases),
        internal_inductance=geo.internal_inductance,
    )
    x = z[0].imag  # [P, P]
    p = x.shape[-1]
    self_x = x.diagonal().mean()
    if p == 1:
        return float(self_x)
    mutual = (x.sum() - x.diagonal().sum()) / (p * (p - 1))
    return float(self_x - mutual)


def _perturb_line(line, kind: str, factor: float, f0: float) -> None:
    geo = line.conductor_geometry
    if kind == "line_length":
        line.length_m = _scaled(line.length_m, factor)
    elif kind == "line_r":
        if geo is not None:
            for c in geo.conductors:
                c.r_dc_ohm_per_m = _scaled(c.r_dc_ohm_per_m, factor)
        else:
            line.series_resistance_ohm_per_m = _scaled(
                _require(line, "series_resistance_ohm_per_m"), factor
            )
    elif kind == "line_x":
        if geo is not None:
            # every phase conductor's self reactance moves by -mu0*f0*log(gmr'/gmr); the
            # mutual terms (and so a Kron-reduced neutral) are unchanged, so this shift is
            # the change of the positive-sequence reactance.
            x_ref = _geometry_reference_reactance(line, f0)
            shift = math.exp(-(factor - 1.0) * x_ref / (_MU0 * f0))
            for c in geo.conductors:
                if not c.is_neutral:
                    c.gmr_m = _scaled(c.gmr_m, shift)
        else:
            line.series_inductance_h_per_m = _scaled(
                _require(line, "series_inductance_h_per_m"), factor
            )


def perturb_grid(grid: "Grid", factors: dict) -> "Grid":
    """A deep copy of ``grid`` with each element scaled by its factor.

    ``factors`` maps a kind of :data:`PERTURBATION_KINDS` to ``{element id: factor}`` (the
    :attr:`GridPerturbation.factors` layout); kinds and elements it does not name are left
    nominal. Raises ``ValueError`` for an element whose perturbed parameter is unresolved
    (a ``type_ref`` line or transformer without stored values).
    """
    out = copy.deepcopy(grid)
    f0 = float(out.base_frequency_hz)
    by_id = {int(b.id): b for b in out.branches}
    by_id.update({int(a.id): a for a in out.appliances})
    for kind, per_element in factors.items():
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind {kind!r}.")
        for eid, factor in per_element.items():
            element = by_id[int(eid)]
            factor = float(factor)
            if kind.startswith("line_"):
                _perturb_line(element, kind, factor, f0)
            elif kind == "transformer_r":
                element.series_resistance_ohm = _scaled(
                    _require(element, "series_resistance_ohm"), factor
                )
            elif kind == "transformer_x":
                element.series_inductance_h = _scaled(
                    _require(element, "series_inductance_h"), factor
                )
            else:  # source_impedance
                element.resistance_ohm = _scaled(element.resistance_ohm, factor)
                element.inductance_h = _scaled(element.inductance_h, factor)
    return out


def _rows(value, start: int, end: int, b: int):
    """Scenarios ``[start:end]`` of a batched value; unbatched values pass through."""
    if isinstance(value, (list, tuple)):
        return type(value)(_rows(v, start, end, b) for v in value)
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == b:
        return value[start:end]
    return value


def _slice_sampled(sampled: "SampledScenarios", start: int, end: int):
    """The solver inputs of scenarios ``[start:end]`` as a batch of their own."""
    from pgml.scenarios import SampledScenarios

    b = int(sampled.n_samples)
    return SampledScenarios(
        operating_point={
            cid: {key: _rows(v, start, end, b) for key, v in entry.items()}
            for cid, entry in sampled.operating_point.items()
        },
        samples={},
        n_samples=end - start,
        config=sampled.config,
        harmonic_injection={
            cid: {
                order: tuple(_rows(c, start, end, b) for c in pair)
                for order, pair in per_order.items()
            }
            for cid, per_order in sampled.harmonic_injection.items()
        },
        node_sources=[
            replace(
                src,
                spectrum={
                    order: tuple(_rows(c, start, end, b) for c in pair)
                    for order, pair in src.spectrum.items()
                },
                source_power_va=_rows(src.source_power_va, start, end, b),
            )
            for src in sampled.node_sources
        ],
        n_steps=sampled.n_steps,
    )


def copy_blocks(n_scenarios: int, n_copies: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` scenario blocks of the copies, sizes differing by at most 1.

    The first ``n_scenarios % n_copies`` copies take one scenario more. Raises
    ``ValueError`` when there are fewer scenarios than copies.
    """
    if n_copies < 1 or n_scenarios < n_copies:
        raise ValueError(
            f"{n_scenarios} scenario(s) cannot be shared by {n_copies} grid copies; each "
            "copy solves at least one."
        )
    base, extra = divmod(int(n_scenarios), int(n_copies))
    blocks, start = [], 0
    for k in range(n_copies):
        end = start + base + (1 if k < extra else 0)
        blocks.append((start, end))
        start = end
    return blocks


def run_perturbed_scenarios(
    grid: "Grid", spec, perturbations: list[GridPerturbation], **kwargs
) -> "ScenarioResult":
    """Solve one scenario batch across perturbed copies of ``grid`` (module docstring).

    Parameters
    ----------
    grid:
        The NOMINAL grid; the batch is drawn on it (through the same ideal-switch
        materialisation :func:`injection_hse.data.scenarios.run_scenarios` applies), so the operating
        points equal those of a nominal dataset with the same spec.
    spec:
        A scenario spec (anything with ``sample(grid)``) or a pre-built
        :class:`pgml.scenarios.SampledScenarios`.
    perturbations:
        The copies (:func:`draw_grid_perturbations`); copy ``k`` solves block ``k`` of
        :func:`copy_blocks`.
    **kwargs:
        Forwarded to :func:`injection_hse.data.scenarios.run_scenarios` for every block
        (``calculation``, ``harmonic_orders``, ``load_shunt``, ``chunk_size``, ...).

    Returns
    -------
    pgml.scenarios.ScenarioResult
        ``v`` concatenated over the blocks in copy order (``[B, ...]``, on each block's
        output device), the full sampled batch with the int64 ``[B]`` record
        :data:`GRID_COPY_RECORD` added, and the failed scenarios at their batch positions.
    """
    from pgml.scenarios import SampledScenarios, ScenarioResult

    from injection_hse._switches import materialize_estimator_grid

    from . import run_scenarios

    if not perturbations:
        raise ValueError("run_perturbed_scenarios needs at least one grid copy.")
    if isinstance(spec, SampledScenarios):
        sampled = spec
    else:
        sampled = spec.sample(materialize_estimator_grid(grid))
        # the blocks are solved as pre-built batches, which carry no order hint: apply the
        # spec's own the way pgml.scenarios.run_scenarios would
        hint = getattr(spec, "harmonic_orders", None)
        if hint:
            kwargs["calculation"] = "harmonic"
            if kwargs.get("harmonic_orders") is None:
                kwargs["harmonic_orders"] = list(hint)
    if kwargs.get("calculation", "power_flow") == "power_flow":
        batched_ndim = 2
    else:
        batched_ndim = 4 if int(sampled.n_steps) > 1 else 3
    blocks = copy_blocks(int(sampled.n_samples), len(perturbations))
    parts, failed = [], []
    converged = True
    copy_index = torch.empty(int(sampled.n_samples), dtype=torch.int64)
    result = None
    for copy_, (start, end) in zip(perturbations, blocks):
        result = run_scenarios(
            copy_.grid, _slice_sampled(sampled, start, end), **kwargs
        )
        v = result.v
        if v.ndim == batched_ndim - 1:  # a one-scenario block loses its scenario axis
            v = v.unsqueeze(0)
        parts.append(v)
        converged = converged and bool(result.converged)
        failed.extend(start + int(i) for i in result.failed_states)
        copy_index[start:end] = int(copy_.index)
    record = replace(sampled, samples={**sampled.samples, GRID_COPY_RECORD: copy_index})
    return ScenarioResult(
        v=torch.cat(parts, dim=0),
        index=result.index,
        sampled=record,
        frequencies_hz=result.frequencies_hz,
        converged=converged,
        failed_states=tuple(failed),
    )


def perturbation_provenance(
    config: GridPerturbationConfig,
    perturbations: list[GridPerturbation],
    n_scenarios: int,
) -> dict:
    """What a perturbed dataset records in ``meta.json`` (``extra_provenance``).

    The perturbation config, the pool, each copy's factors (:meth:`GridPerturbation.record`)
    and the scenario block each copy solved.
    """
    blocks = copy_blocks(int(n_scenarios), len(perturbations))
    return {
        "config": config.model_dump(mode="json"),
        "pool": perturbations[0].pool if perturbations else None,
        "record": GRID_COPY_RECORD,
        "copies": [
            {**p.record(), "scenarios": [start, end]}
            for p, (start, end) in zip(perturbations, blocks)
        ],
    }


__all__ = [
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
