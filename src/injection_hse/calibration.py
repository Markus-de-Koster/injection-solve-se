"""Learned grid-parameter calibration (Stage 1): nudge a few grid parameters to fit the data.

On a REAL grid the description is imperfect — a line length off by some percent, a resistance
from a data sheet rather than the wet ground it actually runs through. Those errors enter the
admittance ``Y``, so an injection-estimation estimator that trusts ``Y`` is capped at the
grid's accuracy when it predicts voltages. Because ``Y(Θ)`` is a DIFFERENTIABLE function of the
grid parameters, the measurement residual can be back-propagated to a SMALL, shared set of
correction factors ``Δ`` and the grid nudged so the predictions fit the measurements again —
beating the nominal-``Y`` cap within the span of the chosen parameterisation.

Identifiability discipline (the only thing that keeps "too many free variables" in check):

- the corrections are LOW-dimensional and GLOBAL (one ``Δ`` shared across every scenario — the
  grid error is a property of the grid, not the operating point; scenario diversity is what
  disentangles it from the per-scenario injections);
- each knob carries a CONFIDENCE (a relative tolerance / std): a tight tolerance both penalises
  a correction harder (the prior) and down-weights it; ``tolerance=0`` FIXES the parameter (the
  user is sure it is correct), so it is not a free variable at all.

:class:`GridCalibration` holds the learnable corrections, applies them to the grid
differentiably (``field = nominal · (1 + Δ)``), and exposes the confidence-weighted regulariser
``Σ (Δ / tolerance)²`` that pulls unsupported corrections back toward 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid


@dataclass(frozen=True)
class CalibrationKnob:
    """One calibratable grid parameter + the user's confidence in its recorded value.

    ``target``/``id`` select the grid element — a branch (``"line"`` / ``"transformer"`` /
    ``"branch"``) or an appliance (``"source"`` / ``"load"`` / ``"generator"`` /
    ``"storage"``) by ``id``; any other target raises. ``field`` is the physical attribute
    to correct (e.g. ``"length_m"``, ``"series_resistance_ohm_per_m"``). ``tolerance`` is
    the RELATIVE confidence — ``0.2`` means "known to ±20%" (a soft prior std on the
    multiplicative correction); ``0.0`` FIXES the parameter (no free variable).
    """

    target: str
    id: int
    field: str
    tolerance: float = 0.1


class GridCalibration(torch.nn.Module):
    """Learnable, confidence-regularised corrections applied to a few grid parameters.

    Captures each knob's NOMINAL value (detached) and a scalar correction ``Δ`` (an
    ``nn.Parameter``, init 0; a fixed knob keeps ``Δ ≡ 0``). :meth:`apply_to_grid` writes
    ``field = nominal · (1 + Δ)`` back onto the grid IN PLACE so the next admittance assembly
    sees the corrected value on the live tape; :meth:`regularizer` is the confidence-weighted
    prior ``Σ (Δ / tolerance)²``. The grid the solver assembles from is the same object passed
    here, so a training step does ``apply_to_grid(); solver.assemble()`` before the solve.
    """

    def __init__(self, grid: "Grid", knobs: Sequence[CalibrationKnob]) -> None:
        super().__init__()
        self._grid = grid
        self.knobs = list(knobs)
        self._nominal: list[Tensor] = []
        self._free: list[bool] = []
        deltas = []
        for knob in self.knobs:
            elem = self._element(knob)
            nominal = torch.as_tensor(
                _to_array(getattr(elem, knob.field)), dtype=torch.float64
            )
            self._nominal.append(nominal)
            free = knob.tolerance > 0.0
            self._free.append(free)
            # Δ inherits the nominal's float64, NOT the ambient torch default:
            # apply_to_grid casts the nominal to Δ's dtype, so a float32 default
            # would silently cap the corrected parameter's precision below the
            # complex128 solve it feeds.
            deltas.append(
                torch.nn.Parameter(
                    torch.zeros((), dtype=nominal.dtype), requires_grad=free
                )
            )
        self.deltas = torch.nn.ParameterList(deltas)

    # -- grid access ----------------------------------------------------------
    _BRANCH_TARGETS = ("line", "transformer", "branch")
    _APPLIANCE_TARGETS = ("source", "load", "generator", "storage")

    def _element(self, knob: CalibrationKnob):
        if knob.target in self._BRANCH_TARGETS:
            pool = self._grid.branches
        elif knob.target in self._APPLIANCE_TARGETS:
            pool = self._grid.appliances
        else:
            raise ValueError(
                f"calibration knob target {knob.target!r} is not recognised "
                f"(expected one of {self._BRANCH_TARGETS + self._APPLIANCE_TARGETS})."
            )
        for e in pool:
            if int(e.id) == int(knob.id):
                return e
        raise ValueError(
            f"calibration knob {knob.target} id={knob.id} not found in grid."
        )

    def apply_to_grid(self) -> "Grid":
        """Write ``field = nominal · (1 + Δ)`` onto the grid (differentiable w.r.t. ``Δ``).

        The nominal snapshot follows ``Δ`` onto its device, so the assembly stays on the model's
        device on a GPU run.
        """
        for knob, nominal, delta in zip(self.knobs, self._nominal, self.deltas):
            elem = self._element(knob)
            value = nominal.to(device=delta.device, dtype=delta.dtype) * (1.0 + delta)
            setattr(elem, knob.field, value)
        return self._grid

    # -- regularizer ----------------------------------------------------------
    def regularizer(self) -> Tensor:
        """Confidence-weighted prior ``Σ (Δ / tolerance)²`` (0 when no free knob)."""
        dev = self.deltas[0].device if len(self.deltas) else torch.device("cpu")
        total = torch.zeros((), dtype=torch.float64, device=dev)
        for knob, free, delta in zip(self.knobs, self._free, self.deltas):
            if free:
                total = total + (delta.double() / knob.tolerance) ** 2
        return total

    def corrections(self) -> dict:
        """``{f"{target}:{id}:{field}": Δ}`` — the learned relative corrections (for reporting)."""
        return {
            f"{k.target}:{k.id}:{k.field}": float(d.detach())
            for k, d in zip(self.knobs, self.deltas)
        }


def _to_array(value):
    """Grid field -> a plain array/scalar for the nominal snapshot (detached)."""
    if isinstance(value, Tensor):
        return value.detach().cpu()
    return value


__all__ = ["CalibrationKnob", "GridCalibration"]
