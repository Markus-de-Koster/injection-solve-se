"""PGL's finite full-row representation of schema-level ideal switches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from pgml.schemas.grid_schema import Switch

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid


#: Finite series resistance used by PGL for a schema-level ideal switch. PGL keeps both
#: switch terminals in its state and half-edge layouts, so it needs a primitive branch stamp;
#: pgml's exact-zero representation instead fuses those rows and has no such primitive.
ESTIMATOR_IDEAL_SWITCH_OHM = 1.0e-4


def materialize_estimator_grid(grid: "Grid") -> "Grid":
    """Return PGL's full-row physics view without mutating ``grid``.

    An exactly ideal :class:`pgml.schemas.grid_schema.Switch` (zero longitudinal R and L)
    is represented by a 100 micro-ohm series resistance. This is below PGL's electrical-tie
    threshold while retaining the finite primitive stamp required by its explicit switch
    half-edges and switchable ``branch_states``. Every nonzero user impedance and every
    non-switch branch is preserved exactly. pgml's exact fusion of an ideal switch reduces
    the row layout instead, and recovering the fused switch's current exactly would need the
    full nodal injections, which a persisted dataset does not carry.

    Tensor-backed switch parameters remain tensors on their original device/dtype. Adding
    the masked constant keeps the identity gradient through ``resistance_ohm``; no caller
    object is modified.
    """
    branches = []
    changed = False
    for branch in grid.branches:
        if not isinstance(branch, Switch):
            branches.append(branch)
            continue

        resistance = branch.resistance_ohm
        inductance = branch.inductance_h
        if isinstance(resistance, Tensor) or isinstance(inductance, Tensor):
            reference = resistance if isinstance(resistance, Tensor) else inductance
            dtype = reference.dtype if reference.is_floating_point() else torch.float64
            device = reference.device
            r = torch.as_tensor(resistance, dtype=dtype, device=device)
            inductance_tensor = torch.as_tensor(inductance, dtype=dtype, device=device)
            ideal = (r == 0) & (inductance_tensor == 0)
            epsilon = torch.as_tensor(
                ESTIMATOR_IDEAL_SWITCH_OHM, dtype=dtype, device=device
            )
            effective_r = r + ideal.to(dtype=dtype) * epsilon
            branches.append(branch.model_copy(update={"resistance_ohm": effective_r}))
            changed = True
        elif float(resistance) == 0.0 and float(inductance) == 0.0:
            branches.append(
                branch.model_copy(update={"resistance_ohm": ESTIMATOR_IDEAL_SWITCH_OHM})
            )
            changed = True
        else:
            branches.append(branch)

    return grid.model_copy(update={"branches": branches}) if changed else grid


__all__ = ["ESTIMATOR_IDEAL_SWITCH_OHM", "materialize_estimator_grid"]
