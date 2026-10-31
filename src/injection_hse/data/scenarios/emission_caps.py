"""IEC 61000-3-2 emission caps at every harmonic order: the standard's envelope, made total.

Every drawn device harmonic ratio in this package is capped at the device's IEC 61000-3-2
emission fraction (:func:`pgml.scenarios.iec61000_3_2_fraction`). pgml's table lists the
orders the standard tabulates explicitly and returns ``0.0`` for an order its class table
does not list, which makes a referenced device SILENT there: the packaged Class A table
stops at h39 (odd) and h10 (even), Class D lists no even order, Class C only h2. A
harmonic analyser resolves every integer order to h50 (IEC 61000-4-7), so a recipe that
solves those orders needs a cap at each of them.

:func:`iec61000_3_2_cap` returns pgml's value wherever the class table lists the order
(identical numbers, the same table override) and otherwise the standard's own envelope:

- Class A (and B = 1.5 x A), odd orders: ``0.15 * 15 / h`` A, the standard's formula for
  15 <= h <= 39, continued above h39 where the standard sets no limit.
- Class A (and B), even orders: ``0.23 * 8 / h`` A, the standard's formula for
  8 <= h <= 40 (pgml tabulates it to h10 only), continued above h40.
- Class C, odd orders: 3 % of the fundamental, the standard's flat limit for
  11 <= h <= 39, continued above h39.
- Class D, odd orders: ``3.85 / h`` mA/W, the standard's formula for 13 <= h <= 39,
  continued above h39.
- Classes C and D at an even order (the standard limits none but C's h2): the Class A
  limit of that order, the standard's general class.

Above h40 the continuation is an extrapolation of the standard's ``1/h`` laws, not a
limit the standard states. Like pgml's fraction, a cap is clamped to ``<= 1``.
"""

from __future__ import annotations

from typing import Optional

from pgml.assembly import phase_voltage_magnitude
from pgml.errors import InputError
from pgml.scenarios import (
    iec61000_3_2_fraction,
    iec61000_3_2_limits,
    resolve_emission_class,
)
from pgml.schemas.grid_schema import Grid

#: Highest order IEC 61000-3-2 sets a limit for; above it every cap is extrapolated.
IEC61000_3_2_MAX_ORDER = 40

#: Class A odd-order envelope numerator [A]: ``0.15 * 15 / h`` (15 <= h <= 39).
_CLASS_A_ODD_A = 0.15 * 15.0
#: Class A even-order envelope numerator [A]: ``0.23 * 8 / h`` (8 <= h <= 40).
_CLASS_A_EVEN_A = 0.23 * 8.0
#: Class B limits are the Class A limits times this factor.
_CLASS_B_FACTOR = 1.5
#: Class C flat odd-order limit [fraction of the fundamental] (11 <= h <= 39).
_CLASS_C_ODD = 0.03
#: Class D odd-order envelope numerator [mA/W]: ``3.85 / h`` (13 <= h <= 39).
_CLASS_D_ODD_MA_PER_W = 3.85


def _envelope_amperes(order: int, cls: str) -> Optional[float]:
    """The Class A / B envelope of ``order`` in amperes, ``None`` for classes C and D."""
    if cls not in ("A", "B"):
        return None
    amps = (_CLASS_A_ODD_A if order % 2 else _CLASS_A_EVEN_A) / order
    return amps * (_CLASS_B_FACTOR if cls == "B" else 1.0)


def iec61000_3_2_cap(
    order: int,
    *,
    emission_class: str,
    p_w: float,
    u_ln_v: float,
    power_factor: float = 1.0,
) -> float:
    """IEC 61000-3-2 emission cap of one order as a fraction of the fundamental current.

    :func:`pgml.scenarios.iec61000_3_2_fraction` wherever the class table lists the
    order, else the standard's envelope (module docstring), so every order ``>= 2`` has a
    cap. The fundamental current is ``I1 = p_w / (u_ln_v * power_factor)`` per phase, as
    in pgml.

    Parameters
    ----------
    order:
        Harmonic order, ``>= 2``.
    emission_class:
        Concrete class ``"A"`` / ``"B"`` / ``"C"`` / ``"D"`` (case-insensitive).
    p_w:
        Device active power per phase [W].
    u_ln_v:
        Line-to-neutral voltage magnitude at the device terminal [V].
    power_factor:
        Circuit power factor; enters ``I1`` and the Class C h3 scaling.

    Returns
    -------
    float
        Cap in ``[0, 1]`` [pu of the device's fundamental current].
    """
    h = int(order)
    if h < 2:
        raise InputError(f"an emission cap needs a harmonic order >= 2, got {order}.")
    cls = str(emission_class).upper()
    if h in iec61000_3_2_limits(cls)["limits"]:
        return iec61000_3_2_fraction(
            h,
            emission_class=cls,
            p_w=p_w,
            u_ln_v=u_ln_v,
            power_factor=power_factor,
        )
    if h % 2 and cls == "C":
        return _CLASS_C_ODD
    if h % 2 and cls == "D":
        return min(_CLASS_D_ODD_MA_PER_W / h * u_ln_v * power_factor / 1000.0, 1.0)
    amps = _envelope_amperes(h, cls if cls in ("A", "B") else "A")
    i1 = p_w / (u_ln_v * power_factor) if u_ln_v * power_factor != 0.0 else 0.0
    if i1 <= 0.0:
        return 1.0  # no fundamental current: the ratio is unbounded, fully clamped
    return min(amps / i1, 1.0)


def iec61000_3_2_caps(
    grid: Grid,
    ids: list[int],
    orders: list[int],
    *,
    emission_class: str = "auto",
    power_factor: float = 1.0,
) -> dict[int, dict[int, float]]:
    """Per-device, per-order emission caps ``{device_id: {order: cap}}`` for a grid.

    :func:`pgml.scenarios.iec61000_3_2_device_caps` with :func:`iec61000_3_2_cap` in place
    of the table lookup: the same per-phase power (nameplate over phase count), terminal
    voltage and class resolution (``"auto"`` from the device's ``consumer_type``), and the
    same values at every order the class table lists. Plain floats, off the autograd tape.
    """
    nodes_by_id = {n.id: n for n in grid.nodes}
    appliances_by_id = {a.id: a for a in grid.appliances}
    caps: dict[int, dict[int, float]] = {}
    for cid in ids:
        dev = appliances_by_id[cid]
        node = nodes_by_id[dev.node]
        n_phase = len(dev.phases)
        p_total = float(dev.p_nom_w)
        p_phase = p_total / n_phase if n_phase else p_total
        u_ln = phase_voltage_magnitude(float(node.u_rated_v), len(node.phases))
        cls = (
            resolve_emission_class(getattr(dev, "consumer_type", None), p_total)
            if str(emission_class).lower() == "auto"
            else emission_class
        )
        caps[cid] = {
            int(order): iec61000_3_2_cap(
                order,
                emission_class=cls,
                p_w=p_phase,
                u_ln_v=u_ln,
                power_factor=power_factor,
            )
            for order in orders
        }
    return caps


__all__ = ["IEC61000_3_2_MAX_ORDER", "iec61000_3_2_cap", "iec61000_3_2_caps"]
