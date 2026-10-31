"""A hand-built three-node single-phase grid for the injection_hse unit tests.

``Source -> n1 --line1--> n2 --line2--> n3`` with a shunt load at ``n3``, single phase
``(A,)`` so every per-phase matrix is 1x1 — small enough for the config / masking /
curriculum tests that need a real :class:`~pgml.schemas.grid_schema.Grid` without a
reference network. Materialised (no ``type_ref``); clean SI numbers.
"""

from __future__ import annotations

from pgml.schemas.grid_schema import (
    Grid,
    Line,
    Load,
    Node,
    Phase,
    Source,
)


def single_phase_chain() -> Grid:
    """Source -> n1 --line1--> n2 --line2--> n3, with a shunt load at n3.

    Single phase ``(A,)`` so every per-phase matrix is 1x1. Chosen with clean SI
    numbers; the test recomputes Y and V in numpy.
    """
    nodes = [
        Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
        Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        Node(id=3, u_rated_v=230.0, phases=(Phase.A,)),
    ]
    source = Source(
        id=10,
        node=1,
        phases=(Phase.A,),
        u_ref_v=(230.0,),
        u_angle_deg=(0.0,),
        resistance_ohm=[[0.1]],
        inductance_h=[[1.0e-3]],
    )
    line1 = Line(
        id=20,
        from_node=1,
        to_node=2,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=100.0,
        series_resistance_ohm_per_m=[[1.0e-3]],
        series_inductance_h_per_m=[[1.0e-6]],
        shunt_capacitance_f_per_m=[[1.0e-9]],
    )
    line2 = Line(
        id=21,
        from_node=2,
        to_node=3,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=50.0,
        series_resistance_ohm_per_m=[[2.0e-3]],
        series_inductance_h_per_m=[[1.5e-6]],
        shunt_capacitance_f_per_m=[[2.0e-9]],
    )
    load = Load(
        id=30,
        node=3,
        phases=(Phase.A,),
        p_nom_w=2000.0,
        q_nom_var=500.0,
    )
    return Grid(
        base_frequency_hz=50.0,
        nodes=nodes,
        branches=[line1, line2],
        appliances=[source, load],
    )
