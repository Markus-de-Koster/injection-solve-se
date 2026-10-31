"""Shared fixtures: a single-phase 3-bus grid, a 3-phase 3-bus grid, and a
single-phase 2-bus grid with a stored harmonic spectrum."""

from __future__ import annotations

import math

import pytest

from pgml.schemas.grid_schema import (
    Grid,
    HarmonicComponent,
    Line,
    Load,
    LoadModel,
    Node,
    Phase,
    Source,
    SpectrumPoint,
    StaticSpectrum,
)

W = 2.0 * math.pi * 50.0

ABC = (Phase.A, Phase.B, Phase.C)


def _line(bid, a, b):
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=(Phase.A,),
        to_phases=(Phase.A,),
        length_m=1.0,
        series_resistance_ohm_per_m=[[0.5]],
        series_inductance_h_per_m=[[0.5 / W]],
        shunt_capacitance_f_per_m=[[0.0]],
    )


@pytest.fixture
def grid3() -> Grid:
    """3-bus radial: source@1, household load@2 (id 10), EV load@3 (id 11)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=3, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[_line(1, 1, 2), _line(2, 2, 3)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="household",
            ),
            Load(
                id=11,
                node=3,
                phases=(Phase.A,),
                p_nom_w=3000.0,
                q_nom_var=800.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="ev_charging",
            ),
        ],
    )


def _line3(bid, a, b):
    """A 3-phase (ABC) line with diagonal R/L and no shunt."""
    z = [[0.0] * 3 for _ in range(3)]
    r = [[0.5 if i == j else 0.0 for j in range(3)] for i in range(3)]
    x = [[0.5 / W if i == j else 0.0 for j in range(3)] for i in range(3)]
    return Line(
        id=bid,
        from_node=a,
        to_node=b,
        from_phases=ABC,
        to_phases=ABC,
        length_m=1.0,
        series_resistance_ohm_per_m=r,
        series_inductance_h_per_m=x,
        shunt_capacitance_f_per_m=z,
    )


@pytest.fixture
def grid_3ph() -> Grid:
    """2-load 3-phase (ABC) feeder: source@1, two 3-phase loads (ids 20, 21)."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=400.0, phases=ABC),
            Node(id=2, u_rated_v=400.0, phases=ABC),
            Node(id=3, u_rated_v=400.0, phases=ABC),
        ],
        branches=[_line3(1, 1, 2), _line3(2, 2, 3)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=ABC,
                u_ref_v=(400.0, 400.0, 400.0),
                u_angle_deg=(0.0, -120.0, 120.0),
                resistance_ohm=[
                    [0.1 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
                inductance_h=[
                    [0.1 / W if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            ),
            Load(
                id=20,
                node=2,
                phases=ABC,
                p_nom_w=3000.0,
                q_nom_var=900.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="pv",
            ),
            Load(
                id=21,
                node=3,
                phases=ABC,
                p_nom_w=6000.0,
                q_nom_var=1500.0,
                load_model=LoadModel.CONST_POWER,
                consumer_type="pv",
            ),
        ],
    )


@pytest.fixture
def grid_spectrum() -> Grid:
    """Single-phase 2-bus grid whose load (id 10) carries a stored StaticSpectrum."""
    return Grid(
        base_frequency_hz=50.0,
        nodes=[
            Node(id=1, u_rated_v=230.0, phases=(Phase.A,)),
            Node(id=2, u_rated_v=230.0, phases=(Phase.A,)),
        ],
        branches=[_line(1, 1, 2)],
        appliances=[
            Source(
                id=1,
                node=1,
                phases=(Phase.A,),
                u_ref_v=(230.0,),
                u_angle_deg=(0.0,),
                resistance_ohm=[[0.1]],
                inductance_h=[[0.1 / W]],
            ),
            Load(
                id=10,
                node=2,
                phases=(Phase.A,),
                p_nom_w=2000.0,
                q_nom_var=500.0,
                load_model=LoadModel.CONST_POWER,
                spectrum=StaticSpectrum(
                    spectrum=SpectrumPoint(
                        components=[
                            HarmonicComponent(
                                order=5, magnitude_pu=0.1, phase_deg=10.0
                            ),
                            HarmonicComponent(
                                order=7, magnitude_pu=0.05, phase_deg=-20.0
                            ),
                        ]
                    )
                ),
            ),
        ],
    )
