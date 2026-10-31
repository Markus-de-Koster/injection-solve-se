"""Grid layout pieces shared by the Mack adapters (no frame needed).

The bus order of the published data (the grid's node order), the meter set, the
per-phase frame rotation, the positive-sequence admittance that gives the PANN its pruning
pattern, and the hop eccentricity that sets its depth.
"""

from __future__ import annotations

from collections import deque

import numpy as np

PHASES = ("a", "b", "c")
#: Admittance standing in for a closed switch (zero impedance) in the pruning pattern [S].
SWITCH_ADMITTANCE_S = 1e6 * (1.0 - 1.0j)


def bus_names(grid) -> list:
    """Bus names in the grid's node order (the published data's bus order)."""
    return [str(n.name) for n in sorted(grid.nodes, key=lambda n: int(n.id))]


def meter_buses(grid, placement_buses: list, slack: str) -> np.ndarray:
    """int64 ``[M]`` ascending bus indices of a placement's meters plus the slack bus."""
    names = bus_names(grid)
    return np.array(sorted({names.index(b) for b in list(placement_buses) + [slack]}), dtype=np.int64)


def admittance_per_order(grid, orders, v_ref: float) -> np.ndarray:
    """complex128 ``[H, B, B]`` positive-sequence bus admittance per order, referred to ``v_ref``.

    Lines: ``z1(h) = (R_s - R_m) + j h w (L_s - L_m)`` per metre from the 3x3 phase matrices
    (self minus mutual), shunt capacitance added as ``j h w (C_s - C_m) / 2`` at each end.
    Transformers: their series impedance at order ``h`` on the secondary side. Closed
    switches: :data:`SWITCH_ADMITTANCE_S`. Every branch admittance in siemens is referred
    to the line-to-neutral level ``v_ref`` by ``(V_base(from bus) / v_ref)^2`` (the
    published matrices use the same referral, checked against pandapower's line R1-R2).
    The matrix is used for its pattern (every bus, every entry with a branch, all four
    complex channels non-zero), not for its values.
    """
    names = bus_names(grid)
    idx = {int(n.id): names.index(str(n.name)) for n in grid.nodes}
    rated = {int(n.id): float(n.u_rated_v) / np.sqrt(3.0) for n in grid.nodes}
    f0 = float(grid.base_frequency_hz)
    out = np.zeros((len(orders), len(names), len(names)), dtype=np.complex128)
    for k, h in enumerate(orders):
        w = 2.0 * np.pi * f0 * h
        for br in grid.branches:
            if not br.in_service or (br.component == "switch" and not br.default_closed):
                continue
            i, j = idx[int(br.from_node)], idx[int(br.to_node)]
            shunt = 0.0
            if br.component == "line":
                r = np.asarray(br.series_resistance_ohm_per_m)
                ll = np.asarray(br.series_inductance_h_per_m)
                c = np.asarray(br.shunt_capacitance_f_per_m)
                z1 = (r[0, 0] - r[0, 1]) + 1j * w * (ll[0, 0] - ll[0, 1])
                y = 1.0 / (z1 * br.length_m)
                shunt = 0.5j * w * (c[0, 0] - c[0, 1]) * br.length_m
            elif br.component == "transformer":
                y = 1.0 / (br.series_resistance_ohm + 1j * w * br.series_inductance_h)
            elif br.component == "switch":
                y = SWITCH_ADMITTANCE_S
            else:
                raise ValueError(f"unknown branch component {br.component!r}.")
            side = int(br.to_node) if br.component == "transformer" else int(br.from_node)
            refer = (rated[side] / v_ref) ** 2
            y, shunt = y * refer, shunt * refer
            out[k, i, j] -= y
            out[k, j, i] -= y
            out[k, i, i] += y + shunt
            out[k, j, j] += y + shunt
    return out


def hop_eccentricity(adjacency: np.ndarray, sources: np.ndarray) -> int:
    """Largest number of hops from the nearest source to any bus (breadth-first search)."""
    n = adjacency.shape[0]
    dist = np.full(n, -1)
    queue = deque()
    for s in sources:
        dist[s] = 0
        queue.append(int(s))
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(adjacency[u]):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(int(v))
    if (dist < 0).any():
        raise ValueError("the grid is not connected to the meters.")
    return int(dist.max())


def phase_rotation(orders) -> np.ndarray:
    """complex128 ``[3, H]``: ``exp(+j·2π/3·p·h)`` per phase ``p`` and order ``h``."""
    return np.exp(2j * np.pi / 3 * np.outer(np.arange(3), np.asarray(orders, dtype=float)))


def index_rows(grid) -> np.ndarray:
    """int64 ``[B, 3]``: the row of each bus and phase in pgml's node-phase index order."""
    from pgml.assembly import node_phase_index

    idx = node_phase_index(grid)
    node_ids = idx.node_ids.cpu().numpy().astype(int)
    phase = idx.phase_codes.cpu().numpy().astype(int)
    order = [int(n.id) for n in sorted(grid.nodes, key=lambda n: int(n.id))]
    rows = np.full((len(order), 3), -1, dtype=np.int64)
    pos = {nid: b for b, nid in enumerate(order)}
    for r, (nid, p) in enumerate(zip(node_ids, phase)):
        rows[pos[nid], p] = r
    if (rows < 0).any():
        raise ValueError("every bus needs three phase rows.")
    return rows
