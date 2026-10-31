"""Measurement model + the per-unit base voltage (sensor set -> node-phase rows).

:class:`MeasurementModel` resolves the sensor set (node names/ids, or the grid's installed
:class:`~pgml.schemas.grid_schema.MeasurementDevice` list via :meth:`MeasurementModel.from_grid`)
to observed node-phase rows of the ``NodePhaseIndex`` (the basis of the ``placement`` mask).
The metered slack is one of the measurements — its rows are part of ``observed_rows`` and are
additionally always-observed (exempt from random masking; see ``SEModule``).
:func:`base_voltage_per_row` is the per-row line-to-neutral reference the metrics divide by
for per-unit.

:func:`central_placement` is the sensor set for a grid nobody named sensors for: the slack
plus per-feeder central nodes. A multi-grid run needs a placement RULE rather than a list of
node names, since its grids are not known when the experiment is configured.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import Tensor

from pgml.assembly import base_voltage_per_row as _pgml_base_voltage_per_row
from pgml.assembly import node_phase_index
from pgml.topology import slack_node_ids

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid


# =============================================================================
# Sensor -> node-phase rows
# =============================================================================
def _resolve_node_id(grid: "Grid", key: Union[int, str]) -> int:
    """Resolve a node id OR node name to a node id (raises if ambiguous/missing)."""
    if isinstance(key, int):
        if any(n.id == key for n in grid.nodes):
            return key
        raise ValueError(f"no node with id {key!r} in the grid.")
    matches = [n.id for n in grid.nodes if n.name == key]
    if not matches:
        raise ValueError(
            f"no node named {key!r} in the grid "
            f"(available: {[n.name for n in grid.nodes][:8]}...)."
        )
    if len(matches) > 1:
        raise ValueError(f"node name {key!r} is ambiguous (ids {matches}).")
    return matches[0]


class MeasurementModel:
    """The physical sensor set -> observed node-phase rows (+ order capability + edges).

    ``voltage_nodes`` (names or ids) carry a FULL-HARMONIC measurement device (voltage
    observed at every solved order); ``fundamental_nodes`` carry a FUNDAMENTAL-ONLY
    monitor (smart-meter class — observed at ``h=1`` only; see :meth:`order_capability`).
    ``slack_known`` marks the slack bus(es) — the MV/LV substation(s) — as metered:
    the slack measurement is an ORDINARY (full-harmonic) measurement (its voltage is
    a model input, part of ``observed_rows``) that is additionally REQUIRED by the
    physics-informed decode, which reads the fundamental slack voltage as the
    network boundary. Every slack node (all in-service ``Source`` nodes,
    ``pgml.topology.slack_node_ids``) is covered. Resolution uses
    ``pgml.assembly.node_phase_index``.
    """

    def __init__(
        self, voltage_nodes, *, fundamental_nodes=(), slack_known: bool = True
    ) -> None:
        self.voltage_nodes = list(voltage_nodes)
        self.fundamental_nodes = list(fundamental_nodes)
        self.slack_known = slack_known
        # per-sensor-node harmonic ceiling (None = every solved order). Config-driven
        # sensors are binary (full vs fundamental-only); ``from_grid`` fills device-specific
        # ceilings from ``MeasurementDevice.max_harmonic_order``.
        self._node_max_order: dict = {key: None for key in self.voltage_nodes}
        self._node_max_order.update({key: 1 for key in self.fundamental_nodes})
        # device-based instrumentation (``from_grid``): edge observability comes from the
        # devices' CT channels instead of the all-incident default.
        self._device_based = False
        # the node subset a RESTRICTED device model was built over (``from_grid(nodes=...)``);
        # ``None`` = every installed device belongs to the model.
        self._device_nodes = None

    @classmethod
    def from_grid(cls, grid: "Grid", *, nodes=None) -> "MeasurementModel":
        """Build the sensor set from the grid's installed measurement devices.

        Sensor nodes are the nodes of every in-service
        :class:`~pgml.schemas.grid_schema.MeasurementDevice` that measures
        voltage; a device with ``max_harmonic_order`` set caps the orders its rows
        observe (``max_harmonic_order=1`` is a fundamental-only smart meter). The
        devices' CT channels additionally define the metered branches
        (:meth:`observed_edges`). ``slack_known`` is ``True`` iff EVERY slack node
        carries a voltage device (a partially metered multi-slack grid cannot
        provide the full network boundary). The intended path for grids with
        attached instrumentation (``Grid.attach_measurement_devices``);
        config-driven experiments keep passing ``voltage_nodes`` explicitly.

        ``nodes`` restricts the model to the devices installed AT those nodes: the
        placement of an estimator that uses a subset of the instrumentation as its
        inputs, with the remaining meters left as independent validation channels
        (their readings never enter the model). Devices elsewhere contribute neither
        voltage rows nor CT channels.
        """
        allowed = None if nodes is None else {int(n) for n in nodes}
        ceiling: dict[int, object] = {}
        for d in grid.measurement_devices:
            if not d.in_service or "voltage" not in d.measured_quantities:
                continue
            node = int(d.node)
            if allowed is not None and node not in allowed:
                continue
            cap = d.max_harmonic_order
            # two devices at one node: the more capable one wins (None = unlimited)
            if node in ceiling and (
                ceiling[node] is None or (cap is not None and cap < ceiling[node])
            ):
                continue
            ceiling[node] = cap
        full = sorted(n for n, cap in ceiling.items() if cap is None or cap > 1)
        fundamental = sorted(n for n, cap in ceiling.items() if cap == 1)
        slack_covered = set(slack_node_ids(grid)) <= set(ceiling)
        model = cls(full, fundamental_nodes=fundamental, slack_known=slack_covered)
        model._node_max_order = dict(ceiling)
        model._device_based = True
        model._device_nodes = allowed
        return model

    @classmethod
    def resolve(
        cls,
        grid: "Grid",
        voltage_nodes,
        *,
        fundamental_nodes=(),
        slack_known: bool = True,
    ) -> "MeasurementModel":
        """The placement a CONFIGURED sensor set denotes on ``grid`` — device-based when it can.

        A config names the sensor NODES; the grid's installed
        :class:`~pgml.schemas.grid_schema.MeasurementDevice` entries additionally carry each
        meter's current channels (which incident branch its CT is on) and its harmonic
        ceiling. When both describe the same observed rows, the device-based model
        (:meth:`from_grid`) is the truer one — a real installation meters ONE branch per
        meter, not every branch it touches — so it is what training, evaluation and
        deployment must all resolve to. Falls back to the configured sensor set when the
        grid carries no devices or its meters sit elsewhere.

        The configured nodes may also be a strict SUBSET of the installed meters — an
        estimator that uses part of the instrumentation as its inputs while the remaining
        meters serve as independent validation channels. The device model is then built over
        the devices at the configured nodes only (their CT channels, their harmonic
        ceilings); the other instruments contribute nothing. Only when the configured set
        matches no arrangement of the installed devices does the plain configured model
        (all-incident edges) apply.

        This is the SINGLE resolution rule: a placement resolved differently at deployment
        than at training feeds the model an edge set it never saw.
        """
        configured = cls(
            voltage_nodes, fundamental_nodes=fundamental_nodes, slack_known=slack_known
        )
        if not grid.measurement_devices:
            return configured
        device_based = cls.from_grid(grid)
        if torch.equal(
            device_based.observed_rows(grid), configured.observed_rows(grid)
        ):
            return device_based
        restricted = cls.from_grid(
            grid,
            nodes=[
                _resolve_node_id(grid, key)
                for key in (*configured.voltage_nodes, *configured.fundamental_nodes)
            ],
        )
        if restricted.voltage_nodes or restricted.fundamental_nodes:
            if torch.equal(
                restricted.observed_rows(grid), configured.observed_rows(grid)
            ):
                return restricted
        return configured

    def observed_rows(self, grid: "Grid") -> Tensor:
        """int64 ``[K]`` node-phase row indices observed by the sensor set.

        Every phase of each sensor node is observed (a measurement device reads all
        phases at its bus); fundamental-only sensor rows are INCLUDED (they are
        observed locations — their per-order coverage is :meth:`order_capability`).
        When ``slack_known``, the slack node rows are included too — the metered
        slack is a measurement like any other, so its (harmonic) voltage is part of
        the model input. Returns a sorted, de-duplicated int64 CPU tensor (the
        placement mask basis).
        """
        idx = node_phase_index(grid)
        rows: list[int] = []
        for key in (*self.voltage_nodes, *self.fundamental_nodes):
            node_id = _resolve_node_id(grid, key)
            rows.extend(idx.rows(node_id))
        rows.extend(self.slack_rows(grid).tolist())
        return torch.tensor(sorted(set(rows)), dtype=torch.int64)

    def order_capability(self, grid: "Grid", orders) -> Optional[Tensor]:
        """Per-row order coverage ``[N, H]`` bool (``None`` for a homogeneous fleet).

        Row ``i`` is ``True`` at order column ``h`` iff the sensor at that row resolves the
        order (``order <= max_harmonic_order``; full-harmonic sensors and the metered slack
        cover every order). Rows carrying no sensor are ``True`` everywhere — irrelevant
        under the placement mask (they are unobserved), and the neutral value for the
        random-masking stages. Returns ``None`` when every sensor is full-harmonic, so the
        homogeneous path stays byte-identical.
        """
        if all(cap is None for cap in self._node_max_order.values()):
            return None
        idx = node_phase_index(grid)
        orders_t = torch.tensor([int(o) for o in orders], dtype=torch.int64)
        cap = torch.ones((int(idx.size), orders_t.numel()), dtype=torch.bool)
        for key, max_order in self._node_max_order.items():
            if max_order is None:
                continue
            node_id = _resolve_node_id(grid, key)
            rows = torch.tensor(idx.rows(node_id), dtype=torch.int64)
            cap[rows] = (orders_t <= int(max_order)).unsqueeze(0)
        # the metered slack is always a full-harmonic measurement
        slack = self.slack_rows(grid)
        if slack.numel():
            cap[slack] = True
        return cap

    def observed_edges(self, grid: "Grid") -> Optional[Tensor]:
        """int64 ``[K]`` observed half-edge indices, or ``None`` for the all-incident default.

        ``None`` (config-driven sensor sets) means every measurement location meters ALL its
        incident branches — the edge observability then derives from the node mask by
        gathering at each half-edge's source row. For DEVICE-BASED instrumentation
        (:meth:`from_grid`) the installed CT channels are authoritative: a half-edge is
        observed iff an in-service current-measuring device sits at its source row's node
        and its channel list (empty = all incident) covers the branch and the row's phase.
        Half-edge indices follow the :func:`injection_hse.data.build_graph` layout.
        """
        if not self._device_based:
            return None
        from pgml.schemas.grid_schema import MeasuredQuantity, Phase

        from .currents import _half_edge_slots

        idx = node_phase_index(grid)
        slots, src_rows = _half_edge_slots(grid, idx)
        phase_of_code = {0: Phase.A, 1: Phase.B, 2: Phase.C, 3: Phase.N}
        node_ids = idx.node_ids.tolist()
        phase_codes = idx.phase_codes.tolist()

        by_node: dict[int, list] = {}
        for d in grid.measurement_devices:
            if (
                not d.in_service
                or MeasuredQuantity.CURRENT not in d.measured_quantities
            ):
                continue
            # A restricted model owns only the devices at its own nodes; a CT on one of the
            # OTHER instruments is a validation channel, not part of the placement.
            if self._device_nodes is not None and int(d.node) not in self._device_nodes:
                continue
            by_node.setdefault(int(d.node), []).append(d)

        observed: list[int] = []
        for e, ((branch_id, _term, _k), src) in enumerate(
            zip(slots, src_rows.tolist())
        ):
            node = int(node_ids[src])
            phase = phase_of_code[int(phase_codes[src])]
            for device in by_node.get(node, ()):
                if device.phases is not None and phase not in device.phases:
                    continue
                channels = device.current_channels
                if not channels:  # no explicit CT list -> all incident branches
                    observed.append(e)
                    break
                if any(
                    int(ch.branch) == branch_id
                    and (ch.phases is None or phase in ch.phases)
                    for ch in channels
                ):
                    observed.append(e)
                    break
        return torch.tensor(sorted(set(observed)), dtype=torch.int64)

    def slack_rows(self, grid: "Grid") -> Tensor:
        """int64 ``[K]`` node-phase rows of the metered slack node(s).

        Empty when ``slack_known`` is ``False``. These rows are always observed
        (never randomly masked — the substation meter does not disappear) and the
        injection decode reads its fundamental network boundary from them. Covers
        every in-service ``Source`` node (multi-slack grids included).
        """
        if not self.slack_known:
            return torch.zeros(0, dtype=torch.int64)
        idx = node_phase_index(grid)
        rows: list[int] = []
        for node_id in slack_node_ids(grid):
            rows.extend(idx.rows(node_id))
        return torch.tensor(sorted(set(rows)), dtype=torch.int64)

    def known_boundary_rows(self, grid: "Grid") -> Tensor:
        """Deprecated alias of :meth:`slack_rows` (the boundary IS the slack measurement)."""
        return self.slack_rows(grid)


# =============================================================================
# Grid-agnostic placement rule (slack + per-feeder central nodes)
# =============================================================================
def central_placement(
    grid: "Grid", fraction: float = 0.1, *, min_per_feeder: int = 1
) -> Tensor:
    """A structural sensor placement for a grid with no installed meters, as node-phase rows.

    A multi-grid run cannot name its sensor nodes per grid the way a single-grid config does,
    so it needs a RULE that applies to any topology. This is that rule: meter the slack (the
    substation, which is always metered in practice and is the boundary the injection decode
    reads) plus a budget of nodes spread over the feeders, each picked for CENTRALITY inside
    its own feeder — minimum eccentricity, so the chosen node's worst-case hop distance to
    the rest of its feeder is as small as possible, and a further pick maximizes its distance
    to the already chosen ones. A central node sees both the upstream flow and the downstream
    branches, which is where a meter is worth most when only a few exist.

    Parameters
    ----------
    grid:
        The grid (feeders are the components of the in-service branch graph with the slack
        node(s) removed; open switches are excluded).
    fraction:
        Share of the NON-SLACK nodes to meter (rounded up), distributed over the feeders in
        proportion to their size.
    min_per_feeder:
        Minimum number of metered nodes per feeder (``1`` = every feeder gets a meter, even
        when the fraction alone would not pay for one; ``0`` = strict budget).

    Returns
    -------
    torch.Tensor
        Sorted, de-duplicated int64 ``[K]`` node-phase ROW indices (every phase of each
        selected node plus every slack row), directly usable as
        ``PlacementMasker(observed_rows=...)`` or as a
        :class:`MeasurementModel` sensor set.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("central_placement fraction must be in (0, 1].")
    from .encodings import _feeder_adjacency, _feeder_partition, _select_central_nodes

    slack, feeders = _feeder_partition(grid)
    adjacency = _feeder_adjacency(grid)
    n_free = sum(len(members) for members in feeders)
    budget = max(1, math.ceil(fraction * n_free)) if n_free else 0

    # proportional allocation with a per-feeder floor; largest remainder breaks ties
    # deterministically (by feeder order) so the placement is reproducible.
    quotas = [
        (len(members) * budget / n_free if n_free else 0.0) for members in feeders
    ]
    counts = [
        min(len(members), max(min_per_feeder, int(quota)))
        for members, quota in zip(feeders, quotas)
    ]
    remainders = sorted(
        range(len(feeders)),
        key=lambda i: (-(quotas[i] - int(quotas[i])), i),
    )
    cursor = 0
    while sum(counts) < budget and cursor < len(remainders) * 4:
        i = remainders[cursor % len(remainders)] if remainders else 0
        if counts[i] < len(feeders[i]):
            counts[i] += 1
        cursor += 1

    idx = node_phase_index(grid)
    rows: list[int] = []
    for node_id in slack:
        rows.extend(idx.rows(node_id))
    for members, count in zip(feeders, counts):
        for node_id in _select_central_nodes(members, adjacency, count):
            rows.extend(idx.rows(node_id))
    return torch.tensor(sorted(set(int(r) for r in rows)), dtype=torch.int64)


# =============================================================================
# Per-row base voltage (the per-unit reference for the metrics)
# =============================================================================
def base_voltage_per_row(grid: "Grid", *, device=None) -> Tensor:
    """Per-row line-to-neutral base voltage ``[N]`` (the per-unit reference), row-aligned.

    Thin pass-through to the public :func:`pgml.assembly.base_voltage_per_row` so the
    per-unit convention is single-sourced in ``pgml`` (no duplication of the
    ``u_rated / sqrt(3)`` rule). Row order matches ``node_phase_index(grid)``, so the
    returned ``base_v`` divides a state/residual ``[..., N]`` directly into per-unit; real
    ``float64`` on ``device`` (a fixed constant, not on the autograd tape).
    """
    return _pgml_base_voltage_per_row(grid, device=device)


__all__ = ["MeasurementModel", "base_voltage_per_row", "central_placement"]
