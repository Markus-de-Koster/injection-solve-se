"""The realized-injection cluster: the per-device localization target.

:class:`InjectionSpec` (the per-device per-order injection TARGET for the optional localization
head) builds from the dataset's REALIZED ``harmonic_injection`` — reproduced deterministically
from the stored config + seed via :func:`_reconstruct_sampled` — so it unifies across the
snapshot (``ScenarioConfig``) and sequence (``CoherentSpectrumConfig``) datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from pgml.assembly import node_phase_index

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid


# =============================================================================
# Per-device injection target (the localization / "where did it originate" label)
# =============================================================================
def _reconstruct_sampled(grid: "Grid", config):
    """Deterministically reproduce a dataset's realized scenario draws from its config.

    Every scenario config is a ``pgml.scenarios.ScenarioSpec``, so redrawing the batch is
    the config's own ``sample(grid)``: the returned ``SampledScenarios`` carries the same
    ``operating_point`` (P/Q) and ``harmonic_injection`` overrides the dataset was solved
    with (reproducible from config + seed). The realized ``harmonic_injection`` is the
    source for the localization target (:class:`InjectionSpec`), uniform across the
    snapshot and sequence datasets.
    """
    from pgml.scenarios import SampledScenarios

    if isinstance(config, SampledScenarios):
        return config
    if not callable(getattr(config, "sample", None)):
        raise TypeError(
            f"cannot reproduce a dataset's draws from {type(config).__name__!r}: it is "
            "neither a SampledScenarios nor a scenario spec (an object with a "
            "sample(grid) method). A dataset whose config class is unknown to this "
            "package reads back as a dict — register it with "
            "injection_hse.data.scenarios.read_dataset(config_types=...)."
        )
    return config.sample(grid)


def _injection_leading_shape(harmonic_injection: dict) -> tuple[int, ...]:
    """The common leading (scenario[, step]) shape of a realized ``harmonic_injection``.

    Each ``(magnitude, phase)`` entry may be a python float / 0-d tensor (broadcast to every
    scenario) or a ``[B]`` / ``[B, T]`` tensor (per scenario / per step). The leading shape is
    the largest-rank leading shape seen across all entries (``()`` if every entry is scalar).
    """
    lead: tuple[int, ...] = ()
    for orders in harmonic_injection.values():
        for mag, _phase in orders.values():
            if isinstance(mag, Tensor) and mag.ndim > len(lead):
                lead = tuple(mag.shape)
    return lead


class InjectionSpec:
    """The per-device per-order injection grid + target, from the REALIZED harmonic injection.

    The localization target/input is **per-device, per-injected-order MAGNITUDE**, laid out
    ``[*B, (T,) D, O, 1]`` (the trailing channel ``C=1`` now; phase is a documented later
    channel). It is assembled from the dataset's REALIZED ``harmonic_injection`` — reproduced
    deterministically from the stored config + seed via :func:`_reconstruct_sampled` (the same
    dispatch ``run_scenarios`` uses) — so it works UNIFORMLY for both a ``ScenarioConfig``
    (snapshot, magnitudes ``[B]``) and a ``CoherentSpectrumConfig`` (sequence, magnitudes
    ``[B, T]``), and is correct on coherent data. The injection is
    ``{device_id: {order: (magnitude, phase)}}`` (order 1 = the implicit reference, never a
    target). This spec resolves, ONCE per (grid, config):

    - **device set ``D``** — the injecting appliances (loads + generators) that appear in the
      realized injection, in GRID ORDER (``grid.appliances`` iteration order, ``in_service``).
    - **order set ``O``** — the SORTED UNION of all injected orders ``h > 1``.
    - **device → node-phase rows** — each device injects at its terminal rows
      (``node_phase_index.rows_for_terminal(node, phases)``), used to POOL node features at the
      device for the head.
    - **assembly** — for each device/order, broadcast its magnitude to the common
      ``[*B, (T,)]`` leading shape and place it at ``[..., pos(device), pos(order)]``;
      device/order pairs the injection does not drive stay 0.
    """

    def __init__(self, grid: "Grid", config) -> None:
        from pgml.schemas.grid_schema import Generator, Load

        idx = node_phase_index(grid)
        sampled = _reconstruct_sampled(grid, config)
        self._injection: dict = sampled.harmonic_injection or {}

        # device set D: injecting appliances present in the realized injection, in GRID ORDER
        driven = set(self._injection.keys())
        device_ids: list[int] = [
            a.id
            for a in grid.appliances
            if isinstance(a, (Load, Generator)) and a.in_service and a.id in driven
        ]
        self.device_ids = device_ids
        self._device_pos = {d: j for j, d in enumerate(device_ids)}

        # order set O: sorted union of injected orders h>1 (order 1 is the implicit reference)
        orders: set[int] = set()
        for dev in device_ids:
            orders.update(int(o) for o in self._injection[dev] if int(o) > 1)
        self.orders = sorted(orders)
        self._order_pos = {o: k for k, o in enumerate(self.orders)}

        # device -> node-phase rows (for the head's pooling), padded to a [D, P_max] index
        # with a [D, P_max] validity mask (devices differ in phase count). A device injects
        # at its TERMINAL node's rows (appliance ``.node`` + ``.phases``).
        appliance_by_id = {a.id: a for a in grid.appliances}
        rows_per_dev = [
            idx.rows_for_terminal(
                appliance_by_id[dev].node, appliance_by_id[dev].phases
            ).tolist()
            for dev in device_ids
        ]
        p_max = max((len(r) for r in rows_per_dev), default=1)
        n_dev = len(device_ids)
        dev_rows = torch.zeros((n_dev, p_max), dtype=torch.int64)
        dev_rows_mask = torch.zeros((n_dev, p_max), dtype=torch.bool)
        for j, rows in enumerate(rows_per_dev):
            for k, r in enumerate(rows):
                dev_rows[j, k] = r
                dev_rows_mask[j, k] = True
        self.device_rows = (
            dev_rows  # [D, P_max] int64 (node-phase row of each device phase)
        )
        self.device_rows_mask = dev_rows_mask  # [D, P_max] bool (valid phase slots)

    @property
    def n_devices(self) -> int:
        return len(self.device_ids)

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    @property
    def is_empty(self) -> bool:
        """``True`` when the dataset carries no per-device harmonic-magnitude injection."""
        return self.n_devices == 0 or self.n_orders == 0

    def assemble_target(self) -> Tensor:
        """Build the per-device per-order magnitude target ``[*B, (T,) D, O, 1]`` (float32).

        Each device/order magnitude (broadcast from its realized value to the common leading
        shape) lands at ``[..., pos(device), pos(order)]``; device/order pairs the injection
        does not drive stay 0. The trailing channel axis is ``C=1`` (phase is a later channel).
        """
        lead = _injection_leading_shape(self._injection)
        block = torch.zeros((*lead, self.n_devices, self.n_orders), dtype=torch.float32)
        for dev in self.device_ids:
            j = self._device_pos[dev]
            for o, (mag, _phase) in self._injection[dev].items():
                o = int(o)
                if o not in self._order_pos:
                    continue  # order 1 (the reference) is not a target
                k = self._order_pos[o]
                m = mag if isinstance(mag, Tensor) else torch.as_tensor(mag)
                block[..., j, k] = m.to(
                    torch.float32
                )  # broadcasts a scalar to [*B,(T)]
        return block.unsqueeze(-1)  # [*B, (T,) D, O, 1]


__all__ = ["InjectionSpec"]
