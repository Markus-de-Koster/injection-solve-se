"""Heterogeneous-graph batching: many grids of different size in one batch.

The single-grid data path can keep one graph, one row layout and one ``[B, N]`` mask, because
every sample is the same network. A multi-grid run cannot: ``N`` differs per member, so a
batch of samples is a batch of DIFFERENT graphs. This module supplies the two pieces that
change:

- :class:`MultiGridDataSource` — one ``ParquetDataset`` per member grid of a
  :class:`~injection_hse.data.multigrid.MultiGridManifest`, concatenated and tagged with the grid each
  item came from, plus the per-grid artifacts a model needs (graph with its structural
  encoding, per-row base voltage, forced-observed slack rows, placement rows). The split is
  by GRID, taken from the manifest — validation and test measure held-out topologies.
- :class:`MultiGridCollate` — the DataLoader collate that folds the sample axis into the
  NODE axis (the torch-geometric convention: a batch of graphs is one disjoint graph),
  emitting ``ptr`` / ``batch`` index vectors and offset ``edge_index``.

The TWO TASKS of the scarcity curriculum are two SOURCES over one corpus. A member may carry a
randomized snapshot dataset and a coherent SEQUENCE dataset beside it; the
source's ``task`` selects which one it opens. Everything else — the member list, the grids and
their graphs, the placements, the grid-level split and the batch contract below — is identical,
so one module trains across both by running its curriculum once per source.

Batch layout (FROZEN — models and the training step are written against this)
-----------------------------------------------------------------------------
``B`` items, item ``b`` drawn from member grid ``g_b`` with ``N_b`` node-phase rows and
``E_b`` directed half-edges; ``N* = sum_b N_b``, ``E* = sum_b E_b``; ``H`` harmonic orders
(uniform across members, validated) and ``T`` steps (uniform, ``1`` for snapshot data).

=========================  =========================  ===============================
key                        shape / dtype              meaning
=========================  =========================  ===============================
``v``                      complex ``[H, N*]``        the state target; ``[T, H, N*]``
                                                      when the members are sequences
``batch``                  int64 ``[N*]``             item index of each node row (PyG)
``ptr``                    int64 ``[B+1]``            node offsets; item ``b`` owns
                                                      ``ptr[b]:ptr[b+1]``
``row_bus``                int64 ``[N*]``             physical-bus index of each row
``bus_ptr``                int64 ``[B+1]``            physical-bus offsets per item
``grid_index``             int64 ``[B]``              member-grid index per item (into
                                                      ``MultiGridDataSource.grids``)
``index``                  int64 ``[B]``              scenario id in the member dataset
``edge_index``             int64 ``[2, E*]``          half-edges, node offsets applied
``edge_batch``             int64 ``[E*]``             item index of each half-edge
``edge_ptr``               int64 ``[B+1]``            half-edge offsets
``node_static``            float32 ``[N*, Fns]``      per-row graph features
                                                      (base ⧺ encoding)
``edge_static``            float32 ``[E*, Fes]``      per-half-edge graph features
``edge_branch_ids``        int64 ``[E*]``             branch id behind each half-edge
``base_voltage``           float64 ``[N*]``           per-row line-to-neutral base
                                                      (the per-unit reference)
``slack_rows``             int64 ``[K]``              GLOBAL rows of the metered
                                                      slack(s)
``reference_rows``         int64 ``[N*]``             GLOBAL angle-reference row of
                                                      every row, in the source's
                                                      configured frame
``observed_rows``          int64 ``[K]``              GLOBAL rows of the fixed
                                                      placement (only when one is
                                                      resolved)
``node_mask``              bool ``[N*]``              observation mask (only when the
                                                      collate was given a masker)
``time_unix_s``            float64 ``[T]``            step time axis (time-stamped
                                                      members only)
=========================  =========================  ===============================

The node axis is the ONLY axis that varies: ``[T, H]`` stay uniform, so every per-order and
per-step operation of the single-grid pipeline carries over unchanged. Row ``i`` of ``v``,
``node_static``, ``base_voltage`` and mask column ``i`` all describe the same node-phase row
of the same item — the alignment invariant the single-grid path states for ``[B, N]``, now
over the concatenated axis. Per-scenario ``sample_*`` inputs are NOT collated (their per-grid
axes cannot stack).

The DEVICE axis (only with ``MultiGridCollate(..., with_injection=True)``)
--------------------------------------------------------------------------
The per-device injection target has its own per-grid axis, so it folds the same way the node
axis does: a batch of items becomes one concatenated DEVICE axis ``D* = sum_b D_b`` with its
own ``ptr`` / ``batch`` pair. ``O`` (the injected orders ``h > 1``) is uniform across members,
validated by :attr:`MultiGridDataSource.injection_orders`; a member whose dataset carries no
per-device injection contributes ``D_b = 0``.

=====================  ============================  ===============================
key                    shape / dtype                 meaning
=====================  ============================  ===============================
``injection_target``   float32 ``[(T,) D*, O, 1]``   per-device per-order injection
                                                     magnitude, in the PER-UNIT current
                                                     base (dimensionless)
``device_ptr``         int64 ``[B+1]``               device offsets; item ``b`` owns
                                                     ``device_ptr[b]:device_ptr[b+1]``
``device_batch``       int64 ``[D*]``                item index of each device
``device_rows``        int64 ``[D*, P_max]``         GLOBAL node rows of a device's
                                                     phases (pad slot ``-1``)
``device_rows_mask``   bool ``[D*, P_max]``          valid slots of ``device_rows``
=====================  ============================  ===============================

``device_rows`` are GLOBAL indices into the batched node axis (member-local rows plus the
item's ``ptr`` offset), so a head pools node features at ``device_rows`` exactly as in the
single-grid path. The pad slot is ``-1``, not ``0``, so a forgotten mask fails loudly instead
of silently pooling the batch's first row.

Per-unit currents (the magnitude decision on the DEVICE axis)
--------------------------------------------------------------
``injection_target`` is emitted divided by the member's per-unit CURRENT base at each device's
rows (:meth:`injection_hse.physics.MultiGridPhysicsBank.device_current_base`) — the current analogue of
:func:`per_unit_state` below. A localization target in amperes is not comparable across
voltage classes (the same relative harmonic emission is orders of magnitude apart between a
400 V and a 20 kV member), so what the batch carries is dimensionless. The PERSISTED dataset
is untouched: the scaling happens here, in the batch.

Per-unit first (the multi-grid normalization decision)
------------------------------------------------------
``v`` is persisted and collated in VOLTS, and the metrics stay in volts / per-unit exactly as
before. Features, however, enter the encoder in PER-UNIT (:func:`per_unit_state`, dividing by
``base_voltage``): a 400 V feeder and a 20 kV feeder differ by a factor 50 in volts and not at
all per-unit, so a single dataset-global :class:`~injection_hse.normalization.HarmonicNormalizer` stays
valid across voltage classes. Normalizing in volts across mixed classes would make the
per-harmonic statistics a mixture of two scales and cost the small harmonic orders their
visibility — precisely what the normalizer exists to protect.

The angle reference frame (the multi-grid ANGLE decision)
---------------------------------------------------------
Voltage MAGNITUDES are comparable across grids once they are per-unit; voltage ANGLES are
not. Every grid carries its own arbitrary global angle reference — the convention of the
source EMF it was solved against, and the vector-group clock a converter folded into it — so
two members can describe the same physical state with a constant per-grid angle offset. A
model trained on many grids then has to guess that offset from the topology, which it cannot,
and a grid whose offset was never seen is estimated with the right magnitude and the wrong
phase.

:func:`slack_referenced_state` removes it: every row is rotated by the FUNDAMENTAL angle of
its own grid's reference row, and order ``h`` rotates by ``exp(-1j*h*phi)`` — a shift of the
TIME reference, so the harmonic phasors stay mutually consistent (exactly what an instrument
locked to a fundamental PLL at the reference bus reports). Magnitudes are untouched, so
per-unit magnitude statistics are unchanged; what remains in the angle is structure the
estimator can explain.

WHICH reference row is the frame's choice, made by
:attr:`injection_hse.config.MultiGridConfig.angle_reference` and carried on the source:

- ``"slack"`` (default) — the same-phase row of the grid's slack (:func:`slack_reference_rows`).
  A transformer between the slack and a node then keeps its vector-group clock in the state.
- ``"lv_busbar"`` — the same-phase row of the grid's LOCAL reference bus, the
  lower-voltage-side node of the transformer nearest the slack (:func:`reference_bus_rows`).
  This is the frame a real LV measurement campaign works in (the reference meter sits at the
  substation's LV busbar, not on the MV side), and it is independent of how the transformer
  above the busbar is modelled. A member without an in-service transformer has no such
  busbar and falls back to the slack frame — the two frames coincide there.
- ``"absolute"`` — no rotation, the raw solver frame.

Whichever is configured, the batch carries ONE row table, ``reference_rows``, so every
consumer (the training step, the normalizer fits, the physics-decode boundary rotation) reads
the same frame.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

from .graph import build_graph
from .measurement import MeasurementModel, base_voltage_per_row, central_placement
from .multigrid import GridEntry, MultiGridManifest, read_grid, task_dataset_dir
from .sources import DataSource, ParquetDataset

if TYPE_CHECKING:  # heavy / optional imports kept out of the import path
    import torch_geometric.data as pyg
    from torch.utils.data import Dataset

    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import PhaseCouplingConfig, StructuralEncodingConfig
    from injection_hse.data.physics import InjectionSpec
    from injection_hse.physics import MultiGridPhysicsBank

#: Default per-graph observed fraction of the multi-grid random masker.
DEFAULT_OBSERVED_FRACTION = 0.30

#: The two curriculum TASKS a multi-grid dataset can carry per member
#: (:class:`injection_hse.config.MultiGridConfig.task`): the randomized snapshots and the coherent
#: sequences generated beside them.
TASKS = ("random", "coherent")


# =============================================================================
# Per-grid artifacts
# =============================================================================
@dataclass(frozen=True)
class GridBundle:
    """Everything a multi-grid batch needs about ONE member grid (built once, cached).

    Attributes
    ----------
    key, split, entry:
        The member's manifest identity (key, ``"train"``/``"val"``/``"test"``, full entry).
    grid:
        The reloaded ``pgml.Grid`` (persisted beside its dataset by the generator).
    source:
        The member's :class:`~injection_hse.data.ParquetDataset` (its whole scenario set — the split
        is by grid, so no scenario-level partition is applied).
    graph:
        The PyG ``Data`` for this grid, INCLUDING the structural encoding columns.
    base_voltage:
        float64 ``[N]`` per-row line-to-neutral base voltage (the per-unit reference).
    slack_rows:
        int64 ``[K]`` rows of the metered slack node(s) — always observed.
    reference_rows:
        int64 ``[N]`` angle-reference row of every row, in the frame the owning source was
        configured for (:attr:`MultiGridDataSource.angle_reference`): the same-phase slack
        row (:func:`slack_reference_rows`) or the same-phase LV-busbar row
        (:func:`reference_bus_rows`).
    observed_rows:
        int64 ``[K]`` rows of the fixed placement (installed measurement devices, or
        :func:`~injection_hse.data.central_placement` when a placement fraction is configured);
        ``None`` when no fixed placement applies.
    row_bus:
        int64 ``[N]`` index of each node-phase row's physical bus, for coherent
        bus-level sensor placement.
    injection_spec:
        The member's :class:`~injection_hse.data.InjectionSpec` — its injecting devices, their injected
        orders and their terminal rows. Taken from the member's ``ParquetDataset`` (which
        assembles one for every dataset), so it costs nothing extra; ``is_empty`` for a member
        whose dataset carries no per-device harmonic injection.
    """

    key: str
    split: str
    entry: GridEntry
    grid: "Grid"
    source: ParquetDataset
    graph: "pyg.Data"
    base_voltage: Tensor
    slack_rows: Tensor
    reference_rows: Tensor
    observed_rows: Optional[Tensor]
    row_bus: Tensor
    injection_spec: Optional[InjectionSpec] = None

    @property
    def n_nodes(self) -> int:
        """Node-phase rows ``N`` of this member."""
        return int(self.graph.num_nodes)

    @property
    def n_edges(self) -> int:
        """Directed half-edges ``E`` of this member."""
        return int(self.graph.edge_index.shape[1])

    @property
    def n_devices(self) -> int:
        """Injecting devices ``D`` of this member (``0`` without an injection)."""
        spec = self.injection_spec
        return 0 if spec is None or spec.is_empty else int(spec.n_devices)

    @property
    def device_rows(self) -> Tensor:
        """int64 ``[D, P_max]`` MEMBER-LOCAL node rows of each device's phases.

        The collate offsets these into the batched node axis; ``[0, 1]`` (empty) for a member
        with no injection. Padded slots are flagged by :attr:`device_rows_mask`.
        """
        spec = self.injection_spec
        if spec is None or spec.is_empty:
            return torch.zeros((0, 1), dtype=torch.int64)
        return spec.device_rows

    @property
    def device_rows_mask(self) -> Tensor:
        """bool ``[D, P_max]`` valid phase slots of :attr:`device_rows`."""
        spec = self.injection_spec
        if spec is None or spec.is_empty:
            return torch.zeros((0, 1), dtype=torch.bool)
        return spec.device_rows_mask


class _GridTagged:
    """Concatenated member datasets, tagging each item with its member-grid index.

    The tag is what lets the collate look up an item's graph, base voltage and slack rows —
    the multi-grid analogue of the switchable-topology variant tag.
    """

    def __init__(
        self, datasets: Sequence["Dataset"], grid_indices: Sequence[int]
    ) -> None:
        self._datasets = list(datasets)
        self._grid_indices = [int(g) for g in grid_indices]
        self._offsets = list(
            itertools.accumulate((len(d) for d in self._datasets), initial=0)
        )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> dict:
        k = bisect.bisect_right(self._offsets, i) - 1
        item = self._datasets[k][i - self._offsets[k]]
        item["grid"] = torch.tensor(self._grid_indices[k], dtype=torch.int64)
        return item


# =============================================================================
# The source
# =============================================================================
class MultiGridDataSource(DataSource):
    """Training data spanning MANY grids: one dataset per member, split by grid.

    Reads a :class:`~injection_hse.data.multigrid.MultiGridManifest` and opens one
    :class:`~injection_hse.data.ParquetDataset` per member grid, each validated against its own
    persisted grid (the per-grid row-layout assertion of the single-grid path is unchanged —
    it now holds per member). The manifest's grid-level split is authoritative: every
    scenario of a train grid is training data, and validation/test consist of grids never
    trained on.

    Parameters
    ----------
    root:
        The multi-grid dataset root (holding ``manifest.json``), or the manifest file.
    manifest:
        A pre-read manifest (skips the file read; ``root`` still locates the datasets).
    task:
        WHICH per-member dataset to open — ``"random"`` (default) the randomized snapshots
        each member carries, ``"coherent"`` the sequence dataset generated beside them
        (:class:`injection_hse.data.multigrid.CoherentSequenceConfig`). The two tasks share the member list,
        the grids, the graphs and the split, so ONE estimator trains across both by building
        one source per task; only the opened dataset directory differs.
    orders:
        The SUBSET of the corpus' generated orders to train on; ``None`` (default) takes
        every order the members carry. The selection is applied per member, so a member that
        was not generated for a requested order raises naming it and the harmonic axis stays
        uniform across the corpus either way.
    encoding:
        :class:`~injection_hse.config.StructuralEncodingConfig` applied when building each member's
        graph (the SAME encoding for every grid — a model sees one feature layout).
    coupling:
        :class:`~injection_hse.config.PhaseCouplingConfig` applied when building each member's graph
        (the SAME edge set rule for every grid, and hence one edge-feature width across the
        corpus — the width follows the config, not what a particular grid happens to couple).
    placement_fraction:
        Fraction of nodes metered by :func:`~injection_hse.data.central_placement` on members that
        carry no installed measurement devices (``None`` = no fixed placement; the slack
        rows are still forced-observed).
    max_grids:
        Cap on the members loaded PER SPLIT (smoke runs).
    keep_failed:
        Keep non-converged scenarios (default drops them, as in the single-grid path).
    per_unit:
        Records the normalization decision on the source (consumed by the training step;
        see the module docstring). Default ``True``.
    angle_reference:
        The ANGLE frame the batches are expressed in — ``"slack"`` (default), ``"lv_busbar"``
        or ``"absolute"``; see :class:`injection_hse.config.MultiGridConfig`. It selects which per-row
        reference table each member carries (:attr:`GridBundle.reference_rows`) and hence
        what the collate emits as ``reference_rows``. It must match the frame the training
        step applies — :func:`build_multigrid_source` passes the config's value.
    with_injection:
        Validate the members' INJECTED orders at construction (default ``False`` = only when
        asked for, since a voltage-only run never reads the device axes). The batch's device
        axes are emitted once an operator bank is attached
        (:meth:`attach_physics_bank`), which is what supplies their per-unit current base.

    Raises
    ------
    ValueError
        If the members disagree on the solved orders or the step count ``T`` — one model
        cannot be trained across datasets with different harmonic axes — or, with
        ``with_injection``, on the injected orders ``O``; if ``angle_reference`` is not
        one of the three frames or ``task`` not one of :data:`TASKS`; or if the requested
        task's dataset is missing from a member.
    """

    def __init__(
        self,
        root,
        *,
        manifest: Optional[MultiGridManifest] = None,
        task: str = "random",
        orders: Optional[list[int]] = None,
        encoding: Optional["StructuralEncodingConfig"] = None,
        coupling: Optional["PhaseCouplingConfig"] = None,
        placement_fraction: Optional[float] = None,
        placement_min_per_feeder: int = 1,
        max_grids: Optional[int] = None,
        keep_failed: bool = False,
        per_unit: bool = True,
        angle_reference: str = "slack",
        with_injection: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.is_file():
            self.root = self.root.parent
        self.manifest = (
            MultiGridManifest.read(self.root) if manifest is None else manifest
        )
        self.task = _validated_task(task)
        self.encoding = encoding
        self.coupling = coupling
        self.placement_fraction = placement_fraction
        self.placement_min_per_feeder = int(placement_min_per_feeder)
        self.per_unit = bool(per_unit)
        self.angle_reference = _validated_frame(angle_reference)

        entries: list[GridEntry] = []
        for split in ("train", "val", "test"):
            selected = self.manifest.split(split)
            if max_grids is not None:
                selected = selected[:max_grids]
            entries.extend(selected)
        if not entries:
            raise ValueError(f"the manifest at {self.root} lists no grids.")

        # the requested SUBSET of the corpus' orders, applied per member (each member's
        # dataset raises if it does not carry one of them); ``None`` = every stored order.
        self._requested_orders = None if orders is None else [int(o) for o in orders]
        self._bundles: list[GridBundle] = [
            self._build_bundle(entry, keep_failed=keep_failed) for entry in entries
        ]
        first = self._bundles[0]
        self._orders = list(first.source.orders)
        self._n_steps = int(first.source.n_steps)
        for bundle in self._bundles[1:]:
            if bundle.source.orders != first.source.orders:
                raise ValueError(
                    f"grid {bundle.key!r} was generated for orders "
                    f"{bundle.source.orders}, grid {first.key!r} for "
                    f"{first.source.orders} (task {self.task!r}); one estimator needs one "
                    "harmonic axis."
                )
            if bundle.source.n_steps != self._n_steps:
                raise ValueError(
                    f"grid {bundle.key!r} has {bundle.source.n_steps} step(s), grid "
                    f"{first.key!r} has {self._n_steps} (task {self.task!r}); the step axis "
                    "must be uniform."
                )
        self._by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for i, bundle in enumerate(self._bundles):
            self._by_split[bundle.split].append(i)
        self._injection_orders: Optional[list[int]] = None
        self._bank: Optional["MultiGridPhysicsBank"] = None
        if with_injection:
            self.injection_orders  # validate the device axes now, not at the first batch

    # -- construction -------------------------------------------------------
    def _build_bundle(self, entry: GridEntry, *, keep_failed: bool) -> GridBundle:
        grid = read_grid(self.root, entry.key)
        path = task_dataset_dir(self.root, entry.key, self.task)
        if not path.is_dir():
            raise ValueError(
                f"grid {entry.key!r} carries no {self.task!r} dataset at {path}; generate "
                "the multi-grid dataset with a MultiGridGenerationConfig.coherent section "
                "(or load it with task='random')."
            )
        source = ParquetDataset(
            str(path),
            grid=grid,
            val_fraction=0.0,  # the split is by GRID, not by scenario
            test_fraction=0.0,
            orders=self._requested_orders,
            keep_failed=keep_failed,
        )
        if source.time_unix_s is not None and source.time_unix_s.ndim != 1:
            raise ValueError(
                f"grid {entry.key!r}: its {self.task!r} dataset starts every sequence at "
                "its own time ([B, T] time axis); a multi-grid batch shares one step time "
                "axis, so generate it with one start_time."
            )
        graph = build_graph(
            grid, orders=source.orders, encoding=self.encoding, coupling=self.coupling
        )
        # a member that carries instrumentation defines its own placement; one that does not
        # (every generated grid) falls back to the structural rule, when one is configured.
        observed: Optional[Tensor] = None
        if grid.measurement_devices:
            observed = MeasurementModel.from_grid(grid).observed_rows(grid)
        elif self.placement_fraction is not None:
            observed = central_placement(
                grid,
                self.placement_fraction,
                min_per_feeder=self.placement_min_per_feeder,
            )
        from pgml.assembly import node_phase_index

        _, row_bus = torch.unique(
            node_phase_index(grid).node_ids.to(torch.int64),
            sorted=True,
            return_inverse=True,
        )
        return GridBundle(
            key=entry.key,
            split=entry.split,
            entry=entry,
            grid=grid,
            source=source,
            graph=graph,
            base_voltage=base_voltage_per_row(grid),
            slack_rows=MeasurementModel([], slack_known=True).slack_rows(grid),
            reference_rows=(
                reference_bus_rows(grid)
                if self.angle_reference == "lv_busbar"
                else slack_reference_rows(grid)
            ),
            observed_rows=observed,
            row_bus=row_bus,
            injection_spec=source.injection_spec,
        )

    # -- the DataSource contract --------------------------------------------
    def train_dataset(self) -> "Dataset":
        return self._dataset("train")

    def val_dataset(self) -> "Dataset":
        return self._dataset("val")

    def test_dataset(self) -> "Dataset":
        """The held-out TEST split — the REAL grids, never seen during training."""
        return self._dataset("test")

    def _dataset(self, split: str) -> "Dataset":
        indices = self._by_split[split]
        return _GridTagged(
            [self._bundles[i].source.train_dataset() for i in indices], indices
        )

    def collate(self) -> "MultiGridCollate":
        """The DataLoader ``collate_fn`` for this source (see :class:`MultiGridCollate`).

        Carries the per-device injection axes exactly when an operator bank is attached
        (:meth:`attach_physics_bank`) — the bank supplies their per-unit current base.
        """
        return MultiGridCollate(
            self, with_injection=self._bank is not None, bank=self._bank
        )

    @property
    def orders(self) -> list[int]:
        return list(self._orders)

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_nodes(self) -> int:
        """Not defined for a multi-grid source (``N`` differs per member).

        Use :attr:`grids` / :meth:`bundle` for the per-member row counts; a model trained
        here must not assume a fixed ``N``.
        """
        raise ValueError(
            "a multi-grid source has no single node count; read it per member from "
            "MultiGridDataSource.grids (GridBundle.n_nodes)."
        )

    def graph(self) -> "pyg.Data":
        """Not defined for a multi-grid source (one graph per member).

        The batch carries the concatenated graph tensors; per-member graphs are on
        :attr:`grids`.
        """
        raise ValueError(
            "a multi-grid source has no single graph; the collate concatenates the "
            "per-member graphs (MultiGridDataSource.grids[i].graph)."
        )

    @property
    def grid(self) -> "Grid":
        """Not defined for a multi-grid source (one grid per member)."""
        raise ValueError(
            "a multi-grid source has no single grid; use MultiGridDataSource.grids."
        )

    # -- multi-grid accessors ------------------------------------------------
    @property
    def grids(self) -> list[GridBundle]:
        """The member bundles, in manifest order (train, then val, then test)."""
        return list(self._bundles)

    def bundle(self, index: int) -> GridBundle:
        """The member bundle at a batch's ``grid_index`` entry."""
        return self._bundles[int(index)]

    @property
    def injection_orders(self) -> list[int]:
        """The injected harmonic orders ``O`` (``h > 1``) shared by every member.

        The device axis of a batch is ``[D*, O, 1]``, so ``O`` must be uniform: members
        injecting at different orders cannot share one head. Members whose dataset carries no
        per-device injection are ignored here (they contribute ``D_b = 0``).

        Raises
        ------
        ValueError
            If two members inject at different order sets, or no member injects at all (a run
            asking for the device axes on a dataset that has none).
        """
        if self._injection_orders is not None:
            return list(self._injection_orders)
        orders: Optional[list[int]] = None
        reference = ""
        for bundle in self._bundles:
            spec = bundle.injection_spec
            if spec is None or spec.is_empty:
                continue
            own = [int(o) for o in spec.orders]
            if orders is None:
                orders, reference = own, bundle.key
            elif own != orders:
                raise ValueError(
                    f"grid {bundle.key!r} injects at orders {own}, grid {reference!r} at "
                    f"{orders}; one per-device injection head needs one order axis "
                    "(regenerate the members with the same MultiGridGenerationConfig "
                    "orders)."
                )
        if orders is None:
            raise ValueError(
                f"no member of the multi-grid dataset at {self.root} carries a per-device "
                "harmonic injection, so there is no device axis to batch; generate it with "
                "harmonic orders (MultiGridGenerationConfig.orders) or train the voltage "
                "state only."
            )
        self._injection_orders = orders
        return list(orders)

    @property
    def physics_bank(self) -> Optional["MultiGridPhysicsBank"]:
        """The attached per-member operator bank (``None`` = voltage-only)."""
        return self._bank

    def attach_physics_bank(self, bank: "MultiGridPhysicsBank") -> None:
        """Attach the per-member operator bank, turning ON the batch's device axes.

        The bank (:class:`injection_hse.physics.MultiGridPhysicsBank`) is built FROM this source, so it
        cannot be a constructor argument; attaching it afterwards is what lets
        :meth:`collate` emit ``injection_target`` / ``device_*`` in the per-unit current base
        the bank defines. Validates the injected orders immediately.
        """
        if len(bank) != len(self._bundles):
            raise ValueError(
                f"the operator bank serves {len(bank)} grid(s) but the source has "
                f"{len(self._bundles)}; the bank must be built from this source."
            )
        self.injection_orders  # loud now rather than in the first collate call
        self._bank = bank

    def split_indices(self, split: str) -> list[int]:
        """Member indices belonging to one split."""
        return list(self._by_split[split])

    def per_unit_states(
        self,
        *,
        split: str = "train",
        max_scenarios_per_grid: Optional[int] = 8,
        angle_reference: Optional[str] = None,
    ) -> list[Tensor]:
        """Per-unit member states for fitting a dataset-global normalizer.

        Returns one complex tensor per member of the split, ``[S, H, N_g]`` (or
        ``[S, T, H, N_g]`` for sequences), already divided by that grid's base voltage. The
        caller encodes each and pools over the NODE axis (``H`` is uniform), which is what
        makes one :class:`~injection_hse.normalization.HarmonicNormalizer` valid across voltage
        classes — see the module docstring.

        ``max_scenarios_per_grid`` caps the scenarios taken per member (``None`` = all), so
        the fit sees every TOPOLOGY rather than mostly the biggest dataset.

        ``angle_reference`` selects the frame and must match the one the training step
        applies (:attr:`injection_hse.config.MultiGridConfig.angle_reference`). ``None`` (default) uses
        the source's own :attr:`angle_reference`; ``"absolute"`` explicitly asks for the raw
        solver frame. A ROW frame (``"slack"`` / ``"lv_busbar"``) other than the source's is
        refused — the per-member reference rows were built for the source's frame, so
        honoring the request would silently rotate against the wrong bus. Per-unit
        MAGNITUDES are rotation-invariant, but the Cartesian components a normalizer also
        standardizes are not: a normalizer fit in one frame and applied in the other is
        mis-scaled.
        """
        frame = self.angle_reference if angle_reference is None else angle_reference
        frame = _validated_frame(frame)
        if frame != "absolute" and frame != self.angle_reference:
            raise ValueError(
                f"angle_reference={frame!r} was requested but this source carries the "
                f"reference rows of the {self.angle_reference!r} frame; build the source "
                "with that frame (injection_hse.data.build_multigrid_source does)."
            )
        out: list[Tensor] = []
        for i in self._by_split[split]:
            bundle = self._bundles[i]
            state = bundle.source.states
            if max_scenarios_per_grid is not None:
                state = state[: int(max_scenarios_per_grid)]
            if frame != "absolute":
                state = slack_referenced_state(
                    state, bundle.reference_rows, orders=self._orders
                )
            out.append(per_unit_state(state, bundle.base_voltage))
        return out


# =============================================================================
# Collate
# =============================================================================
class MultiGridCollate:
    """Collate items from different grids into ONE disjoint graph batch.

    The sample axis folds into the node axis (torch-geometric convention): the batch's
    ``v``/``node_static``/``base_voltage`` are concatenations over the items' node-phase
    rows, and ``ptr``/``batch`` name the item each row belongs to. See the module docstring
    for the frozen key/shape table.

    Parameters
    ----------
    source:
        The :class:`MultiGridDataSource` whose per-grid artifacts the batch assembles from —
        including the angle frame: the emitted ``reference_rows`` are the source's
        :attr:`~MultiGridDataSource.angle_reference` table, mapped to global indices.
    masker:
        Optional :class:`GraphBatchMasker` — when given, the batch carries ``node_mask``
        drawn PER GRAPH (each item gets its own observed fraction) with the metered slack
        rows forced observed. Left ``None`` by default so the mask is drawn in the training
        step from a seeded generator; a masker here draws in the DataLoader worker, which is
        only reproducible with ``num_workers=0`` (or a per-worker seeding scheme).
    generator:
        RNG for the optional masker.
    with_injection:
        Emit the per-device axes (``injection_target`` / ``device_ptr`` / ``device_batch`` /
        ``device_rows`` / ``device_rows_mask``; see the module docstring). Default ``False``
        leaves the batch exactly as a voltage-only run has always seen it.
    bank:
        The :class:`injection_hse.physics.MultiGridPhysicsBank` supplying each member's per-unit CURRENT
        base. REQUIRED with ``with_injection``: the emitted target is dimensionless by
        contract, so there is no unscaled fallback to fall into.
    """

    def __init__(
        self,
        source: MultiGridDataSource,
        *,
        masker: Optional["GraphBatchMasker"] = None,
        generator: Optional[torch.Generator] = None,
        with_injection: bool = False,
        bank: Optional["MultiGridPhysicsBank"] = None,
    ) -> None:
        if with_injection and bank is None:
            raise ValueError(
                "MultiGridCollate(with_injection=True) needs the operator bank that defines "
                "the per-unit current base (bank=MultiGridPhysicsBank(source, ...)); the "
                "batch's injection target is dimensionless by contract."
            )
        self.source = source
        self.masker = masker
        self.generator = generator
        self.with_injection = bool(with_injection)
        self.bank = bank
        if self.with_injection:
            source.injection_orders  # validate the uniform order axis up front

    def __call__(self, items: list[dict]) -> dict:
        if not items:
            raise ValueError("MultiGridCollate received an empty item list.")
        bundles = [self.source.bundle(int(item["grid"])) for item in items]
        sizes = [b.n_nodes for b in bundles]
        edge_sizes = [b.n_edges for b in bundles]
        ptr = torch.tensor([0, *itertools.accumulate(sizes)], dtype=torch.int64)
        bus_sizes = [int(b.row_bus.max().item()) + 1 for b in bundles]
        bus_ptr = torch.tensor([0, *itertools.accumulate(bus_sizes)], dtype=torch.int64)
        edge_ptr = torch.tensor(
            [0, *itertools.accumulate(edge_sizes)], dtype=torch.int64
        )

        states = [item["v"] for item in items]
        lead = states[0].shape[:-1]
        for state in states[1:]:
            if state.shape[:-1] != lead:
                raise ValueError(
                    "multi-grid items must share the [T, H] axes; got "
                    f"{tuple(state.shape[:-1])} and {tuple(lead)}."
                )

        batch: dict = {
            "v": torch.cat(states, dim=-1),
            "batch": torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.int64),
                torch.tensor(sizes, dtype=torch.int64),
            ),
            "ptr": ptr,
            "bus_ptr": bus_ptr,
            "row_bus": torch.cat(
                [b.row_bus + int(off) for b, off in zip(bundles, bus_ptr[:-1])]
            ),
            "edge_batch": torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.int64),
                torch.tensor(edge_sizes, dtype=torch.int64),
            ),
            "edge_ptr": edge_ptr,
            "grid_index": torch.stack([item["grid"] for item in items]),
            "edge_index": torch.cat(
                [b.graph.edge_index + int(off) for b, off in zip(bundles, ptr[:-1])],
                dim=-1,
            ),
            "node_static": torch.cat([b.graph.x for b in bundles], dim=0),
            "edge_static": torch.cat([b.graph.edge_attr for b in bundles], dim=0),
            "edge_branch_ids": torch.cat([b.graph.edge_branch_ids for b in bundles]),
            "base_voltage": torch.cat([b.base_voltage for b in bundles]),
            "slack_rows": _global_rows([b.slack_rows for b in bundles], ptr),
            # one entry per ROW (not a row subset), so the angle frame is available for the
            # whole concatenated axis without a per-member lookup in the training step
            "reference_rows": torch.cat(
                [
                    b.reference_rows.to(torch.int64) + int(off)
                    for b, off in zip(bundles, ptr[:-1])
                ]
            ),
        }
        if "index" in items[0]:
            batch["index"] = torch.stack([item["index"] for item in items])
        if all(b.observed_rows is not None for b in bundles):
            batch["observed_rows"] = _global_rows(
                [b.observed_rows for b in bundles], ptr
            )
        if "time_unix_s" in items[0]:
            batch["time_unix_s"] = items[0]["time_unix_s"]
        if self.masker is not None:
            batch["node_mask"] = self.masker.mask(
                ptr, forced_rows=batch["slack_rows"], generator=self.generator
            )
        if self.with_injection:
            batch.update(self._injection_axes(items, bundles, ptr, lead[:-1]))
        return batch

    # -- the device axis -----------------------------------------------------
    def _injection_axes(
        self,
        items: list[dict],
        bundles: list[GridBundle],
        ptr: Tensor,
        lead: tuple[int, ...],
    ) -> dict:
        """The concatenated per-device axes of a batch (see the module docstring).

        ``lead`` is the items' leading shape ahead of ``[H, N]`` — ``()`` for a snapshot,
        ``(T,)`` for a sequence — which the injection target shares. Every member's block is
        divided by its own per-unit current base before concatenation, so the ``D*`` axis
        carries one dimensionless quantity rather than a mixture of voltage classes.
        """
        n_orders = len(
            self.source.injection_orders
        )  # uniform across members (validated)
        counts = [b.n_devices for b in bundles]
        device_ptr = torch.tensor([0, *itertools.accumulate(counts)], dtype=torch.int64)
        p_max = max(
            (int(b.device_rows.shape[1]) for b in bundles if b.n_devices), default=1
        )

        targets: list[Tensor] = []
        rows: list[Tensor] = []
        masks: list[Tensor] = []
        for item, bundle, offset in zip(items, bundles, ptr[:-1].tolist()):
            n_dev = bundle.n_devices
            if n_dev == 0:
                targets.append(
                    torch.zeros((*lead, 0, n_orders, 1), dtype=torch.float32)
                )
                rows.append(torch.full((0, p_max), -1, dtype=torch.int64))
                masks.append(torch.zeros((0, p_max), dtype=torch.bool))
                continue
            target = item.get("injection_target")
            if target is None:
                raise ValueError(
                    f"grid {bundle.key!r} declares {n_dev} injecting device(s) but its item "
                    "carries no 'injection_target'; the member dataset and its InjectionSpec "
                    "disagree."
                )
            expected = (*lead, n_dev, n_orders, 1)
            if tuple(target.shape) != expected:
                raise ValueError(
                    f"grid {bundle.key!r} has an injection target of shape "
                    f"{tuple(target.shape)}, expected {expected} ([*lead, D, O, 1])."
                )
            base = self.bank.device_current_base(int(item["grid"]))  # [D, O]
            base = base.to(device=target.device, dtype=target.dtype)
            base = torch.where(base > 0, base, torch.ones_like(base))
            targets.append(target / base.unsqueeze(-1))

            local = bundle.device_rows  # [D, P_b]
            valid = bundle.device_rows_mask
            row = torch.full((n_dev, p_max), -1, dtype=torch.int64)
            mask = torch.zeros((n_dev, p_max), dtype=torch.bool)
            width = int(local.shape[1])
            # GLOBAL rows: member-local + the item's node offset, pad slots left at -1
            row[:, :width] = torch.where(
                valid, local.to(torch.int64) + int(offset), torch.full_like(local, -1)
            )
            mask[:, :width] = valid
            rows.append(row)
            masks.append(mask)

        return {
            "injection_target": torch.cat(targets, dim=-3),
            "device_ptr": device_ptr,
            "device_batch": torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.int64),
                torch.tensor(counts, dtype=torch.int64),
            ),
            "device_rows": torch.cat(rows, dim=0),
            "device_rows_mask": torch.cat(masks, dim=0),
        }


def _global_rows(rows: Sequence[Optional[Tensor]], ptr: Tensor) -> Tensor:
    """Concatenate per-item row indices into GLOBAL indices of the batched node axis."""
    parts = [
        r.to(torch.int64) + int(off)
        for r, off in zip(rows, ptr[:-1])
        if r is not None and r.numel()
    ]
    if not parts:
        return torch.zeros(0, dtype=torch.int64)
    return torch.cat(parts)


# =============================================================================
# Masking in the concatenated space
# =============================================================================
class GraphBatchMasker:
    """Random observation mask over a batched node axis, drawn PER GRAPH.

    The multi-grid analogue of :class:`injection_hse.masking.RandomMasker`. The observed fraction must
    apply per member, not to the concatenated axis: a global top-k would spend the whole
    sensor budget on the largest grid in the batch and leave small ones unobserved, so the
    model would never learn the sparse regime on them. Each item therefore observes
    ``round(fraction * N_b)`` of ITS OWN rows (at least ``min_observed``), and the forced
    rows (the metered slack, as GLOBAL indices) are observed on top — exactly the
    single-grid convention, where the substation meter never disappears.

    Vectorized over the whole batch: one score draw plus one sort, no Python loop over the
    member graphs.
    """

    def __init__(
        self,
        observed_fraction: float = DEFAULT_OBSERVED_FRACTION,
        *,
        min_observed: int = 1,
    ) -> None:
        if not 0.0 <= observed_fraction <= 1.0:
            raise ValueError("observed_fraction must be in [0, 1].")
        self.observed_fraction = float(observed_fraction)
        self.min_observed = int(min_observed)

    def mask(
        self,
        ptr: Tensor,
        *,
        forced_rows: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Boolean ``[N*]`` observation mask over the batched node axis (``True`` = observed).

        Parameters
        ----------
        ptr:
            int64 ``[B+1]`` node offsets of the batch (``batch["ptr"]``).
        forced_rows:
            GLOBAL row indices always observed (``batch["slack_rows"]``).
        device, generator:
            Target device and RNG (the draw honors the generator's device).
        """
        return segment_random_mask(
            ptr,
            fraction=self.observed_fraction,
            min_observed=self.min_observed,
            forced_rows=forced_rows,
            device=device,
            generator=generator,
        )


def segment_random_mask(
    ptr: Tensor,
    *,
    fraction: float,
    min_observed: int = 1,
    forced_rows: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Per-segment random observation mask over a concatenated node axis.

    Observes ``max(min_observed, round(fraction * N_b))`` uniformly drawn rows of each
    segment ``ptr[b]:ptr[b+1]``, then forces ``forced_rows`` observed. Fully vectorized: the
    per-segment top-k is obtained by ranking one score draw within its segment (a single
    sort of a segment-major key), so no Python loop over segments appears on the hot path.

    Parameters
    ----------
    ptr:
        int64 ``[B+1]`` segment offsets (``ptr[0] == 0``, ``ptr[-1] == N*``).
    fraction:
        Observed share PER SEGMENT.
    min_observed:
        Lower bound on the observed count per segment.
    forced_rows:
        Global row indices forced observed (``None`` = none).
    device, generator:
        Target device / RNG.

    Returns
    -------
    torch.Tensor
        Boolean ``[N*]`` (``True`` = observed).
    """
    ptr = ptr.to(torch.int64)
    device = ptr.device if device is None else device
    n_total = int(ptr[-1])
    mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    if n_total == 0:
        return mask

    sizes = ptr[1:] - ptr[:-1]
    segment = torch.repeat_interleave(
        torch.arange(sizes.numel(), dtype=torch.int64, device=device), sizes.to(device)
    )
    scores = torch.rand(
        n_total, device=device, generator=generator, dtype=torch.float64
    )
    # segment-major, score-minor key: scores in [0, 1) keep the segment ordering intact
    order = torch.argsort(segment.to(torch.float64) + scores)
    rank = (
        torch.arange(n_total, dtype=torch.int64, device=device)
        - ptr.to(device)[segment[order]]
    )
    keep = torch.clamp(
        torch.round(float(fraction) * sizes.to(torch.float64)).to(torch.int64),
        min=int(min_observed),
    ).clamp(max=sizes)
    mask[order] = rank < keep.to(device)[segment[order]]
    if forced_rows is not None and forced_rows.numel():
        mask[forced_rows.to(device=device, dtype=torch.int64)] = True
    return mask


def placement_mask(rows: Tensor, n_total: int, *, device=None) -> Tensor:
    """Boolean ``[N*]`` mask of a FIXED placement given its global rows.

    The multi-grid counterpart of :class:`injection_hse.masking.PlacementMasker`: the placement is
    already resolved per member and concatenated by the collate
    (``batch["observed_rows"]``), so the mask is a scatter.
    """
    mask = torch.zeros(int(n_total), dtype=torch.bool, device=device)
    if rows.numel():
        mask[rows.to(device=mask.device, dtype=torch.int64)] = True
    return mask


# =============================================================================
# Per-unit conversion
# =============================================================================
def per_unit_state(v: Tensor, base_voltage: Tensor) -> Tensor:
    """Divide a state ``[..., N]`` by the per-row base voltage (volts -> per-unit).

    The multi-grid normalization decision (see the module docstring): features are encoded
    and normalized in per-unit so one normalizer spans every voltage class, while targets
    and metrics stay in volts. Preserves ``v``'s dtype and device; a zero base (a row with no
    rated voltage) passes the value through unscaled rather than producing an infinity.
    """
    base = base_voltage.to(device=v.device, dtype=v.real.dtype)
    base = torch.where(base > 0, base, torch.ones_like(base))
    return v / base


def physical_state(v_pu: Tensor, base_voltage: Tensor) -> Tensor:
    """Inverse of :func:`per_unit_state` (per-unit -> volts)."""
    base = base_voltage.to(device=v_pu.device, dtype=v_pu.real.dtype)
    base = torch.where(base > 0, base, torch.ones_like(base))
    return v_pu * base


# =============================================================================
# Angle reference frame
# =============================================================================
#: The angle frames a multi-grid run can be expressed in (``MultiGridConfig.angle_reference``).
ANGLE_FRAMES = ("slack", "lv_busbar", "absolute")


def _validated_frame(name: str) -> str:
    """One of :data:`ANGLE_FRAMES`, or a ValueError naming the three."""
    if name not in ANGLE_FRAMES:
        raise ValueError(
            f"angle_reference must be one of {ANGLE_FRAMES}; got {name!r}."
        )
    return name


def _validated_task(name: str) -> str:
    """One of :data:`TASKS`, or a ValueError naming both."""
    if name not in TASKS:
        raise ValueError(f"task must be one of {TASKS}; got {name!r}.")
    return name


def slack_reference_rows(grid: "Grid") -> Tensor:
    """int64 ``[N]`` angle-reference row of every node-phase row of one grid.

    Row ``i`` references the row of the grid's slack at the SAME phase — the bus and phase an
    instrument at the substation would lock its phase reference to. The rows follow
    ``pgml.assembly.node_phase_index`` (the layout of the state ``V[..., N]``), and the slack
    rows are the in-service ``Source`` rows (``injection_hse.data.MeasurementModel.slack_rows``); with
    several sources the LOWEST row of a phase wins, so the reference is deterministic.

    A row whose phase carries no slack row at all (e.g. a neutral row on a grid whose source
    is modelled without one) falls back to the FIRST slack row. Its rotation then also removes
    that slack phase's angle, which is a constant per grid — the property the frame exists for
    — but the row does not end up at angle zero the way a same-phase row does.

    Returns a CPU int64 tensor, static per grid (inert metadata, never on the autograd tape).

    Raises
    ------
    ValueError
        If the grid has no in-service ``Source``: there is no angle reference to express the
        state in (and such a grid has no slack boundary to estimate from either).
    """
    from pgml.assembly import node_phase_index

    codes = node_phase_index(grid).phase_codes.to(torch.int64)
    slack = MeasurementModel([], slack_known=True).slack_rows(grid).to(torch.int64)
    if slack.numel() == 0:
        raise ValueError(
            "the grid carries no in-service Source, so its state has no slack angle "
            "reference; add a Source or run with angle_reference='absolute'."
        )
    reference = torch.full_like(
        codes, int(slack[0])
    )  # the no-same-phase-slack fallback
    seen: set[int] = set()
    for row in slack.tolist():  # a handful of slack rows; static per grid, off the tape
        code = int(codes[row])
        if code in seen:  # the first slack row of a phase wins
            continue
        seen.add(code)
        reference[codes == code] = int(row)
    return reference


def reference_bus_rows(grid: "Grid") -> Tensor:
    """int64 ``[N]`` angle-reference row of every row, referenced at the LV BUSBAR.

    The deployment frame: a real LV measurement campaign locks its phase reference to the
    reference meter in the substation, which sits on the LOW-VOLTAGE busbar — not on the
    (MV-side) grid slack. Referencing the state there also takes the feeding transformer out
    of the angle: with a slack reference, every node of the LV network carries that
    transformer's vector-group clock, so the estimator's target moves with a modelling
    decision made ABOVE the busbar it will never observe.

    The reference bus is the lower-rated-voltage-side node of the in-service ``Transformer``
    NEAREST the slack — nearest by hop count over the in-service branch graph (open switches
    excluded) from the in-service ``Source`` nodes, taking each transformer's closer terminal,
    ties broken by the lowest branch id. The low-voltage side is the terminal whose
    ``Node.u_rated_v`` is smaller; when the two ratings are equal (or absent) the schema's own
    convention decides — ``Transformer.to_node`` is the LV winding.

    Rows map per phase exactly as in :func:`slack_reference_rows`: row ``i`` references the
    reference bus' row of the SAME phase, and a phase the reference bus does not carry falls
    back to that bus' FIRST row (a constant per grid, so the frame still removes the member's
    global angle convention, but such a row does not end up at angle zero).

    A grid with NO in-service transformer has no such busbar: the slack IS the local
    reference, and the result is exactly :func:`slack_reference_rows` — on an all-LV corpus
    the two frames coincide.

    Returns a CPU int64 tensor, static per grid (inert metadata, never on the autograd tape).

    Raises
    ------
    ValueError
        If the grid has no in-service ``Source`` (as :func:`slack_reference_rows`): without a
        slack there is no place to start looking for the feeding transformer.
    """
    from pgml.assembly import node_phase_index

    transformer = _nearest_transformer(grid)
    if transformer is None:
        return slack_reference_rows(grid)

    index = node_phase_index(grid)
    codes = index.phase_codes.to(torch.int64)
    bus_rows = torch.nonzero(
        index.node_ids.to(torch.int64) == _low_voltage_node(grid, transformer),
        as_tuple=False,
    ).flatten()
    reference = torch.full_like(codes, int(bus_rows[0]))  # the no-same-phase fallback
    for row in bus_rows.tolist():  # one node's phases; static per grid, off the tape
        reference[codes == int(codes[row])] = int(row)
    return reference


def _conducting_branches(grid: "Grid") -> list:
    """In-service branches that connect two distinct nodes; OPEN switches excluded.

    The current-carrying topology — the same rule ``pgml.topology.branch_edges`` draws: an
    open switch defines no path, so it neither connects the search nor supplies a reference.
    """
    from pgml.schemas.grid_schema import Switch

    return [
        b
        for b in grid.branches
        if b.in_service
        and not (isinstance(b, Switch) and not b.default_closed)
        and int(b.from_node) != int(b.to_node)
    ]


def _nearest_transformer(grid: "Grid"):
    """The in-service ``Transformer`` closest to a slack node, or ``None`` if there is none.

    Breadth-first over :func:`_conducting_branches` from every in-service ``Source`` node
    (multi-slack aware), scoring each transformer by the hop count of its CLOSER terminal;
    the lowest branch id breaks a tie, so the choice is deterministic. Transformers in a
    de-energized island (unreachable from any source) are ignored.
    """
    from collections import deque

    from pgml.schemas.grid_schema import Transformer
    from pgml.topology import slack_node_ids

    try:
        roots = [int(n) for n in slack_node_ids(grid)]
    except ValueError as exc:
        raise ValueError(
            "the grid carries no in-service Source, so its state has no angle reference to "
            "search from; add a Source or run with angle_reference='absolute'."
        ) from exc

    branches = _conducting_branches(grid)
    adjacency: dict[int, list[int]] = {}
    for branch in branches:
        adjacency.setdefault(int(branch.from_node), []).append(int(branch.to_node))
        adjacency.setdefault(int(branch.to_node), []).append(int(branch.from_node))
    hops: dict[int, int] = {r: 0 for r in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in hops:
                hops[neighbor] = hops[node] + 1
                queue.append(neighbor)

    best = None
    for branch in branches:
        if not isinstance(branch, Transformer):
            continue
        reachable = [
            hops[n]
            for n in (int(branch.from_node), int(branch.to_node))
            if n in hops  # a de-energized island carries no reference
        ]
        if not reachable:
            continue
        key = (min(reachable), int(branch.id))
        if best is None or key < best[0]:
            best = (key, branch)
    return None if best is None else best[1]


def _low_voltage_node(grid: "Grid", transformer) -> int:
    """Node id of ``transformer``'s LOW-voltage terminal (the busbar the meter sits on).

    Decided by the terminals' ``Node.u_rated_v``; on an equal (or missing) rating the schema
    convention applies — a ``Transformer``'s ``to`` winding is the LV one. Ratings are read as
    plain numbers (they may be tensors under the float/tensor duality); this is inert
    metadata, never on the autograd tape.
    """
    rated = {
        int(node.id): _plain_number(node.u_rated_v)
        for node in grid.nodes
        if node.u_rated_v is not None
    }
    from_node, to_node = int(transformer.from_node), int(transformer.to_node)
    u_from, u_to = rated.get(from_node), rated.get(to_node)
    if u_from is not None and u_to is not None and u_from != u_to:
        return from_node if u_from < u_to else to_node
    return to_node


def _plain_number(value) -> float:
    """A python float from a plain number OR a 0-d tensor (the float/tensor duality)."""
    if hasattr(value, "detach"):
        return float(value.detach().reshape(()).item())
    return float(value)


def slack_frame_rotation(
    v: Tensor, reference_rows: Tensor, *, orders: Sequence[int]
) -> Tensor:
    """The unit rotation phasors ``[*lead, H, N]`` of the referenced angle frame.

    Entry ``[h, i]`` is ``exp(-1j * h * phi_i)``, ``phi_i`` the FUNDAMENTAL angle of row
    ``i``'s reference row taken from ``v``. Multiplying by it is
    :func:`slack_referenced_state`; multiplying by its conjugate is the inverse. WHICH bus
    each row references is the ``reference_rows`` table handed in — the slack
    (:func:`slack_reference_rows`) or the LV busbar (:func:`reference_bus_rows`); the
    rotation itself is the same construction either way.

    Exposed separately because the frame of a state must be applied to quantities DERIVED
    from it: a nodal injection current or a branch current is rotated by the rotation the
    VOLTAGES define (``v' = D v`` pairs with ``i' = D i``, which is what keeps the residual
    ``i' - (D Y D^{-1}) v'`` a rotation of ``i - Y v``), and the physics — assembling ``Y``,
    solving, forming branch currents — must run in the ABSOLUTE frame because the row-dependent
    diagonal ``D`` does not commute with ``Y``. A caller therefore needs the rotation itself,
    not only its application to the state it was derived from.

    Parameters
    ----------
    v:
        The complex ``[*lead, H, N]`` state defining the frame (volts or per-unit — the frame
        is scale-free).
    reference_rows:
        int64 ``[N]`` reference row of each row (member-local, or GLOBAL for a collated
        multi-grid batch). Moved to ``v``'s device.
    orders:
        The ``H`` solved harmonic orders, aligned with ``v``'s harmonic axis; must contain
        the fundamental ``1``.

    Returns
    -------
    torch.Tensor
        Complex ``[*lead, H, N]`` of unit modulus, on ``v``'s device and dtype.
    """
    order_list = [int(h) for h in orders]
    if 1 not in order_list:
        raise ValueError(
            f"the referenced angle frame is defined by the FUNDAMENTAL phasor, but "
            f"order 1 is not among the solved orders {order_list}."
        )
    if v.shape[-2] != len(order_list):
        raise ValueError(
            f"the state has {int(v.shape[-2])} harmonic rows but {len(order_list)} orders "
            "were given."
        )
    rows = reference_rows.to(device=v.device, dtype=torch.int64)
    if rows.shape != v.shape[-1:]:
        raise ValueError(
            f"reference_rows must be [N] = [{int(v.shape[-1])}]; got {tuple(rows.shape)}."
        )
    real_dtype = v.real.dtype
    fundamental = v.select(-2, order_list.index(1))  # [*lead, N]
    reference = fundamental.index_select(-1, rows)  # [*lead, N]
    # unit phasor of the reference row; the clamp keeps a de-energized (zero) reference from
    # producing 0/0 — it then contributes angle 0, i.e. no rotation at all.
    unit = reference / reference.abs().clamp_min(torch.finfo(real_dtype).tiny)
    phi = torch.angle(unit).unsqueeze(-2)  # [*lead, 1, N]
    h = torch.tensor(order_list, dtype=real_dtype, device=v.device).reshape(-1, 1)
    angle = -(h * phi)  # [*lead, H, N]
    return torch.polar(torch.ones_like(angle), angle)


def slack_referenced_state(
    v: Tensor, reference_rows: Tensor, *, orders: Sequence[int]
) -> Tensor:
    """Rotate a complex state into the REFERENCED angle frame.

    Every row is rotated by the fundamental angle ``phi`` of its reference row (the slack row
    of :func:`slack_reference_rows`, or the LV-busbar row of :func:`reference_bus_rows` —
    whichever table is handed in), order ``h`` by ``exp(-1j*h*phi)``. Scaling the rotation
    with the order makes it a shift of the TIME reference rather than an independent
    per-order twist, so the harmonic phasors stay mutually consistent — the frame a meter
    locked to the reference bus' fundamental PLL reports. The fundamental angle at each
    reference row becomes ``0``; the rotation has unit modulus, so magnitudes are untouched.

    This is what makes states from different grids comparable: shifting a whole grid's time
    reference — order ``h`` rotated by ``exp(1j*h*alpha)``, which for a fundamental-only
    state is a plain global rotation ``exp(1j*alpha)`` — leaves the result UNCHANGED. A
    per-source angle convention (the EMF angle a grid was solved against, a clock a converter
    folded in) is exactly such a shift, so it can no longer separate two members that
    describe the same physical state. A UNIFORM rotation of every order
    (``v·exp(1j*alpha)``) is a different transformation and is invariant only at ``h = 1``,
    so an invariance check on a harmonic state must build the shift per order.

    The rotation is a per-row diagonal ``D(h)`` (rows of different phases reference
    different rows), so it does NOT commute with ``Y(h)``: a nodal injection or residual is
    computed from the ABSOLUTE-frame state and the resulting current rotated afterwards with
    the same ``D``; ``Y`` applied to an already-rotated state yields injections that are not
    the grid's (see :func:`slack_frame_rotation`). On a single-phase grid every row shares
    one reference row, ``D`` is a scalar per order and the difference vanishes, which is why
    a frame test needs a three-phase grid.

    Parameters
    ----------
    v:
        Complex ``[*lead, H, N]`` state (volts or per-unit — the frame is scale-free).
    reference_rows:
        int64 ``[N]`` reference row of each row, member-local for one grid or GLOBAL for a
        collated multi-grid batch (``batch["reference_rows"]``). Moved to ``v``'s device.
    orders:
        The ``H`` solved harmonic orders, aligned with ``v``'s harmonic axis. Must contain
        the fundamental ``1`` — the frame is defined by the fundamental phasor.

    Returns
    -------
    torch.Tensor
        The rotated state, same shape, device and dtype as ``v``.

    Raises
    ------
    ValueError
        If ``1`` is not among ``orders``, or the shapes of ``orders`` / ``reference_rows`` do
        not match ``v``.
    """
    return v * slack_frame_rotation(v, reference_rows, orders=orders)


def build_multigrid_source(
    cfg, *, root: Optional[str] = None, orders: Optional[list[int]] = None
) -> MultiGridDataSource:
    """Construct a :class:`MultiGridDataSource` from a :class:`~injection_hse.config.DataConfig`.

    ``cfg.dataset_path`` (or the explicit ``root``) is the multi-grid dataset root;
    ``cfg.multigrid`` supplies the loader knobs — including the TASK (which per-member dataset
    to open) and the ANGLE FRAME, so the batches a run trains on carry the reference rows its
    training step expects — and ``cfg.structural_encoding`` / ``cfg.phase_coupling`` the
    graph's node encoding and edge set. Called by
    :func:`injection_hse.data.build_data_source` whenever ``cfg.multigrid`` is set.

    ``orders`` (the experiment's :attr:`~injection_hse.config.ExperimentConfig.orders`) trains on that
    SUBSET of the corpus' generated orders; ``None`` (default) takes every order the corpus
    carries. The split fractions and the split seed of :class:`~injection_hse.config.DataConfig` are
    deliberately NOT forwarded — a multi-grid run splits by GRID, following the corpus
    manifest (the config validator rejects a non-default value there rather than ignoring it).
    """
    path = root if root is not None else cfg.dataset_path
    if not path:
        raise ValueError(
            "a multi-grid run requires data.dataset_path (the dataset root)."
        )
    multigrid = cfg.multigrid
    return MultiGridDataSource(
        path,
        task=multigrid.task,
        orders=orders,
        encoding=cfg.structural_encoding,
        coupling=cfg.phase_coupling,
        placement_fraction=multigrid.placement_fraction,
        placement_min_per_feeder=multigrid.placement_min_per_feeder,
        max_grids=multigrid.max_grids,
        keep_failed=cfg.keep_failed,
        per_unit=multigrid.per_unit,
        angle_reference=multigrid.angle_reference,
    )


__all__ = [
    "GridBundle",
    "MultiGridDataSource",
    "MultiGridCollate",
    "GraphBatchMasker",
    "segment_random_mask",
    "placement_mask",
    "per_unit_state",
    "physical_state",
    "slack_frame_rotation",
    "slack_reference_rows",
    "reference_bus_rows",
    "slack_referenced_state",
    "build_multigrid_source",
    "ANGLE_FRAMES",
    "TASKS",
    "DEFAULT_OBSERVED_FRACTION",
]
