"""Data sources: pgml datasets -> model-ready train/val ``torch`` datasets.

- :class:`DataSource` — abstract supplier of train/val ``torch`` datasets (the trainer is
  source-agnostic). Three implementations:

  - :class:`ParquetDataset` — offline datasets written by ``pgml.scenarios.write_dataset`` /
    reloaded by ``read_dataset`` (the bulk path; generate once on the cluster).
  - :class:`TopologyDataSource` — one such dataset per switch configuration of a
    :class:`~injection_hse.data.topology.TopologySet`, concatenated and tagged with the variant each
    sample was generated on (the switchable-topology path).
  - :class:`OnlineScenarioSource` — fresh batches from ``pgml.scenarios.run_scenarios`` (the
    hook for future hard-example / active-learning augmentation).

  A fourth lives in :mod:`injection_hse.data.multigraph`: :class:`~injection_hse.data.MultiGridDataSource`, one
  dataset per MEMBER GRID of a multi-grid manifest, split by grid rather than by scenario.

Each sample's target is the complex state ``V`` ``[H, N]`` (or ``[T, H, N]`` for a sequence);
the per-grid graph is shared across samples. Encoding (:mod:`injection_hse.encoding`) and normalization
(:mod:`injection_hse.normalization`) are applied in the training step, not here.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

from pgml.assembly import node_phase_index

from .graph import build_graph
from .physics import InjectionSpec

if TYPE_CHECKING:  # heavy / optional imports kept out of the import path
    import torch_geometric.data as pyg
    from torch.utils.data import Dataset

    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import PhaseCouplingConfig

    from .topology import TopologySet

_log = logging.getLogger(__name__)

#: Item key of a scenario's OWN step time axis ``[T]`` (a dataset with a start per
#: sequence); :class:`StateCollate` stacks it into the batch's ``"time_unix_s"`` ``[B, T]``.
_SEQUENCE_TIME_KEY = "sequence_time_unix_s"


@dataclass(frozen=True)
class ProfileFactors:
    """Ground-truth per-device load-profile factors of a PROFILED coherent dataset.

    The multiplicative fundamental profile factor :mod:`injection_hse.data.scenarios.profiles` applied to
    each device per step (recorded in the dataset as ``<name>_profile_factor``), exposed for
    profile-attribution studies (which device's activity drove a given step). Present only
    when the coherent config carried a ``LoadProfileConfig`` — see
    :meth:`DataSource.profile_factors`.

    Attributes
    ----------
    factor:
        ``[B, n_dev, T]`` float multiplicative factor per profiled device per step.
    device_ids:
        int64 ``[n_dev]`` profiled device ids (in selector order — the same order as the
        ``factor`` device axis).
    time_unix_s:
        float64 absolute epoch seconds per step: ``[T]`` (one window shared by every
        scenario) or ``[B, T]`` (a start per scenario).
    """

    factor: Tensor
    device_ids: Tensor
    time_unix_s: Tensor


@dataclass(frozen=True)
class CompositionLabels:
    """Ground-truth device-class attribution of a COMPOSED coherent dataset.

    A dataset generated with a :class:`injection_hse.data.scenarios.CompositionConfig` decomposes each
    covered aggregated load into statistical member devices, and records the per-step,
    per-class contribution so an estimator can be scored on (or, later, supervised toward)
    attributing an observed aggregate spectrum to a device mix. This is the pgl-side accessor
    for those labels — see :meth:`DataSource.composition_labels`. Present only when the
    coherent config carried a ``composition``. No model consumes it yet; it is the plumbing +
    ground truth.

    Attributes
    ----------
    class_p_w:
        ``[B, n_agg, n_class, T]`` float — signed per-class active-power contribution per
        covered load per step (the attribution label; may go net-negative under PV).
    class_active:
        ``[B, n_agg, n_class, T]`` int64 — active member count per class per covered load
        per step.
    cap_binding:
        ``[B, n_agg, n_ord, T]`` float ``0/1`` — where the composition's ``max_injection_pu``
        cap clipped the relative injection magnitude (a diagnostic on the residual-THD floor).
    agg_ids:
        ``[n_agg]`` int64 — the covered (composed) load ids, in aggregation order (the
        ``n_agg`` axis order).
    roster_p_rated:
        ``[n_agg, n_class, max_count]`` float — the per-member rated powers (0-padded roster
        sidecar; ``max_count`` = the largest member count over all (load, class) pairs).
    class_names:
        ``n_class`` device-class names, ordered as the ``n_class`` axis of ``class_p_w`` /
        ``class_active`` (from ``config.composition.class_names()``).
    time_unix_s:
        float64 absolute epoch seconds per step, ``[T]`` or ``[B, T]`` (a start per
        scenario); ``None`` only if the config carried no time anchor, which a composition
        always does.
    """

    class_p_w: Tensor
    class_active: Tensor
    cap_binding: Tensor
    agg_ids: Tensor
    roster_p_rated: Tensor
    class_names: list
    time_unix_s: Optional[Tensor]


def _derive_time_unix_s(config, n_steps: int) -> Optional[Tensor]:
    """Absolute per-step epoch axis of a coherent config, or ``None`` when it is timeless.

    A COMPOSED coherent dataset (a :class:`injection_hse.data.scenarios.CompositionConfig`) is
    time-stamped — its activity model is diurnal — but one written before the generator
    recorded ``time_unix_s`` carries only the per-class attribution samples. The absolute
    axis is fully determined by the config (``start_time`` or the seeded
    ``start_time_range`` draw, plus ``step_size_s``), so it is reconstructed here by
    :func:`injection_hse.data.scenarios.sequence_time_axis` (naive timestamps are UTC): ``[T]`` for
    one shared window, ``[B, T]`` for a start per sequence. ``None`` when the config
    carries neither (a fingerprint-only sequence — timeless).
    """
    if getattr(config, "start_time", None) is None and (
        getattr(config, "start_time_range", None) is None
    ):
        return None
    from injection_hse.data.scenarios import sequence_time_axis

    axis = sequence_time_axis(config)
    if axis is not None and int(axis.shape[-1]) != int(n_steps):
        raise ValueError(
            f"the config's time axis has {int(axis.shape[-1])} steps, the data {n_steps}."
        )
    return axis


# =============================================================================
# torch Dataset over a loaded scenario batch
# =============================================================================
class _StateDataset:
    """A ``torch.utils.data.Dataset`` over a contiguous slice of solved scenarios.

    ``__getitem__(i)`` returns ``{"v": complex state, "index": int64 scalar}`` plus any
    realized per-scenario sampled inputs ``{name: Tensor}`` (sequence / aux supervision). The
    state is ``[H, N]`` for a snapshot dataset (``T=1``) or ``[T, H, N]`` for a SEQUENCE
    dataset (``T>1``) — the time axis is preserved so a temporal model sees the whole
    scenario; the index is the SCENARIO (``B``-axis) id. The shared per-grid graph is NOT
    returned per item (it would be duplicated) — the trainer fetches it once via
    :meth:`DataSource.graph`.

    A time-stamped dataset adds its absolute step axis ``[T]``: under ``"time_unix_s"`` when
    one window is shared by every scenario, under ``"sequence_time_unix_s"`` when every
    scenario has its own (a ``start_time_range`` dataset, stored ``[B, T]``) —
    :class:`StateCollate` carries the first once and stacks the second.
    """

    def __init__(
        self,
        v: Tensor,
        samples: dict,
        row_idx: Tensor,
        injection_target: Optional[Tensor] = None,
        time_unix_s: Optional[Tensor] = None,
    ) -> None:
        # v: complex [B, H, N] (snapshot) or [B, T, H, N] (sequence); row_idx: int64 [n_sel]
        # global SCENARIO indices in this split. injection_target: [B, (T,) D, O, 1] (or None);
        # it carries the same (T,) leading axis as v, so the per-scenario slice keeps time.
        # time_unix_s: absolute epoch seconds of a time-stamped coherent dataset, [T] shared by
        # every scenario or [B, T] per scenario, attached per item so the batch carries the
        # step time axis; None otherwise.
        self._v = v
        self._samples = samples
        self._rows = row_idx
        self._injection = injection_target
        self._time_unix_s = time_unix_s

    def __len__(self) -> int:
        return int(self._rows.numel())

    def __getitem__(self, i: int) -> dict:
        g = int(self._rows[i])
        item: dict = {"v": self._v[g], "index": torch.tensor(g, dtype=torch.int64)}
        for name, t in self._samples.items():
            if name == "time_unix_s":
                continue  # the step time axis travels on its own key below
            if t.ndim >= 1 and t.shape[0] == self._v.shape[0]:
                item[f"sample_{name}"] = t[g]
        if self._injection is not None:
            item["injection_target"] = self._injection[g]  # [(T,) D, O, 1]
        if self._time_unix_s is not None:
            if self._time_unix_s.ndim == 2:
                item[_SEQUENCE_TIME_KEY] = self._time_unix_s[g]  # [T] (this scenario's)
            else:
                item["time_unix_s"] = self._time_unix_s  # [T] (shared across scenarios)
        return item


def _split_indices(
    n: int,
    val_fraction: float,
    seed: int,
    *,
    test_fraction: float = 0.0,
    drop: Sequence[int] = (),
) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic train/val/test split of ``range(n)`` (permute by ``seed``).

    ``n`` is the number of SCENARIOS (the ``B`` axis) — splitting by scenario keeps all of a
    scenario's correlated time steps on the SAME side, so adjacent steps never leak between
    the splits (essential once ``T>1`` sequences are loaded whole). The three splits are
    disjoint slices of ONE seeded permutation, in the order ``val | test | train``, so a
    ``test_fraction`` of ``0.0`` leaves the ``val | train`` partition byte-identical to a
    two-way split (the permutation draw is unchanged and the leading ``val`` slice is unchanged).

    ``drop`` lists scenario ids to remove from the permutation BEFORE partitioning (the
    non-converged scenarios): the fractions then apply to the surviving scenarios, so an
    empty ``drop`` reproduces the undropped split exactly. Each returned tensor is a sorted
    int64 CPU tensor of global scenario indices.
    """
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)
    if len(drop):
        drop_mask = torch.zeros(n, dtype=torch.bool)
        drop_mask[
            torch.tensor(sorted(set(int(i) for i in drop)), dtype=torch.int64)
        ] = True
        perm = perm[~drop_mask[perm]]  # keep permutation order, minus the dropped ids
    m = int(perm.numel())
    n_val = int(round(val_fraction * m))
    n_test = int(round(test_fraction * m))
    val = perm[:n_val].sort().values
    test = perm[n_val : n_val + n_test].sort().values
    train = perm[n_val + n_test :].sort().values
    return train, val, test


# =============================================================================
# Data sources
# =============================================================================
class DataSource(ABC):
    """Supplies train/val datasets of (state target + per-grid graph + sampled inputs)."""

    @abstractmethod
    def train_dataset(self) -> "Dataset": ...

    @abstractmethod
    def val_dataset(self) -> "Dataset": ...

    def test_dataset(self) -> "Dataset":
        """The held-out TEST split (empty by default).

        Populated only when the source was built with ``test_fraction > 0``. It is NEVER
        consumed by training, early stopping, checkpoint selection, or the curriculum — it
        exists solely for a final one-shot generalization estimate. The default returns an
        empty dataset so a source that carves no test split still satisfies the contract.
        """
        return _StateDataset(
            torch.zeros(0), {}, torch.zeros(0, dtype=torch.int64), None
        )

    @abstractmethod
    def graph(self) -> "pyg.Data":
        """The (shared) network-structure input for the grid these samples come from."""

    def collate(self):
        """The DataLoader ``collate_fn`` this source's items are batched with.

        :class:`StateCollate` for every fixed-topology source (training and evaluation share
        it, so the batch layout cannot drift). A source whose items come from DIFFERENT
        graphs overrides it — see :class:`injection_hse.data.MultiGridCollate`, which folds the sample
        axis into the node axis instead of stacking it.
        """
        return StateCollate()

    @property
    @abstractmethod
    def orders(self) -> list[int]:
        """The solved harmonic orders ``H`` (for the loss/normalizer/metrics)."""

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Node-phase rows ``N`` (the fixed DNN width / mask length)."""

    @property
    def n_steps(self) -> int:
        """Time steps ``T`` per scenario (``1`` for a snapshot dataset; ``>1`` is a sequence)."""
        return 1

    @property
    def injection_spec(self) -> Optional["InjectionSpec"]:
        """The per-device injection target spec (``None`` if the source carries no injections)."""
        return None

    @property
    def time_unix_s(self) -> Optional[Tensor]:
        """Absolute per-step epoch seconds (``None`` unless a time-stamped coherent dataset).

        Present when the coherent dataset was generated with an absolute time anchor (a
        :class:`~injection_hse.data.scenarios.LoadProfileConfig` or a composition); ``[T]`` when every
        scenario shares one window, ``[B, T]`` (indexed by the scenario id) when each has its
        own start (``CoherentSpectrumConfig.start_time_range``). The periodic time features
        (:mod:`injection_hse.time_features`) are derived from it when ``encoding.time_features`` is
        enabled.
        """
        return None

    def profile_factors(self) -> Optional["ProfileFactors"]:
        """Ground-truth per-device load-profile factors (``None`` unless a PROFILED dataset).

        The multiplicative fundamental profile factor applied per device per step, exposed for
        profile-attribution studies; see :class:`ProfileFactors`. Default ``None``.
        """
        return None

    def composition_labels(self) -> Optional["CompositionLabels"]:
        """Ground-truth device-class attribution (``None`` unless a COMPOSED dataset).

        The per-step, per-class active power / active-member count of a dataset generated
        with a :class:`injection_hse.data.scenarios.CompositionConfig`, exposed for device-attribution
        studies; see :class:`CompositionLabels`. Default ``None`` (no model consumes it yet).
        """
        return None

    @property
    @abstractmethod
    def grid(self) -> "Grid":
        """The grid the samples were generated on (for measurement-row resolution)."""


def _select_orders(
    v: Tensor, stored: Optional[list[int]], requested: list[int], where: str
) -> tuple[list[int], Tensor]:
    """Restrict a stored state to the REQUESTED harmonic orders.

    ``v`` is ``[B, (T,) H, N]`` and ``stored`` the orders its harmonic axis carries (``None``
    when the dataset records no frequencies). The requested orders must be a SUBSET of the
    stored ones — training on a subset of a dataset's spectrum is a legitimate experiment,
    inventing an order it does not contain is not. Returns ``(orders, state)`` with the
    harmonic axis selected in the requested order; an exact match is returned untouched.
    """
    if stored is None:
        # no frequency metadata: the labels are taken on trust, but they must at least
        # describe the stored harmonic axis.
        if len(requested) != int(v.shape[-2]):
            raise ValueError(
                f"dataset {where} records no frequencies and carries {int(v.shape[-2])} "
                f"harmonic row(s), so it cannot be read as orders {requested}."
            )
        return requested, v
    missing = [o for o in requested if o not in stored]
    if missing:
        raise ValueError(
            f"dataset {where} was generated for orders {stored}; it does not contain the "
            f"requested order(s) {missing}. Train on a subset of the generated orders, or "
            "regenerate the dataset."
        )
    if requested == stored:
        return requested, v
    pos = torch.tensor([stored.index(o) for o in requested], dtype=torch.int64)
    return requested, v.index_select(-2, pos)


class ParquetDataset(DataSource):
    """Offline source: a dataset directory written by ``pgml.scenarios.write_dataset``.

    Reloads via ``injection_hse.data.scenarios.read_dataset`` -> complex ``v`` (targets) + the realized
    sampled inputs (sequence / aux supervision) + the node-phase index + the config (for
    reproducibility). A snapshot dataset (``[B, H, N]``) yields one ``[H, N]`` state per item;
    a SEQUENCE dataset (``[B, T, H, N]``, ``T>1``) is KEPT whole — ``__getitem__`` returns the
    scenario's ``[T, H, N]`` sequence (time preserved for a temporal model). ``T`` is
    auto-detected; the ``T=1`` path is unchanged.

    The train/val/test split is by SCENARIO (the ``B`` axis), so all of a scenario's
    correlated time steps stay on one side (no leakage between splits). ``seed`` is the
    SPLIT seed (``DataConfig.split_seed``, separate from the experiment seed so replicate
    runs and a checkpoint-reload evaluation share one partition); ``test_fraction > 0``
    carves the held-out test split. Non-converged ``failed_scenarios`` are dropped from
    every split unless ``keep_failed`` (the count is ``n_dropped_failed``).

    An item is a dict with ``"v"`` (``[H, N]`` or ``[T, H, N]``), ``"index"`` (the scenario
    id), the ``"sample_*"`` inputs, ``"injection_target"`` (``[(T,) D, O, 1]``) when the
    dataset injects, and the step time axis of a profiled or composed coherent dataset;
    :class:`StateCollate` carries a shared ``[T]`` axis once rather than stacking it and
    stacks per-scenario axes (a start per sequence) to ``[B, T]``.

    ``orders`` selects WHICH of the generated harmonic orders the source serves: ``None``
    (default) takes every order the dataset stores, a list takes that SUBSET (the state's
    harmonic axis is selected accordingly). An order the dataset does not contain is an
    error naming it — the harmonic axis is the model's input width, so a silent mismatch
    would mislabel every order.

    The dataset directory stores the node-phase ROW LAYOUT but not the full ``Grid`` (the
    grid is the generator's input, not the result). The grid is therefore supplied
    explicitly (``grid=``) for the graph / measurement model — the grid the dataset was
    generated on, as for :func:`build_data_source` and :func:`injection_hse.train.train_experiment`.
    Its ``node_phase_index`` is asserted to match the dataset's stored layout, node ids AND
    phase codes (a row is a ``(node, phase)`` pair), and a mismatch raises ``ValueError``.

    ``coupling`` (a :class:`~injection_hse.config.PhaseCouplingConfig`, ``None`` = off) selects the
    graph's edge set: the series-only default, or the one that also carries the network's
    phase-to-phase admittance couplings.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        grid: "Grid",
        val_fraction: float = 0.1,
        seed: int = 0,
        orders: Optional[list[int]] = None,
        test_fraction: float = 0.0,
        keep_failed: bool = False,
        coupling: Optional["PhaseCouplingConfig"] = None,
    ) -> None:
        from injection_hse.data.scenarios import read_dataset

        self.dataset_path = dataset_path
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.keep_failed = keep_failed
        self._grid = grid
        self._coupling = coupling

        loaded = read_dataset(dataset_path)
        v = loaded.v
        if v.ndim == 2:  # [B, N] power flow -> insert a singleton harmonic axis
            v = v.unsqueeze(1)  # [B, H=1, N]
        if v.ndim not in (3, 4):
            raise ValueError(
                f"expected [B, H, N] or [B, T, H, N] state, got shape {tuple(v.shape)}."
            )
        # T is auto-detected: a 4-D [B, T, H, N] sequence is KEPT (time preserved), a 3-D
        # [B, H, N] snapshot stays T=1. The B (scenario) axis is what the split uses.
        self._has_time = v.ndim == 4
        self._n_steps = int(v.shape[1]) if self._has_time else 1
        self._v = v
        self._samples = loaded.samples
        n_rows = int(v.shape[-1])

        # Absolute per-step epoch seconds + ground-truth attribution of a time-stamped
        # coherent dataset. A PROFILED dataset (a ``LoadProfileConfig``) stamps ``time_unix_s``
        # + ``<name>_profile_factor`` directly; a COMPOSED dataset (a ``CompositionConfig``)
        # records the per-class attribution samples but NOT ``time_unix_s`` — its absolute
        # step axis is derived from the config's ``start_time`` + ``step_size_s`` below (both
        # paths agree with :func:`injection_hse.data.scenarios.profiles._time_axis`). A snapshot /
        # fingerprint-only sequence carries neither (time features are then all-zero).
        self._time_unix_s: Optional[Tensor] = self._samples.get("time_unix_s")
        if self._time_unix_s is None and self._has_time:
            self._time_unix_s = _derive_time_unix_s(loaded.config, self._n_steps)
        self._profile_factors = self._extract_profile_factors()
        self._composition_labels = self._extract_composition_labels(loaded.config)

        idx = node_phase_index(grid)
        if int(idx.size) != n_rows or not torch.equal(idx.node_ids, loaded.node_ids):
            raise ValueError(
                "grid node_phase_index does not match the dataset row layout "
                f"(grid N={idx.size}, dataset N={n_rows}); pass the grid the "
                "dataset was generated on."
            )
        # node_ids alone do not pin a row: a row is a (node, phase) pair. Reject a phase-code
        # mismatch too (e.g. a grid whose per-node phase order differs from the stored layout),
        # otherwise the state tensor's phase axis would be silently misaligned to the graph.
        if not torch.equal(idx.phase_codes, loaded.phase_codes):
            raise ValueError(
                "grid node_phase_index phase codes do not match the dataset row layout "
                "(same nodes, different per-row phase assignment); pass the grid the "
                "dataset was generated on."
            )

        freqs = loaded.frequencies_hz
        f0 = float(grid.base_frequency_hz)
        stored_orders = (
            [int(round(float(fz) / f0)) for fz in freqs] if freqs is not None else None
        )
        if orders is not None:
            self._orders, self._v = _select_orders(
                self._v, stored_orders, [int(o) for o in orders], dataset_path
            )
        elif stored_orders is not None:
            self._orders = stored_orders
        else:
            self._orders = [1]

        # the per-device injection target (the localization label), assembled from the
        # dataset's REALIZED harmonic injection (uniform for snapshot + coherent); empty when
        # the dataset carries none. Built once ([*B, (T,) D, O, 1]) and sliced per scenario.
        self._injection_spec = InjectionSpec(grid, loaded.config)
        self._injection_target: Optional[Tensor] = None
        if not self._injection_spec.is_empty:
            self._injection_target = self._injection_spec.assemble_target()

        # Non-converged scenarios carry best-effort (non-solution) voltages; drop them from
        # every split unless explicitly kept, so no split trains/validates/tests on a
        # non-solution. A dataset with no failures (or written before convergence metadata
        # existed) is unaffected. Report the count so the drop is visible.
        drop: tuple[int, ...] = ()
        self.n_dropped_failed = 0
        if not keep_failed and loaded.failed_scenarios:
            drop = tuple(
                int(i) for i in loaded.failed_scenarios if 0 <= int(i) < v.shape[0]
            )
            self.n_dropped_failed = len(drop)
            if drop:
                _log.info(
                    "ParquetDataset(%s): dropped %d non-converged scenario(s) of %d from the "
                    "train/val/test splits (keep_failed=False).",
                    dataset_path,
                    self.n_dropped_failed,
                    int(v.shape[0]),
                )

        # split by SCENARIO (the leading B axis), never by step
        train_idx, val_idx, test_idx = _split_indices(
            int(v.shape[0]),
            val_fraction,
            seed,
            test_fraction=test_fraction,
            drop=drop,
        )
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx

    def train_dataset(self) -> "Dataset":
        return _StateDataset(
            self._v,
            self._samples,
            self._train_idx,
            self._injection_target,
            self._time_unix_s,
        )

    def val_dataset(self) -> "Dataset":
        return _StateDataset(
            self._v,
            self._samples,
            self._val_idx,
            self._injection_target,
            self._time_unix_s,
        )

    def test_dataset(self) -> "Dataset":
        """The held-out TEST split (empty unless built with ``test_fraction > 0``).

        Disjoint by scenario from train and val; never read by training. See
        :class:`~injection_hse.config.DataConfig` ``test_fraction``.
        """
        return _StateDataset(
            self._v,
            self._samples,
            self._test_idx,
            self._injection_target,
            self._time_unix_s,
        )

    def _extract_profile_factors(self) -> Optional[ProfileFactors]:
        """Collect the ``<name>_profile_factor`` ground truth from the loaded samples.

        A PROFILED coherent dataset stores ``<name>_profile_factor`` ``[B, n_dev, T]`` +
        ``<name>_profile_device_ids`` ``[n_dev]`` (keyed by the config name) alongside
        ``time_unix_s``. Returns the assembled :class:`ProfileFactors` if all three are
        present, else ``None``.
        """
        factor_key = next(
            (k for k in self._samples if k.endswith("_profile_factor")), None
        )
        if factor_key is None or self._time_unix_s is None:
            return None
        ids_key = factor_key[: -len("_profile_factor")] + "_profile_device_ids"
        device_ids = self._samples.get(ids_key)
        if device_ids is None:
            return None
        return ProfileFactors(
            factor=self._samples[factor_key],
            device_ids=device_ids,
            time_unix_s=self._time_unix_s,
        )

    def _extract_composition_labels(self, config) -> Optional[CompositionLabels]:
        """Collect the device-class attribution ground truth from the loaded samples.

        A COMPOSED coherent dataset stores ``<name>_class_p_w`` ``[B, n_agg, n_class, T]`` +
        ``<name>_class_active`` + ``<name>_cap_binding`` + ``<name>_agg_ids`` +
        ``<name>_roster_p_rated`` (keyed by the config name), and the ``n_class`` ordering
        lives on the config (``composition.class_names()``). Returns the assembled
        :class:`CompositionLabels` when the samples are present, else ``None``.
        """
        p_key = next((k for k in self._samples if k.endswith("_class_p_w")), None)
        if p_key is None:
            return None
        nm = p_key[: -len("_class_p_w")]
        need = ("_class_active", "_cap_binding", "_agg_ids", "_roster_p_rated")
        if any(nm + s not in self._samples for s in need):
            return None
        comp = getattr(config, "composition", None)
        class_names = list(comp.class_names()) if comp is not None else []
        return CompositionLabels(
            class_p_w=self._samples[nm + "_class_p_w"],
            class_active=self._samples[nm + "_class_active"],
            cap_binding=self._samples[nm + "_cap_binding"],
            agg_ids=self._samples[nm + "_agg_ids"],
            roster_p_rated=self._samples[nm + "_roster_p_rated"],
            class_names=class_names,
            time_unix_s=self._time_unix_s,
        )

    def graph(self) -> "pyg.Data":
        return build_graph(self._grid, orders=self._orders, coupling=self._coupling)

    @property
    def states(self) -> Tensor:
        """The stored complex state of EVERY scenario, ``[B, (T,) H, N]`` (unsplit).

        The raw reloaded tensor behind the splits — read by consumers that fit statistics
        over the dataset (e.g. a multi-grid normalizer sample) rather than iterate items.
        """
        return self._v

    @property
    def orders(self) -> list[int]:
        return self._orders

    @property
    def n_nodes(self) -> int:
        return int(self._v.shape[-1])

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def injection_spec(self) -> "InjectionSpec":
        return self._injection_spec

    @property
    def time_unix_s(self) -> Optional[Tensor]:
        return self._time_unix_s

    def profile_factors(self) -> Optional[ProfileFactors]:
        return self._profile_factors

    def composition_labels(self) -> Optional[CompositionLabels]:
        return self._composition_labels

    @property
    def grid(self) -> "Grid":
        return self._grid


class OnlineScenarioSource(DataSource):
    """Online source: fresh batches from ``pgml.scenarios.run_scenarios`` (one config).

    The hook for future hard-example augmentation — a policy that biases the scenario config
    toward states where the estimator struggles (active learning). Generates on the same
    device as training. The graph and the source properties work; ``train_dataset`` /
    ``val_dataset`` raise ``NotImplementedError`` (a planned extension).
    """

    def __init__(
        self,
        scenario_config_path: str,
        *,
        grid: "Grid",
        orders: list[int],
        coupling: Optional["PhaseCouplingConfig"] = None,
    ) -> None:
        self.scenario_config_path = scenario_config_path
        self._grid = grid
        self._orders = list(orders)
        self._coupling = coupling

    def train_dataset(self) -> "Dataset":
        raise NotImplementedError("OnlineScenarioSource is a planned extension.")

    def val_dataset(self) -> "Dataset":
        raise NotImplementedError("OnlineScenarioSource is a planned extension.")

    def graph(self) -> "pyg.Data":
        return build_graph(self._grid, orders=self._orders, coupling=self._coupling)

    @property
    def orders(self) -> list[int]:
        return self._orders

    @property
    def n_nodes(self) -> int:
        return node_phase_index(self._grid).size

    @property
    def grid(self) -> "Grid":
        return self._grid


class _TopologyTagged:
    """Concatenation of the per-variant datasets, tagging each item with its variant index.

    The variant index is what selects a sample's physics operators (admittance, branch-current
    map) in the training step and the topology channels the model conditions on.
    """

    def __init__(self, datasets: Sequence["Dataset"]) -> None:
        self._datasets = list(datasets)
        self._sizes = [len(d) for d in self._datasets]
        self._offsets = list(itertools.accumulate(self._sizes, initial=0))

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> dict:
        k = bisect.bisect_right(self._offsets, i) - 1
        item = self._datasets[k][i - self._offsets[k]]
        item["topology"] = torch.tensor(k, dtype=torch.int64)
        return item


class TopologyDataSource(DataSource):
    """One dataset per switch configuration, presented as ONE source.

    Each variant of a :class:`~injection_hse.data.topology.TopologySet` was generated on the same grid
    with different switch states, so the variants share the row layout, the harmonic orders
    and the union graph; only the physics differs. Their datasets live side by side under one
    root (``<root>/<variant name>``), are split with the SAME seed and fractions — so every
    split covers every topology — and are concatenated here, each item tagged with its
    variant index.

    The graph is built from the grid AS GIVEN (every switchable branch closed = the union
    topology), which is the layout every variant's operators are built on.
    """

    def __init__(
        self,
        root: str,
        *,
        grid: "Grid",
        topology: "TopologySet",
        val_fraction: float = 0.1,
        seed: int = 0,
        orders: Optional[list[int]] = None,
        test_fraction: float = 0.0,
        keep_failed: bool = False,
        coupling: Optional["PhaseCouplingConfig"] = None,
    ) -> None:
        self.root = str(root)
        self.topology = topology
        # the UNION topology defines the layout the model is built on (graph edge set,
        # half-edges, placement); the variants only open switches within it.
        grid = topology.union_grid(grid)
        self._grid = grid
        self._sources = [
            ParquetDataset(
                str(topology.dataset_path(root, k)),
                grid=grid,
                val_fraction=val_fraction,
                seed=seed,
                orders=orders,
                test_fraction=test_fraction,
                keep_failed=keep_failed,
                coupling=coupling,
            )
            for k in range(topology.n_variants)
        ]
        first = self._sources[0]
        for name, source in zip(topology.names[1:], self._sources[1:]):
            if source.orders != first.orders or source.n_steps != first.n_steps:
                raise ValueError(
                    f"topology variant {name!r} was generated with orders "
                    f"{source.orders} / {source.n_steps} steps, variant "
                    f"{topology.names[0]!r} with {first.orders} / {first.n_steps}; "
                    "generate every variant from the same scenario configuration."
                )

    def train_dataset(self) -> "Dataset":
        return _TopologyTagged([s.train_dataset() for s in self._sources])

    def val_dataset(self) -> "Dataset":
        return _TopologyTagged([s.val_dataset() for s in self._sources])

    def test_dataset(self) -> "Dataset":
        return _TopologyTagged([s.test_dataset() for s in self._sources])

    def graph(self) -> "pyg.Data":
        return self._sources[0].graph()

    @property
    def orders(self) -> list[int]:
        return self._sources[0].orders

    @property
    def n_nodes(self) -> int:
        return self._sources[0].n_nodes

    @property
    def n_steps(self) -> int:
        return self._sources[0].n_steps

    @property
    def injection_spec(self) -> "InjectionSpec":
        # the injecting device set follows the grid + scenario configuration, which every
        # variant shares — the switch states do not change it.
        return self._sources[0].injection_spec

    @property
    def time_unix_s(self) -> Optional[Tensor]:
        return self._sources[0].time_unix_s

    @property
    def grid(self) -> "Grid":
        return self._grid

    @property
    def sources(self) -> list["ParquetDataset"]:
        """The per-variant sources, in variant order (for per-topology evaluation)."""
        return list(self._sources)


class StateCollate:
    """Collate scenario dicts into a batch (complex ``v`` stacked over the batch axis).

    The DataLoader ``collate_fn`` for every :class:`DataSource` dataset — training
    and evaluation share it so the batch layout cannot drift between the two. The batch's
    ``"time_unix_s"`` is ``[T]`` when the dataset shares one window across its scenarios and
    ``[B, T]`` when every scenario has its own start.
    """

    def __call__(self, items: list[dict]) -> dict:
        out: dict = {"v": torch.stack([it["v"] for it in items])}
        if items and "index" in items[0]:
            out["index"] = torch.stack([it["index"] for it in items])
        if items and "injection_target" in items[0]:
            out["injection_target"] = torch.stack(
                [it["injection_target"] for it in items]
            )
        # the switch configuration each sample was generated on (switchable topology)
        if items and "topology" in items[0]:
            out["topology"] = torch.stack([it["topology"] for it in items])
        # the absolute step time axis: one window SHARED by every scenario is carried once
        # (``[T]``, not stacked over the batch); a per-scenario axis (a start per sequence)
        # is stacked to ``[B, T]``.
        if items and "time_unix_s" in items[0]:
            out["time_unix_s"] = items[0]["time_unix_s"]
        elif items and _SEQUENCE_TIME_KEY in items[0]:
            out["time_unix_s"] = torch.stack([it[_SEQUENCE_TIME_KEY] for it in items])
        for key in items[0]:
            if key.startswith("sample_"):
                out[key] = torch.stack([it[key] for it in items])
        return out


def build_data_source(
    cfg, *, grid: Optional["Grid"] = None, orders: Optional[list[int]] = None
) -> DataSource:
    """Construct a :class:`DataSource` from a :class:`~injection_hse.config.DataConfig`.

    A ``grid`` is REQUIRED (the dataset stores only the row layout, not the topology /
    parameters needed for the graph and measurement model). A config carrying a
    ``topology`` (:class:`~injection_hse.config.TopologyConfig`) reads ``dataset_path`` as the PARENT
    of one dataset directory per switch configuration and yields a
    :class:`TopologyDataSource`.

    A config carrying a ``multigrid`` (:class:`~injection_hse.config.MultiGridConfig`) reads
    ``dataset_path`` as the ROOT of a multi-grid dataset and yields a
    :class:`injection_hse.data.MultiGridDataSource`; no ``grid`` is required there, since a multi-grid
    dataset persists each member grid beside its own scenarios.

    ``orders`` (the experiment's :attr:`~injection_hse.config.ExperimentConfig.orders`) restricts the
    source to that SUBSET of the generated harmonic orders; ``None`` (default) serves every
    order the data carries, and an order the data does not contain raises naming it.
    """
    if cfg.multigrid is not None:
        from .multigraph import build_multigrid_source

        return build_multigrid_source(cfg, orders=orders)
    if grid is None:
        raise ValueError(
            "build_data_source requires the grid the dataset was generated on."
        )
    if cfg.source == "parquet":
        if not cfg.dataset_path:
            raise ValueError("source='parquet' requires data.dataset_path.")
        topology = topology_set(cfg)
        if topology is not None:
            return TopologyDataSource(
                cfg.dataset_path,
                grid=grid,
                topology=topology,
                val_fraction=cfg.val_fraction,
                seed=cfg.split_seed,
                orders=orders,
                test_fraction=cfg.test_fraction,
                keep_failed=cfg.keep_failed,
                coupling=cfg.phase_coupling,
            )
        return ParquetDataset(
            cfg.dataset_path,
            grid=grid,
            val_fraction=cfg.val_fraction,
            seed=cfg.split_seed,
            orders=orders,
            test_fraction=cfg.test_fraction,
            keep_failed=cfg.keep_failed,
            coupling=cfg.phase_coupling,
        )
    if cfg.source == "online":
        if not cfg.scenario_config_path:
            raise ValueError("source='online' requires scenario_config_path.")
        raise NotImplementedError("OnlineScenarioSource is a planned extension.")
    raise ValueError(f"unknown data source {cfg.source!r}.")


def topology_set(cfg) -> Optional["TopologySet"]:
    """The :class:`~injection_hse.data.topology.TopologySet` a :class:`~injection_hse.config.DataConfig` declares.

    ``None`` for a single-topology run — the common case, and what every consumer built
    before switchable topologies existed sees.
    """
    from .topology import TopologySet

    return None if cfg.topology is None else TopologySet.from_config(cfg.topology)


__all__ = [
    "DataSource",
    "ProfileFactors",
    "CompositionLabels",
    "ParquetDataset",
    "TopologyDataSource",
    "OnlineScenarioSource",
    "StateCollate",
    "build_data_source",
    "topology_set",
]
