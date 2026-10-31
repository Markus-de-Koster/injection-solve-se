"""Physics decoder: predicted nodal injection currents -> node voltages via the network law.

The injection-estimation state estimator predicts the (sparse, structured) nodal **injection
currents** ``I(h)`` — the net current injected *at* each node by its devices, NOT a branch /
line flow — and reconstructs the full voltage state with the KNOWN, differentiable grid:

    V(h) = solve(Y(h), I_pred(h))

the fundamental via an ideal-slack reduced solve held at the measured slack voltage, the
harmonics via the Norton-shunted harmonic system. The output satisfies the nodal network law
``Y(h)·V(h) = I(h)`` by construction — no ground-truth injection enters the RHS, so there is no
measurement leakage. Because ``Y(h)`` is a differentiable function of the grid description, the
same path yields grid-parameter sensitivities (``∂loss/∂Θ``) and, by unfreezing a few
parameters, a learned grid calibration.

``Y(h)`` is grid-constant, so it is assembled ONCE (``assemble``) and reused across the batch;
a learned-calibration run re-assembles it per step from the (perturbed) grid (the assembly is
differentiable, the solve reuses the result).

Across MANY grids (a ``data.multigrid`` run) there is no single operator:
:class:`MultiGridPhysicsBank` holds one :class:`NetworkSolver` per member grid of a
:class:`~injection_hse.data.MultiGridDataSource`, index-aligned with its bundles, plus the per-member
CURRENT BASE that makes injections comparable across voltage classes — the current analogue
of the per-unit voltage base (see :meth:`MultiGridPhysicsBank.i_base`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from injection_hse.train import SEModule


def reference_base_voltage(base_voltage: Tensor) -> Tensor:
    """0-dim float64 REFERENCE base voltage of a grid, from its per-row bases.

    A per-unit system fixes ONE power base per grid and derives every row's current base from
    it (``I_base = S_base/V_base``); this is the voltage that power base is referred to, so it
    sets the magnitude of :meth:`NetworkSolver.i_base` but never its per-row (voltage-class)
    dependence. It is the MEDIAN of the rated row voltages, chosen so that

    - a SINGLE-VOLTAGE-CLASS grid returns exactly that class, which makes the current base
      reduce bit-for-bit to ``s(h)·V_base`` there;
    - a MIXED-class grid returns the class of its middle row, so a feeder carrying a few
      substation-side rows keeps the same convention as a pure grid of the feeder's own class
      — the continuity a multi-grid normalizer pooled over many members needs;
    - the reference is a value the grid actually has (a median never interpolates), and it
      pairs with the MEDIAN singular value :meth:`NetworkSolver.operator_scale` returns: the
      grid's power base is ``S_base(h) = s(h)·V_ref²``, a typical admittance at a typical row.

    Rows with no rated voltage (``V_base <= 0``, kept unscaled by the callers) are excluded
    from the median; a grid with no rated row at all returns ``1``. Returns ``float64`` on the
    input's device, free of data-dependent shapes and host synchronisation (it runs inside the
    per-step feature-frame conversion), and off the autograd tape — the base is a unit
    convention, not a learnable quantity.
    """
    v_base = torch.as_tensor(base_voltage).to(torch.float64)
    rated = v_base > 0
    # sort the unrated rows to the end so the median is taken over the rated block alone
    inf = torch.full_like(v_base, float("inf"))
    ordered, _ = torch.sort(torch.where(rated, v_base, inf))
    lower_median = torch.clamp((rated.sum() - 1).div(2, rounding_mode="floor"), min=0)
    v_ref = ordered.index_select(0, lower_median.reshape(1)).reshape(())
    return torch.where(torch.isfinite(v_ref), v_ref, torch.ones_like(v_ref))


def _factored_nbytes(fac) -> int:
    """Bytes a :class:`pgml.solver.harmonic.FactoredSystem` holds (budget accounting)."""
    total = 0
    if fac.backend == "sparse":
        total += sum(int(lu.nnz) * 16 for lu in fac.sparse.lus)
    elif fac.backend == "dense":
        total += fac.lu.numel() * fac.lu.element_size()
        total += fac.piv.numel() * fac.piv.element_size()
    if fac.y_fs is not None:
        total += fac.y_fs.numel() * fac.y_fs.element_size()
    return total


class OperatorCacheBudget:
    """Shared byte budget of the factor-once operator caches (:class:`NetworkSolver`).

    One instance is shared by every solver that may cache (all members of a
    :class:`MultiGridPhysicsBank`, all switch configurations of a topology set), so the
    total held by ``Z = Y_ff^{-1}`` images, branch-current images and LU factors never
    exceeds the configured budget whatever the batch order builds first. Charging is
    first-come: a solver whose cache no longer fits falls back to the direct
    (factorize-per-call) solve, which is always available.
    """

    def __init__(self, budget_bytes: int) -> None:
        self.remaining = int(budget_bytes)

    def charge(self, nbytes: int) -> bool:
        """Reserve ``nbytes``; ``False`` (and no reservation) when the budget is spent."""
        if nbytes > self.remaining:
            return False
        self.remaining -= int(nbytes)
        return True

    def refund(self, nbytes: int) -> None:
        """Return a reservation (a cache invalidated by re-assembly or a device move)."""
        self.remaining += int(nbytes)


#: The harmonic device-shunt models the decode can assemble ``Y(h)`` with (pgml's
#: ``load_shunt`` names).
_LOAD_SHUNT_MODELS = ("none", "opendss", "motor")

#: The basis of the decode's device shunt. The decode solves a batch with ONE ``Y(h)`` per
#: order, so the shunt is built from the devices' stored P, Q at rated voltage, never from a
#: scenario's operating point.
_DECODE_LOAD_SHUNT_BASIS = "nameplate"


def _check_load_shunt(load_shunt: Optional[str]) -> None:
    """Raise ``ValueError`` unless ``load_shunt`` is ``None`` or a pgml shunt model name."""
    if load_shunt is not None and load_shunt not in _LOAD_SHUNT_MODELS:
        raise ValueError(
            f"load_shunt must be None or one of {_LOAD_SHUNT_MODELS}, got {load_shunt!r}."
        )


def _decode_shunt_kwargs(load_shunt: Optional[str]) -> dict:
    """The device-shunt keywords of the decode's ``assemble_harmonic_ybus`` call.

    ``None`` passes nothing (pgml's assembly defaults, the historical decode); a model name
    pins it on the nameplate basis.
    """
    if load_shunt is None:
        return {}
    return {"load_shunt": load_shunt, "load_shunt_basis": _DECODE_LOAD_SHUNT_BASIS}


#: A closed :class:`~pgml.schemas.grid_schema.Switch` at or below this longitudinal
#: resistance [ohm] (and without inductance) is an electrical TIE: the rows on both sides
#: are the same node for every purpose of the estimator (the schema's ``0`` is ideal).
TIE_SWITCH_MAX_OHM = 1e-2


def tied_rows_of(
    grid: "Grid", slack_rows: Tensor, branch_states: Optional[dict] = None
) -> Tensor:
    """int64 rows electrically identical to the slack — reached through closed switch ties.

    Walks from the slack rows over every in-service, closed
    :class:`~pgml.schemas.grid_schema.Switch` that is a tie (:data:`TIE_SWITCH_MAX_OHM`) and
    whose state no ``branch_states`` entry can change, phase by phase. An injection at such
    a row leaves through the tie into the slack and moves no voltage anywhere, so it is
    not an estimable quantity: the injection head predicts nothing there, the injection
    normalizer and the linear prior skip those rows, and their ``Y·V`` entry — the whole
    feeder's current, like a slack row's — never becomes a target. The rows stay in the
    STATE (their voltage is solved and reported as before), so the row layout, and with it
    a switch-state inference from the estimated voltages, is untouched. Empty for a grid
    without switch ties.
    """
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import Switch

    index = node_phase_index(grid)
    adjacent: dict[int, set] = {}
    for b in grid.branches:
        if not isinstance(b, Switch) or not b.in_service or not b.default_closed:
            continue
        if branch_states and b.id in branch_states:
            continue
        if float(b.resistance_ohm) > TIE_SWITCH_MAX_OHM or float(b.inductance_h) > 0.0:
            continue
        ra = index.rows_for_terminal(int(b.from_node), tuple(b.from_phases)).tolist()
        rb = index.rows_for_terminal(int(b.to_node), tuple(b.to_phases)).tolist()
        for a, c in zip(ra, rb):
            adjacent.setdefault(int(a), set()).add(int(c))
            adjacent.setdefault(int(c), set()).add(int(a))
    seen = {int(r) for r in torch.as_tensor(slack_rows).reshape(-1).tolist()}
    frontier = list(seen)
    tied: set = set()
    while frontier:
        r = frontier.pop()
        for n in adjacent.get(r, ()):
            if n not in seen:
                seen.add(n)
                tied.add(n)
                frontier.append(n)
    return torch.tensor(sorted(tied), dtype=torch.int64)


class NetworkSolver:
    """Differentiable map from predicted nodal injection currents to node voltages.

    Holds the per-order network operator ``Y(h)`` (grid-constant) and the slack rows. :meth:`solve`
    maps ``I_pred [*lead, H, N]`` (and the measured fundamental slack voltage) to ``V [*lead, H,
    N]`` that exactly satisfies ``Y(h)·V(h) = I(h)``. The fundamental (``h = 1``) is solved with
    an ideal slack (``pgml.solver.solve_harmonic`` ``fixed_rows``/``v_fixed``); the harmonics use
    the Norton-shunted system that matches :func:`pgml.solver.solve_harmonic_flow`.
    """

    def __init__(
        self,
        grid: "Grid",
        orders: Sequence[int],
        slack_rows: Tensor,
        *,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        anchor_v_weight: float = 0.0,
        anchor_i_weight: float = 0.0,
        edge_op=None,
        branch_states: Optional[dict] = None,
        tied_rows: Optional[Tensor] = None,
        harmonic_slack: str = "norton",
        load_shunt: Optional[str] = None,
    ) -> None:
        from injection_hse._switches import materialize_estimator_grid

        grid = materialize_estimator_grid(grid)
        if harmonic_slack not in ("norton", "measured"):
            raise ValueError(
                f"harmonic_slack must be 'norton' or 'measured', got {harmonic_slack!r}."
            )
        _check_load_shunt(load_shunt)
        #: The harmonic device-shunt model of ``Y(h)`` (:attr:`injection_hse.config.PhysicsConfig
        #: .load_shunt`), assembled on the nameplate basis; ``None`` keeps
        #: pgml's assembly defaults.
        self.load_shunt = load_shunt
        #: How the slack rows enter the HARMONIC solves: ``"norton"`` (the source's own
        #: shunt admittance with a zero harmonic emf — the harmonics float against the
        #: supply) or ``"measured"`` (the metered slack's harmonic voltage is the Dirichlet
        #: boundary at every order, as it is at the fundamental — the upstream network's
        #: distortion enters the decode through the measurement instead of being modelled).
        self.harmonic_slack = harmonic_slack
        self.orders = [int(h) for h in orders]
        self.has_fundamental = 1 in self.orders
        self.fundamental_pos = self.orders.index(1) if self.has_fundamental else None
        self.harmonic_orders = [h for h in self.orders if h > 1]
        self._grid = grid
        # Optional topology / switch-state mask ``{branch id: state}``: the assembly keeps a
        # listed branch in the network and scales its admittance by the state, so one grid
        # serves every switch configuration of a topology set (:mod:`injection_hse.data.topology`)
        # without changing the row layout the model is built on.
        self._branch_states = branch_states
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else None
        self._f0 = float(grid.base_frequency_hz)
        self.slack_rows = torch.as_tensor(slack_rows, dtype=torch.int64)
        # rows tied to the slack through closed switches (:func:`tied_rows_of`): no
        # estimable injection lives there
        self.tied_rows = (
            torch.zeros(0, dtype=torch.int64)
            if tied_rows is None
            else torch.as_tensor(tied_rows, dtype=torch.int64).reshape(-1)
        )
        if self._device is not None:
            self.slack_rows = self.slack_rows.to(self._device)
            self.tied_rows = self.tied_rows.to(self._device)
        self.harmonic_pos = torch.tensor(
            [self.orders.index(h) for h in self.harmonic_orders], dtype=torch.int64
        )
        # Measurement-anchoring (over-determined solve): the observed voltages / branch currents
        # become soft constraints on top of the predicted injections. Weights <= 0 keep the
        # original slack-only solve (byte-identical). ``edge_op`` (a
        # :class:`~injection_hse.data.currents.BranchCurrentOperator`) is required for current anchoring —
        # it supplies the V -> branch-current linear map materialised in :meth:`assemble`.
        # The weights are PROPERTIES (:meth:`set_anchor_weights`): assigning one refreshes the
        # anchoring state, so no caller has to keep a second copy of it in sync.
        self._anchor_v_weight = float(anchor_v_weight)
        self._anchor_i_weight = float(anchor_i_weight)
        self._edge_op = edge_op
        # Grid-constant anchoring tensors, (re)built in assemble(): per-order voltage->current
        # admittance scale (makes the voltage weight dimensionless) and the dense per-order
        # branch-current map C(h) [H, E, N] (only when current anchoring is on).
        self._v_scale: Optional[Tensor] = None
        self._c_map: Optional[Tensor] = None
        # NOISE-CALIBRATED anchor weights (:meth:`set_anchor_calibration`): the per-order,
        # per-channel inverse-variance weights ``[H, N]`` / ``[H, E]`` (float64, solver
        # device) the anchored solve multiplies the runtime scalars by instead of the manual
        # operator-scale construction. ``None`` = manual weighting (the default).
        self._cal_v: Optional[Tensor] = None
        self._cal_i: Optional[Tensor] = None
        # The structural COUPLING pattern of ``Y(h)`` (:meth:`coupling_mask`), cached until
        # the next :meth:`assemble`.
        self._coupling: Optional[Tensor] = None
        # FACTOR-ONCE operator caches (:meth:`enable_operator_cache`): per-order
        # :class:`pgml.solver.AnchoredSystem` for the anchored decode and the LU
        # factorizations of the plain decode. Off (``None`` budget) = the historical
        # factorize-per-call paths, byte-identical. The caches hold CONSTANT images of
        # ``Y(h)`` and are invalidated whenever the operators change (:meth:`assemble`,
        # :meth:`to`); a ``Y`` on the autograd tape (the learned-calibration path) is
        # never cached.
        self._cache_budget: Optional[OperatorCacheBudget] = None
        self._anchored_systems: dict = {}
        self._anchored_rejected: set = set()
        self._plain_factors: dict = {}
        self._cache_charged: int = 0
        # Y(h): assembled from the (current) grid. Re-run assemble() after mutating the grid
        # (the learned-calibration path) to refresh the operator on the live tape.
        self.y_harmonic: Optional[Tensor] = None
        self.y_fundamental: Optional[Tensor] = None
        self._free_rows: Optional[Tensor] = None
        self.assemble()

    # -- operator assembly ----------------------------------------------------
    def assemble(self) -> "NetworkSolver":
        """(Re)assemble ``Y(h)`` from the grid (differentiable w.r.t. its parameters).

        The operators are assembled and cached in ``complex128`` REGARDLESS of the
        training dtype: a physical feeder's ``Y`` reaches condition numbers of
        1e6–1e9 (SI units mix ~1e-8 S line shunts with ~1e5 S source stamps), so
        even a ``complex64`` ROUNDING of the matrix entries perturbs the solved
        voltages by up to κ·eps ≈ O(1). Only the decode outputs are cast to the
        training dtype (:attr:`solve` / :attr:`injection_targets`); the cached
        operators are small, so the doubled storage is negligible.
        """
        from pgml.assembly import assemble_network_ybus
        from pgml.solver import assemble_harmonic_ybus

        dev = self._device
        if self.harmonic_orders:
            self.y_harmonic = assemble_harmonic_ybus(
                self._grid,
                self.harmonic_orders,
                dtype=torch.complex128,
                device=dev,
                branch_states=self._branch_states,
                **_decode_shunt_kwargs(self.load_shunt),
            )[0]  # [Hh, N, N]
        if self.has_fundamental:
            yf = assemble_network_ybus(
                self._grid,
                [self._f0],
                dtype=torch.complex128,
                device=dev,
                branch_states=self._branch_states,
            ).Y
            # keep a leading H=1 axis ([1, N, N]) so the batched solve broadcasts cleanly.
            self.y_fundamental = yf if yf.ndim == 3 else yf.unsqueeze(0)
        # free rows = the non-slack rows solved by the fundamental ideal-slack reduction (the
        # complement of the fixed slack). Cached for the reduced solve and the anchoring scale.
        n = (
            self.y_harmonic.shape[-1]
            if self.harmonic_orders
            else self.y_fundamental.shape[-1]
        )
        keep = torch.ones(n, dtype=torch.bool, device=self.slack_rows.device)
        if self.slack_rows.numel():
            keep = keep.index_fill(0, self.slack_rows, False)
        self._free_rows = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        # the cached anchoring tensors belong to the PREVIOUS operators; drop them so the next
        # reader (anchoring or the multi-grid current base) rebuilds from the fresh Y(h).
        self._v_scale = None
        self._c_map = None
        self._coupling = None
        self._drop_operator_caches()
        if self.anchoring:
            self._assemble_anchoring()
        return self

    # -- factor-once operator caches ------------------------------------------
    def enable_operator_cache(
        self, budget: Optional[OperatorCacheBudget]
    ) -> "NetworkSolver":
        """Cache the per-order solve factorizations under the shared ``budget``; returns self.

        With a budget installed, :meth:`solve` reuses one :class:`pgml.solver.AnchoredSystem`
        per order for the anchored decode (the ``O(F³)`` inverse image built once, each batch
        answered through the anchored-rows correction) and one LU factorization set for the
        plain decode — the operators are grid-constant, so a training run otherwise pays the
        factorization on EVERY step. Results are algebraically identical to the direct paths
        (machine-rounding differences only). ``None`` disables and releases the held bytes.
        A solver whose ``Y(h)`` carries gradients (the learned-calibration path) ignores the
        cache and keeps the direct, on-tape solve.
        """
        if budget is None:
            self._drop_operator_caches()
        self._cache_budget = budget
        return self

    def _drop_operator_caches(self) -> None:
        """Invalidate the factor-once caches and refund their budget reservation."""
        self._anchored_systems = {}
        self._anchored_rejected = set()
        self._plain_factors = {}
        if self._cache_budget is not None and self._cache_charged:
            self._cache_budget.refund(self._cache_charged)
        self._cache_charged = 0

    def _operators_on_tape(self) -> bool:
        """Whether ``Y(h)`` carries gradients (then nothing may be cached as constant)."""
        return bool(
            (self.y_harmonic is not None and self.y_harmonic.requires_grad)
            or (self.y_fundamental is not None and self.y_fundamental.requires_grad)
        )

    def _charge_cache(self, nbytes: int) -> bool:
        if not self._cache_budget.charge(nbytes):
            return False
        self._cache_charged += int(nbytes)
        return True

    def _anchored_system(self, pos: int, *, need_op: bool):
        """The cached :class:`pgml.solver.AnchoredSystem` of order position ``pos``.

        ``None`` when caching is off, the budget is spent, or the operators are on the
        tape — the caller then takes the direct per-call path. ``need_op``: the batch
        anchors branch currents, so a system cached without the branch-current image is
        rebuilt with it (a stage ladder can turn current anchoring on later).
        """
        if self._cache_budget is None or self._operators_on_tape():
            return None
        sys_ = self._anchored_systems.get(pos)
        if sys_ is not None and (not need_op or sys_.oz is not None):
            return sys_
        if pos in self._anchored_rejected:
            return None
        from pgml.solver import AnchoredSystem

        order = self.orders[pos]
        if order == 1:
            y = self.y_fundamental[0]
            fixed = self.slack_rows if self.slack_rows.numel() else None
        else:
            y = self.y_harmonic[self.harmonic_orders.index(order)]
            fixed = self.slack_rows if self.measured_harmonic_slack else None
        op = self._c_map[pos] if (need_op and self._c_map is not None) else None
        built = AnchoredSystem(y, op=op, fixed_rows=fixed)
        nbytes = built.nbytes()
        if sys_ is not None:
            # replacing the op-less system: its bytes come back before the new charge
            self._cache_budget.refund(sys_.nbytes())
            self._cache_charged -= sys_.nbytes()
        if not self._charge_cache(nbytes):
            self._anchored_rejected.add(pos)
            return None
        self._anchored_systems[pos] = built
        return built

    def _plain_factored(self, key: str):
        """The cached plain-solve factorization (``"fund"`` / ``"harm"``), or ``None``.

        Built through :func:`pgml.solver.harmonic.lu_factor_system` (``backend="auto"``:
        sparse SuperLU on large CPU systems, batched dense LU elsewhere) and charged to
        the budget; the fundamental factors the free-row block with the slack held as the
        Dirichlet boundary, exactly as the direct solve reduces it.
        """
        if self._cache_budget is None or self._operators_on_tape():
            return None
        fac = self._plain_factors.get(key)
        if fac is not None:
            return fac
        if key in self._anchored_rejected:
            return None
        from pgml.solver.harmonic import lu_factor_system

        if key == "fund":
            fac = lu_factor_system(
                self.y_fundamental,
                fixed_rows=self.slack_rows if self.slack_rows.numel() else None,
            )
        else:
            fac = lu_factor_system(
                self.y_harmonic,
                fixed_rows=self.slack_rows if self.measured_harmonic_slack else None,
            )
        nbytes = _factored_nbytes(fac)
        if not self._charge_cache(nbytes):
            self._anchored_rejected.add(key)
            return None
        self._plain_factors[key] = fac
        return fac

    # -- measurement anchoring: the weights own their derived state -------------
    @property
    def anchoring(self) -> bool:
        """Whether the next :meth:`solve` is MEASUREMENT-ANCHORED (either weight positive).

        Derived from the weights rather than stored, so it can never disagree with them: a
        caller changes the anchoring by setting :attr:`anchor_v_weight` /
        :attr:`anchor_i_weight` (or :meth:`set_anchor_weights`) and nothing else.
        """
        return self._anchor_v_weight > 0.0 or self._anchor_i_weight > 0.0

    @property
    def anchor_v_weight(self) -> float:
        """Dimensionless trust in the observed VOLTAGES (``0`` = the slack-only solve)."""
        return self._anchor_v_weight

    @anchor_v_weight.setter
    def anchor_v_weight(self, weight: float) -> None:
        self.set_anchor_weights(v_weight=weight)

    @property
    def can_anchor_current(self) -> bool:
        """Whether a positive :attr:`anchor_i_weight` is servable on this solver.

        ``True`` once the ``V -> branch current`` map exists or can be materialised from a
        :class:`~injection_hse.data.currents.BranchCurrentOperator`. The curriculum checks it BEFORE a
        stage starts, so a schedule that would ask for current anchoring on a solver built
        without metered currents fails at the stage boundary, not mid-epoch.
        """
        return self._c_map is not None or self._edge_op is not None

    @property
    def anchor_i_weight(self) -> float:
        """Trust in the observed BRANCH CURRENTS (``0`` = they do not constrain the solve)."""
        return self._anchor_i_weight

    @anchor_i_weight.setter
    def anchor_i_weight(self, weight: float) -> None:
        self.set_anchor_weights(i_weight=weight)

    def set_anchor_weights(
        self, *, v_weight: Optional[float] = None, i_weight: Optional[float] = None
    ) -> "NetworkSolver":
        """Install the measurement-anchoring weights; returns self.

        THE entry point for the curriculum's anchoring schedule and for a deployment that
        scores a trained model at a chosen trust (``None`` leaves a weight untouched). The
        grid-constant anchoring tensors are materialised on demand — the per-order operator
        scale, and, for current anchoring, the dense ``V -> branch current`` map — so a solver
        built unanchored can be dialled up later without re-assembling ``Y(h)``. A positive
        current weight on a solver built without a
        :class:`~injection_hse.data.currents.BranchCurrentOperator` raises: there is nothing to
        constrain the solve with.
        """
        if v_weight is not None:
            self._anchor_v_weight = float(v_weight)
        if i_weight is not None:
            self._anchor_i_weight = float(i_weight)
        operators_ready = self.y_harmonic is not None or self.y_fundamental is not None
        if operators_ready and self.anchoring:
            self._assemble_anchoring()
        return self

    # -- noise-calibrated anchor weights ------------------------------------------
    @property
    def anchor_calibrated(self) -> bool:
        """Whether the anchored solve weights its channels by the installed noise calibration.

        ``True`` after :meth:`set_anchor_calibration`; ``False`` (the default, and after
        :meth:`clear_anchor_calibration`) means the MANUAL construction — voltage weight
        ``anchor_v_weight · s(h)²`` with ``s(h)`` the operator scale, current weight
        ``anchor_i_weight`` raw.
        """
        return self._cal_v is not None

    @property
    def anchor_voltage_calibration(self) -> Optional[Tensor]:
        """float64 ``[H, N]`` (or ``[H, 1]``) calibrated voltage-row weight at multiplier 1.

        ``σ_I(h, r)² / σ_y(h, r)²`` — the inverse-variance weight of each observed voltage
        row relative to the unit-weight physics block (units of admittance², like the manual
        ``s(h)²`` scale). ``None`` in manual mode.
        """
        return self._cal_v

    @property
    def anchor_current_calibration(self) -> Optional[Tensor]:
        """float64 ``[H, E]`` (or ``[H, 1]``) calibrated current-channel weight at multiplier 1.

        ``σ_I(h, src(e))² / σ_i(h, e)²`` — dimensionless, the current term sharing the
        physics term's units. ``None`` in manual mode or when no current sigma was installed.
        """
        return self._cal_i

    def set_anchor_calibration(
        self,
        *,
        sigma_injection: Tensor,
        sigma_voltage: Tensor,
        sigma_current: Optional[Tensor] = None,
    ) -> "NetworkSolver":
        """Install NOISE-CALIBRATED anchor weights; returns self. Idempotent.

        The anchored solve is the Gaussian MAP estimate of the state given the predicted
        injections (the prior, with error ``σ_I``) and the measurements (with their sensor
        errors ``σ_y`` / ``σ_i``): with the physics block normalised to weight 1, the correct
        weight of every anchored channel is the RATIO of variances ``w = σ_I² / σ_meas²``.
        A manual weight trusts every channel of an order equally and — because the voltage
        weight is scaled by the operator and the current weight is not — trusts a noisy CT
        far more than a noisy voltage sensor; the calibrated weights remove both defects and
        the hand-tuned schedule with them. After this call :attr:`anchor_v_weight` /
        :attr:`anchor_i_weight` act as DIMENSIONLESS MULTIPLIERS on the calibrated weights
        (``1`` = the noise-optimal weight, ``0`` = the channel off); the per-stage schedule
        and the per-batch randomisation still move those multipliers.

        The weights are built once here, off the autograd tape, as float64 constants on the
        solver's device, and broadcast over the batch in :meth:`solve`; only the observation
        masks vary per batch. They do not depend on ``Y(h)`` and survive :meth:`assemble`.

        Parameters
        ----------
        sigma_injection:
            float64 ``[H]`` (one value per order) or ``[H, N]`` (per order and row) — the
            assumed prior error of the predicted injection, in AMPERES (the units of the
            physics residual), ``σ_I(h) = ρ · s_I(h)`` with ``s_I`` the RMS injection
            magnitude of the training rows and ``ρ`` the relative prior error
            (``PhysicsConfig.prior_injection_rel_sigma``). The per-row form is the multi-grid
            one (the per-unit scale times each row's current base).
        sigma_voltage:
            float64 ``[H, N]`` (or ``[H, 1]``) per-REAL-COMPONENT error std of each voltage
            row's measurement at each order, in VOLTS — see :func:`anchor_measurement_sigma`
            for the construction from a sensor model, :func:`clean_sensor_sigma` for a
            noise-free run.
        sigma_current:
            float64 ``[H, E]`` (or ``[H, 1]``) per-real-component error std of each
            branch-current (CT) channel, in AMPERES; ``None`` = no current calibration (a
            positive current multiplier then raises in :meth:`solve`). With a per-row
            ``sigma_injection`` the current channel ``e`` takes the prior error at its SOURCE
            row (the terminal the CT meters), which needs the solver's branch-current
            operator.

        Raises
        ------
        ValueError
            On a shape mismatch against the solver's orders / rows / half-edges, or a
            non-positive sigma (a zero sigma would be an infinitely trusted channel, which
            the correction system cannot represent).
        """
        dev = self.slack_rows.device
        n = (
            self.y_harmonic.shape[-1]
            if self.harmonic_orders
            else self.y_fundamental.shape[-1]
        )
        h = len(self.orders)
        s_inj = torch.as_tensor(sigma_injection, dtype=torch.float64).to(dev)
        s_v = torch.as_tensor(sigma_voltage, dtype=torch.float64).to(dev)
        if s_inj.ndim == 1:
            s_inj = s_inj.unsqueeze(-1)  # [H, 1]
        if s_inj.ndim != 2 or s_inj.shape[0] != h or s_inj.shape[1] not in (1, n):
            raise ValueError(
                f"sigma_injection must be [H={h}] or [H, N={n}]; got {tuple(s_inj.shape)}."
            )
        if s_v.ndim != 2 or s_v.shape[0] != h or s_v.shape[1] not in (1, n):
            raise ValueError(
                f"sigma_voltage must be [H={h}, N={n}] or [H, 1]; got {tuple(s_v.shape)}."
            )
        if not bool((s_inj > 0).all()) or not bool((s_v > 0).all()):
            raise ValueError("anchor calibration sigmas must be strictly positive.")
        cal_v = (s_inj / s_v) ** 2  # [H, N] or [H, 1]

        cal_i = None
        if sigma_current is not None:
            s_i = torch.as_tensor(sigma_current, dtype=torch.float64).to(dev)
            e = self._edge_op.n_edges if self._edge_op is not None else None
            if s_i.ndim != 2 or s_i.shape[0] != h:
                raise ValueError(
                    f"sigma_current must be [H={h}, E] or [H, 1]; got {tuple(s_i.shape)}."
                )
            if e is not None and s_i.shape[1] not in (1, e):
                raise ValueError(
                    f"sigma_current has {s_i.shape[1]} channels but the branch-current "
                    f"operator has {e} half-edges."
                )
            if not bool((s_i > 0).all()):
                raise ValueError("anchor calibration sigmas must be strictly positive.")
            if s_inj.shape[1] == 1:
                s_inj_e = s_inj  # [H, 1]
            else:
                if self._edge_op is None:
                    raise ValueError(
                        "a per-row sigma_injection needs the branch-current operator "
                        "(edge_op) to place each CT channel at its source row."
                    )
                src = self._edge_op.edge_src_rows.to(device=dev, dtype=torch.int64)
                s_inj_e = s_inj.index_select(1, src)  # [H, E]
            cal_i = (s_inj_e / s_i) ** 2  # [H, E] or [H, 1]

        self._cal_v = cal_v
        self._cal_i = cal_i
        return self

    def clear_anchor_calibration(self) -> "NetworkSolver":
        """Return to the MANUAL anchor-weight construction; returns self."""
        self._cal_v = None
        self._cal_i = None
        return self

    def operator_scale(self) -> Tensor:
        """float64 ``[H]`` per-order scale of the SOLVED operator (its MEDIAN singular value).

        ``s(h)`` is a *typical coupling* admittance of the operator each order is actually
        solved with — the slack-free ``Y_ff`` for the fundamental (the reduced ideal-slack
        system), ``Y(h)`` for the harmonics. It is the physical scale that converts between the
        voltage and current worlds, and it is used for two things:

        - MEASUREMENT ANCHORING: the anchor term ``wv_eff·|V - V_meas|²`` (V²) is added to the
          physics residual ``|Y·V - I_pred|²`` (A²), so ``wv_eff`` carries units of
          admittance² and ``wv_eff = anchor_v_weight · s(h)²`` makes ``anchor_v_weight`` a
          DIMENSIONLESS relative trust — ``1`` pulls the poorly-determined (small-σ) directions
          toward the measurements while the physics still governs the stiff ones. (The diagonal
          self-admittance would be dominated by the stiff source-shunt stamps and make
          ``anchor_v_weight = 1`` fully measurement-dominant.)
        - The per-unit CURRENT base: ``s(h)·V_ref²`` is the POWER a typical coupling admittance
          carries at the grid's reference voltage, and dividing it by a row's own base voltage
          gives that row's current base (:meth:`i_base`, and per member
          :meth:`MultiGridPhysicsBank.i_base`).

        Computed lazily from the assembled operators and cached until the next
        :meth:`assemble`, so it costs one set of singular-value decompositions per grid
        whether or not anchoring is on. Lives on the solver's device.
        """
        if self._v_scale is None:
            scales: list[Tensor] = []
            hi = 0
            for h in self.orders:
                if h == 1:
                    yh = self.y_fundamental[0]
                    yh = yh.index_select(0, self._free_rows).index_select(
                        1, self._free_rows
                    )  # the reduced free-row operator the fundamental actually solves
                else:
                    yh = self.y_harmonic[hi]
                    hi += 1
                scale = torch.linalg.svdvals(yh).median().clamp_min(1e-30)
                scales.append(scale.to(torch.float64))
            v_scale = torch.stack(scales)  # [H], complex128 grid -> float64 magnitudes
            if self._device is not None:
                v_scale = v_scale.to(self._device)
            self._v_scale = v_scale
        return self._v_scale

    def i_base(self, base_voltage: Tensor) -> Tensor:
        """float64 ``[H, N]`` per-order, per-row CURRENT base (off the tape).

        ``i_base[h, n] = S_base(h) / V_base[n]`` with the grid-global power base
        ``S_base(h) = s(h)·V_ref²`` — the power a *typical coupling admittance* of this network
        (:meth:`operator_scale`) carries at the grid's reference voltage
        (:func:`reference_base_voltage`), and ``V_base[n]`` that row's line-to-neutral base
        voltage (:func:`injection_hse.data.base_voltage_per_row`, the same reference the per-unit voltage
        uses). Dividing a nodal injection current by it removes BOTH the voltage class (through
        ``V_base``) and the network's stiffness (through ``s(h)``) — the current analogue of
        :func:`injection_hse.data.per_unit_state`, and the frame the injection features live in under
        :attr:`injection_hse.config.NormalizationConfig.per_unit` (and, per member, under
        :attr:`injection_hse.config.MultiGridConfig.per_unit` — :meth:`MultiGridPhysicsBank.i_base`
        delegates here, so the convention has ONE definition).

        THE CURRENT BASE SCALES INVERSELY WITH THE VOLTAGE CLASS. A per-unit system fixes one
        power base per grid and derives ``I_base = S_base/V_base``: the same power at 50× the
        voltage is 1/50 of the current, so the rows of a higher class must be divided by a
        SMALLER base, not a larger one. A base proportional to ``V_base`` would separate the
        classes by ``V_base²`` (a factor 2500 across a 20 kV/400 V transformer) in exactly the
        quantities the per-unit voltage frame exists to make homogeneous — the
        injection features, the metered CT channels and the zero-injection prior sigma the
        noise-calibrated anchor weighting installs per row.

        ``V_ref`` only fixes the grid-global power base, never the class dependence, and is the
        MEDIAN rated row voltage (:func:`reference_base_voltage`) so that a single-voltage-class
        grid reproduces ``s(h)·V_base`` BIT-FOR-BIT and a mixed-class member keeps its dominant
        class on the same convention as a pure member of that class (which is what lets one
        multi-grid normalizer span the corpus).

        A row with no rated voltage keeps base ``1`` (unscaled), matching
        :func:`injection_hse.data.per_unit_state`. The base is a UNIT CONVENTION, not a learnable
        quantity: the operator scale is detached here, so a per-unit feature or target never
        back-propagates into ``Y`` through its own scale.
        """
        scale = self.operator_scale().detach()  # [H] float64
        v_base = torch.as_tensor(base_voltage).to(
            device=scale.device, dtype=torch.float64
        )
        rated = v_base > 0
        ones = torch.ones_like(v_base)
        v_ref = reference_base_voltage(v_base)  # 0-dim, the grid's reference class
        # ``V_ref·(V_ref/V_base)`` rather than ``V_ref²/V_base``: the ratio of a row at the
        # reference class is EXACTLY 1.0, so a single-voltage-class grid lands on
        # ``s(h)·V_base`` bit-for-bit rather than one rounding step away from it.
        row_base = torch.where(
            rated, v_ref * (v_ref / torch.where(rated, v_base, ones)), ones
        )  # [N]
        return scale.unsqueeze(-1) * row_base

    def coupling_mask(self) -> Tensor:
        """bool ``[N, N]`` — which rows ``Y(h)`` couples: ``Y(h)[r, j] != 0`` at ANY order.

        The structural neighbourhood of every node-phase row under the nodal network law
        (the rows whose voltages enter row ``r``'s current balance ``Σ_j Y_rj V_j``): the
        rows of the same node and of the branches incident to it, phase couplings included.
        The diagonal is always set, so a row is its own neighbour. Grid-constant — computed
        lazily from the assembled operators and cached until the next :meth:`assemble`; on the
        solver's device, off the autograd tape.
        """
        if self._coupling is None:
            ops = []
            if self.y_fundamental is not None:
                ops.append(self.y_fundamental[0].detach() != 0)
            if self.y_harmonic is not None:
                ops.append((self.y_harmonic.detach() != 0).any(dim=0))
            coupling = ops[0]
            for other in ops[1:]:
                coupling = coupling | other
            n = int(coupling.shape[-1])
            eye = torch.eye(n, dtype=torch.bool, device=coupling.device)
            self._coupling = coupling | eye
        return self._coupling

    def _assemble_anchoring(self) -> None:
        """Materialise the grid-constant anchoring tensors from the assembled operators.

        The per-order voltage->current scale ``_v_scale`` comes from :meth:`operator_scale`;
        ``_c_map`` ``[H, E, N]`` is the dense per-order V->branch-current map, materialised
        from the sparse :class:`~injection_hse.data.currents.BranchCurrentOperator` by evaluating it on
        the node-voltage identity basis (only when current anchoring is on). Both are cached
        until the next :meth:`assemble` invalidates them, so the per-batch weight
        randomization pays for them once.
        """
        self.operator_scale()

        if self._anchor_i_weight > 0.0 and self._c_map is None:
            if self._edge_op is None:
                raise ValueError(
                    "physics.anchor_current_weight > 0 requires a BranchCurrentOperator "
                    "(edge_op) so the decode can materialise the V -> branch-current map; "
                    "current anchoring also requires measurement.measure_current."
                )
            n = (
                self.y_harmonic.shape[-1]
                if self.harmonic_orders
                else self.y_fundamental.shape[-1]
            )
            dev = self._device
            self._edge_op.to(dev if dev is not None else torch.device("cpu"))
            # C(h)[e, r] = d i_edge_e / d V_r: evaluate the operator on the identity basis
            # (each "sample" n is the unit voltage e_n), giving [N, H, E] -> [H, E, N].
            basis = torch.eye(n, dtype=torch.complex128, device=dev)  # [N, N]
            basis = basis.unsqueeze(1).expand(n, len(self.orders), n)  # [N(lead), H, N]
            cmap = self._edge_op(basis)  # [N, H, E]
            self._c_map = (
                cmap.permute(1, 2, 0).contiguous().to(torch.complex128)
            )  # [H, E, N]
            if dev is not None:
                self._c_map = self._c_map.to(dev)

    def to(self, device) -> "NetworkSolver":
        """Move the cached operators + index tensors to ``device`` (the model's device)."""
        dev = torch.device(device)
        self._device = dev
        if self.y_harmonic is not None:
            self.y_harmonic = self.y_harmonic.to(dev)
        if self.y_fundamental is not None:
            self.y_fundamental = self.y_fundamental.to(dev)
        self.slack_rows = self.slack_rows.to(dev)
        self.tied_rows = self.tied_rows.to(dev)
        self.harmonic_pos = self.harmonic_pos.to(dev)
        if self._free_rows is not None:
            self._free_rows = self._free_rows.to(dev)
        if self._v_scale is not None:
            self._v_scale = self._v_scale.to(dev)
        if self._c_map is not None:
            self._c_map = self._c_map.to(dev)
        if self._cal_v is not None:
            self._cal_v = self._cal_v.to(dev)
        if self._cal_i is not None:
            self._cal_i = self._cal_i.to(dev)
        if self._coupling is not None:
            self._coupling = self._coupling.to(dev)
        if self._edge_op is not None:
            self._edge_op.to(dev)
        # the factor-once caches are device-resident images; rebuild lazily on the new device
        self._drop_operator_caches()
        return self

    # -- the physics decode ---------------------------------------------------
    def solve(
        self,
        i_pred: Tensor,
        v_slack: Tensor,
        *,
        v_meas: Optional[Tensor] = None,
        v_mask: Optional[Tensor] = None,
        i_meas: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
        order_mask: Optional[Tensor] = None,
        edge_order_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predicted injection currents ``[*lead, H, N]`` -> voltages ``[*lead, H, N]``.

        Without anchoring inputs (or when both anchor weights are ``0``) this is the
        slack-only physics decode: the fundamental via the ideal-slack reduction at the
        measured slack voltage, the harmonics via the Norton solve — ``V`` exactly satisfies
        ``Y(h)·V = I_pred``. With ``v_meas``/``i_meas`` and a positive weight it becomes the
        MEASUREMENT-ANCHORED (over-determined) solve: the observed voltage rows (``v_mask``,
        or per-order ``order_mask``) and observed branch currents (``edge_mask`` /
        ``edge_order_mask``, currents from ``i_meas`` mapped through ``C(h)``) are added as soft
        least-squares constraints, so the state is pulled toward every measurement (the slack
        stays hard-fixed). Differentiable w.r.t. ``i_pred``, ``v_slack``, the measurements and
        the grid ``Y(h)``; vectorised over the leading batch/time dims.
        """
        if not self.anchoring or (v_meas is None and i_meas is None):
            return self._solve_plain(i_pred, v_slack)
        return self._solve_anchored(
            i_pred,
            v_slack,
            v_meas=v_meas,
            v_mask=v_mask,
            i_meas=i_meas,
            edge_mask=edge_mask,
            order_mask=order_mask,
            edge_order_mask=edge_order_mask,
        )

    @property
    def measured_harmonic_slack(self) -> bool:
        """Whether the harmonic solves hold the metered slack rows fixed (see ``harmonic_slack``)."""
        return self.harmonic_slack == "measured" and bool(self.slack_rows.numel())

    def _split_slack(self, v_slack: Tensor, lead_ndim: int):
        """``(v_fund [*lead, K], v_harm [*lead, Hh, K] | None)`` from the ``v_slack`` argument.

        ``v_slack`` is the fundamental slack voltage ``[*lead, K]``, or the per-order slack
        voltage ``[*lead, H, K]`` (the order axis in the solver's layout). The harmonic part
        is required under ``harmonic_slack="measured"`` and ignored otherwise.
        """
        if v_slack.ndim == lead_ndim + 2:
            v_f = (
                v_slack.select(-2, self.fundamental_pos)
                if self.has_fundamental
                else v_slack.new_zeros((*v_slack.shape[:-2], v_slack.shape[-1]))
            )
            v_h = (
                v_slack.index_select(-2, self.harmonic_pos.to(v_slack.device))
                if self.harmonic_orders
                else None
            )
            return v_f, v_h
        if self.measured_harmonic_slack and self.harmonic_orders:
            raise ValueError(
                "harmonic_slack='measured' needs the per-order slack voltage [*lead, H, K] "
                f"(got {tuple(v_slack.shape)}): the metered slack's harmonic voltage is the "
                "boundary of every harmonic solve."
            )
        return v_slack, None

    def _solve_plain(self, i_pred: Tensor, v_slack: Tensor) -> Tensor:
        """Slack-only physics decode ``[*lead, H, N]`` -> ``[*lead, H, N]``.

        ``v_slack`` is the measured fundamental slack voltage ``[*lead, len(slack_rows)]`` (a
        known boundary condition; used only by the fundamental ideal-slack solve). The order
        axis (``-2``) is split into the fundamental + harmonic solves and recombined in the
        original order. Differentiable w.r.t. ``i_pred``, ``v_slack`` and the grid ``Y(h)``.

        The linear solves run internally in ``complex128`` and the result is cast
        back to the training dtype: a physical feeder's ``Y`` reaches condition
        numbers of 1e6–1e9 (SI units mix ~1e-8 S shunts with ~1e5 S source
        stamps), so a direct ``complex64`` solve loses up to κ·eps ≈ all of its
        digits — and CPU/CUDA factorizations then disagree at the percent level.
        The upcast costs one cast of the small cached operators per call and
        keeps the decode exact (and device-independent) at any training precision.
        """
        from pgml.solver import solve_harmonic
        from pgml.solver.harmonic import solve_factored

        cdt = self._dtype if self._dtype is not None else i_pred.dtype
        sdt = torch.complex128  # internal solve precision (see docstring)
        i_pred = i_pred.to(sdt)
        v_slack_f, v_slack_h = self._split_slack(v_slack, i_pred.ndim - 2)

        v_fund: Optional[Tensor] = None
        if self.has_fundamental:
            # add an explicit H=1 axis so the batched ideal-slack solve broadcasts the
            # [1, N, N] operator over the leading dims (its unbatched path assumes a single RHS).
            i_f = i_pred.select(-2, self.fundamental_pos).unsqueeze(-2)  # [*lead, 1, N]
            vsl = v_slack_f.to(sdt).unsqueeze(-2)  # [*lead, 1, K]
            fac_f = self._plain_factored("fund")
            if fac_f is not None:
                v_fund = solve_factored(fac_f, i_f, v_fixed=vsl).squeeze(-2)
            else:
                v_fund = solve_harmonic(
                    self.y_fundamental.to(sdt),
                    i_f,
                    fixed_rows=self.slack_rows,
                    v_fixed=vsl,
                ).squeeze(-2)  # [*lead, N]

        v_harm: Optional[Tensor] = None
        if self.harmonic_orders:
            i_h = i_pred.index_select(-2, self.harmonic_pos.to(i_pred.device))
            fac_h = self._plain_factored("harm")
            if self.measured_harmonic_slack:
                # the metered slack's harmonic voltage is the boundary, order by order
                vsl_h = v_slack_h.to(sdt)  # [*lead, Hh, K]
                if fac_h is not None:
                    v_harm = solve_factored(fac_h, i_h, v_fixed=vsl_h)
                else:
                    v_harm = solve_harmonic(
                        self.y_harmonic.to(sdt),
                        i_h,
                        fixed_rows=self.slack_rows,
                        v_fixed=vsl_h,
                    )
            elif fac_h is not None:
                v_harm = solve_factored(fac_h, i_h)  # [*lead, Hh, N]
            else:
                v_harm = solve_harmonic(self.y_harmonic.to(sdt), i_h)

        # recombine into the original order layout (out-of-place; loop is over H, not nodes)
        slices: list[Tensor] = []
        hi = 0
        for h in self.orders:
            if h == 1:
                slices.append(v_fund)
            else:
                slices.append(v_harm.select(-2, hi))
                hi += 1
        return torch.stack(slices, dim=-2).to(cdt)  # [*lead, H, N]

    # -- the measurement-anchored (over-determined) decode --------------------
    def _anchored_order(
        self,
        y: Tensor,
        i_rhs: Tensor,
        *,
        is_fundamental: bool,
        v_slack: Tensor,
        vmask: Optional[Tensor],
        vmeas: Optional[Tensor],
        wv: Optional[Tensor],
        emask: Optional[Tensor],
        imeas: Optional[Tensor],
        c_h: Optional[Tensor],
        wi: "Optional[float | Tensor]",  # manual: float; calibrated: [E] weight; None: off
    ) -> Tensor:
        """One order's anchored solve ``[Bf, N]`` via the reusable ``pgml.solver.solve_anchored``.

        The soft anchors are the observed voltage rows (``wv`` at ``vmask`` rows, toward
        ``vmeas``) and the observed branch currents (``wi`` at ``emask`` edges, toward ``imeas``,
        through the branch-current operator ``c_h``). ``wv`` / ``wi`` are the per-order weights
        — a scalar (manual weighting) or a float64 ``[N]`` / ``[E]`` per-channel tensor (noise
        calibrated) — broadcast against the masks; ``wi`` is ``None`` when the currents are
        not anchored. The fundamental passes the slack as the hard Dirichlet boundary
        (``fixed_rows``); the solve reduces to the non-slack rows, reuses ONE factorization of
        the shared ``Y`` across the batch, and scatters the slack back — no per-sample QR of
        the (shared) physics block.
        """
        from pgml.solver import solve_anchored

        fixed = (
            self.slack_rows
            if (
                (is_fundamental or self.measured_harmonic_slack)
                and self.slack_rows.numel()
            )
            else None
        )
        row_weight = None
        if wv is not None and vmask is not None:
            row_weight = vmask.to(torch.float64) * wv.to(torch.float64)  # [Bf, N]
        op = op_weight = None
        if wi is not None and c_h is not None:
            op = c_h
            op_weight = emask.to(torch.float64) * wi  # [Bf, E]
        return solve_anchored(
            y,
            i_rhs,
            row_weight=row_weight,
            row_target=vmeas if row_weight is not None else None,
            op=op,
            op_weight=op_weight,
            op_target=imeas if op is not None else None,
            fixed_rows=fixed,
            v_fixed=v_slack if fixed is not None else None,
        )

    def _solve_anchored(
        self,
        i_pred: Tensor,
        v_slack: Tensor,
        *,
        v_meas,
        v_mask,
        i_meas,
        edge_mask,
        order_mask,
        edge_order_mask,
    ) -> Tensor:
        """Measurement-anchored decode (see :meth:`solve`). Loops the handful of orders;
        every node/branch/scenario axis stays vectorised inside the per-order least squares."""
        cdt = self._dtype if self._dtype is not None else i_pred.dtype
        sdt = torch.complex128
        h = len(self.orders)
        lead = tuple(i_pred.shape[:-2])
        bf = 1
        for s in lead:
            bf *= int(s)
        n = int(i_pred.shape[-1])

        ip = i_pred.to(sdt).reshape(bf, h, n)
        v_slack_f, v_slack_h = self._split_slack(v_slack, len(lead))
        vsl = v_slack_f.to(sdt).reshape(bf, -1)
        vsl_h = (
            v_slack_h.to(sdt).reshape(bf, len(self.harmonic_orders), -1)
            if (v_slack_h is not None and self.measured_harmonic_slack)
            else None
        )

        do_v = self.anchor_v_weight > 0.0 and v_meas is not None
        omask = vm = None
        if do_v:
            vm = v_meas.to(sdt).reshape(bf, h, n)
            if order_mask is not None:
                omask = order_mask.reshape(bf, n, h).permute(0, 2, 1).contiguous()
            elif v_mask is not None:
                omask = v_mask.reshape(bf, n).unsqueeze(1).expand(bf, h, n)
            else:
                do_v = False

        do_i = (
            self.anchor_i_weight > 0.0
            and i_meas is not None
            and self._c_map is not None
        )
        emask = im = None
        if do_i:
            e = int(self._c_map.shape[1])
            im = i_meas.to(sdt).reshape(bf, h, e)
            if edge_order_mask is not None:
                emask = edge_order_mask.reshape(bf, e, h).permute(0, 2, 1).contiguous()
            elif edge_mask is not None:
                emask = edge_mask.reshape(bf, e).unsqueeze(1).expand(bf, h, e)
            else:
                do_i = False

        # The per-order weights. MANUAL: the voltage weight is the dimensionless trust times
        # the operator scale squared, the current weight enters raw. CALIBRATED: both are the
        # installed inverse-variance weights (per channel) times the runtime multipliers.
        calibrated = self._cal_v is not None
        if calibrated and do_i and self._cal_i is None:
            raise ValueError(
                "the anchored solve is noise-calibrated but no current sigma was installed "
                "(set_anchor_calibration(sigma_current=...)); it cannot weight the "
                "branch-current anchors."
            )

        outs: list[Tensor] = []
        hi = 0
        for pos, order in enumerate(self.orders):
            if order == 1:
                yh = self.y_fundamental[0].to(sdt)
                vsl_pos = vsl
            else:
                yh = self.y_harmonic[hi].to(sdt)
                vsl_pos = vsl_h[:, hi] if vsl_h is not None else vsl
                hi += 1
            if calibrated:
                wv = (
                    self.anchor_v_weight * self._cal_v[pos] if do_v else None
                )  # [N]|[1]
                wi = (
                    self.anchor_i_weight * self._cal_i[pos] if do_i else None
                )  # [E]|[1]
            else:
                wv = self.anchor_v_weight * self._v_scale[pos] ** 2 if do_v else None
                wi = self.anchor_i_weight if do_i else None
            sys_h = self._anchored_system(pos, need_op=do_i)
            if sys_h is not None:
                row_weight = None
                if do_v:
                    wv_t = torch.as_tensor(wv, dtype=torch.float64, device=ip.device)
                    row_weight = omask[:, pos].to(torch.float64) * wv_t
                op_weight = None
                if do_i:
                    wi_t = torch.as_tensor(wi, dtype=torch.float64, device=ip.device)
                    op_weight = emask[:, pos].to(torch.float64) * wi_t
                v_h = sys_h.solve(
                    ip[:, pos],
                    row_weight=row_weight,
                    row_target=vm[:, pos] if do_v else None,
                    op_weight=op_weight,
                    op_target=im[:, pos] if do_i else None,
                    v_fixed=vsl_pos if sys_h.fixed_rows is not None else None,
                )
            else:
                v_h = self._anchored_order(
                    yh,
                    ip[:, pos],
                    is_fundamental=(order == 1),
                    v_slack=vsl_pos,
                    vmask=omask[:, pos] if do_v else None,
                    vmeas=vm[:, pos] if do_v else None,
                    wv=wv,
                    emask=emask[:, pos] if do_i else None,
                    imeas=im[:, pos] if do_i else None,
                    c_h=self._c_map[pos].to(sdt) if do_i else None,
                    wi=wi,
                )
            outs.append(v_h)
        v = torch.stack(outs, dim=1)  # [Bf, H, N]
        return v.reshape(*lead, h, n).to(cdt)

    # -- targets for normalizing / supervising the injection head -------------
    def zero_slack(self, i: Tensor) -> Tensor:
        """``i [*lead, H, N]`` with the slack rows set to zero (out-of-place).

        A slack row's ``Y·V`` entry is the source's own current — the whole feeder's at the
        fundamental, and ``V_h / R_source`` at a harmonic when the slack carries a harmonic
        voltage — never a device injection: the solve holds those rows as its boundary and
        never reads an injection there, so the head's target at them is exactly zero.
        """
        if self.slack_rows.numel() == 0:
            return i
        return i.index_fill(-1, self.slack_rows.to(i.device), 0)

    def zero_tied(self, i: Tensor) -> Tensor:
        """``i [*lead, H, N]`` with the switch-tied rows (:attr:`tied_rows`) set to zero.

        Out of place; the identity when the grid has no ties.
        """
        if self.tied_rows.numel() == 0:
            return i
        return i.index_fill(-1, self.tied_rows.to(i.device), 0)

    def injection_targets(self, v_full: Tensor) -> Tensor:
        """The true nodal injection currents ``I(h) = Y(h)·V_true(h)`` for ``v_full [*lead, H, N]``.

        Used to fit the injection normalizer (and, optionally, to supervise the head directly).
        At non-slack rows this is the device injection the head must learn; the slack-row entries
        are the (free) slack current and are not predicted. Detached from the grid params.
        """
        cdt = self._dtype if self._dtype is not None else v_full.dtype
        # Enforce the documented contract, not just the call-site discipline: the
        # fitted normalizer statistics must be constants, never a live graph.
        # The matmul runs at the cached operators' complex128 precision; the
        # result is cast back to the training dtype below.
        v = v_full.detach().to(torch.complex128)
        outs: list[Tensor] = []
        hi = 0
        for pos, h in enumerate(self.orders):
            vh = v.select(-2, pos)  # [*lead, N]
            if h == 1:
                yv = torch.einsum("ij,...j->...i", self.y_fundamental[0], vh)
            else:
                yv = torch.einsum("ij,...j->...i", self.y_harmonic[hi], vh)
                hi += 1
            outs.append(yv)
        return torch.stack(outs, dim=-2).to(cdt)  # [*lead, H, N]

    @classmethod
    def from_grid(
        cls,
        grid: "Grid",
        orders: Sequence[int],
        *,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        anchor_v_weight: float = 0.0,
        anchor_i_weight: float = 0.0,
        edge_op=None,
        branch_states: Optional[dict] = None,
        harmonic_slack: str = "norton",
        load_shunt: Optional[str] = None,
    ) -> "NetworkSolver":
        """Build a solver, deriving the slack rows from the grid's ``Source`` node(s).

        The slack rows are every in-service ``Source`` terminal; the rows tied to them
        through closed switches (:func:`tied_rows_of`) are recorded as ``tied_rows``.
        ``anchor_v_weight`` / ``anchor_i_weight`` are the anchoring CAPABILITY the solver is
        built for (current anchoring needs ``edge_op``, a
        :class:`~injection_hse.data.currents.BranchCurrentOperator`); the runtime strength is dialled
        later with :meth:`set_anchor_weights`. ``branch_states`` assembles ``Y(h)`` for ONE
        switch configuration of a topology set without changing the row layout;
        ``harmonic_slack`` is :attr:`injection_hse.config.PhysicsConfig.harmonic_slack` and
        ``load_shunt`` :attr:`injection_hse.config.PhysicsConfig.load_shunt`.
        """
        from pgml.assembly import node_phase_index
        from pgml.schemas.grid_schema import Source

        index = node_phase_index(grid)
        srcs = [a for a in grid.appliances if isinstance(a, Source) and a.in_service]
        if srcs:
            slack = torch.cat([index.rows_for_terminal(s.node, s.phases) for s in srcs])
        else:
            slack = torch.zeros(0, dtype=torch.int64)
        return cls(
            grid,
            orders,
            slack,
            dtype=dtype,
            device=device,
            anchor_v_weight=anchor_v_weight,
            anchor_i_weight=anchor_i_weight,
            edge_op=edge_op,
            branch_states=branch_states,
            tied_rows=tied_rows_of(grid, slack, branch_states),
            harmonic_slack=harmonic_slack,
            load_shunt=load_shunt,
        )


def _device_row_mask(injection_spec, n_rows: int, device) -> Optional[Tensor]:
    """bool ``[N]`` rows that carry an injecting DEVICE, or ``None`` when unknown.

    Read off a :class:`~injection_hse.data.InjectionSpec`'s device→terminal-row map (its padded
    ``device_rows`` restricted to the valid phase slots). ``None`` — the caller keeps every
    row — whenever the spec is absent or lists no device, which is what a dataset carrying no
    per-device injection looks like.
    """
    rows = getattr(injection_spec, "device_rows", None)
    valid = getattr(injection_spec, "device_rows_mask", None)
    if rows is None or valid is None or rows.numel() == 0:
        return None
    rows = rows[valid].reshape(-1).to(device=device, dtype=torch.int64)
    if rows.numel() == 0:
        return None
    mask = torch.zeros(n_rows, dtype=torch.bool, device=device)
    return mask.index_fill(0, rows, True)


def _prior_scale_rows(solver, n_rows: int, device, injection_spec) -> Tensor:
    """bool ``[N]`` rows the zero-injection PRIOR SCALE (``aux_scale``) is measured over.

    The prior sigma the noise-calibrated anchor weighting installs is meant to answer *how
    large is a device injection at this order* — so the statistic must be taken over the rows
    that carry a device, and nowhere else:

    - the SLACK rows are dropped: their ``Y·V`` entry is the total feeder current, not a
      device injection, and it is never predicted;
    - the rows with NO device are dropped: their true injection is zero to solver roundoff, so
      pooling them into an RMS reports a device injection smaller than any device actually
      makes — on a feeder where two thirds of the rows are passive junctions the scale comes
      out ~1.7× too tight, and with it the injection prior is over-trusted (weight ~2.9× too
      strong) at exactly the rows whose injection is largest and least certain.

    Without an :class:`~injection_hse.data.InjectionSpec` the device rows are unknown and every non-slack
    row is kept. A persisted ``aux_scale`` is reused as fit and never recomputed, so a
    checkpoint keeps the prior scale its own run was calibrated with.
    """
    keep = torch.ones(n_rows, dtype=torch.bool, device=device)
    if solver.slack_rows.numel():
        keep = keep.index_fill(0, solver.slack_rows.to(device), False)
    if solver.tied_rows.numel():
        # electrically the slack: their ``Y·V`` is a feeder total too
        keep = keep.index_fill(0, solver.tied_rows.to(device), False)
    devices = _device_row_mask(injection_spec, n_rows, device)
    if devices is not None:
        keep = keep & devices
    return keep


def fit_injection_normalizer(
    solver,
    train_v,
    encoding,
    *,
    magnitude_transform="log1p",
    base=None,
    injection_spec=None,
    exclude_tied: bool = True,
):
    """Fit a :class:`~injection_hse.normalization.HarmonicNormalizer` on the TRUE nodal injections.

    The injection-decode estimator predicts currents, whose per-order magnitudes span many
    orders of magnitude (kA at the fundamental, ~A at the harmonics); the normalizer puts them
    in a learnable feature range, the injection-space analogue of the voltage normalizer.
    ``train_v`` is a stack of true states ``[Bt, H, N]`` from the train split. ``encoding`` is
    the INJECTION head's :class:`~injection_hse.encoding.StateEncoding`
    (:func:`injection_hse.encoding.build_injection_encoding` — ``EncodingConfig.injection_kind``, else
    the state encoding), the space the model's raw output is decoded in.

    ``base`` (float64 ``[H, N]``, :meth:`NetworkSolver.i_base`) is the per-unit CURRENT base
    of the per-unit feature frame (:attr:`injection_hse.config.NormalizationConfig.per_unit`): the
    targets are divided by it, so the fitted statistics — and the ``aux_scale`` the
    noise-calibrated anchor weighting reads — describe the DIMENSIONLESS injections the model
    predicts. ``None`` (default) fits on amperes, byte-identical.

    ``aux_scale`` — the zero-injection PRIOR scale the noise-calibrated anchor weighting reads
    — answers *how large is a device injection at this order*, so it is the RMS over the rows
    that carry a device, named by ``injection_spec`` (the source's
    :class:`~injection_hse.data.InjectionSpec`). The slack row's ``Y·V`` entry is the whole feeder
    current and a passive junction's is zero to solver roundoff, so pooling either in reports
    a device injection no device actually makes. Without a spec — a dataset carrying no
    per-device injection — every non-slack row is kept. The fitted FEATURE statistics
    deliberately keep every passive row: the head predicts them too, and their exact zeros
    are part of the distribution it must reproduce. The SLACK rows are zeroed before the
    fit (:meth:`NetworkSolver.zero_slack`): their ``Y·V`` is the source's own current, which
    the head never predicts — left in, the feeder's fundamental current inflates the
    fundamental scale, and a slack carrying a harmonic voltage (an upstream background
    under ``harmonic_slack="measured"``) puts ``V_h / R_source`` — megaamperes for a stiff
    source — into the harmonic scales and the head's output range with it.
    """
    from injection_hse.normalization import HarmonicNormalizer

    i_target = solver.zero_slack(solver.injection_targets(train_v))  # [Bt, H, N]
    if exclude_tied:
        # the switch-tied rows' ``Y·V`` is the feeder's current, not a device injection,
        # and the head is asked for exactly zero there: fit on what it must reproduce
        i_target = solver.zero_tied(i_target)
    if base is not None:
        i_target = i_target / base.to(device=i_target.device, dtype=i_target.real.dtype)
    feats = encoding.encode(i_target).transpose(-3, -2)  # [Bt, N, H, C]
    normalizer = HarmonicNormalizer().fit(
        feats, magnitude_transform=magnitude_transform
    )
    keep = _prior_scale_rows(
        solver, i_target.shape[-1], i_target.device, injection_spec
    )
    device_rows = i_target[..., keep]  # [Bt, H, N_device]
    normalizer.aux_scale = (
        device_rows.abs()
        .pow(2)
        .mean(
            dim=tuple(d for d in range(device_rows.ndim) if d != device_rows.ndim - 2)
        )
        .sqrt()
        .clamp_min(1e-30)
        .to(torch.float64)
        .cpu()
    )  # [H]
    return normalizer


# =============================================================================
# Noise-calibrated anchor weights: the per-channel measurement sigmas
# =============================================================================
def anchor_measurement_sigma(
    typical: Tensor,
    *,
    mag_sigma: Tensor,
    phase_sigma_rad: Tensor,
    floor_sigma: Optional[Tensor] = None,
    order_floor_scale: Optional[Tensor] = None,
) -> Tensor:
    """float64 ``[H, R]`` per-REAL-COMPONENT error std of a sensor fleet's channels.

    The measurement error a :class:`~injection_hse.data.SensorNoise` model applies to a phasor of
    magnitude ``typical(h)`` — a multiplicative magnitude error ``σ_mag``, an additive phase
    error ``σ_phase`` (radians, a perpendicular displacement of the same relative size) and
    an additive range floor ``σ_floor`` per real component — has, per real component,
    the variance::

        σ(h, r)² = ½·(σ_mag(r)² + σ_phase(r)²)·typical(h)² + (σ_floor(r)·order_floor(h))²

    (the reading errors spread isotropically over the two components, hence the ½; the floor
    is already per component). This is the ``σ_meas`` of the inverse-variance anchor weight
    ``σ_I² / σ_meas²`` (:meth:`NetworkSolver.set_anchor_calibration`); feeding it the noise
    model's own sigmas (``SensorNoise.mag_sigma`` / ``phase_sigma_rad`` / ``floor_sigma`` /
    ``order_floor_scale``) makes the weights and the applied noise agree by construction.

    Parameters
    ----------
    typical:
        float ``[H]`` typical phasor magnitude per order (the fitted normalizer's pooled
        ``cartesian_scale``), or ``[H, R]`` per order and channel (its ``row_scale``: each
        channel's own RMS magnitude), in the channel's units.
    mag_sigma, phase_sigma_rad:
        float ``[R]`` (or ``[1]``, broadcasting) class sigmas of the ``R`` channels.
    floor_sigma:
        float ``[R]`` (or ``[1]``) absolute per-component floor at the fundamental range, in
        the channel's units, or ``[H, R]`` when the floor differs per order — an instrument
        floor converted into a per-order FEATURE frame divides by that order's base; ``None``
        = no floor.
    order_floor_scale:
        float ``[H]`` per-order multiplier on the floor; ``None`` = the same floor at every
        order.

    Returns
    -------
    Tensor
        float64 ``[H, R]`` (``[H, 1]`` for broadcasting sigmas), on ``typical``'s device.
    """
    typ = torch.as_tensor(typical, dtype=torch.float64)
    dev = typ.device
    typ = typ.reshape(-1, 1) if typ.ndim < 2 else typ  # [H, 1] | [H, R]
    mag = torch.as_tensor(mag_sigma, dtype=torch.float64).to(dev).reshape(1, -1)
    phase = torch.as_tensor(phase_sigma_rad, dtype=torch.float64).to(dev).reshape(1, -1)
    var = 0.5 * (mag**2 + phase**2) * typ**2  # [H, R]
    if floor_sigma is not None:
        floor = torch.as_tensor(floor_sigma, dtype=torch.float64).to(dev)
        floor = floor.reshape(1, -1) if floor.ndim < 2 else floor  # [1, R] | [H, R]
        if order_floor_scale is not None:
            scale = torch.as_tensor(order_floor_scale, dtype=torch.float64).to(dev)
            floor = floor * scale.reshape(-1, 1)  # [H, R]
        var = var + floor**2
    return var.sqrt()


def clean_sensor_sigma(typical: Tensor, rel_sigma: float) -> Tensor:
    """float64 ``[H, 1]`` (``[H, R]`` for a per-channel ``typical``) measurement sigma of a
    NOISE-FREE channel: ``rel_sigma · typical``.

    A clean sensor is still not infinitely trusted — a zero sigma would be an infinite anchor
    weight, and the anchored correction system's conditioning grows with
    ``w·σ_max(Y⁻¹)²`` — so a small relative sigma (``PhysicsConfig.clean_sensor_rel_sigma``)
    caps the weight and keeps the solve finite. Broadcasts over the channel axis.
    """
    typ = torch.as_tensor(typical, dtype=torch.float64)
    return float(rel_sigma) * (typ.reshape(-1, 1) if typ.ndim < 2 else typ)


# =============================================================================
# Many grids: one operator per member
# =============================================================================
class MultiGridPhysicsBank:
    """One :class:`NetworkSolver` (and its current base) per member grid of a multi-grid run.

    A multi-grid batch mixes grids of different size, so there is no single ``Y(h)``: the
    physics decode has to dispatch per member, exactly as the switchable-topology
    :class:`~injection_hse.data.TopologyBank` dispatches per switch configuration. This bank is that
    dispatch table — index-aligned with :attr:`injection_hse.data.MultiGridDataSource.grids`, so a
    batch's ``grid_index`` entry indexes it directly.

    Besides the solver, each member carries its per-unit CURRENT base
    (:meth:`i_base`): the magnitude decision of the multi-grid path is per-unit
    (:func:`injection_hse.data.per_unit_state` for voltages), and an injection current needs the same
    treatment or a 20 kV feeder's fundamental injection and a 400 V feeder's harmonic
    injection cannot share one normalizer. The base is off the autograd tape (grid constants
    in float64), like every other piece of batch metadata.

    LAZY BY MEMBER. A bank over a few hundred grids must not assemble a few hundred
    admittance matrices (and their singular values) at construction, and a run that never
    touches a member never pays for it: :meth:`solver`, :meth:`i_base` and
    :meth:`current_operator` build on first access and cache. The
    :class:`~injection_hse.data.BranchCurrentOperator` is the heaviest item to hold — building it for
    current ANCHORING additionally makes that member's solver materialise the dense per-order
    ``C(h)`` map ``[H, E, N]`` — so a voltage-only or current-free run never constructs one.

    Parameters
    ----------
    source:
        The :class:`~injection_hse.data.MultiGridDataSource` whose members the bank mirrors.
    orders:
        The solved harmonic orders; ``None`` (default) takes the source's.
    dtype:
        Complex dtype of the decode's inputs/outputs (the solve itself always runs in
        ``complex128``; see :meth:`NetworkSolver.assemble`).
    device:
        Device the operators are built on (``None`` = CPU); :meth:`to` moves them later.
    anchor_v_weight, anchor_i_weight:
        The measurement-anchoring CAPABILITY every member's solver is built with — sized, as
        on the single-grid path, for the largest weight any curriculum stage will request
        (:func:`injection_hse.train.anchor_build_weights`). A positive current weight forces that
        member's :class:`~injection_hse.data.BranchCurrentOperator` to be built with the solver (the
        anchored solve needs the ``V -> branch current`` map). The strength a given stage
        actually decodes with is the RUNTIME weight (:meth:`set_anchor_weights`).
    harmonic_slack:
        ``"norton"`` (default) or ``"measured"`` — how each member's harmonic solves treat
        its metered slack (:attr:`NetworkSolver.harmonic_slack`). Under ``"measured"`` the
        multi-grid decode hands every member its measured slack voltage at EVERY order, so
        a supply's own distortion enters the member through its boundary rows.
    load_shunt:
        The harmonic device-shunt model every member's ``Y(h)`` is assembled with
        (:attr:`NetworkSolver.load_shunt`, nameplate basis); ``None`` (default) keeps
        pgml's assembly defaults.
    """

    def __init__(
        self,
        source,
        orders: Optional[Sequence[int]] = None,
        *,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        anchor_v_weight: float = 0.0,
        anchor_i_weight: float = 0.0,
        harmonic_slack: str = "norton",
        load_shunt: Optional[str] = None,
    ) -> None:
        if harmonic_slack not in ("norton", "measured"):
            raise ValueError(
                f"harmonic_slack must be 'norton' or 'measured', got {harmonic_slack!r}."
            )
        _check_load_shunt(load_shunt)
        #: The decode's harmonic device-shunt model of every member
        #: (:attr:`NetworkSolver.load_shunt`).
        self.load_shunt = load_shunt
        self.source = source
        self._orders = [int(h) for h in (source.orders if orders is None else orders)]
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else None
        self.anchor_v_weight = float(anchor_v_weight)
        self.anchor_i_weight = float(anchor_i_weight)
        #: How every member's harmonic solves treat its metered slack
        #: (:attr:`NetworkSolver.harmonic_slack`): ``"measured"`` holds the slack rows at
        #: their measured harmonic voltage, ``"norton"`` lets them float behind the source.
        self.harmonic_slack = harmonic_slack
        # The stage's RUNTIME anchoring strength (``None`` = the built capability stands).
        # Kept beside the build weights because members are built LAZILY: a stage that dials
        # the anchor down must not become the capability a member built later is stuck with.
        self._runtime_anchor: Optional[tuple[float, float]] = None
        # The NOISE CALIBRATION of the anchor weights, in the per-unit feature frame
        # (:meth:`set_anchor_calibration`); ``None`` = manual weighting. Like the runtime
        # anchor, it is kept on the bank so a member built later inherits it.
        self._anchor_calibration: Optional[dict] = None
        n = len(source.grids)
        self._solvers: list[Optional[NetworkSolver]] = [None] * n
        self._edge_ops: list[Optional[object]] = [None] * n
        self._i_base: list[Optional[Tensor]] = [None] * n
        self._device_base: list[Optional[Tensor]] = [None] * n
        # Shared factor-once cache budget (:meth:`enable_operator_cache`); ``None`` = off.
        self._cache_budget: Optional[OperatorCacheBudget] = None

    # -- identity -----------------------------------------------------------
    def __len__(self) -> int:
        """Member grids the bank serves (== ``len(source.grids)``)."""
        return len(self._solvers)

    @property
    def orders(self) -> list[int]:
        """The solved harmonic orders ``H`` every member's operator is built for."""
        return list(self._orders)

    @property
    def device(self) -> torch.device:
        """Device the member operators live on (CPU until :meth:`to` says otherwise)."""
        return self._device if self._device is not None else torch.device("cpu")

    # -- per-member operators -----------------------------------------------
    def solver(self, index: int) -> NetworkSolver:
        """The member's :class:`NetworkSolver` (built on first access, then cached)."""
        i = int(index)
        solver = self._solvers[i]
        if solver is None:
            bundle = self.source.bundle(i)
            edge_op = self.current_operator(i) if self.anchor_i_weight > 0.0 else None
            solver = NetworkSolver.from_grid(
                bundle.grid,
                self._orders,
                dtype=self._dtype,
                device=self._device,
                anchor_v_weight=self.anchor_v_weight,
                anchor_i_weight=self.anchor_i_weight,
                edge_op=edge_op,
                harmonic_slack=self.harmonic_slack,
                load_shunt=self.load_shunt,
            )
            # registered first: the calibration below reads the member's current base,
            # which is built from this very solver.
            self._solvers[i] = solver
            if self._runtime_anchor is not None:
                self._apply_runtime_anchor(solver)
            if self._anchor_calibration is not None:
                self._apply_anchor_calibration(i, solver)
            if self._cache_budget is not None:
                solver.enable_operator_cache(self._cache_budget)
        return solver

    def enable_operator_cache(self, budget_mb: int) -> "MultiGridPhysicsBank":
        """Cache the members' solve factorizations under ONE shared budget; returns self.

        ``budget_mb`` bounds the TOTAL bytes the factor-once caches of every member may
        hold (:meth:`NetworkSolver.enable_operator_cache`); members are charged in the
        order the batches first solve them, and a member the budget cannot fit keeps the
        direct per-call solve. ``0`` disables the caches on every built member.
        """
        budget = (
            OperatorCacheBudget(int(budget_mb) * 2**20) if int(budget_mb) > 0 else None
        )
        self._cache_budget = budget
        for solver in self._solvers:
            if solver is not None:
                solver.enable_operator_cache(budget)
        return self

    def set_decode_dtype(self, dtype: torch.dtype) -> "MultiGridPhysicsBank":
        """Set the complex dtype every member's decode reads/writes; returns self.

        The TRAINING precision is the module's decision (``TrainConfig.dtype``) and a bank may
        have been built before it was known, so the module re-declares it here: built members
        are updated and later ones inherit it. Only the decode's inputs/outputs are affected —
        the solve itself always runs in ``complex128`` (see :meth:`NetworkSolver.assemble`).
        """
        self._dtype = dtype
        for solver in self._solvers:
            if solver is not None:
                solver._dtype = dtype
        return self

    # -- the curriculum's anchoring schedule ---------------------------------
    def set_anchor_weights(
        self, *, v_weight: float, i_weight: float
    ) -> "MultiGridPhysicsBank":
        """Dial the RUNTIME anchoring strength of every member; returns self.

        The multi-grid counterpart of flipping ``NetworkSolver.anchor_*_weight`` per
        curriculum stage (:meth:`injection_hse.train.SEModule.set_stage_anchor`): the grid-constant
        anchoring tensors stay as built, only the scalars change. Members built AFTER this
        call inherit the same runtime weights, which is why the bank owns the schedule rather
        than the caller looping over the built solvers — a member first drawn in a later batch
        must decode with the stage's strength, not the build capability.

        Raises
        ------
        ValueError
            If a positive current weight is requested from a bank built without the
            branch-current capability (``anchor_i_weight = 0``): its members carry no
            ``V -> branch current`` map to constrain the solve with.
        """
        if float(i_weight) > 0.0 and self.anchor_i_weight <= 0.0:
            raise ValueError(
                "a curriculum stage sets anchor_current_weight > 0 but the operator bank was "
                "built without the branch-current map — build it with anchor_i_weight (the "
                "largest weight any stage requests) and enable measurement.measure_current."
            )
        self._runtime_anchor = (float(v_weight), float(i_weight))
        for solver in self._solvers:
            if solver is not None:
                self._apply_runtime_anchor(solver)
        return self

    def _apply_runtime_anchor(self, solver: NetworkSolver) -> None:
        """Install the runtime anchor scalars on one member's solver."""
        wv, wi = self._runtime_anchor
        solver.set_anchor_weights(v_weight=wv, i_weight=wi)

    # -- noise-calibrated anchor weights ------------------------------------
    def set_anchor_calibration(
        self,
        *,
        sigma_injection: Tensor,
        sigma_voltage: Tensor,
        sigma_current: Optional[Tensor] = None,
        voltage_per_unit: bool = True,
    ) -> "MultiGridPhysicsBank":
        """Install NOISE-CALIBRATED anchor weights on every member; returns self. Idempotent.

        The multi-grid counterpart of :meth:`NetworkSolver.set_anchor_calibration`. A
        multi-grid run has ONE set of fitted scales and one uniform (default-class) noise
        model, both in the per-unit FEATURE frame, while each member's solve runs in its own
        absolute units — so the sigmas are given here per ORDER in the feature frame and
        converted per member with that member's bases (off the tape, float64):

        - the injection prior ``σ_I[h, r] = sigma_injection[h] · i_base[h, r]`` (the
          injection normalizer is fit on currents divided by :meth:`i_base`);
        - the voltage measurement ``σ_y[h, r] = sigma_voltage[h] · V_base[r]`` when
          ``voltage_per_unit`` (the noise model perturbs the per-unit state, so its error is
          per-unit), else ``sigma_voltage[h]`` in volts;
        - the CT channel ``σ_i[h, e] = sigma_current[h] · i_base[h, src(e)]`` (the edge
          features are the half-edge currents over the base at their SOURCE row).

        One dimensionless trust per order therefore reaches every member scaled by its own
        operator and its own voltage classes, which is what makes a single calibration valid
        across the corpus. What the resulting weight ``(σ_I/σ_y)²`` retains is the RATIO of
        the two bases at each row, ``i_base[h, r]/V_base[r] = s(h)·V_ref²/V_base[r]²`` — the
        row's own trade between its KCL equation and its voltage anchor. It is uniform across
        a single-voltage-class member (where it is ``s(h)``, the member's operator scale) and
        correctly favours the KCL equation at the rows of a higher class, whose injection is
        the same power at a smaller current and hence known more tightly in amperes. Members
        built later inherit the calibration.

        Parameters
        ----------
        sigma_injection:
            float64 ``[H]`` prior error of the predicted injection, per-unit
            (``ρ · s_I(h)`` with ``s_I`` the injection normalizer's ``cartesian_scale``).
        sigma_voltage:
            float64 ``[H]`` (or ``[H, 1]``) per-real-component voltage measurement sigma in
            the feature frame (:func:`anchor_measurement_sigma` on the uniform noise model, or
            :func:`clean_sensor_sigma`).
        sigma_current:
            float64 ``[H]`` (or ``[H, 1]``) per-real-component CT sigma in per-unit current;
            ``None`` = no current calibration (the members raise on a positive current
            multiplier). Only consumed by a bank built with the branch-current capability.
        voltage_per_unit:
            Whether the state the noise model perturbs is per-unit
            (``data.multigrid.per_unit``).
        """
        calibration = {
            "sigma_injection": torch.as_tensor(
                sigma_injection, dtype=torch.float64
            ).reshape(-1),
            "sigma_voltage": torch.as_tensor(
                sigma_voltage, dtype=torch.float64
            ).reshape(-1),
            "sigma_current": (
                None
                if sigma_current is None
                else torch.as_tensor(sigma_current, dtype=torch.float64).reshape(-1)
            ),
            "voltage_per_unit": bool(voltage_per_unit),
        }
        h = len(self._orders)
        for key in ("sigma_injection", "sigma_voltage", "sigma_current"):
            value = calibration[key]
            if value is not None and value.numel() != h:
                raise ValueError(
                    f"{key} must hold one value per solved order (H={h}); got "
                    f"{value.numel()}."
                )
        self._anchor_calibration = calibration
        for i, solver in enumerate(self._solvers):
            if solver is not None:
                self._apply_anchor_calibration(i, solver)
        return self

    def clear_anchor_calibration(self) -> "MultiGridPhysicsBank":
        """Return every member (built or later) to MANUAL anchor weighting; returns self."""
        self._anchor_calibration = None
        for solver in self._solvers:
            if solver is not None:
                solver.clear_anchor_calibration()
        return self

    def _apply_anchor_calibration(self, index: int, solver: NetworkSolver) -> None:
        """Convert the feature-frame calibration into one member's absolute units and install it."""
        cal = self._anchor_calibration
        i = int(index)
        i_base = self.i_base(i)  # float64 [H, N], the member's device
        dev = i_base.device
        sigma_inj = cal["sigma_injection"].to(dev).unsqueeze(-1) * i_base  # [H, N]
        if cal["voltage_per_unit"]:
            v_base = self.source.bundle(i).base_voltage.to(
                device=dev, dtype=torch.float64
            )
            v_base = torch.where(v_base > 0, v_base, torch.ones_like(v_base))  # [N]
        else:
            v_base = torch.ones(i_base.shape[-1], dtype=torch.float64, device=dev)
        sigma_v = cal["sigma_voltage"].to(dev).unsqueeze(-1) * v_base.unsqueeze(0)
        sigma_i = None
        if cal["sigma_current"] is not None and self.anchor_i_weight > 0.0:
            src = self.current_operator(i).edge_src_rows.to(
                device=dev, dtype=torch.int64
            )
            e_base = i_base.index_select(1, src)  # [H, E]
            sigma_i = cal["sigma_current"].to(dev).unsqueeze(-1) * e_base
        solver.set_anchor_calibration(
            sigma_injection=sigma_inj, sigma_voltage=sigma_v, sigma_current=sigma_i
        )

    def current_operator(self, index: int):
        """The member's :class:`~injection_hse.data.BranchCurrentOperator` (built on first access).

        The state -> half-edge current map behind measured-current inputs and current
        anchoring. Built lazily because it is the bank's heaviest per-member object: a
        voltage-only multi-grid run holds none of them.

        Its half-edge layout is verified against the member's own graph, so an operator can
        never be paired with a batch whose edge columns mean something else.
        """
        from injection_hse.data.currents import BranchCurrentOperator

        i = int(index)
        op = self._edge_ops[i]
        if op is None:
            bundle = self.source.bundle(i)
            op = BranchCurrentOperator(bundle.grid, self._orders, dtype=self._dtype)
            src = bundle.graph.edge_index[0].to(op.edge_src_rows.dtype)
            if op.n_edges != bundle.n_edges or not torch.equal(op.edge_src_rows, src):
                raise ValueError(
                    f"grid {bundle.key!r}: the branch-current operator's half-edge layout "
                    f"({op.n_edges} edges) does not match its graph ({bundle.n_edges}); the "
                    "edge observations would be gathered at the wrong rows."
                )
            if self._device is not None:
                op.to(self._device)
            self._edge_ops[i] = op
        return op

    # -- the per-unit current base ------------------------------------------
    def i_base(self, index: int) -> Tensor:
        """float64 ``[H, N]`` per-order, per-row CURRENT base of one member (off the tape).

        ``i_base[h, n] = s(h)·V_ref² / V_base[n]``, :meth:`NetworkSolver.i_base` evaluated on
        the member's own operator scale, reference voltage and per-row base voltages — the
        convention has ONE definition and the bank keeps no second formula. Dividing a nodal
        injection current by it gives the dimensionless quantity a cross-grid normalizer can
        pool: it removes the voltage class (INVERSELY, as a power base per row voltage must —
        see :meth:`NetworkSolver.i_base`) and the network's stiffness / size (through
        ``s(h)``, which grows with the short-circuit strength the injection is pushed
        against). Each member's reference is its OWN
        (:func:`reference_base_voltage`), so a member's base never depends on the rest of the
        corpus.

        A row with no rated voltage keeps base ``1`` (unscaled), matching
        :func:`injection_hse.data.per_unit_state`. Cached per member, on the bank's device.

        The base is a UNIT CONVENTION, not a learnable quantity: it is detached from the grid
        parameters, so a per-unit target/feature never back-propagates into ``Y`` through its
        own scale (as with :meth:`NetworkSolver.injection_targets`, the constant is enforced
        here rather than left to call-site discipline).
        """
        i = int(index)
        base = self._i_base[i]
        if base is None:
            base = self.solver(i).i_base(self.source.bundle(i).base_voltage)  # [H, N]
            self._i_base[i] = base
        return base

    def device_current_base(self, index: int) -> Tensor:
        """float64 ``[D, O]`` current base of one member's INJECTING DEVICES (off the tape).

        The per-device localization target is a per-order injection MAGNITUDE in amperes; this
        is the divisor that makes it dimensionless and comparable across voltage classes, the
        device-axis view of :meth:`i_base`. ``D`` are the member's devices and ``O`` its
        injected orders (:class:`~injection_hse.data.InjectionSpec`), and each entry is
        ``i_base[pos(order), row]`` at the device's FIRST valid phase row — a device injects
        the same spectrum on each of its phases and its rows share one node (hence one base
        voltage), so the phase choice is immaterial and taking the first avoids a pooled
        statistic that a padded phase slot could bias. A device with no valid row (nothing to
        pool at) keeps base ``1``.

        Empty (``[0, O]``) for a member whose dataset carries no per-device harmonic
        injection.
        """
        i = int(index)
        cached = self._device_base[i]
        if cached is not None:
            return cached
        spec = self.source.bundle(i).injection_spec
        n_orders = len(self.source.injection_orders)
        base_device = self.device
        if spec is None or spec.is_empty:
            out = torch.zeros((0, n_orders), dtype=torch.float64, device=base_device)
            self._device_base[i] = out
            return out
        missing = [o for o in spec.orders if int(o) not in self._orders]
        if missing:
            raise ValueError(
                f"grid {self.source.bundle(i).key!r} injects at order(s) {missing}, which "
                f"the bank does not solve ({self._orders}); the injection axes cannot be "
                "expressed in the per-unit current base."
            )
        rows = spec.device_rows.to(base_device)  # [D, P_max]
        valid = spec.device_rows_mask.to(base_device)  # [D, P_max]
        first = torch.argmax(valid.to(torch.int64), dim=-1, keepdim=True)  # [D, 1]
        row = rows.gather(-1, first).squeeze(-1)  # [D]
        pos = torch.tensor(
            [self._orders.index(int(o)) for o in spec.orders],
            dtype=torch.int64,
            device=base_device,
        )  # [O]
        base = self.i_base(i).index_select(0, pos).index_select(1, row)  # [O, D]
        base = base.transpose(0, 1).contiguous()  # [D, O]
        base = torch.where(valid.any(dim=-1, keepdim=True), base, torch.ones_like(base))
        self._device_base[i] = base
        return base

    # -- device placement ---------------------------------------------------
    def to(self, device) -> "MultiGridPhysicsBank":
        """Move every BUILT member operator + cached base onto ``device``; returns self.

        Members built later go straight to it. Like
        ``injection_hse.train.SEModule._sync_solver_device``, a solver already on the target device is
        left alone (the bank is a plain object, so Lightning never relocates it with the
        module — the training step calls this once per batch).
        """
        dev = torch.device(device)
        self._device = dev
        for i, solver in enumerate(self._solvers):
            if solver is not None and solver.slack_rows.device != dev:
                solver.to(dev)
            op = self._edge_ops[i]
            if op is not None and op.device != dev:
                op.to(dev)
            if self._i_base[i] is not None:
                self._i_base[i] = self._i_base[i].to(dev)
            if self._device_base[i] is not None:
                self._device_base[i] = self._device_base[i].to(dev)
        return self


def fit_multigrid_injection_normalizer(
    bank: MultiGridPhysicsBank,
    source,
    encoding,
    cfg,
    *,
    split: str = "train",
    max_scenarios_per_grid: Optional[int] = 8,
):
    """Fit ONE dataset-global injection :class:`~injection_hse.normalization.HarmonicNormalizer`.

    The multi-grid analogue of :func:`fit_injection_normalizer`: per member of ``split``, the
    true nodal injections ``I(h) = Y(h)·V_true(h)`` of a few scenarios
    (:meth:`NetworkSolver.injection_targets`), divided by that member's per-unit current base
    (:meth:`MultiGridPhysicsBank.i_base`), encoded and pooled over the node axis — the same
    construction :func:`injection_hse.train.fit_multigrid_normalizer` uses for the voltage state, in the
    current world.

    THE FRAME (why the states are read RAW). ``injection_targets`` applies the member's
    ``Y(h)``, and the referenced angle frame is a row-dependent diagonal rotation
    ``D(h)``, which does NOT commute with ``Y(h)``: ``Y·(D V) ≠ D·(Y V)``, so injections
    computed from an already-rotated state are not the grid's injections at all. The targets
    are therefore computed from the ABSOLUTE-frame states as persisted, and the RESULTING
    CURRENT is rotated afterwards with the SAME ``D`` the state gets
    (:func:`injection_hse.data.slack_frame_rotation` of the member's VOLTAGE states over its
    configured reference rows — the frame is defined by the voltages, exactly as the
    training step applies it to the predicted currents).
    That is the frame-consistent pairing: with ``v' = D v`` and ``i' = D i`` the residual
    ``i' - (D Y D^{-1}) v'`` is the rotation of ``i - Y v``, so the normalized injection
    features the model predicts live in the same frame as the voltage features it sees.
    ``data.multigrid.angle_reference="absolute"`` skips the rotation on both sides.

    THE PRIOR SCALE. ``aux_scale`` pools each member's DEVICE rows — the rows its
    :class:`~injection_hse.data.InjectionSpec` names, the slack rows and passive junctions left out — so
    the zero-injection prior the noise-calibrated anchor weighting installs describes how large
    a device injection is rather than how many rows have none. A member whose dataset carries
    no per-device injection contributes its non-slack rows.

    Parameters
    ----------
    bank:
        The member operators (must be index-aligned with ``source``).
    source:
        The :class:`~injection_hse.data.MultiGridDataSource`.
    encoding:
        The INJECTION head's :class:`~injection_hse.encoding.StateEncoding`
        (:func:`injection_hse.encoding.build_injection_encoding` — ``EncodingConfig.injection_kind``,
        else the state encoding the voltage features use).
    cfg:
        The :class:`~injection_hse.config.ExperimentConfig` (normalization knobs + the angle frame).
    split:
        Which member split to fit on (``"train"``).
    max_scenarios_per_grid:
        Scenarios taken per member (``None`` = all), so the fit sees every TOPOLOGY rather
        than mostly the largest member dataset.

    Returns
    -------
    injection_hse.normalization.HarmonicNormalizer
        Fitted on ``[M, 1, H, C]`` — every (sample, row) of every member.
    """
    from injection_hse.data.multigraph import slack_frame_rotation
    from injection_hse.normalization import HarmonicNormalizer

    multigrid = cfg.data.multigrid
    angle_reference = "slack" if multigrid is None else multigrid.angle_reference
    indices = source.split_indices(split)
    if not indices:
        raise ValueError(f"the multi-grid manifest lists no {split} grids to fit on.")

    parts: list[Tensor] = []
    aux_parts: list[Tensor] = []
    for i in indices:
        bundle = source.bundle(i)
        states = bundle.source.states  # ABSOLUTE frame, [S, (T,) H, N]
        if max_scenarios_per_grid is not None:
            states = states[: int(max_scenarios_per_grid)]
        solver = bank.solver(i)
        # the slack rows' ``Y·V`` is the source's own current (the feeder total at the
        # fundamental, ``V_h / R_source`` at a harmonic the slack carries) — never a device
        # injection the head predicts, so it must not set the feature scales
        i_target = solver.zero_slack(
            solver.zero_tied(solver.injection_targets(states.to(bank.device)))
        )
        i_pu = i_target / bank.i_base(i).to(i_target.real.dtype)
        if angle_reference != "absolute":
            # the rotation the VOLTAGES define (the training step rotates the predicted
            # currents by it): a current's own slack-row angle is the source current's,
            # not the frame's, and a zeroed slack row would define no rotation at all
            i_pu = i_pu * slack_frame_rotation(
                states.to(i_pu.device), bundle.reference_rows, orders=bank.orders
            )
        feats = encoding.encode(i_pu).transpose(-3, -2)  # [S, (T,) N_g, H, C]
        parts.append(feats.reshape(-1, 1, feats.shape[-2], feats.shape[-1]))
        # the member's DEVICE-row per-unit magnitudes for the prior scale (see below)
        keep = _prior_scale_rows(
            solver, i_pu.shape[-1], i_pu.device, bundle.injection_spec
        )
        aux_parts.append(
            i_pu[..., keep].abs().reshape(-1, i_pu.shape[-2], int(keep.sum()))
        )
    normalizer = HarmonicNormalizer().fit(
        torch.cat(parts),
        magnitude_transform=cfg.normalization.magnitude_transform,
        per_harmonic=cfg.normalization.per_harmonic,
    )
    # DEVICE-ROW prior scale (``aux_scale``), per-unit: the RMS injection of a row that
    # actually carries a device, which is what the noise-calibrated anchor weighting reads as
    # the zero-injection prior scale (:func:`_prior_scale_rows`; a member whose dataset
    # carries no per-device injection contributes all its non-slack rows).
    pooled = torch.cat(
        [a.transpose(-2, -1).reshape(-1, a.shape[1]) for a in aux_parts]
    )  # [M, H] — every (sample, device row) magnitude of every member
    normalizer.aux_scale = (
        pooled.pow(2).mean(dim=0).sqrt().clamp_min(1e-30).to(torch.float64).cpu()
    )
    return normalizer


def fit_multigrid_edge_normalizer(
    bank: MultiGridPhysicsBank,
    source,
    encoding,
    cfg,
    *,
    quantity: str = "current",
    split: str = "train",
    max_scenarios_per_grid: Optional[int] = 8,
):
    """Fit ONE dataset-global EDGE :class:`~injection_hse.normalization.HarmonicNormalizer`.

    The measured-branch-current input needs statistics that span the members exactly as the
    voltage and injection normalizers do, so the features are made dimensionless before they
    are pooled:

    - ``quantity="current"`` — the half-edge currents ``I_e(h)`` of a member
      (:meth:`MultiGridPhysicsBank.current_operator`) divided by that member's per-unit
      CURRENT base at the half-edge's SOURCE row, ``i_base[h, src(e)]``
      (:meth:`MultiGridPhysicsBank.i_base`). A CT reads the current at the terminal it sits
      on, so the row it is normalized against is that terminal's.
    - ``quantity="power"`` — the terminal apparent power ``S_e = V_src · I_e*`` divided by
      ``V_base[src(e)] · i_base[h, src(e)]``, the product of the two bases the factors use.

    THE FRAME. As for the injections, the physics runs in the ABSOLUTE frame (the operator
    combines voltages of several rows, whose frame rotations differ) and the RESULT is rotated
    into the run's frame afterwards, with the rotation the voltages define
    (:func:`injection_hse.data.slack_frame_rotation`) gathered at each half-edge's source row — the
    convention :meth:`injection_hse.train.SEModule._multigrid_step` feeds the model. The apparent power
    is invariant under it (``V·I*`` cancels the unit rotation), so the power features are the
    same in both frames.

    Parameters
    ----------
    bank:
        The member operators (index-aligned with ``source``).
    source:
        The :class:`~injection_hse.data.MultiGridDataSource`.
    encoding:
        The run's :class:`~injection_hse.encoding.StateEncoding`.
    cfg:
        The :class:`~injection_hse.config.ExperimentConfig` (normalization knobs + the angle frame).
    quantity:
        ``"current"`` (default) or ``"power"``.
    split:
        Which member split to fit on (``"train"``).
    max_scenarios_per_grid:
        Scenarios taken per member (``None`` = all).

    Returns
    -------
    injection_hse.normalization.HarmonicNormalizer
        Fitted on ``[M, 1, H, C]`` — every (sample, half-edge) of every member.
    """
    from injection_hse.data.multigraph import slack_frame_rotation
    from injection_hse.normalization import HarmonicNormalizer

    if quantity not in ("current", "power"):
        raise ValueError(f"quantity must be 'current' or 'power'; got {quantity!r}.")
    multigrid = cfg.data.multigrid
    angle_reference = "slack" if multigrid is None else multigrid.angle_reference
    indices = source.split_indices(split)
    if not indices:
        raise ValueError(f"the multi-grid manifest lists no {split} grids to fit on.")

    parts: list[Tensor] = []
    for i in indices:
        bundle = source.bundle(i)
        states = bundle.source.states  # ABSOLUTE frame, [S, (T,) H, N]
        if max_scenarios_per_grid is not None:
            states = states[: int(max_scenarios_per_grid)]
        states = states.to(bank.device)
        src = bundle.graph.edge_index[0].to(device=states.device, dtype=torch.int64)
        i_edges = bank.current_operator(i)(states)  # [S, (T,) H, E]
        e_base = bank.i_base(i).index_select(-1, src)  # [H, E]
        if quantity == "current":
            feat = i_edges / e_base.to(i_edges.real.dtype)
            if angle_reference != "absolute":
                rotation = slack_frame_rotation(
                    states, bundle.reference_rows, orders=bank.orders
                )
                feat = feat * rotation.index_select(-1, src).to(feat.dtype)
        else:
            v_base = bundle.base_voltage.to(device=states.device, dtype=torch.float64)
            v_base = torch.where(v_base > 0, v_base, torch.ones_like(v_base))
            s_base = (v_base.index_select(-1, src) * e_base).to(i_edges.real.dtype)
            feat = (states.index_select(-1, src) * i_edges.conj()) / s_base
        feats = encoding.encode(feat).transpose(-3, -2)  # [S, (T,) E, H, C]
        parts.append(feats.reshape(-1, 1, feats.shape[-2], feats.shape[-1]))
    return HarmonicNormalizer().fit(
        torch.cat(parts),
        magnitude_transform=cfg.normalization.magnitude_transform,
        per_harmonic=cfg.normalization.per_harmonic,
    )


def solve_injection(
    module: "SEModule",
    i_pred: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    i_edges: Optional[Tensor] = None,
    drop_orders: tuple = (),
) -> Tensor:
    """Nodal injection currents ``[B, H, N]`` (amperes) -> the decoded state ``[B, H, N]``.

    The module's own network solve at the measured fundamental slack voltage, including
    the DEPLOYED measurement anchoring when the solver anchors and the anchor inputs
    (``mask``, and the measured ``i_edges``/``edge_mask`` of a current-measuring module)
    are given. ``drop_orders`` removes those order indices from the anchor measurements,
    so an order occluded at the model input cannot re-enter through the anchors.

    The decode is affine in ``i_pred`` at fixed measurements, which is what lets a
    non-learned injection prior (zero, or the fitted linear one) be scored through
    exactly the estimator's deployed path.

    Package-internal: shared by the analysis probes and the training preconditioner, not
    part of the public API (not exported in ``__all__``) and free to change between
    releases.
    """
    module._sync_solver_device(i_pred.device)
    v_c = v.to(module._complex_dtype)
    if not (module.solver.anchoring and mask is not None):
        return module.solver.solve(i_pred, module._slack_voltage(v_c))
    b, n = mask.shape
    h = module.n_harmonics
    order_mask = mask.unsqueeze(-1).expand(b, n, h).clone()
    edge_order_mask = None
    if edge_mask is not None and i_edges is not None:
        e = int(edge_mask.shape[-1])
        edge_order_mask = edge_mask.unsqueeze(-1).expand(b, e, h).clone()
    for j in drop_orders:
        order_mask[..., j] = False
        if edge_order_mask is not None:
            edge_order_mask[..., j] = False
    return module.solver.solve(
        i_pred,
        module._slack_voltage(v_c),
        v_meas=v_c,
        v_mask=mask,
        i_meas=i_edges,
        edge_mask=edge_mask,
        order_mask=order_mask,
        edge_order_mask=edge_order_mask,
    )


__all__ = [
    "NetworkSolver",
    "MultiGridPhysicsBank",
    "anchor_measurement_sigma",
    "clean_sensor_sigma",
    "fit_injection_normalizer",
    "fit_multigrid_edge_normalizer",
    "fit_multigrid_injection_normalizer",
    "reference_base_voltage",
]
