"""Physics-derived node INPUT priors: the solver-level primitives behind
:attr:`injection_hse.config.PhysicsConfig.input_priors`.

A message-passing estimator sees the measurements of its few-hop neighbourhood; the
fundamental voltage at an unobserved row is, however, a GLOBAL function of the measurements
— the drop along the whole feeder, which a handful of layers cannot carry — and Kirchhoff's
current law ``I = Y·V`` needs a neighbour voltage MULTIPLIED by an admittance, which additive
edge messages do not represent exactly. Both quantities are cheap linear functions of the
measured state and the known network, so they are supplied as additional per-row input
channels; the network then learns the nonlinear correction on top of the best linear
estimate (the structure of learned-iterative reconstruction), and the channels are per row
and per-unit-scaled, so they carry over to a multi-grid run.

Two priors, each a function of ONE :class:`~injection_hse.physics.NetworkSolver` (one switch
configuration, one member grid) and the masked measured state:

- :func:`kcl_prior` — ``κ_r(h) = Σ_j Y(h)_rj · (mask_j · V_in,j(h))``, the nodal injection the
  OBSERVED neighbourhood implies, plus the fraction of each row's coupled rows that are
  observed (how complete the sum is);
- :func:`linear_state_prior` — the measurement-anchored solve of the ZERO injection prior,
  the MAP state under "no knowledge of the injections", with the prior's own anchor weights
  installed on the solver for that solve only.

Everything here runs under ``no_grad`` (the priors are inputs; nothing in them is learned),
honours the input device and dtype, and is vectorised over the leading batch / time axes.
:meth:`injection_hse.train.SEModule.input_priors` assembles the encoded, normalized channel block from
these (per switch configuration; :meth:`~injection_hse.train.multigrid.MultiGridMixin
._multigrid_input_priors` per member grid).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

#: The prior kinds in the CANONICAL order of their channel blocks (whatever order the config
#: lists them in): the ``kcl`` block (``H × C_inj + 1`` channels) precedes the
#: ``linear_state`` block (``H × C`` channels).
PRIOR_KINDS: tuple[str, ...] = ("kcl", "linear_state")


def prior_kinds(requested: Sequence[str]) -> list[str]:
    """The requested priors in the canonical block order (:data:`PRIOR_KINDS`)."""
    return [kind for kind in PRIOR_KINDS if kind in requested]


def prior_feature_width(
    priors: Sequence[str],
    *,
    n_orders: int,
    n_state_channels: int,
    n_injection_channels: int,
) -> int:
    """Per-row width ``Fp`` of the ``node_prior`` block for the given priors.

    ``kcl`` contributes ``n_orders × n_injection_channels + 1`` (the injection encoding's
    channels per order, plus the observed-neighbourhood fraction), ``linear_state``
    ``n_orders × n_state_channels`` (the state encoding's channels per order); an empty list
    is ``0`` — no block, no model input change.
    """
    width = 0
    if "kcl" in priors:
        width += int(n_orders) * int(n_injection_channels) + 1
    if "linear_state" in priors:
        width += int(n_orders) * int(n_state_channels)
    return width


def kcl_prior(
    solver,
    v_in: Tensor,
    mask: Tensor,
    order_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """The injection the OBSERVED neighbourhood implies, and how complete each sum is.

    Parameters
    ----------
    solver:
        The :class:`~injection_hse.physics.NetworkSolver` whose grid-constant operators ``Y(h)``
        (complex128, cached) define the nodal network law.
    v_in:
        Complex ``[*lead, H, N]`` — the MEASURED input state (noisy when the noise model
        is on), in the solver's units and frame.
    mask:
        bool ``[*lead, N]`` — the per-sample location mask (``True`` = observed).
    order_mask:
        Optional bool ``[*lead, N, H]`` — the per-order observability of a heterogeneous
        fleet; ``None`` = every observed row covers every order.

    Returns
    -------
    (kappa, fraction)
        ``kappa`` complex ``[*lead, H, N]`` in the solver's decode dtype: ``κ_r(h) = Σ_j
        Y(h)_rj · V_j(h)`` over the OBSERVED rows ``j`` (per order under an order mask) —
        exact where row ``r``'s whole coupled neighbourhood is observed, a partial sum
        otherwise. ``fraction`` float64 ``[*lead, N]`` — the fraction of row ``r``'s coupled
        rows (itself included; :meth:`~injection_hse.physics.NetworkSolver.coupling_mask`) observed
        under the location mask, in ``[0, 1]``. Both detached, on ``v_in``'s device.
    """
    with torch.no_grad():
        if order_mask is not None:
            observed = order_mask.transpose(-1, -2)  # [*lead, H, N]
        else:
            observed = mask.unsqueeze(-2)  # [*lead, 1, N]
        v_obs = torch.where(observed, v_in, torch.zeros_like(v_in))
        kappa = solver.injection_targets(v_obs)  # [*lead, H, N]
        coupling = solver.coupling_mask().to(device=mask.device, dtype=torch.float64)
        observed_rows = mask.to(torch.float64)  # [*lead, N]
        # Σ_j A_rj · mask_j over the coupled rows j of each row r
        count = observed_rows @ coupling.transpose(0, 1)  # [*lead, N]
        degree = coupling.sum(dim=-1).clamp_min(1.0)  # [N]
        fraction = count / degree
        return kappa, fraction


def prior_anchor_weights(physics, solver, *, currents: bool) -> tuple[float, float]:
    """The ``(voltage, current)`` anchor weights of the zero-injection prior solve.

    ``physics`` is the :class:`~injection_hse.config.PhysicsConfig`; ``currents`` says whether
    measured branch currents are available to anchor to (the current weight is ``0``
    otherwise). Under ``anchor_weighting="manual"`` the configured
    ``prior_anchor_voltage_weight`` / ``prior_anchor_current_weight`` are used as they are
    (dimensionless voltage trust / raw current weight). On a NOISE-CALIBRATED solver
    (:attr:`~injection_hse.physics.NetworkSolver.anchor_calibrated`) they are multipliers on the
    calibrated weight ``(σ_I/σ_meas)²``, whose ``σ_I = ρ · s_I(h)`` was installed for the
    DECODE's prior (``ρ = prior_injection_rel_sigma``); the zero-injection prior knows
    nothing about the injections, so its sigma is the order's RMS injection itself (``ρ =
    1``), which the multipliers realise by the factor ``1/ρ²``.
    """
    wv = float(physics.prior_anchor_voltage_weight)
    wi = float(physics.prior_anchor_current_weight) if currents else 0.0
    if getattr(solver, "anchor_calibrated", False):
        rho = physics.prior_injection_rel_sigma
        if isinstance(rho, list):
            raise ValueError(
                "the zero-injection prior solve rescales the calibrated weights by one "
                "factor 1/rho^2; a per-order physics.prior_injection_rel_sigma is not "
                "supported with physics-derived input priors."
            )
        scale = 1.0 / float(rho) ** 2
        wv, wi = wv * scale, wi * scale
    return wv, wi


def linear_state_prior(
    solver,
    v_in: Tensor,
    v_slack: Tensor,
    mask: Tensor,
    order_mask: Optional[Tensor],
    i_edges: Optional[Tensor],
    edge_mask: Optional[Tensor],
    edge_order_mask: Optional[Tensor],
    *,
    v_weight: float,
    i_weight: float,
) -> Tensor:
    """The measurement-anchored solve of the ZERO injection prior: complex ``[*lead, H, N]``.

    ``solver.solve(0, v_slack, v_meas=v_in, v_mask=mask, …)`` with the prior's own anchor
    weights (:func:`prior_anchor_weights`) installed on the solver for THIS solve and the
    solver's previous weights restored afterwards — the decode's anchoring schedule is left
    untouched, and the solver's weight setters only flip scalars once the grid-constant
    anchoring tensors exist (the current map is materialised on first use and cached). The
    result is the MAP state under "no knowledge of the injections" at every row: the slack
    voltage propagated through the network, pulled toward every observed voltage row (per
    order under ``order_mask``) and, with a positive ``i_weight``, toward every measured
    branch current ``i_edges`` ``[*lead, H, E]`` at the observed half-edges.

    ``v_in`` complex ``[*lead, H, N]``, ``v_slack`` complex ``[*lead, K]`` (the measured
    fundamental slack voltage, the hard boundary), ``mask`` bool ``[*lead, N]``,
    ``order_mask`` bool ``[*lead, N, H]`` | None, ``edge_mask`` bool ``[*lead, E]`` | None,
    ``edge_order_mask`` bool ``[*lead, E, H]`` | None. Detached, in the solver's decode dtype,
    on ``v_in``'s device; with both weights ``0`` it is the slack-only solve of zero
    injections (the flat slack voltage).
    """
    with torch.no_grad():
        saved = (solver.anchor_v_weight, solver.anchor_i_weight)
        solver.set_anchor_weights(v_weight=float(v_weight), i_weight=float(i_weight))
        try:
            return solver.solve(
                torch.zeros_like(v_in),
                v_slack,
                v_meas=v_in,
                v_mask=mask,
                order_mask=order_mask,
                i_meas=i_edges if i_weight > 0.0 else None,
                edge_mask=edge_mask if i_weight > 0.0 else None,
                edge_order_mask=edge_order_mask if i_weight > 0.0 else None,
            )
        finally:
            solver.set_anchor_weights(v_weight=saved[0], i_weight=saved[1])


__all__ = [
    "PRIOR_KINDS",
    "prior_kinds",
    "prior_feature_width",
    "kcl_prior",
    "prior_anchor_weights",
    "linear_state_prior",
]
