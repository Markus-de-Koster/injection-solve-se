"""The loss preconditioner: curvature-normalised gradients for the injection head.

Behind :attr:`injection_hse.config.LossConfig.preconditioner`. The supervised loss of an injection
decode is, at fixed measurements, an exact quadratic in the head's output ``z``::

    L = || W · G · C · (z - z*) ||²

with ``G`` the decode's injection-to-state map, ``W`` the loss's own per-order
standardisation and ``C`` the head's frame. Its Hessian ``H = CᵀGᵀWᵀWGC`` is grid-constant
and badly conditioned: on the CIGRE LV benchmark the condition number at the 1 % cutoff is
~100 in every order and 30-43 % of the directions are flat (``σ < 1 % σ_max``) — the
difference between injecting at a passive junction and at its neighbour is nearly
invisible in the voltages, so gradient descent resolves the few steep directions and
leaves the rest untouched however long it runs (measured: longer training WORSENS the
deployed fundamental error while the held-out loss keeps falling).

The fix is the textbook one for a known quadratic: precondition the gradient with the
inverse curvature. This module builds, once before training, the exact map ``WGC`` per
order by finite differences on the module's own decode (exact — the decode is affine in
the injection), takes its SVD, and installs a backward-only transform on the head's output:
the forward pass is the identity, the backward pass multiplies ``∂L/∂z`` by

    P = σ_max² · V · diag(1 / (σ² + ε²)) · Vᵀ          ε = floor · σ_max

(``σ_max`` over all orders, so the floor has the same meaning as in the analysis). Along a
steep direction the step becomes the Gauss-Newton step; along a flat one the gradient is
lifted by up to ``1 / floor²`` — but multiplied by a residual sensitivity of ``σ``, so the
step stays bounded. Nothing changes in the forward pass: a checkpoint trained with the
preconditioner infers exactly like one trained without it, and the preconditioner is
rebuilt deterministically from the config and the grid whenever training resumes. A
per-parameter optimiser (AdamW) still normalises each weight's gradient by its own running
scale; what the preconditioner corrects is the MIXING between the head's output directions,
which no diagonal optimiser can see.

Which decode the curvature is read from is the ``preconditioner`` mode: ``"placement"`` —
the deployed measurement-anchored decode at the placement (the loss's own curvature where
the run is judged; it also floors the anchor-induced null directions); ``"plain"`` — the
slack-only decode (mask-independent, the network's own collinearity only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from injection_hse.config import ExperimentConfig

    from .module import SEModule

__all__ = [
    "LossPreconditioner",
    "attach_loss_preconditioner",
    "build_loss_preconditioner",
    "preconditioner_requested",
]


def preconditioner_requested(cfg: "ExperimentConfig") -> bool:
    """Whether ``cfg`` asks for the loss preconditioner."""
    return cfg.loss.preconditioner != "none"


class _PreconditionGradient(torch.autograd.Function):
    """Identity forward; the backward multiplies the gradient by ``P`` per order."""

    @staticmethod
    def forward(ctx, feats: Tensor, matrices: Tensor, rows: Tensor) -> Tensor:
        ctx.save_for_backward(matrices, rows)
        return feats.view_as(feats)

    @staticmethod
    def backward(ctx, grad: Tensor):
        matrices, rows = ctx.saved_tensors
        # grad [*lead, N, H, C] -> the (re, im) block of the non-slack rows, order-major
        g = grad.index_select(-3, rows)  # [*lead, R, H, C]
        lead = g.shape[:-3]
        r, h, c = g.shape[-3:]
        blocks = g.permute(*range(len(lead)), -2, -1, -3).reshape(*lead, h, c * r)
        mixed = torch.einsum("...hj,hij->...hi", blocks, matrices.to(blocks.dtype))
        mixed = mixed.reshape(*lead, h, c, r).permute(
            *range(len(lead)), -1, -3, -2
        )  # [*lead, R, H, C]
        out = grad.clone()
        out.index_copy_(-3, rows, mixed.contiguous())
        return out, None, None


class LossPreconditioner(torch.nn.Module):
    """The per-order gradient preconditioner of the injection head's output.

    Holds ``matrices`` ``[H, 2R, 2R]`` (float64, the ``P`` above, indexed re-block then
    im-block of the ``rows``), the non-slack ``rows`` ``[R]`` they act on, and the
    singular spectra ``[H, 2R]`` they were built from (a diagnostic, and what a resumed
    run checks itself against). A child module of :class:`~injection_hse.train.SEModule`, so it
    follows the device; nothing here is a trained parameter.
    """

    def __init__(
        self,
        matrices: Tensor,
        rows: Tensor,
        spectra: Tensor,
        *,
        mode: str,
        floor: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        self.register_buffer("matrices", matrices.to(torch.float64), False)
        self.register_buffer("rows", rows.to(torch.int64), False)
        self.register_buffer("spectra", spectra.to(torch.float64), False)
        self.mode = str(mode)
        self.floor = float(floor)
        self.sigma_max = float(sigma_max)

    @property
    def n_flat(self) -> list[int]:
        """Per order, how many directions lie under the floor (``σ < floor · σ_max``)."""
        return [
            int((s < self.floor * self.sigma_max).sum()) for s in self.spectra.unbind(0)
        ]

    def forward(self, feats: Tensor) -> Tensor:
        """Identity on ``feats`` ``[*lead, N, H, C]``; preconditions its gradient."""
        if not feats.requires_grad:
            return feats
        return _PreconditionGradient.apply(feats, self.matrices, self.rows)

    def summary(self) -> str:
        return (
            f"loss preconditioner ({self.mode}): σ_max {self.sigma_max:.3e}, floor "
            f"{self.floor:g}, flat directions per order {self.n_flat} of "
            f"{int(self.spectra.shape[-1])}"
        )


def _curvature_maps(module: "SEModule", observed_rows, *, mode: str) -> tuple:
    """``(maps [H, 2N, 2R], rows [R])`` — the loss's map ``W·G·C`` per order.

    Finite differences on the module's own decode chain: a unit in ONE real component of
    the head's output at one non-slack row and one order, pushed through the injection
    inverse-normalisation, the network solve (anchored at the placement or plain) and the
    loss's per-unit, per-order standardisation. Exact, because that chain is affine in
    the head's output at fixed measurements (the Cartesian injection encoding is required
    for that, and the config enforces it).
    """
    from injection_hse.physics import solve_injection

    solver = module.solver
    n = int(module.n_nodes)
    n_h = int(module.n_harmonics)
    c_out = int(module.n_out_channels)
    keep = torch.ones(n, dtype=torch.bool)
    if solver.slack_rows.numel():
        keep = keep.index_fill(0, solver.slack_rows.cpu(), False)
    if module._tied_rows.numel():
        keep = keep.index_fill(0, module._tied_rows.cpu(), False)
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    r = int(rows.numel())
    device = module.base_v.device
    cdt = module._complex_dtype
    real_dt = torch.float64

    mask = None
    if mode == "placement":
        if observed_rows is None or len(observed_rows) == 0:
            raise ValueError(
                "loss.preconditioner='placement' needs the deployed sensor placement "
                "(a 'placement' curriculum stage or measurement.voltage_nodes)."
            )
        mask = torch.zeros((c_out * r + 1, n), dtype=torch.bool, device=device)
        mask[:, torch.as_tensor(list(observed_rows), dtype=torch.int64)] = True

    base_v = module.base_v.to(real_dt)
    s = module.normalizer.cartesian_scale.to(real_dt)  # [H]
    w = module.loss_fn._weights.to(real_dt)  # [H]
    row_scale = (torch.sqrt(w) / s).view(n_h, 1) / base_v.view(1, n)  # [H, N]

    maps = []
    for k in range(n_h):
        feats = torch.zeros(
            (c_out * r + 1, n, n_h, c_out), dtype=module._real_dtype, device=device
        )
        for c in range(c_out):
            for j in range(r):
                feats[1 + c * r + j, rows[j], k, c] = 1.0
        with torch.no_grad():
            i_pred = module._from_injection_frame(
                module.inj_encoding.decode(
                    module.inj_normalizer.inverse_transform(feats)
                ).transpose(-2, -1)
            ).to(cdt)  # [B, H, N]
            v_meas = torch.zeros_like(i_pred)
            sol = solve_injection(module, i_pred, v_meas, mask=mask).to(
                torch.complex128
            )
        resp = (sol[1:, k] - sol[0, k]) * row_scale[k].to(torch.complex128)  # [2R, N]
        real = torch.cat([resp.real, resp.imag], dim=-1)  # [2R, 2N]
        maps.append(real.T)  # [2N, 2R]
    return torch.stack(maps), rows


def build_loss_preconditioner(
    module: "SEModule",
    observed_rows: Optional[Sequence[int]],
    *,
    mode: str,
    floor: float,
) -> LossPreconditioner:
    """Build the preconditioner from the module's own decode (see the module docstring).

    Runs on the module's current device in float64; the solver's runtime anchor weights
    are left exactly as found (``"placement"`` temporarily installs the config's deployed
    voltage/current weights for the probe solves).
    """
    solver = module.solver
    if solver is None or module._bank is not None:
        raise ValueError(
            "the loss preconditioner reads ONE grid's injection decode "
            "(physics.decode='injection' on a single-grid run)."
        )
    if int(module.n_out_channels) != 2:
        raise ValueError(
            "the loss preconditioner needs the Cartesian injection encoding "
            "(encoding.injection_kind='cartesian'): the decode chain is affine in the "
            "head's output only there."
        )
    saved = (float(solver.anchor_v_weight), float(solver.anchor_i_weight))
    try:
        if mode == "placement":
            solver.anchor_v_weight = float(module.cfg.physics.anchor_voltage_weight)
            solver.anchor_i_weight = (
                float(module.cfg.physics.anchor_current_weight)
                if solver.can_anchor_current
                else 0.0
            )
        maps, rows = _curvature_maps(module, observed_rows, mode=mode)
    finally:
        solver.anchor_v_weight, solver.anchor_i_weight = saved
    maps = maps.cpu()
    n_h, two_n, two_r = maps.shape
    spectra = torch.zeros((n_h, two_r), dtype=torch.float64)
    bases = []
    for k in range(n_h):
        _, sv, vt = torch.linalg.svd(maps[k], full_matrices=False)
        spectra[k, : sv.numel()] = sv
        bases.append(vt.T)  # [2R, min(2N, 2R)]
    sigma_max = float(spectra.max())
    if not sigma_max > 0.0:
        raise ValueError(
            "the loss curvature map is identically zero; nothing to precondition."
        )
    eps = float(floor) * sigma_max
    eye = torch.eye(two_r, dtype=torch.float64)
    matrices = torch.zeros((n_h, two_r, two_r), dtype=torch.float64)
    for k in range(n_h):
        v = bases[k]
        sv = spectra[k, : v.shape[1]]
        d = sigma_max**2 / (sv**2 + eps**2)
        matrices[k] = (v * d) @ v.T + (eye - v @ v.T) * (sigma_max**2 / eps**2)
    return LossPreconditioner(
        matrices, rows, spectra, mode=mode, floor=floor, sigma_max=sigma_max
    ).to(module.base_v.device)


def attach_loss_preconditioner(
    module: "SEModule",
    cfg: "ExperimentConfig",
    observed_rows: Optional[Sequence[int]],
) -> Optional[LossPreconditioner]:
    """Build and install the preconditioner BEFORE training; a no-op unless requested."""
    if not preconditioner_requested(cfg):
        return None
    if cfg.physics.decode != "injection":
        raise ValueError(
            "loss.preconditioner needs physics.decode='injection': it preconditions the "
            "injection head's output, which the voltage decode does not have."
        )
    pre = build_loss_preconditioner(
        module,
        observed_rows,
        mode=cfg.loss.preconditioner,
        floor=cfg.loss.preconditioner_floor,
    )
    module.set_loss_preconditioner(pre)
    print(f"  {pre.summary()}", flush=True)
    return pre
