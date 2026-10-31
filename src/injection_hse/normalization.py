"""Per-harmonic normalization of the encoded state (fit on train, persisted with the model).

Harmonic voltages span orders of magnitude (fundamental ≈ 1 pu, h11/h13 ≈ 1e-3 pu). Without
per-order normalization the high orders are numerically invisible and a plain MSE is
dominated by the fundamental. :class:`HarmonicNormalizer` standardizes per harmonic order so
each is perceptible; for the trig encoding this targets the magnitude channel (``log1p`` then
z-score, given the heavy tail), passing ``cosθ/sinθ`` through. The same fitted per-harmonic
statistics standardize the Cartesian residual in the loss so harmonics stay comparable.
Rationale: ``docs/pgl/index.md`` §4–§5.

The fitted statistics are part of the model artifact (``state_dict`` / ``load_state_dict``)
so the dev box and the cluster reproduce identically and inference de-normalizes correctly.

Shapes
------
All feature tensors are ``[*batch, N, H, C]`` (``C`` = encoding channels; trig ``C=3`` with
channel 0 the magnitude; cartesian ``C=2`` = ``(re, im)``). Statistics are per ``(H, C)`` and
broadcast over leading/spatial axes. :meth:`loss_standardize` consumes the Cartesian
``(re, im)`` reconstruction ``[*batch, N, H, 2]``. Honors the input device/dtype; all ops are
plain differentiable real torch (run on CPU/CUDA unchanged).
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


class HarmonicNormalizer:
    """Standardize encoded features per harmonic order; invertible; persistable.

    Statistics (``mean``/``std`` ``[H, C]`` and a per-harmonic Cartesian scale ``[H]``) are
    fit on the TRAIN split only (:meth:`fit`) and travel with the checkpoint
    (:meth:`state_dict` / :meth:`load_state_dict`).

    ``aux_scale`` (``[H]``, ``None`` unless a fit helper set one; persisted alongside the
    statistics) is an optional SECONDARY per-order scale that does not take part in
    :meth:`transform` / :meth:`inverse_transform` / :meth:`loss_standardize`.
    :func:`injection_hse.physics.fit_injection_normalizer` records the SLACK-MASKED RMS injection
    magnitude there — the zero-injection prior scale the noise-calibrated anchor weighting
    reads (:meth:`injection_hse.physics.NetworkSolver.set_anchor_calibration`), kept apart from the
    fitted feature statistics because those deliberately cover every row while the slack
    row's ``Y·V`` entry is the total feeder current rather than a device injection.
    """

    def __init__(self) -> None:
        self.magnitude_transform: str = "log1p"
        # whether statistics are estimated per harmonic order (default) or shared across the
        # harmonic axis. Recorded for introspection; the behaviour is fully baked into the
        # (broadcast-to-[H]) fitted statistics, so transform/inverse/loss need no branch.
        self.per_harmonic: bool = True
        # channel mask: which channels are standardized (magnitude/cartesian) vs passed
        # through (cos/sin). Filled by fit().
        self._standardize_channel: Tensor | None = None  # bool [C]
        self._log_channel: Tensor | None = None  # bool [C]
        self.mean: Tensor | None = None  # [H, C]
        self.std: Tensor | None = None  # [H, C]
        self.cartesian_scale: Tensor | None = None  # [H]
        # Optional secondary per-order scale a fit helper may attach beside the fitted
        # statistics (the injection normalizer carries the SLACK-MASKED RMS injection
        # magnitude here — the honest zero-injection prior scale the noise-calibrated
        # anchor weighting reads; the slack row's entry of Y·V is the total feeder
        # current, which would otherwise dominate the fundamental RMS). ``None`` when the
        # fit did not set one; persisted with the statistics, absent in older checkpoints.
        self.aux_scale: Tensor | None = None  # [H]
        # The RMS complex magnitude of every ROW (node-phase row or half-edge) at every
        # order, ``[H, N]``: the typical size of each channel's own reading, which the
        # noise-calibrated anchor weighting evaluates the sensor error at. The pooled
        # ``cartesian_scale`` mixes rows of different voltage classes and loadings (20 kV
        # beside 0.4 kV rows, feeder heads beside laterals), which misstates a single
        # reading's error by an order of magnitude at the fundamental. ``None`` in older
        # checkpoints (the calibration then falls back to the pooled scale).
        self.row_scale: Tensor | None = None  # [H, N]

    # -- fit -----------------------------------------------------------------
    def fit(
        self,
        feats: Tensor,
        *,
        magnitude_transform: str = "log1p",
        per_harmonic: bool = True,
    ) -> "HarmonicNormalizer":
        """Estimate the standardization statistics of the encoded TRAIN features. Returns self.

        ``feats`` ``[*batch, N, H, C]``. Trig (``C==3``): channel 0 (magnitude) is
        ``log1p``-compressed (if requested) then standardized; channels 1,2 (cos/sin) pass
        through (mean 0, std 1, no transform). Cartesian (``C==2``): both channels are
        standardized. The Cartesian SCALE (for the loss) is the RMS complex magnitude in
        PHYSICAL units, robust to the leading axes.

        ``per_harmonic`` (default ``True``) estimates a SEPARATE statistic for every harmonic
        order, so each order is standardized to unit scale and stays numerically visible
        despite spanning orders of magnitude — the standard behaviour. ``per_harmonic=False``
        shares ONE statistic across the harmonic axis: the standardization mean/std (of the
        transformed magnitude/Cartesian channels) AND the Cartesian loss scale are each pooled
        over all orders and broadcast back to ``[H]``, so every order divides by the same scale
        (the fundamental then dominates, the ablation of per-order visibility). The stored
        ``mean``/``std`` ``[H, C]`` and ``cartesian_scale`` ``[H]`` keep their shapes either
        way (the shared values are simply repeated across ``H``), so transform / inverse / loss
        are unchanged.
        """
        if feats.ndim < 2:
            raise ValueError("feats must be [*batch, N, H, C].")
        # Fitted statistics are CONSTANTS: never retain a caller's autograd graph
        # in the stored mean/std/scale.
        feats = feats.detach()
        c = feats.shape[-1]
        h = feats.shape[-2]
        self.magnitude_transform = magnitude_transform
        self.per_harmonic = per_harmonic
        flat = feats.reshape(-1, h, c)  # [M, H, C]
        per_row = feats.reshape(-1, feats.shape[-3], h, c)  # [M', N, H, C]

        if c == 3:  # trig: standardize magnitude only
            standardize = torch.tensor([True, False, False])
            log_ch = torch.tensor([magnitude_transform == "log1p", False, False])
            mag = flat[..., 0]  # channel 0 (== |V|) [M, H]
        elif c == 2:  # cartesian: standardize both, scale = RMS |re+jim|
            standardize = torch.tensor([True, True])
            log_ch = torch.tensor([False, False])
            mag = flat.pow(2).sum(dim=-1).sqrt()  # |re+jim| [M, H]
        else:
            raise ValueError(f"unsupported channel count C={c} (expected 2 or 3).")
        row_mag = (
            per_row[..., 0] if c == 3 else per_row.pow(2).sum(dim=-1).sqrt()
        )  # [M', N, H]
        row_scale = row_mag.pow(2).mean(dim=0).sqrt().transpose(0, 1)  # [H, N]

        proc = flat.clone()
        if log_ch.any():
            li = log_ch.nonzero(as_tuple=True)[0]
            proc[..., li] = torch.log1p(proc[..., li].clamp_min(0.0))

        if per_harmonic:
            mean = proc.mean(dim=0)  # [H, C]
            std = proc.std(dim=0, unbiased=False)  # [H, C]
            scale = mag.pow(2).mean(dim=0).sqrt()  # [H]
        else:
            # pool across BOTH the leading axes AND the harmonic axis: one shared statistic
            # per channel / one shared physical scale, repeated across every order.
            pooled = proc.reshape(-1, c)  # [M*H, C]
            mean = pooled.mean(dim=0, keepdim=True).expand(h, c).contiguous()  # [H, C]
            std = (
                pooled.std(dim=0, unbiased=False, keepdim=True)
                .expand(h, c)
                .contiguous()
            )
            scale = mag.reshape(-1).pow(2).mean().sqrt().expand(h).contiguous()  # [H]
        # only standardized channels get real mean/std; pass-through -> (0, 1)
        st = standardize.to(mean.device)
        mean = torch.where(st.unsqueeze(0), mean, torch.zeros_like(mean))
        std = torch.where(st.unsqueeze(0), std.clamp_min(_EPS), torch.ones_like(std))

        self._standardize_channel = standardize.to(feats.device)
        self._log_channel = log_ch.to(feats.device)
        self.mean = mean.to(feats.device, feats.dtype)
        self.std = std.to(feats.device, feats.dtype)
        self.cartesian_scale = scale.clamp_min(_EPS).to(feats.device, feats.dtype)
        self.row_scale = row_scale.to(feats.device, feats.dtype)
        return self

    # -- transform / inverse -------------------------------------------------
    def _ensure_fit(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("HarmonicNormalizer must be fit() before use.")

    def _align(self, ref: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Move fitted stats to ``ref``'s device/dtype; return (mean, std, log, stdz)."""
        mean = self.mean.to(ref.device, ref.dtype)
        std = self.std.to(ref.device, ref.dtype)
        log_ch = self._log_channel.to(ref.device)
        return mean, std, log_ch, self._standardize_channel.to(ref.device)

    def transform(self, feats: Tensor) -> Tensor:
        """Normalize encoded features ``[*batch, N, H, C]`` -> normalized, same shape."""
        self._ensure_fit()
        mean, std, log_ch, _ = self._align(feats)
        out = feats
        if bool(log_ch.any()):
            mask = log_ch.view(*([1] * (feats.ndim - 1)), -1)
            out = torch.where(mask, torch.log1p(feats.clamp_min(0.0)), feats)
        return (out - mean) / std

    def inverse_transform(self, feats: Tensor) -> Tensor:
        """Invert :meth:`transform` (e.g. on model outputs before decoding to complex).

        The ``expm1`` argument is clamped before the ``where``: ``torch.where``
        evaluates BOTH branches on every element, and the discarded trig
        channels (cos/sin) are scale-invariant in the decode — the loss never
        bounds their magnitude, so a long-trained model can drift them past the
        float32 ``expm1`` overflow. The overflowed (discarded) lane then turns
        the ``where`` backward into ``0 · inf = NaN`` and one such batch
        poisons every weight. The clamp bound (80 → ``expm1 ≈ 5.5e34``) is far
        beyond any physical magnitude, so selected lanes are unaffected.
        """
        self._ensure_fit()
        mean, std, log_ch, _ = self._align(feats)
        out = feats * std + mean
        if bool(log_ch.any()):
            mask = log_ch.view(*([1] * (feats.ndim - 1)), -1)
            out = torch.where(mask, torch.expm1(out.clamp(max=80.0)), out)
        return out

    def loss_standardize(self, cartesian: Tensor) -> Tensor:
        """Standardize the Cartesian ``(re, im)`` residual per harmonic for the loss.

        ``cartesian`` ``[*batch, N, H, 2]`` -> divided by the fitted per-harmonic Cartesian
        scale ``[H]`` so every order contributes comparably (the per-harmonic loss WEIGHT
        then sets priority on top — see :mod:`injection_hse.loss`). Independent of the encoding (one
        physical scale shared by trig and cartesian).
        """
        self._ensure_fit()
        scale = self.cartesian_scale.to(cartesian.device, cartesian.dtype)
        return cartesian / scale.view(*([1] * (cartesian.ndim - 2)), -1, 1)

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> dict:
        """Serializable fitted statistics (persisted with the checkpoint)."""
        self._ensure_fit()
        return {
            "magnitude_transform": self.magnitude_transform,
            "standardize_channel": self._standardize_channel.cpu(),
            "log_channel": self._log_channel.cpu(),
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "cartesian_scale": self.cartesian_scale.detach().cpu(),
            "aux_scale": (
                self.aux_scale.detach().cpu() if self.aux_scale is not None else None
            ),
            "row_scale": (
                self.row_scale.detach().cpu() if self.row_scale is not None else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted statistics from :meth:`state_dict`."""
        self.magnitude_transform = state["magnitude_transform"]
        self._standardize_channel = state["standardize_channel"].clone()
        self._log_channel = state["log_channel"].clone()
        self.mean = state["mean"].clone()
        self.std = state["std"].clone()
        self.cartesian_scale = state["cartesian_scale"].clone()
        aux = state.get("aux_scale")
        self.aux_scale = aux.clone() if aux is not None else None
        rows = state.get("row_scale")
        self.row_scale = rows.clone() if rows is not None else None


__all__ = ["HarmonicNormalizer"]
