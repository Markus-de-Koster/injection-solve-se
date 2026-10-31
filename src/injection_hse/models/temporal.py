"""Temporal estimator: a pluggable CAUSAL-with-memory wrapper around a spatial model.

The snapshot models estimate each time step INDEPENDENTLY (the ``T`` axis is carried by
the :class:`~injection_hse.io.ModelInput` but not mixed). For node-coherent sequences
(``injection_hse.data.scenarios.CoherentSpectrumConfig``) the disturbance pattern persists over time, so an
estimator that REMEMBERS earlier steps de-mixes the state better. :class:`TemporalEstimator`
adds that memory while keeping the spatial model swappable and the mixing CAUSAL — the estimate
at step ``t`` uses only steps ``<= t`` (plus a running hidden state), which is exactly what the
live / streaming use case needs.

Data flow (``kind="gru"`` — the implemented mechanism)
------------------------------------------------------
For ``node_obs [*B, T, N, Fnd]`` + ``node_mask [*B, T, N]``:

1. Form the per-(node, step) token ``[node_obs ⧺ mask] [*B, T, N, Fnd+1]``.
2. Run a PER-NODE GRU over the ``T`` axis (the node axis is folded into the batch, so each
   node-phase row carries its own recurrent memory): ``[(*B·N), T, Fnd+1]`` -> a temporal
   context ``[(*B·N), T, hidden]``. The GRU hidden state at step ``t`` depends only on inputs
   ``<= t``, so the context is CAUSAL by construction.
3. Project the context to ``Fnd`` and ADD it as a temporal RESIDUAL to ``node_obs`` (so the
   spatial model's input width stays ``Fnd`` — any DNN/GNN/Graphormer slots in unchanged;
   its output width ``Fout`` is the spatial model's own, the recurrence never sees it). Build
   a new :class:`~injection_hse.io.ModelInput` with the enriched ``node_obs`` (mask / static / edges /
   ``node_time`` unchanged) and run the wrapped spatial model, which folds ``T`` into the
   batch and estimates per step.

The optional per-step ``node_time`` features and the optional per-row ``node_prior`` block
(the physics-derived input priors) are FORWARDED to the wrapped spatial model untouched (via
:func:`dataclasses.replace`); the spatial model conditions on them, while the GRU memory
captures the sequence dynamics.

The wrapper is ``N``-agnostic: the node axis is folded into the GRU's batch, so a MULTI-GRID
batch (whose sample axis is already folded into the node axis, :class:`injection_hse.data.MultiGridCollate`)
gives one independent recurrence per concatenated node-phase row — nothing couples two members —
and the enriched observation is handed to the wrapped spatial model with the index vectors
(``node_batch`` / ``node_ptr`` / ``edge_batch``) untouched.

Causality argument: the ONLY operation that mixes across time is the GRU, and a GRU is causal
(step ``t`` reads inputs ``0..t``). The temporal residual at step ``t`` is a function of the
context at ``t`` (hence of observations ``<= t``); the spatial model is applied per step with no
cross-step coupling. Therefore the output at step ``t`` is independent of observations at steps
``> t`` — verified in the tests by perturbing future steps and checking the output at ``t`` is
unchanged. The GRU is sequential over ``T`` by nature (that is what makes it causal); the node
and batch axes are still vectorized.

``"causal_attention"`` (causal-masked temporal attention) is a planned alternative mechanism;
the seam is in place (a ``NotImplementedError`` until added).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

from injection_hse.config import TemporalConfig
from injection_hse.io import ModelInput

from .base import StateEstimator


class TemporalEstimator(StateEstimator):
    """Wrap a spatial :class:`StateEstimator` with a causal temporal memory over ``T``.

    Parameters
    ----------
    cfg:
        The :class:`~injection_hse.config.TemporalConfig` (selects the mechanism + the wrapped spatial
        model).
    spatial:
        The already-constructed spatial estimator (built from ``cfg.spatial`` by
        :func:`injection_hse.models.build_model`).
    n_features:
        ``Fnd`` — the per-node dynamic feature width (the spatial model's input width, and
        the width of the GRU token and its residual).
    n_out_features:
        ``Fout`` — the per-node OUTPUT width of the wrapped spatial model (``None`` =
        ``n_features``). Recorded for introspection only: the recurrence lives on the input
        width and the spatial model's own head emits ``Fout``, so nothing here changes with it.
    """

    def __init__(
        self,
        cfg: TemporalConfig,
        *,
        spatial: StateEstimator,
        n_features: int,
        n_out_features: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial = spatial
        self.n_features = n_features
        self.n_out_features = (
            n_features if n_out_features is None else int(n_out_features)
        )
        hidden = cfg.hidden_dim

        if cfg.temporal == "gru":
            # per-node GRU over T; input = node_obs (Fnd) ⧺ mask channel (1)
            self.memory = nn.GRU(
                input_size=n_features + 1,
                hidden_size=hidden,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
        elif cfg.temporal == "causal_attention":
            raise NotImplementedError(
                "temporal='causal_attention' is a planned mechanism; use 'gru' for now."
            )
        else:
            raise ValueError(
                f"unknown temporal mechanism {cfg.temporal!r} (use 'gru')."
            )
        # project the temporal context back to the node feature width -> a residual on node_obs
        self.context_proj = nn.Linear(hidden, n_features)

    def _temporal_context(self, node_obs: Tensor, node_mask: Tensor) -> Tensor:
        """Causal per-node temporal residual ``[*B, T, N, Fnd]`` from the GRU memory."""
        *lead, t, n, fnd = node_obs.shape
        mask_ch = node_mask.to(node_obs.dtype).unsqueeze(-1)  # [*B, T, N, 1]
        token = torch.cat([node_obs, mask_ch], dim=-1)  # [*B, T, N, Fnd+1]
        # fold (*B, N) into the batch so each node-phase row runs its own causal recurrence
        # over T; the GRU is batch_first -> [(*B·N), T, Fnd+1].
        seq = token.movedim(-3, -2).reshape(-1, t, fnd + 1)  # [(*B·N), T, Fnd+1]
        ctx, _ = self.memory(
            seq
        )  # [(*B·N), T, hidden] (hidden_t depends on inputs <= t)
        ctx = self.context_proj(ctx)  # [(*B·N), T, Fnd]
        ctx = ctx.reshape(*lead, n, t, fnd).movedim(-2, -3)  # [*B, T, N, Fnd]
        return ctx

    def forward(self, model_input: ModelInput) -> Tensor:
        ctx = self._temporal_context(model_input.node_obs, model_input.node_mask)
        enriched = dataclasses.replace(model_input, node_obs=model_input.node_obs + ctx)
        return self.spatial(enriched)  # [*B, T, N, Fout]

    def predict_injection(self, model_input: ModelInput):
        """Delegate the optional injection head to the wrapped spatial model (if any)."""
        return self.spatial.predict_injection(model_input)


__all__ = ["TemporalEstimator"]
