"""DNN baseline: a fixed-topology, fixed-input MLP state estimator.

The simplest reference — flattens the per-node dynamic features over the (fixed) node set and
regresses the full flattened node state. NOT topology-general (the input/output width is tied
to one grid's ``N``); it is the baseline the GNN/Graphormer are measured against.

Tensor contract (see :class:`injection_hse.models.base.StateEstimator`):
``forward(model_input) -> [*B, T, N, Fout]`` with ``Fnd = H * encoding.n_features`` the input
width and ``Fout`` the output width (``n_out_features``; ``Fnd`` unless the injection decode
gives its head an encoding of its own). The DNN
consumes ``node_obs`` + ``node_mask`` (+ the optional per-row ``node_prior`` block and the
optional per-step ``node_time`` features) only
(it ignores ``node_static`` / edges / injection — the interface carries them so the
GNN/Graphormer can use them). The boolean mask is concatenated as an extra per-node channel
so the MLP distinguishes an "observed zero" from a "masked (unknown)" row; when time features
are enabled the per-step global ``node_time`` vector is broadcast to every node and appended
too. The whole ``[N, Fnd+1(+Ftime)]`` block is flattened to one vector per (batch, time)
sample. ``T`` is folded into the batch and estimated per step. Honors the input device/dtype;
fully vectorized over the batch/time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from injection_hse.config import DNNConfig
from injection_hse.io import ModelInput

from .base import (
    StateEstimator,
    broadcast_global_features,
    order_mask_channels,
    prior_channels,
)


class DNN(StateEstimator):
    """Flatten-and-regress MLP over a fixed node set."""

    def __init__(
        self,
        cfg: DNNConfig,
        *,
        n_nodes: int,
        n_features: int,
        n_time_features: int = 0,
        n_topology_features: int = 0,
        n_order_mask_channels: int = 0,
        n_prior_features: int = 0,
        n_out_features: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_nodes = n_nodes
        self.n_features = n_features
        # per-node OUTPUT width (``None`` = the input width): the final projection alone
        # follows it, so a model with ``n_out_features == n_features`` is byte-identical.
        self.n_out_features = (
            n_features if n_out_features is None else int(n_out_features)
        )
        self.n_time_features = n_time_features
        self.n_topology_features = n_topology_features
        self.n_order_mask_channels = n_order_mask_channels
        self.n_prior_features = n_prior_features

        # per node: dynamic features + mask channel + optional per-order observability
        # channels (heterogeneous sensor fleets) + optional physics-derived prior channels
        # + optional per-step time channels
        in_dim = n_nodes * (
            n_features
            + 1
            + n_order_mask_channels
            + n_prior_features
            + n_time_features
            + n_topology_features
        )
        out_dim = n_nodes * self.n_out_features
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, model_input: ModelInput) -> Tensor:
        folded, lead = model_input.fold_time_into_batch()
        features = folded.node_obs  # [BT, N, Fnd]
        mask = folded.node_mask  # [BT, N]
        bt, n, f = features.shape
        if n != self.n_nodes or f != self.n_features:
            raise ValueError(
                f"DNN expects [..., {self.n_nodes}, {self.n_features}], got "
                f"[..., {n}, {f}] (fixed-topology baseline)."
            )
        mask_ch = mask.to(features.dtype).unsqueeze(-1)  # [BT, N, 1]
        parts = [features, mask_ch]
        order_ch = order_mask_channels(
            folded.node_order_mask,
            location_mask=mask,
            n_order_channels=self.n_order_mask_channels,
            dtype=features.dtype,
        )
        if order_ch is not None:
            parts.append(order_ch)  # [BT, N, H]
        prior_ch = prior_channels(
            folded.node_prior,
            batch=bt,
            n_nodes=n,
            n_features=self.n_prior_features,
            device=features.device,
            dtype=features.dtype,
        )
        if prior_ch is not None:
            parts.append(prior_ch)  # [BT, N, Fp]
        time_ch = broadcast_global_features(
            folded.node_time,
            batch=bt,
            n_nodes=n,
            n_features=self.n_time_features,
            device=features.device,
            dtype=features.dtype,
        )
        if time_ch is not None:
            parts.append(time_ch)  # [BT, N, Ftime]
        topology_ch = broadcast_global_features(
            folded.node_topology,
            batch=bt,
            n_nodes=n,
            n_features=self.n_topology_features,
            device=features.device,
            dtype=features.dtype,
        )
        if topology_ch is not None:
            parts.append(topology_ch)  # [BT, N, S]
        x = torch.cat(parts, dim=-1).reshape(bt, -1)  # [BT, N*(F+1[+Ftime])]
        out = self.mlp(x).reshape(bt, n, self.n_out_features)  # [BT, N, Fout]
        return out.reshape(*lead, n, self.n_out_features)  # [*B, T, N, Fout]


__all__ = ["DNN"]
