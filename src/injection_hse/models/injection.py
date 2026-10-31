"""Per-device injection / localization head (the "where did the disturbance originate" output).

A SHARED auxiliary head (model-agnostic — DNN, GNN, Graphormer all gain localization through
it) that predicts the clean, non-superposed per-device injected harmonic-magnitude spectrum
from a model's node-level representation. The observed node voltage is a SUPERPOSITION of every
device's injection propagated through ``Y(h)^-1``; this head de-mixes it back to the source.

How it works (vectorized, no python loop over batch/time/devices)
-----------------------------------------------------------------
Given the model's predicted node features ``[*B, T, N, Fnd]`` (the node-level representation —
the same tensor the state head produces, so the head is genuinely shared) and the
device→node-phase-row incidence from :class:`injection_hse.data.InjectionSpec`:

1. POOL: gather each device's terminal node-phase rows ``device_rows [D, P_max]`` and take a
   masked mean over the valid phase slots (``device_rows_mask [D, P_max]``) → a per-device
   feature ``[*B, T, D, Fnd]``.
2. MAP: a shared MLP applied per device → ``[*B, T, D, O*C]`` reshaped to ``[*B, T, D, O, C]``
   (``C=1`` now; phase is a documented later channel). A ``softplus`` output keeps the
   predicted magnitudes non-negative (the EN 50160-relative injection magnitudes are ``>= 0``).

The head is OPTIONAL and OFF by default — the node-V state estimation is byte-identical when it
is not constructed. Honors the input device/dtype.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class InjectionHead(nn.Module):
    """Shared per-device injection-magnitude head: node features -> ``[*B, T, D, O, C]``.

    Parameters
    ----------
    node_feature_dim:
        ``Fnd`` — width of the per-node representation the head pools (the model's predicted
        node-state feature width ``H * encoding.n_features``).
    n_orders:
        ``O`` — number of injected harmonic orders (the :class:`injection_hse.data.InjectionSpec` order
        set).
    hidden_dim:
        Width of the shared per-device MLP.
    n_channels:
        ``C`` — output channels per (device, order); ``1`` now (magnitude). Phase is a later
        channel.
    dropout:
        Dropout in the MLP (``0`` disables).
    """

    def __init__(
        self,
        *,
        node_feature_dim: int,
        n_orders: int,
        hidden_dim: int = 128,
        n_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_orders = n_orders
        self.n_channels = n_channels
        layers: list[nn.Module] = [nn.Linear(node_feature_dim, hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, n_orders * n_channels)]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        node_features: Tensor,
        device_rows: Tensor,
        device_rows_mask: Tensor,
    ) -> Tensor:
        """Pool node features at each device's rows -> per-device injection ``[*B, T, D, O, C]``.

        Parameters
        ----------
        node_features:
            The model's node-level representation ``[*B, T, N, Fnd]``.
        device_rows:
            ``[D, P_max]`` int64 — the node-phase rows of each device's terminal phases (padded).
        device_rows_mask:
            ``[D, P_max]`` bool — valid (non-pad) phase slots.
        """
        *lead, n, fnd = node_features.shape
        device = node_features.device
        rows = device_rows.to(device)  # [D, P_max]
        rmask = device_rows_mask.to(
            device=device, dtype=node_features.dtype
        )  # [D, P_max]
        d, p_max = rows.shape

        # gather the per-device terminal-row features: [*lead, D, P_max, Fnd]
        flat = node_features.reshape(-1, n, fnd)  # [M, N, Fnd]
        # the leading count is stated explicitly so a DEVICE-FREE input (``D = 0``, a
        # multi-grid batch drawn only from members without an injection) reshapes to an empty
        # result instead of an ambiguous ``-1``.
        gathered = flat[:, rows.reshape(-1), :].reshape(
            int(flat.shape[0]), d, p_max, fnd
        )  # [M, D, P_max, Fnd]
        # masked mean over the valid phase slots -> [M, D, Fnd]
        w = rmask.view(1, d, p_max, 1)
        pooled = (gathered * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        out = self.mlp(pooled)  # [M, D, O*C]
        out = torch.nn.functional.softplus(out)  # non-negative magnitudes
        out = out.reshape(*lead, d, self.n_orders, self.n_channels)  # [*B, T, D, O, C]
        return out


__all__ = ["InjectionHead"]
