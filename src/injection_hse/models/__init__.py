"""State-estimation models: a DNN baseline, a GNN, and a Graphormer (one interface).

See :class:`injection_hse.models.base.StateEstimator` for the shared tensor contract. Construct from
an :class:`~injection_hse.config.ModelConfig` via :func:`build_model`.
"""

from __future__ import annotations

from injection_hse.config import DNNConfig, GNNConfig, GraphormerConfig, TemporalConfig

from .base import StateEstimator
from .dnn import DNN
from .gnn import GNN
from .graphormer import Graphormer
from .injection import InjectionHead
from .temporal import TemporalEstimator


def build_model(
    cfg,
    *,
    n_features: int,
    n_nodes: int,
    n_time_features: int = 0,
    n_topology_features: int = 0,
    n_order_mask_channels: int = 0,
    n_prior_features: int = 0,
    n_edge_features: int = 0,
    n_edge_topology_features: int = 0,
    n_static_features: int | None = None,
    n_edge_static_features: int | None = None,
    n_out_features: int | None = None,
) -> StateEstimator:
    """Construct a :class:`StateEstimator` from a model config.

    ``n_features = H * encoding.n_features``; ``n_nodes`` = node-phase rows (the DNN baseline
    needs it for its fixed input/output width). ``n_out_features`` is the per-node OUTPUT
    width ``Fout`` (``None`` = ``n_features``, the voltage decode's in = out contract): the
    injection decode may give its head an encoding of its own
    (:attr:`injection_hse.config.EncodingConfig.injection_kind`), so the model emits ``H × C_out``
    channels while its node observations carry ``H × C_in``. Only each spatial model's final
    projection follows it; every hidden width and the temporal wrapper's recurrence (which
    lives on the INPUT width) are unchanged, and ``None`` is byte-identical.
    ``n_static_features`` is the width of
    ``node_static`` — the :func:`injection_hse.data.build_graph` base block (``None`` = the default 8)
    plus a :class:`~injection_hse.config.StructuralEncodingConfig`'s ``K`` encoding columns, which the
    graph-consuming models fold into their input token. ``n_time_features`` (0 = off) is the
    width of the optional per-step time features appended to every spatial model's node token
    — 0 reproduces the pre-time-feature construction byte-for-byte. ``n_order_mask_channels``
    (0 = off, ``H`` = on) appends the per-order observability channels of a HETEROGENEOUS
    sensor fleet (fundamental-only smart meters next to full-harmonic PQ meters) to every
    node token (and, for the edge-consuming GNN, to the dynamic edge channel).
    ``n_topology_features`` (0 = off, ``S`` = the number of switchable branches) appends the
    per-sample switch states of a SWITCHABLE TOPOLOGY to every node token, the same
    node-agnostic broadcast as the time features; ``n_edge_topology_features`` (0 = off, 1 =
    on) declares the same topology's per-half-edge conduction channel on the edge-consuming
    GNN. ``n_prior_features`` (0 = off) appends the per-row PHYSICS-DERIVED input priors
    (:attr:`injection_hse.config.PhysicsConfig.input_priors` — ``ModelInput.node_prior``, built by
    :meth:`injection_hse.train.SEModule.input_priors`) to every spatial model's node token; an input
    carrying no ``node_prior`` is zero-filled so the width stays fixed.
    ``n_edge_features`` (0 = off) declares the dynamic edge-observation width (the measured
    branch-current features) so the GNN's edge-observation encoder is built EAGERLY — its
    parameters must exist before the optimizer/checkpoint are constructed.
    ``n_edge_static_features`` is the width of ``edge_static`` — :func:`injection_hse.data.build_graph`'s
    series edge block (``None`` = the default 5) plus the flag columns a
    :class:`~injection_hse.config.PhaseCouplingConfig` graph appends, which the edge-consuming GNN folds
    into its edge message (:func:`injection_hse.data.edge_feature_dim`). All of them default
    to 0, reproducing the previous construction byte-for-byte. A
    :class:`~injection_hse.config.TemporalConfig` builds the wrapped spatial model first (which consumes
    the time features), then the causal :class:`TemporalEstimator` around it (which forwards
    the time features through unchanged), so the temporal model is constructed transparently
    from one config.
    """
    if isinstance(cfg, DNNConfig):
        return DNN(
            cfg,
            n_nodes=n_nodes,
            n_features=n_features,
            n_time_features=n_time_features,
            n_topology_features=n_topology_features,
            n_order_mask_channels=n_order_mask_channels,
            n_prior_features=n_prior_features,
            n_out_features=n_out_features,
        )
    if isinstance(cfg, GNNConfig):
        return GNN(
            cfg,
            n_features=n_features,
            n_time_features=n_time_features,
            n_topology_features=n_topology_features,
            n_order_mask_channels=n_order_mask_channels,
            n_prior_features=n_prior_features,
            n_edge_features=n_edge_features,
            n_edge_topology_features=n_edge_topology_features,
            n_static_features=n_static_features,
            n_edge_static_features=n_edge_static_features,
            n_out_features=n_out_features,
        )
    if isinstance(cfg, GraphormerConfig):
        return Graphormer(
            cfg,
            n_features=n_features,
            n_time_features=n_time_features,
            n_topology_features=n_topology_features,
            n_order_mask_channels=n_order_mask_channels,
            n_prior_features=n_prior_features,
            n_static_features=n_static_features,
            n_out_features=n_out_features,
        )
    if isinstance(cfg, TemporalConfig):
        spatial = build_model(
            cfg.spatial,
            n_features=n_features,
            n_nodes=n_nodes,
            n_time_features=n_time_features,
            n_topology_features=n_topology_features,
            n_order_mask_channels=n_order_mask_channels,
            n_prior_features=n_prior_features,
            n_edge_features=n_edge_features,
            n_edge_topology_features=n_edge_topology_features,
            n_static_features=n_static_features,
            n_edge_static_features=n_edge_static_features,
            n_out_features=n_out_features,
        )
        return TemporalEstimator(
            cfg, spatial=spatial, n_features=n_features, n_out_features=n_out_features
        )
    raise TypeError(f"unknown model config {type(cfg).__name__}.")


__all__ = [
    "StateEstimator",
    "DNN",
    "GNN",
    "Graphormer",
    "TemporalEstimator",
    "InjectionHead",
    "build_model",
]
