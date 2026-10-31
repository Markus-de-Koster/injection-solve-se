"""Data layer: pgml datasets/grids -> model-ready batches (state + graph + measurement).

Bridges ``pgml`` to the learning loop. The public surface is unchanged from the former single
``injection_hse.data`` module (``from injection_hse.data import X`` keeps working); the implementation is split by
concern into this subpackage:

- :mod:`injection_hse.data.graph` — :func:`build_graph` (the network structure as a PyG ``Data``:
  ``edge_index`` + node/edge parameter features, row-aligned to ``assembly.node_phase_index``).
- :mod:`injection_hse.data.coupling` — the optional PHASE-TO-PHASE admittance edges of that graph (the
  mutual impedance of a line, the winding incidence of a transformer), taken from the same
  primitive stamps the Y-bus assembly scatters.
- :mod:`injection_hse.data.measurement` — :class:`MeasurementModel` (sensor set -> observed node-phase
  rows, per-order capability, metered edges) + :func:`base_voltage_per_row` (the per-unit
  reference).
- :mod:`injection_hse.data.currents` — :class:`BranchCurrentOperator` (the state -> half-edge branch
  currents linear map behind the measured-current model input).
- :mod:`injection_hse.data.physics` — :class:`InjectionSpec` (per-device localization target) built from
  the dataset's realized harmonic injection.
- :mod:`injection_hse.data.sources` — :class:`DataSource` ABC + :class:`ParquetDataset` /
  :class:`TopologyDataSource` / :class:`OnlineScenarioSource` + :func:`build_data_source`.
- :mod:`injection_hse.data.topology` — :class:`TopologySet` / :class:`TopologyBank` (one estimator
  across several switch configurations: per-variant operators + the model's topology
  channels).
- :mod:`injection_hse.data.encodings` — structural node encodings (random-walk / Laplacian /
  effective-impedance)
  and the pairwise effective-resistance matrix: the cross-grid positional signal.
- :mod:`injection_hse.data.multigrid` — generating a MULTI-GRID dataset (many grids, one manifest,
  grid-level split) from ``pgg`` exports and real ``pghub`` grids.
- :mod:`injection_hse.data.validate` — the generation GATE: :func:`validate_dataset` /
  :func:`validate_corpus` / :func:`validate_run` measure what a written dataset actually
  contains and compare it against its own config, DIN EN 50160 and (optionally) measured
  populations.
- :mod:`injection_hse.data.multigraph` — loading one: :class:`MultiGridDataSource` +
  :class:`MultiGridCollate` (the sample axis folds into the node axis) +
  :class:`GraphBatchMasker` (per-graph observation fractions) + the cross-grid comparable
  frames :func:`per_unit_state` (magnitude) and :func:`slack_referenced_state` (angle, over
  the reference rows of :func:`slack_reference_rows` or :func:`reference_bus_rows`).

Each sample's target is the complex state ``V``; the per-grid graph is shared across samples.
Encoding (:mod:`injection_hse.encoding`) and normalization (:mod:`injection_hse.normalization`) are applied in the
training step, not here. The graph feature layout is documented in :mod:`injection_hse.data.graph`.
"""

from __future__ import annotations

from .currents import BranchCurrentOperator, upstream_edge_mask
from .encodings import (
    effective_impedance_to_slack,
    encoding_dim,
    laplacian_positional_encoding,
    pairwise_effective_resistance,
    random_walk_encoding,
    structural_node_encoding,
)
from .graph import build_graph, edge_feature_dim
from .measurement import MeasurementModel, base_voltage_per_row, central_placement
from .multigraph import (
    TASKS,
    GraphBatchMasker,
    GridBundle,
    MultiGridCollate,
    MultiGridDataSource,
    per_unit_state,
    physical_state,
    placement_mask,
    reference_bus_rows,
    segment_random_mask,
    slack_frame_rotation,
    slack_reference_rows,
    slack_referenced_state,
)
from .multigrid import (
    CoherentSequenceConfig,
    Ding0Source,
    GridEntry,
    MultiGridGenerationConfig,
    MultiGridManifest,
    PggExportSource,
    SimbenchSource,
    SyntheticFeederSource,
    generate_multigrid_dataset,
)
from .instrument import InstrumentNoise
from .noise import SensorNoise, grid_current_range
from .physics import (
    InjectionSpec,
    _reconstruct_sampled,  # noqa: F401  (private re-export; kept on the package surface)
)
from .sources import (
    CompositionLabels,
    DataSource,
    OnlineScenarioSource,
    ParquetDataset,
    ProfileFactors,
    StateCollate,
    TopologyDataSource,
    build_data_source,
    topology_set,
)
from .topology import (
    TopologyBank,
    TopologySet,
    enumerate_variants,
    merge_groups,
    switch_branch_ids,
    topology_groups,
)
from .validate import (
    Check,
    DatasetValidationError,
    SeriesStats,
    ValidationReport,
    load_expected_ranges,
    validate_corpus,
    validate_dataset,
    validate_path,
    validate_run,
)

__all__ = [
    "MeasurementModel",
    "base_voltage_per_row",
    "central_placement",
    "BranchCurrentOperator",
    "upstream_edge_mask",
    "SensorNoise",
    "InstrumentNoise",
    "grid_current_range",
    "InjectionSpec",
    "build_graph",
    "edge_feature_dim",
    "DataSource",
    "ProfileFactors",
    "CompositionLabels",
    "ParquetDataset",
    "TopologyDataSource",
    "OnlineScenarioSource",
    "StateCollate",
    "build_data_source",
    "topology_set",
    "TopologySet",
    "TopologyBank",
    "enumerate_variants",
    "switch_branch_ids",
    "topology_groups",
    "merge_groups",
    # structural encodings (cross-grid positional signal)
    "encoding_dim",
    "structural_node_encoding",
    "random_walk_encoding",
    "laplacian_positional_encoding",
    "effective_impedance_to_slack",
    "pairwise_effective_resistance",
    # multi-grid: generation
    "MultiGridGenerationConfig",
    "CoherentSequenceConfig",
    "MultiGridManifest",
    "GridEntry",
    "PggExportSource",
    "SimbenchSource",
    "Ding0Source",
    "SyntheticFeederSource",
    "generate_multigrid_dataset",
    # multi-grid: loading + batching
    "TASKS",
    "MultiGridDataSource",
    "MultiGridCollate",
    "GridBundle",
    "GraphBatchMasker",
    "segment_random_mask",
    "placement_mask",
    "per_unit_state",
    "physical_state",
    "slack_frame_rotation",
    "slack_reference_rows",
    "reference_bus_rows",
    "slack_referenced_state",
    # dataset validation (the generation gate)
    "validate_dataset",
    "validate_corpus",
    "validate_run",
    "validate_path",
    "load_expected_ranges",
    "ValidationReport",
    "SeriesStats",
    "Check",
    "DatasetValidationError",
]
