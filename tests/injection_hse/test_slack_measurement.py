"""Slack-as-measurement: forced observation + the injection-decode requirement.

The metered slack is an ordinary measurement whose rows are ALWAYS observed —
part of the placement, exempt from random masking — and the physics-informed
injection decode refuses to run without it (its fundamental solve needs the
measured slack voltage as the network boundary).
"""

from __future__ import annotations

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    ExperimentConfig,
    MaskStageConfig,
    MeasurementConfig,
    PhysicsConfig,
)
from injection_hse.masking import RandomMasker
from injection_hse.normalization import HarmonicNormalizer
from tests.injection_hse.tiny_grids import single_phase_chain

ORDERS = [1, 3]


def _config(*, slack_known: bool, decode: str = "voltage") -> ExperimentConfig:
    return ExperimentConfig(
        name="slack-measurement",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="unused"),
        model=DNNConfig(hidden_dims=[8]),
        measurement=MeasurementConfig(voltage_nodes=[], slack_known=slack_known),
        physics=PhysicsConfig(decode=decode),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="full", kind="full", target_loss=1.0, max_epochs=1)
            ]
        ),
    )


def _module(cfg):
    from injection_hse.data import build_graph
    from injection_hse.train import SEModule

    grid = single_phase_chain()
    graph = build_graph(grid, orders=ORDERS)
    n_nodes = graph.num_nodes
    feats = torch.rand(32, n_nodes, len(ORDERS), 3)
    normalizer = HarmonicNormalizer().fit(feats)
    return SEModule(cfg, n_nodes=n_nodes, normalizer=normalizer, graph=graph), grid


def test_random_mask_keeps_slack_observed():
    module, grid = _module(_config(slack_known=True))
    module.set_masker(RandomMasker(1.0 / 3.0))

    from pgml.assembly import node_phase_index
    from pgml.topology import slack_node_ids

    idx = node_phase_index(grid)
    slack_rows = [r for s in slack_node_ids(grid) for r in idx.rows(s)]
    mask, _ = module._draw_mask((64,), torch.device("cpu"))
    assert bool(mask[:, slack_rows].all()), "slack rows must survive random masking"
    # the non-slack rows still vary (the masker's random stream is untouched)
    other = [r for r in range(module.n_nodes) if r not in slack_rows]
    assert not bool(mask[:, other].all())


def test_unmetered_slack_is_randomly_masked():
    module, grid = _module(_config(slack_known=False))
    module.set_masker(RandomMasker(1.0 / 3.0))
    mask, _ = module._draw_mask((64,), torch.device("cpu"))
    from pgml.assembly import node_phase_index
    from pgml.topology import slack_node_ids

    idx = node_phase_index(grid)
    slack_rows = [r for s in slack_node_ids(grid) for r in idx.rows(s)]
    assert not bool(mask[:, slack_rows].all()), "no forcing without a slack meter"


def test_injection_decode_requires_metered_slack():
    from injection_hse.data import build_graph
    from injection_hse.physics import NetworkSolver
    from injection_hse.train import SEModule

    grid = single_phase_chain()
    graph = build_graph(grid, orders=ORDERS)
    n_nodes = graph.num_nodes
    feats = torch.rand(32, n_nodes, len(ORDERS), 3)
    normalizer = HarmonicNormalizer().fit(feats)
    solver = NetworkSolver.from_grid(grid, ORDERS)

    with pytest.raises(ValueError, match="metered slack"):
        SEModule(
            _config(slack_known=False, decode="injection"),
            n_nodes=n_nodes,
            normalizer=normalizer,
            graph=graph,
            solver=solver,
            inj_normalizer=HarmonicNormalizer().fit(feats),
        )

    # positive control: with the slack metered the module constructs
    SEModule(
        _config(slack_known=True, decode="injection"),
        n_nodes=n_nodes,
        normalizer=normalizer,
        graph=graph,
        solver=solver,
        inj_normalizer=HarmonicNormalizer().fit(feats),
    )
