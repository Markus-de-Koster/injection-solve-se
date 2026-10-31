"""Shared fixtures for the injection_hse (state-estimation) tests.

Builds the full 3-phase CIGRE LV grid (with synthesized Carson geometry + PV, matching
``run/examples/injection_hse/generate_se_dataset.py``) and generates a tiny parquet dataset ONCE per session
so the data-layer / training tests run end-to-end without re-solving per test.
"""

from __future__ import annotations

import warnings

import pytest
import torch

ORDERS = [1, 3, 5, 7, 9, 11, 13]


@pytest.fixture(scope="session")
def cigre_grid():
    """Full 3-phase CIGRE LV grid with Carson geometry + PV (the SE benchmark grid)."""
    from injection_hse._switches import materialize_estimator_grid
    from pgml.convert.pandapower import PhaseMode
    from pgml.grids import (
        add_pv_systems,
        cigre_lv_full_grid,
    )
    from pgml.geometry.synthesis import synthesize_grid_geometry

    warnings.filterwarnings("ignore")
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    synthesize_grid_geometry(grid)
    add_pv_systems(grid, fraction=0.5)
    return materialize_estimator_grid(grid)


@pytest.fixture(scope="session")
def se_dataset(cigre_grid, tmp_path_factory):
    """A tiny parquet SE dataset (256 Sobol scenarios, complex64) -> directory path."""
    from injection_hse.data.scenarios import se_benchmark_scenario_config
    from pgml.scenarios import write_dataset

    from injection_hse.data.scenarios import run_scenarios

    out = tmp_path_factory.mktemp("se_dataset")
    cfg = se_benchmark_scenario_config(cigre_grid, n_samples=256, seed=0)
    result = run_scenarios(
        cigre_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")


@pytest.fixture(scope="session")
def se_sequence_dataset(cigre_grid, tmp_path_factory):
    """A tiny SEQUENCE SE dataset (``[B, T, H, N]``, T>1) from a coherent-spectrum config."""
    from pgml.scenarios import Selector, write_dataset
    from injection_hse.data.scenarios import CoherentSpectrumConfig
    from injection_hse.data.scenarios import run_scenarios

    out = tmp_path_factory.mktemp("se_seq_dataset")
    cfg = CoherentSpectrumConfig(
        name="coherent",
        selector=Selector(component="load"),
        orders=[3, 5, 7],
        n_steps=4,
        n_scenarios=12,
        n_modes=2,
        seed=0,
    )
    result = run_scenarios(
        cigre_grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    return write_dataset(result, out, layout="wide")
