"""The decode's harmonic device shunt (``PhysicsConfig.load_shunt``).

The injection decode solves a whole batch with ONE ``Y(h)`` per order, so it builds the
device shunt from the nameplate. These tests pin that a dataset generated on the nameplate
basis with the same model is decoded with the generator's own ``Y(h)`` (the injection
targets ``Y·V`` reproduce the generator's harmonic right-hand side to rounding), that the
default (``None``) assembles exactly as before the field existed, and that data on the
operating-point basis is NOT reproduced (the test is sensitive to the mismatch).
"""

from __future__ import annotations

import pytest
import torch

ORDERS = [1, 3, 5, 7]
HARMONICS = [3, 5, 7]


def _generate(grid, model, basis, n=4):
    from injection_hse.data.scenarios import run_scenarios, se_benchmark_scenario_config

    cfg = se_benchmark_scenario_config(grid, n_samples=n, seed=0)
    return run_scenarios(
        grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        load_shunt=model,
        load_shunt_basis=basis,
    )


def _generator_system(grid, result, model, basis):
    """``(Y, I)`` the generating solve used for the harmonic orders of ``result``."""
    from pgml.solver import assemble_harmonic_system

    sampled = result.sampled
    y, i, _ = assemble_harmonic_system(
        grid,
        HARMONICS,
        result.v[:, 0],
        operating_point=sampled.operating_point,
        harmonic_injection=sampled.harmonic_injection,
        node_sources=sampled.node_sources or None,
        load_shunt=model,
        load_shunt_basis=basis,
        symmetry="auto",
    )
    return y, i


@pytest.mark.parametrize("model", ["opendss", "none", "motor"])
def test_nameplate_data_is_decoded_with_the_generators_admittance(cigre_grid, model):
    from injection_hse.physics import NetworkSolver

    result = _generate(cigre_grid, model, "nameplate")
    y_gen, i_gen = _generator_system(cigre_grid, result, model, "nameplate")
    solver = NetworkSolver.from_grid(
        cigre_grid, ORDERS, dtype=torch.complex128, load_shunt=model
    )
    assert solver.load_shunt == model
    assert y_gen.shape == solver.y_harmonic.shape  # scenario-independent: [Hh, N, N]
    scale = solver.y_harmonic.abs().amax()
    assert (y_gen - solver.y_harmonic).abs().amax() <= 1e-12 * scale
    # the decode's injection targets reproduce the generator's harmonic right-hand side
    targets = solver.injection_targets(result.v)[:, 1:]
    rel = (targets - i_gen).abs().amax() / i_gen.abs().amax()
    assert rel < 1e-9, float(rel)


def test_operating_point_data_differs_from_every_decode_admittance(cigre_grid):
    from injection_hse.physics import NetworkSolver

    result = _generate(cigre_grid, "opendss", "operating_point")
    y_gen, _ = _generator_system(cigre_grid, result, "opendss", "operating_point")
    solver = NetworkSolver.from_grid(
        cigre_grid, ORDERS, dtype=torch.complex128, load_shunt="opendss"
    )
    assert y_gen.ndim == 4  # [B, Hh, N, N]: one admittance per scenario
    rel = (y_gen - solver.y_harmonic).abs().amax() / solver.y_harmonic.abs().amax()
    assert rel > 1e-7, float(rel)


def test_default_decode_assembles_as_before(cigre_grid):
    """``load_shunt=None`` passes no shunt keyword: the historical ``Y(h)`` exactly."""
    from pgml.solver import assemble_harmonic_ybus

    from injection_hse.physics import NetworkSolver

    solver = NetworkSolver.from_grid(cigre_grid, ORDERS, dtype=torch.complex128)
    assert solver.load_shunt is None
    expected = assemble_harmonic_ybus(cigre_grid, HARMONICS, dtype=torch.complex128)[0]
    assert torch.equal(solver.y_harmonic, expected)


def test_an_unknown_shunt_model_is_rejected(cigre_grid):
    from pydantic import ValidationError

    from injection_hse.config import PhysicsConfig
    from injection_hse.physics import NetworkSolver

    with pytest.raises(ValueError, match="load_shunt"):
        NetworkSolver.from_grid(cigre_grid, ORDERS, load_shunt="nameplate")
    with pytest.raises(ValidationError):
        PhysicsConfig(load_shunt="zip")


def test_the_shunt_model_cannot_change_between_passes():
    """The solver's ``Y(h)`` is built once, so a pass swapping the model is refused."""
    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        MaskStageConfig,
        PhysicsConfig,
    )
    from injection_hse.train import assert_cfg_swappable

    def cfg(load_shunt):
        return ExperimentConfig(
            name="swap",
            orders=ORDERS,
            data=DataConfig(dataset_path="x"),
            model=DNNConfig(hidden_dims=[8]),
            physics=PhysicsConfig(decode="injection", load_shunt=load_shunt),
            curriculum=CurriculumConfig(
                stages=[
                    MaskStageConfig(
                        name="full", kind="full", target_loss=1e-3, max_epochs=1
                    )
                ]
            ),
        )

    assert_cfg_swappable(cfg("opendss"), cfg("opendss"))
    with pytest.raises(ValueError, match="physics.load_shunt"):
        assert_cfg_swappable(cfg(None), cfg("opendss"))
