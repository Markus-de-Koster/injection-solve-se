"""Data generated on perturbed copies of the grid (``injection_hse.data.scenarios.perturbation``).

Pins: the copies are deterministic in (config, pool) and the pools disjoint; every kind
scales exactly the parameter it names (the geometry line's positive-sequence reactance to
rounding) and the nominal grid is never touched; a perturbed batch shares the nominal
batch's operating points, records each scenario's copy, and with zero spread reproduces the
nominal solve (one copy: identically; several: to the rounding of a split batched solve).
"""

from __future__ import annotations

import math

import pytest
import torch

ORDERS = [1, 3, 5]
SOLVE = dict(calculation="harmonic", harmonic_orders=ORDERS, symmetry="auto")


def _config(**overrides):
    from injection_hse.data.scenarios import GridPerturbationConfig, PerturbationSpread

    spreads = dict(
        line_r=PerturbationSpread(shared=0.05, element=0.1),
        line_x=PerturbationSpread(element=0.1),
        line_length=PerturbationSpread(shared=0.02, element=0.02),
        transformer_r=PerturbationSpread(element=0.1),
        transformer_x=PerturbationSpread(element=0.1),
        source_impedance=PerturbationSpread(shared=0.3),
    )
    spreads.update(overrides)
    return GridPerturbationConfig(seed=1, **spreads)


def _same(a, b) -> bool:
    """Equality of operating-point entries (floats, tensors or lists of them)."""
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_copies_are_deterministic_and_the_pools_disjoint(cigre_grid):
    from injection_hse.data.scenarios import PERTURBATION_KINDS, draw_grid_perturbations

    cfg = _config(n_copies=3, n_test_copies=2)
    train = draw_grid_perturbations(cigre_grid, cfg, pool="train")
    again = draw_grid_perturbations(cigre_grid, cfg, pool="train")
    test = draw_grid_perturbations(cigre_grid, cfg, pool="test")
    assert [p.index for p in train] == [0, 1, 2] and len(test) == 2
    assert [p.factors for p in train] == [p.factors for p in again]
    assert set(train[0].factors) == set(PERTURBATION_KINDS)
    train_factors = {f for p in train for f in p.factors["line_r"].values()}
    assert not train_factors & {f for p in test for f in p.factors["line_r"].values()}
    # a kind's draws do not depend on another kind's spread
    from injection_hse.data.scenarios import PerturbationSpread

    other = draw_grid_perturbations(
        cigre_grid, cfg.model_copy(update={"line_x": PerturbationSpread()})
    )
    assert other[1].factors["line_r"] == train[1].factors["line_r"]
    # the shared factor multiplies every element of its kind
    shared = train[0].shared["source_impedance"]
    assert all(f == shared for f in train[0].factors["source_impedance"].values())
    assert train[0].magnitude > 0.0


def test_each_kind_scales_its_parameter_and_leaves_the_nominal_grid(cigre_grid):
    from pgml.schemas.grid_schema import Line, Source, Transformer

    from injection_hse.data.scenarios import draw_grid_perturbations
    from injection_hse.data.scenarios.perturbation import _geometry_reference_reactance

    nominal_json = cigre_grid.model_dump_json()
    copy_ = draw_grid_perturbations(cigre_grid, _config(n_copies=1))[0]
    assert cigre_grid.model_dump_json() == nominal_json
    f = copy_.factors
    f0 = float(cigre_grid.base_frequency_hz)
    nominal = {e.id: e for e in [*cigre_grid.branches, *cigre_grid.appliances]}
    for e in [*copy_.grid.branches, *copy_.grid.appliances]:
        ref = nominal[e.id]
        if isinstance(e, Line):
            assert e.length_m == pytest.approx(ref.length_m * f["line_length"][e.id])
            for c, c0 in zip(
                e.conductor_geometry.conductors, ref.conductor_geometry.conductors
            ):
                assert c.r_dc_ohm_per_m == pytest.approx(
                    c0.r_dc_ohm_per_m * f["line_r"][e.id], rel=1e-12
                )
            ratio = _geometry_reference_reactance(
                e, f0
            ) / _geometry_reference_reactance(ref, f0)
            assert ratio == pytest.approx(f["line_x"][e.id], rel=1e-12)
        elif isinstance(e, Transformer):
            assert e.series_resistance_ohm == pytest.approx(
                ref.series_resistance_ohm * f["transformer_r"][e.id]
            )
            assert e.series_inductance_h == pytest.approx(
                ref.series_inductance_h * f["transformer_x"][e.id]
            )
        elif isinstance(e, Source):
            k = f["source_impedance"][e.id]
            assert e.resistance_ohm[0][0] == pytest.approx(ref.resistance_ohm[0][0] * k)
            assert e.inductance_h[1][1] == pytest.approx(ref.inductance_h[1][1] * k)


def test_explicit_lines_scale_their_matrices():
    from pgml.grids import synthetic_feeder
    from pgml.schemas.grid_schema import Line

    from injection_hse.data.scenarios import perturb_grid

    grid = synthetic_feeder(n_nodes=4, n_feeders=1)
    line = next(b for b in grid.branches if isinstance(b, Line))
    assert line.conductor_geometry is None
    out = perturb_grid(grid, {"line_r": {line.id: 1.2}, "line_x": {line.id: 0.9}})
    new = next(b for b in out.branches if b.id == line.id)
    r0 = line.series_resistance_ohm_per_m[0][0]
    l0 = line.series_inductance_h_per_m[0][0]
    assert new.series_resistance_ohm_per_m[0][0] == pytest.approx(1.2 * r0)
    assert new.series_inductance_h_per_m[0][0] == pytest.approx(0.9 * l0)


def test_a_perturbed_batch_records_its_copies_and_reproduces(cigre_grid):
    from injection_hse.data.scenarios import (
        GRID_COPY_RECORD,
        copy_blocks,
        draw_grid_perturbations,
        run_perturbed_scenarios,
        run_scenarios,
        se_benchmark_scenario_config,
    )

    spec = se_benchmark_scenario_config(cigre_grid, n_samples=7, seed=0)
    cfg = _config(n_copies=3)
    first = run_perturbed_scenarios(
        cigre_grid, spec, draw_grid_perturbations(cigre_grid, cfg), **SOLVE
    )
    second = run_perturbed_scenarios(
        cigre_grid, spec, draw_grid_perturbations(cigre_grid, cfg), **SOLVE
    )
    assert torch.equal(first.v, second.v)
    assert copy_blocks(7, 3) == [(0, 3), (3, 5), (5, 7)]
    assert first.sampled.samples[GRID_COPY_RECORD].tolist() == [0, 0, 0, 1, 1, 2, 2]
    nominal = run_scenarios(cigre_grid, spec, **SOLVE)
    # the same operating points ...
    for cid, entry in nominal.sampled.operating_point.items():
        for key, value in entry.items():
            assert _same(first.sampled.operating_point[cid][key], value), (cid, key)
    # ... on a different network: the harmonic voltages move by more than 1 %
    harm = nominal.v[:, 1:]
    rel = (first.v[:, 1:] - harm).abs().amax() / harm.abs().amax()
    assert rel > 1e-2, float(rel)


def test_zero_spread_reproduces_the_nominal_solve(cigre_grid):
    from injection_hse.data.scenarios import (
        GridPerturbationConfig,
        draw_grid_perturbations,
        run_perturbed_scenarios,
        run_scenarios,
        se_benchmark_scenario_config,
    )

    spec = se_benchmark_scenario_config(cigre_grid, n_samples=5, seed=0)
    nominal = run_scenarios(cigre_grid, spec, **SOLVE)
    one = run_perturbed_scenarios(
        cigre_grid,
        spec,
        draw_grid_perturbations(cigre_grid, GridPerturbationConfig(n_copies=1)),
        **SOLVE,
    )
    assert torch.equal(one.v, nominal.v)
    two = run_perturbed_scenarios(
        cigre_grid,
        spec,
        draw_grid_perturbations(cigre_grid, GridPerturbationConfig(n_copies=2)),
        **SOLVE,
    )
    rel = (two.v - nominal.v).abs().amax() / nominal.v.abs().amax()
    assert rel < 1e-13, float(rel)


def test_a_dataset_round_trips_the_copy_record_and_the_factors(cigre_grid, tmp_path):
    import json

    from pgml.scenarios import write_dataset

    from injection_hse.data.scenarios import (
        GRID_COPY_RECORD,
        draw_grid_perturbations,
        perturbation_provenance,
        read_dataset,
        run_perturbed_scenarios,
        se_benchmark_scenario_config,
    )

    cfg = _config(n_copies=2)
    copies = draw_grid_perturbations(cigre_grid, cfg, pool="test")
    result = run_perturbed_scenarios(
        cigre_grid,
        se_benchmark_scenario_config(cigre_grid, n_samples=4, seed=0),
        copies,
        **SOLVE,
    )
    record = perturbation_provenance(cfg, copies, 4)
    path = write_dataset(
        result, tmp_path / "d", provenance={"grid_perturbation": record}
    )
    loaded = read_dataset(path)
    assert loaded.samples[GRID_COPY_RECORD].tolist() == [0, 0, 1, 1]
    meta = json.loads((path / "meta.json").read_text())["extra_provenance"]
    stored = meta["grid_perturbation"]
    assert stored["pool"] == "test" and stored["copies"][1]["scenarios"] == [2, 4]
    line_id = next(iter(copies[1].factors["line_r"]))
    assert stored["copies"][1]["factors"]["line_r"][str(line_id)] == pytest.approx(
        copies[1].factors["line_r"][line_id]
    )
    assert stored["copies"][0]["magnitude_rms_log"] == pytest.approx(
        copies[0].magnitude
    )


def test_invalid_configs_are_rejected(cigre_grid):
    from pydantic import ValidationError

    from injection_hse.data.scenarios import (
        GridPerturbationConfig,
        PerturbationSpread,
        copy_blocks,
        draw_grid_perturbations,
    )

    with pytest.raises(ValidationError, match="uniform"):
        GridPerturbationConfig(
            distribution="uniform", line_r=PerturbationSpread(element=1.0)
        )
    with pytest.raises(ValueError, match="pool"):
        draw_grid_perturbations(cigre_grid, GridPerturbationConfig(), pool="val")
    with pytest.raises(ValueError, match="at least one"):
        copy_blocks(2, 3)
    assert math.isclose(
        draw_grid_perturbations(cigre_grid, GridPerturbationConfig())[0].magnitude, 0.0
    )


def test_the_shipped_template_validates():
    from pathlib import Path

    import yaml

    from injection_hse.data.scenarios import GridPerturbationConfig

    path = (
        Path(__file__).resolve().parents[3] / "run/configs/pgl_grid_perturbation.yaml"
    )
    block = yaml.safe_load(path.read_text())["grid_perturbation"]
    cfg = GridPerturbationConfig.model_validate(block)
    assert cfg.pool_size("train") == 8 and cfg.pool_size("test") == 4
