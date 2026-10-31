"""The dataset-validation gate: does a generated dataset contain what its config claims?

Covers the three failure modes that shipped unnoticed before this gate existed — a solved
harmonic order nothing injects at, a coherent window generated in the dead of night, and a
task whose emission population is an order of magnitude off — plus the healthy baseline,
the two injection sources (the persisted realized emission vs the ``I(h) = Y(h)*V(h)``
reconstruction), the reference (measured-population) layer, the strict exit code, and the
multi-grid corpus hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from injection_hse.data.validate import (
    DatasetValidationError,
    ValidationReport,
    _cli,
    validate_corpus,
    validate_dataset,
    validate_run,
)

ORDERS = [1, 3, 5]
N_SAMPLES = 8


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def feeder():
    """A small synthetic 3-phase feeder with PV (schema-only, no reference-tool deps)."""
    from pgml.grids import (
        add_pv_systems,
        synthetic_feeder,
    )

    grid = synthetic_feeder(n_nodes=5, n_feeders=2)
    add_pv_systems(grid, fraction=0.5)
    return grid


def _solve(grid, config, orders):
    from pgml.scenarios import run_scenarios

    return run_scenarios(
        grid,
        config,
        calculation="harmonic",
        harmonic_orders=orders,
        symmetry="auto",
        dtype=torch.complex128,
    )


def _write_random(grid, out_dir, *, orders=ORDERS, injected=None, seed=0):
    """A randomized dataset from the calibrated preset, with the grid beside it.

    ``injected`` (default: every order above 1) selects which orders the CONFIG excites,
    independently of the orders that are SOLVED — the way a solved-but-never-injected
    dead order is produced.
    """
    from pgml.scenarios import write_dataset
    from injection_hse.data.scenarios import se_random_scenario_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid.json").write_text(grid.model_dump_json())
    config = se_random_scenario_config(
        grid,
        orders=[
            1,
            *(injected if injected is not None else [o for o in orders if o > 1]),
        ],
        n_samples=N_SAMPLES,
        seed=seed,
    )
    return write_dataset(_solve(grid, config, orders), out_dir / "dataset_random")


def _write_coherent(
    grid, out_dir, *, start_time, orders=ORDERS, seed=1, name="coherent"
):
    from pgml.scenarios import write_dataset
    from injection_hse.data.scenarios import se_coherent_scenario_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid.json").write_text(grid.model_dump_json())
    config = se_coherent_scenario_config(
        grid,
        orders=[o for o in orders if o > 1],
        n_scenarios=4,
        n_steps=4,
        seed=seed,
        step_size_s=900.0,
        start_time=start_time,
    )
    return write_dataset(_solve(grid, config, orders), out_dir / name)


def _rewrite_scaled(src, dst, grid, factor: float, *, samples=None):
    """Copy a dataset with every harmonic order above the fundamental scaled by ``factor``.

    The realized-injection columns are scaled with it, because a mis-scaled emission
    recipe moves BOTH what the generator recorded and the state it produced. Goes through
    the real writer, so the copy is an ordinary dataset — the way a systematically
    mis-scaled task looks on disk.
    """
    from pgml.assembly import node_phase_index
    from pgml.scenarios import write_dataset
    from pgml.scenarios.run import ScenarioResult
    from pgml.scenarios.sampler import SampledScenarios
    from injection_hse.data.scenarios import read_dataset

    loaded = read_dataset(src)
    state = loaded.v.clone()
    state[..., 1:, :] *= factor
    if samples is None:
        samples = {
            name: (value * factor if name.endswith("_mag") else value)
            for name, value in loaded.samples.items()
        }
    result = ScenarioResult(
        v=state,
        index=node_phase_index(grid),
        sampled=SampledScenarios(
            operating_point={},
            samples=samples,
            n_samples=int(state.shape[0]),
            config=loaded.config,
        ),
        frequencies_hz=loaded.frequencies_hz,
    )
    return write_dataset(result, dst)


@pytest.fixture(scope="module")
def healthy(feeder, tmp_path_factory):
    """A healthy randomized dataset (the preset, every solved order injected)."""
    return _write_random(feeder, tmp_path_factory.mktemp("healthy"))


# =============================================================================
# The healthy baseline
# =============================================================================
def test_healthy_dataset_validates_ok(healthy, feeder):
    report = validate_dataset(healthy)

    assert report.verdict == "OK" and not report.failed, report.table()
    assert report.kind == "dataset" and report.orders == tuple(ORDERS)
    assert report.n_scenarios == N_SAMPLES
    # the fingerprint a reuse gate compares on is carried through from meta.json
    assert (
        report.config_hash
        == json.loads((healthy / "meta.json").read_text())["config_hash"]
    )
    # per-order state + injection statistics, per phase where applicable
    assert report.series("v_pu", 1, "*").median == pytest.approx(1.0, abs=0.1)
    assert report.series("v_pu", 3, "a") is not None
    assert report.series("v_h_ratio_pct", 3, "*").median > 0.0
    assert report.series("i_h_emission_pct", 3, "*").median > 0.0
    for quantity in ("thd_pct", "vuf_pct", "vdev_pct", "p_slack_w"):
        assert report.series(quantity) is not None, quantity


def test_the_report_carries_the_generation_stamps(feeder, tmp_path):
    """The recipe and device-library revisions are read from where the writer put them:
    ``extra_provenance`` for a stamped dataset, the top level for an older one."""
    from pgml.scenarios import write_dataset
    from injection_hse.data.scenarios import GENERATION_PROVENANCE, se_random_scenario_config

    config = se_random_scenario_config(feeder, orders=ORDERS, n_samples=4, seed=0)
    result = _solve(feeder, config, ORDERS)
    (tmp_path / "grid.json").write_text(feeder.model_dump_json())
    unstamped = write_dataset(result, tmp_path / "unstamped")
    report = validate_dataset(unstamped)
    assert report.se_preset_version is None and report.device_library_version is None

    stamped = write_dataset(
        result, tmp_path / "stamped", provenance=GENERATION_PROVENANCE
    )
    report = validate_dataset(stamped)
    assert report.se_preset_version == GENERATION_PROVENANCE["se_preset_version"]
    assert (
        report.device_library_version == GENERATION_PROVENANCE["device_library_version"]
    )

    meta_path = stamped / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["device_library_version"] = meta.pop("extra_provenance")[
        "device_library_version"
    ]
    meta_path.write_text(json.dumps(meta))
    report = validate_dataset(stamped)
    assert (
        report.device_library_version == GENERATION_PROVENANCE["device_library_version"]
    )
    assert report.se_preset_version is None


def test_artifacts_are_written_and_round_trip(healthy):
    validate_dataset(healthy)

    reloaded = ValidationReport.model_validate_json(
        (healthy / "validation.json").read_text()
    )
    assert reloaded.verdict == "OK"
    assert (healthy / "validation_summary.csv").is_file()
    assert (healthy / "validation.svg").is_file()

    import polars as pl

    summary = pl.read_csv(healthy / "validation_summary.csv")
    assert {"quantity", "order", "phase", "median", "p99", "verdict"} <= set(
        summary.columns
    )


# =============================================================================
# The three historical failure modes
# =============================================================================
def test_dead_order_fails_and_names_the_order(feeder, tmp_path):
    """A SOLVED order nothing is injected at: |V_h| is identically zero -> FAIL."""
    path = _write_random(
        feeder, tmp_path, injected=[3]
    )  # order 5 solved, never excited

    report = validate_dataset(path)

    assert report.failed
    dead = [c for c in report.failures if "DEAD ORDER" in c.reason]
    assert dead and all(c.order == 5 for c in dead)
    assert "h5" in dead[0].reason and "injects orders [3]" in dead[0].reason
    # the healthy orders are not implicated
    assert not [c for c in report.failures if c.order == 3]


def test_midnight_sequence_flags_the_dead_activity_band(feeder, tmp_path):
    """A coherent window anchored at midnight sits outside every activity preset."""
    path = _write_coherent(feeder, tmp_path, start_time="2024-06-21T00:00:00")

    report = validate_dataset(path)

    band = [c for c in report.all_checks() if c.quantity == "hour_of_day"]
    assert band and band[0].verdict in ("WARN", "FAIL")
    assert "activity band" in band[0].reason


def test_high_activity_anchor_passes_the_band_check(feeder, tmp_path):
    """The preset's own anchor must not trip the check it exists to catch."""
    from injection_hse.data.scenarios import HIGH_ACTIVITY_START_TIME

    path = _write_coherent(feeder, tmp_path, start_time=HIGH_ACTIVITY_START_TIME)

    report = validate_dataset(path)

    assert not [c for c in report.all_checks() if c.quantity == "hour_of_day"]


def test_run_cross_comparison_flags_a_scaled_sequence_set(healthy, feeder, tmp_path):
    """the sequence pass twenty times low at every order: the per-order median ratio catches it.

    The historical bug was 10-30x low, and it moved BOTH the injected emission and the
    harmonic voltage it produced — so both the persisted and the state-derived quantity
    are expected to flag.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "grid.json").write_text(feeder.model_dump_json())
    _rewrite_scaled(healthy, run_dir / "dataset_random", feeder, 1.0)
    _rewrite_scaled(healthy, run_dir / "dataset_coherent", feeder, 0.05)

    report = validate_run(run_dir)

    cross = [c for c in report.checks if c.layer == "cross"]
    assert cross and all(c.verdict == "FAIL" for c in cross)
    assert {c.quantity for c in cross} == {"v_h_ratio_pct", "i_h_emission_pct"}
    assert {c.order for c in cross} == {3, 5}
    assert "below dataset_random" in cross[0].reason
    assert report.failed and set(report.members) == {
        "dataset_random",
        "dataset_coherent",
    }
    assert (run_dir / "validation.json").is_file()


def test_a_faithful_second_task_is_not_flagged(healthy, feeder, tmp_path):
    """The two tasks legitimately differ; only an order-of-magnitude gap is a verdict."""
    run_dir = tmp_path / "run_ok"
    run_dir.mkdir()
    (run_dir / "grid.json").write_text(feeder.model_dump_json())
    _rewrite_scaled(healthy, run_dir / "dataset_random", feeder, 1.0)
    _rewrite_scaled(healthy, run_dir / "dataset_coherent", feeder, 1.5)

    report = validate_run(run_dir)

    assert not [c for c in report.checks if c.layer == "cross"]


# =============================================================================
# The two injection sources: persisted realized emission vs reconstruction
# =============================================================================
def _strip_realized_columns(src, dst, grid):
    """The same dataset as written before realized injections were persisted."""
    from injection_hse.data.scenarios import read_dataset

    loaded = read_dataset(src)
    kept = {
        name: value
        for name, value in loaded.samples.items()
        if not (name.endswith("_mag") or name.endswith("_phase"))
    }
    return _rewrite_scaled(src, dst, grid, 1.0, samples=kept)


def test_persisted_realized_injections_are_preferred(healthy, monkeypatch):
    """The columns are exact and free; the reconstruction is not even attempted."""
    import injection_hse.data.validate as validate_module

    def _never(*args, **kwargs):
        raise AssertionError("I(h) = Y(h)*V(h) reconstructed despite persisted columns")

    monkeypatch.setattr(validate_module, "_injection_series", _never)

    report = validate_dataset(healthy, write=False)

    assert report.series("i_h_emission_pct", 3, "*").median > 0.0
    assert report.series("i_h_ratio_pct", 3, "*") is None
    assert "persisted sample columns" in report.notes[0]


def test_old_dataset_falls_back_to_the_reconstruction(healthy, feeder, tmp_path):
    """A dataset written before the columns existed keeps its full audit."""
    legacy = _strip_realized_columns(healthy, tmp_path / "legacy" / "dataset", feeder)
    (tmp_path / "legacy" / "grid.json").write_text(feeder.model_dump_json())

    report = validate_dataset(legacy, write=False)

    assert report.series("i_h_emission_pct", 3, "*") is None
    assert report.series("i_h_ratio_pct", 3, "*").median > 0.0
    assert "no realized-injection columns" in report.notes[0]


def test_both_sources_report_two_distinct_quantities(healthy):
    """They measure different things (device emission vs nodal net injection)."""
    report = validate_dataset(healthy, write=False, injection_source="both")

    emission = report.series("i_h_emission_pct", 3, "*")
    nodal = report.series("i_h_ratio_pct", 3, "*")
    assert emission is not None and nodal is not None
    assert emission.median != nodal.median
    assert "both quantities measured" in report.notes[0]


# =============================================================================
# The measured-population layer
# =============================================================================
def _reference_file(path: Path, *, quantity: str, order: int, p50: float) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({quantity: {order: {"p50": float(p50)}}}))
    return path


@pytest.mark.parametrize(
    "source,quantity",
    [("auto", "i_h_emission_pct"), ("reconstructed", "i_h_ratio_pct")],
)
def test_reference_median_warns_at_3x_and_fails_at_10x(
    healthy, tmp_path, source, quantity
):
    realized = validate_dataset(healthy, write=False, injection_source=source).series(
        quantity, 3, "*"
    )

    warn = validate_dataset(
        healthy,
        reference=_reference_file(
            tmp_path / f"warn_{source}.yaml",
            quantity=quantity,
            order=3,
            p50=realized.median / 4.0,
        ),
        write=False,
        injection_source=source,
    )
    fail = validate_dataset(
        healthy,
        reference=_reference_file(
            tmp_path / f"fail_{source}.yaml",
            quantity=quantity,
            order=3,
            p50=realized.median / 20.0,
        ),
        write=False,
        injection_source=source,
    )

    assert warn.verdict == "WARN" and not warn.failed
    assert [c.order for c in warn.warnings if c.layer == "reference"] == [3]
    assert fail.failed
    assert [c.order for c in fail.failures if c.layer == "reference"] == [3]


def test_the_shipped_expectations_template_loads(healthy):
    """The tracked template is a valid reference file the validator can apply."""
    from injection_hse.data import load_expected_ranges

    template = Path(__file__).resolve().parents[2] / "run/configs/expected_ranges.yaml"
    ranges = load_expected_ranges(template)

    assert ranges["i_h_ratio_pct"][3].p50 > 0.0
    assert ranges["thd_pct"][None].hard_max == 8.0
    # and it applies to a preset-generated dataset without failing it
    assert not validate_dataset(healthy, reference=template, write=False).failed


# =============================================================================
# Strictness + the CLI
# =============================================================================
def test_strict_raises_and_the_cli_exits_non_zero(feeder, healthy, tmp_path):
    broken = _write_random(feeder, tmp_path / "broken", injected=[3])

    with pytest.raises(DatasetValidationError):
        validate_dataset(broken, strict=True)
    # the artifacts are written BEFORE the raise, so the failure is inspectable
    assert (broken / "validation.json").is_file()

    assert _cli([str(broken), "--strict"]) == 1
    assert _cli([str(broken)]) == 0
    assert _cli([str(healthy), "--strict"]) == 0


def test_cli_detects_the_layout(healthy, feeder, tmp_path):
    from injection_hse.data.validate import detect_kind

    run_dir = tmp_path / "run_kind"
    run_dir.mkdir()
    (run_dir / "grid.json").write_text(feeder.model_dump_json())
    _rewrite_scaled(healthy, run_dir / "dataset_random", feeder, 1.0)

    assert detect_kind(healthy) == "dataset"
    assert detect_kind(run_dir) == "run"
    with pytest.raises(SystemExit):
        detect_kind(tmp_path / "run_kind" / "nothing_here")


# =============================================================================
# Self-declared layer + the multi-grid corpus hook
# =============================================================================
def test_a_declared_varied_quantity_that_never_varies_fails(feeder, tmp_path):
    """The config's own claim is the first reference: a frozen draw is a FAIL."""
    from pgml.scenarios import write_dataset
    from pgml.assembly import node_phase_index
    from pgml.scenarios.run import ScenarioResult
    from pgml.scenarios.sampler import SampledScenarios
    from injection_hse.data.scenarios import read_dataset

    source = _write_random(feeder, tmp_path / "frozen")
    loaded = read_dataset(source)
    samples = dict(loaded.samples)
    samples["load_scale"] = torch.full_like(samples["load_scale"], 0.5)
    frozen = write_dataset(
        ScenarioResult(
            v=loaded.v,
            index=node_phase_index(feeder),
            sampled=SampledScenarios(
                operating_point={},
                samples=samples,
                n_samples=int(loaded.v.shape[0]),
                config=loaded.config,
            ),
            frequencies_hz=loaded.frequencies_hz,
        ),
        tmp_path / "frozen" / "dataset_frozen",
    )
    (tmp_path / "frozen" / "grid.json").write_text(feeder.model_dump_json())

    report = validate_dataset(frozen)

    declared = [c for c in report.failures if c.quantity == "load_scale"]
    assert declared and "realized p1..p99 span" in declared[0].reason


def test_corpus_generation_validates_every_member(tmp_path):
    """The corpus generator validates what it wrote and records the verdict."""
    from injection_hse.data.multigrid import (
        MultiGridGenerationConfig,
        MultiGridManifest,
        SyntheticFeederSource,
        generate_multigrid_dataset,
    )

    config = MultiGridGenerationConfig(
        output_root=str(tmp_path / "corpus"),
        n_scenarios=4,
        orders=(1, 3),
        val_fraction=0.5,
        sources=[SyntheticFeederSource(n_grids=2, node_range=(4, 6), n_feeders=2)],
    )
    manifest = generate_multigrid_dataset(config)

    root = Path(config.output_root)
    assert manifest.validation["verdict"] in ("OK", "WARN")
    assert manifest.validation["report"] == "validation.json"
    assert not manifest.validation["failed_members"]
    # rewritten manifest carries the summary, the full report sits beside it
    assert MultiGridManifest.read(root).validation == manifest.validation
    assert (root / "validation.json").is_file()
    assert (root / "validation_summary.csv").is_file()

    report = validate_corpus(root, write=False)
    assert set(report.members) == {entry.key for entry in manifest.grids}
    assert all(not member.failed for member in report.members.values())
    # every member's own artifacts were written beside its data
    for entry in manifest.grids:
        assert (root / "grids" / entry.key / "validation.json").is_file()
