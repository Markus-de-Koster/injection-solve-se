"""Generate the two state-estimation datasets of the full workflow, without training.

Writes exactly what ``train_se_full_workflow.py`` writes before its first training step —
``grid.json``, ``dataset_random/`` (Task A, Sobol snapshots ``[B, H, N]``) and
``dataset_coherent/`` (Task B, sequences ``[B, T, H, N]``) — plus a small
``manifest.json`` naming them, so a recipe can be inspected (``data_card.py``) and gated
(``headroom_probe.py``) before any GPU time is spent on it. The grid, the scenario recipes
and the dataset writers are the workflow's own functions, imported rather than copied, so
the data matches a training run with the same flags and seed.

Only the generation flags are exposed; every other workflow knob keeps its workflow
default. Example (the current recipe, small CPU sizes)::

    pixi run -e cpu python run/examples/injection_hse/generate_se_task_datasets.py out_dir \\
        --scenario-config run/configs/pgl_se_scenario.yaml \\
        --orders 1,3,5,7,9,11,13,15,17,19 --n-random 2000 --n-scenarios 200 --n-steps 8

TIME AXIS. By default every sequence covers the same ``--profile-start`` window.
``--start-time-range 2024-01-01,2025-01-01`` gives each sequence its own start in that range
(``--start-time-sampling uniform|stratified``, optionally restricted by ``--start-hours`` /
``--start-weekdays``), so the sequences span times of day, weekdays and seasons and the
dataset records a ``[B, T]`` ``time_unix_s``; ``--activity-calendar`` adds the weekly and
seasonal modulation of the composed device activity. ``--skip-random`` writes the sequences
alone.

NETWORK-MODEL ERROR. ``--grid-perturbation <yaml>`` (the workflow's template
``run/configs/pgl_grid_perturbation.yaml``) writes, from the SAME scenario draw (one seed, so
identical operating points and emission), up to three versions of each dataset: the nominal
one (``dataset_random``), one solved on the ``"train"`` pool of perturbed grid copies
(``dataset_random_ensemble``, ``--perturb-copies`` copies) and one on the ``"test"`` pool
(``dataset_random_heldout``, ``--perturb-test-copies`` copies no ensemble copy shares); the
same for ``dataset_coherent``. ``--perturbation-sets`` selects which. Every perturbed sample
records its copy (``grid_copy``) and ``meta.json`` each copy's factors, exactly as the
workflow writes them; ``grid.json`` stays nominal. Because the three share the scenario
draw and the split is by scenario index, a split seed holds out the same scenarios in all
three, so a model fitted on one dataset's training split can be scored on another's
held-out split without operating-point overlap (``headroom_probe.py --test-dataset``).

STORED GRID. ``--grid <grid.json>`` generates on a stored grid instead of the CIGRE LV
build, exactly as the training driver's ``--grid`` does (``export_se_feeder.py`` writes one
from a published feeder): its PV generators, consumer types and installed meters are used
as stored, so ``--pv-fraction`` is rejected. The grid-model options still apply on top of
it (``--device-capacitance``, ``--source-impedance``, ``--load-shunt``); the written
``grid.json`` carries them, so the probe and the data card read the generator's grid. Pass
the STORED grid, not this ``grid.json``, to the training driver with the same flags. The
manifest's ``sensor_names`` are the grid's metered nodes (slack excluded), else the CIGRE
set.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path


try:  # run as a script (run/examples/injection_hse on sys.path) or imported as a package
    from run.examples.injection_hse import train_se_full_workflow as workflow
    from run.examples.injection_hse.headroom_probe import default_sensors
except ModuleNotFoundError:  # pragma: no cover - script run
    import train_se_full_workflow as workflow
    from headroom_probe import default_sensors


def generation_args(argv=None) -> argparse.Namespace:
    """Parse the generation flags into the namespace the workflow's generators read."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", help="Run-like directory to write (created if missing).")
    ap.add_argument("--n-random", type=int, default=2000, help="Task A snapshots.")
    ap.add_argument("--n-scenarios", type=int, default=200, help="Task B sequences.")
    ap.add_argument("--n-steps", type=int, default=8, help="Steps per Task B sequence.")
    ap.add_argument(
        "--orders",
        default="1,3,5,7,9,11,13,15,17,19",
        help="Solved harmonic orders (the workflow's --orders syntax).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scenario-config",
        default=None,
        help="Scenario deviation YAML (the workflow's --scenario-config); default the preset.",
    )
    ap.add_argument(
        "--grid",
        default=None,
        help="Stored grid JSON to generate on (the training driver's --grid) instead of "
        "the CIGRE LV build.",
    )
    ap.add_argument(
        "--pv-fraction",
        type=float,
        default=None,
        help="PV generators added to the CIGRE LV build (default 0.5); a --grid keeps "
        "its own.",
    )
    ap.add_argument(
        "--coherent-mode", default="composed", choices=["composed", "fingerprint"]
    )
    ap.add_argument("--coherent-step-s", type=float, default=900.0)
    ap.add_argument("--n-modes", type=int, default=2)
    ap.add_argument("--dwell", type=float, default=0.9)
    ap.add_argument(
        "--emission-persistence",
        default="scenario",
        choices=["scenario", "device", "class"],
    )
    ap.add_argument("--emission-fraction", default=None)
    ap.add_argument("--background-distortion", default=None)
    ap.add_argument("--background-drift", type=float, default=0.0)
    ap.add_argument("--background-drift-phase-deg", type=float, default=0.0)
    ap.add_argument("--background-sc-mva", type=float, default=20.0)
    ap.add_argument("--activity-scale", type=float, default=1.0)
    ap.add_argument("--behavioral-coupling", type=float, default=0.3)
    ap.add_argument("--cloud-coupling", type=float, default=0.5)
    ap.add_argument("--profile-start", default=workflow._DEFAULT_START_TIME)
    workflow.add_time_axis_arguments(ap)
    ap.add_argument(
        "--skip-coherent", action="store_true", help="Generate Task A snapshots only."
    )
    ap.add_argument(
        "--skip-random", action="store_true", help="Generate Task B sequences only."
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Scenarios per solve slice (a sequence scenario carries all its steps); bounds "
        "the per-scenario Y(h) workspace. 2000 snapshots + 200 x 8 sequences peak near "
        "2.6 GB at 32 and near 12 GB unchunked.",
    )
    ap.add_argument(
        "--load-shunt",
        default=None,
        choices=["none", "opendss", "motor"],
        help="Harmonic device shunt of the generation (as the workflow's --load-shunt).",
    )
    ap.add_argument(
        "--load-shunt-basis",
        default=None,
        choices=["nameplate", "operating_point"],
        help="Basis of --load-shunt (default nameplate when --load-shunt is given).",
    )
    ap.add_argument(
        "--device-capacitance",
        default=None,
        help="Device capacitance YAML (as the workflow's --device-capacitance).",
    )
    ap.add_argument(
        "--source-impedance",
        default="pgml",
        choices=["pgml", "cigre-tb575"],
        help="Source impedance preset (as the workflow's --source-impedance).",
    )
    ap.add_argument(
        "--grid-perturbation",
        default=None,
        help="Perturbation YAML (the workflow's --grid-perturbation): also write the "
        "perturbed-ensemble and held-out-perturbed versions of each dataset.",
    )
    ap.add_argument(
        "--perturb-copies",
        type=int,
        default=None,
        help="Copies of the 'train' pool (overrides the YAML's n_copies).",
    )
    ap.add_argument(
        "--perturb-test-copies",
        type=int,
        default=None,
        help="Copies of the 'test' pool (overrides the YAML's n_test_copies).",
    )
    ap.add_argument(
        "--perturbation-sets",
        default="nominal,ensemble,heldout",
        help="With --grid-perturbation: which versions to write, a subset of "
        "nominal,ensemble,heldout.",
    )
    ap.add_argument("--dtype", default="complex128", choices=list(workflow.DTYPES))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.grid is not None and args.pv_fraction is not None:
        ap.error(
            "--pv-fraction applies to the CIGRE LV build; a --grid keeps its own PV"
        )
    if args.grid is None and args.pv_fraction is None:
        args.pv_fraction = 0.5
    if args.load_shunt_basis is not None and args.load_shunt is None:
        ap.error("--load-shunt-basis needs --load-shunt")
    if args.skip_random and args.skip_coherent:
        ap.error("--skip-random and --skip-coherent leave nothing to generate")
    sets = [s.strip() for s in args.perturbation_sets.split(",") if s.strip()]
    unknown = set(sets) - set(PERTURBATION_SETS)
    if unknown or not sets:
        ap.error(f"--perturbation-sets: choose from {list(PERTURBATION_SETS)}")
    if args.grid_perturbation is None and (
        args.perturb_copies or args.perturb_test_copies
    ):
        ap.error("--perturb-copies / --perturb-test-copies need --grid-perturbation")
    args.perturbation_sets = sets if args.grid_perturbation else ["nominal"]
    # the workflow's regime flags, from the selected sets (read by perturbation_plan)
    args.perturb_train = "perturbed" if "ensemble" in sets else "nominal"
    args.perturb_test = "heldout" if "heldout" in sets else "none"
    return args


def stored_or_benchmark_grid(args):
    """``(grid, label)``: the ``--grid`` file as stored, else the CIGRE LV build + PV.

    The stored grid is read by the workflow's own loader; its label names the PV
    generators and meters it carries. Grid-model options are applied by the caller.
    """
    from pgml.schemas.grid_schema import Generator

    if args.grid is None:
        grid, n_pv = workflow.build_grid(args.pv_fraction)
        return grid, f"CIGRE LV (3-ph), {n_pv} PV"
    path = Path(args.grid)
    grid = workflow.load_grid(path)
    n_gen = sum(isinstance(a, Generator) for a in grid.appliances)
    name = grid.metadata.name or path.stem
    label = (
        f"{name} ({len(grid.nodes)} nodes), {n_gen} generators, "
        f"{len(grid.measurement_devices)} meters"
    )
    return grid, label


#: The dataset versions a perturbation run writes, with their directory suffix and the
#: perturbation-plan entry that solves them (``None`` = the nominal grid).
PERTURBATION_SETS = {
    "nominal": ("", None),
    "ensemble": ("_ensemble", "train"),
    "heldout": ("_heldout", "test"),
}


def main(argv=None) -> None:
    args = generation_args(argv)
    warnings.filterwarnings("ignore")  # the synthesized-GMR notice is expected here
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = workflow.resolve_device(args.device)
    dtype = workflow.DTYPES[args.dtype]
    orders = workflow.parse_orders(args.orders)
    injected = [h for h in orders if h > 1]
    args.scenario_random, args.scenario_coherent = workflow.load_scenario_config(
        args.scenario_config
    )

    grid, label = stored_or_benchmark_grid(args)
    sensor_names = default_sensors(grid)
    # the workflow's grid-model options, applied before the grid is persisted so the
    # estimator built from grid.json sees them
    args.grid_model = workflow.apply_grid_model(grid, args)
    (out_dir / "grid.json").write_text(grid.model_dump_json())
    recipe = workflow.scenario_recipe_identity(args)
    print(f"{label} | orders {orders} | recipe: {recipe}", flush=True)

    plan = workflow.perturbation_plan(grid, args)
    if plan is not None:
        for pool in ("train", "test"):
            copies = plan[pool]
            if copies:
                mags = ", ".join(f"{p.magnitude:.3g}" for p in copies)
                print(f"  {pool} pool: {len(copies)} copies, RMS log-factor {mags}")

    timings = {}
    datasets = {}
    generators = []
    if not args.skip_random:
        generators.append(
            ("dataset_random", workflow.generate_random_dataset, args.n_random)
        )
    if not args.skip_coherent:
        generators.append(
            ("dataset_coherent", workflow.generate_coherent_dataset, args.n_scenarios)
        )
    for stem, generate, size in generators:
        for name in args.perturbation_sets:
            suffix, pool = PERTURBATION_SETS[name]
            copies = None if pool is None else plan[pool]
            t0 = time.perf_counter()
            path = generate(
                grid,
                orders,
                injected,
                args,
                out_dir,
                device,
                dtype,
                dataset_path=out_dir / f"{stem}{suffix}",
                perturbations=copies,
                perturbation_config=None if copies is None else plan["config"],
            )
            key = f"{stem}{suffix}"
            timings[f"{key}_s"] = time.perf_counter() - t0
            datasets[key] = str(Path(path).resolve())
            detail = "" if copies is None else f" on {len(copies)} {pool}-pool copies"
            print(
                f"  {key}: {size} scenarios{detail} in {timings[f'{key}_s']:.1f} s "
                f"-> {path}",
                flush=True,
            )

    manifest = {
        "orders": orders,
        "sensor_names": sensor_names,
        "datasets": datasets,
        "scenario_recipe": recipe,
        "generation": {
            k: getattr(args, k)
            for k in (
                "n_random",
                "n_scenarios",
                "n_steps",
                "seed",
                "pv_fraction",
                "coherent_mode",
                "coherent_step_s",
                "emission_persistence",
                "emission_fraction",
                "background_distortion",
                "dtype",
            )
        },
        "timings_s": timings,
    }
    if args.grid is not None:
        manifest["grid_file"] = workflow.file_fingerprint(args.grid)
    generation = workflow.data_generation_record(args)
    if generation:
        manifest["data_generation"] = generation
    if plan is not None:
        manifest["grid_perturbation"] = {
            "config": plan["config"].model_dump(mode="json"),
            "sets": args.perturbation_sets,
            "copies": {
                pool: [
                    {"index": p.index, "magnitude_rms_log": p.magnitude}
                    for p in plan[pool]
                ]
                for pool in ("train", "test")
                if plan[pool] is not None
            },
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
