"""Command-line entry: regenerate the data, train, evaluate and redraw the figures.

usage::

    python -m se_ablation.reproduce data     --experiment evaluation [--site S] [--smoke]
    python -m se_ablation.reproduce train    --experiment gnn_p|gnn_p_r [--smoke] [--device cpu]
    python -m se_ablation.reproduce evaluate --experiment baselines --data DIR NAME=CKPT ...
    python -m se_ablation.reproduce evaluate --experiment perturbed --data DIR NAME=CKPT ...
    python -m se_ablation.reproduce evaluate --experiment meter_sweep --data DIR NAME=CKPT ...
    python -m se_ablation.reproduce evaluate --experiment placement_search --data DIR --frame CKPT
    python -m se_ablation.reproduce attribution --checkpoint CKPT --data DIR
    python -m se_ablation.reproduce figures  --figure parity|meter_sweep|coupling|table ...
    python -m se_ablation.reproduce verify   NAME=CKPT --data DIR --recorded per_order.csv
    python -m se_ablation.reproduce mack     convert|train|score|table ...
    python -m se_ablation.reproduce list

Every command writes below ``--out`` (default ``outputs/<command>/<experiment>``). The
training experiments and the evaluation data are the YAML files of ``experiments/``
(``list`` prints them); ``--smoke`` runs them at a size that finishes on a CPU in minutes.
The estimators of the Mack et al. baseline run in their own environment (``mack/setup.sh``
installs it); ``mack`` converts the data, trains them there and tabulates their scores
(``mack/README.md``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import experiments as X

OUTPUTS = X.ROOT / "outputs"


def _env() -> dict:
    """The environment of a driver run: the repository's packages importable, MLflow's
    file store allowed, the CUDA allocator of the paper runs."""
    env = dict(os.environ)
    src = str(X.ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=_env(), cwd=X.ROOT).returncode


def _extra(args) -> list[str]:
    extra = []
    if getattr(args, "device", None):
        extra.append(f"--device {args.device}")
    if getattr(args, "seed", None) is not None:
        extra.append(f"--seed {args.seed}")
    return extra + list(getattr(args, "driver_args", None) or [])


def cmd_list(args) -> int:
    for kind in ("train", "data"):
        print(f"{kind}:")
        for name in X.names(kind):
            print(f"  {name:22s} {X.load(name).display}")
    print("evaluate: baselines, perturbed, meter_sweep, placement_search")
    return 0


def cmd_data(args) -> int:
    exp = X.load(args.experiment)
    site = [f"--seed {args.site}"] if args.site is not None else []
    out = args.out or OUTPUTS / "data" / (
        f"{exp.name}_site{args.site or 0}" if exp.kind == "data" else exp.name
    )
    if exp.kind == "data":
        argv = exp.argv(smoke=args.smoke, extra=site + _extra(args))
    else:
        argv = exp.data_argv(smoke=args.smoke, extra=_extra(args))
    return _run([sys.executable, str(X.DRIVER["data"]), str(out), *argv])


def cmd_train(args) -> int:
    exp = X.load(args.experiment)
    if exp.kind != "train":
        raise SystemExit(f"{exp.name} is not a training experiment")
    out = args.out or OUTPUTS / "train" / exp.name
    rc = _run(
        [sys.executable, str(exp.driver), str(out), *exp.argv(smoke=args.smoke, extra=_extra(args))]
    )
    if rc or not exp.continuation:
        return rc
    device = [f"--device {args.device}"] if args.device else []
    return _run(
        [
            sys.executable,
            "-m",
            "se_ablation.continuation",
            str(out),
            str(out / "continued"),
            *exp.continuation_argv(smoke=args.smoke, extra=device),
        ]
    )


def cmd_evaluate(args) -> int:
    import torch

    from .evaluate import Evaluation, summarize_search
    from .names import PLACEMENTS

    torch.set_num_threads(int(os.environ.get("THREADS", torch.get_num_threads())))
    out = args.out or OUTPUTS / "evaluate" / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    runs = dict(a.split("=", 1) for a in args.checkpoints)
    placements = dict(PLACEMENTS)
    for spec in args.extra_placement:
        name, buses = spec.split("=", 1)
        placements[name] = [b.strip() for b in buses.split(",")]
    test_dataset = args.test_dataset
    if args.experiment == "perturbed":
        test_dataset = "dataset_coherent_heldout"
    ev = Evaluation(
        data=args.data,
        out=out,
        noise=args.noise,
        test_dataset=test_dataset,
        placements=placements,
    )
    if args.experiment != "placement_search" or not args.summarize:
        if not runs and not args.frame:
            raise SystemExit("give the network checkpoints (NAME=CKPT) or --frame")
        frame = runs.get(args.frame, args.frame) or next(iter(runs.values()))
    if args.experiment == "baselines":
        path = ev.table(runs, frame, args.placements.split(","))
    elif args.experiment == "perturbed":
        path = ev.table(runs, frame, ["deployed"])
    elif args.experiment == "meter_sweep":
        path = ev.sweep(runs, frame, {int(k) for k in args.ks.split(",")})
    elif args.experiment == "placement_search":
        if args.summarize:
            import json

            import pandas as pd

            frames = [
                pd.read_csv(p) for p in sorted(out.glob("placement_search_noise*_shard*.csv"))
            ]
            piv, facts = summarize_search(frames, placements["deployed"])
            piv.to_csv(out / "placement_search.csv", index=False)
            (out / "placement_search.json").write_text(json.dumps(facts, indent=1))
            print(json.dumps(facts, indent=1))
            return 0
        shard, n_shards = (int(x) for x in args.shard.split("/"))
        path = ev.search(frame, shard, n_shards, limit=args.limit)
    else:
        raise SystemExit(f"unknown evaluation {args.experiment!r}")
    print(f"-> {path}")
    return 0


def cmd_attribution(args) -> int:
    from .attribution import attribution

    out = args.out or OUTPUTS / "attribution"
    attribution(
        Path(args.checkpoint),
        Path(args.data),
        out,
        ig_steps=args.ig_steps,
        max_batches=args.max_batches,
        device=args.device or "cpu",
    )
    print(f"-> {out}")
    return 0


def cmd_figures(args) -> int:
    from . import figures as G

    out = args.out or OUTPUTS / "figures"
    if args.figure == "parity":
        if args.inputs:
            path = G.parity_replot(Path(args.inputs[0]), out)
        else:
            orders = [int(o) for o in args.orders.split(",")]
            path = G.parity_figure(Path(args.checkpoint), Path(args.data), out, orders=orders)
    elif args.figure == "meter_sweep":
        path = G.meter_sweep_figure(args.inputs, out)
    elif args.figure == "coupling":
        path = G.coupling_figure(Path(args.inputs[0]), out)
    elif args.figure == "table":
        path = G.results_table(
            args.inputs,
            out,
            perturbed_csvs=args.perturbed,
            mack_csv=args.mack,
            mack_perturbed_csv=args.mack_perturbed,
        )[0]
    else:
        raise SystemExit(f"unknown figure {args.figure!r}")
    print(f"-> {path}")
    return 0


#: the Mack et al. estimators: model class, extra options of mack/train.py (the settings of
#: the paper: the repository's learning rate 5e-5 with its CyclicLR, a step budget and a
#: plateau stop on the validation split)
MACK = {
    "dnn": ("DNN", ["--max-steps", "60000", "--min-steps", "6000", "--patience-steps", "10000"]),
    "dnn_i": (
        "DNN",
        [
            "--max-steps",
            "60000",
            "--min-steps",
            "6000",
            "--patience-steps",
            "10000",
            "--with-currents",
        ],
    ),
    "pann": (
        "PANN",
        [
            "--max-steps",
            "40000",
            "--max-train-hours",
            "9",
            "--min-steps",
            "6000",
            "--patience-steps",
            "8000",
        ],
    ),
}
MACK_SMOKE = [
    "--max-steps",
    "300",
    "--min-steps",
    "300",
    "--patience-steps",
    "300",
    "--checkpoint-every",
    "5",
]


def cmd_mack(args) -> int:
    mack = X.ROOT / "mack"
    out = args.out or OUTPUTS / "mack"
    if args.stage == "convert":
        run = Path(args.train_run)
        return _run(
            [
                sys.executable,
                str(mack / "adapter.py"),
                str(args.data),
                str(
                    out
                    / (
                        "arrays"
                        if args.test_subdir == "dataset_coherent"
                        else f"arrays_{args.test_subdir}"
                    )
                ),
                "--config",
                str(run / "experiment_sequences.yaml"),
                "--placement",
                "Bus R4,Bus C5,Bus I2",
                "--train-draws",
                "2",
                "--test-seed",
                "0",
                "--currents",
                "upstream",
                "--train-root",
                str(run),
                "--train-config-random",
                str(run / "experiment_snapshots.yaml"),
                "--train-config-coherent",
                str(run / "experiment_sequences.yaml"),
                "--test-subdir",
                args.test_subdir,
            ]
        )
    if args.stage == "train":
        model, opts = MACK[args.model]
        return _run(
            [
                "bash",
                str(mack / "envrun.sh"),
                "python",
                "-I",
                str(mack / "train.py"),
                "--src",
                str(mack / "external" / "pqse_concept_pann"),
                "--data",
                str(args.arrays or out / "arrays"),
                "--run-dir",
                str(out / "runs" / args.model),
                "--model",
                model,
                "--seed-index",
                "0",
                "--lr",
                "5e-5",
                "--checkpoint-every",
                "50",
                "--input-variant",
                "classA",
                "--eval-variants",
                "classA,clean",
                *opts,
                *(MACK_SMOKE if args.smoke else []),
                *args.mack_args,
            ]
        )
    if args.stage == "score":
        model, opts = MACK[args.model]
        return _run(
            [
                "bash",
                str(mack / "envrun.sh"),
                "python",
                "-I",
                str(mack / "train.py"),
                "--src",
                str(mack / "external" / "pqse_concept_pann"),
                "--data",
                str(out / "arrays"),
                "--test-data",
                str(args.arrays),
                "--weights-from",
                str(out / "runs" / args.model),
                "--run-dir",
                str(out / f"runs_{Path(args.arrays).name}" / args.model),
                "--model",
                model,
                "--seed-index",
                "0",
                "--lr",
                "5e-5",
                "--input-variant",
                "classA",
                "--eval-variants",
                "classA,clean",
                *[o for o in opts if o == "--with-currents"],
                *args.mack_args,
            ]
        )
    if args.stage == "table":
        labels = {"dnn": "DNN", "dnn_i": "DNN+I", "pann": "PANN"}
        runs_dir = out / (f"runs_{Path(args.arrays).name}" if args.arrays else "runs")
        runs = [f"{labels[name]}={runs_dir / name}" for name in MACK if (runs_dir / name).is_dir()]
        ref = ["--reference", str(args.reference)] if args.reference else []
        return _run(
            [
                sys.executable,
                str(mack / "protocol_table.py"),
                str(out / ("table" if not args.arrays else f"table_{Path(args.arrays).name}")),
                *runs,
                "--data",
                str(args.arrays or out / "arrays"),
                *ref,
                "--site",
                "0",
                "--placement",
                "deployed",
                "--noise",
                "A",
                "--eval-variant",
                "classA",
            ]
        )
    raise SystemExit(f"unknown stage {args.stage!r}")


def cmd_verify(args) -> int:
    from .verify import main

    return main(args.rest)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m se_ablation.reproduce", description=__doc__.splitlines()[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="the experiments").set_defaults(func=cmd_list)

    p = sub.add_parser("data", help="generate a dataset (evaluation data or training data)")
    p.add_argument("--experiment", default="evaluation")
    p.add_argument("--site", type=int, default=None, help="the data seed (evaluation data)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("driver_args", nargs="*", help="extra driver options after --")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="generate the training data and train one estimator")
    p.add_argument("--experiment", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None, help="override the run's --device (e.g. cpu)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("driver_args", nargs="*", help="extra driver options after --")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="baseline table, meter sweep or placement search")
    p.add_argument(
        "--experiment",
        required=True,
        choices=["baselines", "perturbed", "meter_sweep", "placement_search"],
        help="perturbed: the baselines at the deployed placement on the held-out perturbed "
        "grid copies (dataset_coherent_heldout)",
    )
    p.add_argument("--data", type=Path, required=True, help="evaluation dataset directory")
    p.add_argument("checkpoints", nargs="*", help="NAME=CHECKPOINT of the networks")
    p.add_argument(
        "--frame",
        default=None,
        help="the checkpoint (or NAME) whose module scores the baselines "
        "(default: the first network)",
    )
    p.add_argument("--placements", default="deployed,alternative")
    p.add_argument(
        "--extra-placement",
        action="append",
        default=[],
        help="NAME=Bus a,Bus b,... adds a named placement",
    )
    p.add_argument("--noise", default="A", help="A (class-A instrument) or an accuracy class")
    p.add_argument(
        "--test-dataset",
        default="dataset_coherent",
        help="dataset whose test split is scored (fits and calibrations stay on dataset_coherent)",
    )
    p.add_argument("--ks", default="0,1,2,3,4,5,6", help="meter sweep: meter counts (0 = deployed)")
    p.add_argument("--shard", default="0/1", help="placement search: SHARD/NSHARDS")
    p.add_argument("--limit", type=int, default=None, help="placement search: first N of the shard")
    p.add_argument(
        "--summarize",
        action="store_true",
        help="placement search: combine the shards in --out and rank the placements",
    )
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "attribution", help="integrated gradients and order occlusion through network and solve"
    )
    p.add_argument("--checkpoint", required=True, help="the network checkpoint")
    p.add_argument("--data", required=True, help="the evaluation dataset directory")
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument(
        "--max-batches", type=int, default=0, help="only the first batches of 16 sequences"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_attribution)

    p = sub.add_parser("figures", help="the data figures and the baseline table")
    p.add_argument(
        "--figure", required=True, choices=["parity", "meter_sweep", "coupling", "table"]
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help="the evaluation outputs (meter_sweep, coupling, table) or the kept points "
        "parity_points.parquet (parity: redraw without the checkpoint)",
    )
    p.add_argument("--checkpoint", default=None, help="parity: the network checkpoint")
    p.add_argument("--data", default=None, help="parity: the evaluation dataset directory")
    p.add_argument("--orders", default="1,5,9", help="parity: the orders shown")
    p.add_argument(
        "--perturbed", nargs="*", default=[], help="table: the evaluation on the perturbed copies"
    )
    p.add_argument("--mack", type=Path, default=None, help="table: per_order.csv of mack table")
    p.add_argument(
        "--mack-perturbed",
        type=Path,
        default=None,
        help="table: per_order.csv of mack table on the perturbed copies",
    )
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("mack", help="the Mack et al. estimators (their code, their environment)")
    p.add_argument("stage", choices=["convert", "train", "score", "table"])
    p.add_argument("--data", type=Path, help="convert: the evaluation dataset directory")
    p.add_argument("--train-run", help="convert: the GNN-P training run (its training data)")
    p.add_argument(
        "--test-subdir",
        default="dataset_coherent",
        help="convert: dataset_coherent_heldout scores the perturbed copies",
    )
    p.add_argument("--model", choices=sorted(MACK), help="train: the estimator")
    p.add_argument(
        "--arrays",
        type=Path,
        default=None,
        help="train: the converted arrays (default <out>/arrays); score, table: the "
        "converted perturbed test arrays scored with the trained weights",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="table: per_order.csv of the evaluation (ratios to Physics, Linear-P)",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("mack_args", nargs="*", help="train: further mack/train.py options after --")
    p.set_defaults(func=cmd_mack)

    p = sub.add_parser(
        "verify", help="re-score checkpoints against recorded numbers", add_help=False
    )
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_verify)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
