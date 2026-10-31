"""Re-score trained checkpoints and compare with recorded per-order errors.

Scores one or more network checkpoints (and the baselines in the frame of the first) with
:class:`se_ablation.evaluate.Evaluation` on an evaluation dataset, then joins the result
with a recorded per-order table (columns ``noise, placement, run, estimator, order,
pu_mae, rel`` and optionally ``site``, as written by :func:`se_ablation.figures.baseline_table`
or an earlier evaluation) and reports the largest absolute and relative difference of the
pu-MAE per estimator. A checkpoint is named ``name=path``; ``--recorded-run name=ID`` maps
it to the ``run`` value of its rows in the recorded table (default: the same name), the
baselines are matched on ``run == "baseline"``. ``--placement-map recorded=ours`` maps the
recorded placement labels to the placements of :data:`se_ablation.names.PLACEMENTS`.

Writes ``<out>/verify.csv`` (one row per placement, run, estimator and order with both
values) and prints the summary; exits with status 1 when a relative difference exceeds
``--rtol`` or when no recorded row matched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Evaluation
from .names import PLACEMENTS


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+", help="name=checkpoint path (the first is the frame)")
    ap.add_argument("--data", required=True, type=Path, help="evaluation dataset directory")
    ap.add_argument("--recorded", required=True, type=Path, help="recorded per-order CSV")
    ap.add_argument(
        "--recorded-run",
        action="append",
        default=[],
        help="name=ID: the run label of a checkpoint in the recorded table",
    )
    ap.add_argument(
        "--placement-map",
        action="append",
        default=[],
        help="recorded=ours, e.g. a recorded label of the alternative placement",
    )
    ap.add_argument("--placements", default="deployed,alternative")
    ap.add_argument(
        "--test-dataset",
        default="dataset_coherent",
        help="dataset_coherent_heldout: verify the scores on the perturbed grid copies",
    )
    ap.add_argument("--site", type=int, default=0, help="site column of the recorded table")
    ap.add_argument("--noise", default="A")
    ap.add_argument("--out", type=Path, default=Path("outputs/verify"))
    ap.add_argument("--rtol", type=float, default=1e-6)
    return ap.parse_args(argv)


def compare(
    ours: pd.DataFrame, recorded: pd.DataFrame, run_map: dict, placement_map: dict, site: int | None
) -> pd.DataFrame:
    """Join our per-order rows with the recorded ones (``pu_mae`` and ``rel`` of both)."""
    rec = recorded.copy()
    if "site" in rec.columns and site is not None:
        rec = rec[rec.site == site]
    if "copy" in rec.columns:
        rec = rec[rec["copy"].isna() | (rec["copy"].astype(str) == "all")]
    rec["placement"] = rec.placement.map(lambda p: placement_map.get(p, p))
    back = {v: k for k, v in run_map.items()}
    rec["run"] = rec.run.map(lambda r: back.get(r, r))
    ours = (
        ours[ours["copy"].isna() | (ours["copy"].astype(str) == "all")]
        if "copy" in ours.columns
        else ours
    )
    keys = ["noise", "placement", "run", "estimator", "order"]
    rec["noise"] = rec.noise.astype(str)
    ours = ours.assign(noise=ours.noise.astype(str))
    m = ours[keys + ["pu_mae", "rel"]].merge(
        rec[keys + ["pu_mae", "rel"]], on=keys, suffixes=("", "_recorded")
    )
    m["abs_diff"] = (m.pu_mae - m.pu_mae_recorded).abs()
    m["rel_diff"] = m.abs_diff / m.pu_mae_recorded.abs()
    return m


def main(argv=None) -> int:
    args = parse(argv)
    runs = dict(a.split("=", 1) for a in args.checkpoints)
    run_map = {n: n for n in runs} | dict(a.split("=", 1) for a in args.recorded_run)
    placement_map = dict(a.split("=", 1) for a in args.placement_map)
    args.out.mkdir(parents=True, exist_ok=True)
    ev = Evaluation(
        data=args.data,
        out=args.out,
        noise=args.noise,
        test_dataset=args.test_dataset,
        placements=dict(PLACEMENTS),
    )
    frame = next(iter(runs.values()))
    ev.table(runs, frame, args.placements.split(","))
    ours = pd.DataFrame(ev.records)
    m = compare(ours, pd.read_csv(args.recorded), run_map, placement_map, args.site)
    m.to_csv(args.out / "verify.csv", index=False)
    if not len(m):
        print("no recorded row matched (check --recorded-run, --placement-map, --site)")
        return 1
    summary = m.groupby(["placement", "run", "estimator"]).agg(
        orders=("order", "size"), max_abs_diff=("abs_diff", "max"), max_rel_diff=("rel_diff", "max")
    )
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(summary.to_string(float_format=lambda x: f"{x:.2e}"))
    worst = float(np.nanmax(m.rel_diff))
    print(
        f"\n{len(m)} matched rows; largest relative pu-MAE difference {worst:.2e} "
        f"(tolerance {args.rtol:.0e})"
    )
    return 0 if worst <= args.rtol else 1


if __name__ == "__main__":
    raise SystemExit(main())
