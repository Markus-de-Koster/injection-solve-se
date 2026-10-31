"""The scores of trained Mack et al. estimators in the tables of the state-estimation evaluation.

The evaluation of injection_hse's state-estimation networks reports, per order, the pu-MAE of the
complex voltage over the test states and the rows the placement does not read
(``pu_mae``), and the relative error ``rel`` = ``pu_mae`` over the mean true magnitude of
those rows in per unit at that order. :mod:`train` writes, on arrays of
:mod:`adapter`, the summed per-state error of the plain phasor error (metric ``full``)
over exactly those rows; this module turns it into:

- ``<out>/per_order.csv``: one row per estimator and order in the evaluation's per-order
  schema (``noise, placement, run, estimator, order, pu_mae, rel, site``);
- ``<out>/summary.csv``: one row per estimator in the schema of its baseline table, with the
  ratios to the physics solve and to the linear map taken from the evaluation's own per-order
  table (``--reference``; rows ``run == "baseline"``, estimators ``physics solve`` and
  ``linear map, best decode per order``): h1 ratio, the mean over the harmonic orders of the
  per-order ratios, and the number of harmonic orders better than each reference.

Without ``--reference`` only the absolute errors are written (the ratios can be formed once
the references of the same test data exist). On arrays whose test split has several sites
(the perturbed grid copies of :mod:`adapter`), ``<out>/per_site.csv`` also gives the
per-order errors of every site, with its RMS log-factor from ``meta.json``.

Usage::

    python protocol_table.py <out_dir> --data <adapter out> --reference <per_order.csv>
        --site 0 --placement deployed --noise A --eval-variant classA NAME=RUN_DIR [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PHYSICS = "physics solve"
LINEAR = "linear map, best decode per order"


def per_order(run_dir: Path, true_mean_pu: np.ndarray, orders, variant: str) -> pd.DataFrame:
    """``order, pu_mae, rel`` of one run on the evaluation variant (metric ``full``)."""
    s = pd.read_csv(run_dir / "scores_sites.csv")
    s = s[(s.metric == "full") & (s.eval_variant == variant)]
    g = s.groupby("order")[["sum", "n"]].sum()
    mae = (g["sum"] / g["n"]).reindex(orders).to_numpy()
    return pd.DataFrame(dict(order=orders, pu_mae=mae, rel=mae / true_mean_pu))


def per_site(run_dir: Path, true_mean_site: np.ndarray, sites: list, orders, variant: str) -> pd.DataFrame:
    """``site, order, pu_mae, rel`` of one run per test site (metric ``full``)."""
    s = pd.read_csv(run_dir / "scores_sites.csv")
    s = s[(s.metric == "full") & (s.eval_variant == variant)]
    rows = []
    for k, name in enumerate(sites):
        g = s[s.site == name].groupby("order")[["sum", "n"]].sum()
        mae = (g["sum"] / g["n"]).reindex(orders).to_numpy()
        rows += [dict(site=name, order=o, pu_mae=m, rel=m / t)
                 for o, m, t in zip(orders, mae, true_mean_site[k])]
    return pd.DataFrame(rows)


def summary_row(label: str, s: pd.Series, r: pd.Series, phys: pd.Series, lin: pd.Series,
                **key) -> dict:
    """The baseline-table entry of one estimator (``s``, ``r``: pu-MAE and rel by order)."""
    h = s.index[s.index > 1]
    return dict(**key, estimator=label, trained_on_site=key.get("site", ""),
                h1_pu_mae=s.loc[1], harm_pu_mae=s.loc[h].mean(),
                h1_rel=r.loc[1], harm_rel=r.loc[h].mean(),
                h1_vs_physics=s.loc[1] / phys.loc[1],
                harm_vs_physics=(s.loc[h] / phys.loc[h]).mean(),
                h1_vs_linear_map=s.loc[1] / lin.loc[1],
                harm_vs_linear_map=(s.loc[h] / lin.loc[h]).mean(),
                orders_better_than_physics=int((s.loc[h] < phys.loc[h]).sum()),
                orders_better_than_linear_map=int((s.loc[h] < lin.loc[h]).sum()))


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = np.load(Path(args.data) / "scoring.npz")
    orders = [int(o) for o in sc["orders"]]
    true_mean = sc["true_mean_pu"]
    phys = lin = None
    if args.reference:
        ref = pd.read_csv(args.reference)
        ref = ref[(ref.placement == args.placement) & (ref.noise.astype(str) == args.noise)
                  & (ref.run == "baseline")]
        if "site" in ref:
            ref = ref[ref.site == args.site]

        def series(est):
            d = ref[ref.estimator == est].set_index("order")["pu_mae"].sort_index()
            if not len(d):
                raise ValueError(f"the reference has no {est!r} rows at this placement and site.")
            return d

        phys, lin = series(PHYSICS), series(LINEAR)
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    sites = meta["splits"]["test"]
    levels = meta.get("grid_copy_rms_log", {})
    rows, table, by_site = [], [], []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        po = per_order(Path(path), true_mean, orders, args.eval_variant)
        label = f"{name} [Mack]"
        for rec in po.itertuples():
            rows.append(dict(noise=args.noise, placement=args.placement, run=name, estimator=label,
                             order=rec.order, pu_mae=rec.pu_mae, rel=rec.rel, site=args.site))
        s_, r_ = po.set_index("order")["pu_mae"], po.set_index("order")["rel"]
        if phys is not None:
            table.append(summary_row(label, s_, r_, phys, lin, site=args.site, placement=args.placement))
        else:
            h = s_.index[s_.index > 1]
            table.append(dict(site=args.site, placement=args.placement, estimator=label,
                              h1_pu_mae=s_.loc[1], harm_pu_mae=s_.loc[h].mean(),
                              h1_rel=r_.loc[1], harm_rel=r_.loc[h].mean()))
        if len(sites) > 1 and "true_mean_pu_site" in sc:
            ps = per_site(Path(path), sc["true_mean_pu_site"], sites, orders, args.eval_variant)
            ps.insert(0, "estimator", label)
            ps.insert(2, "rms_log_factor", ps.site.map(levels))
            by_site.append(ps)
    if by_site:
        pd.concat(by_site).to_csv(out / "per_site.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "per_order.csv", index=False)
    summ = pd.DataFrame(table)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summ.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    ap.add_argument("--data", required=True)
    ap.add_argument("--reference", default="", help="the evaluation's per-order table (optional)")
    ap.add_argument("--site", type=int, default=0)
    ap.add_argument("--placement", default="deployed")
    ap.add_argument("--noise", default="A")
    ap.add_argument("--eval-variant", default="classA")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
