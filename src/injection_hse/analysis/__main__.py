"""Command-line entry: ``python -m injection_hse.analysis <run_dir>``.

Runs :func:`injection_hse.analysis.analyze_run` self-contained from a finished run
directory (the grid is reloaded from the run's ``grid.json``) and writes the
paper figure set to ``<run_dir>/analysis/``.
"""

from __future__ import annotations

import argparse

from .report import analyze_run


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m injection_hse.analysis",
        description="Interpretability analysis of a finished SE training run "
        "(PDF+SVG paper figures into <run_dir>/analysis/).",
    )
    ap.add_argument("run_dir", help="A finished run directory (manifest.json inside).")
    ap.add_argument(
        "--dataset",
        default="coherent",
        choices=["coherent", "random"],
        help="Evaluation dataset (default: coherent, the sequences a two-task run is "
        "deployed on): the band ablation scores its test split (the validation split "
        "when the run has no test split) and the sensor occlusion reads it. Attribution "
        "and the physical error binning always use the randomized snapshot "
        "dataset (the one spanning operating conditions).",
    )
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    ap.add_argument(
        "--ig-steps",
        type=int,
        default=32,
        help="Integrated-gradients path resolution (default 32).",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of samples for the attribution figures (default 64).",
    )
    ap.add_argument(
        "--prior-samples",
        type=int,
        default=2000,
        help="TRAIN states of the scored dataset the linear injection prior is fitted on "
        "for the band ablation when the checkpoint carries none (default 2000; a "
        "sequence contributes every step; injection-decode runs only).",
    )
    ap.add_argument(
        "--band-samples",
        type=int,
        default=64,
        help="Scenarios of the evaluation split the band ablation scores (default 64; a "
        "sequence is scored with all its steps).",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Quantile bins for the scenario-feature error curves (default 5).",
    )
    ap.add_argument(
        "--geo-positions",
        default=None,
        help="JSON with explicit node coordinates for the node-importance grid map "
        "(keys: node names, zero-based node numbers, or ids -> [x, y]; e.g. "
        "run/configs/cigre_lv_geo.json for the CIGRE LV benchmark). "
        "Default: automatic spring layout.",
    )
    args = ap.parse_args()

    analysis_dir = analyze_run(
        args.run_dir,
        dataset=args.dataset,
        device=args.device,
        ig_steps=args.ig_steps,
        n_attribution_samples=args.samples,
        n_prior_samples=args.prior_samples,
        n_band_samples=args.band_samples,
        bins=args.bins,
        geo_positions=args.geo_positions,
    )
    artifacts = sorted(p.name for p in analysis_dir.iterdir())
    print(f"analysis -> {analysis_dir.resolve()}\nartifacts: {artifacts}")


if __name__ == "__main__":
    main()
