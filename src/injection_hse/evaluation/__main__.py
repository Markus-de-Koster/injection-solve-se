"""``python -m injection_hse.evaluation <run_dir>`` — evaluate a finished run from its directory.

Loads the run's ``manifest.json`` + serialized ``grid.json`` + final checkpoint and writes the
full evaluation plot set into ``<run_dir>/evaluation/``. The grid is reloaded from ``grid.json``
(the experiment writes one next to the manifest), so no training code or grid rebuild is needed
— evaluation is self-contained from the run directory.
"""

from __future__ import annotations

import argparse

from .report import evaluate_run


def _parse_anchor(text):
    """``'wv'`` or ``'wv/wi'`` -> ``(wv, wi)``; a bare weight means equal trust."""
    if not text:
        return None
    wv, _, wi = str(text).partition("/")
    return (float(wv), float(wi) if wi else float(wv))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m injection_hse.evaluation",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument(
        "run_dir",
        help="A finished run directory (contains manifest.json + grid.json).",
    )
    ap.add_argument(
        "--dataset",
        default="coherent",
        choices=["coherent", "random"],
        help="Which dataset split to evaluate (default: coherent).",
    )
    ap.add_argument("--device", default="cpu", help="Torch device for inference.")
    ap.add_argument(
        "--heatmap-phases",
        default=None,
        help="Comma-separated phases for the per-phase error heatmaps (e.g. 'a,b,c'); "
        "default: every line phase present in the grid.",
    )
    ap.add_argument(
        "--anchor",
        default=None,
        help="Deploy these anchoring weights for THIS evaluation: 'wv' (equal current trust) "
        "or 'wv/wi'. The anchor is a deployment decision, so comparing settings needs no "
        "retrain; pair it with --eval-dirname to keep the outputs apart.",
    )
    ap.add_argument(
        "--no-anchor",
        action="store_true",
        help="Disable the injection decode's measurement anchoring at evaluation (score the "
        "SAME weights with the inference-time anchoring off — the slack-only decode).",
    )
    ap.add_argument(
        "--eval-dirname",
        default="evaluation",
        help="Output subdirectory name (default: evaluation). Use a distinct name with "
        "--no-anchor so the anchored and non-anchored evaluations do not overwrite each other.",
    )
    args = ap.parse_args()

    heatmap_phases = None
    if args.heatmap_phases:
        from pgml.schemas.grid_schema import Phase

        heatmap_phases = [
            Phase(tok.strip().lower())
            for tok in args.heatmap_phases.split(",")
            if tok.strip()
        ]

    eval_dir = evaluate_run(
        args.run_dir,
        dataset=args.dataset,
        device=args.device,
        heatmap_phases=heatmap_phases,
        anchor=not args.no_anchor,
        anchor_weights=_parse_anchor(args.anchor),
        eval_dirname=args.eval_dirname,
    )
    artifacts = sorted(p.name for p in eval_dir.iterdir())
    print(f"evaluation -> {eval_dir.resolve()}\n  artifacts: {artifacts}")


if __name__ == "__main__":
    main()
