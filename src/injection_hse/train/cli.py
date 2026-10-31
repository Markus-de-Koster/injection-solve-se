"""``python -m injection_hse.train`` — run a recorded training config."""

from __future__ import annotations

import torch


def main(argv=None) -> None:
    """``python -m injection_hse.train --config <run.yaml>`` — execute a recorded training run.

    The re-entry point for a config a run directory already holds: a
    :class:`injection_hse.workflow.TrainingRunConfig` (``training_run.yaml`` — the whole multi-pass
    curriculum) or a bare :class:`~injection_hse.config.ExperimentConfig` (one pass, wrapped). The file
    is the ONLY source of the run's parameters; the two overrides here relocate it
    (``--out``) and pick the device (``--device``). Prints the resolved identity — passes,
    orders, datasets, and the provenance of the code about to train — before the first fit.
    """
    import argparse
    import json
    import sys

    import yaml

    from ..workflow import TrainingRunConfig, _example, run_training

    ap = argparse.ArgumentParser(
        prog="python -m injection_hse.train",
        description="Run a recorded injection_hse training config (the run directory's "
        "training_run.yaml, or a single experiment config).",
    )
    ap.add_argument("--config", help="Training run config (YAML or JSON).")
    ap.add_argument(
        "--json-schema", action="store_true", help="Print the config JSON Schema."
    )
    ap.add_argument(
        "--example", action="store_true", help="Print an example run config as YAML."
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu', 'cuda' or 'cuda:<i>'. Sets each pass's accelerator "
        "and the device the memory preflight probes; default = the config's accelerator "
        "(no preflight).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Run directory override: every pass writes its checkpoints and its tracking "
        "store there, and the resolved config is dumped beside them.",
    )
    args = ap.parse_args(argv)

    if args.json_schema:
        json.dump(TrainingRunConfig.model_json_schema(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if args.example:
        sys.stdout.write(
            yaml.safe_dump(_example().model_dump(mode="json"), sort_keys=False)
        )
        return
    if not args.config:
        ap.error("--config is required (or use --json-schema / --example).")

    cfg = TrainingRunConfig.load(args.config)
    if args.out:
        cfg = cfg.with_output_dir(args.out)
    device = None
    if args.device:
        device = torch.device(args.device)
        accelerator = "cpu" if device.type == "cpu" else "gpu"
        cfg = cfg.model_copy(
            update={
                "passes": [
                    p.model_copy(
                        update={
                            "experiment": p.experiment.model_copy(
                                update={
                                    "train": p.experiment.train.model_copy(
                                        update={"accelerator": accelerator}
                                    )
                                }
                            )
                        }
                    )
                    for p in cfg.passes
                ]
            }
        )

    from pgml.provenance import code_provenance

    provenance = code_provenance()
    print(cfg.identity(), flush=True)
    print(
        f"  code: pgml {provenance['pgml_version']} @ {provenance['git_sha'][:12]}"
        + (" (dirty tree)" if provenance.get("git_dirty") else "")
        + f" | torch {provenance['torch_version']}",
        flush=True,
    )
    result = run_training(cfg, device=device, out_dir=args.out)
    print(
        "\nTRAINING DONE.\n  checkpoints: "
        + ", ".join(str(p) for p in result.checkpoints),
        flush=True,
    )
