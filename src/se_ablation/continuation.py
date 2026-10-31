"""Continue a finished training run in its last pass, with its optimizer state.

usage::

    python -m se_ablation.continuation <run dir> <out dir> [--epochs N] [--anneal-epochs M]
        [--lr LR] [--warmup W] [--device cuda]

The run directory is the output directory of the workflow driver (``training_run.yaml``,
``grid.json``, ``checkpoints/``). From its final checkpoint (``init_from``) and with that
checkpoint's AdamW state (``train.carry_optimizer_state``), two stages copy the run's last
stage (its placement or random sensor sets, its anchor-weight ranges):

1. ``<last>_continue``: a linear learning-rate warm-up over ``--warmup`` epochs to ``--lr``,
   then constant, ended by the run's plateau stop (its patience and minimum improvement) or
   after ``--epochs`` further epochs;
2. ``<last>_reanneal``: a cosine anneal of the learning rate over ``--anneal-epochs`` with
   the optimizer state carried on.

Both keep their best validation weights. Before training, the starting point is validated
on the run's own validation splits (``border_validation.json``: the final checkpoint on the
sequence and on the snapshot data, and the end-of-sequence-pass checkpoint under the last
stage's anchor weights); these passes are part of the procedure, since they draw from the
module's reading-noise streams before the continuation does.

Writes into ``<out dir>``: ``checkpoints/stage_<last>_continue.ckpt`` and
``checkpoints/stage_<last>_reanneal.ckpt`` (the final model), ``training_history.parquet``,
``border_validation.json`` and ``manifest.json`` (the run's manifest with the final
checkpoint replaced, so the run's analysis tools read the continued model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="the training run (the driver's output directory)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--epochs", type=int, default=92, help="epoch cap after the warm-up")
    ap.add_argument("--anneal-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def validate(module, setup, experiment, pass_index, stage, ckpt) -> dict:
    """Validation metrics of ``ckpt`` on pass ``pass_index`` under ``stage``'s placement
    and anchor weights."""
    import lightning as L

    from injection_hse.curriculum import apply_init_checkpoint
    from injection_hse.train import StageManager
    from injection_hse.train.dtypes import _lightning_precision

    apply_init_checkpoint(module, str(ckpt))
    module.cfg = experiment
    manager = StageManager(experiment, **setup.stage_kwargs)
    module.set_masker(manager._masker_for(stage))
    module.set_stage_anchor(
        v_weight=stage.anchor_voltage_weight
        if stage.anchor_voltage_weight is not None
        else experiment.physics.anchor_voltage_weight,
        i_weight=stage.anchor_current_weight
        if stage.anchor_current_weight is not None
        else experiment.physics.anchor_current_weight,
        v_range=stage.anchor_voltage_range,
        i_range=stage.anchor_current_range,
    )
    trainer = L.Trainer(
        accelerator=experiment.train.accelerator,
        devices=1,
        precision=_lightning_precision(experiment),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        # no_grad instead of inference_mode: tensors the module caches during this pass
        # (solver operands) are reused by the training that follows, and inference-mode
        # tensors cannot be saved for backward
        inference_mode=False,
    )
    (metrics,) = trainer.validate(module, datamodule=setup.datamodules[pass_index], verbose=False)
    keep = (
        "val_pu_mae",
        "val_pu_mae_unobs",
        "val_pu_mae_unobs_h1",
        "val_pu_mae_unobs_harm",
        "val_loss",
    )
    return {k: float(v) for k, v in metrics.items() if k in keep}


def continue_run(
    run: Path, out: Path, *, epochs=92, anneal_epochs=15, lr=3e-4, warmup=3, device="cuda"
) -> Path:
    """Continue ``run`` into ``out`` (see the module docstring); returns the final
    checkpoint."""
    from pgml.schemas.grid_schema import Grid

    from injection_hse.workflow import TrainingRunConfig, build_training_setup, run_training

    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run / "checkpoints"
    base = TrainingRunConfig.load(run / "training_run.yaml")
    task_a, task_b = base.passes[0].experiment, base.passes[1].experiment
    last = task_b.curriculum.stages[-1]
    final = ckpt_dir / f"stage_{last.name}.ckpt"
    seq_end = ckpt_dir / f"stage_{task_b.curriculum.stages[0].name}.ckpt"
    cont = last.model_copy(
        update=dict(
            name=f"{last.name}_continue",
            max_epochs=epochs + warmup,
            lr=lr,
            scheduler="none",
            warmup_epochs=warmup,
            target_loss=1e-12,
        )
    )
    reanneal = last.model_copy(
        update=dict(
            name=f"{last.name}_reanneal",
            max_epochs=anneal_epochs,
            lr=lr,
            scheduler="cosine",
            warmup_epochs=0,
            target_loss=1e-12,
        )
    )

    def with_train(exp, stages):
        train = exp.train.model_copy(
            update={
                "carry_optimizer_state": True,
                "checkpoint_dir": str(out / "checkpoints"),
                "mlflow_tracking_uri": f"file:{out / 'mlruns'}",
            }
        )
        return exp.model_copy(
            update={
                "train": train,
                "curriculum": exp.curriculum.model_copy(update={"stages": stages}),
            }
        )

    cfg = base.model_copy(
        update={
            "passes": [
                base.passes[0].model_copy(update={"experiment": with_train(task_a, [])}),
                base.passes[1].model_copy(
                    update={"experiment": with_train(task_b, [cont, reanneal])}
                ),
            ],
            "init_from": str(final),
            "start_stage": cont.name,
        }
    )
    cfg = TrainingRunConfig.model_validate(cfg.model_dump())
    grid = Grid.model_validate_json((run / "grid.json").read_text())
    setup = build_training_setup(cfg, grid=grid)
    module = setup.module

    border = {
        "final_B": validate(module, setup, task_b, 1, last, final),
        "final_A": validate(module, setup, task_a, 0, task_a.curriculum.stages[-1], final),
    }
    if seq_end.exists() and seq_end != final:
        seq_stage = task_b.curriculum.stages[0]
        border["seq_B_seq"] = validate(module, setup, task_b, 1, seq_stage, seq_end)
        border["seq_B_anneal"] = validate(module, setup, task_b, 1, last, seq_end)
    (out / "border_validation.json").write_text(json.dumps(border, indent=2))
    print(json.dumps(border, indent=2), flush=True)

    result = run_training(cfg, setup=setup, device=torch.device(device), out_dir=out)
    result.history.write_parquet(out / "training_history.parquet")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["final_checkpoint"] = str(result.checkpoints[-1])
    manifest["continued_from"] = str(final)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("continuation done:", [str(p) for p in result.checkpoints], flush=True)
    return Path(result.checkpoints[-1])


def main(argv=None) -> int:
    a = parse(argv)
    continue_run(
        a.run,
        a.out,
        epochs=a.epochs,
        anneal_epochs=a.anneal_epochs,
        lr=a.lr,
        warmup=a.warmup,
        device=a.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
