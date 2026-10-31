"""One-call evaluation of a finished run: manifest + checkpoint -> the full plot set.

:func:`evaluate_run` reads the ``manifest.json`` a training run writes (sensor placement,
orders, dataset paths, final checkpoint, the A→B switch), rebuilds the trained estimator, runs
it over the chosen dataset, and writes every evaluation figure into ``out_dir/evaluation/`` —
so a finished run is evaluated with a single call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

from pgml.schemas.grid_schema import Phase

from .error_heatmap import plot_error_heatmap
from .history import plot_loss_components, plot_loss_curves, read_history
from .inference import load_se_module, predict_dataset, prediction_metrics
from .loss_heatmap import loss_term_maps, plot_loss_heatmap
from .insight_plots import (
    plot_error_by_order,
    plot_error_cdf,
    plot_error_vs_sensor_distance,
    plot_parity_per_harmonic,
    worst_scenario,
)
from .state_plots import plot_state_vs_prediction

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

_MANIFEST = "manifest.json"
_HISTORY = "training_history.parquet"

# phase code (node_phase_index convention) -> Phase, for the per-phase heatmaps
_CODE_PHASE = {0: Phase.A, 1: Phase.B, 2: Phase.C, 3: Phase.N}


def _resolve_heatmap_phases(prediction, requested) -> list:
    """The phases to draw a per-phase error heatmap for (default: a/b/c present in the grid).

    ``requested`` may be a list of :class:`~pgml.schemas.grid_schema.Phase` (or their string
    values, e.g. ``"a"``); ``None`` selects every line phase (A/B/C) that actually has rows in
    the prediction's node-phase layout — so a single-phase grid yields one heatmap, a balanced
    three-phase grid yields three.
    """
    import numpy as np

    present = {int(c) for c in np.unique(prediction.phase_codes)}
    if requested is None:
        return [_CODE_PHASE[c] for c in (0, 1, 2) if c in present]
    out = []
    for ph in requested:
        ph = ph if isinstance(ph, Phase) else Phase(ph)
        out.append(ph)
    return out


def _load_grid(out_dir: Path, grid):
    """Return ``grid`` if given, else the serialized grid the run wrote (``grid.json``)."""
    if grid is not None:
        return grid
    gp = out_dir / "grid.json"
    if gp.exists():
        from pgml.schemas.grid_schema import Grid

        return Grid.model_validate_json(gp.read_text())
    raise FileNotFoundError(
        f"evaluate_run needs the grid the run trained on: pass grid=... or place a serialized "
        f"grid at {gp} (the experiment writes one automatically next to the manifest)."
    )


def _write_metrics_summary(
    prediction,
    eval_dir: Path,
    *,
    stem: str = "metrics",
    anchor_weights: "tuple[float, float] | None" = None,
) -> None:
    """Write ``<stem>.json`` + ``<stem>_per_order.csv`` (pu-MAE + relative + THD errors).

    The unobserved-row numbers are the headline estimation quality: an aggregate block (pu
    MAE/RMSE + voltage-THD error) and a per-order block pairing each order's pu-MAE with its
    per-harmonic RELATIVE error (residual over the order's own signal scale). A silent order's
    relative error is ``nan`` (guarded), which serialises as ``null`` in JSON / an empty CSV
    cell. ``stem`` selects the file name (``"metrics"`` for the validation split, ``"metrics_test"``
    for the held-out test split — the two summaries are written side by side, clearly labeled).

    ``anchor_weights`` ``(wv, wi)`` records the measurement anchoring the scored decode was
    DEPLOYED with (``None`` for a voltage decode, which has no solve to anchor) — the same
    weights read back at inference, and the difference between two otherwise identical runs.
    """
    errors = prediction_metrics(prediction)
    summary = {
        "anchor": (
            None
            if anchor_weights is None
            else {
                "voltage_weight": anchor_weights[0],
                "current_weight": anchor_weights[1],
            }
        ),
        "overall": {
            "pu_mae_unobs": errors["mae_unobs"],
            "pu_rmse_unobs": errors["rmse_unobs"],
            "mae_unobs_phys": errors["mae_unobs_phys"],
            "thd_mae_unobs": errors["thd_mae_unobs"],
            "rel_unobs": errors["rel_unobs"],
        },
        "per_order": {
            str(o): {
                "pu_mae_unobs": errors[o]["mae_unobs"],
                "rel_unobs": errors[o]["rel_unobs"],
                "mae_unobs_phys": errors[o]["mae_unobs_phys"],
            }
            for o in prediction.orders
        },
    }
    (eval_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    lines = ["order,pu_mae_unobs,rel_unobs,mae_unobs_phys"]
    for o in prediction.orders:
        e = errors[o]
        rel = e["rel_unobs"]
        rel_s = "" if rel != rel else f"{rel:.6g}"  # nan -> empty cell
        lines.append(f"{o},{e['mae_unobs']:.6g},{rel_s},{e['mae_unobs_phys']:.6g}")
    (eval_dir / f"{stem}_per_order.csv").write_text("\n".join(lines) + "\n")


def evaluate_run(
    out_dir,
    *,
    grid: "Optional[Grid]" = None,
    dataset: str = "coherent",
    device: Union[str, str] = "cpu",
    heatmap_phases: Optional[Sequence[Phase]] = None,
    anchor: bool = True,
    anchor_weights: "tuple[float, float] | None" = None,
    eval_dirname: str = "evaluation",
    loss_heatmap_batches: Optional[int] = 32,
) -> Path:
    """Evaluate a finished run and write the full plot set to ``out_dir/evaluation/``.

    Reads ``out_dir/manifest.json`` (the sensor placement, orders, dataset paths, final
    checkpoint, and the A→B switch), reloads the trained :class:`~injection_hse.train.SEModule`, runs it
    over the ``dataset`` (``"coherent"`` or ``"random"``) validation split under the placement,
    and writes:

    - ``loss_curves.svg`` — the two-task training history (if ``training_history.parquet``
      is present), and ``loss_components.svg`` — the same history decomposed into the
      supervised loss terms and the per-order errors behind the aggregates;
    - ``error_heatmap_{mean,max}_{phase}.svg/.html`` — per-order x node-phase pu-``|error|``,
      ONE figure per phase (``heatmap_phases``; default every line phase a/b/c present in the
      grid), the fundamental scaled separately from the harmonics;
    - ``loss_heatmap_a.svg/.html`` — the supervised LOSS (voltage / injection / total) per
      injection node on phase A, fundamental beside harmonics: where the two terms disagree
      is where consumption and generation cancel and the voltages cannot resolve the split
      (``loss_heatmap_batches`` caps the averaging pass; ``None`` = the whole split);
    - ``error_by_order.svg`` / ``error_cdf.svg`` (+ ``error_cdf_linear.svg``, the same CDF on
      a linear x-axis) / ``error_vs_distance.svg`` / ``parity.svg`` (+ ``parity.png``, a
      300-DPI raster drawn with a larger point budget) — the insight plots;
    - ``metrics.json`` + ``metrics_per_order.csv`` — the numeric summary on the VALIDATION
      split: the deployed measurement-anchoring weights, the aggregate unobserved pu-MAE/RMSE
      and voltage-THD error, plus the per-order pu-MAE AND per-harmonic RELATIVE error
      (residual normalized by each order's own signal scale, so small orders are not hidden by
      the fundamental-referenced pu base);
    - ``metrics_test.json`` + ``metrics_test_per_order.csv`` — the SAME numeric summary on the
      held-out TEST split (written only when the run set ``DataConfig.test_fraction > 0``; the
      test split is never touched during training / early-stop, so these are a final one-shot
      generalization estimate, clearly separated from the validation numbers above);
    - ``state_h{order}.svg/.html`` + ``state_3d.html`` — ground-truth vs estimate for the
      WORST sample (the hardest case), sensors marked.

    ``grid`` may be omitted when the run wrote a ``grid.json`` (the standard experiment does):
    it is then reloaded automatically, so evaluation is self-contained from the run directory.
    ``heatmap_phases`` selects which phases get an error heatmap (default: a/b/c as present).

    ``anchor`` (default True) keeps the injection-decode's measurement anchoring active at
    evaluation (the deployed decode). ``anchor=False`` disables it — the decode is slack-only —
    so a model can be scored WITH vs WITHOUT inference-time anchoring on the same weights;
    ``anchor_weights`` ``(wv, wi)`` instead deploys a CHOSEN trust — the anchor is a
    deployment decision, so two settings can be compared on one trained model without a
    retrain. ``eval_dirname`` then names a separate output subdirectory (e.g.
    ``"evaluation_no_anchor"``)
    so the two do not overwrite each other. Returns the ``out_dir/<eval_dirname>`` directory.
    """
    from injection_hse.config import MeasurementConfig

    out_dir = Path(out_dir)
    grid = _load_grid(out_dir, grid)
    manifest = json.loads((out_dir / _MANIFEST).read_text())
    eval_dir = out_dir / eval_dirname
    eval_dir.mkdir(parents=True, exist_ok=True)

    key = "dataset_coherent" if dataset == "coherent" else "dataset_random"
    dataset_path = manifest["datasets"][key]
    checkpoint_path = manifest["final_checkpoint"]
    sensor_names = manifest["sensor_names"]
    orders = manifest["orders"]

    measurement = MeasurementConfig(voltage_nodes=sensor_names, slack_known=True)
    module = load_se_module(
        checkpoint_path, grid=grid, dataset_path=dataset_path, device=device
    )
    if anchor_weights is not None and module.solver is not None:
        # Score the SAME weights at a CHOSEN deployed trust rather than the one the run
        # trained at: the anchor is a deployment decision, so comparing settings should not
        # need a retrain (or a mutated checkpoint, which would make the run dir disagree
        # with itself about what it deploys).
        wv, wi = (float(w) for w in anchor_weights)
        module.solver.set_anchor_weights(v_weight=wv, i_weight=wi)
    elif not anchor and module.solver is not None:
        # score the SAME weights with the inference-time anchoring switched off (slack-only
        # decode) — isolates how much the anchoring alone contributes at inference.
        module.solver.set_anchor_weights(v_weight=0.0, i_weight=0.0)
    # what this evaluation actually DEPLOYED, recorded beside the numbers it produced: the
    # anchored decode carries a large share of the accuracy, so a metrics file without its
    # anchor weights cannot be compared to another one.
    deployed_anchor = (
        None
        if module.solver is None
        else (
            float(module.solver.anchor_v_weight),
            float(module.solver.anchor_i_weight),
        )
    )

    # Build the data source ONCE (the module's own split settings — val/test fraction, seed —
    # travel with its config), so the validation and the held-out TEST split share one
    # partition and the test split is never re-derived under different fractions.
    from injection_hse.data import build_data_source

    data_cfg = module.cfg.data.model_copy(
        update={"source": "parquet", "dataset_path": str(dataset_path)}
    )
    source = build_data_source(data_cfg, grid=grid)

    prediction = predict_dataset(
        module,
        source_or_path=source,
        grid=grid,
        measurement=measurement,
        split="val",
        device=device,
    )

    # -- numeric metric summary (pu-MAE + per-harmonic relative error + THD error) --
    # VALIDATION split (metrics.json), and additionally the held-out TEST split
    # (metrics_test.json) when the run carved one (DataConfig.test_fraction > 0). The test
    # split is NEVER touched during training / early-stop / checkpoint selection — this is the
    # only place it is read, for a final one-shot generalization estimate labeled test vs val.
    _write_metrics_summary(prediction, eval_dir, anchor_weights=deployed_anchor)
    if len(source.test_dataset()) > 0:
        test_prediction = predict_dataset(
            module,
            source_or_path=source,
            grid=grid,
            measurement=measurement,
            split="test",
            device=device,
        )
        _write_metrics_summary(
            test_prediction,
            eval_dir,
            stem="metrics_test",
            anchor_weights=deployed_anchor,
        )

    # -- training history (if present) — SVG only (HTML stage annotations overlap) --
    hist_path = out_dir / _HISTORY
    if hist_path.exists():
        history = read_history(hist_path)
        plot_loss_curves(history, manifest, out_svg=str(eval_dir / "loss_curves.svg"))
        # a compact 1-column paper variant: train/val loss only, enlarged fonts, no title
        plot_loss_curves(
            history,
            manifest,
            curves="loss",
            paper=True,
            out_svg=str(eval_dir / "loss_curves_paper.svg"),
            out_pdf=str(eval_dir / "loss_curves_paper.pdf"),
        )
        # the decomposition behind those curves — which supervised term drives the total, and
        # which harmonic band the estimate actually improves in (skipped for a history with
        # neither diagnostic recorded)
        plot_loss_components(
            history, manifest, out_svg=str(eval_dir / "loss_components.svg")
        )

    # -- where the loss sits: per injection node, per term, fundamental vs harmonics --
    # The signature of a cancelling consumption/generation pair is a LOW voltage loss beside
    # a HIGH injection loss: the voltages cannot resolve the split, so the injection term
    # penalizes what no measurement reveals. Skipped when it cannot be built (e.g. a grid
    # with no injection appliance on the plotted phase).
    try:
        maps = loss_term_maps(
            module,
            source_or_path=source,
            grid=grid,
            measurement=measurement,
            split="val",
            device=device,
            max_batches=loss_heatmap_batches,
        )
        plot_loss_heatmap(
            maps,
            out_svg=str(eval_dir / "loss_heatmap_a.svg"),
            out_html=str(eval_dir / "loss_heatmap_a.html"),
        )
    except (
        ValueError,
        KeyError,
    ) as exc:  # diagnostic only — never fail the report on it
        print(f"  loss heatmap skipped: {exc}", flush=True)

    # -- error overview (mean = typical, max = worst-case; fundamental scaled separately) --
    # one heatmap per phase so the per-phase error pattern is not averaged across phases.
    phases = _resolve_heatmap_phases(prediction, heatmap_phases)
    for stat in ("mean", "max"):
        for ph in phases:
            plot_error_heatmap(
                prediction,
                statistic=stat,
                phase=ph,
                out_svg=str(eval_dir / f"error_heatmap_{stat}_{ph.value}.svg"),
                out_html=str(eval_dir / f"error_heatmap_{stat}_{ph.value}.html"),
            )
    plot_error_by_order(prediction, out_svg=str(eval_dir / "error_by_order.svg"))
    plot_error_cdf(prediction, out_svg=str(eval_dir / "error_cdf.svg"))
    plot_error_cdf(
        prediction, xscale="linear", out_svg=str(eval_dir / "error_cdf_linear.svg")
    )
    plot_error_vs_sensor_distance(
        prediction, out_svg=str(eval_dir / "error_vs_distance.svg")
    )
    plot_parity_per_harmonic(
        prediction,
        out_svg=str(eval_dir / "parity.svg"),
        out_png=str(eval_dir / "parity.png"),
    )

    # -- state vs prediction for the WORST sample (the hardest case) --
    worst = worst_scenario(prediction)
    fundamental = orders[0]
    plot_state_vs_prediction(
        prediction,
        order=fundamental,
        phase=Phase.A,
        scenario=worst,
        out_svg=str(eval_dir / f"state_h{fundamental}.svg"),
        out_html=str(eval_dir / f"state_h{fundamental}.html"),
        out_html_3d=str(eval_dir / "state_3d.html"),
        orders_3d=orders,
    )
    # a representative harmonic too (the first injected order, if any)
    injected = [o for o in orders if o > 1]
    if injected:
        h = injected[0]
        plot_state_vs_prediction(
            prediction,
            order=h,
            phase=Phase.A,
            scenario=worst,
            out_svg=str(eval_dir / f"state_h{h}.svg"),
            out_html=str(eval_dir / f"state_h{h}.html"),
        )

    return eval_dir


__all__ = ["evaluate_run"]
