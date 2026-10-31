"""Attribution through the network and the anchored solve, on the test sequences.

The deployment setting of the paper's table: the test split of ``<data>/dataset_coherent``
(200 sequences of 8 steps = 1 600 states), the deployed placement (the supply and the three
meters, each with its upstream CT), class-A readings drawn exactly as the evaluation draws
them (:mod:`se_ablation.evaluate`: voltage noise seed 0, current noise seed 7, batches of 16
sequences in split order) and the checkpoint's deployed anchor weights. The temporal network
sees whole sequences with their time features. The anchored decode reads the noisy readings
(anchors fixed); integrated gradients scale only the network's input features, from the
all-zero input (the network's encoding of "unmeasured") with the placement mask held fixed,
64 midpoint-rule steps.

Target per output order ``h``: ``F_h`` = the summed per-unit estimated ``|V̂_h|`` over the
unobserved node-phase rows, i.e. the voltages after the solve.

Outputs (``out``):

- ``attr_harmonic_coupling_solve.csv`` (+ PDF / SVG): the integrated-gradient coupling of
  ``F_h`` to each input order (absolute attribution mass of the voltage and current features
  of that order) and the order occlusion (mean ``|Δ|V̂_h||`` per unobserved row when the
  inputs of one order are zeroed, anchors kept);
- ``attr_order_importance_solve.csv``: the integrated-gradient share per input order;
- ``h1_error_sign_check.csv``: the h1 pu-MAE change when the h3–h19 inputs or the h1 inputs
  are zeroed (anchors kept), the signed integrated gradient of the h1 error over the same
  inputs, and the h1 output's attribution share from those inputs;
- ``completeness.csv``: per order ``F(x)``, ``F(0)`` (all-zero inputs), the network share
  ``|F(x) − F(0)| / F(x)``, the signed integrated-gradient sum against ``F(x) − F(0)``
  (completeness gap), the mean ``|ΔV̂|`` against the zero-input estimate and against the
  anchors-only solve (zero network output), and the pu-MAE of the estimate and of the
  anchors-only solve.

Runs on the CPU in the checkpoint's dtype; the attributions are accumulated in float64.
``max_batches`` limits the pass to the first batches of 16 sequences (a quick check).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .frame import BATCH, NOISE_SEED
from .names import PLACEMENTS


def attribution(
    checkpoint: Path,
    data: Path,
    out: Path,
    *,
    sensors=None,
    ig_steps: int = 64,
    chunk: int = 4,
    max_batches: int = 0,
    device: str = "cpu",
) -> Path:
    """Write the attribution through network and solve of ``checkpoint`` on the test
    sequences of ``data`` to ``out`` (``chunk`` sequences per gradient pass)."""
    from pgml.schemas.grid_schema import Grid
    from torch.utils.data import DataLoader

    from injection_hse.analysis.attribution import (
        _decode_state,
        _edge_features,
        _masked_features,
        _model_output,
        _prior_features,
        _rnn_grad_safe,
    )
    from injection_hse.analysis.style import save_paper_figure
    from injection_hse.data import (
        MeasurementModel,
        StateCollate,
        base_voltage_per_row,
        build_data_source,
    )
    from injection_hse.evaluation import load_se_module
    from injection_hse.masking import build_masker

    out.mkdir(parents=True, exist_ok=True)
    data = Path(data)
    grid = Grid.model_validate_json((data / "grid.json").read_text())
    module = load_se_module(
        str(checkpoint), grid=grid, dataset_path=str(data / "dataset_coherent"), device=device
    )
    module.eval()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["anchor_weights"]
    module.set_stage_anchor(v_weight=float(weights[0]), i_weight=float(weights[1]))
    mm = MeasurementModel.resolve(grid, list(sensors or PLACEMENTS["deployed"]), slack_known=True)
    module.set_masker(
        build_masker(
            "placement",
            observed_rows=mm.observed_rows(grid),
            observed_edge_rows=mm.observed_edges(grid),
        )
    )
    cfg = module.cfg.data.model_copy(
        update={"dataset_path": str(data / "dataset_coherent"), "num_workers": 0}
    )
    source = build_data_source(cfg, grid=grid)
    loader = DataLoader(
        source.test_dataset(),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        collate_fn=StateCollate(),
    )
    module._noise.reseed(NOISE_SEED)
    module._edge_noise.reseed(NOISE_SEED + 7)

    base = base_voltage_per_row(grid).to(torch.float64).view(1, 1, -1)
    orders = list(module.orders)
    h, c = module.n_harmonics, module.n_channels
    mask_row = torch.zeros(module.n_nodes, dtype=torch.bool)
    mask_row[mm.observed_rows(grid)] = True
    unobs_row = (~mask_row).to(torch.float64).view(1, 1, -1)

    coupling = np.zeros((h, h))
    occl = np.zeros((h, h))
    acc = {k: np.zeros(h) for k in ("F_x", "F_0", "IG", "d0", "dphys", "mae", "mae_phys")}
    err = {"none": 0.0, "harmonic inputs (h3-h19)": 0.0, "h1 inputs": 0.0}
    sign_ig = {"harmonic inputs (h3-h19)": 0.0, "h1 inputs": 0.0}
    n_unobs = 0.0
    alphas = ((torch.arange(ig_steps, dtype=torch.float64) + 0.5) / ig_steps).tolist()
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        v_true = batch["v"].to(torch.complex128)  # [S, T, H, N]
        s_all, t = v_true.shape[:2]
        with torch.no_grad():
            # the readings as the module draws them in inference (the same stream order)
            v_in = module._noise.apply(v_true.to(module._complex_dtype))
            i_edges = module._edge_noise.apply(module._derive_edge_currents(v_true, None))
            i_edges = i_edges.to(v_in.dtype)
            node_time_all = module._build_node_time(batch, (s_all, t), v_in.device)
        for a0 in range(0, s_all, chunk):
            sl = slice(a0, a0 + chunk)
            rows = v_true[sl].shape[0] * t
            vt = v_true[sl].reshape(rows, h, -1)
            vi = v_in[sl].reshape(rows, h, -1)
            ie = i_edges[sl].reshape(rows, h, -1)
            mb = mask_row.unsqueeze(0).expand(rows, -1).clone()
            nt = node_time_all[sl] if node_time_all is not None else None
            unobs = unobs_row.expand(rows, 1, -1)
            n_unobs += float(unobs.sum())
            with torch.no_grad():
                x = _masked_features(module, vi, mb)
                ex, em, ie2 = _edge_features(module, vi, mb, i_edges=ie)
                pr = _prior_features(module, vi, mb, em, ie2)
            ec = int(ex.shape[-1]) // h if ex is not None else 0

            def run(xi, exi, zero_out=False):
                o = _model_output(module, xi, mb, exi, em, pr, steps=t, node_time=nt)
                if zero_out:
                    o = torch.zeros_like(o)
                return _decode_state(module, o, vi, mask=mb, edge_mask=em, i_edges=ie2).to(
                    torch.complex128
                )

            def mag(z):
                return z.abs() / base

            def h1_err(z):
                return float(
                    ((z[:, 0, :] - vt[:, 0, :]).abs() / base[:, 0, :] * unobs[:, 0, :]).sum()
                )

            with torch.no_grad():
                full = run(x, ex)
                zin = run(torch.zeros_like(x), torch.zeros_like(ex) if ex is not None else None)
                zout = run(x, ex, zero_out=True)
                acc["F_x"] += (mag(full) * unobs).sum(dim=(0, 2)).numpy()
                acc["F_0"] += (mag(zin) * unobs).sum(dim=(0, 2)).numpy()
                acc["d0"] += (((full - zin).abs() / base) * unobs).sum(dim=(0, 2)).numpy()
                acc["dphys"] += (((full - zout).abs() / base) * unobs).sum(dim=(0, 2)).numpy()
                acc["mae"] += (((full - vt).abs() / base) * unobs).sum(dim=(0, 2)).numpy()
                acc["mae_phys"] += (((zout - vt).abs() / base) * unobs).sum(dim=(0, 2)).numpy()
                err["none"] += h1_err(full)
                ref = mag(full)
                groups = {j: [j] for j in range(h)}
                groups["harmonic inputs (h3-h19)"] = list(range(1, h))
                groups["h1 inputs"] = [0]
                for key, idx in groups.items():
                    xo, exo = x.clone(), (ex.clone() if ex is not None else None)
                    for j in idx:
                        xo[..., j * c : (j + 1) * c] = 0.0
                        if exo is not None:
                            exo[..., j * ec : (j + 1) * ec] = 0.0
                    pc = run(xo, exo)
                    if isinstance(key, int):
                        occl[:, key] += ((mag(pc) - ref).abs() * unobs).sum(dim=(0, 2)).numpy()
                    else:
                        err[key] += h1_err(pc)
            gx = [torch.zeros_like(x, dtype=torch.float64) for _ in range(h)]
            ge = (
                [torch.zeros_like(ex, dtype=torch.float64) for _ in range(h)]
                if ex is not None
                else None
            )
            gex = torch.zeros_like(x, dtype=torch.float64)
            gee = torch.zeros_like(ex, dtype=torch.float64) if ex is not None else None
            with _rnn_grad_safe():
                for a in alphas:
                    xa = (x * a).detach().requires_grad_(True)
                    leaves = [xa]
                    exa = None
                    if ex is not None:
                        exa = (ex * a).detach().requires_grad_(True)
                        leaves.append(exa)
                    pred = run(xa, exa)
                    m = mag(pred) * unobs
                    for k in range(h):
                        g = torch.autograd.grad(m[:, k, :].sum(), leaves, retain_graph=True)
                        gx[k] += g[0].to(torch.float64)
                        if ex is not None:
                            ge[k] += g[1].to(torch.float64)
                    e1 = (
                        (pred[:, 0, :] - vt[:, 0, :]).abs() / base[:, 0, :] * unobs[:, 0, :]
                    ).sum()
                    g = torch.autograd.grad(e1, leaves)
                    gex += g[0].to(torch.float64)
                    if ex is not None:
                        gee += g[1].to(torch.float64)
            xd = x.detach().to(torch.float64)
            exd = ex.detach().to(torch.float64) if ex is not None else None
            for k in range(h):
                ax = (xd * gx[k] / ig_steps).reshape(rows, -1, h, c)
                acc["IG"][k] += float(ax.sum())
                per_in = ax.abs().sum(dim=(0, 1, 3)).numpy()
                if ex is not None:
                    ae = (exd * ge[k] / ig_steps).reshape(rows, -1, h, ec)
                    acc["IG"][k] += float(ae.sum())
                    per_in = per_in + ae.abs().sum(dim=(0, 1, 3)).numpy()
                coupling[k] += per_in
            sx = (xd * gex / ig_steps).reshape(rows, -1, h, c).sum(dim=(0, 1, 3))
            if ex is not None:
                sx = sx + (exd * gee / ig_steps).reshape(rows, -1, h, ec).sum(dim=(0, 1, 3))
            sign_ig["h1 inputs"] += float(sx[0])
            sign_ig["harmonic inputs (h3-h19)"] += float(sx[1:].sum())
        print(f"batch {bi}: unobserved entries {n_unobs:.0f}", flush=True)

    occl /= n_unobs
    imp = coupling.sum(axis=0)
    pd.DataFrame({"input_order": orders, "ig_mass": imp, "share": imp / imp.sum()}).to_csv(
        out / "attr_order_importance_solve.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "output_order": o_out,
                "input_order": o_in,
                "ig_coupling_solve": coupling[i, j],
                "order_occlusion_solve": occl[i, j],
            }
            for i, o_out in enumerate(orders)
            for j, o_in in enumerate(orders)
        ]
    ).to_csv(out / "attr_harmonic_coupling_solve.csv", index=False)
    base_err = err["none"] / n_unobs
    sign_rows = []
    for key in ("harmonic inputs (h3-h19)", "h1 inputs"):
        e = err[key] / n_unobs
        share = coupling[0, 1:].sum() if key.startswith("harmonic") else coupling[0, 0]
        sign_rows.append(
            {
                "inputs": key,
                "h1_pu_mae_before": base_err,
                "h1_pu_mae_occluded": e,
                "occlusion_change_pct": 100 * (e / base_err - 1),
                "signed_ig_of_h1_error": sign_ig[key] / n_unobs,
                "h1_row_ig_share": float(share / coupling[0].sum()),
            }
        )
    pd.DataFrame(sign_rows).to_csv(out / "h1_error_sign_check.csv", index=False)
    comp = []
    for k, o in enumerate(orders):
        d = acc["F_x"][k] - acc["F_0"][k]
        comp.append(
            {
                "order": o,
                "F_x": acc["F_x"][k],
                "F_0": acc["F_0"][k],
                "network_share": abs(d) / acc["F_x"][k],
                "IG_sum": acc["IG"][k],
                "F_x_minus_F_0": d,
                "completeness_gap": abs(acc["IG"][k] - d) / abs(d) if d else np.nan,
                "mean_abs_dV_vs_zero_input_pu": acc["d0"][k] / n_unobs,
                "mean_abs_dV_vs_anchors_only_pu": acc["dphys"][k] / n_unobs,
                "pu_mae_estimate": acc["mae"][k] / n_unobs,
                "pu_mae_anchors_only": acc["mae_phys"][k] / n_unobs,
            }
        )
    cdf = pd.DataFrame(comp)
    harm = cdf[cdf.order > 1].mean(numeric_only=True).to_dict()
    harm["order"] = "harm (mean h3-h19)"
    pd.concat([cdf, pd.DataFrame([harm])], ignore_index=True).to_csv(
        out / "completeness.csv", index=False
    )
    save_paper_figure(coupling_plot(orders, coupling, occl), out, "attr_harmonic_coupling_solve")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(sign_rows).to_string(index=False))
        print("input-order shares", np.round(imp / imp.sum(), 3).tolist())
    return out


def coupling_plot(orders, coupling, occlusion):
    """The two-panel order coupling (integrated gradients through the solve, occlusion with
    the anchors kept), each row normalized, log colour scale."""
    from injection_hse.analysis.report import plot_harmonic_coupling

    fig = plot_harmonic_coupling(list(orders), np.asarray(coupling), np.asarray(occlusion))
    titled = [a for a in fig.axes if a.get_title()]
    fs = titled[1].title.get_fontsize()
    titled[0].set_title("IG through the solve", fontsize=fs)
    titled[1].set_title("occlusion, anchors kept", fontsize=fs)
    return fig
