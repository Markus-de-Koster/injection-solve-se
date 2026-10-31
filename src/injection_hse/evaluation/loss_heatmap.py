"""Where the supervised loss sits: per injection node, per term, fundamental vs harmonics.

The state estimate is under-determined wherever consumption and generation cancel. A load and
a generator on one bus reach the network only through their SUM, and so do two of them on
NEIGHBOURING buses with no meter in between: ``+20 kW`` against ``−20 kW`` produces the same
voltage profile as ``+2 kW`` against ``−2 kW``. The voltage term is therefore satisfied by a
whole family of injections, while the injection term penalizes every member of it but one —
so at those nodes a LOW voltage loss coexists with a HIGH injection loss.

That signature is what this heatmap exposes. Rows are the injection nodes (any bus carrying a
load or a generator); columns are the two supervised terms and their blend, each at the
fundamental and averaged over the harmonic orders. The cancelling family is a fundamental-only
degeneracy — a load and an inverter distort differently, so their harmonic injections do not
cancel — which is why the fundamental and harmonic halves are drawn side by side: the pattern
to look for is a bright injection column at the fundamental that is NOT bright at the
harmonics, i.e. an ambiguity the spectrum resolves.

Comparing a voltage cell against an injection cell is only meaningful because the blend
normalizes the two terms onto a common scale
(:attr:`injection_hse.config.LossConfig.injection_state_loss_normalize`); on a raw-blend run the
injection column is larger everywhere for reasons that have nothing to do with identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from pgml.evaluation.style import positive_log_norm, save_figure, save_html
from pgml.schemas.grid_schema import Phase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pgml.schemas.grid_schema import Grid

    from ..data import DataSource
    from ..train import SEModule

_PHASE_CODE = {Phase.A: 0, Phase.B: 1, Phase.C: 2, Phase.N: 3}
_PHASE_LETTER = {0: "a", 1: "b", 2: "c", 3: "n"}
# Column layout: the two terms and their blend, at the fundamental then over the harmonics.
# Column labels. The injection term is the residual on the nodal injection CURRENT
# ``I(h) = Y(h)·V(h)`` — NOT a power — so it is never labelled P, and "I" alone would read as
# a current rather than as "the injection term". Spell the terms out; the panels already
# carry the band, so no f/h subscript is needed.
_TERMS = (("voltage", "$L_{volt}$"), ("injection", "$L_{inj}$"), ("total", "$L_{tot}$"))


@dataclass
class LossTermMaps:
    """Per-row, per-order loss of each supervised term, averaged over an evaluation split.

    ``voltage`` / ``injection`` / ``total`` are ``[N, H]`` in the loss's own units (the
    weighted, per-harmonic-standardized residual — :meth:`injection_hse.loss.StateEstimationLoss
    .per_node_order`), so a cell is directly the contribution that row and order makes to the
    reported loss. ``injection`` carries the blend's own scaling, which is what puts it on the
    voltage map's scale and makes a cell-for-cell comparison mean something; it is ``None``
    for a run without the direct injection term (a voltage decode, or
    ``injection_state_loss_split=0``), and ``total`` is then the voltage map. Otherwise
    ``total == (1 - s) * voltage + s * injection`` holds cell for cell (``s`` the blend
    split).
    ``injection_rows`` are the rows whose node carries a load or a generator;
    ``observed_rows`` the metered placement.
    """

    voltage: np.ndarray  # [N, H]
    injection: Optional[np.ndarray]  # [N, H]
    total: np.ndarray  # [N, H]
    orders: list[int]
    node_ids: np.ndarray  # int64 [N]
    phase_codes: np.ndarray  # int64 [N]
    injection_rows: np.ndarray  # int64 [K]
    observed_rows: np.ndarray  # int64 [M]
    grid: "Grid"

    @property
    def has_injection_term(self) -> bool:
        return self.injection is not None


def _injection_rows(grid) -> np.ndarray:
    """Rows whose node carries an in-service load, generator or storage."""
    from pgml.assembly import node_phase_index
    from pgml.schemas.grid_schema import InjectionAppliance

    index = node_phase_index(grid)
    nodes = {
        int(a.node)
        for a in grid.appliances
        if isinstance(a, InjectionAppliance) and a.in_service
    }
    rows = sorted({r for n in sorted(nodes) for r in index.rows(n)})
    return np.asarray(rows, dtype=np.int64)


def loss_term_maps(
    module: "SEModule",
    *,
    source_or_path: Union[str, "DataSource"],
    grid: "Grid",
    measurement,
    split: str = "val",
    device: Union[str, torch.device] = "cpu",
    max_batches: Optional[int] = None,
) -> LossTermMaps:
    """Average each supervised loss term per row and order over a dataset split.

    Runs the module exactly as :func:`~injection_hse.evaluation.predict_dataset` does — the same
    placement masker, the same eval-mode forward — but keeps the loss's own per-row, per-order
    intermediate instead of the decoded state. ``max_batches`` caps the pass (the maps are
    means; a few hundred samples already resolve the pattern).
    """
    from pgml.assembly import node_phase_index
    from torch.utils.data import DataLoader

    from ..data import MeasurementModel, StateCollate, build_data_source
    from ..masking import build_masker

    if isinstance(source_or_path, (str, bytes)) or hasattr(
        source_or_path, "__fspath__"
    ):
        # inherit the checkpoint's own split settings, as predict_dataset does, so "val"
        # here is the same partition training validated on
        data_cfg = module.cfg.data.model_copy(
            update={"source": "parquet", "dataset_path": str(source_or_path)}
        )
        source = build_data_source(data_cfg, grid=grid)
    else:
        source = source_or_path

    model = MeasurementModel.resolve(
        grid,
        measurement.voltage_nodes,
        fundamental_nodes=measurement.fundamental_nodes,
        slack_known=measurement.slack_known,
    )
    observed_rows = model.observed_rows(grid)
    module.set_masker(
        build_masker(
            "placement",
            observed_rows=observed_rows,
            order_capability=(
                model.order_capability(grid, list(module.orders))
                if measurement.heterogeneous_orders
                else None
            ),
            observed_edge_rows=(
                model.observed_edges(grid) if measurement.measure_current else None
            ),
        )
    )
    datasets = {
        "train": source.train_dataset,
        "val": source.val_dataset,
        "test": source.test_dataset,
    }
    if split not in datasets:
        raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}.")
    loader = DataLoader(
        datasets[split](),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=StateCollate(),
    )

    module = module.to(device).eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0
    module._collect_loss_maps = True
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = {
                    k: (t.to(device) if torch.is_tensor(t) else t)
                    for k, t in batch.items()
                }
                module._step(batch, train=False)
                seen = 0
                for name, value in module._loss_maps.items():
                    flat = value.reshape(-1, *value.shape[-2:])  # [B*, N, H]
                    summed = flat.sum(dim=0).to(torch.float64).cpu()
                    totals[name] = (
                        summed if name not in totals else totals[name] + summed
                    )
                    seen = int(flat.shape[0])
                count += seen
    finally:
        module._collect_loss_maps = False
        module._loss_maps = {}

    if not count:
        raise ValueError(f"the {split!r} split yielded no loss maps to average.")
    maps = {k: (v / count).numpy() for k, v in totals.items()}

    idx = node_phase_index(grid)
    node_ids = idx.node_ids.cpu().numpy().astype(np.int64)
    phase_codes = idx.phase_codes.cpu().numpy().astype(np.int64)
    return LossTermMaps(
        voltage=maps["voltage"],
        injection=maps.get("injection"),
        total=maps["total"],
        orders=list(module.orders),
        node_ids=node_ids,
        phase_codes=phase_codes,
        injection_rows=_injection_rows(grid),
        observed_rows=observed_rows.cpu().numpy().astype(np.int64),
        grid=grid,
    )


def _band_columns(
    maps: LossTermMaps,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``({term: [N]} fundamental, {term: [N]} harmonic mean)`` from the per-order maps."""
    orders = list(maps.orders)
    fund = [i for i, o in enumerate(orders) if o == 1]
    harm = [i for i, o in enumerate(orders) if o != 1]
    available = {"voltage": maps.voltage, "total": maps.total}
    if maps.injection is not None:
        available["injection"] = maps.injection

    def band(mat: np.ndarray, cols: list[int]) -> np.ndarray:
        return mat[:, cols].mean(axis=1) if cols else np.full(mat.shape[0], np.nan)

    return (
        {k: band(v, fund) for k, v in available.items()},
        {k: band(v, harm) for k, v in available.items()},
    )


def plot_loss_heatmap(
    maps: LossTermMaps,
    *,
    phase: Optional[Phase] = Phase.A,
    out_svg: Optional[str] = None,
    out_html: Optional[str] = None,
    log: bool = True,
    cmap: str = "magma",
    title: Optional[str] = None,
):
    """Loss per injection node (y) and per term-band column (x), fundamental beside harmonics.

    Columns are the voltage / injection / total loss terms, at the fundamental in the left
    panel and averaged over the harmonic orders in the right one — drawn as two panels with a gap and
    INDEPENDENT colour scales, since the fundamental loss is decades above the harmonic one
    and a shared scale would render the harmonic half flat (the same reason
    :func:`~injection_hse.evaluation.plot_error_heatmap` splits its panels). Within a panel the three
    terms DO share a scale: that comparison is the point, and the loss normalization is what
    makes it legitimate.

    Rows are the injection nodes only — the buses that can host a cancelling
    consumption/generation pair — filtered to one ``phase`` (default :attr:`Phase.A`; ``None``
    keeps every phase). Sensor rows are marked, because a cancelling pair is only ambiguous
    while nothing between the two buses is metered. ``log`` puts both panels on a log colour
    scale (loss values span decades). Writes ``out_svg`` / ``out_html`` when given and returns
    the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt
    from pgml.evaluation import node_numbering

    rows = maps.injection_rows
    if phase is not None:
        want = _PHASE_CODE[phase]
        rows = np.asarray(
            [r for r in rows if int(maps.phase_codes[int(r)]) == want], dtype=np.int64
        )
    if rows.size == 0:
        raise ValueError(
            "no injection rows to plot for the requested phase "
            f"({phase}); the grid may carry no load/generator on it."
        )

    numbering = node_numbering(maps.grid)
    labels = [
        f"{numbering.get(int(maps.node_ids[int(r)]), int(maps.node_ids[int(r)]))}"
        f"·{_PHASE_LETTER[int(maps.phase_codes[int(r)])]}"
        for r in rows
    ]
    observed = set(int(r) for r in maps.observed_rows)
    sensor_mask = np.array([int(r) in observed for r in rows])

    fund, harm = _band_columns(maps)
    keys = [k for k, _ in _TERMS if k in fund]
    tick_labels = {k: short for k, short in _TERMS}

    panels = []
    for band, name in ((fund, "fundamental"), (harm, "harmonics")):
        mat = np.stack([band[k][rows] for k in keys], axis=1)  # [n_rows, n_terms]
        if np.all(~np.isfinite(mat)):
            continue
        panels.append((mat, [tick_labels[k] for k in keys], name))
    if not panels:
        raise ValueError(
            "the loss maps carry neither a fundamental nor a harmonic order."
        )

    height = max(4.0, 0.16 * len(rows) + 1.5)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(2.1 * len(panels) + 3.4, height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.28},
        squeeze=False,
    )
    for ax, (mat, xticks, name) in zip(axes[0], panels):
        norm = positive_log_norm(mat) if log else None
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower", norm=norm)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)
        ax.set_title(name, fontsize=9)
        fig.colorbar(im, ax=ax, label="loss", pad=0.02)
        if ax is axes[0][0]:
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(labels, fontsize=6)
            _mark_sensor_rows(ax, sensor_mask)
            ax.set_ylabel("Injection node·phase")
        else:
            ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)

    if out_svg is not None:
        save_figure(fig, out_svg)
    if out_html is not None:
        _plotly_loss_heatmap(panels, labels, out_html)
    return fig


def _mark_sensor_rows(ax, sensor_mask: np.ndarray) -> None:
    """Red y-tick labels on the metered rows (a cancelling pair needs no meter between)."""
    if not sensor_mask.any():
        return
    for tick, is_sensor in zip(ax.get_yticklabels(), sensor_mask):
        if is_sensor:
            tick.set_color("tab:red")
            tick.set_fontweight("bold")


def _plotly_loss_heatmap(panels, labels, out_html) -> None:
    """The interactive counterpart (one subplot per band, shared row axis)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        subplot_titles=[name for _, _, name in panels],
        horizontal_spacing=0.12,
    )
    for col, (mat, xticks, _name) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=[t.replace("$", "").replace("_", "") for t in xticks],
                y=labels,
                colorscale="Magma",
                colorbar=dict(title="loss", x=1.02 if col == len(panels) else None),
                showscale=col == len(panels),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        xaxis_title="term",
        yaxis_title="Injection node·phase",
        width=520 + 120 * len(panels),
        height=max(480, 16 * len(labels)),
    )
    save_html(fig, out_html)


__all__ = ["LossTermMaps", "loss_term_maps", "plot_loss_heatmap"]
