"""HistoryCollector accumulates across multiple fit calls; the loss-curve plots write.

The history has two sources — the collected ``training_history.parquet`` and the MLflow file
store a run logs to — and ONE reader and plot serve both, so the tests below also pin the
reshaping of a store into the collected frame's shape.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

import json

import lightning as L
import polars as pl
from pytest import approx
from torch.utils.data import DataLoader, TensorDataset

from injection_hse.evaluation.history import (
    HistoryCollector,
    mlflow_history,
    plot_loss_components,
    plot_loss_curves,
    read_history,
    read_mlflow_metrics,
    read_training_history,
    stage_manifest,
)


class _Tiny(L.LightningModule):
    """A minimal LightningModule logging the metrics the collector reads."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)

    def training_step(self, batch, _):
        loss = self.lin(batch[0]).sum()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss = self.lin(batch[0]).sum()
        self.log("val_loss", loss)
        self.log("val_pu_mae", torch.tensor(0.6))
        self.log("val_pu_mae_unobs", torch.tensor(0.5))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


class _SentinelTrainLoss(_Tiny):
    """Exposes an epoch-mean ``train_epoch_loss`` distinct from the per-step ``train_loss``.

    The per-step ``train_loss`` logged in :meth:`training_step` is the last minibatch; the
    property returns a fixed sentinel. A collector that records the epoch MEAN must pick up the
    sentinel, proving it reads the property and not ``callback_metrics["train_loss"]``.
    """

    SENTINEL = 123.0

    @property
    def train_epoch_loss(self) -> float:
        return self.SENTINEL


class _Decomposed(_Tiny):
    """Logs the loss-term and per-order diagnostics on top of the headline metrics."""

    ORDERS = (1, 3, 5)

    def validation_step(self, batch, _):
        super().validation_step(batch, _)
        self.log("val_voltage_loss", torch.tensor(0.2))
        self.log("val_injection_state_loss", torch.tensor(0.9))
        self.log("val_pu_mae_obs", torch.tensor(0.1))
        self.log("val_pu_mae_unobs_fund", torch.tensor(0.4))
        self.log("val_pu_mae_unobs_harm", torch.tensor(0.05))
        for i, order in enumerate(self.ORDERS):
            self.log(f"val_pu_mae_h{order}", torch.tensor(0.6 / (i + 1)))
            self.log(f"val_pu_mae_unobs_h{order}", torch.tensor(0.5 / (i + 1)))
            self.log(f"val_rel_unobs_h{order}", torch.tensor(0.02 * (i + 1)))


def _loader():
    return DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)


def test_collector_records_epoch_mean_train_loss(tmp_path):
    hc = HistoryCollector()
    module = _SentinelTrainLoss()
    trainer = L.Trainer(
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
        callbacks=[hc],
        num_sanity_val_steps=0,
    )
    trainer.fit(module, _loader(), _loader())
    df = hc.to_frame()
    # every recorded training loss is the module's epoch-mean property, NOT the last minibatch
    assert df["train_loss"].to_list() == [module.SENTINEL, module.SENTINEL]


def test_collector_spans_multiple_fits(tmp_path):
    hc = HistoryCollector()
    module = _Tiny()
    for _ in range(3):  # three "stages"
        trainer = L.Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            callbacks=[hc],
            num_sanity_val_steps=2,
        )
        trainer.fit(module, _loader(), _loader())

    df = hc.to_frame()
    # one record per validation epoch across all three fits; stage index increments per fit
    assert df.height == 6
    assert df["stage_index"].to_list() == [0, 0, 1, 1, 2, 2]
    assert df["global_epoch"].to_list() == [0, 1, 2, 3, 4, 5]  # monotonic

    path = hc.write_parquet(tmp_path / "history.parquet")
    assert path.exists()
    reloaded = read_history(path)
    assert reloaded.height == 6


def test_plot_loss_curves_writes(tmp_path):
    hc = HistoryCollector()
    module = _Tiny()
    for _ in range(2):
        trainer = L.Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            callbacks=[hc],
            num_sanity_val_steps=0,
        )
        trainer.fit(module, _loader(), _loader())

    manifest = {
        "switch_stage_index": 1,
        "stages": {
            "0": {
                "task": "A",
                "stage_name": "full",
                "kind": "full",
                "observed_fraction": None,
            },
            "1": {
                "task": "B",
                "stage_name": "placement",
                "kind": "placement",
                "observed_fraction": None,
            },
        },
    }
    svg = tmp_path / "loss.svg"
    html = tmp_path / "loss.html"
    fig = plot_loss_curves(
        hc.to_frame(), manifest, out_svg=str(svg), out_html=str(html)
    )
    assert svg.exists() and svg.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0
    # the A->B switch stage boundary is drawn as a vertical line
    assert len(fig.axes[0].lines) >= 3  # train + val + pu-mae curves
    plt.close(fig)


def _fit(module, callback, tmp_path, fits=2):
    for _ in range(fits):
        L.Trainer(
            max_epochs=2,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            callbacks=[callback],
            num_sanity_val_steps=0,
        ).fit(module, _loader(), _loader())


def test_collector_records_the_loss_terms_and_per_order_metrics(tmp_path):
    """Whatever diagnostics the module logs land in the history, one column each.

    The headline columns keep their leading positions, the per-order columns follow the
    orders the run actually solved, and the ``*_harm`` band aggregate is NOT mistaken for a
    per-order column (its name shares the per-order prefix).
    """
    hc = HistoryCollector()
    _fit(_Decomposed(), hc, tmp_path)
    df = hc.to_frame()

    assert df.columns[:6] == [
        "global_epoch",
        "stage_index",
        "train_loss",
        "val_loss",
        "val_pu_mae",
        "val_pu_mae_unobs",
    ]
    for key in ("val_voltage_loss", "val_injection_state_loss", "val_pu_mae_obs"):
        assert key in df.columns
    for order in _Decomposed.ORDERS:
        assert f"val_pu_mae_unobs_h{order}" in df.columns
        assert f"val_rel_unobs_h{order}" in df.columns
    assert df["val_voltage_loss"].to_list() == [approx(0.2)] * df.height
    assert df["val_pu_mae_unobs_h1"].to_list() == [approx(0.5)] * df.height
    # the band aggregate shares the per-order prefix but is not itself an order
    assert df["val_pu_mae_unobs_harm"].to_list() == [approx(0.05)] * df.height
    assert not any(c.endswith("_h") for c in df.columns)


def test_collector_without_diagnostics_keeps_the_headline_schema(tmp_path):
    """A module logging none of them yields exactly the four headline metric columns."""
    hc = HistoryCollector()
    _fit(_Tiny(), hc, tmp_path, fits=1)
    assert hc.to_frame().columns == [
        "global_epoch",
        "stage_index",
        "train_loss",
        "val_loss",
        "val_pu_mae",
        "val_pu_mae_unobs",
    ]


def test_plot_loss_components_writes(tmp_path):
    """Three panels: the loss terms, the per-order pu-MAE, the per-order relative error."""
    hc = HistoryCollector()
    _fit(_Decomposed(), hc, tmp_path)
    manifest = {
        "switch_stage_index": 1,
        "stages": {
            "0": {"task": "A", "stage_name": "full", "kind": "full"},
            "1": {"task": "B", "stage_name": "placement", "kind": "placement"},
        },
    }
    svg = tmp_path / "components.svg"
    fig = plot_loss_components(hc.to_frame(), manifest, out_svg=str(svg))
    assert svg.exists() and svg.stat().st_size > 0
    assert len(fig.axes) == 3
    # the loss panel carries the total plus both terms; each order panel one curve per order
    # (counted by legend label — ``axvline`` also appends to ``ax.lines``)
    labelled = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in fig.axes]
    assert labelled[0] == [
        "total (reported)",
        "voltage term",
        "injection term",
        "sequences",
    ]
    orders = [f"h{o}" for o in _Decomposed.ORDERS]
    assert labelled[1][: len(orders)] == orders
    assert labelled[2][: len(orders)] == orders
    plt.close(fig)


def test_plot_loss_components_skips_a_history_without_diagnostics(tmp_path):
    """Nothing to decompose -> no figure (an older run's history stays plottable elsewhere)."""
    hc = HistoryCollector()
    _fit(_Tiny(), hc, tmp_path, fits=1)
    assert plot_loss_components(hc.to_frame(), {"stages": {}}) is None


def test_a_diagnostic_that_starts_late_does_not_break_the_frame(tmp_path):
    """A column null for the first hundreds of epochs must still write.

    The identifiability diagnostics report nothing while a stage is fully observed, so they
    are null for every row of the first stage — hundreds of epochs. Inferring a column's type
    from the leading rows would type that column Null and fail on its first real value, at
    ``write_parquet``, i.e. AFTER the entire run has trained.
    """
    hc = HistoryCollector()
    hc.stage_index = 0
    for epoch in range(150):
        hc.global_epoch = epoch
        rec = {
            "global_epoch": epoch,
            "stage_index": 0,
            "train_loss": 1.0,
            "val_loss": 1.0,
            "val_pu_mae": 1.0,
            "val_pu_mae_unobs": 0.0,
        }
        if epoch >= 120:  # the diagnostic only starts once rows go unobserved
            rec["val_inj_rel_cancel_h1"] = 0.641099
        hc.records.append(rec)

    df = hc.to_frame()
    assert df["val_inj_rel_cancel_h1"].dtype == pl.Float64
    assert df["val_inj_rel_cancel_h1"][0] is None
    assert df["val_inj_rel_cancel_h1"][-1] == approx(0.641099)
    path = hc.write_parquet(tmp_path / "late.parquet")
    assert read_history(path).height == 150


def test_an_empty_history_keeps_the_headline_schema(tmp_path):
    """No records at all still yields the headline columns (a run that never validated)."""
    df = HistoryCollector().to_frame()
    assert df.height == 0
    assert df.columns == [
        "global_epoch",
        "stage_index",
        "train_loss",
        "val_loss",
        "val_pu_mae",
        "val_pu_mae_unobs",
    ]


# =============================================================================
# The other source: the MLflow file store
# =============================================================================
def _store(root: Path, series: dict[str, list[tuple[int, float, int]]]) -> Path:
    """An MLflow FILE store holding ``{metric: [(timestamp_ms, value, step)]}``."""
    run = root / "mlruns" / "exp" / "runid"
    (run / "metrics").mkdir(parents=True)
    (run / "meta.yaml").write_text("start_time: 100\n")
    for name, points in series.items():
        (run / "metrics" / name).write_text(
            "".join(f"{t} {v} {s}\n" for t, v, s in points), "utf-8"
        )
    return root / "mlruns"


#: Two curriculum stages of a multi-grid run: the step counter restarts at the boundary,
#: the training loss is logged per batch AND as the epoch mean at each validation epoch.
_TWO_STAGE_STORE = {
    "train_loss": [
        (1, 1.0, 10),
        (2, 0.5, 20),
        (3, 0.25, 30),
        (4, 0.2, 10),
        (5, 0.1, 20),
    ],
    "train_loss_epoch": [(3, 0.4, 30), (5, 0.15, 20)],
    "val_loss": [(3, 0.8, 30), (5, 0.3, 20)],
    "val_pu_mae": [(3, 0.9, 30), (5, 0.4, 20)],
    "val_pu_mae_unobs": [(3, 0.7, 30), (5, 0.2, 20)],
    "val_pu_mae_worst_grid": [(3, 1.2, 30), (5, 0.6, 20)],
    "epoch": [(1, 0.0, 10), (3, 1.0, 30)],
}


def test_a_tracking_store_reshapes_into_the_collected_frame(tmp_path):
    """One row per VALIDATION epoch, the stage index from the store's step restarts."""
    history = mlflow_history(read_mlflow_metrics(_store(tmp_path, _TWO_STAGE_STORE)))

    assert history.columns[:6] == [
        "global_epoch",
        "stage_index",
        "train_loss",
        "val_loss",
        "val_pu_mae",
        "val_pu_mae_unobs",
    ]
    assert history["global_epoch"].to_list() == [0, 1]
    assert history["stage_index"].to_list() == [0, 1]  # one per Trainer fit
    # the epoch MEAN, which is what the collected parquet records — not the last per-batch
    # training loss (0.25 / 0.1) that happens to sit at the same step
    assert history["train_loss"].to_list() == approx([0.4, 0.15])
    assert history["val_pu_mae"].to_list() == approx([0.9, 0.4])
    # a multi-grid run's worst member travels as its own column; Lightning's own epoch
    # counter is bookkeeping and does not
    assert history["val_pu_mae_worst_grid"].to_list() == approx([1.2, 0.6])
    assert "epoch" not in history.columns


def test_a_store_without_validation_metrics_still_yields_curves(tmp_path):
    """A train-only store anchors on its training points rather than plotting nothing."""
    store = _store(tmp_path, {"train_loss": [(1, 1.0, 10), (2, 0.5, 20)]})
    history = mlflow_history(read_mlflow_metrics(store))
    assert history.height == 2
    assert history["train_loss"].to_list() == approx([1.0, 0.5])
    assert history["val_loss"].to_list() == [None, None]


def test_an_empty_store_is_an_empty_history_and_no_figure(tmp_path):
    history = mlflow_history(read_mlflow_metrics(tmp_path / "nothing-here"))
    assert history.height == 0 and "val_pu_mae" in history.columns
    assert plot_loss_curves(history, {}) is None


def test_the_store_plots_through_the_one_loss_curve_figure(tmp_path):
    """Same figure as a collected run's, plus the cross-grid spread when it was logged."""
    history = mlflow_history(read_mlflow_metrics(_store(tmp_path, _TWO_STAGE_STORE)))
    fig = plot_loss_curves(history, stage_manifest(["full", "placement"]))
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "train loss" in labels and "val pu-MAE (worst grid)" in labels
    # the stage boundary is drawn and labelled from the curriculum's stage names
    assert "placement" in [text.get_text() for text in fig.axes[0].texts]
    plt.close(fig)


def test_read_training_history_prefers_the_collected_parquet(tmp_path):
    """Both sources present: the parquet wins, and manifest.json supplies the labels."""
    _store(tmp_path, _TWO_STAGE_STORE)
    pl.DataFrame(
        {
            "global_epoch": [0, 1, 2],
            "stage_index": [0, 0, 1],
            "train_loss": [1.0, 0.5, 0.25],
            "val_loss": [1.1, 0.6, 0.3],
            "val_pu_mae": [0.9, 0.5, 0.2],
            "val_pu_mae_unobs": [0.8, 0.4, 0.1],
        }
    ).write_parquet(tmp_path / "training_history.parquet")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"stages": {"0": {"stage_name": "full", "kind": "full"}}})
    )

    history, manifest = read_training_history(tmp_path)
    assert history.height == 3  # the store holds 2 epochs, the parquet 3
    assert manifest["stages"]["0"]["stage_name"] == "full"


def test_read_training_history_falls_back_to_the_store(tmp_path):
    """No parquet (a multi-grid run): the store answers, and stage_names label it."""
    _store(tmp_path, _TWO_STAGE_STORE)
    history, manifest = read_training_history(
        tmp_path, stage_names=["full", "placement"]
    )
    assert history["stage_index"].to_list() == [0, 1]
    assert manifest["stages"]["1"]["stage_name"] == "placement"

    empty, no_labels = read_training_history(tmp_path / "elsewhere")
    assert empty.height == 0 and no_labels == {}
