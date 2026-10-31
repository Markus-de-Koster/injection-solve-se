"""Run a trained state estimator over a dataset -> a plain-numpy :class:`Prediction`.

The bridge from a fitted :class:`~injection_hse.train.SEModule` (+ checkpoint) to the framework-agnostic
plot functions. Everything here runs OFF the autograd tape: the module is evaluated under
``torch.no_grad()`` (via :meth:`~injection_hse.train.SEModule.infer`) and the complex estimate / target /
mask are detached to numpy, exactly as :mod:`pgml.evaluation` does for the solver outputs.

A :class:`Prediction` folds any time axis into the sample axis (``S = B`` for a snapshot
dataset, ``S = B * T`` for a coherent sequence dataset) and carries the per-unit base voltage,
the node-phase row layout, and the sensor placement so every plot is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:  # heavy / optional imports kept out of the import path
    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import MeasurementConfig
    from injection_hse.data import DataSource
    from injection_hse.train import SEModule


# =============================================================================
# Prediction container (plain numpy; consumed by every plot function)
# =============================================================================
@dataclass(frozen=True)
class Prediction:
    """A trained estimator's output over a dataset split, ready for plotting.

    All arrays are plain numpy (detached from torch). The time axis (if any) is folded into
    the leading sample axis ``S`` (so ``S = B`` for a snapshot dataset, ``S = B * T`` for a
    coherent sequence dataset). Shapes:

    - ``target_v`` complex ``[S, H, N]`` — the ground-truth state (the SE label);
    - ``pred_v``   complex ``[S, H, N]`` — the estimated state (physical units);
    - ``mask``     bool ``[S, N]`` (``True`` = the node-phase row was observed for that sample);
    - ``base_v``   float64 ``[N]`` — per-row line-to-neutral base (the per-unit denominator);
    - ``orders``   the ``H`` harmonic orders (``orders[k]`` ↔ harmonic axis ``k``);
    - ``node_ids`` / ``phase_codes`` int64 ``[N]`` — the node-phase row layout;
    - ``observed_rows`` int64 ``[K]`` — the placement rows (all phases of each sensor, plus
      the metered slack rows — the slack is a measurement like any other);
    - ``sensor_names`` the sensor node names/ids the placement resolves to;
    - ``grid`` the :class:`~pgml.schemas.grid_schema.Grid` (for topology in the plots);
    - ``scenario_index`` int64 ``[S]`` — the dataset SCENARIO (``B``-axis) id of every sample
      (repeated over the ``T`` steps of a sequence), so a sample can be matched with its
      scenario's records; ``None`` for a prediction built without it.
    """

    target_v: np.ndarray  # complex [S, H, N]
    pred_v: np.ndarray  # complex [S, H, N]
    mask: np.ndarray  # bool    [S, N]
    orders: list[int]
    base_v: np.ndarray  # float64 [N]
    node_ids: np.ndarray  # int64  [N]
    phase_codes: np.ndarray  # int64  [N]
    observed_rows: np.ndarray  # int64 [K]
    sensor_names: list
    grid: "Grid"
    scenario_index: Optional[np.ndarray] = None  # int64 [S]

    # -- convenience ----------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.target_v.shape[0])

    @property
    def n_orders(self) -> int:
        return int(self.target_v.shape[1])

    @property
    def n_rows(self) -> int:
        return int(self.target_v.shape[-1])

    def abs_error_pu(self) -> np.ndarray:
        """Per-sample per-order per-row per-unit absolute error ``[S, H, N]``.

        ``|pred - target|`` divided by the per-row line-to-neutral base voltage — the same
        per-unit residual the training metrics report (:func:`injection_hse.metrics.state_errors`).
        """
        err = np.abs(self.pred_v - self.target_v)  # [S, H, N]
        return err / self.base_v[None, None, :]


# =============================================================================
# Checkpoint -> module
# =============================================================================
def load_se_module(
    checkpoint_path,
    *,
    grid: "Grid",
    dataset_path: str,
    device: Union[str, torch.device] = "cpu",
) -> "SEModule":
    """Rebuild a trained :class:`~injection_hse.train.SEModule` from a stage checkpoint.

    Builds the data source on ``dataset_path`` + ``grid`` from the
    :class:`~injection_hse.config.ExperimentConfig` STORED in the checkpoint (``pgl_config``) and hands
    it to :meth:`injection_hse.train.SEModule.from_checkpoint`, which restores the fitted normalizer
    BEFORE constructing the module (the loss resolves its ``physical`` per-harmonic weights
    from the normalizer's scale at construction; an unfit normalizer raises) and then loads
    the trained weights. Moved to ``device`` and put in ``eval`` mode.

    The ``grid`` and ``dataset_path`` must be the ones the run trained on (the dataset stores
    only the node-phase row layout, asserted to match the grid). Returns the module ready for
    :func:`predict_dataset`.
    """
    from injection_hse.config import ExperimentConfig
    from injection_hse.data import build_data_source
    from injection_hse.train import SEModule

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["pgl_config"])
    data_cfg = cfg.data.model_copy(update={"dataset_path": str(dataset_path)})
    source = build_data_source(data_cfg, grid=grid)
    return SEModule.from_checkpoint(
        checkpoint_path, grid=grid, device=device, source=source
    )


# =============================================================================
# Module + dataset -> Prediction
# =============================================================================
def predict_dataset(
    module: "SEModule",
    *,
    source_or_path: Union[str, "DataSource"],
    grid: "Grid",
    measurement: "MeasurementConfig",
    split: str = "val",
    device: Union[str, torch.device] = "cpu",
    max_samples: Optional[int] = None,
) -> Prediction:
    """Run ``module`` over a dataset split under the sensor placement -> a :class:`Prediction`.

    Installs the :class:`~injection_hse.data.MeasurementModel` placement masker on the module (so the
    evaluation reflects the real sensor scarcity, regardless of the curriculum stage the
    module last trained), iterates the requested split's dataloader calling
    :meth:`~injection_hse.train.SEModule.infer`, and concatenates the complex estimate / target / mask
    over the batch (folding any ``T`` into the sample axis). All arrays are detached to numpy.

    Parameters
    ----------
    source_or_path:
        A :class:`~injection_hse.data.DataSource` or a dataset directory path. A PATH is built into a
        :class:`~injection_hse.data.ParquetDataset` on the run's grid from ``module.cfg.data`` (the
        checkpoint's ``val_fraction`` / ``test_fraction`` / ``split_seed`` carried over), so
        ``"val"`` is the split training validated on; a bare ``DataConfig`` would fall back
        to its defaults and let validation rows overlap the training set.
    measurement:
        The sensor placement, resolved through :meth:`injection_hse.data.MeasurementModel.resolve`
        exactly as training resolved it. Its observed rows include the metered slack rows
        (the slack is a measurement like any other), so no decode-specific slack correction
        is applied.
    split:
        ``"val"`` (default), ``"train"``, ``"test"`` (the held-out split; empty unless the
        run used ``DataConfig.test_fraction > 0`` — an empty split raises), ``"all"`` (train + val concatenated —
        the held-out test split is NOT included, so a report over ``"all"`` never touches
        it), or ``"full"`` (train + val + test concatenated — EVERY surviving scenario in
        ``source_or_path``, regardless of the checkpoint's own split fractions).

        ``"full"`` is for evaluating an ENTIRELY EXTERNAL dataset the checkpoint never
        trained or validated on (e.g. one generated by an independent reference engine) —
        a PATH-based ``source_or_path`` builds its train/val/test partition from the
        checkpoint's OWN ``val_fraction``/``test_fraction``/``split_seed`` (see the
        ``source_or_path`` note above), which is meaningless on a dataset the checkpoint
        never split in the first place; ``"all"`` would then silently EXCLUDE whatever
        fraction that inherited config carves out as "test", even though nothing here was
        ever actually held out. ``"full"`` sidesteps that by including every non-dropped
        (converged) scenario:

        .. code-block:: python

            from pgml.evaluation.oracles import write_opendss_dataset

            sampled = scenario_config.sample(grid)
            dataset_dir = write_opendss_dataset(
                grid, sampled, "opendss_testset", harmonic_orders=[1, 5, 7]
            )
            module = load_se_module(checkpoint_path, grid=grid, dataset_path=str(dataset_dir))
            pred = predict_dataset(
                module, source_or_path=str(dataset_dir), grid=grid,
                measurement=measurement, split="full",
            )
            metrics = prediction_metrics(pred)
    max_samples:
        Cap the number of SCENARIOS evaluated (the tail is dropped) — handy for a quick look
        at a large dataset. ``None`` evaluates the whole split.
    """
    from pgml.assembly import node_phase_index
    from torch.utils.data import ConcatDataset, DataLoader

    from injection_hse.data import MeasurementModel, build_data_source
    from injection_hse.masking import build_masker
    from injection_hse.data import StateCollate

    dev = torch.device(device)
    module.to(dev)
    module.eval()

    if isinstance(source_or_path, (str, Path)):
        # Derive the DataConfig from the CHECKPOINT's own data settings (carried on
        # ``module.cfg``) so the train/val partition matches what training used —
        # ``val_fraction``/``split_seed`` and the other split fields must carry over. A bare
        # ``DataConfig(...)`` would silently fall back to the defaults (val_fraction=0.1,
        # split_seed=0), letting "val" rows overlap the training set for any run that used
        # different values.
        data_cfg = module.cfg.data.model_copy(
            update={"source": "parquet", "dataset_path": str(source_or_path)}
        )
        source = build_data_source(data_cfg, grid=grid)
    else:
        source = source_or_path

    # Install the placement masker so the evaluation reflects the real sensor scarcity.
    # ``observed_rows`` already contains the metered slack rows — the slack is a
    # measurement like any other (matching training, where the slack rows are forced
    # observed in every mask), so no decode-specific correction is needed. The sensor set is
    # resolved through the SAME rule training used, so a grid carrying installed devices is
    # scored on THEIR current channels rather than on every incident branch.
    mm = MeasurementModel.resolve(
        grid,
        measurement.voltage_nodes,
        fundamental_nodes=measurement.fundamental_nodes,
        slack_known=measurement.slack_known,
    )
    observed_rows = mm.observed_rows(grid)
    module.set_masker(
        build_masker(
            "placement",
            observed_rows=observed_rows,
            order_capability=(
                mm.order_capability(grid, list(module.orders))
                if measurement.heterogeneous_orders
                else None
            ),
            observed_edge_rows=(
                mm.observed_edges(grid) if measurement.measure_current else None
            ),
        )
    )

    if split == "train":
        ds = source.train_dataset()
    elif split == "val":
        ds = source.val_dataset()
    elif split == "test":
        ds = source.test_dataset()
    elif split == "all":
        ds = ConcatDataset([source.train_dataset(), source.val_dataset()])
    elif split == "full":
        ds = ConcatDataset(
            [source.train_dataset(), source.val_dataset(), source.test_dataset()]
        )
    else:
        raise ValueError(
            f"split must be 'train'|'val'|'test'|'all'|'full', got {split!r}."
        )

    loader = DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=StateCollate()
    )

    preds: list[np.ndarray] = []
    targs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    scenarios: list[np.ndarray] = []
    seen = 0
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = module.infer(batch)
        pred_v = out["pred_v"]  # [*lead, H, N]
        target_v = out["target_v"]
        mask = out["mask"]  # [*lead, N]
        n = pred_v.shape[-1]
        h = pred_v.shape[-2]
        # fold any leading time axis into the sample axis
        pred_v = pred_v.reshape(-1, h, n).detach().cpu().numpy()
        target_v = target_v.reshape(-1, h, n).detach().cpu().numpy()
        mask = mask.reshape(-1, n).detach().cpu().numpy()
        preds.append(pred_v)
        targs.append(target_v)
        masks.append(mask)
        index = batch["index"].detach().cpu().numpy().astype(np.int64)
        scenarios.append(np.repeat(index, pred_v.shape[0] // max(index.size, 1)))
        seen += int(pred_v.shape[0])
        if max_samples is not None and seen >= max_samples:
            break

    if not preds:
        raise ValueError(
            f"the {split!r} split is empty — nothing to predict. A 'test' split needs the run "
            "to have set DataConfig.test_fraction > 0."
        )
    pred_v = np.concatenate(preds, axis=0)
    target_v = np.concatenate(targs, axis=0)
    mask = np.concatenate(masks, axis=0).astype(bool)
    scenario_index = np.concatenate(scenarios, axis=0)
    if max_samples is not None:
        pred_v = pred_v[:max_samples]
        target_v = target_v[:max_samples]
        mask = mask[:max_samples]
        scenario_index = scenario_index[:max_samples]

    idx = node_phase_index(grid)
    base_v = _base_v_numpy(grid)
    return Prediction(
        target_v=target_v,
        pred_v=pred_v,
        mask=mask,
        orders=list(module.orders),
        base_v=base_v,
        node_ids=idx.node_ids.cpu().numpy().astype(np.int64),
        phase_codes=idx.phase_codes.cpu().numpy().astype(np.int64),
        observed_rows=observed_rows.cpu().numpy().astype(np.int64),
        sensor_names=list(measurement.voltage_nodes),
        grid=grid,
        scenario_index=scenario_index,
    )


def _base_v_numpy(grid: "Grid") -> np.ndarray:
    """Per-row L-N base voltage as a float64 numpy array (the per-unit denominator)."""
    from injection_hse.data import base_voltage_per_row

    return base_voltage_per_row(grid).detach().cpu().numpy().astype(np.float64)


def prediction_metrics(prediction: Prediction) -> dict:
    """The full :func:`injection_hse.metrics.state_errors` dict for a :class:`Prediction`.

    Runs the shared metric (per-unit MAE/RMSE, per-harmonic RELATIVE error, and voltage-THD
    error) on the prediction's estimate/target/mask under its per-row base voltage. The single
    scoring seam every evaluation report and table reads, so pu-MAE, the per-harmonic relative
    error, and the THD error are reported consistently.
    """
    from injection_hse.metrics import state_errors

    return state_errors(
        torch.as_tensor(prediction.pred_v),
        torch.as_tensor(prediction.target_v),
        mask=torch.as_tensor(prediction.mask),
        orders=prediction.orders,
        base_v=torch.as_tensor(prediction.base_v),
    )


def prediction_metrics_by_group(prediction: Prediction, groups) -> dict:
    """:func:`prediction_metrics` per group of scenarios -> ``{group: metrics dict}``.

    ``groups`` is an integer label per dataset SCENARIO (indexed by scenario id, e.g. a
    dataset's per-scenario sample record such as the grid copy a scenario was solved on);
    every sample is scored in the group of its scenario
    (:attr:`Prediction.scenario_index`). Only groups present in the prediction appear, in
    ascending order. Raises ``ValueError`` for a prediction without scenario ids.
    """
    from dataclasses import replace

    if prediction.scenario_index is None:
        raise ValueError(
            "prediction_metrics_by_group needs Prediction.scenario_index "
            "(predict_dataset records it)."
        )
    labels = np.asarray(groups)[prediction.scenario_index]
    out: dict = {}
    for g in np.unique(labels):
        sel = labels == g
        out[g.item()] = prediction_metrics(
            replace(
                prediction,
                target_v=prediction.target_v[sel],
                pred_v=prediction.pred_v[sel],
                mask=prediction.mask[sel],
                scenario_index=prediction.scenario_index[sel],
            )
        )
    return out


__all__ = [
    "Prediction",
    "load_se_module",
    "predict_dataset",
    "prediction_metrics",
    "prediction_metrics_by_group",
]
