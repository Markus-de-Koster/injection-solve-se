"""What a training run publishes about ITS OWN loss and where its error sits.

The reported ``train_loss``/``val_loss`` is a single number covering a blend of supervised
terms over harmonic orders whose magnitudes differ by decades — it can fall while the state
estimate stands still, and its per-unit metrics can look flat while one band regresses. These
tests pin the diagnostics that make it readable: the supervised loss is always published split
into its parts, and every solved order reports its own error next to the aggregate.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def se_module(cigre_grid, se_dataset):
    """A voltage-decode :class:`~injection_hse.train.SEModule` plus one collated batch."""
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        DNNConfig,
        ExperimentConfig,
        MaskStageConfig,
    )
    from injection_hse.data import StateCollate, base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.masking import build_masker
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.train import SEModule

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(se_dataset)), grid=cigre_grid
    )
    enc = build_encoding("trig")
    train_v = torch.stack([source.train_dataset()[i]["v"] for i in range(32)])
    normalizer = HarmonicNormalizer().fit(enc.encode(train_v).transpose(-3, -2))
    cfg = ExperimentConfig(
        name="loss-tracking",
        seed=0,
        orders=list(source.orders),
        data=DataConfig(dataset_path="x"),
        model=DNNConfig(hidden_dims=[32]),
        # a partially observed stage: the unobserved metrics are the ones under test, and a
        # fully observed stage leaves them vacuously zero
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-12,
                    max_epochs=1,
                )
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=normalizer,
        graph=source.graph(),
        base_v=base_voltage_per_row(cigre_grid),
    )
    # the stage masker is installed by the StageManager at fit time; do it here so the
    # unobserved metrics under test are non-vacuous
    module.set_masker(build_masker("random", observed_fraction=0.5))
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    return module, batch, list(source.orders)


def _logged(module, batch) -> dict[str, float]:
    """``{metric: value}`` of one validation step, captured from the module's ``log`` calls."""
    captured: dict[str, float] = {}

    def _capture(name, value, **_kwargs):
        captured[name] = float(value.detach() if torch.is_tensor(value) else value)

    module.log = _capture  # type: ignore[method-assign]
    module.validation_step(batch, 0)
    return captured


def test_validation_reports_every_solved_order(se_module):
    """One per-unit and one relative-error column per order, plus the two band aggregates.

    The aggregate ``val_pu_mae_unobs`` averages the orders, so a regression confined to one
    band is invisible in it; the per-order columns are what make the band identifiable, and
    ``*_fund``/``*_harm`` are the compact pair the live progress view reads.
    """
    module, batch, orders = se_module
    logged = _logged(module, batch)

    for order in orders:
        assert logged[f"val_pu_mae_h{order}"] > 0.0
        assert logged[f"val_pu_mae_unobs_h{order}"] > 0.0
        assert logged[f"val_rel_unobs_h{order}"] > 0.0
    assert logged["val_pu_mae_unobs_fund"] == pytest.approx(
        logged["val_pu_mae_unobs_h1"]
    )
    harmonics = [logged[f"val_pu_mae_unobs_h{h}"] for h in orders if h != 1]
    assert logged["val_pu_mae_unobs_harm"] == pytest.approx(
        sum(harmonics) / len(harmonics)
    )
    # the aggregate is the mean over ALL orders — the fundamental included
    every = harmonics + [logged["val_pu_mae_unobs_h1"]]
    assert logged["val_pu_mae_unobs"] == pytest.approx(
        sum(every) / len(every), rel=1e-5
    )


def test_the_voltage_term_is_published_for_every_run(se_module):
    """Without a blend the reported loss IS the voltage term — and still says so.

    The progress view and the training history read the component, so a run that never
    enables :attr:`~injection_hse.config.LossConfig.injection_state_loss_split` must publish it too;
    otherwise its curve is missing exactly where a blended run's would be comparable.
    """
    module, batch, _ = se_module
    loss = module._step(batch, train=False)[0]
    assert set(module._loss_components) == {"voltage"}
    assert torch.equal(module._loss_components["voltage"], loss.detach())
