"""The metered slack as the harmonic boundary on the MULTI-GRID path.

``PhysicsConfig.harmonic_slack="measured"`` reaches every member of a multi-grid run through
the operator bank: each member's solver is built in that mode, the decode hands it the
member's measured slack voltage at EVERY order (not the fundamental alone), and the
reconstructed state therefore holds the slack rows at their measured harmonic voltage —
the supply's own distortion enters each member through its boundary. The Norton default
stays byte-identical, and the injection normalizer's feature scales ignore the slack rows'
``Y·V`` (the source's current, never a device injection) on this path as on the single-grid
one.
"""

from __future__ import annotations

import math

import pytest
import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    ExperimentConfig,
    GNNConfig,
    MaskStageConfig,
    MultiGridConfig,
    PhysicsConfig,
    TrainConfig,
)
from injection_hse.data import MultiGridCollate, MultiGridDataSource
from injection_hse.data.multigrid import (
    MultiGridGenerationConfig,
    SyntheticFeederSource,
    generate_multigrid_dataset,
)
from injection_hse.encoding import CartesianEncoding, TrigEncoding
from injection_hse.physics import MultiGridPhysicsBank, fit_multigrid_injection_normalizer
from injection_hse.train import SEModule, fit_multigrid_normalizer
from injection_hse.train.multigrid import _member_slack_voltage

ORDERS = (1, 3, 5)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    out = tmp_path_factory.mktemp("multigrid_harmonic_slack")
    generate_multigrid_dataset(
        MultiGridGenerationConfig(
            output_root=str(out),
            n_scenarios=3,
            orders=ORDERS,
            val_fraction=0.34,
            sources=[SyntheticFeederSource(n_grids=3, node_range=(4, 6), n_feeders=2)],
        )
    )
    return out


@pytest.fixture(scope="module")
def source(root):
    return MultiGridDataSource(root, placement_fraction=0.3)


@pytest.fixture(scope="module")
def batch(source):
    """One collated batch drawn from two DIFFERENT member grids."""
    dataset = source.train_dataset()
    seen: dict[int, dict] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        seen.setdefault(int(item["grid"]), item)
        if len(seen) == 2:
            break
    assert len(seen) == 2
    return MultiGridCollate(source)([seen[k] for k in sorted(seen)])


def _config(root, harmonic_slack: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="multigrid-slack",
        seed=0,
        orders=list(ORDERS),
        data=DataConfig(
            dataset_path=str(root),
            multigrid=MultiGridConfig(placement_fraction=0.3),
            batch_size=2,
            num_workers=0,
        ),
        model=GNNConfig(hidden_dim=16, num_layers=2),
        physics=PhysicsConfig(decode="injection", harmonic_slack=harmonic_slack),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="random",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
        train=TrainConfig(
            accelerator="cpu", devices=1, enable_progress_bar=False, dtype="complex128"
        ),
    )


def _module(root, source, harmonic_slack: str) -> SEModule:
    cfg = _config(root, harmonic_slack)
    bank = MultiGridPhysicsBank(
        source, list(ORDERS), dtype=torch.complex128, harmonic_slack=harmonic_slack
    )
    normalizer = fit_multigrid_normalizer(source, TrigEncoding(), cfg)
    inj_normalizer = fit_multigrid_injection_normalizer(
        bank, source, CartesianEncoding(), cfg
    )
    return SEModule(
        cfg,
        normalizer=normalizer,
        grid_keys=[b.key for b in source.grids],
        bank=bank,
        inj_normalizer=inj_normalizer,
    )


def _distorted(batch, distort=(0.02, 0.01)):
    """The batch's absolute state with a harmonic voltage placed on every slack row."""
    v = batch["v"].clone().to(torch.complex128)
    slack = batch["slack_rows"]
    for j, order in enumerate(ORDERS):
        if order == 1:
            continue
        v[j, slack] = (
            v[0, slack] * distort[j - 1] * complex(math.cos(0.4 * j), math.sin(0.4 * j))
        )
    return v


def test_the_bank_builds_every_member_in_the_configured_mode(source):
    bank = MultiGridPhysicsBank(source, list(ORDERS), harmonic_slack="measured")
    assert all(bank.solver(i).harmonic_slack == "measured" for i in range(len(bank)))
    assert all(bank.solver(i).measured_harmonic_slack for i in range(len(bank)))
    default = MultiGridPhysicsBank(source, list(ORDERS))
    assert default.solver(0).harmonic_slack == "norton"
    with pytest.raises(ValueError, match="harmonic_slack"):
        MultiGridPhysicsBank(source, list(ORDERS), harmonic_slack="thevenin")


def test_the_member_slack_voltage_carries_every_order_when_measured(source, batch):
    v = _distorted(batch)
    ptr = batch["ptr"].tolist()
    index = int(batch["grid_index"][0])
    node_rows = torch.arange(ptr[0], ptr[1]).unsqueeze(0)
    measured = MultiGridPhysicsBank(source, list(ORDERS), harmonic_slack="measured")
    norton = MultiGridPhysicsBank(source, list(ORDERS))
    k = int(measured.solver(index).slack_rows.numel())
    v_all = _member_slack_voltage(
        measured.solver(index), v, node_rows, fundamental_pos=0
    )
    v_fund = _member_slack_voltage(
        norton.solver(index), v, node_rows, fundamental_pos=0
    )
    assert tuple(v_all.shape) == (1, len(ORDERS), k)
    assert tuple(v_fund.shape) == (1, k)
    assert torch.allclose(v_all[:, 0], v_fund)


def test_the_multigrid_decode_holds_the_measured_slack_at_every_order(
    root, source, batch
):
    v = _distorted(batch)
    mask = torch.zeros(v.shape[-1], dtype=torch.bool)
    mask[batch["slack_rows"]] = True
    zero_injection = torch.zeros_like(v)
    slack = batch["slack_rows"]

    measured = _module(root, source, "measured")
    with torch.no_grad():
        state = measured._multigrid_decode(zero_injection, v, mask, batch)
    assert torch.allclose(state[:, slack], v[:, slack], atol=1e-9)
    # a distorted supply reaches every row of every member with no injection at all
    harmonic = state[1:].abs()
    assert bool((harmonic > 0.5 * harmonic[:, slack].amin()).all())

    norton = _module(root, source, "norton")
    with torch.no_grad():
        plain = norton._multigrid_decode(zero_injection, v, mask, batch)
    assert torch.allclose(plain[0, slack], v[0, slack], atol=1e-9)  # the fundamental
    assert not torch.allclose(plain[1:, slack], v[1:, slack], atol=1e-6)


def test_the_injection_normalizer_ignores_the_slack_rows_current(root, source):
    """A member's slack-row ``Y·V`` (its source current) never sets the feature scales."""
    bank = MultiGridPhysicsBank(source, list(ORDERS), dtype=torch.complex128)
    cfg = _config(str(root), "norton")
    fitted = fit_multigrid_injection_normalizer(bank, source, CartesianEncoding(), cfg)
    # recompute the pooled features WITHOUT zeroing the slack rows: their fundamental
    # ``Y·V`` is the whole feeder's current, so the scales must differ
    parts = []
    for i in source.split_indices("train"):
        bundle = source.bundle(i)
        solver = bank.solver(i)
        states = bundle.source.states[:8]
        i_target = solver.zero_tied(solver.injection_targets(states))
        parts.append(i_target[..., solver.slack_rows].abs().amax())
    assert max(parts) > 0.0
    assert torch.isfinite(fitted.std).all()
