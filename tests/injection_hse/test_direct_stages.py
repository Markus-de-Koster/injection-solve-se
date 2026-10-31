"""``--no-curriculum`` stages: one stage per pass, at the placement or on random sensor sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "run" / "examples" / "injection_hse"))


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        epochs=7,
        target_full=1e-6,
        target_placement=1e-6,
        placement_lr=None,
        placement_scheduler=None,
        anneal_epochs=3,
        direct_random_fraction=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_direct_stages_collapse_a_pass_to_one_placement_stage():
    import train_se_full_workflow as wf

    args = _args()
    ladder = wf._snapshot_stages(args)
    (stage,) = wf._direct_stages(ladder, args)
    assert stage.name == "placement" and stage.kind == "placement"
    assert stage.max_epochs == sum(s.max_epochs for s in ladder)
    assert wf._direct_random(wf._anneal_stage(args), args).kind == "placement"


def test_direct_random_fraction_trains_on_random_sensor_sets():
    import train_se_full_workflow as wf

    args = _args(direct_random_fraction=0.1)
    ladder = wf._sequence_stages(args)
    (stage,) = wf._direct_stages(ladder, args)
    # anchor-schedule keys and checkpoint names unchanged
    assert stage.name == "placement"
    assert stage.kind == "random" and stage.observed_fraction == 0.1
    assert stage.max_epochs == sum(s.max_epochs for s in ladder)
    anneal = wf._direct_random(wf._anneal_stage(args), args)
    assert anneal.kind == "random" and anneal.observed_fraction == 0.1
