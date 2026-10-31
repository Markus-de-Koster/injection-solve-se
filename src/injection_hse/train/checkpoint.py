"""What a checkpoint carries, and how it becomes a live module again.

A stage checkpoint holds the trained weights, the fitted normalizer(s), the config that
trained them, the anchoring weights active when it was written, and the identity of the row
layout the weights index against. Turning one back into a module is ONE decision, which is
why both entry points land here: :meth:`injection_hse.train.SEModule.from_checkpoint` for a single-grid
run (where one grid rebuilds the input) and :func:`load_multigrid_checkpoint` for a run
trained across many grids (where the corpus does). ``from_checkpoint`` dispatches to the
latter, and :func:`injection_hse.analysis.multigrid.load_multigrid_module` reaches it through a run
directory.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from pgml.schemas.grid_schema import Grid

    from ..config import ExperimentConfig
    from ..data.multigraph import MultiGridDataSource
    from .module import SEModule


def _check_grid_fingerprint(checkpoint: dict, grid: Grid) -> None:
    """Refuse a grid whose row-layout fingerprint disagrees with the checkpoint's.

    The checkpoint stores ``grid_fingerprint`` — a stable hash of the node-phase row-layout
    contract the weights index against (``pgml.topology.layout_fingerprint``). Reloading against
    a grid with the same shape but a different layout (relabeled / reordered nodes, changed
    slack or base frequency) would silently mislabel every ``[..., N]`` state row, so a mismatch
    raises. A checkpoint saved before the fingerprint existed carries no key: it loads with a
    warning (its provenance is unknown — the caller vouches for the grid).
    """
    from pgml.topology import layout_fingerprint

    stored = checkpoint.get("grid_fingerprint")
    if stored is None:
        warnings.warn(
            "checkpoint carries no grid fingerprint (saved before grid identity was "
            "recorded); loading against the supplied grid without verifying the row layout "
            "— confirm it is the grid the model was trained on.",
            stacklevel=3,
        )
        return
    current = layout_fingerprint(grid)
    if current == stored:
        return
    ckpt_ids = checkpoint.get("grid_node_ids")
    from pgml.assembly import node_phase_index

    idx = node_phase_index(grid)
    grid_ids = idx.node_ids.cpu().tolist()
    detail = ""
    if ckpt_ids is not None:
        if len(ckpt_ids) != len(grid_ids):
            detail = f" (checkpoint N={len(ckpt_ids)}, grid N={len(grid_ids)})"
        else:
            diff = next(
                (i for i, (a, b) in enumerate(zip(ckpt_ids, grid_ids)) if a != b),
                None,
            )
            if diff is not None:
                detail = (
                    f" (first differing row {diff}: checkpoint node {ckpt_ids[diff]}, "
                    f"grid node {grid_ids[diff]})"
                )
    raise ValueError(
        "grid does not match the checkpoint's trained row layout: fingerprint "
        f"{current} != {stored}{detail}. Pass the exact grid the model was trained on "
        "(node order, phases, slack, and base frequency must match)."
    )


def _checkpoint_normalizer(state: dict, key: str, *, required: bool):
    """Restore an auxiliary normalizer from the checkpoint (``None`` when not needed)."""
    from ..normalization import HarmonicNormalizer

    if key not in state:
        if required:
            raise ValueError(
                f"the checkpoint carries no {key!r}, but the run's config needs it; "
                "the weights and the config disagree."
            )
        return None
    normalizer = HarmonicNormalizer()
    normalizer.load_state_dict(state[key])
    return normalizer


def load_multigrid_checkpoint(
    checkpoint,
    source: MultiGridDataSource,
    *,
    config: Optional[ExperimentConfig] = None,
    device="cpu",
) -> SEModule:
    """Rebuild a MULTI-GRID module from a checkpoint + the corpus it trained on.

    The multi-grid arm of :meth:`injection_hse.train.SEModule.from_checkpoint`, which delegates here:
    one grid cannot rebuild a many-grid input (every batch carries its own concatenated
    graph), so the member layout comes from ``source`` and the physics from a
    :class:`~injection_hse.physics.MultiGridPhysicsBank` built over it.

    The checkpoint's own normalizers are restored rather than refit: they are exact (the loss
    resolves its per-harmonic weights from them at construction) and cost nothing, whereas a
    refit would re-read every train member. The measurement-anchoring weights ACTIVE when the
    checkpoint was written are deployed — as on the single-grid path — so the reloaded decode
    is the one that trained; the operator bank is built for whichever is larger, those weights
    or the largest any curriculum stage requests.

    Parameters
    ----------
    checkpoint:
        A checkpoint path, or an already-loaded checkpoint dict.
    source:
        The :class:`~injection_hse.data.MultiGridDataSource` over the corpus. Its member keys name the
        per-grid metrics, and the operator bank mirrors its members.
    config:
        Config override, for a caller holding a truer copy than the checkpoint's — a run
        directory's ``config.yaml`` with ``data.dataset_path`` repointed at the corpus that is
        actually present. Default: the config the checkpoint carries.
    device:
        Where to place the module.

    Returns
    -------
    SEModule
        In ``eval`` mode, with the checkpoint's weights and normalizers loaded. The masking
        policy is the module default (fully observed) — a caller scoring a specific curriculum
        stage installs that stage's masker itself.
    """
    from ..config import ExperimentConfig
    from ..normalization import HarmonicNormalizer
    from .anchoring import anchor_build_weights
    from .dtypes import training_dtypes
    from .module import SEModule

    state = (
        checkpoint
        if isinstance(checkpoint, dict)
        else torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    cfg = (
        config
        if config is not None
        else ExperimentConfig.model_validate(state["pgl_config"])
    )
    if cfg.data.multigrid is None:
        raise ValueError(
            "load_multigrid_checkpoint needs a data.multigrid run; rebuild a single-grid "
            "checkpoint with SEModule.from_checkpoint(path, grid=...)."
        )

    normalizer = HarmonicNormalizer()
    normalizer.load_state_dict(state["normalizer"])
    # the weights the checkpoint was written with (a checkpoint from before they were
    # recorded falls back to the config's global ones)
    v_weight, i_weight = (
        float(w)
        for w in state.get(
            "anchor_weights",
            [cfg.physics.anchor_voltage_weight, cfg.physics.anchor_current_weight],
        )
    )

    bank = None
    injection_orders = None
    if (
        cfg.physics.needs_solver
        or cfg.loss.predict_injection
        or cfg.measurement.measure_current
    ):
        from ..physics import MultiGridPhysicsBank

        _, complex_dtype = training_dtypes(cfg.train.dtype)
        build_v, build_i = anchor_build_weights(
            cfg.physics,
            [cfg.curriculum],
            measure_current=cfg.measurement.measure_current,
        )
        bank = MultiGridPhysicsBank(
            source,
            list(cfg.orders),
            dtype=complex_dtype,
            anchor_v_weight=max(build_v, v_weight),
            anchor_i_weight=max(build_i, i_weight),
            harmonic_slack=cfg.physics.harmonic_slack,
            load_shunt=cfg.physics.load_shunt,
        )
        if cfg.loss.predict_injection:
            # the device axes are emitted only for a run that consumes them; attaching also
            # validates that the members share one injected-order axis.
            source.attach_physics_bank(bank)
            injection_orders = source.injection_orders

    module = SEModule(
        cfg,
        normalizer=normalizer,
        grid_keys=[bundle.key for bundle in source.grids],
        bank=bank,
        inj_normalizer=_checkpoint_normalizer(
            state, "inj_normalizer", required=cfg.physics.needs_solver
        ),
        edge_normalizer=_checkpoint_normalizer(
            state, "edge_normalizer", required=cfg.measurement.measure_current
        ),
        power_normalizer=_checkpoint_normalizer(
            state, "power_normalizer", required=cfg.measurement.measure_power
        ),
        injection_orders=injection_orders,
    )
    module.load_state_dict(state["state_dict"])
    module.set_stage_anchor(v_weight=v_weight, i_weight=i_weight)
    module.to(torch.device(device))
    module.eval()
    return module
