"""Exact overlap of solved states between the splits of injection_hse task datasets.

Every state (one ``[H, N]`` complex128 array of a snapshot, or of one step of a sequence) of
the evaluated split is hashed by its bytes and looked up among the states of every split of
the reference datasets, so an identical state anywhere is found, whatever its position.
Also reports whether the evaluated dataset is an index-wise prefix of a reference dataset of
the same kind (item ``i`` equal to reference item ``i`` for every item) and, for items that are
not identical, the smallest relative difference to the reference item at the same index.

``--nearest`` adds, for every state of the evaluated split, the relative distance to its
nearest state in each reference split and to the reference state at the same index and step:
``||v - v'|| / ||v'||`` over all orders and rows in per unit of each row's base (Euclidean),
with the same statistics of the evaluated split against the other splits of its own dataset
(states drawn alike, never shared) as the scale of an ordinary neighbour.

Each dataset is read through the data source of an experiment configuration (its
``data`` section; split seed and fractions), so the splits are those the networks used.

Usage::

    python overlap.py <out.json> --eval ROOT:SUBDIR:CONFIG:SPLIT|all
        --ref ROOT:SUBDIR:CONFIG [--ref ...] [--nearest]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapter as P  # noqa: E402

SPLITS = ("train", "val", "test")


def split_items(root: Path, subdir: str, config: Path, split: str):
    """``(dataset indices [n], states complex128 [n, T, H, N])`` of one split."""
    _, _, source = P.load_source(root, config, subdir)
    ds = getattr(source, f"{split}_dataset")()
    idx, states = [], []
    for i in range(len(ds)):
        item = ds[i]
        v = item["v"].numpy().astype(np.complex128)
        states.append(v if v.ndim == 3 else v[None])
        idx.append(int(item["index"]))
    return np.array(idx), np.stack(states)


def digests(states: np.ndarray) -> list:
    """sha256 of every state ``[H, N]`` of ``[n, T, H, N]``, flattened over items and steps."""
    flat = np.ascontiguousarray(states.reshape(-1, *states.shape[-2:]))
    return [hashlib.sha256(s.tobytes()).hexdigest() for s in flat]


def per_unit_vectors(states: np.ndarray, base: np.ndarray):
    """``[n, T, H, N]`` complex -> float64 ``[n·T, 2·H·N]`` real vectors in per unit (float64:
    the distances come from the inner-product form, which float32 resolves to only ~1e-3)."""
    import torch

    z = states.reshape(-1, *states.shape[-2:]) / base[None, None, :]
    return torch.as_tensor(np.concatenate([z.real, z.imag], axis=-1).reshape(z.shape[0], -1),
                           dtype=torch.float64)


def nearest(e, r, chunk: int = 256) -> np.ndarray:
    """float ``[n_e]`` relative distance of each row of ``e`` to its nearest row of ``r``."""
    import torch

    out = []
    for a in range(0, e.shape[0], chunk):
        d = torch.cdist(e[a: a + chunk], r)
        j = d.argmin(1)
        out.append((d[torch.arange(d.shape[0]), j] / r[j].norm(dim=1)).numpy())
    return np.concatenate(out)


def quantiles(x: np.ndarray) -> dict:
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in (0.0, 0.01, 0.5, 0.99)}


def nearest_entries(evals: dict, r_idx: np.ndarray, r_states: np.ndarray, base: np.ndarray) -> dict:
    """``{eval split: {"nearest": ..., "same_index": ...}}`` against one reference split."""
    rv = per_unit_vectors(r_states, base)
    out = {}
    for e_split, (e_idx, e_states, _) in evals.items():
        ev = per_unit_vectors(e_states, base)
        entry = {"nearest": quantiles(nearest(ev, rv))}
        if r_states.shape[1:] == e_states.shape[1:]:
            common = np.intersect1d(e_idx, r_idx)
            if common.size:
                pos_e = {int(k): j for j, k in enumerate(e_idx)}
                pos_r = {int(k): j for j, k in enumerate(r_idx)}
                a = per_unit_vectors(np.stack([e_states[pos_e[int(k)]] for k in common]), base)
                b = per_unit_vectors(np.stack([r_states[pos_r[int(k)]] for k in common]), base)
                entry["same_index"] = quantiles(((a - b).norm(dim=1) / b.norm(dim=1)).numpy())
        out[e_split] = entry
    return out


def row_base(root: Path) -> np.ndarray:
    """float ``[N]`` line-to-neutral base voltage of every row of the dataset's grid."""
    from pgml.schemas.grid_schema import Grid

    from injection_hse.data import base_voltage_per_row

    return base_voltage_per_row(Grid.model_validate_json((root / "grid.json").read_text())).numpy()


def run(args) -> None:
    root, sub, cfg, which = args.eval.split(":")
    eval_splits = SPLITS if which == "all" else (which,)
    evals = {sp: split_items(Path(root), sub, Path(cfg), sp) for sp in eval_splits}
    evals = {sp: (i, s, digests(s)) for sp, (i, s) in evals.items() if i.size}
    report = dict(eval=dict(root=root, subdir=sub, splits={
        sp: dict(items=int(i.size), states=len(h), unique_states=len(set(h)))
        for sp, (i, s, h) in evals.items()}), refs=[])
    near = {sp: {} for sp in evals}
    if args.nearest:
        base = row_base(Path(root))

        def add_near(label, r_idx, r_states):
            for e_split, entry in nearest_entries(evals, r_idx, r_states, base).items():
                near[e_split][label] = entry
                print(json.dumps({"eval": e_split, "ref": label, **entry}), flush=True)

        for sp in SPLITS:
            if sp not in evals:
                add_near(f"own {sp}", *split_items(Path(root), sub, Path(cfg), sp))
    for spec in args.ref:
        r_root, r_sub, r_cfg = spec.split(":")
        rec = dict(root=r_root, subdir=r_sub, splits={})
        for r_split in SPLITS:
            r_idx, r_states = split_items(Path(r_root), r_sub, Path(r_cfg), r_split)
            if not r_idx.size:
                continue
            r_hash = set(digests(r_states))
            entry = dict(items=int(r_idx.size), states=int(r_states.shape[0] * r_states.shape[1]))
            if args.nearest:
                add_near(f"{r_sub} {r_split}", r_idx, r_states)
            for e_split, (e_idx, e_states, e_hash) in evals.items():
                found = dict(eval_states_found=int(sum(h in r_hash for h in e_hash)))
                if r_states.shape[1:] == e_states.shape[1:]:
                    common = np.intersect1d(e_idx, r_idx)
                    if common.size:
                        pos_e = {int(k): j for j, k in enumerate(e_idx)}
                        pos_r = {int(k): j for j, k in enumerate(r_idx)}
                        a = np.stack([e_states[pos_e[int(k)]] for k in common])
                        b = np.stack([r_states[pos_r[int(k)]] for k in common])
                        axes = tuple(range(1, a.ndim))
                        same = np.all(a == b, axis=axes)
                        rel = np.abs(a - b).max(axis=axes) / np.abs(b).max(axis=axes)
                        found.update(same_index_items=int(common.size),
                                     same_index_identical=int(same.sum()),
                                     same_index_min_rel_diff_of_the_others=(
                                         float(rel[~same].min()) if (~same).any() else None))
                entry[f"eval_{e_split}"] = found
            del r_states, r_hash
            rec["splits"][r_split] = entry
            print(json.dumps({"ref": r_sub, "split": r_split, **entry}), flush=True)
        report["refs"].append(rec)
    if args.nearest:
        report["nearest"] = near
    Path(args.out).write_text(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--eval", required=True, help="ROOT:SUBDIR:CONFIG:SPLIT")
    ap.add_argument("--ref", action="append", required=True, help="ROOT:SUBDIR:CONFIG")
    ap.add_argument("--nearest", action="store_true",
                    help="also the nearest-neighbour distances of the evaluated states")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
