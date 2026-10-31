"""A state-estimation task dataset as the input and target tensors of the Mack et al. estimators.

Reads the sequence datasets of ``generate_se_task_datasets.py`` (``<root>/grid.json``,
``<root>/dataset_coherent``), on which the state-estimation networks of ``injection_hse``
are evaluated, and writes them in the layout of the published estimators: one sample per
state and phase, every phase in the frame of phase a, volts referred to the slack level,
the meters of the placement plus the slack bus as inputs, all 44 buses as targets, so
:mod:`train` runs on it unchanged.

What is specific to this data:

- **Splits.** The train / validation / test split of the injection_hse data source the networks were
  evaluated with: ``ExperimentConfig.data`` of an experiment configuration (``--config``:
  its ``split_seed``, ``val_fraction``, ``test_fraction``), with the dataset path replaced.
  Sequences of ``T`` states; every state is a sample, in the split's order.
- **Training data.** By default the train / validation splits of the same dataset. With
  ``--train-root`` the train / validation splits of another generated root instead (its
  ``dataset_random`` snapshots with ``--train-config-random`` and its ``dataset_coherent``
  sequences with ``--train-config-coherent``, each split by its own configuration), e.g.
  the data a network was trained on; the test split stays that of ``DATASET``.
- **Currents.** ``--currents upstream|incident`` also writes the CT readings of the
  placement (:class:`CurrentReadings`): ``c_<variant>`` ``[Q, 3, K, H]``, per phase the ``K``
  read half-edges whose source row is that phase (in amperes, the frame of phase a);
  ``meta.json`` lists them (``ct_channels``). Their error is the configuration's current
  model, seeded with the voltage stream's seed plus :data:`CURRENT_SEED_OFFSET` (the
  evaluation's 0 / 7).
- **Readings.** The instrument model of the configuration (``noise``:
  ``InstrumentNoise.from_config``), applied as the network module applies it: to the states
  cast to the training dtype (``train.dtype``), batch by batch over ``[16, T, H, N]``
  sequence batches in the split's order, the held part of the error per sequence. The test
  readings take the evaluation seed (``--test-seed``, default 0, the seed the evaluation
  re-seeds the voltage stream with), so the test inputs are the networks' evaluation inputs.
  Training and validation readings take their own seeds; the training split is read
  ``--train-draws`` times with independent draws (each a full copy of the training states).
- **Scoring rows.** The rows the placement does not read (every row outside the meters and
  the slack, MV and tied rows included), as the evaluation scores them; ``scoring.npz`` also
  holds the mean true magnitude per order over those rows and the test states, the
  denominator of the relative error. The evaluation's metric is the plain phasor error
  (:mod:`scoring` metric ``full``); no slack response is stored (``g_s`` = 0), so the
  ``slack_ref`` column equals ``full`` on these arrays.
- **Perturbed test grids.** ``--test-subdir dataset_coherent_heldout`` scores the test split
  of the held-out perturbed version of the dataset (the same scenarios, each solved on one of
  the test-pool grid copies). The estimators are not told: the admittance pattern stays the
  nominal grid's. The arrays' ``site`` is the copy, ``meta.json`` lists each copy's RMS
  log-factor and ``scoring.npz`` the mean true magnitude per copy and order.
- **Fingerprint.** ``fingerprint_test.json``: sha256 of the test readings (complex64, states in
  split order, the observed rows in row order, ``[S, H, R_obs]``) and of the true test states,
  and float sums per order, to compare with the readings the networks were evaluated on.

Writes ``<out>/arrays_{train,val,test}.npz`` (``v``, ``x_classA`` or the configured variant
name, ``x_clean``, optionally ``c_classA``, ``site``, ``interval``, ``is_event``), ``admittance.npz``, ``scoring.npz``,
``meta.json``, ``fingerprint_test.json``.

Usage::

    python adapter.py <dataset_root> <out_dir> --config <experiment.yaml>
        [--placement "Bus R4,Bus C5,Bus I2"] [--train-draws 4] [--test-seed 0]
        [--train-root ROOT --train-config-random CFG --train-config-coherent CFG]
        [--currents upstream]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout as A  # noqa: E402

#: Sequences per batch of the evaluation loop (the order of the draws on the noise stream).
EVAL_BATCH = 16
#: Seeds of the training and validation readings (the test readings take ``--test-seed``).
TRAIN_SEED = 1000
VAL_SEED = 2000
#: The current stream's seed: the voltage stream's seed plus this (the evaluation's 0 / 7).
CURRENT_SEED_OFFSET = 7
DEPLOYED = ("Bus R4", "Bus C5", "Bus I2")


def load_source(root: Path, config: Path, subdir: str = "dataset_coherent"):
    """``(cfg, grid, source)``: the data source of an experiment configuration on ``root``.

    Only the configuration's ``data``, ``noise`` and ``train`` sections are read
    (``injection_hse.config.DataConfig``, ``NoiseConfig``, ``TrainConfig``); the dataset path is
    replaced by ``root/subdir``.
    """
    from types import SimpleNamespace

    from pgml.schemas.grid_schema import Grid

    from injection_hse.config import DataConfig, NoiseConfig, TrainConfig
    from injection_hse.data import build_data_source

    doc = yaml.safe_load(Path(config).read_text())
    cfg = SimpleNamespace(data=DataConfig.model_validate(doc.get("data", {})),
                          noise=NoiseConfig.model_validate(doc.get("noise", {})),
                          train=TrainConfig.model_validate(doc.get("train", {})))
    grid = Grid.model_validate_json((root / "grid.json").read_text())
    data_cfg = cfg.data.model_copy(
        update={"dataset_path": str(root / subdir), "num_workers": 0})
    return cfg, grid, build_data_source(data_cfg, grid=grid)


def noise_model(cfg, grid, orders, n: int):
    """The configuration's voltage reading model (instrument or accuracy class) over ``n`` rows."""
    from injection_hse.data import InstrumentNoise, SensorNoise

    if cfg.noise.instrument is not None:
        return InstrumentNoise.from_config(cfg.noise, n_nodes=n, grid=grid, orders=orders)
    return SensorNoise.from_config(cfg.noise, n_nodes=n, grid=grid, orders=orders)


class CurrentReadings:
    """The CT channels a placement reads and their instrument model.

    The half-edge currents come from :class:`injection_hse.data.currents.BranchCurrentOperator` (weights
    at the training dtype) applied to the states as loaded, as the network module derives
    them; a half-edge is read where its source row is observed and, with
    ``current_channels="upstream"``, it is the toward-slack half-edge of its node
    (:func:`injection_hse.data.upstream_edge_mask`; the slack keeps its feeder heads). The error is the
    voltage model's current sibling (``for_edges`` with the grid's CT range), its own stream.
    """

    def __init__(self, grid, orders, noise, observed_rows, channels: str, dtype):
        from injection_hse.data import build_graph, grid_current_range, upstream_edge_mask
        from injection_hse.data.currents import BranchCurrentOperator

        self.op = BranchCurrentOperator(grid, list(orders), dtype=dtype)
        graph = build_graph(grid)
        if not torch.equal(self.op.edge_src_rows.cpu(), graph.edge_index[0].cpu()):
            raise ValueError("the current operator and the graph disagree on the half-edges.")
        mask = torch.zeros(int(torch.as_tensor(noise.nominal).numel()), dtype=torch.bool)
        mask[torch.as_tensor(observed_rows)] = True
        read = mask[self.op.edge_src_rows.cpu()]
        if channels == "upstream":
            read = read & upstream_edge_mask(grid, graph.edge_index).cpu()
        elif channels != "incident":
            raise ValueError(f"unknown current channel rule {channels!r}.")
        self.edges = torch.nonzero(read).flatten().numpy()
        self.src_rows = self.op.edge_src_rows.cpu().numpy()[self.edges]
        self.dst_rows = graph.edge_index[1].cpu().numpy()[self.edges]
        self.noise = noise.for_edges(self.op.edge_src_rows, current_range_a=grid_current_range(grid))


def read_split(dataset, noise, seed: int, dtype, currents: "CurrentReadings" = None,
               current_seed: int = None):
    """``(v complex128 [S·T, H, N], y complex64 [S·T, H, N], i complex64 [S·T, H, E] | None)``.

    ``read_split.grid_copy`` then holds int ``[S·T]`` the perturbed grid copy of each state
    (``sample_grid_copy`` of a held-out perturbed dataset; -1 on a nominal one).

    True states, voltage readings and (with ``currents``) the read half-edge currents, drawn
    as the network module draws them in evaluation: each stream re-seeded (``seed``, and
    ``current_seed`` for the currents), batches of :data:`EVAL_BATCH` items in the split's
    order, the voltages cast to ``dtype`` before their error, the currents derived from the
    states as loaded and cast to ``dtype`` after theirs.
    """
    from torch.utils.data import DataLoader

    from injection_hse.data import StateCollate

    noise.reseed(int(seed))
    if currents is not None:
        currents.noise.reseed(int(current_seed))
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, num_workers=0,
                        collate_fn=StateCollate())
    vs, ys, cs, gc = [], [], [], []
    for batch in loader:
        v = batch["v"]
        copy = batch.get("sample_grid_copy")
        steps = int(np.prod(v.shape[1:-2])) if v.dim() > 3 else 1
        gc.append(np.repeat(np.asarray(copy if copy is not None else np.full(v.shape[0], -1)).reshape(-1),
                            steps))
        y = noise.apply(v.to(dtype))
        vs.append(v.reshape(-1, *v.shape[-2:]).to(torch.complex128).numpy())
        ys.append(y.reshape(-1, *y.shape[-2:]).numpy())
        if currents is not None:
            i = currents.noise.apply(currents.op(v)).to(dtype)
            i = i.reshape(-1, *i.shape[-2:])[..., torch.as_tensor(currents.edges)]
            cs.append(i.numpy())
    read_split.grid_copy = np.concatenate(gc)
    return np.concatenate(vs), np.concatenate(ys), (np.concatenate(cs) if cs else None)


def to_samples(v: np.ndarray, rows: np.ndarray, meters: np.ndarray, scale: np.ndarray,
               rot: np.ndarray) -> np.ndarray:
    """``[S, H, N]`` -> ``[S, 3, B, H]`` (or the meters' ``[S, 3, M, H]`` with ``meters``)."""
    sel = slice(None) if meters is None else meters
    out = v[:, :, rows[sel]] * scale[sel][None, None] * rot[None, :, None, :]  # [S, H, B, 3]
    return np.transpose(out, (0, 3, 2, 1)).astype(np.complex64)


def fingerprint(y_obs: np.ndarray, v: np.ndarray, orders) -> dict:
    """sha256 and per-order sums of the test readings at the observed rows and of the states."""
    yc = np.ascontiguousarray(y_obs.astype(np.complex64))
    vc = np.ascontiguousarray(v.astype(np.complex128))
    return dict(
        readings_shape=list(yc.shape), readings_dtype="complex64",
        readings_sha256=hashlib.sha256(yc.tobytes()).hexdigest(),
        states_sha256=hashlib.sha256(vc.tobytes()).hexdigest(),
        orders=list(orders),
        readings_abs_sum_per_order=[float(x) for x in np.abs(yc.astype(np.complex128)).sum(axis=(0, 2))],
        readings_re_sum_per_order=[float(x) for x in yc.real.astype(np.float64).sum(axis=(0, 2))],
        readings_im_sum_per_order=[float(x) for x in yc.imag.astype(np.float64).sum(axis=(0, 2))],
        states_abs_sum_per_order=[float(x) for x in np.abs(vc).sum(axis=(0, 2))],
    )


def run(args) -> None:
    from injection_hse.data import MeasurementModel

    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, grid, source = load_source(root, Path(args.config))
    test_source = (source if args.test_subdir == "dataset_coherent"
                   else load_source(root, Path(args.config), args.test_subdir)[2])
    test_ds = test_source.test_dataset()
    h = int(test_ds[0]["v"].shape[-2])
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    orders = tuple(int(o) for o in (args.orders.split(",") if args.orders else manifest_orders(manifest, h)))
    if len(orders) != h:
        raise ValueError(f"{len(orders)} orders given, the states carry {h}.")
    from injection_hse.data import base_voltage_per_row

    rows = A.index_rows(grid)
    base = base_voltage_per_row(grid).numpy().astype(float)
    slack_rows = MeasurementModel.resolve(grid, [], slack_known=True).observed_rows(grid).numpy()
    names = A.bus_names(grid)
    slack_buses = sorted({int(np.flatnonzero((rows == r).any(1))[0]) for r in slack_rows})
    if len(slack_buses) != 1:
        raise ValueError(f"expected one slack bus, found {slack_buses}.")
    slack = names[slack_buses[0]]
    v_ref = float(base[slack_rows[0]])
    placement = [b.strip() for b in args.placement.split(",")]
    meters = A.meter_buses(grid, placement, slack)
    scale = (v_ref / base)[rows]
    rot = A.phase_rotation(orders).T  # [H, 3]
    observed = MeasurementModel.resolve(grid, placement, slack_known=True).observed_rows(grid).numpy()
    if not set(observed) == set(rows[meters].reshape(-1).tolist()):
        raise ValueError("the observed rows are not the meter buses' rows.")
    unobs = np.setdiff1d(np.arange(base.size), observed)
    dtype = getattr(torch, str(cfg.train.dtype))
    variant = args.variant
    noise = noise_model(cfg, grid, orders, base.size)

    currents = None
    if args.currents != "none":
        currents = CurrentReadings(grid, orders, noise, observed, args.currents, dtype)
        # per phase: the half-edges whose source row is phase p, in edge order
        phase_of_row = np.zeros(base.size, dtype=int)
        for p in range(3):
            phase_of_row[rows[:, p]] = p
        by_phase = [np.flatnonzero(phase_of_row[currents.src_rows] == p) for p in range(3)]
        if len({len(b) for b in by_phase}) != 1:
            raise ValueError("the phases read different numbers of CT channels.")
        bus_of_row = np.zeros(base.size, dtype=int)
        for b in range(rows.shape[0]):
            bus_of_row[rows[b]] = b
        ct_channels = [dict(edge=int(currents.edges[j]), phase=A.PHASES[p],
                            at=names[bus_of_row[currents.src_rows[j]]],
                            toward=names[bus_of_row[currents.dst_rows[j]]])
                       for p in range(3) for j in by_phase[p]]

    def phase_currents(i: np.ndarray) -> np.ndarray:
        """``[S, H, E_read]`` amperes -> ``[S, 3, K, H]`` per phase, in the frame of phase a."""
        out = np.stack([i[:, :, by_phase[p]] for p in range(3)], axis=1)  # [S, 3, H, K]
        out = out * rot.T[None, :, :, None]
        return np.transpose(out, (0, 1, 3, 2)).astype(np.complex64)

    plan = {"test": [(test_source, "test", args.test_seed)]}
    copies = None
    if args.train_root:
        troot = Path(args.train_root)
        t_sources = [load_source(troot, Path(c), sub)[2] for sub, c in
                     (("dataset_random", args.train_config_random),
                      ("dataset_coherent", args.train_config_coherent)) if c]
    else:
        t_sources = [source]
    plan["train"] = [(src, "train", TRAIN_SEED + d) for d in range(args.train_draws) for src in t_sources]
    plan["val"] = [(src, "val", VAL_SEED) for src in t_sources]
    counts = {}
    for split, parts in plan.items():
        parts_v, parts_x, parts_c, parts_i, interval = [], [], [], [], []
        n_states = 0
        for src, sp, s in parts:
            ds = getattr(src, f"{sp}_dataset")()
            v, y, i = read_split(ds, noise, s, dtype, currents, s + CURRENT_SEED_OFFSET)
            parts_v.append(to_samples(v, rows, None, scale, rot))
            parts_x.append(to_samples(y.astype(np.complex128), rows, meters, scale, rot))
            parts_c.append(to_samples(v, rows, meters, scale, rot))
            if i is not None:
                parts_i.append(phase_currents(i))
            interval.append(np.arange(v.shape[0], dtype=np.int32))
            n_states += v.shape[0]
            if split == "test":
                fp = fingerprint(y[:, :, observed], v, orders)
                fp.update(seed=int(s), batch_sequences=EVAL_BATCH, train_dtype=str(cfg.train.dtype),
                          observed_rows=observed.tolist(), n_states=int(v.shape[0]))
                if i is not None:
                    ic = np.ascontiguousarray(i.astype(np.complex64))
                    fp.update(currents_seed=int(s + CURRENT_SEED_OFFSET), current_edges=currents.edges.tolist(),
                              currents_shape=list(ic.shape),
                              currents_sha256=hashlib.sha256(ic.tobytes()).hexdigest(),
                              currents_abs_sum_per_order=[float(x) for x in np.abs(
                                  ic.astype(np.complex128)).sum(axis=(0, 2))])
                (out / "fingerprint_test.json").write_text(json.dumps(fp, indent=1))
                true_mean_pu = (np.abs(v[:, :, unobs]) / base[unobs]).mean(axis=(0, 2))
                test_copy = read_split.grid_copy
                if (test_copy >= 0).any():
                    copies = sorted(int(c) for c in np.unique(test_copy))
                    true_mean_site = np.stack([(np.abs(v[test_copy == c][:, :, unobs]) / base[unobs]
                                                ).mean(axis=(0, 2)) for c in copies])
            del v, y, i
        arr = {"v": np.concatenate(parts_v), f"x_{variant}": np.concatenate(parts_x),
               "x_clean": np.concatenate(parts_c)}
        del parts_v, parts_x, parts_c
        if parts_i:
            arr[f"c_{variant}"] = np.concatenate(parts_i)
        n = arr["v"].shape[0]
        site = np.zeros(n, dtype=np.int32)
        if split == "test" and copies is not None:
            site = np.searchsorted(copies, test_copy).astype(np.int32)
        arr.update(site=site, interval=np.concatenate(interval),
                   is_event=np.zeros(n, dtype=bool))
        np.savez(out / f"arrays_{split}.npz", **arr)
        counts[split] = dict(parts=[f"{sp}@{getattr(src, 'dataset_path', '')}" for src, sp, _ in parts],
                             states_read=int(n_states), samples=int(3 * n),
                             seeds=[int(s) for _, _, s in parts])
        del arr
    y_adm = A.admittance_per_order(grid, orders, v_ref)
    adjacency = np.abs(y_adm[0]) > 1e-8
    np.savez(out / "admittance.npz", freqs_hz=np.array(orders) * float(grid.base_frequency_hz), Y=y_adm)
    np.savez(out / "scoring.npz", rows=rows, base=base, unobs=unobs, slack_rows=slack_rows,
             g_s=np.zeros((len(orders), base.size, slack_rows.size), dtype=np.complex128), v_ref=v_ref, orders=np.array(orders),
             phase_rotation=A.phase_rotation(orders), true_mean_pu=true_mean_pu,
             **({"true_mean_pu_site": true_mean_site} if copies is not None else {}))
    copy_levels = {}
    if copies is not None:
        tmeta = json.loads((root / args.test_subdir / "meta.json").read_text())
        rec = tmeta.get("extra_provenance", {}).get("grid_perturbation", {})
        copy_levels = {f"copy{int(c['index'])}": float(c["magnitude_rms_log"])
                       for c in rec.get("copies", [])}
    meta = dict(
        dataset=str(root.resolve()), config=str(Path(args.config).resolve()),
        placement_buses=placement, buses=names, meter_buses=[names[i] for i in meters],
        meter_index=meters.tolist(), slack_bus=slack, orders=list(orders), v_ref=v_ref,
        phases=list(A.PHASES), phase_frame="a", variants=[variant, "clean"],
        noise=cfg.noise.model_dump(mode="json"), train_dtype=str(cfg.train.dtype),
        split=dict(split_seed=cfg.data.split_seed, val_fraction=cfg.data.val_fraction,
                   test_fraction=cfg.data.test_fraction),
        splits={"train": ["train"], "val": ["val"],
                "test": [f"copy{c}" for c in copies] if copies is not None else ["test"]},
        counts=counts, test_subdir=args.test_subdir, grid_copy_rms_log=copy_levels,
        pann_layers=A.hop_eccentricity(adjacency, meters),
        admittance_edges=int((adjacency.sum() - adjacency.shape[0]) // 2),
        scored_rows=int(unobs.size),
        rule="every state of the split; the training split read train_draws times",
        train_root=args.train_root, current_channels=args.currents,
        ct_channels=(ct_channels if currents is not None else []),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_provenance(out, vars(args))
    print(json.dumps({k: meta[k] for k in ("meter_buses", "pann_layers", "scored_rows", "counts")}),
          flush=True)


def write_provenance(out: Path, args: dict) -> None:
    """``provenance_mack_adapter.json``: arguments, versions and the imported checkouts."""
    import os
    import time

    import injection_hse
    import pgml

    rec = dict(args=args, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
               imported_from={"injection_hse": str(Path(injection_hse.__file__).resolve().parents[2]),
                              "pgml": str(Path(pgml.__file__).resolve().parents[2])},
               shas={k: os.environ.get(f"MACK_{k.upper()}_SHA") for k in ("code", "injection_hse", "pgml")})
    (out / "provenance_mack_adapter.json").write_text(json.dumps(rec, indent=1, default=str))


def manifest_orders(manifest: dict, h: int) -> list:
    """The harmonic orders recorded in a dataset manifest."""
    for key in ("orders", "harmonic_orders"):
        if key in manifest:
            return manifest[key]
    for value in manifest.values():
        if isinstance(value, dict):
            for key in ("orders", "harmonic_orders"):
                if key in value and len(value[key]) == h:
                    return value[key]
    raise ValueError("the manifest records no orders; pass --orders.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", help="root of a generate_se_task_datasets.py output")
    ap.add_argument("out")
    ap.add_argument("--config", required=True, help="experiment configuration (YAML) of the evaluated networks")
    ap.add_argument("--placement", default=",".join(DEPLOYED))
    ap.add_argument("--orders", default="", help="comma list (default: the dataset manifest's)")
    ap.add_argument("--variant", default="classA", help="name of the noisy input variant")
    ap.add_argument("--train-draws", type=int, default=4,
                    help="independent reading draws of the training split")
    ap.add_argument("--test-seed", type=int, default=0, help="seed of the test readings")
    ap.add_argument("--test-subdir", default="dataset_coherent",
                    help="dataset of DATASET whose test split is scored, e.g. dataset_coherent_heldout "
                    "(the held-out perturbed grid copies; the scores are then per copy)")
    ap.add_argument("--train-root", default="",
                    help="root of the training data (default: the train / val splits of DATASET)")
    ap.add_argument("--train-config-random", default="",
                    help="with --train-root: configuration of its dataset_random (snapshots)")
    ap.add_argument("--train-config-coherent", default="",
                    help="with --train-root: configuration of its dataset_coherent (sequences)")
    ap.add_argument("--currents", default="none", choices=["none", "upstream", "incident"],
                    help="also write the CT readings of the placement (c_<variant>)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
