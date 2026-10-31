"""Baseline table, per-order errors, meter sweep and placement search.

Scores, on the test split of an evaluation dataset (the evaluation set of the paper: 2 000
sequences of 8 steps generated with the training recipe and its own seed; test split 200
sequences = 1 600 states), at named meter placements:

- ``physics``: the anchored solve with zero injections, per-channel reading sigmas,
  ``ρ = 1`` (weighted least squares with zero-injection pseudo-measurements);
- the linear map (ridge, :func:`injection_hse.analysis.baselines.fit_linear_injection_prior`
  on the noisy training split, refitted per placement), decoded unanchored and anchored
  with ``ρ`` = its own validation error; the best decode per order is reported separately;
- the statistics-based estimator (linear MMSE from the training injections' mean and
  covariance and the exact measurement map), unanchored;
- each network checkpoint: as deployed (its own anchor weights), with the corrected decode
  (per-channel sigmas, ``ρ`` = its own validation injection error) and unanchored.

The baselines are scored in the frame of one network checkpoint (``frame``: its reading
noise, solver and injection scale ``s_I(h)``); fits and calibrations use the evaluation
set's own training and validation splits. The reading noise is the checkpoint's own model
(``noise="A"``: the class-A instrument) or an accuracy class swapped in at evaluation.

Metric: the pu-MAE of the complex voltage on the unobserved rows, per order (``pu_mae``),
and the relative error ``rel`` = that over the mean unobserved ``|V_h|`` in per unit. With
``test_dataset = "dataset_coherent_heldout"`` the test states are those of the held-out
perturbed grid copies (every estimator still decodes with the nominal ``Y(h)``); the scores
are then also given per copy (column ``copy``; ``"all"`` = pooled).

Outputs (CSV, one row per placement × estimator × order):
``baseline_table_noise<N>_<placements>.csv``, beside it ``ci_noise<N>_<placements>.csv``
(h1 and the mean over h3–h19: mean and half-width of the 95 % confidence interval over the
test sequences, each sequence's 8 states averaged); with a sweep
``placement_sweep_noise<N>_k<ks>.csv``; with a search shard
``placement_search_noise<N>_shard<i>of<n>.csv``.
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import frame as F
from .names import PLACEMENTS


#: steps per test sequence (the states of one sequence are consecutive in every pass)
N_STEPS = 8


class Score(tuple):
    """``(pu_mae [H], rel [H])`` with ``per_copy``: ``{copy: (pu_mae, rel)}`` or ``{}``, and
    ``ci``: ``{"h1"|"harm": (mean, half-width of the 95 % CI, n_sequences), "seq": the
    per-sequence errors [n_sequences, H]}`` or ``{}``."""

    per_copy: dict = {}
    ci: dict = {}


def sequence_ci(v_est, v_true, n_steps: int = N_STEPS) -> dict:
    """Mean and 95 % confidence half-width over test sequences of the pu-MAE at h1 and of
    the mean over the harmonic orders (``v_est``, ``v_true``: ``[S, H, R]`` per unit,
    sequences of ``n_steps`` consecutive states).

    The states of one sequence are correlated, so the sequence (its states averaged) is the
    sampling unit: half-width ``1.96 · std(ddof=1) / sqrt(n_sequences)``. ``{}`` when the
    states do not form whole sequences.
    """
    e = np.abs(v_est - v_true).mean(axis=2)  # [S, H]
    if e.shape[0] % n_steps or e.shape[0] < 2 * n_steps:
        return {}
    return ci_from_sequences(e.reshape(-1, n_steps, e.shape[1]).mean(axis=1))


def ci_from_sequences(seq: np.ndarray) -> dict:
    """The confidence intervals of :func:`sequence_ci` from per-sequence errors
    ``[n_sequences, H]`` (kept under ``"seq"``)."""
    out = {"seq": seq}
    for part, x in (("h1", seq[:, 0]), ("harm", seq[:, 1:].mean(axis=1))):
        out[part] = (float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), len(x))
    return out


def _score(v_hat, v_true):
    err = np.abs(v_hat - v_true).mean(axis=(0, 2))
    return err, err / np.abs(v_true).mean(axis=(0, 2))


def feeder_combinations(grid):
    """Every placement with one LV meter per feeder (R, I, C), in a fixed order (720 on
    the CIGRE LV benchmark: 18 × 2 × 20)."""
    names = sorted(str(n.name) for n in grid.nodes)
    feeder = {f: [n for n in names if n.startswith(f"Bus {f}") and n != f"Bus {f}0"] for f in "RIC"}
    return [list(c) for c in itertools.product(feeder["R"], feeder["I"], feeder["C"])]


@dataclass
class Evaluation:
    """One evaluation over a dataset directory (``grid.json`` + ``dataset_coherent``)."""

    data: Path
    out: Path
    noise: str = "A"
    test_dataset: str = "dataset_coherent"
    placements: dict = field(default_factory=lambda: dict(PLACEMENTS))
    #: grid copy of every test state when the test data are perturbed copies
    copies: "np.ndarray | None" = None
    records: list = field(default_factory=list)
    ci_records: list = field(default_factory=list)

    # ------------------------------------------------------------------ loading
    def load(self, ckpt: Path, sensors, dataset="dataset_coherent"):
        grid = F.load_grid(self.data)
        module, source, ck_w = F.load_module(grid, ckpt, self.data / dataset)
        if F.noise_kind(module) != self.noise:
            F.swap_noise(module, grid, self.noise)
        F.set_placement(module, grid, sensors)
        return grid, module, source, ck_w

    def test_source(self, ckpt, sensors, src):
        """The source whose test split is scored; records the grid copy of each state for
        perturbed test data."""
        if self.test_dataset == "dataset_coherent":
            return src
        _, _, src_t, _ = self.load(Path(ckpt), sensors, dataset=self.test_dataset)
        ds = src_t.test_dataset()
        copies = []
        for i in range(len(ds)):
            item = ds[i]
            n_steps = int(item["v"].shape[0]) if item["v"].dim() == 3 else 1
            copies += [int(torch.as_tensor(item["sample_grid_copy"]).reshape(-1)[0])] * n_steps
        self.copies = np.asarray(copies)
        return src_t

    # ------------------------------------------------------------------ scoring
    def scores(self, v_hat, data, rows):
        """``(pu_mae [H], rel [H])`` on the unobserved rows (and per grid copy, if any)."""
        unobs, base_u = rows["unobs"], rows["base_v"][rows["unobs"]]
        v_true = data["v"].numpy()[:, :, unobs] / base_u
        v_est = v_hat[:, :, unobs] / base_u
        out = Score(_score(v_est, v_true))
        out.ci = sequence_ci(v_est, v_true)
        out.per_copy = {}
        if self.copies is not None and len(self.copies) == v_true.shape[0]:
            out.per_copy = {
                int(c): _score(v_est[self.copies == c], v_true[self.copies == c])
                for c in np.unique(self.copies)
            }
        return out

    def add(self, placement, run, estimator, res, orders, **extra):
        for c, (e_c, r_c) in getattr(res, "per_copy", {}).items():
            for h, e, r in zip(orders, e_c, r_c):
                self.records.append(
                    {
                        "noise": self.noise,
                        "placement": placement,
                        "run": run,
                        "estimator": estimator,
                        "order": int(h),
                        "pu_mae": float(e),
                        "rel": float(r),
                        "copy": c,
                        **extra,
                    }
                )
        if getattr(res, "per_copy", None):
            extra = {**extra, "copy": "all"}
        for part, (mean, half, n) in (
            (k, v) for k, v in getattr(res, "ci", {}).items() if k != "seq"
        ):
            self.ci_records.append(
                {
                    "noise": self.noise,
                    "placement": placement,
                    "run": run,
                    "estimator": estimator,
                    "part": part,
                    "pu_mae_mean": mean,
                    "ci95_half_width": half,
                    "n_sequences": n,
                }
            )
        err, rel = res[0], res[1]
        for h, e, r in zip(orders, err, rel):
            self.records.append(
                {
                    "noise": self.noise,
                    "placement": placement,
                    "run": run,
                    "estimator": estimator,
                    "order": int(h),
                    "pu_mae": float(e),
                    "rel": float(r),
                    **extra,
                }
            )
        print(
            f"{placement:12s} {run:18s} {estimator:44s} h1 {err[0]:.3e} "
            f"harm {np.mean(err[1:]):.3e}  rel h1 {rel[0]:.2e} harm {np.mean(rel[1:]):.2e}",
            flush=True,
        )

    @staticmethod
    def injection_rho(mod, inj_val, val, s_inj, orders, rows):
        """Per-order relative injection error of ``inj_val`` on the validation split."""
        i_true = F.true_injection(mod, val["v"])
        dev = np.intersect1d(
            np.flatnonzero(np.sqrt((np.abs(i_true) ** 2).mean((0, 1))) > 0), rows["free"]
        )
        e = inj_val.numpy() - i_true
        s_np = s_inj.numpy()
        return np.array(
            [
                np.sqrt(
                    (
                        0.5
                        * np.abs(e[:, k, dev]) ** 2
                        / (s_np[k, dev] if s_np.shape[1] > 1 else s_np[k, 0]) ** 2
                    ).mean()
                )
                for k in range(len(orders))
            ]
        )

    def score_network(self, name, ckpt, pname, sensors):
        """A network checkpoint as deployed, with the corrected decode and unanchored."""
        grid, mod, src, ck_w = self.load(Path(ckpt), sensors)
        rows = F.row_info(mod, grid)
        orders = list(mod.orders)
        if mod.cfg.physics.anchor_weighting != "noise_calibrated":
            F.use_manual(mod, *ck_w)
        test = F.collect(mod, self.test_source(ckpt, sensors, src), "test")
        val = F.collect(mod, src, "val")
        self.add(
            pname,
            name,
            "network, as deployed",
            self.scores(F.decode(mod, test["i_net"], test), test, rows),
            orders,
        )
        F.use_noise_calibrated(mod, 1.0, 1.0)
        s_inj = F.anchor_sigmas(mod)[0].reshape(len(orders), -1)
        rho = self.injection_rho(mod, val["i_net"], val, s_inj, orders, rows)
        s_v, s_i = F.sensor_sigmas(mod, val)
        sv = torch.as_tensor(np.maximum(s_v, 1e-12))
        si = torch.as_tensor(np.maximum(s_i, 1e-12))
        F.install(mod, (s_inj * torch.as_tensor(rho)[:, None]).squeeze(-1), sv, si, 1.0, 1.0)
        self.add(
            pname,
            name,
            "network, corrected decode",
            self.scores(F.decode(mod, test["i_net"], test), test, rows),
            orders,
        )
        F.use_manual(mod, 0.0, 0.0)
        self.add(
            pname,
            name,
            "network, unanchored",
            self.scores(F.decode(mod, test["i_net"], test), test, rows),
            orders,
        )
        print(f"{pname:12s} {name:18s} rho {np.round(rho, 3)}", flush=True)

    def score_baselines(self, ckpt, pname, sensors, cache):
        """Physics, the linear map (three decodes) and the statistics-based estimators."""
        from injection_hse.analysis.baselines import fit_linear_injection_prior

        grid, mod, src, _ = self.load(Path(ckpt), sensors)
        rows = F.row_info(mod, grid)
        orders = list(mod.orders)
        free = rows["free"]
        F.use_noise_calibrated(mod, 1.0, 1.0)
        if "v_train" not in cache:
            cache["v_train"] = F.states(src.train_dataset())
            i_train = F.true_injection(mod, cache["v_train"])
            x = F.real_stack(i_train[:, :, free])
            cache["mu"], cache["sigma"] = x.mean(0), np.cov(x, rowvar=False)
            del x, i_train
        v_train = cache["v_train"]
        s_inj = F.anchor_sigmas(mod)[0].reshape(len(orders), -1)
        mask = rows["mask"].unsqueeze(0).expand(v_train.shape[0], -1).clone()
        ridge = fit_linear_injection_prior(mod, v_train, mask, noisy=True)
        mod.set_injection_prior(ridge)
        F.use_noise_calibrated(mod, 1.0, 1.0)
        test = F.collect(mod, self.test_source(ckpt, sensors, src), "test")
        val = F.collect(mod, src, "val")
        s_v, s_i = F.sensor_sigmas(mod, val)
        sv = torch.as_tensor(np.maximum(s_v, 1e-12))
        si = torch.as_tensor(np.maximum(s_i, 1e-12))
        F.install(mod, s_inj.squeeze(-1), sv, si, 1.0, 1.0)
        self.add(
            pname,
            "baseline",
            "physics solve",
            self.scores(F.decode(mod, torch.zeros_like(test["i_net"]), test), test, rows),
            orders,
        )
        rho = self.injection_rho(mod, val["i_prior"], val, s_inj, orders, rows)
        F.install(mod, (s_inj * torch.as_tensor(rho)[:, None]).squeeze(-1), sv, si, 1.0, 1.0)
        anch = self.scores(F.decode(mod, test["i_prior"], test), test, rows)
        self.add(pname, "baseline", "linear map, anchored (rho = its error)", anch, orders)
        frame = F.Frame(mod, rows, test)  # leaves the solver plain
        unanch = self.scores(F.decode(mod, test["i_prior"], test), test, rows)
        self.add(pname, "baseline", "linear map, unanchored", unanch, orders)

        def _best(a, u):
            return (np.minimum(a[0], u[0]), np.where(a[0] <= u[0], a[1], u[1]))

        best = Score(_best(anch, unanch))
        best.ci = {}
        if "seq" in anch.ci and "seq" in unanch.ci:
            pick = anch[0] <= unanch[0]  # [H]: the decode the best row takes per order
            best.ci = ci_from_sequences(np.where(pick[None, :], anch.ci["seq"], unanch.ci["seq"]))
        best.per_copy = {c: _best(anch.per_copy[c], unanch.per_copy[c]) for c in anch.per_copy}
        self.add(pname, "baseline", "linear map, best decode per order", best, orders)
        y, b = frame.readings(test)
        r_diag = frame.noise_cov(s_v, s_i)
        xh = F.lmmse(frame, y, cache["mu"], cache["sigma"], r_diag)
        self.add(
            pname,
            "baseline",
            "statistics-based (linear MMSE)",
            self.scores(
                frame.voltages(F.complex_unstack(xh, len(orders), len(free)), b), test, rows
            ),
            orders,
        )
        print(f"{pname:12s} linear map rho {np.round(rho, 3)}", flush=True)

    # ------------------------------------------------------------------ table
    def table(self, runs: dict, frame_ckpt, placements: list[str]):
        """Baselines and networks at every placement in ``placements``; writes the per-order
        table and the confidence intervals beside it."""
        cache: dict = {}
        tag = "_".join(placements)
        path = self.out / f"baseline_table_noise{self.noise}_{tag}.csv"
        for pname in placements:
            sensors = self.placements[pname]
            self.score_baselines(frame_ckpt, pname, sensors, cache)
            for name, ckpt in runs.items():
                self.score_network(name, ckpt, pname, sensors)
            pd.DataFrame(self.records).to_csv(path, index=False)
            pd.DataFrame(self.ci_records).to_csv(
                self.out / f"ci_noise{self.noise}_{tag}.csv", index=False
            )
        return path

    # ------------------------------------------------------------------ sweep / search
    def sweep(self, runs, frame_ckpt, ks, explicit=None, tag=None):
        """Meter sweep at the placements with ``k`` in ``ks`` (``0`` = the deployed one):
        physics, the statistics-based estimator and the networks as deployed. The random
        placements are drawn for every k (seeded) and then selected, so a split sweep
        scores the same placements as a whole one. ``explicit(grid)`` replaces them by a
        given list ``[(label, k, sensors)]`` (the placement search)."""
        deployed = self.placements["deployed"]
        grid, mod, src, _ = self.load(Path(frame_ckpt), deployed)
        names = {int(n.id): str(n.name) for n in grid.nodes}
        candidates = sorted(
            n for n in names.values() if n not in ("Bus 0", "Bus R0", "Bus I0", "Bus C0")
        )
        rng = np.random.default_rng(0)
        placements = [("deployed", 3, deployed)]
        for k in range(1, 7):
            for d in range(8):
                placements.append(
                    (f"random k{k} #{d}", k, list(rng.choice(candidates, size=k, replace=False)))
                )
        placements = [p for p in placements if (0 if p[0] == "deployed" else p[1]) in ks]
        if explicit is not None:
            placements = explicit(grid)
        F.use_noise_calibrated(mod, 1.0, 1.0)
        rows0 = F.row_info(mod, grid)
        free = rows0["free"]
        f = len(free)
        orders = list(mod.orders)
        x = F.real_stack(F.true_injection(mod, F.states(src.train_dataset()))[:, :, free])
        mu, sig = x.mean(0), np.cov(x, rowvar=False)
        del x
        s_inj = F.anchor_sigmas(mod)[0].reshape(len(orders), -1).squeeze(-1)
        test = F.collect(mod, src, "test")
        s_v, s_i = F.sensor_sigmas(mod, F.collect(mod, src, "val"))
        nets = {}
        for name, ck in runs.items():
            _, m, s, w = self.load(Path(ck), deployed)
            if m.cfg.physics.anchor_weighting != "noise_calibrated":
                F.use_manual(m, *w)
            nets[name] = (m, s)
        name = (
            f"placement_search_noise{self.noise}_{tag}.csv"
            if tag
            else f"placement_sweep_noise{self.noise}_k{'_'.join(str(k) for k in sorted(ks))}.csv"
        )
        recs = []
        for label, k, sensors in placements:
            t0 = time.time()
            n_states = test["v"].shape[0]
            F.set_placement(mod, grid, sensors)
            m_, e_ = F.placement_masks(mod)
            data = dict(
                test,
                mask=m_.unsqueeze(0).expand(n_states, -1).clone(),
                edge_mask=e_.unsqueeze(0).expand(n_states, -1).clone(),
            )
            rows = F.row_info(mod, grid)
            F.install(
                mod,
                s_inj,
                torch.as_tensor(np.maximum(s_v, 1e-12)),
                torch.as_tensor(np.maximum(s_i, 1e-12)),
                1.0,
                1.0,
            )
            out = {
                "physics solve": self.scores(
                    F.decode(mod, torch.zeros_like(test["i_net"]), data), data, rows
                )
            }
            frame = F.Frame(mod, rows, data)
            y, b = frame.readings(data)
            r_diag = frame.noise_cov(s_v, s_i)
            xh = F.lmmse(frame, y, mu, sig, r_diag)
            out["statistics-based"] = self.scores(
                frame.voltages(F.complex_unstack(xh, len(orders), f), b), data, rows
            )
            for net, (m, s) in nets.items():
                F.set_placement(m, grid, sensors)
                nt = F.collect(m, s, "test")
                out[f"{net}, as deployed"] = self.scores(
                    F.decode(m, nt["i_net"], nt), nt, F.row_info(m, grid)
                )
            for est, (err, rel) in out.items():
                for h, e, r in zip(orders, err, rel):
                    recs.append(
                        {
                            "noise": self.noise,
                            "placement": label,
                            "k": k,
                            "sensors": " ".join(sensors),
                            "estimator": est,
                            "order": h,
                            "pu_mae": float(e),
                            "rel": float(r),
                        }
                    )
            print(
                f"{label:16s} {' '.join(s_.replace('Bus ', '') for s_ in sensors):22s} "
                + "  ".join(
                    f"{e}: h1 {v[0][0]:.2e} harm {np.mean(v[0][1:]):.2e}" for e, v in out.items()
                )
                + f"  ({time.time() - t0:.0f} s)",
                flush=True,
            )
            pd.DataFrame(recs).to_csv(self.out / name, index=False)
        return self.out / name

    def search(self, frame_ckpt, shard: int, n_shards: int, limit: int | None = None):
        """Physics and the statistics-based estimator at every placement with one meter
        per feeder, the placements ``i`` with ``i % n_shards == shard`` (the first ``limit``
        of them when given)."""

        def explicit(grid):
            combos = feeder_combinations(grid)
            chosen = [(" ".join(c), 3, c) for i, c in enumerate(combos) if i % n_shards == shard]
            return chosen[:limit] if limit else chosen

        return self.sweep({}, frame_ckpt, set(), explicit=explicit, tag=f"shard{shard}of{n_shards}")


def _buses(label: str) -> list[str]:
    """The bus names in a placement label (``"Bus R4 Bus I2 Bus C5"``)."""
    return re.findall(r"Bus \w+", label)


def summarize_search(frames: list[pd.DataFrame], deployed: list[str]) -> tuple[pd.DataFrame, dict]:
    """The placement search as one row per placement (``<estimator>|h1``,
    ``<estimator>|harm`` = mean over the harmonic orders) and the facts the paper quotes:
    the rank of the deployed placement for the statistics-based estimator at h1 and at the
    harmonics (1 = best) and the LMMSE-best placement, the one minimising
    ``h1 / min(h1) + harm / min(harm)`` of the statistics-based estimator."""
    d = pd.concat(frames, ignore_index=True)
    d["part"] = np.where(d.order == 1, "h1", "harm")
    piv = d.groupby(["placement", "estimator", "part"]).pu_mae.mean().unstack(["estimator", "part"])
    piv.columns = [f"{e}|{p}" for e, p in piv.columns]
    piv = piv.sort_index()
    key = "statistics-based"
    h1, harm = piv[f"{key}|h1"], piv[f"{key}|harm"]
    best = (h1 / h1.min() + harm / harm.min()).idxmin()
    facts = {"n_placements": int(len(piv)), "lmmse_best": best}
    dep = next((p for p in piv.index if set(_buses(p)) == set(deployed)), None)
    if dep is not None:
        facts.update(
            {
                "deployed": dep,
                "deployed_rank_h1": int(h1.rank(method="min")[dep]),
                "deployed_rank_harm": int(harm.rank(method="min")[dep]),
            }
        )
    return piv.reset_index(), facts
