"""adapter on a small generated v4 task dataset: rows, units, readings, fingerprint, tables."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import adapter as P
import protocol_table as T

ORDERS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def _repo_root() -> Path:
    import injection_hse

    return Path(injection_hse.__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = _repo_root()
    script = root / "run" / "examples" / "injection_hse" / "generate_se_task_datasets.py"
    if not script.exists():
        pytest.skip("the repository's run/ directory is not available")
    out = tmp_path_factory.mktemp("v4")
    subprocess.run([sys.executable, str(script), str(out), "--scenario-config",
                    "run/configs/pgl_se_scenario_v4.yaml", "--orders", ",".join(map(str, ORDERS)),
                    "--device-capacitance", "run/configs/pgl_device_capacitance.yaml",
                    "--source-impedance", "cigre-tb575", "--load-shunt", "opendss",
                    "--load-shunt-basis", "nameplate", "--n-scenarios", "40", "--n-steps", "8",
                    "--seed", "0", "--skip-random"], cwd=root, check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    from injection_hse.config import DataConfig, InstrumentErrorConfig, NoiseConfig, TrainConfig

    doc = {"data": DataConfig(split_seed=0, val_fraction=0.2, test_fraction=0.1).model_dump(mode="json"),
           "noise": NoiseConfig(enabled=True, seed=0,
                                instrument=InstrumentErrorConfig.class_a()).model_dump(mode="json"),
           "train": TrainConfig(dtype="complex64").model_dump(mode="json")}
    path = tmp_path_factory.mktemp("cfg") / "experiment.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


@pytest.fixture(scope="module")
def adapted(dataset, config, tmp_path_factory):
    out = tmp_path_factory.mktemp("adapted")
    P.run(type("Args", (), dict(dataset=str(dataset), out=str(out), config=str(config),
                                placement=",".join(P.DEPLOYED), orders="", variant="classA",
                                train_draws=2, test_seed=0, test_subdir="dataset_coherent", train_root="", train_config_random="",
                                train_config_coherent="", currents="upstream"))())
    return out


def test_layout_meters_and_scored_rows(adapted):
    meta = json.loads((adapted / "meta.json").read_text())
    assert meta["meter_buses"] == ["Bus 0", "Bus R4", "Bus I2", "Bus C5"]
    assert meta["orders"] == list(ORDERS) and meta["pann_layers"] == 7
    s = np.load(adapted / "scoring.npz")
    assert s["unobs"].size == 132 - 12          # every row but the meters' and the slack's
    assert not set(s["unobs"]) & {0, 1, 2, 15, 16, 17, 66, 67, 68, 84, 85, 86}
    c = meta["counts"]
    assert len(c["train"]["seeds"]) == 2 and len(c["test"]["seeds"]) == 1
    assert c["test"]["states_read"] % 8 == 0
    ct = meta["ct_channels"]
    assert len(ct) == 18 and {d["at"] for d in ct} == {"Bus R4", "Bus I2", "Bus C5", "Bus 0"}
    assert {(d["at"], d["toward"]) for d in ct} == {("Bus R4", "Bus R3"), ("Bus I2", "Bus I1"),
                                                   ("Bus C5", "Bus C4"), ("Bus 0", "Bus R0"),
                                                   ("Bus 0", "Bus I0"), ("Bus 0", "Bus C0")}
    z = np.load(adapted / "arrays_test.npz")
    assert z["c_classA"].shape[1:] == (3, 6, len(ORDERS))


def test_targets_are_the_split_states_in_the_phase_a_frame(adapted, dataset, config):
    _, _, source = P.load_source(dataset, config)
    v = torch.stack([source.test_dataset()[i]["v"] for i in range(len(source.test_dataset()))])
    v = v.reshape(-1, *v.shape[-2:]).numpy()                                   # [S, H, N]
    z, s = np.load(adapted / "arrays_test.npz"), np.load(adapted / "scoring.npz")
    rows, base, v_ref, rot = s["rows"], s["base"], float(s["v_ref"]), s["phase_rotation"]
    for p in range(3):
        back = z["v"][:, p] / rot[p][None, None, :] * (base[rows[:, p]] / v_ref)[None, :, None]
        ref = np.transpose(v[:, :, rows[:, p]], (0, 2, 1))
        assert np.abs(back - ref).max() <= 1e-6 * np.abs(ref).max()
    true_mean = (np.abs(v[:, :, s["unobs"]]) / base[s["unobs"]]).mean(axis=(0, 2))
    assert np.allclose(s["true_mean_pu"], true_mean, rtol=1e-12)


def test_class_a_readings_follow_the_instrument_and_are_reproducible(adapted, dataset, config):
    z = np.load(adapted / "arrays_test.npz")
    x, clean = z["x_classA"], z["x_clean"]
    rel = np.abs(x - clean) / np.abs(clean)
    # h1: 0.1 % limit at coverage 2 -> about 0.05 % std; harmonics up to 5 % of reading
    assert 1e-5 < np.median(rel[..., 0]) < 2e-3
    assert np.median(rel[..., 1:]) > 3 * np.median(rel[..., 0])
    cfg, grid, source = P.load_source(dataset, config)
    s = np.load(adapted / "scoring.npz")
    noise = P.noise_model(cfg, grid, ORDERS, s["base"].size)
    _, y1, _ = P.read_split(source.test_dataset(), noise, 0, torch.complex64)
    _, y2, _ = P.read_split(source.test_dataset(), noise, 0, torch.complex64)
    assert np.array_equal(y1, y2)
    fp = json.loads((adapted / "fingerprint_test.json").read_text())
    obs = np.array(fp["observed_rows"])
    assert hashlib.sha256(np.ascontiguousarray(y1[:, :, obs]).tobytes()).hexdigest() == fp["readings_sha256"]
    _, y3, _ = P.read_split(source.test_dataset(), noise, 5, torch.complex64)
    assert not np.array_equal(y1, y3)


def test_the_held_error_is_constant_over_a_sequence(dataset, config):
    cfg, grid, source = P.load_source(dataset, config)
    s_base = P.noise_model(cfg, grid, ORDERS, 132)
    v, y, _ = P.read_split(source.test_dataset(), s_base, 0, torch.complex64)
    k = 0                                     # h1: the held gain (0.05 %) dominates the floor
    row = 15                                  # Bus R4.a
    ratio = (y[:, k, row] / v[:, k, row]).reshape(-1, 8)
    within = np.abs(ratio - ratio.mean(1, keepdims=True)).std()   # the floor, drawn per reading
    between = np.abs(ratio.mean(1) - ratio.mean()).std() + np.abs(ratio.mean(1) - 1).mean()
    assert within < 0.3 * between


def test_protocol_table_matches_the_evaluation_formulas(tmp_path, adapted):
    s = np.load(adapted / "scoring.npz")
    orders = [int(o) for o in s["orders"]]
    run = tmp_path / "run"
    run.mkdir()
    mae = np.linspace(1e-3, 1e-4, len(orders))
    pd.DataFrame([dict(model="DNN", seed_index=0, train_variant="classA", eval_variant="classA",
                       site="test", metric=m, order=o, sum=mae[j] * 10 * (1 if m == "full" else 0.5), n=10)
                  for m in ("full", "slack_ref") for j, o in enumerate(orders)]
                 ).to_csv(run / "scores_sites.csv", index=False)
    phys = 2 * mae
    ref = pd.DataFrame([dict(noise="A", placement="deployed", run="baseline", estimator=e, order=o,
                             pu_mae=(phys[j] if e == T.PHYSICS else mae[j] / 2), rel=0.0, site=0)
                        for e in (T.PHYSICS, T.LINEAR) for j, o in enumerate(orders)])
    ref.to_csv(tmp_path / "ref.csv", index=False)
    T.run(type("Args", (), dict(out=str(tmp_path / "out"), runs=[f"DNN={run}"], data=str(adapted),
                                reference=str(tmp_path / "ref.csv"), site=0, placement="deployed",
                                noise="A", eval_variant="classA"))())
    po = pd.read_csv(tmp_path / "out" / "per_order.csv")
    assert np.allclose(po.pu_mae, mae) and np.allclose(po.rel, mae / s["true_mean_pu"])
    row = pd.read_csv(tmp_path / "out" / "summary.csv").iloc[0]
    assert row.h1_vs_physics == pytest.approx(0.5) and row.harm_vs_linear_map == pytest.approx(2.0)
    assert row.orders_better_than_physics == len(orders) - 1 and row.orders_better_than_linear_map == 0


def test_phase_fix_positions_follow_the_order_values():
    import train

    orders = list(ORDERS)
    assert [orders[j] for j, o in enumerate(orders) if o in train.FIX_ORDERS] == [3, 9, 15]
    assert [j for j, o in enumerate(range(1, 21)) if o in train.FIX_ORDERS] == train.FIX_HARMONICS


def test_ct_readings_are_the_branch_currents_with_the_class_a_error(dataset, config):
    cfg, grid, source = P.load_source(dataset, config)
    noise = P.noise_model(cfg, grid, ORDERS, 132)
    obs = [0, 1, 2, 15, 16, 17, 66, 67, 68, 84, 85, 86]
    cur = P.CurrentReadings(grid, ORDERS, noise, obs, "upstream", torch.complex64)
    v, _, i = P.read_split(source.test_dataset(), noise, 0, torch.complex64, cur, 7)
    exact = cur.op(torch.as_tensor(v)).numpy()[..., cur.edges]
    rel = np.abs(i[:, 0] - exact[:, 0]) / np.abs(exact[:, 0])
    assert 1e-5 < np.median(rel) < 5e-3                 # h1: 0.1 % limit, coverage 2
    _, _, i2 = P.read_split(source.test_dataset(), noise, 0, torch.complex64, cur, 7)
    assert np.array_equal(i, i2)


def test_training_data_from_another_root(dataset, config, tmp_path):
    out = tmp_path / "a"
    P.run(type("Args", (), dict(dataset=str(dataset), out=str(out), config=str(config),
                                placement=",".join(P.DEPLOYED), orders="", variant="classA",
                                train_draws=1, test_seed=0, test_subdir="dataset_coherent", train_root=str(dataset),
                                train_config_random="", train_config_coherent=str(config),
                                currents="none"))())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["counts"]["train"]["states_read"] == 8 * 28 and "c_classA" not in np.load(out / "arrays_test.npz")


def test_overlap_finds_every_state_of_a_dataset_in_itself(dataset, config, tmp_path):
    import overlap as O

    O.run(type("Args", (), dict(out=str(tmp_path / "o.json"),
                                eval=f"{dataset}:dataset_coherent:{config}:test",
                                ref=[f"{dataset}:dataset_coherent:{config}"], nearest=True))())
    rep = json.loads((tmp_path / "o.json").read_text())
    n = rep["eval"]["splits"]["test"]["states"]
    test_in_test = rep["refs"][0]["splits"]["test"]["eval_test"]
    assert test_in_test["eval_states_found"] == n
    assert test_in_test["same_index_identical"] == test_in_test["same_index_items"]
    assert rep["refs"][0]["splits"]["train"]["eval_test"]["eval_states_found"] == 0
    near = rep["nearest"]["test"]
    assert near["dataset_coherent test"]["nearest"]["q99"] < 1e-6          # itself
    assert near["dataset_coherent train"]["nearest"]["q00"] > 1e-6         # never shared


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    root = _repo_root()
    out = tmp_path_factory.mktemp("v4pert")
    subprocess.run([sys.executable, str(root / "run" / "examples" / "injection_hse" / "generate_se_task_datasets.py"),
                    str(out), "--scenario-config", "run/configs/pgl_se_scenario_v4.yaml",
                    "--orders", ",".join(map(str, ORDERS)),
                    "--device-capacitance", "run/configs/pgl_device_capacitance.yaml",
                    "--source-impedance", "cigre-tb575", "--load-shunt", "opendss",
                    "--load-shunt-basis", "nameplate", "--n-scenarios", "60", "--n-steps", "8",
                    "--seed", "0", "--skip-random", "--grid-perturbation",
                    "run/configs/pgl_grid_perturbation.yaml", "--perturbation-sets", "nominal,heldout",
                    "--perturb-test-copies", "4"], cwd=root, check=True, capture_output=True)
    return out


def test_held_out_perturbed_test_split_is_scored_per_grid_copy(perturbed, config, tmp_path):
    out = tmp_path / "pert"
    P.run(type("Args", (), dict(dataset=str(perturbed), out=str(out), config=str(config),
                                placement=",".join(P.DEPLOYED), orders="", variant="classA",
                                train_draws=1, test_seed=0, test_subdir="dataset_coherent_heldout",
                                train_root="", train_config_random="", train_config_coherent="",
                                currents="none"))())
    meta = json.loads((out / "meta.json").read_text())
    copies = meta["splits"]["test"]
    assert all(c.startswith("copy") for c in copies) and len(copies) >= 2
    assert set(copies) <= set(meta["grid_copy_rms_log"])
    z, s = np.load(out / "arrays_test.npz"), np.load(out / "scoring.npz")
    assert z["site"].max() == len(copies) - 1
    assert s["true_mean_pu_site"].shape == (len(copies), len(ORDERS))
    # the same scenarios as the nominal test split, solved on other grids
    nominal = tmp_path / "nom"
    P.run(type("Args", (), dict(dataset=str(perturbed), out=str(nominal), config=str(config),
                                placement=",".join(P.DEPLOYED), orders="", variant="classA",
                                train_draws=1, test_seed=0, test_subdir="dataset_coherent",
                                train_root="", train_config_random="", train_config_coherent="",
                                currents="none"))())
    zn = np.load(nominal / "arrays_test.npz")
    assert np.array_equal(zn["interval"], z["interval"])
    d = np.abs(zn["v"] - z["v"]).max() / np.abs(zn["v"]).max()
    assert 1e-5 < d < 0.2
    # per-copy table without a reference
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([dict(model="DNN", seed_index=0, train_variant="classA", eval_variant="classA",
                       site=c, metric="full", order=o, sum=1e-3 * (k + 1) * 8, n=8)
                  for k, c in enumerate(copies) for o in ORDERS]).to_csv(run / "scores_sites.csv", index=False)
    T.run(type("Args", (), dict(out=str(tmp_path / "t"), runs=[f"DNN={run}"], data=str(out), reference="",
                                site=0, placement="deployed", noise="A", eval_variant="classA"))())
    ps = pd.read_csv(tmp_path / "t" / "per_site.csv")
    assert np.allclose(ps[ps.site == copies[1]].pu_mae, 2e-3)
    assert ps.rms_log_factor.notna().all()
    summ = pd.read_csv(tmp_path / "t" / "summary.csv")
    assert summ.h1_pu_mae.iloc[0] == pytest.approx(1e-3 * np.mean(np.arange(1, len(copies) + 1)), rel=0.3)
