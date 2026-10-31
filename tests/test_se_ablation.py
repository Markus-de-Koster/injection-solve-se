"""Tests of the reproduction's own code: experiments, the linear MMSE frame, the evaluation
summaries, the table, the verification join and the vendoring rename."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from se_ablation import experiments as X
from se_ablation import frame as F
from se_ablation.evaluate import (
    ci_from_sequences,
    feeder_combinations,
    sequence_ci,
    summarize_search,
)
from se_ablation.figures import results_table
from se_ablation.names import NAMES, PLACEMENTS, TABLE_ROWS
from se_ablation.verify import compare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_vendored import rename  # noqa: E402


# ----------------------------------------------------------------------------- experiments
@pytest.mark.parametrize("name", X.names())
def test_every_experiment_resolves_to_existing_configs(name):
    exp = X.load(name)
    argv = exp.argv()
    assert exp.driver.is_file()
    for opt in ("--scenario-config", "--device-capacitance", "--grid-perturbation"):
        if opt in argv:
            assert Path(argv[argv.index(opt) + 1]).is_file(), opt
    assert "{configs}" not in " ".join(argv)


def test_smoke_and_extra_arguments_come_last_and_win():
    exp = X.load("gnn_p")
    argv = exp.argv(smoke=True, extra=["--device cpu"])
    last = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert last["--n-random"] == "400" and last["--device"] == "cpu"
    assert exp.argv()[exp.argv().index("--seed") + 1] == "0"


def _opts(argv):
    return {
        argv[i]: (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True)
        for i in range(len(argv))
        if argv[i].startswith("--")
    }


def test_paper_arguments_of_the_experiments():
    assert set(X.names("train")) == {"gnn_p", "gnn_p_r"}
    gnn_p = _opts(X.load("gnn_p").argv())
    assert gnn_p["--prior-residual"] is True and gnn_p["--prior-fit-states"] == "40000"
    assert gnn_p["--carry-optimizer-state"] is True and gnn_p["--stage-warmup"] == "3"
    assert gnn_p["--epochs"] == "28" and gnn_p["--seed"] == "0"
    assert X.load("gnn_p").continuation_argv() == []
    r = X.load("gnn_p_r")
    opts = _opts(r.argv())
    assert opts["--direct-random-fraction"] == "0.1" and "--prior-residual" not in opts
    cont = _opts(r.continuation_argv())
    assert cont == {
        "--epochs": "92",
        "--anneal-epochs": "15",
        "--warmup": "3",
        "--lr": "3e-4",
        "--device": "cuda",
    }
    smoke = _opts(r.continuation_argv(smoke=True, extra=["--device cpu"]))
    assert smoke["--epochs"] == "1" and smoke["--device"] == "cpu"
    ev = _opts(X.load("evaluation").argv())
    assert ev["--perturbation-sets"] == "nominal,heldout" and ev["--perturb-test-copies"] == "8"


def test_training_data_arguments_keep_only_generation_options():
    argv = X.load("gnn_p").data_argv()
    assert "--lr" not in argv and "--measure-currents" not in argv
    last = len(argv) - 1 - argv[::-1].index("--n-random")
    assert argv[last + 1] == "200000"  # argparse keeps the later value of the run


# ----------------------------------------------------------------------------- frame
def test_real_stack_round_trip():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 3, 4)) + 1j * rng.standard_normal((5, 3, 4))
    assert np.array_equal(F.complex_unstack(F.real_stack(z), 3, 4), z)


def test_block_real_is_the_complex_product():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    x = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    y = F.block_real(m) @ np.concatenate([x.real, x.imag])
    np.testing.assert_allclose(y[:3] + 1j * y[3:], m @ x, rtol=1e-13, atol=1e-13)


def test_lmmse_equals_the_gaussian_posterior_mean():
    """Against the information form ``(Σ⁻¹ + HᵀR⁻¹H)⁻¹(Σ⁻¹μ + HᵀR⁻¹y)`` in float64."""
    rng = np.random.default_rng(2)
    d, m, s = 6, 4, 3
    a = rng.standard_normal((d, d))
    sigma = a @ a.T + 0.1 * np.eye(d)
    mu = rng.standard_normal(d)
    frame = type("Frame", (), {"H": rng.standard_normal((m, d))})()
    r = rng.uniform(0.1, 1.0, m)
    y = rng.standard_normal((s, m))
    got = F.lmmse(frame, y, mu, sigma, r)
    h, ri, si = frame.H, np.diag(1 / r), np.linalg.inv(sigma)
    post = np.linalg.inv(si + h.T @ ri @ h)
    want = (post @ (si @ mu[:, None] + h.T @ ri @ y.T)).T
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_sequence_ci_uses_the_sequence_as_the_unit():
    rng = np.random.default_rng(3)
    v_true = rng.standard_normal((5 * 8, 3, 4)) + 0j
    v_est = v_true + rng.uniform(0, 1, (5 * 8, 3, 4))
    ci = sequence_ci(v_est, v_true, n_steps=8)
    seq = np.abs(v_est - v_true).mean(axis=2).reshape(5, 8, 3).mean(axis=1)
    assert ci["h1"][0] == pytest.approx(seq[:, 0].mean())
    assert ci["h1"][1] == pytest.approx(1.96 * seq[:, 0].std(ddof=1) / np.sqrt(5))
    assert ci["harm"][0] == pytest.approx(np.abs(v_est - v_true)[:, 1:].mean())
    assert ci["h1"][2] == 5
    assert sequence_ci(v_est[:7], v_true[:7], n_steps=8) == {}
    assert ci_from_sequences(seq)["harm"][0] == pytest.approx(ci["harm"][0])


# ----------------------------------------------------------------------------- evaluation
@pytest.fixture(scope="module")
def cigre_grid():
    from pgml.convert.pandapower import PhaseMode
    from pgml.grids import cigre_lv_full_grid

    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    return grid


def test_placement_search_covers_one_meter_per_feeder(cigre_grid):
    combos = feeder_combinations(cigre_grid)
    assert len(combos) == 18 * 2 * 20 == 720
    assert sorted(PLACEMENTS["deployed"]) in [sorted(c) for c in combos]
    names = {str(n.name) for n in cigre_grid.nodes}
    assert all(b in names for p in PLACEMENTS.values() for b in p)


def test_search_summary_ranks_and_selects():
    rows = []
    for label, h1, harm in (
        ("Bus R4 Bus I2 Bus C5", 2.0, 2.0),
        ("Bus R6 Bus I2 Bus C4", 1.0, 1.1),
        ("Bus R1 Bus I1 Bus C1", 0.9, 3.0),
    ):
        for order, value in ((1, h1), (3, harm), (5, harm)):
            rows.append(
                {
                    "placement": label,
                    "estimator": "statistics-based",
                    "order": order,
                    "pu_mae": value,
                }
            )
    piv, facts = summarize_search([pd.DataFrame(rows)], PLACEMENTS["deployed"])
    assert facts["lmmse_best"] == "Bus R6 Bus I2 Bus C4"
    assert facts["deployed_rank_h1"] == 3 and facts["deployed_rank_harm"] == 2
    assert len(piv) == 3


def _per_order(run, estimator, values, placement="deployed"):
    return [
        {
            "noise": "A",
            "placement": placement,
            "run": run,
            "estimator": estimator,
            "order": o,
            "pu_mae": v,
            "rel": v / 10,
        }
        for o, v in zip((1, 3, 5), values)
    ]


def test_results_table_values_and_bold_rule(tmp_path):
    rows = []
    for placement in ("deployed", "alternative"):
        rows += (
            _per_order("baseline", "physics solve", [4.0, 2.0, 4.0], placement)
            + _per_order(
                "baseline", "linear map, best decode per order", [1.0, 1.0, 1.0], placement
            )
            + _per_order("baseline", "statistics-based (linear MMSE)", [1.04, 1.2, 1.2], placement)
            + _per_order("gnn_p", "network, as deployed", [2.0, 1.0, 1.0], placement)
        )
    csv = tmp_path / "baseline_table_noiseA_deployed_alternative.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    ci = tmp_path / "ci_noiseA_deployed_alternative.csv"
    pd.DataFrame(
        [
            {
                "noise": "A",
                "placement": "deployed",
                "run": "gnn_p",
                "estimator": "network, as deployed",
                "part": "h1",
                "pu_mae_mean": 2.0,
                "ci95_half_width": 0.1,
                "n_sequences": 200,
            }
        ]
    ).to_csv(ci, index=False)
    pert = tmp_path / "pert.csv"
    prow = _per_order("gnn_p", "network, as deployed", [3.0, 1.0, 1.0]) + _per_order(
        "baseline", "physics solve", [5.0, 2.0, 2.0]
    )
    pd.DataFrame(
        [dict(r, copy="all") for r in prow] + [dict(prow[0], copy="0", pu_mae=99.0)]
    ).to_csv(pert, index=False)
    mack = tmp_path / "mack.csv"
    pd.DataFrame(
        [dict(r, run="DNN") for r in _per_order("x", "DNN [Mack]", [3.0, 1.5, 1.5])]
    ).to_csv(mack, index=False)
    path, tex = results_table([csv, ci], tmp_path, perturbed_csvs=[pert], mack_csv=mack)
    t = pd.read_csv(path).set_index("key")
    assert t.loc["gnn_p", "deployed_h1"] == pytest.approx(2e4)
    assert t.loc["gnn_p", "deployed_h1_ci95"] == pytest.approx(1e3)
    assert t.loc["gnn_p", "perturbed_h1"] == pytest.approx(3e4)  # pooled, not the copy
    assert t.loc["mack_dnn", "deployed_harm"] == pytest.approx(1.5e4)
    assert np.isnan(t.loc["mack_dnn", "alternative_h1"])
    assert t.loc["gnn_p", "rel_h3"] == pytest.approx(10.0)
    # best 1.0 (Linear-P); LMMSE at 1.04 is within 5 %, GNN-P at 2.0 is not
    assert t.loc["linear", "bold_deployed_h1"] and t.loc["lmmse", "bold_deployed_h1"]
    assert not t.loc["gnn_p", "bold_deployed_h1"]
    assert list(t.index) == [k for k, *_ in TABLE_ROWS]
    assert t.loc["gnn_p", "estimator"] == NAMES["gnn_p"]
    body = tex.read_text()
    assert "\\textbf{10000$^{*}$}" in body  # refitted at the alternative placement
    assert "Physics & 40000 & 30000" in body


def test_verify_joins_on_mapped_runs_and_placements():
    ours = pd.DataFrame(
        _per_order("gnn_p", "network, as deployed", [2.0, 1.0, 1.0], placement="alternative")
    )
    rec = pd.DataFrame(
        _per_order("OLD", "network, as deployed", [2.0, 1.0, 1.1], placement="alt_recorded")
    )
    m = compare(ours, rec, {"gnn_p": "OLD"}, {"alt_recorded": "alternative"}, site=None)
    assert len(m) == 3
    assert m.rel_diff.max() == pytest.approx(0.1 / 1.1)


# ----------------------------------------------------------------------------- vendoring
def test_rename_rule():
    src = (
        "from pgl.data import x\nimport pgl\n'pgl'\nrun/examples/pgl/a.py\n"
        "run/configs/pgl_se_scenario_v4.yaml\npgl_config\npower-grid-learn\ntests/pgl/b\n"
    )
    out = rename(src)
    assert out == (
        "from injection_hse.data import x\nimport injection_hse\n'injection_hse'\n"
        "run/examples/injection_hse/a.py\nrun/configs/pgl_se_scenario_v4.yaml\n"
        "pgl_config\npower-grid-learn\ntests/injection_hse/b\n"
    )


def test_no_vendored_module_imports_a_removed_one():
    removed = (
        "injection_hse.inference",
        "injection_hse.analysis.multigrid",
        "injection_hse.analysis.ablation",
        "injection_hse.analysis.feasibility",
        "injection_hse.analysis.headroom",
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, injection_hse, injection_hse.analysis, injection_hse.evaluation, "
            "injection_hse.train, injection_hse.workflow, injection_hse.curriculum, "
            "injection_hse.data.scenarios; print(','.join(sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    loaded = set(out.stdout.strip().split(","))
    assert not loaded & set(removed)
    assert not any(m == "pgl" or m.startswith("pgl.") for m in loaded)


# ----------------------------------------------------------------------------- end to end
@pytest.mark.slow
def test_smoke_training_run_writes_a_checkpoint(tmp_path):
    """GNN-P-R at smoke size on the CPU, with its continuation (a few minutes)."""
    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "se_ablation.reproduce",
            "train",
            "--experiment",
            "gnn_p_r",
            "--smoke",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    assert (out / "checkpoints" / "stage_placement_anneal.ckpt").is_file()
    assert (out / "training_history.parquet").is_file()
    assert (out / "continued" / "checkpoints" / "stage_placement_anneal_reanneal.ckpt").is_file()
