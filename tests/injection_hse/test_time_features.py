"""Periodic time features + their plumbing into the models and the SEModule.

Covers the feature math (hand-computed sin/cos for known UTC timestamps), the validity-flag
semantics for timeless data, the model input width / byte-identity when the feature is OFF,
the timeless zero-fill contract, and the SEModule end-to-end wiring (time actually conditions
the estimate when ON; is fully ignored when OFF).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import torch

from injection_hse.config import (
    CurriculumConfig,
    DataConfig,
    DNNConfig,
    EncodingConfig,
    ExperimentConfig,
    GNNConfig,
    GraphormerConfig,
    MaskStageConfig,
)
from injection_hse.data import base_voltage_per_row, build_graph
from injection_hse.encoding import TrigEncoding
from injection_hse.io import build_model_input
from injection_hse.models import build_model
from injection_hse.normalization import HarmonicNormalizer
from injection_hse.time_features import TIME_FEATURE_DIM, time_features
from injection_hse.train import SEModule

ORDERS = [1, 3, 5, 7, 9, 11, 13]


def _epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _epoch_t(*isos: str) -> torch.Tensor:
    # float64 is REQUIRED: an epoch (~1.7e9 s) truncated to float32 loses ~100 s of
    # precision, which shifts the sub-hour phases. The library keeps time_unix_s in float64.
    return torch.tensor([_epoch(iso) for iso in isos], dtype=torch.float64)


# =============================================================================
# feature math (hand-computed against the calendar)
# =============================================================================
def test_time_features_shape_and_valid_flag():
    t = _epoch_t("2024-06-21T12:00:00", "2024-01-01T00:00:00")
    feats = time_features(t)
    assert feats.shape == (2, TIME_FEATURE_DIM) == (2, 7)
    # the validity flag is the last channel, 1 for a real timestamp
    assert torch.allclose(feats[:, -1], torch.ones(2, dtype=torch.float64))


def test_time_of_day_noon_and_midnight():
    # noon UTC -> tod = 0.5 -> angle = pi -> (sin, cos) = (0, -1)
    noon = time_features(_epoch_t("2024-06-21T12:00:00"))[0]
    assert math.isclose(noon[0].item(), 0.0, abs_tol=1e-9)
    assert math.isclose(noon[1].item(), -1.0, abs_tol=1e-9)
    # midnight UTC -> tod = 0 -> angle = 0 -> (sin, cos) = (0, 1)
    midnight = time_features(_epoch_t("2024-01-01T00:00:00"))[0]
    assert math.isclose(midnight[0].item(), 0.0, abs_tol=1e-9)
    assert math.isclose(midnight[1].item(), 1.0, abs_tol=1e-9)


def test_day_of_week_matches_calendar():
    # cross-check the epoch-offset weekday arithmetic against Python's calendar,
    # INDEPENDENTLY: 2024-06-21 is a Friday (weekday() == 4), so at 12:00 the continuous
    # day-of-week index is 4.5 -> the sin/cos of 2*pi*4.5/7.
    iso = "2024-06-21T12:00:00"
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    assert dt.weekday() == 4  # Friday (Monday = 0)
    dow_cont = dt.weekday() + (dt.hour + dt.minute / 60.0) / 24.0  # 4.5
    ang = 2.0 * math.pi * dow_cont / 7.0
    feats = time_features(_epoch_t(iso))[0]
    assert math.isclose(feats[2].item(), math.sin(ang), abs_tol=1e-9)
    assert math.isclose(feats[3].item(), math.cos(ang), abs_tol=1e-9)


def test_day_of_year_is_a_periodic_unit_circle_point():
    # the seasonal channel uses a fixed 365.25-day epoch frame (matching the profile
    # generator); assert it is a valid unit-circle point and matches the documented scalar
    # formula computed independently with python floats.
    seconds_per_day = 86400.0
    seconds_per_year = 365.25 * seconds_per_day
    e = _epoch("2024-09-15T06:00:00")
    doy = (e % seconds_per_year) / seconds_per_year
    ang = 2.0 * math.pi * doy
    feats = time_features(torch.tensor([e], dtype=torch.float64))[0]
    assert math.isclose(feats[4].item(), math.sin(ang), abs_tol=1e-9)
    assert math.isclose(feats[5].item(), math.cos(ang), abs_tol=1e-9)
    assert math.isclose(feats[4].item() ** 2 + feats[5].item() ** 2, 1.0, abs_tol=1e-9)


# =============================================================================
# model input width + byte-identity when disabled
# =============================================================================
def _graph():
    import torch_geometric.data as pyg

    n, e = 6, 8
    return pyg.Data(
        x=torch.randn(n, 8),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, 5),
    )


def test_model_input_width_unchanged_when_disabled():
    f, n = 3 * 2, 6  # H*C, node-phase rows
    # DNN: N*(F+1) with no time, N*(F+1+Ftime) with time
    d0 = build_model(DNNConfig(hidden_dims=[16]), n_features=f, n_nodes=n)
    d7 = build_model(
        DNNConfig(hidden_dims=[16]), n_features=f, n_nodes=n, n_time_features=7
    )
    assert d0.mlp[0].in_features == n * (f + 1)
    assert d7.mlp[0].in_features == n * (f + 1 + 7)
    # GNN / Graphormer: F+1+Fns (+Ftime)
    g0 = build_model(
        GNNConfig(conv="sage", hidden_dim=16, num_layers=1), n_features=f, n_nodes=n
    )
    g7 = build_model(
        GNNConfig(conv="sage", hidden_dim=16, num_layers=1),
        n_features=f,
        n_nodes=n,
        n_time_features=7,
    )
    assert g0.encoder[0].in_features == f + 1 + 8
    assert g7.encoder[0].in_features == f + 1 + 8 + 7
    q0 = build_model(
        GraphormerConfig(hidden_dim=16, num_layers=1, num_heads=2),
        n_features=f,
        n_nodes=n,
    )
    q7 = build_model(
        GraphormerConfig(hidden_dim=16, num_layers=1, num_heads=2),
        n_features=f,
        n_nodes=n,
        n_time_features=7,
    )
    assert q0.input_proj.in_features == f + 1 + 8
    assert q7.input_proj.in_features == f + 1 + 8 + 7


def test_timeless_none_equals_explicit_zeros():
    # a time-ENABLED model fed no timestamp (node_time=None) must equal the same model fed an
    # explicit all-zero time vector (valid flag = 0) — the timeless contract.
    torch.manual_seed(0)
    f, n = 3 * 2, 6
    g = _graph()
    m = build_model(
        GNNConfig(conv="gine", hidden_dim=16, num_layers=2),
        n_features=f,
        n_nodes=n,
        n_time_features=7,
    )
    m.eval()
    b = 4
    obs = torch.randn(b, n, f)
    mask = torch.ones(b, n, dtype=torch.bool)
    with torch.no_grad():
        out_none = m(build_model_input(g, obs, mask, has_time=False))
        out_zero = m(
            build_model_input(g, obs, mask, has_time=False, node_time=torch.zeros(b, 7))
        )
    assert torch.equal(out_none, out_zero)


def test_time_features_shift_the_output():
    # a real (non-zero) time vector must change the estimate of a time-enabled model.
    torch.manual_seed(0)
    f, n = 3 * 2, 6
    g = _graph()
    m = build_model(
        GNNConfig(conv="gine", hidden_dim=16, num_layers=2),
        n_features=f,
        n_nodes=n,
        n_time_features=7,
    )
    m.eval()
    b = 4
    obs = torch.randn(b, n, f)
    mask = torch.ones(b, n, dtype=torch.bool)
    nt = time_features(
        torch.full((b,), _epoch("2024-06-21T12:00:00"), dtype=torch.float64)
    ).to(torch.float32)
    with torch.no_grad():
        out_zero = m(build_model_input(g, obs, mask, has_time=False))
        out_time = m(build_model_input(g, obs, mask, has_time=False, node_time=nt))
    assert not torch.allclose(out_zero, out_time)


# =============================================================================
# ModelInput plumbing (snapshot lift + time-fold)
# =============================================================================
def test_model_input_lifts_and_folds_node_time():
    g = _graph()
    n, f = 6, 6
    b = 3
    obs = torch.randn(b, n, f)
    mask = torch.ones(b, n, dtype=torch.bool)
    nt = torch.randn(b, 7)
    mi = build_model_input(g, obs, mask, has_time=False, node_time=nt)
    assert mi.node_time.shape == (b, 1, 7)  # snapshot lifted to T=1
    folded, lead = mi.fold_time_into_batch()
    assert lead == (b, 1)
    assert folded.node_time.shape == (b * 1, 7)


# =============================================================================
# SEModule end-to-end wiring
# =============================================================================
def _module(cigre_grid, v, *, time_features_on: bool):
    cfg = ExperimentConfig(
        name="tf",
        seed=0,
        orders=ORDERS,
        data=DataConfig(source="parquet", dataset_path="x", batch_size=4),
        model=DNNConfig(hidden_dims=[32]),
        encoding=EncodingConfig(time_features=time_features_on),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(
                    name="m",
                    kind="random",
                    observed_fraction=0.5,
                    target_loss=1e-9,
                    max_epochs=1,
                )
            ]
        ),
    )
    g = build_graph(cigre_grid)
    feats = TrigEncoding().encode(v).transpose(-3, -2)  # [*lead, N, H, C]
    norm = HarmonicNormalizer().fit(feats)
    return SEModule(
        cfg,
        n_nodes=g.num_nodes,
        normalizer=norm,
        graph=g,
        base_v=base_voltage_per_row(cigre_grid),
    )


def _seq_batch(n, b=4, t=3):
    v = (torch.randn(b, t, len(ORDERS), n) + 1j * torch.randn(b, t, len(ORDERS), n)).to(
        torch.complex64
    )
    return v


def test_semodule_disabled_ignores_time(cigre_grid):
    # time OFF: a batch carrying time_unix_s must be byte-identical to one without it.
    n = build_graph(cigre_grid).num_nodes
    v = _seq_batch(n)
    tu = torch.full((v.shape[1],), _epoch("2024-06-21T12:00:00"), dtype=torch.float64)
    mod = _module(cigre_grid, v, time_features_on=False)
    assert mod._n_time_features == 0
    state = mod._mask_gen.get_state()
    out_no = mod.infer({"v": v})["pred_v"]
    mod._mask_gen.set_state(state)
    out_tu = mod.infer({"v": v, "time_unix_s": tu})["pred_v"]
    assert torch.equal(out_no, out_tu)


def test_semodule_enabled_conditions_on_time(cigre_grid):
    # time ON: two different absolute times (same v + same mask) yield different estimates.
    n = build_graph(cigre_grid).num_nodes
    v = _seq_batch(n)
    mod = _module(cigre_grid, v, time_features_on=True)
    assert mod._n_time_features == 7
    tu_noon = torch.full(
        (v.shape[1],), _epoch("2024-06-21T12:00:00"), dtype=torch.float64
    )
    tu_mid = torch.full(
        (v.shape[1],), _epoch("2024-01-01T00:00:00"), dtype=torch.float64
    )
    state = mod._mask_gen.get_state()
    out_noon = mod.infer({"v": v, "time_unix_s": tu_noon})["pred_v"]
    mod._mask_gen.set_state(state)
    out_mid = mod.infer({"v": v, "time_unix_s": tu_mid})["pred_v"]
    assert out_noon.shape == v.shape
    assert not torch.allclose(out_noon, out_mid)


def test_semodule_enabled_timeless_snapshot_runs(cigre_grid):
    # time ON but a TIMELESS snapshot batch (no time_unix_s): the model fills zeros
    # (valid flag = 0) — the SAME input contract, so training across both is possible.
    n = build_graph(cigre_grid).num_nodes
    b = 4
    v = (torch.randn(b, len(ORDERS), n) + 1j * torch.randn(b, len(ORDERS), n)).to(
        torch.complex64
    )
    mod = _module(cigre_grid, v, time_features_on=True)
    loss, mask, pred_v, target_v, inj_loss, phys_loss, _ = mod._step(
        {"v": v}, train=True
    )
    assert torch.isfinite(loss) and loss.requires_grad
    assert pred_v.shape == (b, len(ORDERS), n)
