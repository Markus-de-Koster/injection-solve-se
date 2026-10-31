"""GPU gate: the injection-decode estimator + differentiable solve run + match on CUDA.

Regression guard for the device sync: the ``NetworkSolver`` is a plain object, so Lightning
does not move its cached ``Y(h)`` / slack tensors with the module — the module must relocate
them. This builds the injection-decode ``SEModule``, runs a step on CPU and on CUDA, and asserts
the reconstructed voltages match (skips cleanly without CUDA).
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

ORDERS = [1, 5, 7]


def _grid():
    from pgml.convert.pandapower import PhaseMode
    from pgml.evaluation.oracles import cigre_lv_full_grid
    from pgml.geometry.synthesis import synthesize_grid_geometry

    warnings.filterwarnings("ignore")
    grid, _ = cigre_lv_full_grid(phase_mode=PhaseMode.THREE_PHASE)
    synthesize_grid_geometry(grid)
    return grid


def _module_and_batch(grid, dataset_dir):
    import lightning as L

    from injection_hse.config import (
        CurriculumConfig,
        DataConfig,
        ExperimentConfig,
        GNNConfig,
        LossConfig,
        MaskStageConfig,
        MeasurementConfig,
        PhysicsConfig,
    )
    from injection_hse.data import base_voltage_per_row, build_data_source
    from injection_hse.encoding import build_encoding
    from injection_hse.normalization import HarmonicNormalizer
    from injection_hse.physics import NetworkSolver, fit_injection_normalizer
    from injection_hse.data import StateCollate
    from injection_hse.train import SEModule

    source = build_data_source(
        DataConfig(source="parquet", dataset_path=str(dataset_dir)), grid=grid
    )
    orders = list(source.orders)
    enc = build_encoding("trig")
    tv = torch.stack([source.train_dataset()[i]["v"] for i in range(16)])
    norm = HarmonicNormalizer().fit(enc.encode(tv).transpose(-3, -2))
    solver = NetworkSolver.from_grid(grid, orders)
    inj_norm = fit_injection_normalizer(solver, tv, enc)
    cfg = ExperimentConfig(
        name="inj",
        seed=0,
        orders=orders,
        data=DataConfig(dataset_path="x"),
        model=GNNConfig(),
        physics=PhysicsConfig(decode="injection"),
        loss=LossConfig(),
        measurement=MeasurementConfig(voltage_nodes=["Bus R4"], slack_known=True),
        curriculum=CurriculumConfig(
            stages=[
                MaskStageConfig(name="f", kind="full", target_loss=1e-9, max_epochs=1)
            ]
        ),
    )
    L.seed_everything(0, workers=True)
    module = SEModule(
        cfg,
        n_nodes=source.n_nodes,
        normalizer=norm,
        graph=source.graph(),
        base_v=base_voltage_per_row(grid),
        solver=solver,
        inj_normalizer=inj_norm,
    )
    batch = StateCollate()([source.train_dataset()[i] for i in range(8)])
    return module, batch


def test_injection_decode_cpu_cuda_parity(tmp_path):
    from pgml.scenarios import (
        ParameterSpec,
        ScenarioConfig,
        Selector,
        Uniform,
        run_scenarios,
        write_dataset,
    )

    grid = _grid()
    injected = [h for h in ORDERS if h > 1]
    cfg = ScenarioConfig(
        n_samples=24,
        seed=0,
        method="sobol",
        parameters=[
            ParameterSpec(
                name="load_scale",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.5, high=1.0),
                field="pq",
                mode="scale",
                per="each",
            ),
            ParameterSpec(
                name="load_spectrum",
                selector=Selector(component="load"),
                distribution=Uniform(low=0.0, high=1.0),
                field="h_mag",
                orders=injected,
                harmonic_reference="en50160",
            ),
        ],
    )
    result = run_scenarios(
        grid,
        cfg,
        calculation="harmonic",
        harmonic_orders=ORDERS,
        symmetry="auto",
        dtype=torch.complex64,
    )
    ds = write_dataset(result, tmp_path / "ds", layout="wide")

    # CPU reference
    module, batch = _module_and_batch(grid, ds)
    with torch.no_grad():
        out_cpu = module.infer(batch)["pred_v"]

    # CUDA: the solver's cached Y/slack must follow the module to the GPU
    module_c, batch_c = _module_and_batch(grid, ds)
    module_c = module_c.to("cuda")
    batch_c = {
        k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch_c.items()
    }
    with torch.no_grad():
        out_cuda = module_c.infer(batch_c)["pred_v"]

    assert out_cuda.is_cuda
    torch.testing.assert_close(out_cuda.cpu(), out_cpu, rtol=1e-4, atol=1e-4)
