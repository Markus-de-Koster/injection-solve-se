"""The decode frame every scored estimator shares: placement, anchored solve, linear maps.

All estimators of the study output nodal injections and are turned into voltages by the
same network solve of a trained checkpoint's module (its grid, its admittance per order,
its reading-noise model, its injection scale ``s_I(h)``). The helpers here load such a
module on an evaluation dataset, set a meter placement, run the module over a split while
capturing every input of its decode, re-decode arbitrary injections through the anchored
solve, and build the linear pieces of the statistics-based estimator:

- the anchored decode is affine in the injection at a fixed reading, so its injection
  Jacobian ``G[h] = dV/dI`` follows exactly from unit injections against an all-zero
  reading (:func:`injection_jacobian`);
- with the plain (slack-only) decode ``V = G·I + b`` and the branch-current operator, the
  readings are ``y = H·I + b_y + n`` (:class:`Frame`); with the injection mean ``μ`` and
  covariance ``Σ`` the linear MMSE estimate is
  ``Î = μ + Σ·Hᵀ·(H·Σ·Hᵀ + R)⁻¹·(y − b_y − H·μ)`` (:func:`lmmse`).

Shapes: states ``[S, H, N]`` complex128 (``S`` states, ``H`` orders, ``N`` node-phase
rows), branch currents ``[S, H, E]`` (``E`` directed half-edges), the real-stacked
injection layout ``[S, H·2F]`` (per order: the real block, then the imaginary block over the
``F`` free rows). Everything runs on the CPU in float64 / complex128.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: seed of the reading noise of the scored split (voltages); the currents use seed + 7
NOISE_SEED = 0
#: batch size of the passes over a split
BATCH = 16
#: the deployed placement: one meter per feeder (plus the always-metered supply)
DEPLOYED = ["Bus R4", "Bus C5", "Bus I2"]


# ----------------------------------------------------------------------------- module
def load_grid(root: Path):
    """The ``grid.json`` of a dataset or run directory."""
    from pgml.schemas.grid_schema import Grid

    return Grid.model_validate_json((Path(root) / "grid.json").read_text())


def load_module(grid, ckpt_path: Path, dataset: Path, sensors=DEPLOYED):
    """A checkpoint's module on ``dataset`` (CPU, no loader workers) at ``sensors``.

    Returns ``(module, source, [v_weight, i_weight])``: the data source carries the
    checkpoint's own split settings (fractions, seed), so its test split is the one every
    estimator is scored on; the anchor weights are those the checkpoint deploys.
    """
    from injection_hse.config import ExperimentConfig
    from injection_hse.data import build_data_source
    from injection_hse.train import SEModule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["pgl_config"])
    data_cfg = cfg.data.model_copy(update={"dataset_path": str(dataset), "num_workers": 0})
    source = build_data_source(data_cfg, grid=grid)
    module = SEModule.from_checkpoint(ckpt_path, grid=grid, device="cpu", source=source)
    _install_placement(module, grid, sensors)
    return module, source, [float(w) for w in ckpt["anchor_weights"]]


def _install_placement(module, grid, sensors):
    """The checkpoint's placement as loaded: CT channels only when the run measured them."""
    from injection_hse.data import MeasurementModel
    from injection_hse.masking import build_masker

    mm = MeasurementModel.resolve(grid, sensors, slack_known=True)
    module.set_masker(
        build_masker(
            "placement",
            observed_rows=mm.observed_rows(grid),
            observed_edge_rows=(
                mm.observed_edges(grid) if module.cfg.measurement.measure_current else None
            ),
        )
    )


def set_placement(module, grid, sensors):
    """Meter ``sensors`` (bus names) plus the slack, each with its upstream CT channels."""
    from injection_hse.data import MeasurementModel
    from injection_hse.masking import build_masker

    mm = MeasurementModel.resolve(grid, sensors, slack_known=True)
    module.set_masker(
        build_masker(
            "placement",
            observed_rows=mm.observed_rows(grid),
            observed_edge_rows=mm.observed_edges(grid),
        )
    )


def swap_noise(module, grid, kind: str):
    """Replace the module's reading noise by ``kind`` (``"A"``: the class-A instrument, else
    an accuracy class such as ``"0.1"``) on the voltage rows and the CT channels, keeping the
    module's noise seed."""
    from injection_hse.config import InstrumentErrorConfig, NoiseConfig
    from injection_hse.data import InstrumentNoise, SensorNoise, grid_current_range

    seed = int(module.cfg.noise.seed)
    if kind == "A":
        cfg = NoiseConfig(enabled=True, seed=seed, instrument=InstrumentErrorConfig.class_a())
        module._noise = InstrumentNoise.from_config(
            cfg, n_nodes=module.n_nodes, grid=grid, orders=module.orders
        )
    else:
        cfg = NoiseConfig(enabled=True, default_class=kind, seed=seed)
        module._noise = SensorNoise.from_config(
            cfg, n_nodes=module.n_nodes, grid=grid, orders=module.orders
        )
    if module._edge_noise is not None:
        module._edge_noise = module._noise.for_edges(
            module._edge_src_rows, current_range_a=grid_current_range(grid)
        )
    module.cfg = module.cfg.model_copy(update={"noise": cfg})


def noise_kind(module) -> str:
    """``"A"`` for the class-A instrument model, else the module's accuracy class."""
    return "A" if module.cfg.noise.instrument is not None else str(module.cfg.noise.default_class)


# ----------------------------------------------------------------------------- anchors
def use_noise_calibrated(module, v_mult: float = 1.0, i_mult: float = 1.0):
    """Switch the module's solver to noise-calibrated anchor weights (multipliers v / i)."""
    from injection_hse.train.anchoring import install_anchor_calibration

    physics = module.cfg.physics.model_copy(update={"anchor_weighting": "noise_calibrated"})
    module.cfg = module.cfg.model_copy(update={"physics": physics})
    if not install_anchor_calibration(module):
        raise RuntimeError("the module's solver does not accept an anchor calibration")
    module.set_stage_anchor(v_weight=v_mult, i_weight=i_mult)


def use_manual(module, v_weight: float, i_weight: float):
    """Fixed anchor weights (``0, 0``: the plain, slack-only decode)."""
    for solver in module._solvers:
        solver.clear_anchor_calibration()
    physics = module.cfg.physics.model_copy(update={"anchor_weighting": "manual"})
    module.cfg = module.cfg.model_copy(update={"physics": physics})
    module.set_stage_anchor(v_weight=v_weight, i_weight=i_weight)


def anchor_sigmas(module):
    """``(sigma_injection [H], sigma_voltage [H, N|1], sigma_current [H, E|1])`` of the
    noise-calibrated decode at a relative injection error ``ρ = 1``."""
    from injection_hse.train.anchoring import (
        _per_unit_calibration_bases,
        anchor_calibration_sigmas,
    )

    physics = module.cfg.physics.model_copy(update={"prior_injection_rel_sigma": 1.0})
    inj = module.inj_normalizer
    return anchor_calibration_sigmas(
        physics,
        state_scale=module.normalizer.cartesian_scale,
        injection_scale=inj.aux_scale if inj.aux_scale is not None else inj.cartesian_scale,
        edge_scale=module.edge_normalizer.cartesian_scale,
        voltage_noise=module._noise,
        current_noise=module._edge_noise,
        **(_per_unit_calibration_bases(module) or {}),
    )


def install(module, s_inj, s_v, s_i, v_mult, i_mult):
    """Install explicit sigmas (injection ``[H]``, voltage ``[H, N]``, current ``[H, E]``)."""
    for solver in module._solvers:
        solver.set_anchor_calibration(sigma_injection=s_inj, sigma_voltage=s_v, sigma_current=s_i)
    module.set_stage_anchor(v_weight=v_mult, i_weight=i_mult)


def _accuracy_class_sigma(noise, typical_hr: np.ndarray) -> torch.Tensor:
    """``[H, R]`` per-component sigma of an accuracy-class noise model for channels of the
    typical magnitude ``typical_hr``: ``σ² = ½(σ_mag² + σ_phase²)·|x|² + (σ_floor·floor_h)²``."""
    mag = noise.mag_sigma.double().cpu().numpy().reshape(1, -1)
    ph = noise.phase_sigma_rad.double().cpu().numpy().reshape(1, -1)
    var = 0.5 * (mag**2 + ph**2) * typical_hr**2
    floor = noise.floor_sigma
    if floor is not None:
        f = floor.double().cpu().numpy()
        f = f.reshape(1, -1) if f.ndim < 2 else f
        scale = noise.order_floor_scale
        if scale is not None:
            f = f * scale.double().cpu().numpy().reshape(-1, 1)
        var = var + f**2
    return torch.as_tensor(np.sqrt(var))


def channel_sigma(noise, typical_hr) -> torch.Tensor:
    """Per-channel sigma ``[H, R]`` at each channel's own typical magnitude: the instrument
    model's own per-reading sigma, else the accuracy-class formula."""
    if hasattr(noise, "reading_sigma"):
        return noise.reading_sigma(torch.as_tensor(np.asarray(typical_hr), dtype=torch.float64))
    return _accuracy_class_sigma(noise, typical_hr)


def sensor_sigmas(module, data):
    """``(s_v [H, N], s_i [H, E])``: each channel's sigma at its RMS magnitude on ``data``."""
    v_rms = np.sqrt((np.abs(data["v"].numpy()) ** 2).mean(0))
    i_rms = np.sqrt(
        (
            np.abs(module._derive_edge_currents(data["v"].to(torch.complex128), None).numpy()) ** 2
        ).mean(0)
    )
    return (
        channel_sigma(module._noise, v_rms).numpy(),
        channel_sigma(module._edge_noise, i_rms).numpy(),
    )


# ----------------------------------------------------------------------------- rows
def placement_masks(module):
    """``(mask [N] bool, edge_mask [E] bool)`` of the module's fixed placement (slack included)."""
    lead = (1,)
    mask, order_mask = module._draw_mask(lead, torch.device("cpu"))
    if order_mask is not None:
        raise ValueError("an order mask is not expected at a fixed placement")
    v0 = torch.zeros((1, module.n_harmonics, module.n_nodes), dtype=module._complex_dtype)
    _, edge_mask, _, _ = module._edge_observations(
        v0, mask, None, lead, torch.device("cpu"), v_true=v0
    )
    return mask[0].clone(), (None if edge_mask is None else edge_mask[0].clone())


def row_info(module, grid) -> dict:
    """The row sets of the placement: observed / unobserved / slack / tied / free rows, the
    per-row line-to-neutral base voltage and the masks."""
    from pgml.assembly import node_phase_index

    from injection_hse.data import base_voltage_per_row

    idx = node_phase_index(grid)
    n = int(idx.size)
    mask, edge_mask = placement_masks(module)
    slack = module._slack_rows.cpu().numpy().astype(int)
    tied = module._tied_rows.cpu().numpy().astype(int)
    free = np.setdiff1d(np.arange(n), np.union1d(slack, tied))
    obs = np.flatnonzero(mask.numpy())
    unobs = np.flatnonzero(~mask.numpy())
    base_v = base_voltage_per_row(grid).numpy().astype(float)
    node_ids = idx.node_ids.cpu().numpy().astype(int)
    names = {int(nd.id): str(nd.name) for nd in grid.nodes}
    labels = [f"{names.get(int(node_ids[r]), node_ids[r])}" for r in range(n)]
    return dict(
        n=n,
        mask=mask,
        edge_mask=edge_mask,
        slack=slack,
        tied=tied,
        free=free,
        obs=obs,
        unobs=unobs,
        base_v=base_v,
        labels=labels,
    )


def injection_jacobian(module, rows: dict) -> np.ndarray:
    """``G [H, N, N]`` complex128: ``G[h, r, j] = dV_r / dI_j`` (volts per ampere) of the
    module's CURRENT decode (anchored at the placement when its solver anchors)."""
    n_h, n = module.n_harmonics, rows["n"]
    solver = module.solver
    old_dtype = solver._dtype
    solver._dtype = torch.complex128  # keep the solve's complex128 result (no downcast)
    try:
        b = n + 1
        ib = torch.zeros((b, n_h, n), dtype=torch.complex128)
        for j in range(n):
            ib[j + 1, :, j] = 1.0
        v0 = torch.zeros((b, n_h, n), dtype=torch.complex128)
        mask = rows["mask"].unsqueeze(0).expand(b, n).clone()
        e_mask = (
            None
            if rows["edge_mask"] is None
            else rows["edge_mask"].unsqueeze(0).expand(b, -1).clone()
        )
        i_edges = (
            None
            if e_mask is None
            else torch.zeros((b, n_h, e_mask.shape[-1]), dtype=torch.complex128)
        )
        with torch.no_grad():
            v = (
                module._decode_state(
                    ib,
                    v_in=v0,
                    mask=mask,
                    order_mask=None,
                    i_edges=i_edges,
                    edge_mask=e_mask,
                    edge_order_mask=None,
                    topo=None,
                )
                .to(torch.complex128)
                .numpy()
            )
    finally:
        solver._dtype = old_dtype
    g = v[1:] - v[0:1]  # [N(cols), H, N(rows)]
    return np.ascontiguousarray(g.transpose(1, 2, 0))  # [H, rows, cols]


# ----------------------------------------------------------------------------- passes
def _prior_injection(module, i_pred, kw):
    """The checkpoint's linear-prior injections for one batch (zeros without a prior)."""
    if getattr(module, "injection_prior", None) is None:
        return torch.zeros_like(i_pred)
    with torch.no_grad():
        pf = module._injection_features(
            module._prior_residual_injection(kw["v_in"], kw["i_edges"], kw["mask"], kw["edge_mask"])
        )
        pf = module._pin_tied_features(pf)
        return module._zero_tied(
            module._from_injection_frame(
                module.inj_encoding.decode(module.inj_normalizer.inverse_transform(pf)).transpose(
                    -2, -1
                )
            )
        )


def _collect(module, dataset):
    """Run the module over ``dataset``; capture the reading and every injection.

    The reading noise is reseeded first (:data:`NOISE_SEED`, currents ``+ 7``), so every
    estimator scored on a split sees the same noise draw. Returns, flattened over the
    sequence steps, ``v`` (true state), ``v_in`` (noisy reading), ``mask``, ``i_edges``
    (noisy CT reading), ``edge_mask``, ``i_net`` (the module's injections), ``i_prior``
    (its linear prior's injections, zeros without one) and ``pred_v`` (its decode).
    """
    from torch.utils.data import DataLoader

    from injection_hse.data import StateCollate

    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=StateCollate()
    )
    if module._noise is not None:
        module._noise.reseed(NOISE_SEED)
    if getattr(module, "_edge_noise", None) is not None:
        module._edge_noise.reseed(NOISE_SEED + 7)

    captured = {}
    orig = module._decode_state

    def spy(i_pred, **kw):
        captured.setdefault("calls", []).append((i_pred.detach().clone(), kw))
        return orig(i_pred, **kw)

    module._decode_state = spy
    out = {
        k: [] for k in ("v", "v_in", "mask", "i_edges", "edge_mask", "i_net", "i_prior", "pred_v")
    }
    try:
        for batch in loader:
            captured.clear()
            res = module.infer(batch)
            # the first call is the deployed (anchored) decode; a run with the plain-decode
            # term decodes the same injections a second time without anchors
            i_pred, kw = captured["calls"][0]
            v = batch["v"].to(torch.complex128)
            i_prior = _prior_injection(module, i_pred, kw)
            lead = v.shape[:-2]
            flat = lambda t: t.reshape(-1, *t.shape[len(lead) :])  # noqa: E731
            out["v"].append(flat(v))
            out["v_in"].append(flat(kw["v_in"].to(torch.complex128)))
            out["mask"].append(flat(kw["mask"]))
            out["i_edges"].append(flat(kw["i_edges"].to(torch.complex128)))
            out["edge_mask"].append(flat(kw["edge_mask"]))
            out["i_net"].append(flat(i_pred.to(torch.complex128)))
            out["i_prior"].append(flat(i_prior.to(torch.complex128)))
            out["pred_v"].append(flat(res["pred_v"].to(torch.complex128)))
    finally:
        module._decode_state = orig
    return {k: torch.cat(v_) for k, v_ in out.items()}


def collect(module, source, split: str):
    """:func:`_collect` on the ``"train"``, ``"val"`` or ``"test"`` split of ``source``."""
    datasets = {
        "train": source.train_dataset,
        "val": source.val_dataset,
        "test": source.test_dataset,
    }
    return _collect(module, datasets[split]())


def decode(module, inj, data, *, clean=False, chunk=512):
    """The module's current decode (its anchor weights) of injections ``[S, H, N]`` against
    the readings in ``data`` (``clean``: the true state and its currents instead)."""
    solver = module.solver
    old = solver._dtype
    solver._dtype = torch.complex128
    parts = []
    try:
        with torch.no_grad():
            for a in range(0, inj.shape[0], chunk):
                sl = slice(a, a + chunk)
                if clean:
                    v_rd = data["v"][sl]
                    i_rd = module._derive_edge_currents(v_rd.to(torch.complex128), None)
                else:
                    v_rd, i_rd = data["v_in"][sl], data["i_edges"][sl]
                parts.append(
                    module._decode_state(
                        inj[sl],
                        v_in=v_rd,
                        mask=data["mask"][sl],
                        order_mask=None,
                        i_edges=i_rd,
                        edge_mask=data["edge_mask"][sl],
                        edge_order_mask=None,
                        topo=None,
                    ).to(torch.complex128)
                )
    finally:
        solver._dtype = old
    return torch.cat(parts).numpy()


def true_injection(module, v):
    """``Y·V`` of true states ``[S, H, N]`` (tied rows zeroed), complex128 numpy."""
    solver = module.solver
    old = solver._dtype
    solver._dtype = torch.complex128
    try:
        with torch.no_grad():
            return module._zero_tied(solver.injection_targets(v)).to(torch.complex128).numpy()
    finally:
        solver._dtype = old


def states(dataset):
    """Every state of a split, flattened over the sequence steps: ``[S, H, N]`` complex128."""
    v = torch.stack([dataset[i]["v"] for i in range(len(dataset))])
    return v.reshape(-1, *v.shape[-2:]).to(torch.complex128)


# ----------------------------------------------------------------------------- linear MMSE
def real_stack(z):
    """complex ``[S, H, F]`` -> real ``[S, H*2F]`` (per order: Re block, Im block)."""
    return np.concatenate([z.real, z.imag], axis=-1).reshape(z.shape[0], -1)


def complex_unstack(x, h, f):
    """The inverse of :func:`real_stack`: real ``[S, H*2F]`` -> complex ``[S, H, F]``."""
    x = x.reshape(x.shape[0], h, 2 * f)
    return x[..., :f] + 1j * x[..., f:]


def block_real(m):
    """complex ``[M, F]`` -> real ``[[Re, -Im], [Im, Re]]`` ``[2M, 2F]``."""
    return np.block([[m.real, -m.imag], [m.imag, m.real]])


def _block_diag(blocks):
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols))
    r = c = 0
    for b in blocks:
        out[r : r + b.shape[0], c : c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


class Frame:
    """The plain-decode map ``G``, the measurement map ``H`` and ``R`` at one placement.

    Building the frame switches the module's solver to the plain (slack-only) decode.
    """

    def __init__(self, module, rows, data):
        use_manual(module, 0.0, 0.0)
        self.G = injection_jacobian(module, rows)  # [H, N, N] plain decode
        self.free = rows["free"]
        self.orders = list(module.orders)
        slack = set(int(r) for r in rows["slack"])
        mask = data["mask"][0].numpy()
        self.vrows = np.array([r for r in np.flatnonzero(mask) if r not in slack])
        self.erows = np.flatnonzero(data["edge_mask"][0].numpy())
        g_states = torch.as_tensor(self.G.transpose(2, 0, 1))  # [Ncols, H, Nrows]
        ge = module._derive_edge_currents(g_states, None).numpy()  # [Ncols, H, E]
        self.Ge = ge.transpose(1, 2, 0)  # [H, E, Ncols]
        f = self.free
        blocks = [
            block_real(np.vstack([self.G[k][self.vrows][:, f], self.Ge[k][self.erows][:, f]]))
            for k in range(len(self.orders))
        ]
        self.H = _block_diag(blocks)  # [Hm, Hx]
        self.module = module

    def readings(self, data):
        """``(y - b_y)`` real ``[S, Hm]`` and the offset ``b`` ``[S, H, N]`` of the plain decode."""
        zero = torch.zeros_like(data["i_net"])
        b = decode(self.module, zero, data)  # plain decode of zero injection
        be = self.module._derive_edge_currents(torch.as_tensor(b), None).numpy()
        y_v = data["v_in"].numpy()[:, :, self.vrows] - b[:, :, self.vrows]
        y_e = data["i_edges"].numpy()[:, :, self.erows] - be[:, :, self.erows]
        y = np.concatenate([y_v, y_e], axis=-1)  # [S, H, M]
        return real_stack(y), b

    def noise_cov(self, s_v, s_i):
        """Diagonal ``R`` (per real component) from the per-channel sigmas ``[H, N]`` / ``[H, E]``."""
        diag = []
        for k in range(len(self.orders)):
            sk = np.concatenate([s_v[k, self.vrows], s_i[k, self.erows]])
            diag.append(np.concatenate([sk, sk]) ** 2)
        return np.maximum(np.concatenate(diag), 1e-24)

    def voltages(self, i_hat, b):
        """``V̂ = G·Î + b`` from free-row injections complex ``[S, H, F]``."""
        v = b.copy()
        for k in range(len(self.orders)):
            v[:, k, :] += i_hat[:, k, :] @ self.G[k][:, self.free].T
        return v


def lmmse(frame, y, mu, sigma, r_diag):
    """The linear MMSE injection estimate (real-stacked ``[S, H·2F]``) from the readings
    ``y`` (offset removed), the prior mean ``mu`` / covariance ``sigma`` and the diagonal
    noise covariance ``r_diag``."""
    H = frame.H
    s = H @ sigma @ H.T + np.diag(r_diag)
    k = np.linalg.solve(s, H @ sigma).T  # Σ Hᵀ S⁻¹ (S symmetric)
    return mu + (y - mu @ H.T) @ k.T
