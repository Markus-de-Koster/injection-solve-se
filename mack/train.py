"""Train and score one Mack et al. (2024) estimator on the arrays of ``adapter.py``.

Runs in the estimators' own environment (TensorFlow 2.13, their pinned versions) and uses
their code unchanged (github.com/th-koeln-iet/pqse_concept_pann, ``--src`` the clone): the
experiment configurations (``experiments.exp_{dnn,cnn,pann}_config``: Adam, batch 16 384,
leaky ReLU, skip connections, the triangular ``CyclicLR`` with min ``lr/5``, max ``2·lr``,
step 200 batches), the model classes and their ``train`` (``model.fit`` on min-max scaled
arrays, the best-``val_loss`` checkpoint reloaded at the end), the preprocessing
(the phase fix at the orders 3, 6, ..., 18 the data carry, see :func:`fix_phase_angles`; the input split into Re, Im,
``|V|·sign(cos φ)``, ``sin φ``; the target into Re, Im; ``DimMinMaxScaler`` per order and
channel over samples and buses, separate input and target scalers fitted on the training
split; ``convert_y_mats`` for the PANN's admittances).

What differs from their CIGRE LV experiment, and why:

- the data: the arrays of ``adapter.py``, one sample per state and phase; inputs the
  meters of the placement plus the slack bus, read clean or with the instrument's reading
  errors (``--input-variant``); the input of a meter replaces the true voltage at that bus
  before the phase fix, which then sees the array their code sees;
- the phase fix: their ``fix_phase_angle_problem`` can raise ``IndexError`` on such arrays (a
  handful of target voltages below 1e-9 V), so :func:`fix_phase_angles` applies the operation
  its docstring describes; it moves those entries by less than 2e-9 V;
- the PANN depth: their rule "one layer per hop of the meter set's eccentricity" (6 for
  their meters) gives 7 for this placement (``meta.json``: ``pann_layers``);
- the base learning rate 5e-5 of the repository's configuration (``--lr``), with its CyclicLR;
- the budget in optimizer steps (``--max-steps``) and wall-clock hours (``--max-train-hours``,
  checked after every epoch), and a plateau stop on the validation
  sites (no improvement of the best ``val_loss`` by >= 1 % within ``--patience-steps``,
  checked after ``--min-steps``); the validation loss is evaluated once per pass over the
  training split, as their ``fit`` does;
- the best-``val_loss`` checkpoint is kept in host memory and written every
  ``--checkpoint-every`` epochs (the same weights as their ``ModelCheckpoint(save_best_only)``,
  including its initial threshold 1e-2);
- the split-complex conversion is cast to float32 chunk by chunk and scaled in float32 (the
  Keras model computes in float32 and casts the float64 arrays it is given; the scaled values
  differ by float32 rounding only), which bounds the host memory of large training splits;
- ``--weights-from`` scores an earlier run's best checkpoint on ``--test-data`` without
  training (the scalers are refitted on the same training arrays, so they are the run's own);
- ``--with-currents`` (DNN only): the CT readings of the arrays follow the voltage meters on
  the node axis of the input, split like the voltages and min-max scaled with a scaler of
  their own; the DNN flattens its input, so only the first layer widens.

Writes into ``--run-dir``: ``history.csv`` (per epoch: loss, val_loss, lr), ``result.json``
(configuration, steps, best epoch, scaled losses, timings), ``scores_sites.csv`` (per test
site, order, input variant and metric of :mod:`scoring`: the summed per-interval error and
the interval count) and ``pred_h1_<variant>.npz`` (the h1 predictions of every test
sample, volts referred to the slack level, each phase in its own frame).

Usage::

    python -I train.py --src <clone> --data <adapter out> --run-dir <run> --model {DNN,CNN,PANN}
        [--seed-index 0] [--lr 5e-5] [--max-steps 20000] [--max-train-hours 0] [--min-steps 6000]
        [--patience-steps 4000] [--input-variant clean] [--eval-variants clean,pqida,class0.1]
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
#: Harmonic orders of their phase-angle fix (their 0-based positions 2, 5, ..., 17 of h1..h20).
FIX_ORDERS = (3, 6, 9, 12, 15, 18)
#: Positions of :data:`FIX_ORDERS` on the order axis h1..h20 of their data.
FIX_HARMONICS = [2, 5, 8, 11, 14, 17]
#: Initial threshold of their ModelCheckpoint (a checkpoint is written only below it).
CHECKPOINT_THRESHOLD = 1e-2


def fix_phase_angles(arr, harmonics=FIX_HARMONICS, threshold=1e-9, angle_cutoff=0.4):
    """Their ``fix_phase_angle_problem`` on ``[S, B, H]`` complex arrays, indexed as documented.

    Per listed order (0-based), where ``|V| < threshold`` and the phase lies more than
    ``angle_cutoff`` rad from the order's most frequent phase (``scipy.stats.mode`` over all
    entries), the phase is replaced by that mode; magnitudes are kept. Their implementation
    looks the extracted near-zero phases up by sample index instead of by position, which
    fails (``IndexError``) when there are fewer near-zero entries than the largest affected
    sample index; this is the operation their docstring describes. Modifies ``arr`` in place
    and returns it, as theirs does; returns the number of replaced phases as well.
    """
    import numpy as np
    from scipy import stats

    n = 0
    for f in harmonics:
        h = arr[..., f]
        mag, phase = np.abs(h), np.angle(h)
        mode = stats.mode(phase.ravel())[0]
        sel = (mag < threshold) & (np.abs(phase - mode) > angle_cutoff)
        if sel.any():
            phase[sel] = mode
            arr[..., f] = mag * np.exp(1j * phase)
            n += int(sel.sum())
    return arr, n


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="root of the pqse_concept_pann clone")
    ap.add_argument("--data", required=True, help="output directory of adapter.py")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True, choices=["DNN", "CNN", "PANN"])
    ap.add_argument("--seed-index", type=int, default=0,
                    help="seeds numpy 1+10k, tensorflow 2+10k, random 3+10k (k=0: their seeds)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="base learning rate of the CyclicLR (the repository's configuration: 5e-5)")
    ap.add_argument("--max-train-hours", type=float, default=0.0,
                    help="if > 0: stop training after the epoch that exceeds this wall-clock time")
    ap.add_argument("--max-steps", type=int, default=20000, help="optimizer step budget")
    ap.add_argument("--min-steps", type=int, default=6000, help="no plateau stop before this many steps")
    ap.add_argument("--patience-steps", type=int, default=4000,
                    help="stop when the best val_loss has not improved by >= 1 %% within this many steps")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="epochs between checkpoint writes")
    ap.add_argument("--input-variant", default="clean", help="clean, pqida or class0.1")
    ap.add_argument("--eval-variants", default="clean,pqida,class0.1")
    ap.add_argument("--weights-from", default="",
                    help="score only: rebuild the scalers and the model from --data as in training, "
                    "load the best checkpoint of this earlier run directory and skip training")
    ap.add_argument("--test-data", default="",
                    help="arrays directory whose test split is scored (default --data)")
    ap.add_argument("--with-currents", action="store_true",
                    help="DNN only: the CT readings of the arrays (c_<variant>) widen the input")
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="if > 0: a seeded subset of the training samples (smoke runs)")
    return ap.parse_args()


def main():
    args = parse_args()
    src = os.path.abspath(args.src)
    sys.path.insert(0, os.path.join(src, "src"))
    sys.path.insert(0, os.path.join(src, "src", "pqse_concept_pann"))
    sys.path.insert(0, HERE)

    import numpy as np
    import tensorflow as tf

    import experiments as E  # their module; sets the seeds 1 / 2 / 3 at import
    from preprocessing import convert_y_mats, mask_data, scale_data
    from tools import DimMinMaxScaler, SplitComplexConverter, SplitComplexMode

    import scoring as S

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    data_orders = [int(o) for o in meta["orders"]]
    fix_positions = [j for j, o in enumerate(data_orders) if o in FIX_ORDERS]
    meters = list(meta["meter_index"])
    n_bus = len(meta["buses"])

    k = args.seed_index
    np.random.seed(1 + 10 * k)
    tf.random.set_seed(2 + 10 * k)
    random.seed(3 + 10 * k)

    modes = [SplitComplexMode.CARTESIAN, SplitComplexMode.EXPONENTIAL]
    input_converter = SplitComplexConverter(modes, target_axis=3)
    target_converter = SplitComplexConverter(modes[0], target_axis=3)
    fix_counts = {}

    def prepare(v_bus, x_meter, tag):
        """Their read_network_data on one split: phase fix on the full array, meters masked."""
        v = v_bus.reshape(-1, n_bus, v_bus.shape[-1])            # [Q*3, B, H]
        x = x_meter.reshape(-1, len(meters), x_meter.shape[-1])
        full = v.copy()
        full[:, meters] = x
        full, n_in = fix_phase_angles(full, harmonics=fix_positions)
        x_in = convert32(input_converter, mask_data(full, meters, 1))
        del full
        y, n_out = fix_phase_angles(v.copy(), harmonics=fix_positions)
        counts = fix_counts.setdefault(tag, {"near_zero": 0, "replaced_inputs_array": 0, "replaced_targets": 0})
        counts["near_zero"] += int(sum((np.abs(v[..., f]) < 1e-9).sum() for f in fix_positions))
        counts["replaced_inputs_array"] += n_in
        counts["replaced_targets"] += n_out
        return x_in, convert32(target_converter, y)

    def convert32(converter, z, chunk=65536):
        """Their split-complex conversion, chunk by chunk into float32 (their float64 result
        rounded once; the model computes in float32)."""
        first = converter.convert(z[:1])
        out = np.empty((z.shape[0],) + first.shape[1:], dtype=np.float32)
        for a in range(0, z.shape[0], chunk):
            out[a: a + chunk] = converter.convert(z[a: a + chunk])
        return out

    def prepare_currents(c_bus):
        """CT readings ``[Q, 3, K, H]`` -> their input split ``[Q·3, K, H, 4]`` float32."""
        return convert32(input_converter, c_bus.reshape(-1, c_bus.shape[2], c_bus.shape[-1]))

    t0 = time.time()
    tr = np.load(os.path.join(args.data, "arrays_train.npz"))
    va = np.load(os.path.join(args.data, "arrays_val.npz"))
    v_tr, x_tr = tr["v"], tr[f"x_{args.input_variant}"]
    if args.max_train_samples:
        pick = np.sort(np.random.default_rng(k).choice(v_tr.shape[0], args.max_train_samples // 3, replace=False))
        v_tr, x_tr = v_tr[pick], x_tr[pick]
    x_train, y_train = prepare(v_tr, x_tr, "train")
    del v_tr, x_tr
    x_val, y_val = prepare(va["v"], va[f"x_{args.input_variant}"], "val")
    current_scaler = None
    if args.with_currents:
        if args.model != "DNN":
            raise SystemExit("--with-currents widens the flat input of the DNN only.")
        c_tr = tr[f"c_{args.input_variant}"]
        if args.max_train_samples:
            c_tr = c_tr[pick]
        current_scaler = DimMinMaxScaler()
        c_train = current_scaler.fit_transform(prepare_currents(c_tr), (0, 1))
        c_val = current_scaler.transform(prepare_currents(va[f"c_{args.input_variant}"]))
        del c_tr
    adm = np.load(os.path.join(args.data, "admittance.npz"))
    freqs = [int(round(f)) for f in adm["freqs_hz"]]
    network_data = {
        "grid_name": "corpus", "scaler": DimMinMaxScaler(), "data_format": "channels_last",
        "mask_x_test": meters, "mask_y_test": None, "split_complex_modes": modes,
        "input_converter": input_converter, "target_converter": target_converter,
        "frequencies": freqs,
        "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val,
        # their scale_data and model classes expect a test split; the validation split stands in
        "x_test": x_val, "y_test": y_val,
        "input_dims": x_train.shape[1:], "output_dims": y_train.shape[1:],
        "Y_matrices_per_frequency": convert_y_mats(dict(zip(freqs, adm["Y"])), data_format="channels_last",
                                                   modes=input_converter.modes),
    }
    scale_data(network_data, axes=(0, 1), sep_output_scaler=True)
    for key in ("x_train_n", "y_train_n", "x_val_n", "y_val_n", "x_test_n", "y_test_n"):
        network_data[key] = network_data[key].astype(np.float32, copy=False)
    if current_scaler is not None:
        # the CT channels follow the voltage meters on the node axis, each quantity min-max
        # scaled with its own scaler (their scaler pools the node axis; volts and amperes
        # would share one range otherwise); the DNN flattens the widened input
        network_data["x_train_n"] = np.concatenate([network_data["x_train_n"], c_train], axis=1)
        network_data["x_val_n"] = np.concatenate([network_data["x_val_n"], c_val], axis=1)
        network_data["x_test_n"] = network_data["x_val_n"]
        network_data["input_dims"] = network_data["x_train_n"].shape[1:]
        del c_train, c_val
    for key in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"):
        network_data[key] = np.zeros((1,) + network_data[key].shape[1:], dtype=np.float32)
    del x_train, y_train, x_val, y_val
    t_load = time.time() - t0

    E.data_path = os.path.join(run_dir, "keras")
    E.GRID = "corpus"
    cfg = {"DNN": E.exp_dnn_config, "CNN": E.exp_cnn_config, "PANN": E.exp_pann_config}[args.model]
    hyperparams, callbacks, model_class, custom_layer, exp_path, label = cfg()
    if args.model == "PANN":
        hyperparams["num_hidden_layers"] = int(meta["pann_layers"])
    n_train = network_data["x_train_n"].shape[0]
    bs = hyperparams["batch_size"]
    steps_per_epoch = math.ceil(n_train / bs)
    epochs = math.ceil(args.max_steps / steps_per_epoch)
    hyperparams["epochs"] = epochs
    hyperparams["learning_rate"] = args.lr
    assert sum(isinstance(c, E.CyclicLR) for c in callbacks) == 1
    ckpt = [c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint)]
    assert len(ckpt) == 1
    ckpt_path = ckpt[0].filepath
    callbacks = [E.CyclicLR(min_lr=args.lr / 5, max_lr=args.lr * 2, step_size=200.)]

    t_epoch = []

    class EpochTimer(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.time()

        def on_epoch_end(self, epoch, logs=None):
            t_epoch.append(time.time() - self._t)

    class BestInMemory(tf.keras.callbacks.Callback):
        """ModelCheckpoint(save_best_only, monitor=val_loss, initial_value_threshold) in memory."""

        def __init__(self):
            super().__init__()
            self.best, self.best_w, self.best_epoch, self.dirty = CHECKPOINT_THRESHOLD, None, -1, False

        def on_epoch_end(self, epoch, logs=None):
            v = logs.get("val_loss")
            if v is not None and v < self.best:
                self.best, self.best_w, self.best_epoch, self.dirty = v, self.model.get_weights(), epoch, True
            if (epoch + 1) % args.checkpoint_every == 0 and self.dirty:
                self._write()

        def _write(self):
            cur = self.model.get_weights()
            self.model.set_weights(self.best_w)
            self.model.save_weights(ckpt_path)
            self.model.set_weights(cur)
            self.dirty = False

        def on_train_end(self, logs=None):
            if self.dirty:
                self._write()

    patience_epochs = math.ceil(args.patience_steps / steps_per_epoch)
    min_epochs = math.ceil(args.min_steps / steps_per_epoch)
    stop_info = {}

    class RelPlateauStop(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_epoch = np.inf, 0

        def on_epoch_end(self, epoch, logs=None):
            v = logs.get("val_loss")
            if v is not None and v < 0.99 * self.best:
                self.best, self.best_epoch = v, epoch
            if epoch + 1 >= min_epochs and epoch - self.best_epoch >= patience_epochs:
                stop_info.update(epoch=epoch + 1, reason="plateau", best=float(self.best),
                                 best_epoch=self.best_epoch + 1)
                self.model.stop_training = True

    class TimeBudget(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if args.max_train_hours > 0 and time.time() - self._t0 > 3600.0 * args.max_train_hours:
                stop_info.update(epoch=epoch + 1, reason="max_train_hours")
                self.model.stop_training = True

    best_keeper = BestInMemory()
    callbacks += [best_keeper, RelPlateauStop(), TimeBudget(),
                  tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")), EpochTimer()]

    if model_class == E.PANN:
        m = model_class(hyperparams, network_data, callbacks, custom_layer=custom_layer)
    else:
        m = model_class(hyperparams, network_data, callbacks)
    model = m.create_model()
    n_params = int(model.count_params())
    info = dict(model=args.model, label=label, seed_index=k,
                seeds=dict(numpy=1 + 10 * k, tensorflow=2 + 10 * k, random=3 + 10 * k),
                input_variant=args.input_variant, with_currents=args.with_currents, lr=args.lr, hyperparams=hyperparams,
                n_train_samples=int(n_train), n_val_samples=int(network_data["x_val_n"].shape[0]),
                steps_per_epoch=steps_per_epoch, max_epochs=epochs, max_steps=args.max_steps,
                min_steps=args.min_steps, patience_steps=args.patience_steps,
                max_train_hours=args.max_train_hours, orders=data_orders, phase_fix_orders=[
                    data_orders[j] for j in fix_positions],
                phase_fix_near_zero_entries=fix_counts, n_params=n_params,
                input_dims=list(network_data["input_dims"]), output_dims=list(network_data["output_dims"]),
                tf_version=tf.__version__, gpus=[d.name for d in tf.config.list_physical_devices("GPU")],
                time_load_s=t_load)
    print(json.dumps(info, default=str), flush=True)

    t1 = time.time()
    if args.weights_from:
        src_ckpt = tf.train.latest_checkpoint(os.path.join(
            os.path.abspath(args.weights_from), "keras", "corpus", label, "weights"))
        if src_ckpt is None:
            raise SystemExit(f"no checkpoint under {args.weights_from}.")
        model.load_weights(src_ckpt).expect_partial()
        hist = {"loss": [], "val_loss": []}
        info["weights_from"] = src_ckpt
    else:
        model, hist = m.train(model, exp_path, load_weights=False)
    t_train = time.time() - t1
    val_loss = [float(v) for v in hist.get("val_loss", [])]
    best_epoch = best_keeper.best_epoch + 1
    val_best = float(model.evaluate(network_data["x_val_n"], network_data["y_val_n"], batch_size=bs, verbose=0))

    # ------------------------------------------------------------------ test sites
    test_dir = args.test_data or args.data
    te = np.load(os.path.join(test_dir, "arrays_test.npz"))
    sc = np.load(os.path.join(test_dir, "scoring.npz"))
    test_meta = json.load(open(os.path.join(test_dir, "meta.json")))
    if test_meta["meter_index"] != meters or test_meta["orders"] != data_orders:
        raise SystemExit("the test arrays have other meters or orders than the training arrays.")
    rows, base, v_ref = sc["rows"], sc["base"], float(sc["v_ref"])
    unobs, slack_rows, g_s = sc["unobs"], sc["slack_rows"], sc["g_s"]
    orders = [int(o) for o in sc["orders"]]
    rot = sc["phase_rotation"][None, :, None, :]  # [1, 3, 1, H]: the adapter's phase frame
    n_rows = base.size
    to_phys = (base / v_ref)[None, None, :]
    site, sites = te["site"], test_meta["splits"]["test"]
    in_scaler, out_scaler = network_data["input_scaler"], network_data["scaler"]
    score_rows, test_mse = [], {}
    t2 = time.time()
    for variant in [s for s in args.eval_variants.split(",") if f"x_{s}" in te
                    and (current_scaler is None or f"c_{s}" in te)]:
        se, cnt = 0.0, 0
        h1 = []
        for kk in range(len(sites)):
            sel = np.flatnonzero(site == kk)
            v_bus = te["v"][sel]
            x_in, y_true = prepare(v_bus, te[f"x_{variant}"][sel], f"test_{variant}")
            y_n = out_scaler.transform(y_true).astype(np.float32)
            x_n = in_scaler.transform(x_in).astype(np.float32)
            if current_scaler is not None:
                c_n = current_scaler.transform(prepare_currents(te[f"c_{variant}"][sel]))
                x_n = np.concatenate([x_n, c_n.astype(np.float32)], axis=1)
            y_pred_n = model.predict(x_n, batch_size=bs, verbose=0)
            se += float(((y_n - y_pred_n) ** 2).sum())
            cnt += y_n.size
            y_pred = out_scaler.revert(y_pred_n.astype(np.float64))
            v_hat_bus = (y_pred[..., 0] + 1j * y_pred[..., 1]).reshape(v_bus.shape) / rot
            h1.append(v_hat_bus[..., 0].astype(np.complex64))
            v_hat = S.to_rows(v_hat_bus, rows, n_rows) * to_phys
            v_true = S.to_rows(v_bus.astype(np.complex128) / rot, rows, n_rows) * to_phys
            err = S.interval_errors(v_hat, v_true, g_s, slack_rows, unobs, base)
            score_rows += S.site_rows(err, np.zeros(sel.size, dtype=int), [sites[kk]], orders,
                                      model=args.model, seed_index=k, train_variant=args.input_variant,
                                      eval_variant=variant)
        test_mse[variant] = se / cnt
        np.savez_compressed(os.path.join(run_dir, f"pred_h1_{variant}.npz"), v_hat_h1=np.concatenate(h1),
                            site=site, interval=te["interval"])
    t_test = time.time() - t2
    with open(os.path.join(run_dir, "scores_sites.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)

    epochs_run = len(val_loss)
    info["test_data"] = os.path.abspath(test_dir)
    out = dict(info, epochs_run=epochs_run, steps_run=epochs_run * steps_per_epoch,
               best_epoch=best_epoch, steps_at_best=best_epoch * steps_per_epoch,
               val_mse_best=val_best, val_loss_min=min(val_loss) if val_loss else None,
               plateau_stop=stop_info or None, test_mse_scaled=test_mse,
               phase_fix_near_zero_entries=fix_counts,
               val_loss_history=val_loss, train_loss_history=[float(v) for v in hist.get("loss", [])],
               time_train_s=t_train, time_epoch_median_s=float(np.median(t_epoch)) if t_epoch else None,
               time_test_s=t_test)
    with open(os.path.join(run_dir, "result.json"), "w") as fh:
        json.dump(out, fh, default=str)
    print("RESULT", json.dumps({key: out[key] for key in ("epochs_run", "steps_run", "best_epoch", "val_mse_best",
                                                          "test_mse_scaled", "time_train_s")}), flush=True)


if __name__ == "__main__":
    main()
