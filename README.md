# Injection-domain learned state estimation for harmonic distribution networks: reproduction

This repository regenerates the data, trains the paper's networks, evaluates every
estimator and redraws every data figure and the results table of the paper
*Injection-Domain Learned State Estimation for Harmonic Distribution Networks: Accuracy and
Feature Attribution*. It needs nothing besides this folder and the published power-flow
package [`pgml`](https://github.com/power-grid-ml/pgml) (`power-grid-ml` 0.6.0, pinned to
its release commit `21fcafc`); the Mack et al. (2024) baselines run with their own published
code in their own environment ([mack/README.md](mack/README.md)).

Everything the paper does not state (the grid model, the full data recipe, the instrument
model, the decode of every estimator, every hyperparameter and stopping rule, the
evaluation protocol, hardware and run times) is in [DETAILS.md](DETAILS.md).

## Estimators

| name | what | where |
|---|---|---|
| Physics | the anchored decode of zero injections (weighted least squares with zero-injection pseudo-measurements) | `se_ablation.evaluate` |
| Linear-P [ours] | ridge regression from all readings to all nodal injections, fitted on labelled states of the site at the placement | `se_ablation.evaluate` |
| LMMSE [ours] | linear MMSE estimate from the injection mean and covariance and the exact measurement model | `se_ablation.frame`, `se_ablation.evaluate` |
| GNN-P [ours] | temporal GNN (GINE + GRU) predicting a correction to the linear map, trained at the deployed placement | `experiments/gnn_p.yaml` |
| GNN-P-R [ours] | the temporal GNN without the linear start, trained on random sensor sets, continued and re-annealed | `experiments/gnn_p_r.yaml` |
| DNN / DNN+I / PANN [Mack] | the published estimators of Mack et al. (2024), their code and schedule, on GNN-P's training data | `mack/` |

## Layout

| path | what |
|---|---|
| `src/injection_hse/` | the estimator code: data generation recipes, instrument model, graph networks, the anchored injection decode, training, evaluation and attribution helpers (a subset of the study's code base, see [Vendored code](#vendored-code)) |
| `run/examples/injection_hse/` | the drivers: `train_se_full_workflow.py` (generate a run's data, train, evaluate the run) and `generate_se_task_datasets.py` (data only) |
| `run/configs/` | the scenario recipe (`pgl_se_scenario_v4.yaml`), the device capacitances, the grid perturbations, the dataset validation ranges |
| `experiments/` | one YAML per experiment: the exact driver arguments of the paper run (and of the continuation), and the smoke sizes |
| `src/se_ablation/` | the paper's evaluation (table, perturbed grids, meter sweep, placement search), the continuation, the attribution, the figures, checkpoint verification and the command-line entry |
| `mack/` | the adapter that runs the published Mack et al. (2024) estimators on this data, in their own environment |
| `tests/` | the unit tests of the vendored code (`tests/injection_hse`, `tests/differentiability`, `tests/gpu`) and of this repository's own code (`tests/test_se_ablation.py`) |
| `tools/check_vendored.py` | the static equivalence check of the vendored files |

## Install

[pixi](https://pixi.sh) sets up the environment from `pixi.toml` / `pixi.lock` (Python 3.13,
PyTorch, Lightning, PyTorch Geometric; `pgml` from GitHub at the pinned commit):

```bash
pixi install -e cpu       # no GPU: data generation, evaluation, attribution, figures, tests
pixi install              # CUDA 12 (the default environment): training at the paper's sizes
pixi run -e cpu test      # the test suite
```

## One entry point

```bash
pixi run reproduce list                                     # the experiments
pixi run reproduce data  --experiment evaluation [--smoke]
pixi run reproduce train --experiment gnn_p|gnn_p_r [--smoke] [--device cpu]
pixi run reproduce evaluate --experiment baselines   --data <data> gnn_p=<ckpt> gnn_p_r=<ckpt>
pixi run reproduce evaluate --experiment perturbed   --data <data> gnn_p=<ckpt> gnn_p_r=<ckpt>
pixi run reproduce evaluate --experiment meter_sweep --data <data> gnn_p=<ckpt> gnn_p_r=<ckpt>
pixi run reproduce evaluate --experiment placement_search --data <data> --frame <ckpt> --shard i/n
pixi run reproduce attribution --checkpoint <ckpt> --data <data>
pixi run reproduce figures --figure parity|meter_sweep|coupling|table ...
pixi run reproduce mack convert|train|score|table ...
pixi run reproduce verify gnn_p=<ckpt> --data <data> --recorded <per-order csv>
```

(`pixi run reproduce` is `python -m se_ablation.reproduce`; with the CPU environment,
`pixi run -e cpu reproduce`.) Outputs go to `outputs/<command>/<experiment>` unless `--out`
is given. `--smoke` runs an experiment at a size that finishes on a CPU in minutes (400
snapshots, 40 sequences, one epoch per pass); `--device` and `--seed` override the run's
values; further driver options follow `--`. The evaluation scores the networks under the
names `gnn_p` and `gnn_p_r`; the table and the figures look them up by these names.

The training experiments (`experiments/*.yaml`):

| experiment | estimator | run |
|---|---|---|
| `gnn_p` | GNN-P [ours] | driver run with the linear start (`--prior-residual --prior-fit-states 40000`), the optimizer state carried across the passes, 3-epoch warm-up per pass; final checkpoint `<run>/checkpoints/stage_placement_anneal.ckpt` |
| `gnn_p_r` | GNN-P-R [ours] | driver run on random sensor sets (`--direct-random-fraction 0.1`), then the continuation in its last pass (`se_ablation.continuation`); final checkpoint `<run>/continued/checkpoints/stage_placement_anneal_reanneal.ckpt` |

Each training run generates its own data (200 000 snapshots, 20 000 sequences of 8 steps,
seed 0), trains, and evaluates itself on its own splits.

## Paper → commands

Let `D=outputs/data/evaluation_site0` (the evaluation data), `T=outputs/train`,
`P=$T/gnn_p/checkpoints/stage_placement_anneal.ckpt` (GNN-P) and
`R=$T/gnn_p_r/continued/checkpoints/stage_placement_anneal_reanneal.ckpt` (GNN-P-R).

| paper | commands |
|---|---|
| Fig. 1 (CIGRE LV one-line diagram), Fig. 2 (injection-domain estimation) | `figures/diagrams/*.graphml`, see [Diagrams](#diagrams) |
| evaluation data (the table and every figure) | `reproduce data --experiment evaluation --out $D` |
| the two networks | `reproduce train --experiment gnn_p`, `reproduce train --experiment gnn_p_r` |
| Table II, deployed and alternative placement, relative error per order | `reproduce evaluate --experiment baselines --data $D --out outputs/evaluate/baselines gnn_p=$P gnn_p_r=$R` |
| Table II, perturbed grid copies | `reproduce evaluate --experiment perturbed --data $D --out outputs/evaluate/perturbed gnn_p=$P gnn_p_r=$R` |
| Table II, Mack et al. rows | `reproduce mack convert --data $D --train-run $T/gnn_p`; `reproduce mack train --model dnn` (and `dnn_i`, `pann`); `reproduce mack table`; perturbed: `reproduce mack convert --data $D --train-run $T/gnn_p --test-subdir dataset_coherent_heldout`, `reproduce mack score --model <m> --arrays outputs/mack/arrays_dataset_coherent_heldout` for each model, `reproduce mack table --arrays outputs/mack/arrays_dataset_coherent_heldout` |
| Table II itself | `reproduce figures --figure table outputs/evaluate/baselines/baseline_table_noiseA_deployed_alternative.csv outputs/evaluate/baselines/ci_noiseA_deployed_alternative.csv --perturbed outputs/evaluate/perturbed/baseline_table_noiseA_deployed.csv --mack outputs/mack/table/per_order.csv --mack-perturbed outputs/mack/table_arrays_dataset_coherent_heldout/per_order.csv` (writes `table2.csv` and the LaTeX rows `table2.tex`) |
| the alternative placement (buses 7, 22, 27: the LMMSE-best of the 720 one-meter-per-feeder placements) | `reproduce evaluate --experiment placement_search --data $D --frame $P --shard i/12 --out outputs/evaluate/search` for `i = 0..11`, then the same with `--summarize` |
| Fig. 3 (parity, GNN-P, h1/h5/h9) | `reproduce figures --figure parity --checkpoint $P --data $D` (keeps the points; `reproduce figures --figure parity outputs/figures/parity_points.parquet` redraws) |
| Fig. 4 (meter sweep) | `reproduce evaluate --experiment meter_sweep --data $D gnn_p=$P gnn_p_r=$R`, then `reproduce figures --figure meter_sweep outputs/evaluate/meter_sweep/placement_sweep_noiseA_k0_1_2_3_4_5_6.csv` |
| Fig. 5 and Sec. VII (order coupling of GNN-P-R through network and solve; the h1 attribution share from h3–h19, the occlusion changes, completeness) | `reproduce attribution --checkpoint $R --data $D`, then `reproduce figures --figure coupling outputs/attribution/attr_harmonic_coupling_solve.csv`; the quoted numbers are in `outputs/attribution/h1_error_sign_check.csv` and `completeness.csv` |

The meter sweep accepts `--ks` (a comma list of meter counts, `0` = the deployed placement)
to split it over several jobs; the figure takes all the CSVs.

## Re-scoring trained checkpoints

`reproduce verify` scores given checkpoints (and the baselines in the frame of the first)
on an evaluation dataset and compares every per-order pu-MAE with a recorded table,
reporting the largest absolute and relative difference:

```bash
pixi run reproduce verify gnn_p=<checkpoint> gnn_p_r=<checkpoint> --data $D \
    --recorded <baseline_table csv> [--recorded-run gnn_p=<label>] [--placement-map <label>=alternative]
```

## Hardware and run times

The networks were trained on one NVIDIA L40S (48 GB) per run: GNN-P 4 h 51 min of training
(5 h 12 min for the whole driver job with data generation and the run's own evaluation);
GNN-P-R 6 h 13 min for the driver job and 4 h 03 min for the continuation. The Mack et al.
estimators trained on one RTX A2000 (12 GB): DNN 41 min, DNN+I 44 min, PANN 2 h 16 min. The
evaluation (2–4 min per table on 16 CPU threads), the meter sweep (about 40 min in total),
the placement search and the attribution (about 35 min on 32 threads) run on CPUs. On a GPU
with less memory, pass a smaller `--batch-size` after `--`; `--effective-batch 1024` keeps
the optimization batch through gradient accumulation.

## Vendored code

`src/injection_hse`, `run/examples/injection_hse`, `run/configs` and `tests/injection_hse`
are copied from the study's code base at one commit, with one mechanical change: the
package is renamed to `injection_hse`. `VENDORED.json` lists every file with its source
path and the commit, and the files that deviate further, each with the reason;
`tools/check_vendored.py --source <checkout>` re-derives every file from the source and
lists every deviation. Configuration files keep their names (and their bytes, so the
SHA-256 a run records for its recipe file is the one of the paper runs). Docstrings keep
the source's cross-references; a few name modules that are not part of this subset (the
deployment engine for measured data, the multi-grid report, the ablation registry, the
feasibility and headroom analyses). One unit test of the multi-grid data path, which the
paper's runs do not use, is left out because it fails in the source code base as well
(`VENDORED.json`).

## AI usage disclosure

This repository was assembled with Claude Opus 5.5 (Anthropic): it pulled the required files
from the study's code base and tested the reproduction of the paper's results automatically,
so that those results are reproducible from this repository alone.

For the paper itself, the simulation, training and analysis code was largely generated with
Claude Opus and Claude Fable (Anthropic) under the authors' specification and review; the
same systems assisted in drafting and editing the text of all sections except the
references. The authors verified all code, results and text and take full responsibility
for them.
