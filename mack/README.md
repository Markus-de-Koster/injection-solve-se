# Mack et al. (2024) estimators on the study's data

The DNN and PANN state estimators of Mack et al. (2024), run with their published code and
in their pinned environment on the data of this study. Their code is not copied here:
`setup.sh` fetches it at a fixed commit.

| file | what |
|---|---|
| `setup.sh` | clones `github.com/th-koeln-iet/pqse_concept_pann` at commit `6d0b1e96bbb5b1231dd872b002780acf587482db` into `mack/external/` and installs the environment of `pixi.toml` / `pixi.lock` (Python 3.11, TensorFlow 2.13.0, Keras 2.13.1, CUDA 11.8, cuDNN 8.6) |
| `envrun.sh` | runs a command in that environment with the CUDA libraries, `ptxas` and `libdevice` on the search paths |
| `adapter.py` | converts an evaluation dataset (test split) and a training run's data (training and validation splits) into their array layout, with the class-A readings of the evaluation (runs in the main environment) |
| `layout.py` | bus order, meter set, per-phase frame rotation, the positive-sequence admittance pattern and the hop eccentricity that sets the PANN depth |
| `train.py` | trains and scores one estimator with their configuration, model classes and training loop (runs in their environment); its docstring lists every deviation from their CIGRE LV experiment and why |
| `scoring.py` | the per-order voltage error at the unobserved rows |
| `protocol_table.py` | their scores in the per-order and summary schema of the paper's evaluation, with the ratios to Physics and Linear-P |
| `overlap.py` | the leakage check: exact and nearest-neighbour overlap of the evaluation states with a training run's states |

## Run

```bash
bash mack/setup.sh                                              # once
pixi run reproduce mack convert --data outputs/data/evaluation_site0 --train-run outputs/train/gnn_p
pixi run reproduce mack train --model dnn                       # DNN [Mack]
pixi run reproduce mack train --model dnn_i                     # DNN+I [Mack]: plus the CT readings
pixi run reproduce mack train --model pann                      # PANN [Mack]
pixi run reproduce mack table                                   # their per-order scores
# the held-out perturbed grid copies: the trained weights scored unchanged
pixi run reproduce mack convert --data outputs/data/evaluation_site0 --train-run outputs/train/gnn_p \
    --test-subdir dataset_coherent_heldout
pixi run reproduce mack score --model dnn --arrays outputs/mack/arrays_dataset_coherent_heldout   # and dnn_i, pann
pixi run reproduce mack table --arrays outputs/mack/arrays_dataset_coherent_heldout
```

`convert` reads the deployed placement (R4, C5, I2 and the supply) with the upstream CT
channels, two independent class-A reading draws of the training split, and the evaluation's
reading draw of the test split (voltage seed 0, current seed 7). `--test-subdir
dataset_coherent_heldout` converts the held-out perturbed grid copies instead (into
`outputs/mack/arrays_dataset_coherent_heldout`; the models keep the nominal admittance
pattern); `score` scores them with the trained weights (`mack/train.py --weights-from`, the
scalers refitted on the same training arrays). `table` writes `per_order.csv` (the input of
`reproduce figures --figure table --mack`) and `summary.csv`; `--reference <evaluation
baseline_table csv>` adds the ratios to Physics and Linear-P. `--smoke` trains for 300 steps.

Settings of the paper runs (`se_ablation.reproduce.MACK`): base learning rate 5e-5 with the
repository's CyclicLR, seed set 0, the best validation checkpoint (written every 50 epochs);
DNN and DNN+I at most 60 000 steps, stopped after 6 000 steps when the validation loss has
not improved by 1 % within 10 000 steps; PANN at most 40 000 steps or 9 hours, patience
8 000 steps.

Tests: `cd mack && python -m pytest -q tests` (main environment).
