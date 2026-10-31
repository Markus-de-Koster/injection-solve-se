# Details not in the paper

Every value below is taken from the code (`src/injection_hse`, `src/se_ablation`), the
recipe files (`run/configs`) or the experiment arguments (`experiments/*.yaml`); the file
that defines it is named. Estimator names: Physics, Linear-P [ours], LMMSE [ours],
GNN-P [ours], GNN-P-R [ours], DNN / DNN+I / PANN [Mack].

## 1. Grid model

- **Network.** The three-phase CIGRE European LV benchmark as built by `pgml`
  (`cigre_lv_full_grid`), 44 buses (132 node-phase rows), 20 kV supply, loads at their
  published buses and ratings.
- **Lines** (the driver's `build_grid`). Every line gets a synthesized conductor geometry,
  and its impedance at every order follows from Carson's equations on that geometry. The
  geometry is chosen so that a three-phase line reproduces its published positive-sequence
  impedance Z1 and zero-sequence reactance X0, a single-phase line its R1 + jX1.
- **PV.** A unity-power-factor PV generator at every second load in the order of the load
  list (`pgml.grids.add_pv_systems`, fraction 0.5: 8 generators), rated at half the host
  load's nameplate active power, consumer type `pv`.
- **Supply impedance** (`--source-impedance cigre-tb575`): the published benchmark supply,
  100 MVA short-circuit power, R/X = 1 (R = X = 2.828 Ω at 20 kV). At the harmonics the
  supply is a passive Norton source (`physics.harmonic_slack: norton`); the data contain no
  upstream background distortion.
- **Device capacitance** (`--device-capacitance run/configs/pgl_device_capacitance.yaml`):
  the EMC and X-capacitors of electronic loads and the output filters of PV inverters, one
  wye capacitive shunt per load and per PV generator at its node,
  `C per phase = density [µF/kW] × rated active power [kW] / phases`, the density drawn once
  per device, log-uniform, seed 0: loads 0.2–2.7 µF/kW (every consumer type), PV generators
  1–4 µF/kW. The shunts are part of `grid.json`, so the data and every estimator's `Y(h)`
  contain them.
- **Load shunt** (`--load-shunt opendss --load-shunt-basis nameplate`): every load and PV
  generator is, at `h > 1`, a harmonic current source in parallel with OpenDSS's default
  harmonic load model (its admittance split into a series and a parallel R-L branch), built
  from the device's stored nameplate P, Q at rated voltage, identically in the data and in
  every estimator's `Y(h)`.
- **Orders.** The odd orders h = 1, 3, …, 19 are solved; 3–19 are injected.
- **Bus numbering.** The paper numbers the buses from 0 in node order (`Bus 0`, `Bus R0` …
  `Bus R18`, `Bus I0` … `Bus I2`, `Bus C0` … `Bus C20`); bus k is node id k + 1 of
  `grid.json` (`se_ablation.names.bus_number`). The deployed meters R4, I2, C5 are buses 5,
  22, 28; the alternative ones R6, I2, C4 are buses 7, 22, 27.

## 2. Data recipe (`run/configs/pgl_se_scenario_v4.yaml`)

### 2.1 Operating point (both datasets)

| draw | distribution | coupling |
|---|---|---|
| load P, Q scale | Uniform(0, 1) per load | Gaussian copula on one shared demand factor, ρ = 0.5; per-phase imbalance, fractional SD 0.15 |
| PV P, Q scale | Uniform(0, 1) | one factor shared by all PV generators |
| slack voltage scale | Normal(1, 0.0333²) | shared |

### 2.2 Randomized snapshots (`random:` section)

Sobol sampling. **Loads**: each load injects, per order, a current that is a ratio of its own
fundamental current, `Uniform(0, a)`, redrawn per snapshot and per load. The upper bound
`a` per consumer type is `√3 ×` the `|I₁|²`-weighted RMS ratio the device-class
composition produces for that type, so a load type carries the same harmonic current energy
in both datasets (loads of other types emit no harmonics in the snapshots):

| order | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 |
|---|---|---|---|---|---|---|---|---|---|
| household | 0.0915 | 0.0518 | 0.0277 | 0.0155 | 0.00965 | 0.00788 | 0.00624 | 0.00545 | 0.00373 |
| office | 0.278 | 0.183 | 0.119 | 0.0574 | 0.0326 | 0.0241 | 0.0168 | 0.0152 | 0.0132 |
| workshop | 0.135 | 0.0897 | 0.0576 | 0.0246 | 0.0176 | 0.0131 | 0.00879 | 0.00688 | 0.00728 |

Load emission angles: Normal(0°, 32.5°) at h3–h9, Normal(0°, 65°) at h11–h19, drawn once per
device for the whole dataset (`per: fixed`), around the device's reference spectrum (mean
resultant lengths exp(−SD²/2) = 0.85 and 0.53).

**PV generators**: a rated ratio per order, Uniform, drawn once per device:

| order | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 |
|---|---|---|---|---|---|---|---|---|---|
| PV ratio | 0.01–0.04 | 0.02–0.07 | 0.01–0.05 | 0.005–0.03 | 0.002–0.045 | 0.002–0.035 | 0.001–0.03 | 0.001–0.06 | 0.001–0.025 |

and angles as the loads'. **PV emission law** (`injection_hse.data.scenarios.emission`): the
harmonic current is complex-affine in the loading `λ = |I₁| / |I₁ at rating|`,
`I_h(λ) ∝ A_h + B_h·λ`, normalised so the rated emission equals the drawn ratio: the
proportional phasor `r_h·I₁(λ)` is multiplied by `z(λ) / (λ·z(1))` with
`z(λ) = f·e^{jδ} + (1 − f)·λ`, plus an explicit phase shift `s·(λ − 1)`. Per device and
order, drawn once: floor share `f ~ Uniform(0.43, 0.73)`, floor angle
`δ ~ Uniform(100°, 150°)`, slope `s ~ Uniform(−25°, 25°)`. Below `λ = 0.05` the law is
evaluated at 0.05 (the harmonic current falls to zero with the fundamental).

### 2.3 Device-class sequences (`coherent:` section)

Sequences of T = 8 steps of 900 s, every sequence starting on 21 June 2024, 16:00
(`start_time`; the steps cover 16:00–17:45 of one summer day, the generator's high-activity
window), so the variation between sequences comes from the random draws, not from the time
of day. Each load is decomposed into a roster of member
devices drawn per consumer type; each member has a rated power, a power factor and a
per-order emission ratio from its class ranges, a phase per order from its class range (drawn
once per device), the emission law of 2.2 (floor, floor angle, slope from the class ranges),
and an activity that follows a diurnal profile with on/off switching (dwell probability per
step) and correlated loading (AR(1), ρ = 0.85).

| class | rated power | PF | h3 | h5 | h7 | h9 | h11 | h13 | h15 | h17 | h19 | phase slope | activity | switching | on/off dwell | loading min / mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_linear | 150–1500 W | 0.97 | – | – | – | – | – | – | – | – | – | – | household | continuous | 0.85–0.98 | 0.3 / 0.6–1.0 |
| electronics_smps | 20–400 W | 0.97 | 0.5–0.85 | 0.25–0.6 | 0.1–0.4 | 0.05–0.25 | 0.03–0.15 | 0.02–0.1 | 0.015–0.08 | 0.01–0.06 | 0.008–0.05 | ±20° | household | discrete | 0.5–0.8 | 0.1 / 0.15–0.9 |
| ev_charger | 3.7–11 kW | 0.98 | 0.02–0.08 | 0.03–0.1 | 0.02–0.06 | 0.01–0.04 | 0.005–0.025 | 0.005–0.025 | 0.003–0.016 | 0.003–0.015 | 0.003–0.014 | ±15° | ev | discrete | 0.6–0.82 | 0.2 / 0.25–1.0 |
| pv_inverter (generation) | 1–8 kW | 1.0 | 0.01–0.04 | 0.02–0.07 | 0.01–0.05 | 0.005–0.03 | 0.002–0.045 | 0.002–0.035 | 0.001–0.03 | 0.001–0.06 | 0.001–0.025 | ±25° | pv | continuous | 0.85–0.98 | 0.05 / 0.2–1.0 |
| inverter_drive | 0.5–3 kW | 0.92 | 0.08–0.3 | 0.15–0.5 | 0.08–0.35 | 0.03–0.18 | 0.02–0.12 | 0.015–0.08 | 0.01–0.05 | 0.008–0.04 | 0.005–0.03 | ±30° | household | discrete | 0.6–0.85 | 0.15 / 0.2–1.0 |
| heat_pump_inverter | 2–12 kW | 0.95 | 0.05–0.15 | 0.1–0.3 | 0.05–0.2 | 0.02–0.08 | 0.01–0.05 | 0.008–0.035 | 0.005–0.025 | 0.004–0.02 | 0.003–0.015 | ±20° | flat | discrete | 0.9–0.98 | 0.15 / 0.2–0.95 |
| resistive_heating | 0.5–3 kW | 1.0 | – | – | – | – | – | – | – | – | – | – | flat | discrete | 0.5–0.75 | 0.9 / 0.95–1.0 |

Every emitting class: phase ranges ±55° at h3–h9 and ±105° at h11–h19 (uniform half-widths
with the mean resultant lengths of the snapshots' normal angles); emission floor share
0.43–0.73 at 100–150°. `electronics_smps` follows IEC 61000-3-2 class D. The inverter drive
switches between two states, heating (95 % power, spectrum × 0.2) and inverter (35 % power,
spectrum × 1.6), equally weighted. Loading jitter 0.05 (heat pump 0.08, resistive 0.02).

The bidirectional household inverter is the `pv_inverter` member class: 0–1 per household
load (each of the six household loads), with the spectrum, emission law and activity profile
of a PV inverter.

Compositions (member count range, power share):

| consumer type | members |
|---|---|
| household | base_linear 1 (1.5), electronics_smps 1–3 (1.0), resistive_heating 0–1 (1.0), ev_charger 0–1 (2.0), heat_pump_inverter 0–1 (2.0), pv_inverter 0–1 (1.0) |
| office | base_linear 1 (2.0), electronics_smps 3–10 (2.0), inverter_drive 1–3 (2.5), heat_pump_inverter 0–2 (1.5) |
| restaurant | base_linear 1 (2.0), inverter_drive 1–3 (2.5), electronics_smps 1–4 (1.0) |
| heat_pump | base_linear 1 (1.0), heat_pump_inverter 1–2 (3.0) |
| workshop | base_linear 1 (1.5), inverter_drive 2–4 (3.0), electronics_smps 1–4 (1.0), resistive_heating 0–1 (0.5) |
| ev_charging | base_linear 0–1 (0.5), ev_charger 1–2 (1.0) |
| pv | pv_inverter 1 (1.0) |
| any other | base_linear 1 (2.0), electronics_smps 1–2 (1.0), inverter_drive 0–1 (1.0) |

The roster is scaled to the load's nominal power (`scale_to_nominal`), the injection capped
at 3 pu of it; behavioural coupling 0.3, cloud coupling 0.5, activity scale 1. The PV
generators of the grid, which no composition covers, keep a fixed emission angle per order
drawn from Normal(0°, 32.5°). Profile: daily amplitude 1, level 0.85–1.15, amplitude jitter
0.8–1.2, phase offset 1 h, weekend contrast 0.15, seasonal amplitude 0.15 (peak day 15), PV
seasonal amplitude 0.4 (peak day 172), PV daylight 12 ± 4 h, short-term AR(1) ρ = 0.9,
σ = 0.05.

### 2.4 Sizes and splits

| dataset | size | split (train / val / test) | seed |
|---|---|---|---|
| training run, snapshots | 200 000 | 70 / 20 / 10 % | 0 |
| training run, sequences | 20 000 × 8 steps | 70 / 20 / 10 % (by sequence) | 0 |
| evaluation data (`experiments/evaluation.yaml`) | 2 000 × 8 steps | 70 / 20 / 10 %: 1 400 / 400 / 200 sequences | 0 |

The seed draws the operating points and the device population (rosters, emission angles
and ratios); the capacitance densities have their own seed (0). The training runs and the
evaluation data use the same seed, so they share one device population (one site), but the
evaluation data are a separate generation (2 000 instead of 20 000 sequences): a
nearest-neighbour check of the evaluation scenarios against the training sequences of
GNN-P found no duplicated operating point (closest distance 2.6e-4 of the largest voltage
magnitude), while the harmonic-angle fingerprints of the devices are shared. The split seed
is 0. Data are solved in complex128.

## 3. Network-model error (`run/configs/pgl_grid_perturbation.yaml`)

Perturbed deep copies of the grid: every element's factor is the product of a factor shared
by its kind and its own factor, log-normal (the spreads are the SD of the log-factor):

| quantity | shared | per element |
|---|---|---|
| line resistance | 0.05 | 0.10 |
| line reactance | 0 | 0.05 |
| line length (R, L, C together) | 0.02 | 0.05 |
| transformer resistance | 0 | 0.10 |
| transformer reactance | 0 | 0.05 |
| supply impedance | 0.30 | 0 |

Seed 0, a train pool and a disjoint test pool of copies. The evaluation data hold the same
2 000 scenarios solved on the nominal grid (`dataset_coherent`) and on 8 test-pool copies
(`dataset_coherent_heldout`, each scenario on one copy; RMS log-factors 0.079, 0.088, 0.076,
0.075, 0.082, 0.082, 0.088, 0.096, i.e. 7.5–9.6 %). The test split is the same 200
sequences in both (1 600 states, 120–248 per copy). Every estimator is trained or fitted on
nominal data and decodes with the nominal `Y(h)`; only the scored test states come from the
perturbed copies; the table reports the pooled pu-MAE over the 8 copies (per copy in the
evaluation CSV, column `copy`). The training runs (`--perturb-test heldout`) also write and
score a perturbed test set of their own (4 test-pool copies); the paper does not use it.

## 4. Measurements and the class-A instrument

**Placement.** The deployed placement meters buses 5, 22 and 28 (R4, I2, C5: one per
feeder) and the supply (`Bus 0`, the slack, always metered). Every metered bus also reads
the current of its upstream branch (`--current-channels upstream`), the supply meter the
three feeder-head branches; currents enter the networks as edge features on the directed
edges (per phase).

**Instrument** (`injection_hse.config.InstrumentErrorConfig`, `--noise-class A`): an IEC
61000-4-30 class-A power-quality instrument (harmonics per IEC 61000-4-7 class I), the phase
limits and the current fundamental band of a class-A instrument with a 5 A current input of
10 A full scale. Every voltage and current channel is read by the same model; a limit is
read as 2 standard deviations. `U_nom` is the bus's nominal line-to-neutral voltage,
`I_nom` the CT range of the grid (`injection_hse.data.grid_current_range`: the total
in-service rated power of loads and generators over three times the largest line-to-neutral
voltage, one figure for every CT), FSR = 2·`I_nom`.

| quantity | proportional part (at or above the threshold) | phase | threshold | floor (every reading) |
|---|---|---|---|---|
| voltage, h1 | ±0.1 % of reading | ±0.02° | 10 % `U_nom` | ±0.01 % of `U_nom` |
| voltage, h2–h50 | ±5 % of reading | ±0.5° | 1 % `U_nom` | ±0.05 % of `U_nom` |
| current, h1 | ±0.1 % of reading | ±0.1° | 10 % FSR | ±0.01 % of FSR |
| current, h2–h50 | ±5 % of reading | ±0.5° | 3 % `I_nom` | ±0.15 % of `I_nom` |

The regime of each reading follows its own true magnitude. At or above the threshold the
reading carries a gain and a phase error, held per scenario (sequence), channel and order
(`stable_fraction = 1`: a calibration error, constant over the 8 steps, redrawn for every
sequence and every training batch), plus the floor; below the threshold only the floor
applies, an isotropic complex error drawn anew for every reading. On the test split of the
evaluation data at the deployed placement (measured on the true magnitudes), every h1
voltage reading is in the proportional regime, but only 18–25 % of the harmonic voltage
readings at h3–h9 and almost none at h11–h19 (0–0.2 %), so about 90 % of the harmonic
voltage readings carry only the random floor; of the CT readings 49 % are proportional at
h1, 38–46 % at h3–h9, 16 % at h11 and h13, 4 % at h15 and h17 and none at h19. The readings
carry the error in training and in every evaluation; the targets never do. In training the
noise is drawn on the input device (`--noise-draw-device input`) from the run's seed; every
evaluation reseeds the voltage stream with 0 and the current stream with 7 before scoring a
split, so all estimators see the same reading draw.

## 5. Estimators and their decode

### 5.1 Per-unit encoding of the network inputs

`--per-unit` (`injection_hse.config.NormalizationConfig.per_unit`): every voltage is
expressed in per unit of its bus's line-to-neutral base voltage, every current (the metered
branch currents and the injections the networks predict) in per unit of the per-order,
per-row current base `i_base(h, r) = s(h)·V_ref² / V_base(r)`, where `s(h)` is the scale of a
typical coupling admittance of `Y(h)` (`injection_hse.physics.NetworkSolver.i_base`); the MV
and LV rows thus share one scale. Each phasor enters as `(|x|, cos θ, sin θ)`, the magnitude
`log(1 + |x|)`-compressed and z-scored per order on the training split; the currents have a
normalizer of their own. Unobserved entries are zeroed and the mask is an extra input
channel; the currents enter as edge features. Further inputs: periodic features of the
absolute time; static node attributes (rated voltage, phase, slack flag, distance from the
transformer, degree) and edge attributes (series impedance, length, branch type). The
physics solves, anchors and reports in absolute units.

### 5.2 The anchored injection decode

Every estimator except the Mack et al. models outputs the nodal injections `Î(h)`
(`injection_hse.physics.NetworkSolver`). The fundamental is solved with the slack held at its
measured voltage, the harmonics with the Norton-shunted harmonic system; with anchoring, the
metered voltage rows and the metered branch currents enter as soft least-squares constraints
next to `Y(h)·V = Î(h)` (the slack stays a hard constraint). Two ways to weight them:

- **manual** (the networks): one voltage and one current weight per stage. In training the
  weights are drawn uniformly per batch from the stage's range (`--anchor-schedule`), so the
  network cannot co-adapt to one anchor strength; validation and deployment use the range
  midpoints:

  | stage | voltage weight | current weight | deployed (midpoint) |
  |---|---|---|---|
  | `placement` (snapshot and sequence pass) | 0.5–2 | 0.05–0.2 | 1.25 / 0.125 |
  | `placement_anneal` (and the continuation of GNN-P-R) | 0.5–3 | 0.05–0.3 | 1.75 / 0.175 |

- **noise-calibrated** (Physics, Linear-P anchored): the Gaussian MAP estimate given the
  injections with error `σ_I(h) = ρ(h)·s_I(h)` (`s_I` = the training-set RMS injection of the
  order, the injection normalizer's scale) and the readings with each channel's own
  instrument sigma: every reading weighs `σ_I² / σ_reading²`. The per-channel sigma is the
  instrument model's per-reading sigma at the channel's RMS magnitude on the validation split
  (`se_ablation.frame.sensor_sigmas`).

**Decode per estimator in the table:**

| estimator | decode |
|---|---|
| Physics | zero injections, anchored, noise-calibrated weights from the instrument model's per-reading sigmas, `ρ = 1` (weighted least squares with zero-injection pseudo-measurements) |
| Linear-P [ours] | the better of two decodes per order: anchored with noise-calibrated weights at `ρ(h)` = its own relative injection error on the validation split, and unanchored. The evaluation writes both decodes (`linear map, anchored (rho = its error)`, `linear map, unanchored`) and the table takes the per-order best (`linear map, best decode per order`). On this data the rule picks the anchored decode at h1 and the unanchored one at h3–h19, at the deployed and at the alternative placement and on the perturbed grids (deployed: h1 anchored 6.33 against unanchored 6.62, harmonics anchored 2.35 against unanchored 1.71, ×1e-4 pu) |
| LMMSE [ours] | unanchored (it conditions on the readings itself) |
| GNN-P [ours], GNN-P-R [ours] | as deployed: their own manual anchor weights at the range midpoints of their last stage |
| DNN / DNN+I / PANN [Mack] | their own voltage output (no decode) |

The evaluation also writes the networks with the noise-calibrated decode (`network,
corrected decode`, `ρ` = their own validation injection error) and unanchored; the paper
does not use these.

### 5.3 Linear-P [ours]

The ridge regression from the measurement features (real and imaginary parts of all
readings, all orders jointly) to all nodal injections of the non-slack rows
(`injection_hse.analysis.baselines.fit_linear_injection_prior`): closed form on standardized
features with a diagonal loading of 1e-9 of the feature variance, fitted on the noisy
training split (70 %: 1 400 sequences = 11 200 states) of the evaluation data of the site,
refitted per placement (labelled data of the alternative placement for that column).

### 5.4 LMMSE [ours]

With the plain (slack-only) decode `V = G·I + b` and the branch-current operator, the
readings are `y = H·I + b_y + n` (`se_ablation.frame.Frame`; `H` = the observed rows of `G`
and of the current map, per order, real-stacked: the exact linear measurement map). With
the mean `μ` and covariance `Σ` of the true injections of the training split of the
evaluation data (real-stacked over all orders and free rows, so cross-order and cross-bus
structure is kept) and the diagonal reading covariance `R` from the per-sensor sigmas of the
validation split:

    Î = μ + Σ·Hᵀ·(H·Σ·Hᵀ + R)⁻¹·(y − b_y − H·μ),    V̂ = G·Î + b

decoded unanchored. Only `μ` and `Σ` come from data (no readings, no placement), so the same
statistics serve every placement; `H` and `R` are recomputed per placement.

### 5.5 Networks

| | GNN-P [ours] | GNN-P-R [ours] |
|---|---|---|
| spatial model | GINE, 6 layers, width 128, dropout 0 | the same |
| temporal model | per-node GRU, width 128, 1 layer, over the 8 steps | the same |
| output | nodal injections, Cartesian (`--injection-encoding cartesian`) | the same |
| linear start | output = Linear-P prediction + learned correction (`--prior-residual`); the ridge prior is fitted on 40 000 states of the network's own training data at the deployed placement (`--prior-fit-states 40000`) and refitted for the sequence pass; the correction's output layer starts at zero | none |
| sensor sets in training | the deployed placement | random 10 % of the buses per sample (the supply always), in every stage including the anneal and the continuation (`--direct-random-fraction 0.1`) |
| decode | anchored injection decode (5.2, manual weights) | the same |

Loss: squared error of the decoded voltages per order, divided by the order's typical
magnitude `s_h` (fitted on the training data) and weighted by mean-one weights `w_h ∝ s_h`
(`harmonic_weight: physical`), averaged over all samples, rows and orders.

### 5.6 DNN / DNN+I / PANN [Mack]

`mack/`: their published code at commit `6d0b1e9` and their pinned environment
(TensorFlow 2.13), trained with their learning-rate schedule on the same seed-0 training
data as GNN-P. Adam, batch 16 384, their triangular CyclicLR around the repository's base
learning rate 5e-5 (min 1e-5, max 1e-4, step 200 batches); training data: the training split
of GNN-P's own data (140 000 snapshots and 14 000 × 8 sequence states), read twice with
independent class-A draws (seeds 1000, 1001); validation: its validation split (seed 2000);
test: the evaluation data's test split with the evaluation's reading draw; one sample per
state and phase in the frame of phase a; inputs the slack and the three meters (DNN+I also
their 18 CT channels); budgets DNN / DNN+I 60 000 steps (plateau stop after 6 000 steps
without a 1 % improvement within 10 000), PANN 40 000 steps or 9 h (patience 8 000), best
validation checkpoint; PANN depth 7 (their rule, one layer per hop of the meter set's
eccentricity). The input layout fixes the meter set, so they are scored at the deployed
placement only; on the perturbed copies the trained weights are scored unchanged (`mack
score`; the PANN keeps the nominal admittance pattern).

## 6. Training

| setting | value |
|---|---|
| optimizer | AdamW, lr 3e-4, weight decay 0, eps 1e-8 |
| gradient clipping | 0.3 |
| batch | micro-batch 256, effective 1 024 (4 accumulation steps) |
| precision | float32 / complex64 model (`--train-dtype complex64`), data in complex128 |
| passes (`--no-curriculum`) | snapshots, sequences, and the sequences again with a cosine-annealed learning rate (`placement_anneal`) |
| epochs per pass (caps) | snapshots 3 × `--epochs`, sequences 2 × `--epochs`, anneal `--anneal-epochs` = 20: GNN-P 84 / 56 / 20 (`--epochs 28`), GNN-P-R 66 / 44 / 20 (`--epochs 22`) |
| stopping | per pass at the epoch cap, at the target loss (1e-6; anneal 1e-9), or when the validation loss has not improved by 5e-6 for 20 epochs; GNN-P ran 60 / 24 / 20 epochs (the first two passes ended by the plateau stop) |
| learning-rate schedule | constant (`--placement-scheduler none`), cosine in the anneal pass |
| optimizer state | GNN-P: carried across the passes, with a 3-epoch linear warm-up at the start of every pass after the first (`--carry-optimizer-state --stage-warmup 3`); GNN-P-R: fresh per pass |
| checkpoint | `checkpoints/stage_placement_anneal.ckpt` (the snapshot and the sequence pass share the name `placement`, so the sequence pass's checkpoint replaces the snapshot pass's) |
| seed | 0 |

**Continuation of GNN-P-R** (`se_ablation.continuation`, `continuation:` in
`experiments/gnn_p_r.yaml`): from the final checkpoint, in its last stage (random sensor
sets, the anneal stage's anchor-weight ranges), with the checkpoint's AdamW state carried
over: a 3-epoch linear warm-up to 3e-4, constant 3e-4 until the plateau stop (patience 20,
minimum improvement 5e-6 on the validation pu-MAE) or 92 further epochs, then a 15-epoch
cosine re-anneal with the state carried on; each keeps its best validation weights. The
run used the whole budget (95 + 15 = 110 epochs). The paper scores the re-annealed
checkpoint `continued/checkpoints/stage_placement_anneal_reanneal.ckpt`.

The paper runs passed the job defaults first and the run's arguments after them (argparse
keeps the last value); `experiments/*.yaml` lists both in that order.

## 7. Evaluation

- **Data**: the test split of the evaluation data (200 sequences = 1 600 states); fits
  (Linear-P, the LMMSE statistics) use its training split, calibrations (`ρ`, the
  per-channel sigmas) its validation split. The baselines are computed in the frame of the
  first network checkpoint given (its reading-noise model, solver and injection scale; the
  two networks share the training data, so either gives the same baseline scores).
- **Metric**: per order, the pu-MAE of the complex voltage over the test states and the
  unobserved node-phase rows (every row the placement does not read; MV and tied rows
  included), `|V̂ − V| / V_base`; relative error = that over the mean unobserved `|V_h|` in
  per unit at the order. "Harmonics" = the mean over h3–h19 of the per-order values.
- **Confidence intervals** (`se_ablation.evaluate.sequence_ci`): the 8 states of a sequence
  are correlated, so the sequence is the sampling unit: mean and half-width
  1.96·s/√200 of the per-sequence pu-MAE (h1, and the mean over h3–h19), written as
  `ci_*.csv` and as the `_ci95` columns of `table2.csv`.
- **Table** (`se_ablation.figures.results_table`): deployed and alternative placement,
  the perturbed grid copies (deployed, pooled over the 8 copies), and the relative error per
  order at the deployed placement. Bold: the best value of each column and every value within
  5 % of it.
- **Placements** (`se_ablation.names.PLACEMENTS`): deployed (buses 5, 22, 28 = R4, I2, C5);
  alternative (buses 7, 22, 27 = R6, I2, C4), the LMMSE-best placement of the search below,
  for which no network was trained; Linear-P is refitted there, LMMSE recomputed, the
  networks are unchanged.
- **Placement search**: all 18 × 2 × 20 = 720 placements with one meter on each of the
  residential (R1–R18), industrial (I1–I2) and commercial (C1–C20) feeders; Physics and
  LMMSE. The LMMSE-best placement minimises `h1 / min(h1) + harm / min(harm)` of LMMSE
  (`se_ablation.evaluate.summarize_search`).
- **Meter sweep**: k = 1 … 6 metered LV buses (each with its upstream CT; the supply always
  metered), 8 random placements per k drawn from the 40 LV buses other than R0, I0, C0
  (`numpy.random.default_rng(0)`), plus the deployed placement; Physics, LMMSE and the
  networks as deployed. The figure shows the median over the 8 placements of each count
  (line) and their range (band) for Physics, LMMSE and GNN-P-R, and the deployed placement as
  a star; GNN-P only as its star, since it corrects a linear map fitted at the deployed
  placement.
- **Parity** (`se_ablation.figures.parity_figure`): GNN-P on the test split at the deployed
  placement with its own reading model, every (state, node-phase row), h1, h5, h9.
- **Attribution** (`se_ablation.attribution`): GNN-P-R on the 200 test sequences at the
  deployed placement with the evaluation's reading draw and its deployed anchor weights.
  Integrated gradients (64 midpoint-rule steps from the all-zero input, the mask held fixed)
  of the summed per-unit estimated `|V̂_h|` on the unobserved rows after the anchored solve
  (anchors fixed) with respect to the network inputs, summed per input order; order
  occlusion zeroes the inputs of one order (or of h3–h19, or of h1) and re-runs network and
  solve with every anchor kept. The completeness check (`completeness.csv`) compares the
  signed attribution sum with `F(x) − F(0)` per order.

## 8. Hardware and run times

| run | hardware | time |
|---|---|---|
| GNN-P training (driver job: data, training, the run's own evaluation) | 1 × NVIDIA L40S (48 GB) | 4 h 51 min training, 5 h 12 min job |
| GNN-P-R training (driver job) | 1 × NVIDIA L40S | 6 h 13 min job |
| GNN-P-R continuation (110 epochs) | 1 × NVIDIA L40S | 4 h 03 min |
| DNN / DNN+I / PANN [Mack] | 1 × NVIDIA RTX A2000 (12 GB) | 41 min / 44 min / 2 h 16 min |
| evaluation data (2 000 sequences, nominal and 8 perturbed copies) | CPU, 16 threads | under 1 min |
| table, one placement set | CPU, 16 threads | 2–4 min |
| meter sweep (49 placements) | CPU, 16 threads | about 40 min in total |
| attribution (200 test sequences) | CPU, 32 threads | about 36 min |
