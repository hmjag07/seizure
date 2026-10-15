# Pipeline

Research question: how robust are persistent-homology features to mid-recording sensor faults (electrode pops/motion artefacts, gradual signal decline, intermittent dropout) in multimodal wearable seizure detection, compared with standard feature-based and deep-learning detectors?

Persistent homology (TDA) is the main method. A classic-feature model and a small CNN are baselines.

## Overview

```
 1. EXTRACT DATA     stream labels → plan time ranges → stream only those ranges → clean → window
        │
 2. INJECT FAULTS    create damaged copies (pops, decline, dropout) at known times
        │
 3. HEALTH CHECK     per sensor, per window: healthy / degraded / failed
        │
 4. FEATURES         TDA-A: delay embedding per signal
                     TDA-B: network of sensors
                     (+ classic features and raw signal for the baselines)
        │
 5. TRAIN            classifier on features + health scores, with random sensor failures
        │
 6. TUNE             TDA settings, model settings, alarm threshold
        │
 7. RUN "LIVE"       stream recordings second by second, as a monitor would
        │
 8. EVALUATE         how much worse each model gets under each fault → results for the paper
```

---

## Step 1: Extract the data

Data is streamed from the online sources and never stored in the repo. [data-access.md](data-access.md) explains the file formats, how streaming works byte by byte, how stored numbers become physical units, and every check that was run.

**Progress (2026-10-15):**

| Part | Status | Where |
|---|---|---|
| 1.1 Understand both datasets | ✓ Done | This section, [data-access.md](data-access.md) |
| 1.2 Pick pilot patients | ✓ Done: 30 patients, 189 seizures | [`pilot.py`](../src/seizure/data/pilot.py), [`pilot_patients.csv`](../results/tables/pilot_patients.csv) |
| 1.3 Streaming and unit conversion | ✓ Done and verified | [`streaming.py`](../src/seizure/data/streaming.py) |
| 1.3 Plan which time ranges to fetch | Next | |
| 1.4 Clean the signals | To do | `preprocess.py` |
| 1.5 Windows | To do | `windows.py` |
| 1.6 Balance | To do | |

### 1.1 Datasets

#### SeizeIT2 (primary)

OpenNeuro `ds005873`, organised in the BIDS standard. Recorded 2020–2022 in five European epilepsy monitoring units (Leuven, Freiburg, Aachen, Karolinska, Coimbra) using the Sensor Dot wearable, alongside the hospital's standard video-EEG. All patients have focal epilepsy.

| Property | Whole dataset | Training range (sub-001 to sub-096), as scanned |
|---|---|---|
| Patients | 125 | 96 |
| Recording hours | about 11,640 | 8,715 |
| Runs (continuous recordings) | | 2,072 (median 1.0 h, longest 24.2 h) |
| Seizures | 886 | 702 |

**Signals per run** (16 when everything is present):

| Modality | Signals | Sampling rate | Notes |
|---|---|---|---|
| EEG | `BTEleft SD`, `CROSStop SD` | 256 Hz | Behind-the-ear electrodes: one same-side channel and one cross-head channel, placed on the side where seizures were expected to start. |
| ECG | `ECG SD` | 256 Hz | Chest |
| EMG | `EMG SD` | 256 Hz | Left deltoid (upper arm) |
| Motion, neck device | `EEG SD ACC X/Y/Z`, `EEG SD GYR A/B/C` | 25 Hz | 3 accelerometer + 3 gyroscope axes |
| Motion, chest device | `ECGEMG SD ACC X/Y/Z`, `ECGEMG SD GYR A/B/C` | 25 Hz | 3 accelerometer + 3 gyroscope axes |

**Which modalities each run has** (training range):

| Modalities present | Runs | Share |
|---|---|---|
| EEG + ECG + EMG + motion | 1,867 | 90.1% |
| EEG + ECG + EMG (no motion) | 181 | 8.7% |
| EEG + motion (no ECG, no EMG) | 22 | 1.1% |
| EEG only | 2 | 0.1% |

EEG is always present. The missing modalities are real technical failures and setup errors, so the dataset itself shows that sensors fail in practice: the motivation for this project.

**Labels** are in each run's `eeg/..._events.tsv`: start time, duration, seizure type, hemisphere, brain lobe, and whether the patient was awake or asleep. Other event types are `bckg` (no seizure) and `impd` (staff checking electrode contact, which produces artefacts). The codes are explained in `events.json` and in [data-access.md](data-access.md#the-label-files-eventstsv).

**Seizure types in the training range** (702 seizures):

| Group | Meaning | Seizures | Share |
|---|---|---|---|
| FIA | Focal impaired awareness | 289 | 41% |
| FA | Focal aware | 274 | 39% |
| other | Unknown awareness, unknown onset or unclassified | 106 | 15% |
| FBTC | Focal to bilateral tonic-clonic (convulsive) | 33 | 5% |

Awake: 344 (49%), asleep: 274 (39%), unknown: 84 (12%). The most common specific codes are `sz_foc_a_m_hyperkinetic` (146), `sz_foc_ia_nm` (105) and `sz_foc_ia_m_hyperkinetic` (88).

**Official split** (from the SeizeIT2 paper): sub-001 to sub-096 for training, sub-097 to sub-125 for validation. This project uses sub-097 to sub-125 as its final, untouched test set.

#### CHB-MIT (second)

PhysioNet `chbmit/1.0.0`. Scalp EEG recorded at Boston Children's Hospital.

| Property | Value |
|---|---|
| Patients | 22 (23 cases; chb21 is a later recording of chb01) |
| Channels | About 23 bipolar scalp channels. The project uses the 18 shared across nearly all recordings. `T8-P8` appears twice in many files (identical copies). |
| Sampling rate | 256 Hz |
| Files | One EDF per hour |
| Labels | `chbXX-summary.txt` |

CHB-MIT has many EEG channels but no ECG or motion, so it complements SeizeIT2: it tests the network-of-sensors features (TDA-B, Step 4B) where there are enough nodes to form meaningful shapes.

### 1.2 Pilot patients

#### Why a pilot subset

Processing all 96 training patients would not fit the one-week model deadline, especially because persistent homology is slow. A subset of 30 patients is large enough for 5-fold cross-validation grouped by patient (6 test patients per fold) and small enough to process quickly. More patients can be added in weeks 2–3.

#### How they were chosen

The patients were **not** hand-picked for expected good results. That would be cherry-picking: the model would only learn easy cases, then do worse on the untouched test patients, and reviewers would rightly question it. Instead, the rules below were fixed first, written down, and then applied by code ([`pilot.py`](../src/seizure/data/pilot.py)). Anyone can re-run it and get the same 30 patients.

| # | Rule | Why |
|---|---|---|
| 1 | Scan only sub-001 to sub-096 | sub-097 to sub-125 are the final test set. Their labels are not even read, so nothing about them can influence choices. |
| 2 | A seizure counts only if its run has all four modalities (EEG, ECG, EMG, motion) | The project tests what happens when sensors fail. A seizure recorded without ECG or motion cannot be used to test ECG or motion failing. This is checked **per run**, not per patient, because some patients have all modalities in some runs but not others. |
| 3 | A patient is eligible with **at least 3** such seizures | Patients with 1–2 seizures contribute very little to learn from, and with 30 slots, each one should carry real data. |
| 4 | Draw **30** eligible patients **at random**, seed 42 | Random choice avoids selection bias. The fixed seed makes the draw reproducible. |
| 5 | **Balance check:** the pilot's share of each seizure type and of sleep seizures must be within **15 percentage points** of the eligible pool's | Guards against an unlucky draw (e.g. almost no sleep seizures). If it had failed, the plan was to redo the draw within groups (stratified sampling). |
| 6 | Use at most **10 seizures per patient**, chosen at random with the same seed | A few patients have dozens of seizures (sub-039 has 75). Without a cap they would dominate what the model learns. |

#### What happened at each rule

| Stage | Patients | Runs | Seizures |
|---|---|---|---|
| SeizeIT2, all | 125 | | 886 |
| Rule 1: training range | 96 | 2,072 | 702 |
| Rule 2: seizures in runs with all 4 modalities | 87 with at least one | 1,867 | 631 (71 dropped) |
| Rule 3: at least 3 usable seizures | **50 eligible** | | 581 |
| Rule 4: random draw | **30 pilot** | 752 (744 with all 4 modalities) | 357 |
| Rule 6: cap of 10 per patient | 30 | | **189 used** |

Why the 46 patients were not eligible:

| Usable seizures | Patients |
|---|---|
| 0 | 9 (sub-048, 049, 054, 058, 081, 082, 087, 090, 091) |
| 1 | 24 |
| 2 | 13 |

The 50 eligible patients: sub-001, 002, 004, 011, 012, 015, 021, 022, 025, 026, 028, 029, 030, 031, 034, 035, 036, 039, 040, 042, 043, 044, 046, 047, 050, 051, 052, 055, 056, 059, 060, 062, 064, 066, 067, 068, 071, 072, 073, 074, 076, 077, 079, 080, 084, 085, 086, 089, 092, 094.

#### The 30 pilot patients

| # | Patient | Usable seizures | Used (after cap) | Usable seizures during sleep | Seizure types (usable) | Hours with all 4 modalities |
|---|---|---|---|---|---|---|
| 1 | sub-002 | 15 | 10 | 5 | FA:2, FIA:13 | 99.2 |
| 2 | sub-004 | 4 | 4 | 1 | FA:4 | 18.5 |
| 3 | sub-015 | 9 | 9 | 5 | FBTC:1, FIA:8 | 88.6 |
| 4 | sub-021 | 4 | 4 | 3 | FIA:4 | 111.8 |
| 5 | sub-022 | 7 | 7 | 1 | FIA:7 | 113.5 |
| 6 | sub-025 | 3 | 3 | 1 | FA:2, FBTC:1 | 98.0 |
| 7 | sub-031 | 3 | 3 | 1 | FA:2, FBTC:1 | 107.3 |
| 8 | sub-034 | 30 | 10 | 9 | FIA:30 | 95.3 |
| 9 | sub-035 | 55 | 10 | 55 | FA:55 | 72.3 |
| 10 | sub-039 | 75 | 10 | 56 | FA:35, other:40 | 91.9 |
| 11 | sub-040 | 3 | 3 | 0 | FIA:3 | 53.7 |
| 12 | sub-043 | 5 | 5 | 0 | FIA:5 | 87.7 |
| 13 | sub-046 | 4 | 4 | 3 | FA:2, FBTC:2 | 97.6 |
| 14 | sub-055 | 8 | 8 | 8 | other:8 | 42.2 |
| 15 | sub-056 | 13 | 10 | 9 | FIA:13 | 62.7 |
| 16 | sub-060 | 25 | 10 | 3 | FA:22, other:3 | 48.5 |
| 17 | sub-064 | 3 | 3 | 0 | FA:2, FBTC:1 | 41.3 |
| 18 | sub-066 | 4 | 4 | 0 | FIA:2, other:2 | 117.8 |
| 19 | sub-067 | 4 | 4 | 0 | FIA:4 | 35.6 |
| 20 | sub-071 | 6 | 6 | 0 | FIA:3, other:3 | 194.8 |
| 21 | sub-072 | 20 | 10 | 9 | FIA:20 | 126.1 |
| 22 | sub-076 | 3 | 3 | 0 | FIA:3 | 86.6 |
| 23 | sub-077 | 9 | 9 | 0 | FA:7, FIA:2 | 61.6 |
| 24 | sub-079 | 15 | 10 | 0 | FA:1, FBTC:1, FIA:13 | 95.5 |
| 25 | sub-080 | 8 | 8 | 0 | FBTC:1, FIA:7 | 149.1 |
| 26 | sub-084 | 4 | 4 | 0 | FIA:4 | 55.0 |
| 27 | sub-086 | 4 | 4 | 0 | FA:1, FIA:2, other:1 | 84.3 |
| 28 | sub-089 | 5 | 5 | 0 | FIA:5 | 102.5 |
| 29 | sub-092 | 6 | 6 | 0 | FIA:6 | 106.3 |
| 30 | sub-094 | 3 | 3 | 0 | FIA:3 | 43.8 |
| | **Total** | **357** | **189** | **169** | | **2,589** |

The list is also in [`configs/default.yaml`](../configs/default.yaml) (`pilot_patients`), and every seizure with its `used` flag is in [`seizeit2_seizures.csv`](../results/tables/seizeit2_seizures.csv).

#### Balance check (rule 5)

| | Eligible pool (581 seizures) | Pilot, before cap (357) | Within 15 points? | Pilot, after cap (189 used) |
|---|---|---|---|---|
| FA (focal aware) | 38% | 38% | ✓ | 24% |
| FIA (focal impaired awareness) | 44% | 44% | ✓ | 61% |
| FBTC (tonic-clonic) | 3% | 2% | ✓ | 4% |
| other | 15% | 16% | ✓ | 11% |
| During sleep | 44% | 47% | ✓ | 32% |

The random draw matched the eligible pool closely, so no stratified redraw was needed.

#### Things to know about this pilot set

1. **The cap shifts the mix.** After capping, FIA rises from 44% to 61% and sleep seizures fall from 47% to 32%. The patients who lost the most seizures to the cap were mostly FA and asleep: sub-035 (55 seizures, all FA, all asleep) and sub-039 (75, mostly asleep). This is the intended effect of giving each patient an equal voice, and should be stated in the paper.
2. **Few tonic-clonic seizures (7 used).** These large convulsive seizures are the easiest for motion sensors to catch. With so few, results describe mainly the harder, quieter focal seizures, which is realistic for this dataset. The paper should not make specific claims about tonic-clonic seizures.
3. **sub-067 has a real sensor fault:** its neck accelerometer Z axis is stuck at maximum for 96% of an hour checked ([data-access.md](data-access.md#56-a-real-sensor-fault-sub-067)). It stays in the pilot, because real faults are part of what is being studied, and the health check (Step 3) should flag it.

#### Output files

| File | Contents |
|---|---|
| [`results/tables/pilot_patients.csv`](../results/tables/pilot_patients.csv) | One row per pilot patient (the table above) |
| [`results/tables/seizeit2_seizures.csv`](../results/tables/seizeit2_seizures.csv) | All 702 training-range seizures: patient, run, onset, duration, type, group, vigilance, hemisphere, lobe, whether its run has all modalities, and whether it is `used` |
| [`results/tables/seizeit2_runs.csv`](../results/tables/seizeit2_runs.csv) | All 2,072 training-range runs: modalities present and length. Needed to sample background data (1.6). |

Re-run with `PYTHONPATH=src python -m seizure.data.pilot`.

### 1.3 Stream only what is needed

1. Fetch the label files only (tiny) to find every seizure. ✓ Done in 1.2.
2. Plan the time ranges needed: seizures, 5–10 min around them, and sampled background.
3. Fetch those ranges with HTTP range requests, one request per chunk (e.g. 10 min), not per window.
4. Convert stored numbers to physical units. ✓ `read_physical`, verified (below).
5. Process in memory: clean, window, inject faults, compute features.
6. Keep only the features (see caching in [data-access.md](data-access.md#8-caching)). Discard the raw signal.

**Unit conversion, summary of checks** (full details in [data-access.md](data-access.md#5-checking-the-conversion-is-correct-2026-10-15)):

| Check | Expected | Result | Status |
|---|---|---|---|
| ECG heart rate (sub-001, 3 times of day) | 50–80 beats/min | 59–66 beats/min | ✓ pass |
| Behind-the-ear EEG amplitude | low tens of µV | 7–14 µV | ✓ pass |
| CHB-MIT scalp EEG amplitude | tens of µV | 40–58 µV | ✓ pass |
| All 2,976 pilot files continuous, 1 s records, lengths match labels | yes | yes | ✓ pass |
| Accelerometer at rest (gravity) | 1 g | 0.5 or 0.167, depending on patient | ✗ dataset problem, fixed in 1.4 |
| Gyroscope unit | degrees/s | labelled `g` | ✗ dataset problem, normalise per patient |

### 1.4 Clean the signals

| Signal | Cleaning | Why |
|---|---|---|
| EEG | Band-pass about 0.5–40 Hz, 50 Hz notch filter | Removes slow drift and high-frequency noise; 50 Hz is European mains interference (all SeizeIT2 centres are in Europe). |
| ECG | Band-pass about 0.5–40 Hz, R-peak detection to get heart rate | Heart rate and its variability change during many seizures. |
| EMG | High-pass about 20 Hz | Muscle activity is high-frequency; this removes movement and drift. |
| Accelerometer | **Gravity calibration first** (below), then low-pass; compute total acceleration | |
| Gyroscope | Low-pass; compute rotation magnitude | Unit is unreliable (labelled `g`), so only relative values are used after normalisation. |
| All | Normalise each signal per patient (z-score) | Puts patients on a comparable scale. |

All filters must be **causal** (use only past samples), because Step 7 simulates a live monitor that cannot see the future.

**Accelerometer gravity calibration.** The accelerometer values in SeizeIT2 are on two inconsistent scales: gravity reads about 0.5 for some patients and 0.167 for others, instead of 1 g ([data-access.md](data-access.md#54-accelerometer-gravity)). A still accelerometer always measures exactly 1 g of gravity, whichever way it is tilted. So for each device in each recording:
1. Find rest periods: 10-second blocks where the total acceleration `√(x² + y² + z²)` barely varies.
2. Measure the average total acceleration during rest.
3. Divide all three axes by that value, so gravity reads 1 g.

This is the standard method for calibrating wearable accelerometers using gravity (van Hees et al., *J. Appl. Physiol.* 2014). Without it, "total acceleration" would mean different things for different patients, and a model could learn to tell device configurations apart instead of detecting seizures.

### 1.5 Windows
Produce one output every **1 s**, from a window looking back over the recent signal:
- EEG, EMG, motion: last **4 s**
- ECG: last **10–20 s**, to include enough heartbeats

**Label:** a window is *seizure* if it overlaps a seizure in the labels, otherwise *background*.

### 1.6 Balance
- Keep all seizure windows, plus the 5–10 min around each seizure.
- Sample background windows (the SeizeIT2 benchmark used 5× as many background windows as seizure windows). Include walking, eating and sleep.

---

## Step 2: Inject faults

Clean windows are damaged deliberately in code, so it is known exactly what is broken, when, and how badly.

| Fault | Simulation | Severity knob |
|---|---|---|
| Electrode pop / motion artefact | Sudden large spikes or step jumps that decay back | How large and how frequent |
| Gradual decline (rising impedance) | Slowly shrink the signal and add growing noise over minutes | How fast and how far it declines |
| Intermittent dropout | Short random bursts set to zero / flat | Fraction of time missing |

- Faults start **partway through a recording**, not from the beginning.
- Severity levels: 0, 25, 50, 75, 100%.
- Affected sensors: one EEG channel, both EEG, ECG only, motion only, combinations.
- Save the ground-truth fault labels (sensor, fault, seconds). These are the training labels for Step 3.
- Faults are also injected into training data (random type, sensor, severity). See Step 5.3.

---

## Step 3: Health check (fault detection)

For every sensor in every window, decide whether it is **healthy, degraded or failed**.

**Signal-quality features:**
- Variance: near zero means flatline or dropout.
- Kurtosis and maximum jump: spikes or pops.
- Line length.
- Fraction of flat or clipped samples: dropout or saturation.
- Baseline drift and amplitude trend over the last few minutes: gradual decline.
- Correlation with the sensor's own recent history: sudden change in behaviour.

**Classifier:** a small model (e.g. random forest) per sensor type, trained on the labels from Step 2.

**Output:** a health score from 0 to 1 per sensor per window.
- **Healthy:** use the sensor normally.
- **Degraded:** use it, but trust it less.
- **Failed:** ignore it.

The checker's own accuracy (faults caught, and how quickly) is reported as part of the results.

---

## Step 4: Feature extraction

### 4A. TDA-A: delay embedding (per signal)
Captures rhythm and pattern within each signal.

1. Take one window of one signal, e.g. 4 s of EEG = 1,024 samples.
2. Delay embedding: each point is `(x[t], x[t+τ], x[t+2τ], …)`. Choose τ by mutual information and the dimension m (usually 3–5) by false nearest neighbours, once on training data, then keep them fixed.
3. Downsample to about 300 points.
4. Persistent homology (ripser): H0 (clusters) and H1 (loops).
5. Vectorise the diagram:
   - Statistics: number of features, total persistence, longest loop, persistent entropy.
   - Optionally: a persistence image or Betti curve.

Applied to each EEG channel, ECG, EMG, and the accelerometer and gyroscope magnitudes. Plain heart rate and heart-rate variability are added for ECG.

### 4B. TDA-B: network of sensors
Captures how sensors move together. During a seizure, regions synchronise and sensors across modalities change together.

1. **Nodes:**
   - SeizeIT2: 2 EEG + ECG + EMG + 2 motion devices × (3 accelerometer + 3 gyroscope axes) = 16 nodes. The 12 motion axes are strongly correlated within each device, so using per-device acceleration and rotation magnitudes instead (2 EEG + ECG + EMG + 4 magnitudes = 8 nodes) is an option to compare.
   - CHB-MIT: 18 EEG channels.
2. **Distance** between each pair of healthy nodes = 1 − similarity (correlation or a frequency-based similarity).
3. **Persistent homology** on the distance matrix.
4. **Features** that don't depend on node count: how early nodes merge (synchronisation), and how many loops form and how long they persist.
5. **Failed sensors are left out of the network.**

Why it helps with robustness: persistent homology is stable, so small changes to the point set give small changes to the diagram. Removing a redundant sensor should barely change the shape. That gives the paper a testable hypothesis:

> Topological features of the sensor network degrade gracefully when redundant sensors fail, more gracefully than features tied to specific channels.

Limitation: 8–16 nodes in SeizeIT2, many of them motion axes that move together, gives limited shape, so TDA-B is expected to be stronger on CHB-MIT and TDA-A to carry more weight on SeizeIT2.

### 4C. Baseline features
- **Classic features:** EEG band powers, line length, heart rate, heart-rate variability, motion energy.
- **Raw windows:** input to the CNN baseline.

Features are cached for each version of the data (clean, and each fault type and severity), because persistent homology is the slowest step.

---

## Step 5: Train

### 5.1 Inputs
```
[ TDA-A features per sensor ] + [ TDA-B network features ] + [ health score per sensor ]
```
A failed sensor's features are set to missing (NaN).

### 5.2 Classifier
XGBoost or LightGBM:
- They handle missing values natively.
- They train fast.
- They give feature importances, which are useful for explaining results.

### 5.3 Sensor-dropout augmentation
During training, randomly remove sensors and inject faults of random type and severity. Without this, the model depends on every sensor and collapses when one fails.

### 5.4 Class imbalance
Use class weights (e.g. `scale_pos_weight`).

### 5.5 Patient-level splits
A patient's data never appears in both training and testing. Within the pilot, use 5-fold cross-validation grouped by patient.

### 5.6 Baselines
1. Classic features + XGBoost.
2. Small 1D CNN on raw windows, with failed sensors filled with zeros.

All models use the same data, splits and fault augmentation.

---

## Step 6: Tuning

1. **TDA settings:** window length, τ, m, number of points, which diagram summaries to use.
2. **Classifier:** depth, learning rate, number of trees (grid search or Optuna inside the grouped cross-validation).
3. **Alarm rule.** The SeizeIT2 benchmark raises an alarm when at least 8 of the last 10 one-second segments are classified as seizure. Tune:
   - the probability threshold,
   - the X-of-Y rule,
   - weighting each second by its health scores.

   This sets the trade-off between sensitivity and false alarms.
4. **CHB-MIT:** same pipeline and code, but a **separate model**. Its sensors differ (18 scalp EEG channels, no ECG or motion), so the SeizeIT2 model can't be fine-tuned on it directly.

---

## Step 7: Simulated real-time run

```
for each second t in the recording:
    1. take the latest buffer of each sensor (4 s, or 10–20 s for ECG)
    2. health check  → health score per sensor
    3. features      → TDA-A + TDA-B, using healthy sensors only
    4. model         → seizure probability for second t
    5. alarm rule    → alarm on/off
    6. log: probability, alarm, health scores, processing time
```

- Use past data only; filters are causal.
- Faults start partway through, following the injected schedules.
- Processing time per second is measured. If it is well under 1 s, the method is suitable for real-time monitoring.

---

## Step 8: Evaluate

### Metrics (aligned with the SeizeIT2 benchmark)
- **Sensitivity**, using any-overlap scoring.
- **False alarms** per hour and per 24 h.
- **Detection delay.**
- **AUROC / AUPR.**
- **Fault detection:** faults caught and detection time.

### Experiment grid
For each model (TDA, classic, CNN), each fault type (3), each severity (0–100%), and each affected sensor set, record every metric.

### Paper outputs
1. **Robustness curves:** performance against fault severity, one line per model. This is the main figure.
2. **Table by sensor lost:** which sensor's failure hurts most.
3. **False alarms caused by faults,** with and without the health check.
4. **Ablations:** without the health check, without dropout augmentation, without TDA-B, without TDA-A.
5. **Clean-data comparison** with the SeizeIT2 benchmarks (SVM: 71.1% sensitivity at 11.0 false alarms/hour; ChronoNet: 84.2% at 100.5 false alarms/hour).
6. **CHB-MIT results.**

### Success target (set 2026-10-15, before any results)
**When 60% of channels fail, sensitivity drops by at most 15 percentage points compared with clean data** (e.g. 70% → at least 55%).

- The alarm threshold is tuned once on clean validation data, then kept fixed under faults.
- False alarms per hour are always reported alongside sensitivity.
- Measured on held-out patients (SeizeIT2 sub-097 to sub-125), never on the pilot patients used for tuning.
- The TDA model's drop is also compared with the classic and CNN baselines.
- On SeizeIT2, the target applies only to failure combinations where at least one EEG channel survives. With about 6 sensors, 60% failing can remove both EEG channels; those cases are still run and reported, but have no target.

---

## Timeline (model in 1 week)

| Day | Work |
|---|---|
| 1 | Streaming loader, labels, cleaning, windowing (Step 1) |
| 2 | Fault injection and health check (Steps 2–3) |
| 3 | TDA-A features; check speed (Step 4A) |
| 4 | TDA-B and classic features (Steps 4B, 4C) |
| 5 | XGBoost with dropout augmentation; first cross-validation results (Step 5) |
| 6 | CNN baseline; alarm-rule tuning (Steps 5.6, 6) |
| 7 | Streaming replay; first robustness curves (Steps 7, 8) |

Weeks 2–3: full experiment grid, CHB-MIT, ablations, writing.

**Split:**
- Person A: Steps 1–3.
- Person B: Step 4 (start on synthetic signals until data is ready).
- Together: Step 5 onward.

## References
- Bhagubai et al., *SeizeIT2: Wearable Dataset Of Patients With Focal Epilepsy*, arXiv:2502.01224 (2025). https://arxiv.org/html/2502.01224v1
- CHB-MIT Scalp EEG Database, PhysioNet. https://physionet.org/content/chbmit/1.0.0/
