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

Data is streamed from the online sources and never stored in the repo. See [data-access.md](data-access.md) for URLs, the streaming test and caching.

### 1.1 Datasets

**SeizeIT2** (primary). OpenNeuro `ds005873`, BIDS format.
- 125 patients, 886 focal seizures, more than 11,000 hours.
- Per patient: `sub-XXX/ses-01/{eeg,ecg,emg,mov}/`, one EDF per run, e.g. `sub-001_ses-01_task-szMonitoring_run-01_eeg.edf`. Runs can be about 18 h long.
- Signals:
  - EEG: `BTEleft SD`, `CROSStop SD` (2 behind-the-ear channels), 256 Hz
  - ECG: `ECG SD`, 256 Hz
  - EMG: `EMG SD`, 256 Hz
  - Motion: 3 accelerometer and 3 gyroscope axes, 25 Hz
- Labels are in `eeg/..._events.tsv`: `onset`, `duration`, `eventType`, lateralization, localization, vigilance, confidence, channels. The top-level `events.json` explains the event-type codes.
- Official split: sub-001 to sub-096 for training, sub-097 to sub-125 for validation.
- About 3% of recordings are already missing ECG/EMG/motion because of real technical failures, which supports the motivation for this project.

**CHB-MIT** (second). PhysioNet `chbmit/1.0.0`.
- 22 patients (23 cases), 256 Hz, about 23 scalp EEG channels, 1-hour EDF files.
- Seizure times are in `chbXX-summary.txt`.
- Use the 18 channels shared across nearly all recordings.

### 1.2 Pilot patients
Chosen by rule, then at random, not by expected results (`python -m seizure.data.pilot`, 2026-10-15):
1. Only sub-001 to sub-096 are scanned.
2. A seizure counts only if its run has all four modalities (EEG, ECG, EMG, motion).
3. Eligible: at least 3 such seizures. 50 of 96 patients qualify.
4. 30 eligible patients are drawn at random (seed 42).
5. Balance check: the pilot's share of asleep seizures and of each seizure type (focal aware, focal impaired awareness, focal-to-bilateral tonic-clonic, other) is within 15 points of the eligible pool's share. Passed.
6. At most 10 seizures per patient are used, chosen at random (same seed): 189 seizures in total.

Result: 30 patients, 357 usable seizures (189 after the cap), about 2,600 hours with all modalities. The patient list is in `configs/default.yaml`; per-patient details are in `results/tables/pilot_patients.csv`.

- Keep sub-097 to sub-125 untouched as the final test set. This also allows direct comparison with the SeizeIT2 paper's benchmarks.

### 1.3 Stream only what is needed
1. Fetch the label files only (tiny) to find every seizure.
2. Plan the time ranges needed: seizures, 5–10 min around them, and sampled background.
3. Fetch those ranges with HTTP range requests, one request per chunk (e.g. 10 min), not per window.
4. Process in memory: clean, window, inject faults, compute features.
5. Keep only the features (see caching in [data-access.md](data-access.md)). Discard the raw signal.

### 1.4 Clean the signals
- **EEG:** band-pass about 0.5–40 Hz, 50 Hz notch filter.
- **ECG:** band-pass about 0.5–40 Hz, R-peak detection to get heart rate.
- **EMG:** high-pass about 20 Hz.
- **Motion:** low-pass; compute total acceleration and rotation magnitude.
- Normalise each signal per patient (z-score).

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
   - SeizeIT2: 2 EEG + ECG + EMG + 3 accelerometer axes + 3 gyroscope axes, about 10 nodes.
   - CHB-MIT: 18 EEG channels.
2. **Distance** between each pair of healthy nodes = 1 − similarity (correlation or a frequency-based similarity).
3. **Persistent homology** on the distance matrix.
4. **Features** that don't depend on node count: how early nodes merge (synchronisation), and how many loops form and how long they persist.
5. **Failed sensors are left out of the network.**

Why it helps with robustness: persistent homology is stable, so small changes to the point set give small changes to the diagram. Removing a redundant sensor should barely change the shape. That gives the paper a testable hypothesis:

> Topological features of the sensor network degrade gracefully when redundant sensors fail, more gracefully than features tied to specific channels.

Limitation: about 10 nodes in SeizeIT2 gives limited shape, so TDA-B is expected to be stronger on CHB-MIT and TDA-A to carry more weight on SeizeIT2.

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
