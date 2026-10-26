# Results

This document gives the final method and the final results. The complete record of how they were reached, including every intermediate experiment, is in [research-log.md](research-log.md). Data access and unit checks are in [data-access.md](data-access.md).

**Summary.** On 26 held-out SeizeIT2 patients, the final system (classic + TDA features, XGBoost, a sensor health check, training with missing and faulty sensors, and thresholds that adapt to the working sensors) caught 76.3% of seizures at 8.1 false alarms per hour. With 5 of its 8 sensors failing or degrading mid-recording, it stayed within 15 percentage points of that in every fault condition tested. TDA features did not improve clean-data detection. They did improve detection with dead sensors (+4.6 points, p < 0.001). A spectrogram CNN + GRU trained the same way was the most robust model overall.

## 1. Question and success target

**Question.** How robust are persistent-homology (TDA) features to sensor faults that begin partway through a recording, in multimodal wearable seizure detection, compared with standard feature-based and deep-learning detectors?

**Success target**, fixed on 15 October 2026 before any results:

- When 60% of channels fail (5 of 8 on SeizeIT2, 11 of 18 on CHB-MIT), sensitivity drops by at most 15 percentage points against the same system on clean data.
- On SeizeIT2 the target applies when at least one EEG channel survives. Sets in which both EEG channels fail are still run and reported.
- False alarms per hour are reported alongside sensitivity, against a budget of 11 per hour (the SeizeIT2 benchmark's operating point).

## 2. Data

| | SeizeIT2 (primary) | CHB-MIT (second) |
|---|---|---|
| Signals | 2 behind-the-ear EEG, ECG, EMG, 2 motion devices (accelerometer + gyroscope), cleaned into 8 signals | 18 bipolar scalp EEG channels |
| Training patients | 90 (sub-001 to sub-096 with a recording that has all four modalities): 551,429 s, 24,882 s of seizure | 24 recordings of 23 people (chb21 is chb01 recorded later; grouped as one person): 150,026 s, 156 seizures |
| Test patients | 26 held-out patients (sub-097 to sub-125 with an all-modality recording): 152 seizures, 216,507 s (8,809 s of seizure) | Cross-validation only |
| Seconds used | Each seizure ±10 min, plus seizure-free background totalling 5× the seizure time, at least 30 min from any seizure | Each seizure ±5 min, plus background from seizure-free files |

Data is streamed from the public buckets with HTTP range requests; no raw signal is stored in the repository. Accelerometers were recalibrated to gravity, because the dataset stores them on two inconsistent scales. Genuinely broken motion sensors (11% of the pilot's chunks) were detected and handled (section 5.7).

## 3. The final system

| Component | Choice |
|---|---|
| Cleaning | Causal filters only (past samples): EEG band-pass 0.5–40 Hz + mains notch (50 Hz SeizeIT2, 60 Hz CHB-MIT); ECG 0.5–40 Hz; EMG high-pass 20 Hz; motion magnitude, low-pass 10 Hz |
| Output rate and windows | One decision per second, from look-back windows of 4 s (ECG 15 s). A second is labelled seizure if its newest second overlaps a seizure |
| Classic features (37) | EEG band powers (delta to gamma), total power, line length, standard deviation; ECG heart rate and RMSSD; EMG log RMS, line length, zero-crossing rate; motion mean, std, max, line length |
| TDA-A features (56) | Per signal: delay embedding (τ from average mutual information, m from false nearest neighbours, on training data), 150 points, persistent homology H0 and H1 (ripser), summarised as H0 total, max and entropy and H1 count, total, max and entropy |
| Context | For each feature, the 10 s mean and its change against the preceding minute (past only) |
| Normalisation | Robust z-score per patient: (x − median) / (1.4826 × MAD) |
| Classifier | XGBoost, 400 trees, depth 6, learning rate 0.05, class weighting by the seizure ratio; missing (switched-off) sensors handled natively |
| Robust training | Training data plus a copy with 1–6 random sensors removed in every row, plus injected-fault rows (5% of post-onset seconds of each fault condition, on 1–5 random sensors, truly failed windows removed) |
| Health check | XGBoost, 3 classes (healthy, degraded, failed) on 5 unit-free signal-quality features and their context; failed-probability averaged over 15 s; switch-off at 0.5; back on after 30 s below it; plus a rule that switches off motion axes pinned at full scale or frozen |
| Threshold | One threshold per set of working sensors, chosen in advance on training patients as the most sensitive with at most 8 false alarms per hour (CHB-MIT and the CNN: one per number of working channels) |
| Alarm | On when at least 5 of the last 10 seconds are above the threshold |

**Baselines.** The same system with classic features only. A spectrogram CNN + GRU: a log-power spectrogram per sensor, a small 2-D CNN per sensor group, and a GRU over the last 90 seconds. It was trained end to end with balanced sampling, sensor dropout and injected faults. A hybrid adds the CNN's score to XGBoost classic + TDA.

## 4. Evaluation protocol

- **Patients, not seconds, are the unit.** 5-fold cross-validation with whole patients per fold; every prediction comes from a model that never saw that patient.
- **Thresholds and health-check cut-offs are chosen on other patients only**, at no more than 8 false alarms per hour, and judged against 11.
- **Scoring follows the SeizeIT2 benchmark.** A seizure is detected if any alarm overlaps it. False alarms are alarms that overlap no seizure, per hour of non-seizure data.
- **Faults.** Three types are injected into the raw signal from an onset at 20–60% of each chunk, at severities of 25, 50, 75 and 100%:
  - electrode pops: decaying steps up to 20× the signal's spread;
  - gradual decline: the signal shrinks into noise over 5 minutes;
  - intermittent dropout: flat bursts of 0.5–5 s covering the severity's share of the time.

  Faults hit 5 of 8 sensors (6 random sets with an EEG channel fault-free), each sensor alone, or both EEG channels.
- **Dead sensors.** 102 sets of 0–7 removed sensors.
- **Statistics.** Wilson 95% intervals for clean sensitivity. Patient bootstrap (2,000 resamples of patients) for paired differences; p is the share of resamples in which the system with the addition is not better.

## 5. Results

### 5.1 Held-out patients, clean data

26 patients, 152 seizures.

| System | Caught [95% CI] | False alarms / h | AUROC | AUPR | Median delay |
|---|---|---|---|---|---|
| **Main: classic + TDA, XGBoost** | **76.3% [69.0, 82.4]** | **8.1** | 0.888 | 0.508 | 8 s |
| Classic only, XGBoost | 75.0% [67.6, 81.2] | 8.4 | 0.886 | 0.472 | 9 s |
| Spectrogram CNN + GRU | 77.6% [70.4, 83.5] | 6.5 | 0.882 | 0.497 | 7 s |
| Hybrid (XGBoost + CNN score) | 77.0% [69.7, 83.0] | 8.0 | 0.884 | 0.523 | 7 s |
| XGBoost reference for the CNN and hybrid | 72.4% [64.8, 78.9] | 7.9 | 0.881 | 0.487 | 7.5 s |

Every system is within the 11 per hour budget.

### 5.2 Held-out patients, 5 of 8 sensors dead

12 sets, each with at least one EEG channel working.

| System | Caught | Mean drop | Worst drop | Sets within 15 points | False alarms / h |
|---|---|---|---|---|---|
| **Main** | **69.2%** | **7.1** | 15.8 | **11 of 12** | 8.0 |
| Classic only | 64.6% | 10.4 | 25.7 | 8 of 12 | 7.6 |
| Spectrogram CNN + GRU | 75.1% | 2.5 | 9.2 | 12 of 12 | 7.2 |
| Hybrid | 69.8% | 7.1 | 22.4 | 11 of 12 | 7.3 |
| XGBoost reference | 68.2% | 4.2 | 22.4 | 11 of 12 | 9.6 |

With one fixed threshold instead, the same sets give 77–83% caught at 12–16 false alarms per hour, over budget. Adapting the threshold to the working sensors is what keeps the comparison at equal false alarms.

### 5.3 Held-out patients, injected faults on 5 of 8 sensors

Seizures caught by the main system, mean over 6 sensor sets (clean: 76.3%):

| Fault | Health check | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Electrode pops | on | 77.6% | 77.1% | 73.0% | 72.1% |
| | off | 75.3% | 74.4% | 65.7% | 65.8% |
| Gradual decline | on | 75.4% | 73.2% | 73.5% | 70.8% |
| | off | 72.3% | 67.3% | 66.8% | 60.9% |
| Intermittent dropout | on | 72.0% | 71.6% | 71.1% | 69.6% |
| | off | 64.4% | 57.6% | 45.5% | 27.2% |

Average over all 72 scenarios, with the health check:

| System | Caught | False alarms / h |
|---|---|---|
| **Main** | **73.1%** | **7.9** |
| Main without the health check | 61.9% | 5.8 |
| Classic only | 72.2% | 8.3 |
| Spectrogram CNN + GRU | 79.0% | 8.3 |
| Hybrid | 71.9% | 7.4 |
| XGBoost reference | 68.1% | 8.0 |

- **Main system.** With the health check, all 12 fault conditions and all 72 sensor sets stayed within 15 points of clean (mean drop 3.2). No scenario exceeded 11 false alarms per hour (worst 10.4). A fault on a single sensor left it at 71.7–79.0%.
- **Spectrogram CNN.** All 72 scenarios within 15 points (worst drop 3.3).
- **Hybrid and XGBoost reference.** 69 and 68 of 72 scenarios within 15 points.

### 5.4 Held-out patients, paired comparisons

Patient bootstrap, difference in points of seizures caught [95% interval], one-sided p.

| Comparison | Clean | 5 of 8 dead | Faults on 5 of 8 |
|---|---|---|---|
| TDA: main vs classic only | +1.3 [−4.4, 11.4], p = 0.44 | **+4.6 [1.3, 9.2], p < 0.001** | +0.9 [−2.3, 7.1], p = 0.39 |
| Spectrogram CNN vs XGBoost reference | +5.3 [−0.9, 11.5], p = 0.06 | **+7.0 [0.4, 17.1], p = 0.017** | **+11.0 [5.7, 16.8], p < 0.001** |
| Hybrid vs XGBoost reference | +4.6 [−2.4, 12.1], p = 0.13 | +1.7 [−2.7, 6.7], p = 0.22 | +3.8 [−1.5, 8.6], p = 0.10 |

### 5.5 What each part contributes (ablation)

Cross-validation, 90 training patients; faults on 5 of 8 sensors (72 scenarios). The difference is the full system minus the variant.

| Part removed | Clean: caught, FA / h | Faults: caught, FA / h | Difference under faults [95%], p |
|---|---|---|---|
| None (full system) | 67.5% [54, 81], 7.3 | 63.5% [51, 76], 7.1 | |
| 10 s / 60 s context features | 53.4%, 7.6 | 50.5%, 7.1 | +13.0 [8.6, 17.3], p < 0.001 |
| Health check (faulty sensors stay on, one fixed threshold) | 67.5%, 7.3 | 52.2%, 5.1 | +11.2 [8.7, 13.8], p < 0.001 |
| All robust training (no dropout copy, no fault rows) | 67.3%, 7.5 | 56.0%, 7.2 | +7.5 [5.4, 9.7], p < 0.001 |
| TDA features | 67.5%, 7.5 | 61.1%, 7.2 | +2.4 [0.1, 4.8], p = 0.019 |
| Fault training only (dropout copy kept) | 67.3%, 7.5 | 61.4%, 6.9 | +2.1 [1.2, 3.1], p < 0.001 |
| TDA size summaries (shape kept) | 68.5%, 7.5 | 64.0%, 7.0 | −0.5 [−1.4, 0.4], p = 0.85 |
| TDA shape summaries (size kept) | 68.3%, 7.8 | 62.8%, 7.4 | +0.6 [−0.9, 2.2], p = 0.21 |
| Sensor-aware threshold (one fixed threshold) | 67.5%, 7.3 | 69.4%, 10.3 [9.3, 11.4] | catches more, but false alarms rise above the 8 per hour used for selection |

Sensitivity is lower here than on the held-out patients, because the 90 training patients include many with only one or two seizures.

### 5.6 CHB-MIT

Cross-validation, 23 people, 18 channels; thresholds chosen per number of dead channels.

| Model | Clean: caught, FA / h | 11 of 18 dead: caught (mean drop) | Faults on 11 of 18 + health check: caught (mean drop) |
|---|---|---|---|
| Classic only | 80.8%, 7.5 | 74.3% (6.5) | 76.9% (3.9) |
| Classic + TDA | 84.6%, 7.3 | 78.3% (6.3) | 81.5% (4.4) |
| Classic + TDA + TDA-B (sensor network) | 87.8%, 7.8 | 77.4% (10.4) | 80.4% (3.5) |

- The faults column is from the fault-trained versions of each model (clean: 80.8%, 85.9% and 84.0%); its drops are against those.
- Every robustness target is met: all 8 dead-channel sets within 15 points for classic and classic + TDA (7 of 8 for + TDA-B), and all 72 fault scenarios within 15 points for all three.
- TDA added +3.8 [0.7, 7.9] points on clean data (p = 0.010), +4.0 [0.5, 8.2] with 11 channels dead (p = 0.014), and +4.6 [1.8, 7.7] under faults (p < 0.001).
- TDA-B added +3.2 [0.0, 6.8] on clean data (p = 0.035), and nothing once channels were lost (−0.9 and −1.1, not significant).

### 5.7 Health check and real sensor failures

- **Injected faults** (cross-validation, 90 patients):
  - 84.1% of truly failed windows switched off, and 30.2% of degraded ones.
  - Sensors with no fault switched off by mistake: EEG 0.1–0.2%, EMG 0.1%, ECG 1.1%, accelerometers 0.1–0.6%, gyroscopes 18–26%. A gyroscope at rest reads a constant value, which resembles a dropout.
- **Held-out patients** (health check of the first held-out test, trained on the 30 pilot patients): 77–94% of failed windows switched off, depending on the fault type.
- **Real broken motion sensors in SeizeIT2:**
  - Wrongly scaled accelerometers (48 sensor-chunks) were rejected at cleaning in every case.
  - Sensors pinned at full scale, or dead devices (68 sensor-chunks), were switched off in 99.7% of their seconds by the health check and stuck rule together.
  - The stuck rule wrongly switched off 0.6% of seconds of healthy motion sensors.

### 5.8 Processing time

Simulated live run on a laptop CPU: 3 patients, 1,800 seconds through the whole system. Mean 86 ms per second of data, 99th percentile 102 ms, maximum 120 ms: about 10× faster than real time. TDA-A features take 42.7 ms of this.

## 6. What did not work

| Finding | Evidence |
|---|---|
| TDA's clean-data advantage on the 30-patient pilot did not hold | Pilot: +7.1 points (76.9% vs 69.8%). 90-patient cross-validation: no difference. Held-out: +1.3, p = 0.44 |
| The sensor-network variant (TDA-B) did not help on SeizeIT2 | Mean −0.09 points over 71 dead-sensor sets; 33 better, 33 worse, 5 tied; Wilcoxon p = 0.54. At most 8 nodes, half of them correlated motion signals |
| A CNN on raw signal windows was weak | About 41% caught at the same false-alarm rate; adding a GRU over frozen encodings did not help. Spectrograms with end-to-end training fixed this (69%) |
| One fixed threshold breaks under sensor loss | Normal training: 76.9% → 25.1% with 5 of 8 dead. Dropout training: 93.5% at 28 false alarms per hour |
| Labelling faults by overall severity left half-flat sensors on | 49.9% caught at 50% dropout. Labelling each window by the damage inside it raised this to 75.6% |
| The first held-out test exceeded the false-alarm budget | 13.8 per hour against 11, with thresholds chosen at 11. Choosing at 8 per hour fixed this in the second test (6.5–8.4 per hour) |

## 7. Use of the held-out patients

The held-out group was scored twice:

- **v1:** with the models trained on the 30 pilot patients.
- **v2:** with the final systems, after Phases A–C had addressed v1's two weaknesses (false alarms over budget, no clean-data gain from TDA) using training patients only.

v2's protocol was written down before the held-out data was read again, and nothing was changed after either run. Both runs are reported, but the group is no longer an untouched test set.

| Claim | Cross-validation (90 patients) | Held-out v1 | Held-out v2 |
|---|---|---|---|
| Clean: ≥ 60% caught at ≤ 11 false alarms / h | 67.5% at 7.3 | 81.6% at 13.8 (budget missed) | 76.3% at 8.1 |
| 5 of 8 dead: mean drop ≤ 15 points | 6–10 | 9.1 | 7.1 |
| Faults on 5 of 8 with health check: conditions within target | 12 of 12 | 12 of 12 | 12 of 12 |
| TDA improves clean detection | no | no | no |
| TDA improves detection with sensors lost | +2.4 under faults (p = 0.019) | smaller drops | +4.6 with dead sensors (p < 0.001); +0.9 under faults (n.s.) |

## 8. Limitations

1. **Seizure-centred data.** Test data is concentrated around seizures, not continuous 24-hour monitoring. False-alarm rates are measured on harder-than-average data and are not directly comparable with the SeizeIT2 benchmark (SVM: 71.1% at 11.0 false alarms per hour on full recordings).
2. **Held-out group used twice** (section 7). It has 26 patients and 152 seizures, so clean sensitivity carries an interval of about ±7 points.
3. **Normalisation.** Each patient's features are normalised with that patient's whole recording, without labels. A deployed monitor would need a calibration period or a running estimate.
4. **Simulated faults.** Pops, declines and dropouts are simulations. At 25% severity a decline is already easy to detect, so real mild declines may be subtler. The CNN's faults are injected into the cached 64 Hz signal, close to but not identical to the feature build's.
5. **Multiple comparisons.** Many bootstrap comparisons are reported without correction; p-values are one-sided.
6. **Untuned settings.** XGBoost settings are defaults; TDA settings were not tuned beyond the choice of τ and m.
7. **Gyroscopes are over-flagged** by the health check when the patient is still.
8. **Seizure types.** Few tonic-clonic seizures (5% of the training range), and seizures of 10 s or less are often missed (35% caught on the pilot).
9. **CHB-MIT:**
   - cross-validation only;
   - ±5 min of seizure context, chosen for compute;
   - three differently wired files of chb12 left out.

## 9. Where the numbers come from

| Section | Table (results/tables) | Code |
|---|---|---|
| 5.1–5.4 | final2_clean.csv, final2_dead_sensors.csv, final2_faults_main.csv, final2_faults_dl.csv, final2_stats.csv | models/final_test_v2.py |
| 5.5 | phaseC_ablation.csv | models/phase_c.py |
| 5.6 | phaseD_dead_channels.csv, phaseD_faults.csv, phaseD_a3_clean.csv, phaseD_stats.csv | models/phase_d.py |
| 5.7 | phaseA3_health_*.csv, final_health_switch_off.csv, phaseC4_real_faults*.csv | models/phase_a3.py, models/final_test.py, faults/real_faults.py |
| 5.8 | phaseC3_timing.csv | models/live.py |
| 6 | model_clean_summary.csv, phaseB1_dead_sensors.csv, cnn_dead_sensors.csv, robustness_*.csv, faults_eval*.csv, final_clean.csv, phaseA_thresholds.csv | see [results/README.md](../results/README.md) |

Every table and figure is listed, with its status, in [results/README.md](../results/README.md).
