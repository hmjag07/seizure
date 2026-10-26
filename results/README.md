# Results index

Every table and figure in this folder, the stage that produced it, and its status. The final results are summarised in [docs/results.md](../docs/results.md). Each stage is described in full, with all its numbers, in [docs/lab-notebook.md](../docs/lab-notebook.md).

**Status:**

| Status | Meaning |
|---|---|
| **Final** | Behind the final results in [docs/results.md](../docs/results.md), section 5 |
| Stage | The result of an earlier stage, replaced or extended by a later one; kept as the record of how the final system was reached |
| Input | Data selection or settings read by later code (do not move or rename) |
| Bootstrap | Per-patient counts that the statistics scripts resample |

All files stay at these paths because the code reads and writes them by name.

**Name conventions:**

| Pattern | Meaning |
|---|---|
| `final_*` | First held-out test (v1), models trained on the 30 pilot patients |
| `final2_*` | Second held-out test (v2), the final systems |
| `*_severity_labels` | Health check before window-level fault labels ("fix D") |
| `*f_*`, e.g. `cnn_specf`, `cnn_faultsf`, `hybridf` | After fault training of the spectrogram CNN (Phase B5) |
| `cnn_*` / `cnn2_*` | Raw-signal CNN v1 / v2 (Phase B2) |
| `cnn_spec_*` | Spectrogram CNN + GRU (Phase B3) |

**Shared column names:**

| Column | Meaning |
|---|---|
| `sensitivity` | Seizures caught ÷ seizures |
| `fa_per_h` | False alarms per hour of non-seizure data |
| `drop_pts` | Clean sensitivity minus this sensitivity, in percentage points |
| `eeg_alive` | At least one EEG channel working |
| `dead` / `sensors_dead` | Number / names of removed sensors |
| `mode` | A = no health check, B = health check, C = perfect health check |
| `training` | `clean` = normal training, `dropout` = sensor-dropout training |
| `folds_over_budget` | Folds above 11 false alarms per hour |
| `switched_off` | Share of sensor-seconds the health check turned off |
| `lo`, `hi` | 95% bootstrap interval |
| `p_not_better` | One-sided bootstrap p-value |

## Tables (`tables/`)

### Data selection (Step 1, Phase A1, held-out, Phase D)

| File | Status | Contents | Made by |
|---|---|---|---|
| seizeit2_runs.csv | Input | All 2,072 training-range runs: modalities present, length | data/pilot.py |
| seizeit2_seizures.csv | Input | All 702 training-range seizures, with the `used` flag | data/pilot.py |
| pilot_patients.csv | Input | The 30 pilot patients | data/pilot.py |
| fetch_plan.csv | Input | The pilot's 304 chunks to stream | data/fetch_plan.py |
| extra_seizures.csv, extra_fetch_plan.csv | Input | The other 66 training patients | data/fullset.py |
| heldout_runs.csv, heldout_seizures.csv, heldout_fetch_plan.csv | Input | Held-out patients sub-097 to sub-125 | data/heldout.py |
| chbmit_runs.csv, chbmit_seizures.csv, chbmit_seizures_used.csv, chbmit_fetch_plan.csv | Input | CHB-MIT files, seizures and plan | data/chbmit.py |

### Features (Step 4)

| File | Status | Contents | Made by |
|---|---|---|---|
| tda_params.csv, chbmit_tda_params.csv | Input | TDA delay τ and dimension m per signal | features/build.py, features/build_any.py |
| features_counts.csv, features_missing.csv, features_auroc.csv, features_speed.csv | Stage | Pilot feature summary: window counts, missing values, single-feature AUROC, TDA time per window | features/report.py |

### First models and dead sensors, 30 pilot patients (Steps 5 and 8)

| File | Status | Contents | Made by |
|---|---|---|---|
| model_clean_summary.csv, model_clean_folds.csv | Stage (folds file is also an Input) | First clean-data models: classic, classic + TDA, classic + TDA shape | models/train.py |
| robustness_dead_sensors.csv | Stage | 102 dead-sensor sets, normal training, fixed threshold | models/robustness.py |
| robustness_dead_sensors_dropout.csv | Stage | The same with sensor-dropout training | models/robustness.py dropout |
| robustness_sensor_aware.csv | Stage | The same with sensor-aware thresholds | models/sensor_aware.py |

### Health check and injected faults, 30 pilot patients (Steps 2, 3 and 8)

| File | Status | Contents | Made by |
|---|---|---|---|
| health_confusion.csv, health_switch_off.csv, health_by_fault.csv, health_by_sensor.csv, health_cutoffs.csv | Stage (cut-offs are also an Input) | Health check with window-level labels: confusion matrix, switch-off by class, by fault, by sensor, cut-offs | faults/train_health.py |
| health_confusion_severity_labels.csv, health_switch_off_severity_labels.csv, health_by_fault_severity_labels.csv, health_by_sensor_severity_labels.csv, health_cutoffs_severity_labels.csv | Stage | The same before fix D | faults/train_health.py (earlier run) |
| faults_eval.csv | Stage | 192 fault scenarios × 3 models × 3 modes | faults/evaluate.py |
| faults_eval_severity_labels.csv | Stage | The same before fix D | faults/evaluate.py (earlier run) |

### First held-out test, v1

| File | Status | Contents | Made by |
|---|---|---|---|
| final_clean.csv, final_dead_sensors.csv, final_faults.csv | Stage | v1: clean data, 102 dead-sensor sets, 192 fault scenarios | models/final_test.py |
| final_health_switch_off.csv | **Final** (section 5.7) | Held-out health check: share switched off by true window class | models/final_test.py |

### Phase A: 90 training patients, thresholds, fault training

| File | Status | Contents | Made by |
|---|---|---|---|
| phaseA_thresholds.csv | Stage | Threshold strategies compared on unseen folds | models/phase_a.py a2 |
| phaseA3_clean.csv, phaseA3_faults.csv | Stage | A1 vs A3 models, clean and under faults | models/phase_a3.py |
| phaseA3_health_cutoffs.csv, phaseA3_health_switch_off.csv, phaseA3_health_clean_sensors.csv | **Final** (section 5.7; cut-offs are also an Input) | The 90-patient health check | models/phase_a3.py health |

### Phase B: sensor network and CNNs

| File | Status | Contents | Made by |
|---|---|---|---|
| phaseB1_dead_sensors.csv | Stage | classic + TDA vs + TDA-B, 102 dead-sensor sets | models/phase_b1.py |
| cnn_dead_sensors.csv, cnn2_dead_sensors.csv | Stage | Raw-signal CNN v1 and v2 | models/cnn.py |
| cnn_spec_dead_sensors.csv, hybrid_dead_sensors.csv, cnn_spec_stats.csv | Stage | Spectrogram CNN, hybrid and XGBoost reference before fault training | models/cnn_spec.py |
| cnn_spec_per_patient.csv | Bootstrap | | models/cnn_spec.py |
| cnn_faults.csv, cnn_faults_stats.csv | Stage | The three models under injected faults, before fault training | models/cnn_faults.py |
| cnn_faults_per_patient.csv | Bootstrap | | models/cnn_faults.py |
| cnn_specf_dead_sensors.csv, hybridf_dead_sensors.csv, cnn_specf_stats.csv, cnn_faultsf.csv, cnn_faultsf_stats.csv | Stage | The same after fault training; the cross-validation results behind the held-out systems 3–5 | `CNN_TAG=f` runs of models/cnn_spec.py and models/cnn_faults.py |
| cnn_specf_per_patient.csv, cnn_faultsf_per_patient.csv | Bootstrap | | as above |

### Phase C: ablation, live timing, real broken sensors

| File | Status | Contents | Made by |
|---|---|---|---|
| phaseC_ablation.csv | **Final** (section 5.5) | Each part removed: sensitivity and false alarms with intervals, difference vs the full system, p | models/phase_c.py stats |
| phaseC_per_patient.csv | Bootstrap | | models/phase_c.py eval |
| phaseC3_timing.csv | **Final** (section 5.8) | Milliseconds per step of the live run | models/live.py |
| phaseC4_real_faults.csv, phaseC4_real_faults_healthy_rule.csv | **Final** (section 5.7) | Real broken motion sensors; false switch-offs of each rule version | faults/real_faults.py |

### Phase D: CHB-MIT

| File | Status | Contents | Made by |
|---|---|---|---|
| phaseD_dead_channels.csv | **Final** (section 5.6) | 0–17 dead channels × 8 sets, fixed vs count-aware threshold | models/phase_d.py dead |
| phaseD_count_thresholds.csv | Stage | Count-aware thresholds per model, fold and number of dead channels | models/phase_d.py dead |
| phaseD_a3_clean.csv, phaseD_faults.csv | **Final** (section 5.6) | Fault-trained models: clean data, and injected faults on 11 of 18 and single channels | models/phase_d.py faults |
| phaseD_health_by_fault.csv, phaseD_health_clean_channels.csv, phaseD_health_cutoffs.csv, phaseD_health_switch_off.csv | Stage | CHB-MIT health check | models/phase_d.py health |
| phaseD_stats.csv | **Final** (section 5.6) | Bootstrap: TDA, TDA-B and fault training | models/phase_d.py stats |
| phaseD_dead_per_patient.csv, phaseD_faults_per_patient.csv | Bootstrap | | models/phase_d.py |

### Phase E: second held-out test, v2

| File | Status | Contents | Made by |
|---|---|---|---|
| final2_clean.csv | **Final** (section 5.1) | Five systems on clean data, with Wilson intervals | models/final_test_v2.py test |
| final2_dead_sensors.csv | **Final** (section 5.2) | Systems × threshold rule × 102 dead-sensor sets | models/final_test_v2.py test |
| final2_faults_main.csv | **Final** (section 5.3) | Main and classic systems, 192 fault scenarios, with and without the health check | models/final_test_v2.py test |
| final2_faults_dl.csv | **Final** (section 5.3) | CNN, hybrid and XGBoost reference, 72 fault scenarios | models/final_test_v2.py test |
| final2_stats.csv | **Final** (section 5.4) | Bootstrap comparisons | models/final_test_v2.py test |
| final2_count_thresholds.csv | Stage | Thresholds per number of switched-off sensors (the CNN's are logits) | models/final_test_v2.py train |
| final2_per_patient.csv | Bootstrap | | models/final_test_v2.py test |

## Figures (`figures/`)

Figures 1–16 are drawn by `python -m seizure.evaluation.plots` and figures 17–33 by `python -m seizure.evaluation.plots_phases`, from the tables above.

| Figure | Status | Shows |
|---|---|---|
| 1_timelines | Stage | Second-by-second seizure probability around three pilot seizures: long and detected, short and detected, missed |
| 2_tradeoff | Stage | Seizures caught vs false alarms per hour for every threshold, three feature sets (pilot) |
| 3_progression | Stage | How normalisation, the 5-of-10 alarm and context features raised sensitivity from 44% to 77% (pilot) |
| 4_by_seizure_length | Stage | Detection rate by seizure length (pilot) |
| 5_per_fold | Stage | Sensitivity and false alarms per fold (pilot) |
| 6_precision_recall | Stage | Precision–recall curves of the three feature sets (pilot) |
| 7_robustness_dead_sensors | Stage | Collapse with dead sensors under a fixed threshold, normal training (pilot) |
| 8_single_sensor_loss | Stage | Drop when each single sensor dies, fixed threshold (pilot) |
| 9_dropout_before_after | Stage | Normal vs sensor-dropout training under a fixed threshold (pilot) |
| 10_sensor_aware | Stage | Dead sensors at equal false alarms (sensor-aware thresholds) (pilot) |
| 11_faults_5of8 | Stage | Injected faults on 5 of 8 sensors, health check before fix D (pilot) |
| 12_faults_single | Stage | Injected fault on one sensor (pilot) |
| 13_window_labels_before_after | Stage | Effect of fix D (pilot) |
| 14_final_pilot_vs_heldout | Stage | Pilot vs held-out v1: sensitivity and false alarms |
| 15_final_dead_sensors | Stage | Held-out v1, dead sensors |
| 16_final_faults_5of8 | Stage | Held-out v1, injected faults on 5 of 8 sensors |
| 17_health_check | **Final** | What the 90-patient health check switches off |
| 18_a1_30_vs_90 | Stage | 30 vs 90 training patients |
| 19_a2_threshold_rules | Stage | False alarms on unseen folds per threshold rule |
| 20_a3_fault_training | Stage | Drop under faults with and without fault training |
| 21_b1_tda_network | Stage | TDA-B adds nothing on SeizeIT2 |
| 22_b2_cnn_baseline | Stage | Raw-signal CNNs vs XGBoost |
| 23_c_ablation | **Final** | What each part contributes, with 95% intervals |
| 24_c3_live_timing | **Final** | Processing time per second of data |
| 25_c4_real_faults | **Final** | Real broken motion sensors and the stuck rule |
| 26_d_dead_channels | **Final** | CHB-MIT: dead channels; fixed vs count-aware threshold |
| 27_d_faults | **Final** | CHB-MIT: injected faults on 11 of 18 channels |
| 28_d_bootstrap | **Final** | CHB-MIT: what TDA and TDA-B add |
| 29_d_health | **Final** | CHB-MIT health check |
| 30_spectrogram_cnn_hybrid | Stage | Spectrogram CNN and hybrid vs XGBoost with dead sensors (before fault training) |
| 31_spectrogram_cnn_faults | Stage | The three models under injected faults (before fault training) |
| 32_spectrogram_cnn_fault_training | Stage | The spectrogram CNN before and after fault training |
| 33_final2_heldout | **Final** | Held-out v2: five systems on clean data, with 5 of 8 sensors dead, and under injected faults |
