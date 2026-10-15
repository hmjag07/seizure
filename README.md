# Seizure detection under sensor faults

Final-year project.

**Research question:** How robust are persistent-homology (topological data analysis) features to mid-recording sensor faults (electrode pops/motion artefacts, gradual signal decline, intermittent dropout) in multimodal wearable seizure detection (EEG, ECG, motion), compared with standard feature-based and deep-learning detectors?

Existing TDA-based seizure detection studies generally assume every channel stays intact for the whole recording. This project studies channels that were working and then fail, degrade or drop out partway through.

## Docs
- [Pipeline](docs/pipeline.md): data extraction, fault injection, fault detection, features, training, tuning, simulated real-time run, evaluation.
- [Data access](docs/data-access.md): streaming SeizeIT2 and CHB-MIT online (nothing downloaded in full).

## Datasets
- [SeizeIT2](https://openneuro.org/datasets/ds005873) (primary): behind-the-ear EEG, ECG, EMG, motion.
- [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) (second): multi-channel scalp EEG.

## Layout
```
configs/            shared settings (default.yaml) + per-experiment overrides
src/seizure/
  data/             Step 1: streaming, labels, cleaning, windowing
  faults/           Steps 2-3: fault injection, health check
  features/         Step 4: TDA-A, TDA-B, classic features
  models/           Steps 5-6: XGBoost, CNN baseline, alarm rule
  realtime/         Step 7: simulated live replay
  evaluation/       Step 8: metrics, robustness curves
notebooks/          exploration only; move reusable code into src/
tests/              pytest
results/            figures/ and tables/ for the paper (runs/ is ignored)
paper/              manuscript
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"          # add ,cnn for the CNN baseline
```
