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
