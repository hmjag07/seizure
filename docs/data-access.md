# Data access

Both datasets are read online. Raw signal is never downloaded in full or committed to the repo.

## Sources

| Dataset | Base URL |
|---|---|
| SeizeIT2 (OpenNeuro `ds005873`) | `https://s3.amazonaws.com/openneuro.org/ds005873/` |
| CHB-MIT (PhysioNet `chbmit/1.0.0`) | `https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/` |

Both are public cloud storage that supports HTTP range requests, so a client can request a specific byte range of a file.

## How streaming works

EDF files store data in fixed-length records (1 s in both datasets).
1. Read the header (256 bytes + 256 bytes per signal) to get the signal list, samples per record and header size.
2. Seconds `[t0, t1)` then sit at a byte range that can be computed directly. Request only that range.
3. Records are little-endian 16-bit integers, signal after signal. Convert to physical units with each signal's physical and digital min/max from the header.

[`src/seizure/data/streaming.py`](../src/seizure/data/streaming.py) does this with the standard library only. Run `python -m seizure.data.streaming <edf_url>` to test it.

## Test results (2026-10-14)

| File | Total length | Fetched | Data transferred |
|---|---|---|---|
| SeizeIT2 sub-001 run-01 EEG | 18.1 h (74 MB) | 4 s | 4.5 KB |
| SeizeIT2 sub-001 run-01 ECG | 18.1 h | 4 s | 2.5 KB |
| SeizeIT2 sub-001 run-01 motion | 18.1 h | 4 s | 2.9 KB |
| SeizeIT2 sub-001 run-01 EMG | 18.1 h | 4 s | 2.5 KB |
| CHB-MIT chb01_03 | 1 h (42 MB) | 4 s | 47 KB |

Observed in the files:
- **SeizeIT2 signals:**
  - EEG: `BTEleft SD`, `CROSStop SD`, 256 Hz
  - ECG: `ECG SD`, 256 Hz
  - EMG: `EMG SD`, 256 Hz
  - Motion: `EEG SD ACC X/Y/Z` and `EEG SD GYR A/B/C`, 25 Hz
  - Most files also contain an `EDF Annotations` signal, which is skipped.
- **SeizeIT2 labels:** `sub-XXX/ses-01/eeg/..._events.tsv`, a few hundred bytes. Example (sub-001 run-01):
  ```
  onset     duration  eventType  ...  recordingDuration
  14080.00  84.00     impd       ...  65319.00
  23838.00  8.00      impd       ...  65319.00
  ```
- **CHB-MIT:** 256 Hz, bipolar channels (`FP1-F7`, `F7-T7`, …), 1-hour files. Seizure times are in `chbXX/chbXX-summary.txt`.

## Usage pattern

1. Fetch all label files first (tiny).
2. Plan the time ranges needed: seizures, the minutes around them, and sampled background.
3. Fetch each range as one chunk (e.g. 10 min), not one request per window. A 10-minute SeizeIT2 EEG chunk is about 700 KB.
4. Window, inject faults and extract features in memory.

## Caching

Raw signal is never saved. Computed features may be saved to `cache/` (git-ignored) so experiments don't re-stream and recompute persistent homology every run. The cache can be deleted and rebuilt from the online sources at any time.

**Open question:** confirm with the supervisor whether a local feature cache is acceptable or whether zero local storage is required.
