# Data access

This document explains how the project reads its data, what the data files look like inside, how raw numbers are turned into real physical units, which checks were run to prove it works, and what problems were found in the data along the way.

**Key principle:** both datasets are read directly from the internet, piece by piece. Raw signal is never downloaded in full and never committed to the repository. Only small text files (labels, file lists) and derived results are kept.

---

## 1. Where the data lives

| Dataset | Role in the project | Host | Base URL |
|---|---|---|---|
| SeizeIT2 (OpenNeuro `ds005873`) | Primary dataset: wearable EEG, ECG, EMG and motion | Amazon S3 (OpenNeuro's public bucket) | `https://s3.amazonaws.com/openneuro.org/ds005873/` |
| CHB-MIT (PhysioNet `chbmit/1.0.0`) | Second dataset: multi-channel scalp EEG | Amazon S3 (PhysioNet's public bucket) | `https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/` |

Both are public cloud storage buckets. Two properties of these buckets make the whole approach possible:

1. **HTTP range requests.** A client can ask for "bytes 1,139,024 to 1,143,575 of this file" instead of the whole file. The server returns only those bytes.
2. **File listings.** S3 returns a list of every file under a folder (`?list-type=2&prefix=...`), so the code can discover which recordings exist for each patient without any manual index.

### Why stream instead of downloading?

| | Download everything | Stream what is needed (this project) |
|---|---|---|
| SeizeIT2 size | Hundreds of GB (about 11,640 hours of recordings) | Only the minutes around seizures plus sampled background |
| Local disk needed | Very large | Almost none |
| Time before first result | Hours to days of downloading | Seconds |
| Risk of committing patient data by accident | Real | None: raw signal never touches the disk |

---

## 2. How the SeizeIT2 files are organised

SeizeIT2 follows the BIDS standard (Brain Imaging Data Structure), which fixes the folder and file names.

```
ds005873/
├── participants.tsv              one row per patient (ID, sex)
├── events.json                   explains every code used in the label files
├── README                        dataset description from the authors
└── sub-001/
    └── ses-01/
        ├── eeg/
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.edf      the EEG signal
        │   ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.json     recording metadata
        │   └── sub-001_ses-01_task-szMonitoring_run-01_events.tsv   seizure labels for this run
        ├── ecg/   ..._run-01_ecg.edf, ..._run-01_ecg.json
        ├── emg/   ..._run-01_emg.edf, ..._run-01_emg.json
        └── mov/   ..._run-01_mov.edf, ..._run-01_mov.json
```

- A **patient** (`sub-XXX`) has one session (`ses-01`) split into several **runs** (`run-01`, `run-02`, ...). A run is one continuous recording, from a few minutes up to about 24 hours.
- Each run has up to **four EDF files**, one per modality (EEG, ECG, EMG, motion), all starting at the same moment. When a device failed or was not set up, that modality's file is simply missing.
- Each run has **one label file** (`events.tsv`) in the `eeg` folder. The labels apply to all four modalities of that run.

### The signals inside each file

| Modality | File | Signal name(s) | What it measures | Placement | Sampling rate | Unit in file |
|---|---|---|---|---|---|---|
| EEG | `_eeg.edf` | `BTEleft SD`, `CROSStop SD` | Brain electrical activity | Electrodes behind the ear; one same-side channel and one cross-head channel | 256 Hz | µV |
| ECG | `_ecg.edf` | `ECG SD` | Heart electrical activity | Chest | 256 Hz | µV |
| EMG | `_emg.edf` | `EMG SD` | Muscle electrical activity | Left deltoid (upper arm) | 256 Hz | µV |
| Motion, neck device | `_mov.edf` | `EEG SD ACC X/Y/Z` | Acceleration in 3 directions | Device on upper back/neck (the EEG unit) | 25 Hz | g |
| | | `EEG SD GYR A/B/C` | Rotation in 3 directions | same | 25 Hz | labelled `g` (wrong, see 5.5) |
| Motion, chest device | `_mov.edf` | `ECGEMG SD ACC X/Y/Z` | Acceleration in 3 directions | Device on left chest (the ECG/EMG unit) | 25 Hz | g |
| | | `ECGEMG SD GYR A/B/C` | Rotation in 3 directions | same | 25 Hz | labelled `g` (wrong, see 5.5) |

Every file also contains an extra signal called `EDF Annotations`. It holds timestamps and text notes, not a measurement, so the code skips it.

In total that is **2 EEG + 1 ECG + 1 EMG + 12 motion = 16 signals** per run when everything is present.

### The label files (`events.tsv`)

Each run's label file is a small tab-separated table (a few hundred bytes). Example, sub-001 run-03:

```
onset     duration  eventType     lateralization  localization  vigilance  confidence  channels  dateTime             recordingDuration
23059.00  6.00      impd          n/a             n/a           n/a        n/a         n/a       2000-01-01 00:00:00  64979.00
23248.00  5.00      impd          n/a             n/a           n/a        n/a         n/a       2000-01-01 00:00:00  64979.00
57975.00  72.00     sz_foc_ia_nm  left            temp          asleep     n/a         n/a       2000-01-01 00:00:00  64979.00
```

| Column | Meaning | Example |
|---|---|---|
| `onset` | Start of the event, in seconds from the start of the recording | `57975.00` = 16 h 6 min 15 s in |
| `duration` | Length of the event in seconds | `72.00` |
| `eventType` | What happened (codes below) | `sz_foc_ia_nm` |
| `lateralization` | Brain hemisphere where the seizure started | `left`, `right`, `bi`, `un` (unknown) |
| `localization` | Brain lobe where the seizure started | `temp` (temporal), `front` (frontal), ... |
| `vigilance` | Whether the patient was awake or asleep | `awake`, `asleep`, `un` |
| `confidence` | Annotator's confidence, 0 to 1 | usually `n/a` |
| `dateTime` | Recording start (anonymised to 2000-01-01) | |
| `recordingDuration` | Total length of the run in seconds | `64979.00` = 18.05 h |

**Event types.** The full list is in `events.json`. The ones that matter here:

| Code | Meaning | How the project uses it |
|---|---|---|
| `bckg` | Background: no seizure. A run with no events has a single `bckg` row covering the whole run. | Background data |
| `impd` | Impedance check: staff testing the electrodes. Signal is likely artefact. | Not a seizure. Worth excluding from background, because it looks like a sensor fault. |
| `sz_...` | A seizure. The rest of the code describes its type (below). | Seizure data |

Seizure codes are built from parts: `sz` + onset (`foc` = focal, `gen` = generalised, `uo` = unknown onset) + awareness (`a` = aware, `ia` = impaired awareness, `ua` = unknown awareness) + motor features (`m` = motor, `nm` = non-motor, `um` = unknown) + detail (e.g. `hyperkinetic`, `automatisms`, `tonic`). For example:
- `sz_foc_ia_nm` = focal, impaired awareness, non-motor
- `sz_foc_a_m_hyperkinetic` = focal, aware, motor, with hyperkinetic movement (thrashing, pedalling)
- `sz_foc_f2b` = focal to bilateral tonic-clonic (starts focal, spreads to both sides, ends in a convulsion)

For analysis, `labels.py` groups them into four coarse types:

| Group | Codes | Description |
|---|---|---|
| FA | `sz_foc_a...` | Focal aware: the patient stays aware |
| FIA | `sz_foc_ia...` | Focal impaired awareness: awareness is reduced or lost |
| FBTC | `sz_foc_f2b` | Focal to bilateral tonic-clonic: the large convulsive kind |
| other | everything else starting with `sz` | Unknown awareness, unknown onset, generalised, or unclassified |

### CHB-MIT in brief

| Property | Value |
|---|---|
| Patients | 22 (23 cases; chb21 is a later recording of chb01) |
| Recording type | Scalp EEG in a hospital, mostly children |
| Files | One EDF per hour: `chbXX/chbXX_NN.edf` |
| Channels | About 23 bipolar channels (`FP1-F7`, `F7-T7`, ...). `T8-P8` appears twice in many files, as identical copies (verified on chb01_03). |
| Sampling rate | 256 Hz |
| Labels | Seizure start/end times in `chbXX/chbXX-summary.txt` |

---

## 3. How streaming works

### 3.1 The EDF format

EDF (European Data Format) is the standard file format for biosignals. Both datasets use it. An EDF file has two parts:

```
┌───────────────────────────┬──────────────────────────────────────────────────────────┐
│ HEADER                    │ DATA RECORDS                                             │
│ 256 bytes (fixed part)    │ record 0 │ record 1 │ record 2 │ ... │ record N-1        │
│ + 256 bytes per signal    │  (1 s)   │  (1 s)   │  (1 s)   │     │  (1 s)            │
└───────────────────────────┴──────────────────────────────────────────────────────────┘
```

**The header** describes everything needed to read the rest. The fixed part (first 256 bytes):

| Bytes | Field | Example (sub-001 EEG) |
|---|---|---|
| 0–7 | Format version | `0` |
| 8–87 | Patient ID | (anonymised) |
| 88–167 | Recording ID | |
| 168–183 | Start date and time | |
| 184–191 | **Total header size in bytes** | `1024` |
| 192–235 | Reserved: `EDF+C` (continuous) or `EDF+D` (discontinuous) | `EDF+C` |
| 236–243 | **Number of data records** | `65319` |
| 244–251 | **Duration of one record, in seconds** | `1` |
| 252–255 | **Number of signals** | `3` |

Then, for each signal, 256 more bytes are stored field by field (all labels first, then all units, and so on):

| Field | Size per signal | Meaning | Example (`BTEleft SD`) |
|---|---|---|---|
| label | 16 bytes | Signal name | `BTEleft SD` |
| transducer | 80 bytes | Sensor type | (empty) |
| physical dimension | 8 bytes | Unit | `uV` |
| physical minimum | 8 bytes | Real value of the lowest stored number | `-1311` |
| physical maximum | 8 bytes | Real value of the highest stored number | `1306` |
| digital minimum | 8 bytes | Lowest stored number | `-32768` |
| digital maximum | 8 bytes | Highest stored number | `32767` |
| prefiltering | 80 bytes | Filters applied by the device | |
| samples per record | 8 bytes | Samples of this signal in one record | `256` |
| reserved | 32 bytes | | |

**The data records** follow the header. Each record holds one second of every signal, one signal after another, each sample stored as a 16-bit whole number (2 bytes, little-endian):

```
record k:  [ 256 samples of BTEleft SD ][ 256 samples of CROSStop SD ][ 57 samples of EDF Annotations ]
           └──────── 512 bytes ────────┘└───────── 512 bytes ─────────┘└────────── 114 bytes ──────────┘
           = 569 samples = 1,138 bytes per record
```

### 3.2 From "seconds I want" to "bytes to request"

Because every record has the same size and covers exactly one second, the byte position of any second can be calculated directly:

```
record size   = 2 bytes × (total samples per record, all signals)
start byte    = header size + (first second) × record size
end byte      = header size + (last second + 1) × record size − 1
```

**Worked example:** seconds 1000 to 1004 of sub-001 run-01 EEG.

| Quantity | Calculation | Value |
|---|---|---|
| Header size | 256 + 3 signals × 256 | 1,024 bytes |
| Record size | 2 × (256 + 256 + 57) | 1,138 bytes |
| Start byte | 1,024 + 1000 × 1,138 | 1,139,024 |
| End byte | 1,024 + 1004 × 1,138 − 1 | 1,143,575 |
| Bytes downloaded | | **4,552 bytes (4.5 KB)** |
| Whole file | 1,024 + 65,319 × 1,138 | 74,334,046 bytes (74 MB) |

So 4 seconds of an 18-hour recording costs 4.5 KB instead of 74 MB.

The steps the code follows:
1. Request bytes 0–255 to read the fixed header (size, number of records, record length, number of signals).
2. Request the rest of the header to read each signal's name, unit, scaling and samples per record.
3. Compute the byte range for the wanted seconds and request only that.
4. Split each record into its signals and join the records together.
5. Convert the stored whole numbers into physical units (section 4).

### 3.3 Why "continuous" matters

The calculation above assumes record *k* is exactly second *k* of the recording. That is only guaranteed in **EDF+C** (continuous) files. **EDF+D** (discontinuous) files can have gaps, for example when a device was unplugged for 10 minutes. Record *k* could then be second *k + 600*, and every seizure label after the gap would point to the wrong data.

The SeizeIT2 JSON sidecar files say `"RecordingType": "discontinuous"`, so this was checked directly:
- The EDF files themselves are marked `EDF+C`.
- Reading the timestamp stored in each record's `EDF Annotations` field at the start, quarter, middle, three-quarter and end points of two 18-hour runs (sub-001 run-01 and run-03) showed record *k* = second *k* in every case.
- The headers of **all 2,976 pilot files** (744 runs × 4 modalities) were then checked: all `EDF+C`, all with 1-second records, and in all of them the number of records matches the `recordingDuration` in the label file to within 1 second.

As a safety net, `read_header` raises an error if it ever opens an `EDF+D` file, so a gapped file can never silently produce misaligned data.

### 3.4 Data transferred in practice

Streaming test (2026-10-14), fetching 4 seconds from the middle of each file:

| File | Total length | Fetched | Data transferred |
|---|---|---|---|
| SeizeIT2 sub-001 run-01 EEG | 18.1 h (74 MB) | 4 s | 4.5 KB |
| SeizeIT2 sub-001 run-01 ECG | 18.1 h | 4 s | 2.5 KB |
| SeizeIT2 sub-001 run-01 motion | 18.1 h | 4 s | 2.9 KB |
| SeizeIT2 sub-001 run-01 EMG | 18.1 h | 4 s | 2.5 KB |
| CHB-MIT chb01_03 | 1 h (42 MB) | 4 s | 47 KB |

CHB-MIT costs more per second because it has 23 channels instead of 2–3. A 10-minute SeizeIT2 EEG chunk is 600 × 1,138 bytes ≈ 683 KB.

### 3.5 Network errors

Over thousands of requests, an occasional connection drops (seen once during the 2,976-file header check: `SSL: UNEXPECTED_EOF_WHILE_READING`). `get_range` therefore retries up to 4 times, waiting 1, 2 and 4 seconds between attempts, before giving up.

---

## 4. Converting stored numbers to physical units

### 4.1 Why conversion is needed

EDF does not store "17.5 µV". To save space, each sample is stored as a whole number (a "digital value"), between −32,768 and 32,767 for SeizeIT2 and between −2,048 and 2,047 for CHB-MIT. These numbers mean nothing on their own. The header says how to map them back to real units.

### 4.2 The formula

Each signal has four numbers in the header: digital minimum and maximum (the smallest and largest stored numbers) and physical minimum and maximum (the real values those correspond to). Every value in between lies on a straight line:

```
physical = phys_min + (digital − dig_min) × (phys_max − phys_min) / (dig_max − dig_min)
```

This simplifies to `physical = digital × gain + offset`, where

```
gain   = (phys_max − phys_min) / (dig_max − dig_min)      size of one step, in real units
offset = phys_min − dig_min × gain
```

The code computes `gain` and `offset` once per signal when reading the header, then converts a whole chunk in a single numpy operation.

### 4.3 Worked examples

| Signal | dig min / max | phys min / max | Gain (one step) | Offset | Stored value → physical |
|---|---|---|---|---|---|
| SeizeIT2 `BTEleft SD` (EEG, sub-001) | −32768 / 32767 | −1311 / 1306 µV | 0.0399 µV | −2.48 µV | 500 → 500 × 0.0399 − 2.48 = **17.5 µV** |
| SeizeIT2 `ECG SD` (sub-001) | −32768 / 32767 | −1360 / 1380 µV | 0.0418 µV | 10.02 µV | 5000 → **219.1 µV** |
| SeizeIT2 `EEG SD ACC X` (sub-001) | −32768 / 32767 | −1 / 1 g | 0.0000305 g | 0.0000153 g | −12714 → **−0.388 g** |
| CHB-MIT `FP1-F7` (chb01) | −2048 / 2047 | −800 / 800 µV | 0.391 µV | 0.195 µV | 100 → **39.3 µV** |

The gain is also the **resolution**: the smallest change the file can represent. SeizeIT2 EEG resolves 0.04 µV; CHB-MIT, stored with 12-bit numbers, resolves 0.39 µV. Both are far finer than real EEG changes (tens of µV).

**Each signal has its own scaling, and it differs between files.** The physical range for SeizeIT2 EEG was −1311 to 1306 µV in one file and −1726 to 1445 µV in another, so the conversion is always done per signal, per file, using that file's own header.

### 4.4 What `read_physical` returns

`read_physical(url, header, t0, t1)` returns exactly the samples in `[t0, t1)` seconds, as a dictionary:

```python
{
    "BTEleft SD":  (array of 1024 floats in µV, 256.0, "uV"),
    "CROSStop SD": (array of 1024 floats in µV, 256.0, "uV"),
}
```

- `EDF Annotations` is skipped.
- If a file uses the same name twice, the second copy gets a suffix instead of overwriting the first. CHB-MIT lists `T8-P8` twice; the result contains `T8-P8` and `T8-P8 (2)` (identical data, verified on chb01_03).

---

## 5. Checking the conversion is correct (2026-10-15)

You cannot tell by looking whether an EEG value of 17.5 µV is right. So the conversion was checked against signals where the right answer is known in advance from physiology or physics.

### 5.1 ECG: heart rate

**Method:** take 60 seconds of ECG, band-pass filter 5–30 Hz to make the heartbeats stand out, detect peaks at least 0.33 s apart, count them.

**Expected:** a resting or sleeping adult has 50–80 beats per minute.

| Recording | Time in run | Beats in 60 s | ECG range (1st–99th percentile) |
|---|---|---|---|
| sub-001 run-01 | 2,000 s (33 min) | 66 | −155 to 463 µV |
| sub-001 run-01 | 30,000 s (8.3 h, night) | 66 | −186 to 365 µV |
| sub-001 run-01 | 50,000 s (13.9 h) | 59 | −103 to 221 µV |

**Result: pass.** Plausible heart rates and amplitudes of a few hundred µV, typical for chest ECG.

### 5.2 EEG: amplitude

**Method:** 60 seconds, band-pass filtered 0.5–40 Hz (the EEG frequency range), standard deviation.

**Expected:** scalp EEG is typically tens of µV. Behind-the-ear electrodes sit further from the brain, so lower values are expected.

| Recording | Channel | Std at 2,000 s | Std at 30,000 s | Std at 50,000 s |
|---|---|---|---|---|
| SeizeIT2 sub-001 run-01 | `BTEleft SD` | 8.1 µV | 7.1 µV | 7.0 µV |
| SeizeIT2 sub-001 run-01 | `CROSStop SD` | 13.0 µV | 12.8 µV | 14.0 µV |
| CHB-MIT chb01_03 (unfiltered, first 8 channels) | | 40–58 µV | | |

**Result: pass.** The cross-head channel is larger than the same-side channel, as expected, because its electrodes are further apart.

### 5.3 Recording format

**Result: pass.** All 2,976 pilot files are continuous (`EDF+C`) with 1-second records, and their lengths match the label files (section 3.3).

### 5.4 Accelerometer: gravity

**Method:** an accelerometer that is not moving measures only gravity, which is always **1 g**, whichever way the device is tilted. The total acceleration `√(x² + y² + z²)` of a still device should therefore be 1 g. To find still periods, 300 seconds were split into 10-second blocks, and blocks where the total acceleration varied by less than 0.01 were treated as rest. This was done at 30% and 60% of the way through run-01 for every third pilot patient.

**Expected:** 1.0 g on both devices for every patient.

| Patient | Neck device at rest | Chest device at rest |
|---|---|---|
| sub-001 | 0.48–0.49 | 0.52 |
| sub-002 | 0.38–0.51 | 0.49–0.50 |
| sub-039 | 0.48–0.51 | 0.52–0.53 |
| sub-021 | 0.166–0.168 | 0.166–0.169 |
| sub-031 | 0.166–0.167 | 0.164 |
| sub-046 | 0.166–0.168 | 0.166–0.167 |
| sub-060 | 0.167 | 0.170–0.171 |
| sub-076 | 0.168 | 0.169 |
| sub-080 | 0.165 | 0.168 |
| sub-089 | 0.166–0.167 | 0.167 |
| sub-067 | 32.77 (sensor fault, section 5.6) | 32.77 |

**Result: fail, but the problem is in the dataset, not in the conversion.**

How we know the conversion is right: the smallest step between converted accelerometer values is **1/1024 g on every axis, in every file checked**, even though each axis has a different header range. For example:
- sub-001 X axis: header range ±1 g, smallest stored step 32 → 32 × 2/65535 = 0.000977 g = 1/1024 g.
- sub-021 Z axis: header range −0.175 to 0.139 g, smallest stored step 208 → 208 × 0.314/65535 = 0.000997 g ≈ 1/1024 g (within the rounding of the header values).

A 1/1024 g step is a typical native resolution for this kind of sensor, so the header scaling is being applied correctly and consistently. Also, in sub-001 the total stayed the same (0.48–0.49) while the individual axes changed as the patient changed position, which means all three axes share one scale.

What is wrong: the values in the files are on **two different scales**. Gravity reads about **0.5** for some patients and about **0.167** for most others, exactly a factor of 3 apart, and neither is 1. The most likely cause is two device configurations that were not converted consistently when the dataset was prepared. The header cannot reveal this, so it has to be corrected using gravity itself.

**Fix (planned, `preprocess.py`):** for each device in each recording, find rest periods, measure the average total acceleration there, and divide all three axes by it, so gravity reads exactly 1 g. This is the standard way to calibrate wearable accelerometers using gravity (van Hees et al., *J. Appl. Physiol.* 2014).

### 5.5 Gyroscope units

The gyroscope signals (`GYR A/B/C`) are labelled `g` in the header. That cannot be right: a gyroscope measures rotation speed (degrees per second), not acceleration. With no reliable unit, gyroscope signals are treated as arbitrary units and normalised per patient (z-score). Rotation magnitude remains usable as a relative measure.

### 5.6 A real sensor fault: sub-067

In one hour of sub-067 run-01 (seconds 10,000–13,600), the neck accelerometer's Z axis (`EEG SD ACC Z`) is stuck at the largest possible stored value (32,767) **96% of the time**. It has only 67 distinct values in an hour, against 228–263 for the X and Y axes of the same device. This is a genuine sensor failure in the published data: the kind this project studies. The health check (pipeline Step 3) should flag it, and it is a useful real-world example for the paper.

A survey of the motion data in all 304 planned chunks (2026-10-16) found that sub-067 is not alone: **33 chunks (11%) across 10 patients** have at least one dead, saturated or wrongly scaled accelerometer or gyroscope axis. The details, and how these sensors are detected and left out, are in [pipeline Step 1.4](pipeline.md#14-clean-the-signals-preprocesspy).

### 5.7 Summary

| Check | Expected | Result | Status |
|---|---|---|---|
| ECG heart rate | 50–80 beats/min | 59–66 beats/min | ✓ pass |
| Behind-the-ear EEG amplitude | low tens of µV | 7–14 µV | ✓ pass |
| CHB-MIT scalp EEG amplitude | tens of µV | 40–58 µV | ✓ pass |
| File format | continuous, 1 s records | 2,976 / 2,976 files | ✓ pass |
| Accelerometer at rest | 1 g | 0.5 or 0.167 depending on patient | ✗ dataset problem; rescale using gravity |
| Gyroscope unit | deg/s | labelled `g` | ✗ dataset problem; normalise per patient |
| Duplicate channel names (CHB-MIT) | all kept | `T8-P8` kept twice | ✓ fixed |
| Motion sensors working | all axes vary normally | 33 of 304 planned chunks have a dead, saturated or wrongly scaled axis | ✗ real sensor faults; detected and left out (pipeline 1.4) |

---

## 6. Usage pattern

1. **Fetch all label files first.** They are tiny (a few hundred bytes each), so scanning all 2,072 training-range runs is quick.
2. **Plan the time ranges needed:** each seizure, the minutes around it, and sampled background.
3. **Fetch each range as one chunk** (e.g. 10 minutes), not one request per 1-second window. One request per window would mean thousands of slow round trips; one per chunk keeps it fast.
4. **Process in memory:** convert units, clean, window, inject faults, extract features.
5. **Keep only the features.** Discard the raw signal.

## 7. Code

| File | Function | What it does |
|---|---|---|
| [`streaming.py`](../src/seizure/data/streaming.py) | `get_range(url, start, end)` | Downloads one byte range, retrying dropped connections |
| | `read_header(url)` | Reads the header: signals, units, sampling rates, scaling. Rejects `EDF+D` files. |
| | `read_seconds(url, h, t0, t1)` | Raw stored numbers for the records covering `[t0, t1)` |
| | `read_physical(url, h, t0, t1)` | Signals over exactly `[t0, t1)` in physical units |
| [`labels.py`](../src/seizure/data/labels.py) | `list_keys(prefix)` | Lists files in the S3 bucket under a folder |
| | `scan_subject(sub)` | Finds a patient's runs, which modalities each has, and every seizure |
| | `scan_seizeit2(subjects)` | Scans many patients in parallel |
| | `seizure_group(event_type)` | Maps a seizure code to FA / FIA / FBTC / other |
| [`pilot.py`](../src/seizure/data/pilot.py) | `main()` | Applies the pilot-selection rules (pipeline Step 1.2) and writes the result tables |

Quick test of streaming and unit conversion on any EDF URL:

```
PYTHONPATH=src python -m seizure.data.streaming <edf_url>
```

Example output (CHB-MIT chb01_03, seconds 1000–1004):

```
chb01_03.edf
  records: 3600 x 1.0 s -> 1.0 h
  FP1-F7             256 Hz  n=1024  range [ -143.590,   134.994] uV
  F7-T7              256 Hz  n=1024  range [  -99.048,    99.048] uV
  ...
```

## 8. Caching

Raw signal is never saved. Computed features may be saved to `cache/` (git-ignored) so experiments do not re-stream and recompute persistent homology on every run. The cache can be deleted and rebuilt from the online sources at any time.

**Open question:** confirm with the supervisor whether a local feature cache is acceptable or whether zero local storage is required.
