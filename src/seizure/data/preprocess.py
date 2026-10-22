"""Causal filtering, accelerometer gravity calibration and per-patient normalisation.

Every filter is causal (sosfilt/lfilter, never filtfilt), so the output at time t uses
only samples up to t, as a live monitor would.
"""
import numpy as np
from scipy.signal import butter, iirnotch, lfilter, sosfilt

# EEG montages differ by patient (BTEleft+CROSStop, BTEleft+BTEright, BTEright+CROSStop).
# EEG1 is the first channel in the file, EEG2 the second.
EEG_NAMES = ("BTEleft SD", "BTEright SD", "CROSStop SD")
DEVICES = {"neck": "EEG SD", "chest": "ECGEMG SD"}
SIGNALS = ("EEG1", "EEG2", "ECG", "EMG", "ACC_neck", "ACC_chest", "GYR_neck", "GYR_chest")
FS = {"EEG1": 256, "EEG2": 256, "ECG": 256, "EMG": 256,
      "ACC_neck": 25, "ACC_chest": 25, "GYR_neck": 25, "GYR_chest": 25}

REST_BLOCK_S = 10
REST_REL_STD = 0.02  # block counts as rest if |a| varies by < 2% of its mean
# Gravity reads ~0.167 or ~0.5 on working devices; values outside this range come from
# dead axes (~0) or axes stored on a different scale (~32.8), so the accelerometer is unusable.
SCALE_RANGE = (0.1, 1.5)


def stuck(x):
    """True for a dead axis (constant) or one pinned at its own min or max for over half the samples.

    A healthy axis at rest also repeats one value (a still gyroscope reads 0), but that value
    sits inside its range, not at an edge, so it is not flagged.
    """
    lo, hi = x.min(), x.max()
    return lo == hi or max(np.mean(x == lo), np.mean(x == hi)) > 0.5


def valid_scale(s):
    return s is not None and SCALE_RANGE[0] <= s <= SCALE_RANGE[1]


def _sos(btype, cutoff, fs, order=4):
    return butter(order, cutoff, btype=btype, fs=fs, output="sos")


def clean_eeg(x, fs):
    y = sosfilt(_sos("bandpass", [0.5, 40], fs), x)
    b, a = iirnotch(50, 30, fs)
    return lfilter(b, a, y)


def clean_ecg(x, fs):
    return sosfilt(_sos("bandpass", [0.5, 40], fs), x)


def clean_emg(x, fs):
    return sosfilt(_sos("highpass", 20, fs), x)


def acc_xyz(raw, device):
    """3 x n accelerometer array for a device, or None if any axis is missing."""
    keys = [f"{DEVICES[device]} ACC {k}" for k in "XYZ"]
    if not all(k in raw for k in keys):
        return None
    return np.stack([raw[k][0] for k in keys])


def gravity_scale(acc_chunks, fs=25):
    """Median |a| over rest blocks across chunks: the reading that corresponds to 1 g.

    SeizeIT2 accelerometers are on inconsistent scales (gravity reads ~0.5 or ~0.167),
    so dividing by this value makes gravity read 1 g. None if no rest block is found.
    """
    n = REST_BLOCK_S * fs
    rest = []
    for acc in acc_chunks:
        mag = np.linalg.norm(acc, axis=0)
        blocks = mag[: len(mag) // n * n].reshape(-1, n)
        mean, std = blocks.mean(1), blocks.std(1)
        rest.append(mean[(mean > 0) & (std < REST_REL_STD * mean)])
    rest = np.concatenate(rest) if rest else np.array([])
    return float(np.median(rest)) if len(rest) else None


def clean_motion(raw, scales):
    """Dynamic acceleration (|a| - 1 g) and rotation magnitude per device, low-passed at 10 Hz.

    A device's accelerometer or gyroscope is left out (features become NaN) when any of its
    axes is stuck in this chunk, or when the accelerometer's gravity scale is implausible.
    """
    out, sos = {}, _sos("lowpass", 10, 25)
    for dev, name in DEVICES.items():
        acc = acc_xyz(raw, dev)
        if acc is not None and valid_scale(scales.get(dev)) and not any(stuck(a) for a in acc):
            out[f"ACC_{dev}"] = sosfilt(sos, np.linalg.norm(acc / scales[dev], axis=0) - 1.0)
        gyr = [f"{name} GYR {k}" for k in "ABC"]
        if all(k in raw for k in gyr) and not any(stuck(raw[k][0]) for k in gyr):
            out[f"GYR_{dev}"] = sosfilt(sos, np.linalg.norm(np.stack([raw[k][0] for k in gyr]), axis=0))
    return out


def clean_chunk(eeg, ecg, emg, mov, scales):
    """Raw read_physical() dicts for one chunk -> {signal name: cleaned array} (SIGNALS naming)."""
    out = {}
    eeg_ch = [k for k in eeg if k in EEG_NAMES]
    for name, key in zip(("EEG1", "EEG2"), eeg_ch):
        out[name] = clean_eeg(eeg[key][0], eeg[key][1])
    if "ECG SD" in ecg:
        out["ECG"] = clean_ecg(ecg["ECG SD"][0], ecg["ECG SD"][1])
    if "EMG SD" in emg:
        out["EMG"] = clean_emg(emg["EMG SD"][0], emg["EMG SD"][1])
    out.update(clean_motion(mov, scales))
    return out


def clean_one(raw, name, scales):
    """One cleaned signal (SIGNALS naming) from raw read_physical() dicts, without the stuck-axis check.

    Used for injected faults: the stuck check looks at the whole chunk, which would let a fault
    that starts later remove the sensor from the start. Returns None if the signal is absent.
    """
    if name in ("EEG1", "EEG2"):
        keys = [k for k in raw["eeg"] if k in EEG_NAMES]
        i = 0 if name == "EEG1" else 1
        return clean_eeg(raw["eeg"][keys[i]][0], FS[name]) if len(keys) > i else None
    if name == "ECG":
        return clean_ecg(raw["ecg"]["ECG SD"][0], FS[name]) if "ECG SD" in raw["ecg"] else None
    if name == "EMG":
        return clean_emg(raw["emg"]["EMG SD"][0], FS[name]) if "EMG SD" in raw["emg"] else None
    kind, dev = name.split("_")
    keys = [f"{DEVICES[dev]} {kind} {a}" for a in ("XYZ" if kind == "ACC" else "ABC")]
    if not all(k in raw["mov"] for k in keys):
        return None
    xyz = np.stack([raw["mov"][k][0] for k in keys])
    sos = _sos("lowpass", 10, 25)
    if kind == "GYR":
        return sosfilt(sos, np.linalg.norm(xyz, axis=0))
    if not valid_scale(scales.get(dev)):
        return None
    return sosfilt(sos, np.linalg.norm(xyz / scales[dev], axis=0) - 1.0)


def robust_stats(arrays):
    """Median and robust std (1.4826 x MAD) of concatenated arrays, for per-patient z-scoring."""
    x = np.concatenate(arrays)
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med))) or 1.0
