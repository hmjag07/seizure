"""Baseline features: band powers, line length, HR/HRV, motion energy."""
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, lfilter, sosfilt, welch

EEG_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def _line_length(w):
    return np.abs(np.diff(w, axis=1)).mean(1)


def eeg_features(w, fs, name):
    f, p = welch(w, fs=fs, nperseg=fs, axis=1)
    out = {f"{name}_{b}": np.log10(p[:, (f >= lo) & (f < hi)].sum(1) + 1e-12) for b, (lo, hi) in EEG_BANDS.items()}
    out[f"{name}_logpow"] = np.log10(p[:, (f >= 1) & (f < 40)].sum(1) + 1e-12)
    out[f"{name}_linelen"] = _line_length(w)
    out[f"{name}_std"] = w.std(1)
    return out


def emg_features(w, name="EMG"):
    return {f"{name}_logrms": np.log10(w.std(1) + 1e-12),
            f"{name}_linelen": _line_length(w),
            f"{name}_zcr": (np.diff(np.sign(w), axis=1) != 0).mean(1)}


def motion_features(w, name):
    return {f"{name}_mean": np.abs(w).mean(1), f"{name}_std": w.std(1),
            f"{name}_max": np.abs(w).max(1), f"{name}_linelen": _line_length(w)}


def ecg_rpeaks(x, fs, ref_s=10):
    """R-peak times (s, from the start of x): 5-20 Hz band-pass, squared, 0.1 s moving average.

    The height threshold is 30% of the 99th percentile over the previous ref_s seconds only,
    so an artefact raises it for ref_s seconds instead of hiding every beat in the chunk.
    """
    y = sosfilt(butter(2, [5, 20], btype="bandpass", fs=fs, output="sos"), x) ** 2
    n = int(0.1 * fs)
    e = lfilter(np.ones(n) / n, 1, y)
    step = fs // 4  # 4 updates per second keeps the rolling percentile cheap
    ref = pd.Series(e[::step]).rolling(ref_s * 4, min_periods=4).quantile(0.99).bfill().to_numpy()
    thr = 0.3 * np.repeat(ref, step)[: len(e)]
    peaks, _ = find_peaks(e, distance=int(0.3 * fs))
    return peaks[e[peaks] > thr[peaks]] / fs


def hr_features(peak_times, ends, length_s, fetch_t0):
    """Heart rate (bpm) and RMSSD (ms) from R-peaks within each window (t - length_s, t]."""
    t = peak_times + fetch_t0
    hr, rmssd = np.full(len(ends), np.nan), np.full(len(ends), np.nan)
    lo, hi = np.searchsorted(t, ends - length_s, side="right"), np.searchsorted(t, ends, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        rr = np.diff(t[a:b])
        rr = rr[(rr > 0.3) & (rr < 2.0)]  # 30-200 bpm
        if len(rr) >= 3:
            hr[i] = 60 / np.median(rr)
            rmssd[i] = 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))
    return {"ECG_hr": hr, "ECG_rmssd": rmssd}
