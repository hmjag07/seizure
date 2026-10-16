"""One output per second from look-back windows, labelled seizure or background.

At each whole second t, every signal contributes the window that ends at t:
4 s for EEG, EMG and motion, 15 s for ECG (enough heartbeats for heart rate).
A window is labelled seizure if its newest second (t-1, t] overlaps a seizure.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from seizure.data.fetch_plan import WARMUP_S

WINDOW_S = {"EEG1": 4, "EEG2": 4, "ECG": 15, "EMG": 4,
            "ACC_neck": 4, "ACC_chest": 4, "GYR_neck": 4, "GYR_chest": 4}


def window_ends(fetch_t0, t0, t1):
    """Whole seconds at which an output is produced: after warm-up, inside [t0, t1]."""
    first = int(np.ceil(max(t0, fetch_t0 + WARMUP_S)))
    return np.arange(first, int(t1) + 1)


def _stops(ends, fs, fetch_t0):
    return np.round((ends - fetch_t0) * fs).astype(int)


def fits(x, fs, length_s, fetch_t0, ends):
    """Mask of ends whose full window lies inside x (x starts at fetch_t0)."""
    stops = _stops(ends, fs, fetch_t0)
    return (stops >= int(length_s * fs)) & (stops <= len(x))


def windows(x, fs, length_s, fetch_t0, ends):
    """(len(ends), length_s * fs) view of x, one row per window ending at each time in ends.

    Call with ends already filtered by fits() for every signal, so rows line up across signals.
    """
    n = int(length_s * fs)
    return sliding_window_view(x, n)[_stops(ends, fs, fetch_t0) - n]


def label(ends, intervals):
    """1 where the newest second (t-1, t] overlaps any (onset, offset) interval."""
    y = np.zeros(len(ends), dtype=np.int8)
    for on, off in intervals:
        y[(ends > on) & (ends - 1 < off)] = 1
    return y
