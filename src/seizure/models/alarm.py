"""X-of-Y alarm rule with optional health weighting."""
import numpy as np


def alarm(prob, threshold=0.5, x=5, y=10):
    """Alarm at second t when at least x of the last y seconds (t-y+1 .. t) have prob >= threshold.

    prob must be one contiguous run of seconds. Uses past seconds only, like a live monitor;
    the first y-1 seconds can never alarm.
    """
    hits = (np.asarray(prob) >= threshold).astype(int)
    count = np.convolve(hits, np.ones(y, dtype=int))[: len(hits)]
    out = count >= x
    out[: y - 1] = False
    return out
