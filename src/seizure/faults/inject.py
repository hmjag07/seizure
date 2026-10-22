"""Electrode pop, gradual decline and intermittent dropout, with ground-truth fault labels.

Each fault damages one raw signal from an onset time onwards, at a severity s between 0 and 1,
and returns the damaged copy plus the effective severity at every sample (0 = healthy), which is
the ground truth for the health check (Step 3). Amplitudes are relative to the signal's own robust
spread before the onset, so the same severity means the same damage on every sensor.

pop       Sudden steps that decay back (electrode pops, cable or motion artefacts).
          Rate grows with s (up to 6 per minute); step size up to 20 x the signal's spread.
decline   Rising impedance: over RAMP_S the signal shrinks to (1 - s) of its size while noise
          grows to s x its spread; it then stays at that level. At s = 1 it is pure noise.
dropout   Short flat bursts (0.5-5 s) at zero, covering a fraction s of the time after onset.
          At s = 1 the signal is flat from the onset (a dead sensor).
"""
import numpy as np

FAULTS = ("pop", "decline", "dropout")
SEVERITIES = (0.25, 0.5, 0.75, 1.0)
RAMP_S = 300  # decline reaches full severity 5 minutes after onset
POP_RATE_PER_MIN = 6
POP_SIZE = 20
POP_DECAY_S = (0.3, 2.0)
BURST_S = (0.5, 5.0)


def spread(x):
    """Robust standard deviation (1.4826 x MAD)."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) or float(np.std(x)) or 1.0


def pop(x, fs, start, s, rng, scale):
    y, eff = x.copy(), np.zeros(len(x))
    n_after = len(x) - start
    n_pops = rng.poisson(s * POP_RATE_PER_MIN * n_after / fs / 60)
    for p in rng.integers(start, len(x), n_pops):
        size = s * POP_SIZE * scale * rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
        tau = rng.uniform(*POP_DECAY_S) * fs
        k = np.arange(len(x) - p)
        y[p:] += size * np.exp(-k / tau)
    eff[start:] = s
    return y, eff


def decline(x, fs, start, s, rng, scale):
    y, eff = x.copy(), np.zeros(len(x))
    k = np.arange(len(x) - start)
    progress = np.minimum(k / (RAMP_S * fs), 1.0)
    level = s * progress
    y[start:] = x[start:] * (1 - level) + rng.normal(0, 1, len(k)) * level * scale
    eff[start:] = level
    return y, eff


def dropout(x, fs, start, s, rng, scale):
    y, eff = x.copy(), np.zeros(len(x))
    eff[start:] = s
    if s >= 1:
        y[start:] = 0.0
        return y, eff
    n_after, target = len(x) - start, s * (len(x) - start)
    covered, mask = 0, np.zeros(len(x), bool)
    while covered < target:
        length = int(rng.uniform(*BURST_S) * fs)
        p = rng.integers(start, len(x))
        seg = mask[p:p + length]
        covered += (~seg).sum()
        mask[p:p + length] = True
        if mask[start:].all():
            break
    y[mask] = 0.0
    return y, eff


def inject(fault, x, fs, start, s, rng, scale=None):
    """Damaged copy of x from sample `start` on, and the effective severity per sample."""
    scale = spread(x[:start]) if scale is None else scale
    return {"pop": pop, "decline": decline, "dropout": dropout}[fault](x, fs, start, s, rng, scale)
