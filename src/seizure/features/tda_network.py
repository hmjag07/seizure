"""TDA-B: persistent homology on the network of sensors (docs/pipeline.md, Step 4B and Phase B).

Per second t, from the cleaned, downsampled signals (data/signals.py):
1. Activity: each working sensor's log RMS in 0.25 s steps over the last WINDOW_S seconds, after the
   patient's normalisation (40 values for 10 s), so sensors with different sampling rates share one
   time base.
2. Distance between two sensors = 1 - |Pearson correlation of their activities|: sensors that rise and
   fall together are close.
3. Persistent homology (ripser) of that distance matrix: H0 (how the sensors merge into one group) and
   H1 (loops).
4. Features that do not depend on how many sensors are working: mean / min / max of the H0 merge
   distances and their entropy, H1 count / total / max, mean |correlation|, and the share of sensor pairs
   with |correlation| > 0.5.
A dead or switched-off sensor is left out of the network; with fewer than 3 working sensors the features
are missing.
"""
import numpy as np
from ripser import ripser

STEP_S = 0.25
WINDOW_S = 10
COUPLED = 0.5
NAMES = ("b_h0_mean", "b_h0_min", "b_h0_max", "b_h0_entropy", "b_h1_count", "b_h1_total", "b_h1_max",
         "b_corr_mean", "b_coupled_share")


def activity(x, fs, med, sd):
    """log RMS of the normalised signal in STEP_S blocks; block k covers [k * STEP_S, (k + 1) * STEP_S)."""
    n = int(round(STEP_S * fs))
    z = (x[: len(x) // n * n] - med) / sd
    return np.log10(np.sqrt((z.reshape(-1, n) ** 2).mean(1)) + 1e-6)


def _entropy(life):
    life = life[life > 0]
    if len(life) == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def network_features(acts, ends, fetch_t0, alive=None):
    """TDA-B features for each window end (whole seconds). acts: {sensor: activity series starting at
    fetch_t0}; alive: optional {sensor: bool array over ends}, False = sensor switched off at that second."""
    per_s = int(round(1 / STEP_S))
    L = WINDOW_S * per_s
    names = sorted(acts)
    out = np.full((len(ends), len(NAMES)), np.nan, dtype="float32")
    for i, t in enumerate(ends):
        stop = int(round((t - fetch_t0) * per_s))
        if stop < L:
            continue
        use = [s for s in names if (alive is None or alive[s][i]) and len(acts[s]) >= stop]
        if len(use) < 3:
            continue
        A = np.stack([acts[s][stop - L:stop] for s in use])
        A = A - A.mean(1, keepdims=True)
        norm = np.sqrt((A ** 2).sum(1))
        ok = norm > 1e-9
        if ok.sum() < 3:
            continue
        A, norm = A[ok], norm[ok]
        r = np.clip((A @ A.T) / np.outer(norm, norm), -1, 1)
        D = 1 - np.abs(r)
        np.fill_diagonal(D, 0)
        h0, h1 = ripser(D, maxdim=1, distance_matrix=True)["dgms"]
        d0 = h0[np.isfinite(h0[:, 1]), 1]
        l1 = h1[:, 1] - h1[:, 0] if len(h1) else np.array([])
        iu = np.triu_indices(len(D), 1)
        ar = np.abs(r[iu])
        out[i] = [d0.mean(), d0.min(), d0.max(), _entropy(d0), len(l1), l1.sum() if len(l1) else 0.0,
                  l1.max() if len(l1) else 0.0, ar.mean(), (ar > COUPLED).mean()]
    return {f"TDAB_{n}": out[:, j] for j, n in enumerate(NAMES)}
