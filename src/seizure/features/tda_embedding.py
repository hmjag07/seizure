"""TDA-A: delay embedding + persistent homology per signal.

A window x is turned into a point cloud of delay vectors (x[t], x[t+tau], ..., x[t+(m-1)tau]).
Rhythmic activity traces loops in this cloud; noise makes a shapeless blob. Persistent homology
(ripser) measures the clusters (H0) and loops (H1), summarised as a few numbers per window.
"""
import numpy as np
from ripser import ripser
from sklearn.neighbors import NearestNeighbors

N_POINTS = 150  # 300 took 130 ms per diagram, 150 takes ~21 ms (measured 2026-10-16)


def ami_lag(x, max_lag, bins=16):
    """Delay tau: first local minimum of the average mutual information between x[t] and x[t+lag]."""
    q = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    mi = []
    for lag in range(1, max_lag + 1):
        joint = np.histogram2d(q[:-lag], q[lag:], bins=bins)[0]
        p = joint / joint.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        mi.append(float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()))
        if len(mi) >= 3 and mi[-2] < mi[-3] and mi[-2] <= mi[-1]:
            return lag - 1
    return int(np.argmin(mi)) + 1


def fnn_dim(x, tau, max_m=6, rtol=10.0, atol=2.0, threshold=0.05):
    """Dimension m: smallest m where under `threshold` of nearest neighbours are false (Kennel et al. 1992)."""
    sd = x.std()
    for m in range(1, max_m):
        n = len(x) - m * tau
        if n < 20:
            return m
        emb = embed(x, m, tau)[:n]
        dist, idx = NearestNeighbors(n_neighbors=2).fit(emb).kneighbors(emb)
        d, j = dist[:, 1], idx[:, 1]
        extra = np.abs(x[np.arange(n) + m * tau] - x[j + m * tau])
        false = (extra / np.maximum(d, 1e-12) > rtol) | (np.sqrt(d ** 2 + extra ** 2) / sd > atol)
        if false.mean() < threshold:
            return m
    return max_m


def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.stack([x[i * tau: i * tau + n] for i in range(m)], axis=1)


def _summary(dgm):
    """Count, total persistence, longest bar and persistent entropy of one diagram."""
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life) & (life > 0)]
    if len(life) == 0:
        return 0, 0.0, 0.0, 0.0
    p = life / life.sum()
    return len(life), float(life.sum()), float(life.max()), float(-(p * np.log(p)).sum())


def window_features(x, m, tau, n_points=N_POINTS):
    pts = embed(x, m, tau)
    if len(pts) > n_points:
        pts = pts[np.linspace(0, len(pts) - 1, n_points).astype(int)]
    h0, h1 = ripser(pts, maxdim=1)["dgms"]
    _, h0_tot, h0_max, h0_ent = _summary(h0)
    h1_n, h1_tot, h1_max, h1_ent = _summary(h1)
    return h0_tot, h0_max, h0_ent, h1_n, h1_tot, h1_max, h1_ent


NAMES = ("H0_total", "H0_max", "H0_entropy", "H1_count", "H1_total", "H1_max", "H1_entropy")


def tda_features(w, m, tau, name):
    """Features for each row of a (n_windows, n_samples) array."""
    f = np.array([window_features(row, m, tau) for row in w], dtype=float).reshape(len(w), len(NAMES))
    return {f"{name}_{k}": f[:, i] for i, k in enumerate(NAMES)}
