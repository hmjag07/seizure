"""Signal-quality features and the healthy/degraded/failed classifier (docs/lab-notebook.md, Step 3).

Quality features are computed per sensor per window and do not depend on the sensor's units:
q_logstd   log10 of the spread of the normalised, cleaned window: falls for dropout and decline
q_range    (max - min) of the normalised window: rises with pops
q_kurt     kurtosis of the cleaned window: spiky windows (pops) score high
q_flat     share of consecutive raw samples that are identical: flat bursts (dropout)
q_jump     largest raw sample-to-sample jump divided by the typical one: sudden steps (pops)
Context versions (_m10, _d60, as in train.py) are added before the classifier, so slow changes
such as a gradual decline show up as a drift against the minute before.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from seizure.data import preprocess as pp
from seizure.data.windows import WINDOW_S, windows

QUALITY = ("q_logstd", "q_range", "q_kurt", "q_flat", "q_jump")
EEG_KEYS = pp.EEG_NAMES


def raw_signal(raw, name):
    """The unfiltered signal behind a cleaned signal: the electrode channel, or the axis magnitude."""
    if name in ("EEG1", "EEG2"):
        keys = [k for k in raw["eeg"] if k in EEG_KEYS]
        idx = 0 if name == "EEG1" else 1
        return raw["eeg"][keys[idx]][0] if len(keys) > idx else None
    if name == "ECG":
        return raw["ecg"].get("ECG SD", (None,))[0]
    if name == "EMG":
        return raw["emg"].get("EMG SD", (None,))[0]
    kind, dev = name.split("_")
    axes = "XYZ" if kind == "ACC" else "ABC"
    keys = [f"{pp.DEVICES[dev]} {kind} {a}" for a in axes]
    if not all(k in raw["mov"] for k in keys):
        return None
    return np.linalg.norm(np.stack([raw["mov"][k][0] for k in keys]), axis=0)


def quality_features(name, clean, raw_x, stats, fetch_t0, ends, fs=None, length_s=None):
    """Quality features of one sensor for the windows ending at ends (fs and window default to SeizeIT2's)."""
    fs, L = fs or pp.FS[name], length_s or WINDOW_S[name]
    med, sd = stats.get(name, (0.0, 1.0))
    w = windows((clean - med) / sd, fs, L, fetch_t0, ends)
    r = windows(raw_x, fs, L, fetch_t0, ends)
    d = np.abs(np.diff(r, axis=1))
    out = {f"{name}_q_logstd": np.log10(w.std(1) + 1e-6),
           f"{name}_q_range": w.max(1) - w.min(1),
           f"{name}_q_kurt": np.nan_to_num(kurtosis(w, axis=1), nan=0.0),
           f"{name}_q_flat": (d == 0).mean(1),
           f"{name}_q_jump": d.max(1) / (np.median(d, axis=1) + 1e-9)}
    return {k: v.astype("float32") for k, v in out.items()}


CONTEXT = ([f"{q}_m10" for q in QUALITY] + [f"{q}_d60" for q in QUALITY]
           + ["q_jump_max60", "q_flat_m60", "q_logstd_d300"])


def add_context(df, keys):
    """Context features within each series (rows sorted by series, then time); past seconds only.

    _m10    mean over the last 10 s
    _d60    that mean minus the mean of the 60 s before it
    q_jump_max60   largest jump in the last 60 s (remembers recent pops)
    q_flat_m60     mean flat share over the last 60 s (remembers recent dropouts)
    q_logstd_d300  10 s strength now minus 5 minutes ago (slow decline)
    """
    gid = df.groupby(keys, sort=False).ngroup().to_numpy()
    g = df[list(QUALITY)].groupby(gid)
    m10 = g.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
    m60 = g.rolling(60, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
    max60 = g.rolling(60, min_periods=1).max().reset_index(level=0, drop=True).reindex(df.index)
    df[[f"{q}_m10" for q in QUALITY]] = m10.to_numpy(dtype="float32")
    df[[f"{q}_d60" for q in QUALITY]] = (m10 - m60.groupby(gid).shift(10)).to_numpy(dtype="float32")
    df["q_jump_max60"] = max60["q_jump"].to_numpy(dtype="float32")
    df["q_flat_m60"] = m60["q_flat"].to_numpy(dtype="float32")
    df["q_logstd_d300"] = (m10["q_logstd"] - m10["q_logstd"].groupby(gid).shift(300)).to_numpy(dtype="float32")
    return df


def decide(p_failed, gid, cut, smooth=15, recover=30):
    """Switch a sensor off when its failed-probability, averaged over the last `smooth` seconds,
    reaches `cut`; switch it back on only after `recover` seconds in a row below `cut`.
    Rows must be sorted by series (gid), then time."""
    s = pd.Series(p_failed).groupby(gid).rolling(smooth, min_periods=1).mean().to_numpy()
    out = np.zeros(len(s), bool)
    state, below, prev = False, 0, None
    for i in range(len(s)):
        if gid[i] != prev:
            state, below, prev = False, 0, gid[i]
        if s[i] >= cut:
            state, below = True, 0
        elif state:
            below += 1
            if below >= recover:
                state = False
        out[i] = state
    return out


def stuck_trailing(axes, fs, ends, fetch_t0, window_s=60, constant=True, limits=None, rest_value=None,
                   rest_tol=0.0, companion=None):
    """Per-axis stuck rule on the past only (Phase C4). True at each window end where any axis, over the
    last window_s seconds,
      - is exactly constant (if `constant`). A constant equal to `rest_value` (a gyroscope reads exactly 0
        at rest: dead-zone) counts only when a `companion` axis (the same device's accelerometer) is also
        exactly constant, i.e. the whole device is dead. `rest_tol` is how close counts as equal (the
        stored zero is offset by a fraction of one quantisation step); or
      - sits at a pinned edge for more than half of the samples. The edge is the axis's full-scale limit
        when `limits` gives (low, high) per axis, the window's own min / max when limits == "window",
        and the pinned test is skipped when limits is None.
    The "window" edge also fires on healthy axes at rest (they flicker between one or two quantised
    values), so the full-scale edge is the one used. This is the live counterpart of preprocess.stuck,
    which looks at the whole chunk. It exists because the learned health check sees the axis magnitude,
    in which a single frozen axis hides behind the two moving ones."""
    n = int(window_s * fs)
    stops = np.round((np.asarray(ends) - fetch_t0) * fs).astype(int)
    out = np.zeros(len(stops), bool)

    def frozen(x, stop):
        seg = x[max(0, stop - n):stop]
        return len(seg) > 0 and seg.min() == seg.max()

    for j, x in enumerate(axes):
        for i, stop in enumerate(stops):
            if out[i]:
                continue
            seg = x[max(0, stop - n):stop]
            if len(seg) == 0:
                continue
            lo_w, hi_w = seg.min(), seg.max()
            if lo_w == hi_w:
                if constant and (rest_value is None or abs(lo_w - rest_value) > rest_tol
                                 or any(frozen(c, stop) for c in (companion or []))):
                    out[i] = True
                continue
            if limits is None:
                continue
            lo, hi = (lo_w, hi_w) if limits == "window" else limits[j]
            tol = 1e-6 * max(abs(lo), abs(hi), 1.0)
            if max(np.mean(np.abs(seg - lo) <= tol), np.mean(np.abs(seg - hi) <= tol)) > 0.5:
                out[i] = True
    return out


def stuck_motion(mov, header, sensor, fs, ends, fetch_t0, window_s=60):
    """The stuck rule as used by the system for one motion sensor (ACC_/GYR_ + neck/chest): full-scale
    pinning or exact constancy; a gyroscope frozen at exactly 0 only if its accelerometer is frozen too."""
    kind, dev = sensor.split("_")
    keys = [f"{pp.DEVICES[dev]} {kind} {a}" for a in ("XYZ" if kind == "ACC" else "ABC")]
    lim, steps = [], []
    for k in keys:
        i = header["labels"].index(k)
        lim.append((header["offset"][i] + header["gain"][i] * header["dmin"][i],
                    header["offset"][i] + header["gain"][i] * header["dmax"][i]))
        steps.append(abs(header["gain"][i]))
    axes = [mov[k][0] for k in keys]
    if kind == "ACC":
        return stuck_trailing(axes, fs, ends, fetch_t0, window_s, limits=lim)
    acc = [mov[f"{pp.DEVICES[dev]} ACC {a}"][0] for a in "XYZ" if f"{pp.DEVICES[dev]} ACC {a}" in mov]
    return stuck_trailing(axes, fs, ends, fetch_t0, window_s, limits=lim, rest_value=0.0,
                          rest_tol=2 * max(steps), companion=acc)


def health_class(eff):
    """0 healthy, 1 degraded, 2 failed, from the effective fault severity at the window's end."""
    return np.where(eff <= 0, 0, np.where(eff < 0.6, 1, 2)).astype(np.int8)
