"""Fault-injected features for every pilot patient (docs/research-log.md, Steps 2-3).

    python -m seizure.faults.build_faulty [sub ...]

For every chunk, fault type (pop, decline, dropout) and severity (25, 50, 75, 100%), a fault onset
is drawn between 20% and 60% of the way through the chunk (seeded, the same for every sensor, so
faults on several sensors can later be combined). Then, for each sensor separately, its raw signal
is damaged from the onset on, cleaned, and its features (classic + TDA-A, as in Step 4) are
recomputed for the seconds after the onset. Seconds before the onset, and all other sensors, keep
their clean features from cache/features, so any combination of faulty sensors can be assembled
without recomputing.

Health-check quality features (faults/health.py) are stored for the clean data and for every
faulty sensor, with the effective fault severity at each second as the ground truth.

Output per patient in cache/faults/: <sub>_faulty.parquet (one row per second, fault, severity
and sensor) and <sub>_quality.parquet (clean quality features).
"""
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from seizure.data import preprocess as pp
from seizure.data.windows import WINDOW_S, fits, label, window_ends
from seizure.faults.health import quality_features, raw_signal
from seizure.faults.inject import FAULTS, SEVERITIES, inject, spread
from seizure.features import classic
from seizure.features.build import CACHE, SUFFIX, load_params, load_plan, patient_setup, sensor_features, stream

OUT = CACHE.parent / f"faults{SUFFIX}"
ONSET_RANGE = (0.2, 0.6)
MOTION_POP_FLOOR = 0.05  # motion pops are at least 5% of gravity, since a still axis barely varies
WORKERS = 6


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def sensor_channels(raw, name):
    """(modality, key) of the raw channels behind one cleaned signal."""
    if name in ("EEG1", "EEG2"):
        keys = [k for k in raw["eeg"] if k in pp.EEG_NAMES]
        i = 0 if name == "EEG1" else 1
        return [("eeg", keys[i])] if len(keys) > i else []
    if name == "ECG":
        return [("ecg", "ECG SD")] if "ECG SD" in raw["ecg"] else []
    if name == "EMG":
        return [("emg", "EMG SD")] if "EMG SD" in raw["emg"] else []
    kind, dev = name.split("_")
    keys = [f"{pp.DEVICES[dev]} {kind} {a}" for a in ("XYZ" if kind == "ACC" else "ABC")]
    return [("mov", k) for k in keys] if all(k in raw["mov"] for k in keys) else []


def damage(raw, name, fault, s, onset, fetch_t0, scales, rng):
    """Copy of raw with one sensor's channels damaged from onset; effective severity per sample."""
    new = {m: dict(d) for m, d in raw.items()}
    eff = None
    for m, k in sensor_channels(raw, name):
        x, fs, unit = raw[m][k]
        start = int(round((onset - fetch_t0) * fs))
        scale = spread(x[:start])
        if m == "mov":
            dev = name.split("_")[1]
            scale = max(scale, MOTION_POP_FLOOR * (scales.get(dev) or 1.0))
        y, e = inject(fault, x, fs, start, s, rng, scale)
        new[m][k] = (y, fs, unit)
        eff = e if eff is None else np.maximum(eff, e)
    return new, eff


def generic(name, feats):
    """Drop the sensor from feature names ('EEG1_linelen' -> 'linelen', 'TDA_EEG1_H1_max' -> 'TDA_H1_max',
    'ECG_hr' -> 'hr'), so rows of every sensor share one narrow set of columns."""
    out = {}
    for k, v in feats.items():
        if k.startswith(f"TDA_{name}_"):
            out["TDA_" + k.removeprefix(f"TDA_{name}_")] = v
        else:
            out[k.removeprefix(f"{name}_")] = v
    return out


def chunk_rows(sub, row, raw, ctx, params):
    scales, stats = ctx["scales"][row.run], ctx["stats"]
    clean = pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"], scales)
    ends = window_ends(row.fetch_t0, row.t0, row.t1)
    ok = np.ones(len(ends), bool)
    for name, x in clean.items():
        ok &= fits(x, pp.FS[name], WINDOW_S[name], row.fetch_t0, ends)
    ends = ends[ok]
    base = {"subject": sub, "run": row.run, "kind": row.kind}
    sz = label(ends, ctx["sz"][row.run])

    quality = {"t": ends, "label": sz, **base}
    for name, x in clean.items():
        r = raw_signal(raw, name)
        if r is not None:
            quality.update(quality_features(name, x, r, stats, row.fetch_t0, ends))

    faulty = []
    for fault in FAULTS:
        for s in SEVERITIES:
            rng_onset = np.random.default_rng(seed(sub, row.run, row.t0, fault, s))
            span = row.t1 - row.t0
            onset = float(row.t0 + rng_onset.uniform(*ONSET_RANGE) * span)
            after = ends > onset
            if not after.any():
                continue
            e_after = ends[after]
            for name in clean:  # sensors already broken in the clean data are skipped
                rng = np.random.default_rng(seed(sub, row.run, row.t0, fault, s, name))
                new, eff = damage(raw, name, fault, s, onset, row.fetch_t0, scales, rng)
                if eff is None:
                    continue
                x = pp.clean_one(new, name, scales)
                if x is None:
                    continue
                fs = pp.FS[name]
                out = {"t": e_after, "label": sz[after], "fault": fault, "severity": s, "sensor": name,
                       "onset": onset, "eff": eff[np.minimum((e_after - row.fetch_t0) * fs, len(eff)).astype(int) - 1],
                       **base}
                feats = {}
                if name == "ECG":
                    peaks = classic.ecg_rpeaks(x, fs)
                    feats.update(classic.hr_features(peaks, e_after, WINDOW_S["ECG"], row.fetch_t0))
                feats.update(sensor_features(name, x, stats, params, row.fetch_t0, e_after))
                feats.update(quality_features(name, x, raw_signal(new, name), stats, row.fetch_t0, e_after))
                out.update({k: np.asarray(v, dtype="float32") for k, v in generic(name, feats).items()})
                faulty.append(pd.DataFrame(out))
    return pd.DataFrame(quality), faulty


def build_patient(sub, params):
    t0 = time.time()
    ctx = patient_setup(sub)
    quality, faulty = [], []
    for row in ctx["plan"].itertuples():
        raw = ctx["bg"].pop(row.Index, None) or stream(sub, row.run, row.fetch_t0, row.t1, headers=ctx["headers"])
        q, f = chunk_rows(sub, row, raw, ctx, params)
        quality.append(q)
        faulty += f
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(quality, ignore_index=True).to_parquet(OUT / f"{sub}_quality.parquet")
    fdf = pd.concat(faulty, ignore_index=True)
    fdf["eff"] = fdf["eff"].astype("float32")
    fdf.to_parquet(OUT / f"{sub}_faulty.parquet")
    return sub, len(fdf), round(time.time() - t0)


def main():
    params = load_params()
    subs = sys.argv[1:] or sorted(load_plan().subject.unique())
    subs = [s for s in subs if not (OUT / f"{s}_faulty.parquet").exists()]
    print(f"{len(subs)} patients to build", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        for sub, n, secs in ex.map(build_patient, subs, [params] * len(subs)):
            print(f"{sub}: {n} faulty rows, {secs} s", flush=True)


if __name__ == "__main__":
    main()
