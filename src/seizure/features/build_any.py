"""The feature, signal-cache and fault builds for any dataset, through the adapter (data/datasets.py).

    python -m seizure.features.build_any <dataset> params           # TDA delay and dimension (background)
    python -m seizure.features.build_any <dataset> speed            # TDA time per window and channel
    python -m seizure.features.build_any <dataset> features [sub]   # Step 4: classic + TDA-A per second
    python -m seizure.features.build_any <dataset> signals [sub]    # cleaned, downsampled copy (TDA-B, CNN)
    python -m seizure.features.build_any <dataset> faults [sub]     # Steps 2-3: injected faults + quality
    python -m seizure.features.build_any <dataset> wlabels [sub]    # window-level fault labels (fix D)

The steps are the SeizeIT2 ones (features/build.py, data/signals.py, faults/build_faulty.py,
faults/window_labels.py) with every dataset-specific detail taken from the adapter; the same seeds, fault
types, severities, onsets and labelling rules. Outputs go to cache/<what>_<dataset>/.
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from seizure.data import datasets
from seizure.data import preprocess as pp
from seizure.data.signals import downsample
from seizure.data.windows import fits, label, window_ends, windows
from seizure.faults.build_faulty import ONSET_RANGE, generic, seed
from seizure.faults.health import quality_features
from seizure.faults.inject import FAULTS, SEVERITIES, inject, spread
from seizure.faults.window_labels import FLAT_FAILED, POP_DEGRADED, POP_FAILED
from seizure.features import classic, tda_embedding as tda
from seizure.features.build import TABLES

WORKERS = int(os.environ.get("BUILD_WORKERS", 6))


def params_file(ds):
    return TABLES / ("tda_params.csv" if ds.name == "seizeit2" else f"{ds.name}_tda_params.csv")


def load_params(ds):
    p = pd.read_csv(params_file(ds))
    return {r.signal: (int(r.m), int(r.tau)) for r in p.itertuples()}


def sensor_features(ds, name, x, stats, params, fetch_t0, ends):
    """Classic and TDA-A features of one cleaned signal for the windows ending at ends."""
    fs, L = ds.fs[name], ds.window_s[name]
    med, sd = stats.get(name, (0.0, 1.0))
    w = windows((x - med) / sd, fs, L, fetch_t0, ends)
    out, kind = {}, ds.kind(name)
    if kind == "eeg":
        out.update(classic.eeg_features(w, fs, name))
    elif kind == "emg":
        out.update(classic.emg_features(w))
    elif kind == "motion":
        out.update(classic.motion_features(w, name))
    if name in params:
        out.update(tda.tda_features(w, *params[name], name=f"TDA_{name}"))
    return out


def common_ends(ds, sig, row):
    ends = window_ends(row.fetch_t0, row.t0, row.t1)
    ok = np.ones(len(ends), bool)
    for name, x in sig.items():
        ok &= fits(x, ds.fs[name], ds.window_s[name], row.fetch_t0, ends)
    return ends[ok]


def chunk_features(ds, sub, row, raw, ctx, params):
    sig = ds.clean(raw, ctx, row)
    ends = common_ends(ds, sig, row)
    out = {"subject": sub, "run": row.run, "kind": row.kind, "t": ends,
           "label": label(ends, ctx["sz"][row.run]), "impd": label(ends, ctx["impd"][row.run])}
    for name in ds.signals:
        if name in sig and ds.kind(name) == "ecg":
            peaks = classic.ecg_rpeaks(sig[name], ds.fs[name])
            out.update(classic.hr_features(peaks, ends, ds.window_s[name], row.fetch_t0))
    for name in ds.signals:
        if name in sig:
            out.update(sensor_features(ds, name, sig[name], ctx["stats"], params, row.fetch_t0, ends))
    return pd.DataFrame(out)


def chunks(ds, sub, ctx):
    """(row, raw) for every chunk of the patient's plan; background chunks come from setup()."""
    for row in ctx["plan"].itertuples():
        raw = ctx["bg"].pop(row.Index, None) or ds.stream(sub, row, ctx)
        yield row, raw


# ---------------------------------------------------------------- params and speed

def choose_params(ds, n_patients=10, windows_per_signal=20, seed_=42):
    """Median AMI delay and FNN dimension, as features/build.choose_params. For a dataset whose signals
    are all one kind (CHB-MIT: 18 EEG channels), one pooled value is used for every signal."""
    rng = random.Random(seed_)
    plan = ds.load_plan().query("kind == 'background'")
    subs = rng.sample(sorted(plan.subject.unique()), min(n_patients, plan.subject.nunique()))
    taus, ms = {}, {}
    for sub in subs:
        ctx = ds.setup(sub)
        row = ctx["plan"].query("kind == 'background'").iloc[0]
        raw = ctx["bg"][row.name]
        for name, x in ds.clean(raw, ctx, row).items():
            fs, n = ds.fs[name], int(ds.window_s[name] * ds.fs[name])
            x = x[int(30 * fs):]
            med, sd = pp.robust_stats([x])
            x = (x - med) / sd
            for s in random.Random(f"{seed_}-{sub}-{name}").sample(range(len(x) - n), windows_per_signal):
                w = x[s: s + n]
                tau = tda.ami_lag(w, max_lag=64 if fs >= 256 else 10)
                key = ds.kind(name) if len({ds.kind(s) for s in ds.signals}) == 1 else name
                taus.setdefault(key, []).append(tau)
                ms.setdefault(key, []).append(tda.fnn_dim(w, tau))
        print("done", sub, flush=True)
    rows = []
    for name in ds.signals:
        key = name if name in taus else ds.kind(name)
        if key not in taus:
            continue
        rows.append(dict(signal=name, tau=int(np.median(taus[key])), m=int(np.median(ms[key])),
                         tau_iqr=f"{np.percentile(taus[key], 25):.0f}-{np.percentile(taus[key], 75):.0f}",
                         m_iqr=f"{np.percentile(ms[key], 25):.0f}-{np.percentile(ms[key], 75):.0f}",
                         n_windows=len(taus[key]), pooled=key != name, patients=" ".join(subs)))
    pd.DataFrame(rows).to_csv(params_file(ds), index=False)
    print(pd.DataFrame(rows).drop(columns="patients").to_string(index=False))


def speed(ds, n_windows=100):
    """ms per window for TDA-A and for all classic features, on one background chunk."""
    params = load_params(ds)
    sub = sorted(ds.load_plan().subject.unique())[0]
    ctx = ds.setup(sub)
    row = ctx["plan"].query("kind == 'background'").iloc[0]
    sig = ds.clean(ctx["bg"][row.name], ctx, row)
    ends = common_ends(ds, sig, row)[:n_windows]
    t_tda = t_cl = 0.0
    for name, x in sig.items():
        w = windows(x / x.std(), ds.fs[name], ds.window_s[name], row.fetch_t0, ends)
        t0 = time.perf_counter()
        tda.tda_features(w, *params[name], name=name)
        t_tda += time.perf_counter() - t0
        t0 = time.perf_counter()
        classic.eeg_features(w, ds.fs[name], name) if ds.kind(name) == "eeg" else None
        t_cl += time.perf_counter() - t0
    print(f"{ds.name}: {len(sig)} signals; per second of data: TDA-A {1000 * t_tda / len(ends):.1f} ms, "
          f"classic {1000 * t_cl / len(ends):.1f} ms ({1000 * t_tda / len(ends) / len(sig):.1f} ms TDA per signal)")


# ---------------------------------------------------------------- features

def build_features(args):
    ds, sub, params = args
    t0 = time.time()
    ctx = ds.setup(sub)
    df = pd.concat([chunk_features(ds, sub, row, raw, ctx, params) for row, raw in chunks(ds, sub, ctx)],
                   ignore_index=True)
    out = ds.folder("features")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{sub}.parquet")
    return sub, len(df), int(df.label.sum()), round(time.time() - t0)


# ---------------------------------------------------------------- signal cache (TDA-B, CNN)

def target_fs(ds, name):
    return {"eeg": 64, "ecg": 64, "emg": 128}.get(ds.kind(name), ds.fs[name])


def build_signals(args):
    ds, sub, _ = args
    t0 = time.time()
    ctx = ds.setup(sub)
    arrays, meta_chunks = {}, []
    for i, (row, raw) in enumerate(chunks(ds, sub, ctx)):
        clean = ds.clean(raw, ctx, row)
        for name, x in clean.items():
            arrays[f"{i}|{name}"] = downsample(x, ds.fs[name], target_fs(ds, name)).astype("float32")
        meta_chunks.append(dict(i=i, run=row.run, kind=row.kind, t0=row.t0, t1=row.t1, fetch_t0=row.fetch_t0,
                                signals=sorted(clean)))
    meta = dict(chunks=meta_chunks, fs={s: target_fs(ds, s) for s in ds.signals},
                stats={k: list(map(float, v)) for k, v in ctx["stats"].items()})
    out = ds.folder("signals")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{sub}.npz", meta=np.array(json.dumps(meta)), **arrays)
    return sub, len(meta_chunks), round(sum(a.nbytes for a in arrays.values()) / 1e6, 1), round(time.time() - t0)


# ---------------------------------------------------------------- injected faults (Steps 2-3)

def damage(ds, raw, name, fault, s, onset, fetch_t0, rng, floor=0.0):
    """Copy of raw with one signal's channels damaged from onset; effective severity per sample.
    floor: smallest fault scale (SeizeIT2 motion: 5% of gravity in the device's units, ds.scale_floor)."""
    new = {m: dict(d) for m, d in raw.items()}
    eff = None
    for m, k in ds.channels(raw, name):
        x, fs, unit = raw[m][k]
        start = int(round((onset - fetch_t0) * fs))
        scale = max(spread(x[:start]), floor)
        y, e = inject(fault, x, fs, start, s, rng, scale)
        new[m][k] = (y, fs, unit)
        eff = e if eff is None else np.maximum(eff, e)
    return new, eff


def fault_chunk(ds, sub, row, raw, ctx, params):
    stats = ctx["stats"]
    clean = ds.clean(raw, ctx, row)
    ends = common_ends(ds, clean, row)
    base = {"subject": sub, "run": row.run, "kind": row.kind}
    sz = label(ends, ctx["sz"][row.run])
    quality = {"t": ends, "label": sz, **base}
    for name, x in clean.items():
        r = ds.raw_signal(raw, name)
        if r is not None:
            quality.update(quality_features(name, x, r, stats, row.fetch_t0, ends, ds.fs[name], ds.window_s[name]))
    faulty = []
    for fault in FAULTS:
        for s in SEVERITIES:
            rng_onset = np.random.default_rng(seed(sub, row.run, row.t0, fault, s))
            onset = float(row.t0 + rng_onset.uniform(*ONSET_RANGE) * (row.t1 - row.t0))
            after = ends > onset
            if not after.any():
                continue
            e_after = ends[after]
            for name in clean:
                rng = np.random.default_rng(seed(sub, row.run, row.t0, fault, s, name))
                new, eff = damage(ds, raw, name, fault, s, onset, row.fetch_t0, rng, ds.scale_floor(name, ctx, row))
                if eff is None:
                    continue
                x = ds.clean_one(new, name, ctx, row)
                if x is None:
                    continue
                fs = ds.fs[name]
                out = {"t": e_after, "label": sz[after], "fault": fault, "severity": s, "sensor": name,
                       "onset": onset,
                       "eff": eff[np.minimum((e_after - row.fetch_t0) * fs, len(eff)).astype(int) - 1], **base}
                feats = {}
                if ds.kind(name) == "ecg":
                    feats.update(classic.hr_features(classic.ecg_rpeaks(x, fs), e_after, ds.window_s[name],
                                                     row.fetch_t0))
                feats.update(sensor_features(ds, name, x, stats, params, row.fetch_t0, e_after))
                feats.update(quality_features(name, x, ds.raw_signal(new, name), stats, row.fetch_t0, e_after,
                                              fs, ds.window_s[name]))
                out.update({k: np.asarray(v, dtype="float32") for k, v in generic(name, feats).items()})
                faulty.append(pd.DataFrame(out))
    return pd.DataFrame(quality), faulty


def build_faults(args):
    ds, sub, params = args
    t0 = time.time()
    ctx = ds.setup(sub)
    quality, faulty = [], []
    for row, raw in chunks(ds, sub, ctx):
        q, f = fault_chunk(ds, sub, row, raw, ctx, params)
        quality.append(q)
        faulty += f
    out = ds.folder("faults")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(quality, ignore_index=True).to_parquet(out / f"{sub}_quality.parquet")
    fdf = pd.concat(faulty, ignore_index=True)
    fdf["eff"] = fdf["eff"].astype("float32")
    fdf.to_parquet(out / f"{sub}_faulty.parquet")
    return sub, len(fdf), round(time.time() - t0)


# ---------------------------------------------------------------- window-level fault labels (fix D)

def n_channels(ds, name):
    return 3 if ds.kind(name) == "motion" else 1


def build_wlabels(args):
    """As faults/window_labels.py: replay the seeded faults on dummy arrays to see what each window got."""
    from seizure.faults.window_labels import replay
    ds, sub, _ = args
    t0 = time.time()
    plan = ds.load_plan().query("subject == @sub")
    folder = ds.folder("faults")
    f = pd.read_parquet(folder / f"{sub}_faulty.parquet",
                        columns=["run", "t", "fault", "severity", "sensor", "onset", "eff"])
    out = []
    for row in plan.itertuples():
        mine = f[(f.run == row.run) & (f.onset >= row.t0) & (f.onset <= row.t1)]
        for (fault, s, sensor), g in mine.groupby(["fault", "severity", "sensor"]):
            fs = ds.fs[sensor]
            n = int(round((row.t1 - row.fetch_t0) * fs))
            start = int(round((float(g.onset.iloc[0]) - row.fetch_t0) * fs))
            t = g.t.to_numpy()
            stop = np.minimum(np.round((t - row.fetch_t0) * fs).astype(int), n)
            L = int(ds.window_s[sensor] * fs)
            if fault == "decline":
                dmg = g.eff.to_numpy(dtype=float)
                cls = np.where(dmg >= 0.6, 2, np.where(dmg > 0, 1, 0))
            else:
                rng = np.random.default_rng(seed(sub, row.run, row.t0, fault, s, sensor))
                flat, step = replay(fault, s, n, fs, start, rng, n_channels(ds, sensor))
                if fault == "dropout":
                    cs = np.concatenate([[0], np.cumsum(flat)])
                    dmg = (cs[stop] - cs[np.maximum(stop - L, 0)]) / L
                    cls = np.where(dmg >= FLAT_FAILED, 2, np.where(dmg > 0, 1, 0))
                else:
                    wmax = maximum_filter1d(step, size=L, origin=(L - 1) // 2)
                    dmg = wmax[np.maximum(stop - 1, 0)]
                    cls = np.where(dmg >= POP_FAILED, 2, np.where(dmg >= POP_DEGRADED, 1, 0))
            out.append(pd.DataFrame({"run": row.run, "t": t, "fault": fault, "severity": s, "sensor": sensor,
                                     "wclass": cls.astype("int8"), "wdamage": dmg.astype("float32")}))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(folder / f"{sub}_wlabels.parquet")
    return sub, len(res), round(time.time() - t0)


STEPS = {"features": (build_features, "features", "{}.parquet"),
         "signals": (build_signals, "signals", "{}.npz"),
         "faults": (build_faults, "faults", "{}_faulty.parquet"),
         "wlabels": (build_wlabels, "faults", "{}_wlabels.parquet")}


def main():
    ds = datasets.get(sys.argv[1])
    step = sys.argv[2]
    if step == "params":
        return choose_params(ds)
    if step == "speed":
        return speed(ds)
    fn, folder, pattern = STEPS[step]
    subs = sys.argv[3:] or sorted(ds.load_plan().subject.unique())
    subs = [s for s in subs if not (ds.folder(folder) / pattern.format(s)).exists()]
    params = load_params(ds) if step in ("features", "faults") else None
    print(f"{ds.name} {step}: {len(subs)} patients to build, {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        for res in ex.map(fn, [(ds, s, params) for s in subs]):
            print(*res, flush=True)


if __name__ == "__main__":
    main()
