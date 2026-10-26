"""Stream, clean, window and featurise the pilot patients (docs/research-log.md, Steps 1.3-1.5, 4A, 4C).

    python -m seizure.features.build params          # choose TDA delay/dimension on background data
    python -m seizure.features.build features [sub]  # all pilot patients, or one

Per patient, chunk by chunk to keep memory low:
1. Stream the motion file for every chunk and estimate the gravity scale per device per run.
2. Stream and clean the background chunks; per-signal median and robust std give the
   patient's normalisation.
3. For every chunk: clean, normalise, cut 1 s windows, label, compute classic and TDA-A features.
Output: cache/features/<subject>.parquet, one row per window.
"""
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from seizure.data import preprocess as pp
from seizure.data.labels import edf_url, read_events
from seizure.data.streaming import read_header, read_physical
from seizure.data.windows import WINDOW_S, fits, label, window_ends, windows
from seizure.features import classic, tda_embedding as tda

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "results" / "tables"
# SEIZURE_SET picks the patient set: pilot (default), heldout (data/heldout.py) or extra (the other 66
# training patients, data/fullset.py). Each has its own fetch plan and cache folders.
SET = os.environ.get("SEIZURE_SET", "pilot")
HELDOUT = SET == "heldout"
SUFFIX = "" if SET == "pilot" else f"_{SET}"
PLAN_FILE = TABLES / ("fetch_plan.csv" if SET == "pilot" else f"{SET}_fetch_plan.csv")
CACHE = ROOT / "cache" / f"features{SUFFIX}"
MODALITIES = ("eeg", "ecg", "emg", "mov")


def load_plan():
    p = pd.read_csv(PLAN_FILE, dtype={"run": str})
    p["run"] = p["run"].str.zfill(2)
    return p


def stream(sub, run, t0, t1, modalities=MODALITIES, headers=None):
    """read_physical for several modalities of one run, in parallel."""
    def one(m):
        url = edf_url(sub, run, m)
        h = headers[(run, m)] if headers is not None else read_header(url)
        return read_physical(url, h, t0, t1)
    with ThreadPoolExecutor(len(modalities)) as ex:
        return dict(zip(modalities, ex.map(one, modalities)))


def gravity_scales(sub, plan, headers):
    """{run: {device: scale}}, falling back to the patient's median when a run has no rest.

    Chunks with a stuck axis are ignored. A scale outside preprocess.SCALE_RANGE is kept as is,
    so clean_motion rejects that run's accelerometer instead of borrowing another run's scale.
    """
    per_run = {}
    for run, rows in plan.groupby("run"):
        movs = [stream(sub, run, r.fetch_t0, r.t1, ("mov",), headers)["mov"] for r in rows.itertuples()]
        per_run[run] = {}
        for dev in pp.DEVICES:
            accs = [a for a in (pp.acc_xyz(m, dev) for m in movs) if a is not None and not any(map(pp.stuck, a))]
            per_run[run][dev] = pp.gravity_scale(accs)
    for dev in pp.DEVICES:
        vals = [s[dev] for s in per_run.values() if pp.valid_scale(s[dev])]
        for s in per_run.values():
            if s[dev] is None and vals:
                s[dev] = float(np.median(vals))
    return per_run


def sensor_features(name, x, stats, params, fetch_t0, ends):
    """Classic and TDA-A features of one cleaned signal (not ECG heart rate) for windows ending at ends."""
    fs, L = pp.FS[name], WINDOW_S[name]
    med, sd = stats.get(name, (0.0, 1.0))
    w = windows((x - med) / sd, fs, L, fetch_t0, ends)
    out = {}
    if name.startswith("EEG"):
        out.update(classic.eeg_features(w, fs, name))
    elif name == "EMG":
        out.update(classic.emg_features(w))
    elif name != "ECG":
        out.update(classic.motion_features(w, name))
    if name in params:
        out.update(tda.tda_features(w, *params[name], name=f"TDA_{name}"))
    return out


def chunk_features(sub, row, raw, scales, stats, seizures, impd, params):
    sig = pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"], scales)
    ends = window_ends(row.fetch_t0, row.t0, row.t1)
    ok = np.ones(len(ends), bool)
    for name, x in sig.items():
        ok &= fits(x, pp.FS[name], WINDOW_S[name], row.fetch_t0, ends)
    ends = ends[ok]
    out = {"subject": sub, "run": row.run, "kind": row.kind, "t": ends,
           "label": label(ends, seizures), "impd": label(ends, impd)}
    if "ECG" in sig:
        peaks = classic.ecg_rpeaks(sig["ECG"], pp.FS["ECG"])
        out.update(classic.hr_features(peaks, ends, WINDOW_S["ECG"], row.fetch_t0))
    for name in pp.SIGNALS:
        if name in sig:
            out.update(sensor_features(name, sig[name], stats, params, row.fetch_t0, ends))
    return pd.DataFrame(out)


def patient_setup(sub):
    """Everything per patient that does not depend on a chunk: plan, headers, seizure and impedance
    intervals, gravity scales, and normalisation statistics from the background chunks.
    Also returns the streamed background chunks, so they need not be streamed twice."""
    plan = load_plan().query("subject == @sub")
    runs = sorted(plan.run.unique())
    headers = {(r, m): read_header(edf_url(sub, r, m)) for r in runs for m in MODALITIES}
    events = {r: read_events(sub, r) for r in runs}
    sz = {r: [(float(e["onset"]), float(e["onset"]) + float(e["duration"])) for e in ev
              if e["eventType"].startswith("sz")] for r, ev in events.items()}
    impd = {r: [(float(e["onset"]), float(e["onset"]) + float(e["duration"])) for e in ev
                if e["eventType"] == "impd"] for r, ev in events.items()}
    scales = gravity_scales(sub, plan, headers)

    # Patient normalisation from background chunks only (no seizure data).
    bg = {row.Index: stream(sub, row.run, row.fetch_t0, row.t1, headers=headers)
          for row in plan.query("kind == 'background'").itertuples()}
    collected = {}
    for idx, raw in bg.items():
        for name, x in pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"],
                                      scales[plan.loc[idx, "run"]]).items():
            collected.setdefault(name, []).append(x[int(30 * pp.FS[name]):])  # skip filter warm-up
    stats = {name: pp.robust_stats(xs) for name, xs in collected.items()}
    return dict(plan=plan, headers=headers, sz=sz, impd=impd, scales=scales, stats=stats, bg=bg)


def build_subject(sub, params):
    t_start = time.time()
    ctx = patient_setup(sub)
    plan, scales, bg = ctx["plan"], ctx["scales"], ctx["bg"]
    frames = []
    for row in plan.itertuples():
        raw = bg.pop(row.Index, None) or stream(sub, row.run, row.fetch_t0, row.t1, headers=ctx["headers"])
        frames.append(chunk_features(sub, row, raw, scales[row.run], ctx["stats"], ctx["sz"][row.run],
                                     ctx["impd"][row.run], params))
    df = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / f"{sub}.parquet")
    g = {r: {d: (round(v, 3) if v else None) for d, v in s.items()} for r, s in scales.items()}
    return sub, len(df), int(df.label.sum()), round(time.time() - t_start), g


def load_params():
    p = pd.read_csv(TABLES / "tda_params.csv")
    return {r.signal: (int(r.m), int(r.tau)) for r in p.itertuples()}


def choose_params(n_patients=10, windows_per_signal=20, seed=42):
    """Median AMI delay and FNN dimension per signal, from background windows of pilot patients."""
    rng = random.Random(seed)
    plan = load_plan().query("kind == 'background'")
    subs = rng.sample(sorted(plan.subject.unique()), n_patients)
    taus, ms = {}, {}
    for sub in subs:
        row = plan.query("subject == @sub").iloc[0]
        raw = stream(sub, row.run, row.fetch_t0, row.t1)
        scales = {d: pp.gravity_scale([a] if (a := pp.acc_xyz(raw["mov"], d)) is not None else [])
                  for d in pp.DEVICES}
        for name, x in pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"], scales).items():
            fs, n = pp.FS[name], int(WINDOW_S[name] * pp.FS[name])
            x = x[int(30 * fs):]
            med, sd = pp.robust_stats([x])
            x = (x - med) / sd
            # One random stream per patient and signal, so skipping a broken signal
            # does not change which windows another signal gets.
            for s in random.Random(f"{seed}-{sub}-{name}").sample(range(len(x) - n), windows_per_signal):
                w = x[s: s + n]
                tau = tda.ami_lag(w, max_lag=64 if fs == 256 else 10)
                taus.setdefault(name, []).append(tau)
                ms.setdefault(name, []).append(tda.fnn_dim(w, tau))
        print("done", sub, flush=True)
    rows = [dict(signal=k, tau=int(np.median(taus[k])), m=int(np.median(ms[k])),
                 tau_iqr=f"{np.percentile(taus[k], 25):.0f}-{np.percentile(taus[k], 75):.0f}",
                 m_iqr=f"{np.percentile(ms[k], 25):.0f}-{np.percentile(ms[k], 75):.0f}",
                 n_windows=len(taus[k]), patients=" ".join(subs)) for k in pp.SIGNALS if k in taus]
    pd.DataFrame(rows).to_csv(TABLES / "tda_params.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="patients").to_string(index=False))


def main():
    if sys.argv[1] == "params":
        choose_params()
        return
    params = load_params()
    subs = sys.argv[2:] or sorted(load_plan().subject.unique())
    with ProcessPoolExecutor(4) as ex:
        for sub, n, n_sz, secs, scales in ex.map(build_subject, subs, [params] * len(subs)):
            print(f"{sub}: {n} windows, {n_sz} seizure, {secs} s, gravity scales {scales}", flush=True)


if __name__ == "__main__":
    main()
