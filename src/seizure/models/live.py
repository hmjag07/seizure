"""Phase C3: simulated live run and its timing (docs/research-log.md, Step 7).

    python -m seizure.models.live

Plays recordings second by second, as a bedside monitor would, through the complete system:
  1. cleaning     the last 75 s of every sensor, re-filtered each second (conservative: a live device
                  would keep its filter state and only filter the new second)
  2. features     classic + TDA-A for the windows ending now (features/build.sensor_features)
  3. quality      health-check quality features for each sensor (faults/health.py)
  4. context      10 s / 60 s context of the seizure features and the health features, from the history
  5. health       the health check for every sensor, smoothing and cut-off (faults/health.decide rule)
  5b. stuck rule  per-axis stuck check on the last 60 s of each motion sensor (faults/health.stuck_motion)
  6. model        the seizure model (A3 classic + TDA) on the current second, with switched-off sensors blanked
  7. alarm        5 of the last 10 seconds
Each step is timed every second. Uses training-set patients only (pilot) and fold-0 models, so no
held-out data is touched; the point is speed, not accuracy. Writes results/tables/phaseC3_timing.csv.
"""
import time
from collections import deque

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data import preprocess as pp
from seizure.data.windows import WINDOW_S
from seizure.faults.health import CONTEXT, QUALITY, quality_features, raw_signal, stuck_motion
from seizure.faults.train_health import features as health_features
from seizure.features import classic
from seizure.features.build import TABLES, load_params, load_plan, patient_setup, sensor_features, stream
from seizure.models.robustness import MODELS_DIR
from seizure.models.train import context_columns

PATIENTS = ("sub-004", "sub-076", "sub-031")
SECONDS_PER_PATIENT = 600
BUFFER_S = 75


def context_row(history, cols):
    """_m10 and _d60 of the latest row, from a list of feature dicts (oldest first)."""
    h = pd.DataFrame(list(history))[cols]
    m10 = h.iloc[-10:].mean()
    prev = h.iloc[max(0, len(h) - 70):max(1, len(h) - 10)].mean() if len(h) > 10 else m10 * np.nan
    return {**{f"{c}_m10": m10[c] for c in cols}, **{f"{c}_d60": m10[c] - prev[c] for c in cols}}


def main():
    params = load_params()
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "a3_classic+TDA_fold0.json")
    feat_names = list(model.get_booster().feature_names)
    health = XGBClassifier()
    health.load_model(MODELS_DIR / "health90_fold0.json")
    base_cols = [c for c in feat_names if not c.endswith(("_m10", "_d60"))]
    ctx_cols = [c for c in context_columns(base_cols)]
    timings = []
    for sub in PATIENTS:
        ctx = patient_setup(sub)
        row = ctx["plan"].query("kind == 'seizure'").iloc[0]
        raw = stream(sub, row.run, row.fetch_t0, row.t1, headers=ctx["headers"])
        scales, stats = ctx["scales"][row.run], ctx["stats"]
        start = int(row.fetch_t0 + BUFFER_S)
        history, q_hist = deque(maxlen=70), {s: deque(maxlen=310) for s in pp.SIGNALS}
        hits = deque(maxlen=10)
        p_fail = {s: deque(maxlen=15) for s in pp.SIGNALS}
        for t in range(start, min(start + SECONDS_PER_PATIENT, int(row.t1))):
            tm = {}
            t0 = time.perf_counter()
            a = t - BUFFER_S - row.fetch_t0
            cut = {m: {k: (v[0][int(a * v[1]):int((t - row.fetch_t0) * v[1])], v[1], v[2])
                       for k, v in d.items()} for m, d in raw.items()}
            sig = pp.clean_chunk(cut["eeg"], cut["ecg"], cut["emg"], cut["mov"], scales)
            tm["cleaning"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ends, f0 = np.array([t]), t - BUFFER_S
            feats = {}
            if "ECG" in sig:
                peaks = classic.ecg_rpeaks(sig["ECG"], pp.FS["ECG"])
                feats.update(classic.hr_features(peaks, ends, WINDOW_S["ECG"], f0))
            for name, x in sig.items():
                feats.update(sensor_features(name, x, stats, params, f0, ends))
            feats = {k: float(np.asarray(v)[0]) for k, v in feats.items()}
            tm["features (classic + TDA-A)"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for name, x in sig.items():
                q = quality_features(name, x, raw_signal(cut, name), stats, f0, ends)
                q_hist[name].append({c: float(q[f"{name}_{c}"][0]) for c in QUALITY})
            tm["quality"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            history.append({c: feats.get(c, np.nan) for c in base_cols})
            xrow = {**history[-1], **context_row(history, ctx_cols)}
            hrows = []
            for name in sig:
                hq = pd.DataFrame(list(q_hist[name]))
                m10, m60 = hq.iloc[-10:].mean(), hq.iloc[-70:-10].mean() if len(hq) > 10 else hq.iloc[-10:].mean() * np.nan
                r = {c: hq[c].iloc[-1] for c in QUALITY}
                r.update({f"{c}_m10": m10[c] for c in QUALITY})
                r.update({f"{c}_d60": m10[c] - m60[c] for c in QUALITY})
                r["q_jump_max60"] = hq.q_jump.iloc[-60:].max()
                r["q_flat_m60"] = hq.q_flat.iloc[-60:].mean()
                r["q_logstd_d300"] = (hq.q_logstd.iloc[-10:].mean() - hq.q_logstd.iloc[-310:-300].mean()
                                      if len(hq) > 300 else np.nan)
                r["sensor"] = name
                hrows.append(r)
            tm["context"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            hp = health.predict_proba(health_features(pd.DataFrame(hrows)[[*QUALITY, *CONTEXT, "sensor"]]))[:, 2]
            off = set()
            for name, pf in zip(sig, hp):
                p_fail[name].append(pf)
                if np.mean(p_fail[name]) >= 0.5:
                    off.add(name)
            tm["health check"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for name in sig:
                if name.startswith(("ACC_", "GYR_")) and stuck_motion(
                        cut["mov"], ctx["headers"][(row.run, "mov")], name, 25, ends, f0)[0]:
                    off.add(name)
            tm["stuck rule"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            X = pd.DataFrame([xrow]).reindex(columns=feat_names).astype("float32")
            for name in off:
                X.loc[:, [c for c in feat_names if c.removeprefix("TDA_").startswith(name + "_")]] = np.nan
            prob = float(model.predict_proba(X)[:, 1][0])
            tm["seizure model"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            hits.append(prob >= 0.4)
            _alarm = sum(hits) >= 5
            tm["alarm"] = time.perf_counter() - t0
            tm["total"] = sum(tm.values())
            timings.append(dict(subject=sub, t=t, **tm))
        print(f"{sub}: {SECONDS_PER_PATIENT} s simulated", flush=True)
    df = pd.DataFrame(timings)
    steps = [c for c in df.columns if c not in ("subject", "t")]
    summary = pd.DataFrame({"mean_ms": df[steps].mean() * 1000, "median_ms": df[steps].median() * 1000,
                            "p95_ms": df[steps].quantile(0.95) * 1000, "p99_ms": df[steps].quantile(0.99) * 1000,
                            "max_ms": df[steps].max() * 1000}).round(1)
    summary.to_csv(TABLES / "phaseC3_timing.csv")
    print(summary.to_string())
    print(f"seconds simulated: {len(df)}; share of seconds processed within 1 s: {(df.total < 1).mean():.4f}; "
          f"within 0.2 s: {(df.total < 0.2).mean():.4f}")


if __name__ == "__main__":
    main()
