"""Health check: healthy / degraded / failed per sensor per second (docs/research-log.md, Step 3).

    python -m seizure.faults.train_health

Training data: the clean quality features (healthy) and the fault-injected ones (faults/build_faulty.py),
labelled by what the fault did inside each window (faults/window_labels.py): healthy, degraded or failed.
The first version labelled every second by the fault's overall severity (results kept with the suffix
_severity_labels); those labels are still stored as y_severity. A faulty sensor's series is the clean seconds before the fault onset followed
by the faulty seconds, so the context features see the change.

Inputs: the five quality features, their context (health.add_context: 10 s mean, change against the
minute before, 60 s memory of jumps and flat samples, 5-minute strength trend) and which sensor it is.
XGBoost, multi-class, the same 5 patient-grouped folds as the seizure model. A random 20% of whole
series (not single seconds) is kept, so the series stay continuous for smoothing.

Switch-off decision (health.decide): the failed-probability is averaged over 15 s; the sensor is
switched off when that reaches a cut-off and back on after 30 s below it. The cut-off of each fold is
chosen on the other folds' patients as the one with the best F1 for "failed" (balancing failed sensors
left on against working sensors switched off).

Writes cache/models/health_fold<k>.json, cache/predictions/health_oof.parquet and
results/tables/health_*.csv.
"""
import time
import zlib

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.faults.build_faulty import OUT
from seizure.faults.health import CONTEXT, QUALITY, add_context, decide, health_class
from seizure.features.build import CACHE, TABLES

MODELS_DIR = CACHE.parent / "models"
SAMPLE = 0.2  # share of series kept per patient, to fit in memory
SEED = 42
CLASSES = ("healthy", "degraded", "failed")
CUTS = np.round(np.arange(0.2, 0.86, 0.05), 2)
PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", n_jobs=-1, random_state=SEED, objective="multi:softprob")


def keep_series(keys, sample=SAMPLE):
    """True for about `sample` of the series, decided by a hash of the series key (reproducible)."""
    return np.array([zlib.crc32(k.encode()) % 1000 < sample * 1000 for k in keys])


def patient_rows(sub, out=OUT, sample=SAMPLE, signals=SIGNALS):
    """Clean and faulty quality rows of one patient, with context features and labels."""
    # Two adjacent background chunks share their boundary second (once in the data: sub-055); keep one.
    q = (pd.read_parquet(out / f"{sub}_quality.parquet").sort_values(["run", "t"])
         .drop_duplicates(["run", "t"]))
    q["seg"] = ((q.t.diff() != 1) | (q.run != q.run.shift())).cumsum()
    clean = []
    for s in signals:
        cols = [f"{s}_{c}" for c in QUALITY]
        if cols[0] not in q:
            continue
        part = q[["subject", "run", "seg", "t", "label"] + cols].dropna(subset=cols)
        clean.append(part.rename(columns=dict(zip(cols, QUALITY))).assign(sensor=s))
    clean = pd.concat(clean, ignore_index=True).assign(fault="none", severity=0.0, onset=np.nan, eff=0.0)

    f = pd.read_parquet(out / f"{sub}_faulty.parquet",
                        columns=["subject", "run", "t", "label", "fault", "severity", "sensor", "onset", "eff",
                                 *QUALITY]).drop_duplicates(["run", "t", "fault", "severity", "sensor"])
    seg_of = q.set_index(["run", "t"]).seg
    f["seg"] = seg_of.reindex(pd.MultiIndex.from_frame(f[["run", "t"]])).to_numpy()
    # Each faulty series: the clean seconds of its chunk before the onset, then the faulty seconds.
    starts = f.groupby(["seg", "fault", "severity", "sensor"]).onset.first().reset_index()
    pre = starts.merge(clean.drop(columns=["fault", "severity", "onset"]), on=["seg", "sensor"])
    pre = pre[pre.t <= pre.onset]
    rows = pd.concat([clean, pre, f], ignore_index=True)
    rows["sid"] = (sub + "|" + rows.fault + "|" + rows.severity.astype(str) + "|" + rows.sensor
                   + "|" + rows.seg.astype(str))
    rows = rows[keep_series(rows.sid.to_numpy(), sample)].sort_values(["sid", "t"]).reset_index(drop=True)
    rows = add_context(rows, "sid")
    # Keep clean series whole, and faulty series from the onset on (their clean lead-in only fed the context).
    rows = rows[(rows.fault == "none") | (rows.t > rows.onset)].reset_index(drop=True)
    # Window-level labels (window_labels.py): what the fault did inside each window. Clean rows are healthy.
    wl = pd.read_parquet(out / f"{sub}_wlabels.parquet").drop_duplicates(["run", "t", "fault", "severity", "sensor"])
    keys = ["run", "t", "fault", "severity", "sensor"]
    rows = rows.merge(wl, on=keys, how="left")
    rows["y"] = np.where(rows.fault == "none", 0, rows.wclass.fillna(0)).astype("int8")
    rows["y_severity"] = health_class(rows.eff.to_numpy())  # the first labels, kept for comparison
    return rows


def features(df, signals=SIGNALS):
    X = df[[*QUALITY, *CONTEXT]].copy()
    for s in signals:
        X[f"is_{s}"] = (df.sensor == s).astype("int8")
    return X


def f1_failed(off, y):
    tp = (off & (y == 2)).sum()
    return 2 * tp / max(off.sum() + (y == 2).sum(), 1)


def main():
    t0 = time.time()
    fold_of = (pd.read_parquet(CACHE.parent / "predictions" / "clean_oof.parquet", columns=["subject", "fold"])
               .drop_duplicates().set_index("subject").fold)
    subs = sorted(p.name.removesuffix("_faulty.parquet") for p in OUT.glob("*_faulty.parquet"))
    df = pd.concat([patient_rows(s) for s in subs], ignore_index=True)
    df["fold"] = df.subject.map(fold_of)
    print(f"{len(subs)} patients, {len(df)} rows, classes {np.bincount(df.y)}, {time.time() - t0:.0f} s", flush=True)

    X = features(df)
    prob = np.zeros((len(df), 3), dtype="float32")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for k in sorted(df.fold.unique()):
        tr, te = (df.fold != k).to_numpy(), (df.fold == k).to_numpy()
        m = XGBClassifier(**PARAMS)
        m.fit(X[tr], df.y[tr])
        m.save_model(MODELS_DIR / f"health_fold{k}.json")
        prob[te] = m.predict_proba(X[te])
    df["pred"] = prob.argmax(1)
    df[["p_healthy", "p_degraded", "p_failed"]] = prob
    print(f"trained, {time.time() - t0:.0f} s", flush=True)

    # Switch-off decisions for every cut-off, then each fold's cut-off from the other folds.
    gid = df.groupby("sid", sort=False).ngroup().to_numpy()
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    offs = {c: decide(df.p_failed.to_numpy(), gid, c) for c in CUTS}
    cuts, off = {}, np.zeros(len(df), bool)
    for k in sorted(df.fold.unique()):
        other = fold != k
        cuts[k] = max(CUTS, key=lambda c: f1_failed(offs[c][other], y[other]))
        off[fold == k] = offs[cuts[k]][fold == k]
    df["off"] = off
    df["off_old"] = df.pred == 2  # previous rule: failed if it is the most likely class, no smoothing
    pd.Series(cuts, name="cutoff").rename_axis("fold").to_csv(TABLES / "health_cutoffs.csv")

    (CACHE.parent / "predictions").mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in CONTEXT]).to_parquet(CACHE.parent / "predictions" / "health_oof.parquet")

    conf = pd.crosstab(df.y.map(dict(enumerate(CLASSES))), df.pred.map(dict(enumerate(CLASSES))),
                       normalize="index").round(3)
    conf.to_csv(TABLES / "health_confusion.csv")
    switched = (df.groupby(df.y.map(dict(enumerate(CLASSES))))[["off_old", "off"]].mean().round(3)
                .rename(columns={"off_old": "switched_off_before", "off": "switched_off_now"}))
    switched.to_csv(TABLES / "health_switch_off.csv")
    by = (df[df.fault != "none"].assign(flagged=lambda d: d.pred > 0)
          .groupby(["fault", "severity"]).flagged.mean().unstack().round(3))
    by.to_csv(TABLES / "health_by_fault.csv")
    per_sensor = (df[df.fault == "none"].groupby("sensor")[["off_old", "off"]].mean().round(3)
                  .rename(columns={"off_old": "healthy_switched_off_before", "off": "healthy_switched_off_now"}))
    per_sensor.to_csv(TABLES / "health_by_sensor.csv")
    print(f"cut-offs per fold: {cuts}")
    print("Rows = true class, columns = most likely class (share of each true class):")
    print(conf.to_string())
    print("\nShare switched off, by true class (before: most likely = failed; now: smoothed + cut-off):")
    print(switched.to_string())
    print("\nHealthy seconds switched off, by sensor:")
    print(per_sensor.to_string())
    print("\nShare of faulty seconds flagged (degraded or failed), by fault and severity:")
    print(by.to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
