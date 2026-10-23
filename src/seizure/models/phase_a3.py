"""Phase A3: training with injected faults, evaluated on all 90 training patients.

    python -m seizure.models.phase_a3 health    # health check on the 90-patient folds
    python -m seizure.models.phase_a3 models    # A1 and A3 fold models
    python -m seizure.models.phase_a3 eval      # clean data and the 192 fault scenarios
    python -m seizure.models.phase_a3 all       # the three in order

A1 model: sensor-dropout training (Step 5.3), as in phase_a.py.
A3 model: the same, plus fault-augmented rows. For each of the 12 fault conditions (3 faults x 4 severities)
a random set of 1-5 sensors is made faulty from the onset (faults/evaluate.assemble); 5% of the seconds
after the onset are added to training, with every truly failed window blanked, because the health
check would switch that sensor off. So the model learns from what reaches it after the health check.

Everything uses the same 5 patient-grouped folds over the 90 patients (pilot + extra). The health check is
retrained on those folds (7% of whole series per patient, to fit in memory), so no patient is judged by
a health check or seizure model that saw them. Thresholds are chosen on the other folds at <= 8 false
alarms per hour (train.SELECT_BUDGET); results are judged against the 11 per hour budget.

Writes cache/models/{health90,a1,a3}_*.json and results/tables/phaseA3_*.csv.
"""
import itertools
import random
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import event_scores, window_scores
from seizure.faults import evaluate as ev
from seizure.faults import train_health as th
from seizure.faults.health import decide
from seizure.faults.inject import FAULTS, SEVERITIES
from seizure.features.build import ROOT, TABLES
from seizure.models.phase_a import load_all
from seizure.models.robustness import MODELS_DIR, sensor_columns
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, SELECT_BUDGET, feature_sets

PILOT_FAULTS, EXTRA_FAULTS = ROOT / "cache" / "faults", ROOT / "cache" / "faults_extra"
HEALTH_SAMPLE = 0.07
AUG_SHARE = 0.05
NAMES = ("classic", "classic+TDA")
TRAININGS = ("A1", "A3")


def faults_dir(subject):
    return PILOT_FAULTS if (PILOT_FAULTS / f"{subject}_faulty.parquet").exists() else EXTRA_FAULTS


def fold_map(data):
    return data.df.groupby("subject").fold.first()


def train_health(data):
    t0 = time.time()
    fmap = fold_map(data)
    df = pd.concat([th.patient_rows(s, faults_dir(s), HEALTH_SAMPLE) for s in fmap.index], ignore_index=True)
    df["fold"] = df.subject.map(fmap)
    print(f"health: {len(df)} rows, classes {np.bincount(df.y)}, {time.time() - t0:.0f} s", flush=True)
    X = th.features(df)
    p_failed = np.zeros(len(df))
    for k in range(N_FOLDS):
        tr, te = (df.fold != k).to_numpy(), (df.fold == k).to_numpy()
        m = XGBClassifier(**th.PARAMS)
        m.fit(X[tr], df.y[tr])
        m.save_model(MODELS_DIR / f"health90_fold{k}.json")
        p_failed[te] = m.predict_proba(X[te])[:, 2]
    gid = df.groupby("sid", sort=False).ngroup().to_numpy()
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    offs = {c: decide(p_failed, gid, c) for c in th.CUTS}
    cuts, off = {}, np.zeros(len(df), bool)
    for k in range(N_FOLDS):
        o = fold != k
        cuts[k] = float(max(th.CUTS, key=lambda c: th.f1_failed(offs[c][o], y[o])))
        off[fold == k] = offs[cuts[k]][fold == k]
    pd.Series(cuts, name="cutoff").rename_axis("fold").to_csv(TABLES / "phaseA3_health_cutoffs.csv")
    sw = pd.Series(off).groupby(df.y.map(dict(enumerate(th.CLASSES)))).mean().round(3)
    clean_off = pd.Series(off[df.fault == "none"]).groupby(df.sensor[df.fault == "none"].to_numpy()).mean().round(3)
    pd.concat({"switched_off_by_true_class": sw}).to_csv(TABLES / "phaseA3_health_switch_off.csv")
    clean_off.rename("healthy_sensor_switched_off").to_csv(TABLES / "phaseA3_health_clean_sensors.csv")
    print(f"health cut-offs {cuts}\nswitched off by true class:\n{sw.to_string()}\n"
          f"healthy sensors switched off:\n{clean_off.to_string()}\n{time.time() - t0:.0f} s", flush=True)


def fault_augmentation(data):
    """Rows for A3: 12 fault conditions, each on a random set of 1-5 sensors, 5% of the seconds after
    the onset, truly failed windows blanked. Returns (features, labels, folds)."""
    rng, pyrng = np.random.default_rng(SEED), random.Random(SEED)
    by_sensor = sensor_columns(data.cols)
    parts = []
    for fault, sev in itertools.product(FAULTS, SEVERITIES):
        t0 = time.time()
        f = ev.load_condition(data, faults_dir, fault, sev)
        sset = tuple(pyrng.sample(SIGNALS, pyrng.randint(1, 5)))
        X, failed = ev.assemble(data, f, sset)
        after = np.zeros(len(X), bool)
        after[f[f.sensor.isin(sset)].pos.to_numpy()] = True
        rows = np.flatnonzero(after & (rng.random(len(X)) < AUG_SHARE))
        A = X.iloc[rows].copy()
        for s in sset:
            A.loc[failed[s][rows], by_sensor[s]] = np.nan
        parts.append((A, data.df.label.to_numpy()[rows], data.fold[rows]))
        print(f"augmentation {fault} {sev}: sensors {' '.join(sset)}, {len(rows)} rows, {time.time() - t0:.0f} s",
              flush=True)
        del X, f
    return (pd.concat([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def blank_random_sensors(arr, idx_by_sensor, rng):
    """In place: 1-6 random sensors blanked per row (the same rule as robustness.dropout_copy)."""
    n = len(arr)
    k = rng.integers(1, 7, n)
    dead = rng.random((n, len(SIGNALS))).argsort(1).argsort(1) < k[:, None]
    for j, s in enumerate(SIGNALS):
        if idx_by_sensor[s]:
            arr[np.ix_(dead[:, j], idx_by_sensor[s])] = np.nan


def train_models(data):
    """One pre-allocated table per fold: clean rows, a dropout copy, then the fault-augmented rows.
    A1 trains on the first two parts (a view, no copy), A3 on all three."""
    fsets = feature_sets(data.cols)
    aug_X, aug_y, aug_fold = fault_augmentation(data)
    for nm in NAMES:
        fc = fsets[nm]
        pos = {c: i for i, c in enumerate(fc)}
        idx_by_sensor = {s: [pos[c] for c in cs] for s, cs in sensor_columns(fc).items()}
        for k in range(N_FOLDS):
            t0 = time.time()
            tr = data.fold != k
            keep = aug_fold != k
            n, n_aug = int(tr.sum()), int(keep.sum())
            table = np.empty((2 * n + n_aug, len(fc)), dtype="float32")
            table[:n] = data.df.loc[tr, fc].to_numpy(dtype="float32")
            table[n:2 * n] = table[:n]
            blank_random_sensors(table[n:2 * n], idx_by_sensor, np.random.default_rng(SEED + k))
            table[2 * n:] = aug_X.loc[keep, fc].to_numpy(dtype="float32")
            y_clean = data.df.label.to_numpy()[tr]
            y = np.concatenate([y_clean, y_clean, aug_y[keep]])
            for training, rows in (("A1", 2 * n), ("A3", len(table))):
                X = pd.DataFrame(table[:rows], columns=fc, copy=False)
                yt = y[:rows]
                m = XGBClassifier(**PARAMS, scale_pos_weight=(yt == 0).sum() / (yt == 1).sum())
                m.fit(X, yt)
                m.save_model(MODELS_DIR / f"{training.lower()}_{nm}_fold{k}.json")
            del table
            print(f"{nm} fold {k}: A1 and A3 trained ({n_aug} fault rows), {time.time() - t0:.0f} s", flush=True)


def load(training, nm):
    out = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"{training.lower()}_{nm}_fold{k}.json")
        out.append(m)
    return out


def evaluate(data):
    t_start = time.time()
    fsets = feature_sets(data.cols)
    models = {(tr, nm): load(tr, nm) for tr in TRAININGS for nm in NAMES}
    thresholds = {key: ev.Thresholds(data, ms, fsets[key[1]], *key, budget=SELECT_BUDGET)
                  for key, ms in models.items()}
    print(f"thresholds ready, {time.time() - t_start:.0f} s", flush=True)

    clean = []
    for (tr, nm), ms in models.items():
        prob, thr = ev.predict(data, ms, data.df[data.cols], fsets[nm], thresholds[(tr, nm)], None)
        p = data.df[["subject", "run", "t", "label", "fold"]].assign(hit=(prob >= thr).astype(float))
        fold_fa = []
        for k in range(N_FOLDS):
            d, n, fa, h = data.events[k].score(p.hit.to_numpy()[data.fold == k] > 0.5, *ALARM)
            fold_fa.append(fa / h)
        e = event_scores(p, "hit", 0.5, *ALARM)
        clean.append(dict(training=tr, model=nm, **window_scores(p.label, prob), sensitivity=e["sensitivity"],
                          fa_per_h=e["fa_per_h"], median_delay_s=e["median_delay_s"],
                          folds_over_budget=sum(f > FA_BUDGET for f in fold_fa),
                          fa_folds=" ".join(f"{f:.1f}" for f in fold_fa)))
    clean = pd.DataFrame(clean).round(4)
    clean.to_csv(TABLES / "phaseA3_clean.csv", index=False)
    print(clean.to_string(index=False), flush=True)

    health = [XGBClassifier() for _ in range(N_FOLDS)]
    for k, m in enumerate(health):
        m.load_model(MODELS_DIR / f"health90_fold{k}.json")
    ev.HEALTH_CUTS.update(pd.read_csv(TABLES / "phaseA3_health_cutoffs.csv").set_index("fold").cutoff.to_dict())
    clean_q = pd.concat([pd.read_parquet(faults_dir(s) / f"{s}_quality.parquet") for s in data.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(data.df[["subject", "run", "t"]])).reset_index(drop=True))
    out = TABLES / "phaseA3_faults.csv"
    rows = ev.run_scenarios(data, faults_dir, list(itertools.product(FAULTS, SEVERITIES)), ev.sensor_sets(),
                            models, thresholds, fsets, health, clean_q, out, modes=("A", "B"))
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"done in {time.time() - t_start:.0f} s")


def main():
    step = sys.argv[1]
    data = ev.Data(raw=load_all())
    print(f"{data.df.subject.nunique()} patients, {len(data.df)} seconds", flush=True)
    if step in ("health", "all"):
        train_health(data)
    if step in ("models", "all"):
        train_models(data)
    if step in ("eval", "all"):
        evaluate(data)


if __name__ == "__main__":
    main()
