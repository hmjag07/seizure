"""Sensor-aware thresholds: robustness at an equal false-alarm budget (docs/pipeline.md, Steps 6 and 8).

    python -m seizure.models.sensor_aware

A monitor knows which sensors are dead (the health check, Step 3), so it can use a threshold set
for that combination of working sensors. For each combination in the robustness test and each
fold, the threshold is the one with the highest sensitivity at <= FA_BUDGET false alarms per hour
on the OTHER folds' patients (with the same sensors dead); it is then applied to the held-out fold.
This compares every failure scenario at the same false-alarm budget as clean data.

Uses the fold models saved by robustness.py (normal and dropout training).
Writes results/tables/robustness_sensor_aware.csv.
"""
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.evaluation.metrics import FastEvents
from seizure.features.build import TABLES
from seizure.models.robustness import EEG, MODELS_DIR, combinations, sensor_columns
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, feature_sets, prepare

THRESHOLDS = np.unique(np.round(np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.96, 0.025),
                                                [0.97, 0.98, 0.99]]), 3))


def load_models(training, name):
    models = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"{training}_{name}_fold{k}.json")
        models.append(m)
    return models


def best_threshold(fe, prob, look_ahead=3, budget=FA_BUDGET):
    """Highest sensitivity at <= budget false alarms/h (default FA_BUDGET).

    False alarms fall as the threshold rises, so a halving search finds the lowest threshold within
    budget (about 6 scorings instead of 46); the next few thresholds up are also checked in case
    one catches more seizures. Ties go to the higher threshold.
    """
    def score(i):
        d, n, fa, h = fe.score(prob >= THRESHOLDS[i], *ALARM)
        return d / n, fa / h

    lo, hi = 0, len(THRESHOLDS) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if score(mid)[1] <= budget:
            hi = mid
        else:
            lo = mid + 1
    best, best_sens = len(THRESHOLDS) - 1, -1.0
    for i in range(lo, min(lo + look_ahead + 1, len(THRESHOLDS))):
        sens, fa = score(i)
        if fa <= budget and sens >= best_sens:
            best, best_sens = i, sens
    return THRESHOLDS[best]


def main():
    df, cols = prepare()
    meta = df[["subject", "run", "t", "label", "fold"]]
    folds = df.fold.to_numpy()
    other = {k: FastEvents(meta[folds != k]) for k in range(N_FOLDS)}
    held = {k: FastEvents(meta[folds == k]) for k in range(N_FOLDS)}
    combos = combinations()
    rows = []
    for training in ("clean", "dropout"):
        for name, fc in feature_sets(cols).items():
            t0 = time.time()
            models, by_sensor = load_models(training, name), sensor_columns(fc)
            for combo in combos:
                prob = np.empty(len(df))
                for k, m in enumerate(models):
                    X = df.loc[folds == k, fc].copy()
                    for s in combo:
                        X[by_sensor[s]] = np.nan
                    prob[folds == k] = m.predict_proba(X)[:, 1]
                det = n = fa = 0
                hours = 0.0
                thrs = []
                for k in range(N_FOLDS):
                    thr = best_threshold(other[k], prob[folds != k])
                    d, nk, f, h = held[k].score(prob[folds == k] >= thr, *ALARM)
                    det, n, fa, hours = det + d, n + nk, fa + f, hours + h
                    thrs.append(thr)
                rows.append(dict(training=training, model=name, dead=len(combo),
                                 sensors_dead=" ".join(combo) or "none", eeg_alive=not EEG <= set(combo),
                                 sensitivity=det / n, detected=det, seizures=n, false_alarms=fa,
                                 fa_per_h=fa / hours, thresholds=" ".join(f"{t:g}" for t in thrs)))
            print(f"{training} {name}: {len(combos)} combinations in {time.time() - t0:.0f} s", flush=True)

    res = pd.DataFrame(rows)
    clean = res[res.dead == 0].set_index(["training", "model"]).sensitivity
    res["drop_pts"] = ((clean.reindex(pd.MultiIndex.from_frame(res[["training", "model"]])).to_numpy()
                        - res.sensitivity) * 100).round(1)
    res.round(4).to_csv(TABLES / "robustness_sensor_aware.csv", index=False)

    alive = res[res.eeg_alive]
    print(alive.groupby(["training", "model", "dead"]).agg(
        sens=("sensitivity", "mean"), sens_min=("sensitivity", "min"), fa=("fa_per_h", "mean"),
        drop=("drop_pts", "mean"), drop_max=("drop_pts", "max")).round(3).to_string())
    print(res[res.dead == 1].pivot_table(index="sensors_dead", columns=["training", "model"],
                                         values="drop_pts").to_string())
    five = alive[alive.dead == 5]
    print("\nTarget (5 of 8 dead, EEG alive, drop <= 15 points, at <= 11 FA/h):")
    print(five.groupby(["training", "model"]).agg(
        mean_drop=("drop_pts", "mean"), worst_drop=("drop_pts", "max"), mean_fa=("fa_per_h", "mean"),
        share_passing=("drop_pts", lambda d: (d <= 15).mean())).round(3).to_string())
    print("\nBoth EEG dead:")
    print(res[~res.eeg_alive].groupby(["training", "model", "dead"])[["sensitivity", "fa_per_h"]]
          .mean().round(3).unstack([0, 1]).to_string())


if __name__ == "__main__":
    main()
