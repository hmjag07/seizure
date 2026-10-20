"""Dead-sensor robustness test (docs/pipeline.md, Step 8).

    python -m seizure.models.robustness

A dead sensor gives no usable signal, so all of its features (base, TDA and context) are set to
missing. For 0-7 dead sensors out of 8, sampled combinations are scored with the clean-trained
fold models and each fold's clean-data threshold, kept fixed as a deployed monitor would.
Target: with 5 of 8 sensors dead (about 60%) and at least one EEG channel alive, sensitivity
drops by at most 15 percentage points.

Writes results/tables/robustness_dead_sensors.csv and saves the fold models to cache/models/.
"""
import itertools
import random
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import event_scores, window_scores
from seizure.features.build import CACHE, TABLES
from seizure.models.train import ALARM, N_FOLDS, PARAMS, SEED, feature_sets, prepare

MODELS_DIR = CACHE.parent / "models"
EEG = {"EEG1", "EEG2"}
PER_LEVEL = 12  # combinations with at least one EEG alive, per number of dead sensors
PER_LEVEL_NO_EEG = 6  # combinations with both EEG dead


def sensor_columns(cols):
    """{sensor: its columns}. 'ACC_neck_std_m10' and 'TDA_ACC_neck_H1_max' both belong to ACC_neck."""
    out = {s: [] for s in SIGNALS}
    for c in cols:
        base = c.removeprefix("TDA_")
        out[next(s for s in SIGNALS if base.startswith(s + "_"))].append(c)
    return out


def combinations(seed=SEED):
    rng = random.Random(seed)
    combos = [()]
    for k in range(1, len(SIGNALS)):
        allk = list(itertools.combinations(SIGNALS, k))
        alive = [c for c in allk if not EEG <= set(c)]
        dead = [c for c in allk if EEG <= set(c)]
        if k == 1:
            combos += allk
            continue
        combos += rng.sample(alive, min(PER_LEVEL, len(alive))) + rng.sample(dead, min(PER_LEVEL_NO_EEG, len(dead)))
    return combos


def train_fold_models(df, cols, name):
    models = []
    for k in range(N_FOLDS):
        tr = df.fold != k
        y = df.label[tr].to_numpy()
        m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        m.fit(df.loc[tr, cols], y)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        m.save_model(MODELS_DIR / f"clean_{name}_fold{k}.json")
        models.append(m)
    return models


def main():
    df, cols = prepare()
    thresholds = (pd.read_csv(TABLES / "model_clean_folds.csv")
                  .set_index(["model", "fold"]).threshold)
    combos = combinations()
    print(f"{len(combos)} sensor combinations", flush=True)
    rows = []
    for name, fc in feature_sets(cols).items():
        t0 = time.time()
        models = train_fold_models(df, fc, name)
        by_sensor = sensor_columns(fc)
        print(f"{name}: trained in {time.time() - t0:.0f} s", flush=True)
        for combo in combos:
            p = df[["subject", "run", "t", "label", "fold"]].copy()
            p["prob"], p["hit"] = np.nan, 0.0
            for k, m in enumerate(models):
                te = df.fold == k
                X = df.loc[te, fc].copy()
                for s in combo:
                    X[by_sensor[s]] = np.nan
                p.loc[te, "prob"] = m.predict_proba(X)[:, 1]
                # 1.0 where the fold's own clean-data threshold is reached, so one 0.5 cut applies to all folds.
                p.loc[te, "hit"] = (p.loc[te, "prob"] >= thresholds[(name, k)]).astype(float)
            e = event_scores(p, "hit", 0.5, *ALARM)
            rows.append(dict(model=name, dead=len(combo), sensors_dead=" ".join(combo) or "none",
                             eeg_alive=not EEG <= set(combo), **window_scores(p.label, p.prob), **e))
        print(f"{name}: {len(combos)} combinations scored in {time.time() - t0:.0f} s", flush=True)

    res = pd.DataFrame(rows).round(4)
    clean = res[res.dead == 0].set_index("model").sensitivity
    res["drop_pts"] = ((clean.reindex(res.model).to_numpy() - res.sensitivity) * 100).round(1)
    res.to_csv(TABLES / "robustness_dead_sensors.csv", index=False)

    summary = (res[res.eeg_alive].groupby(["model", "dead"])
               .agg(combos=("sensitivity", "size"), sens_mean=("sensitivity", "mean"),
                    sens_min=("sensitivity", "min"), drop_mean=("drop_pts", "mean"),
                    drop_max=("drop_pts", "max"), fa_mean=("fa_per_h", "mean")).round(3))
    print(summary.to_string())
    print(res[res.dead == 1].pivot(index="sensors_dead", columns="model", values="drop_pts").to_string())
    five = res[(res.dead == 5) & res.eeg_alive]
    print("\nTarget (5 of 8 dead, EEG alive, drop <= 15 points):")
    print(five.groupby("model").agg(mean_drop=("drop_pts", "mean"), worst_drop=("drop_pts", "max"),
                                    share_passing=("drop_pts", lambda d: (d <= 15).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
