"""Gradient-boosted classifier with patient-grouped CV (docs/pipeline.md, Step 5).

    python -m seizure.models.train

1. Per-patient normalisation of every feature (median and MAD over all of the patient's
   seconds, without using labels).
2. Short-term context: for key features, the mean over the last 10 s and its change from
   the mean of the minute before (past seconds only, within each chunk).
3. XGBoost on three feature sets with the same 5 patient-grouped folds, so every patient is
   predicted by a model that never saw them.
4. Alarm threshold per fold, chosen on the other folds' patients only: the threshold with the
   highest sensitivity at <= FA_BUDGET false alarms per hour, under the 5-of-10 alarm rule.

Out-of-fold predictions go to cache/predictions/, scores to results/tables/model_clean_*.csv.
Sensor-dropout augmentation (5.3) comes later.
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from seizure.evaluation.metrics import event_scores, window_scores
from seizure.features.build import CACHE, TABLES
from seizure.features.report import META, load

SEED = 42
N_FOLDS = 5
SHAPE = ("H0_entropy", "H1_count", "H1_entropy")
TDA_CONTEXT = ("H0_total", "H1_total", "H1_max") + SHAPE
FA_BUDGET = 11  # false alarms per hour, as in success target T1
ALARM = (5, 10)  # x of the last y seconds
THRESHOLDS = np.round(np.arange(0.1, 0.96, 0.05), 2)
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="aucpr")


def normalise(df, cols):
    """Robust z-score per patient: (x - median) / (1.4826 * MAD) over all of the patient's seconds."""
    med = df.groupby("subject")[cols].transform("median")
    mad = (df[cols] - med).abs().groupby(df.subject).transform("median") * 1.4826
    df[cols] = ((df[cols] - med) / mad.replace(0, np.nan)).astype("float32")


def segments(df):
    """Id of each run of consecutive seconds (one fetched chunk); df must be sorted."""
    return ((df.t.diff() != 1) | (df.subject != df.subject.shift()) | (df.run != df.run.shift())).cumsum()


def add_context(df, cols):
    """Mean over the last 10 s (_m10) and its change from the mean of the 60 s before that (_d60)."""
    seg = segments(df)
    g = df[cols].groupby(seg)
    m10 = g.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    m60 = g.rolling(60, min_periods=1).mean().reset_index(level=0, drop=True)
    new = pd.concat([m10.add_suffix("_m10"), (m10 - m60.groupby(seg).shift(10)).add_suffix("_d60")], axis=1)
    return pd.concat([df, new.astype("float32")], axis=1), list(new.columns)


def context_columns(cols):
    return [c for c in cols if not c.startswith("TDA_") or c.endswith(TDA_CONTEXT)]


def feature_sets(cols):
    """Classic, classic + TDA, classic + TDA shape only; context columns follow their base feature."""
    def base(c):
        return c.removesuffix("_m10").removesuffix("_d60")
    classic = [c for c in cols if not c.startswith("TDA_")]
    tda = [c for c in cols if c.startswith("TDA_")]
    return {"classic": classic,
            "classic+TDA": classic + tda,
            "classic+TDA_shape": classic + [c for c in tda if base(c).endswith(SHAPE)]}


def folds(df):
    """Fold number per row; whole patients per fold."""
    fold = np.empty(len(df), dtype=int)
    for k, (_, test) in enumerate(GroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                  .split(df, groups=df.subject)):
        fold[test] = k
    return fold


def cross_validate(df, cols, fold):
    prob = np.empty(len(df))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        y = df.label.to_numpy()[tr]
        model = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        model.fit(df.loc[tr, cols], y)
        prob[te] = model.predict_proba(df.loc[te, cols])[:, 1]
    return prob


def pick_threshold(p, col):
    """Threshold with the highest sensitivity at <= FA_BUDGET false alarms per hour."""
    best, best_sens = THRESHOLDS[-1], -1.0
    for thr in THRESHOLDS:
        e = event_scores(p, col, thr, *ALARM)
        if e["fa_per_h"] <= FA_BUDGET and e["sensitivity"] > best_sens:
            best, best_sens = thr, e["sensitivity"]
    return best


def evaluate(p, col):
    """Per-fold threshold chosen on the other folds, applied to the held-out fold."""
    per_fold, totals = [], dict(seizures=0, detected=0, false_alarms=0, hours=0.0)
    for k, g in p.groupby("fold"):
        thr = pick_threshold(p[p.fold != k], col)
        e = event_scores(g, col, thr, *ALARM)
        per_fold.append(dict(fold=k, threshold=thr, **window_scores(g.label, g[col]), **e))
        totals["seizures"] += e["seizures"]
        totals["detected"] += e["detected"]
        totals["false_alarms"] += e["false_alarms"]
        totals["hours"] += (g.label == 0).sum() / 3600
    return per_fold, dict(seizures=totals["seizures"], detected=totals["detected"],
                          sensitivity=totals["detected"] / totals["seizures"],
                          false_alarms=totals["false_alarms"], fa_per_h=totals["false_alarms"] / totals["hours"],
                          median_delay_s=float(np.median([f["median_delay_s"] for f in per_fold])))


def prepare():
    df = load().sort_values(["subject", "run", "t"]).reset_index(drop=True)
    df["fold"] = folds(df)
    cols = [c for c in df.columns if c not in META + ["fold"]]
    normalise(df, cols)
    df, ctx = add_context(df, context_columns(cols))
    return df, cols + ctx


def main():
    df, cols = prepare()
    fold_patients = df.groupby("fold").subject.unique().map(lambda s: " ".join(sorted(s)))
    print(fold_patients.to_string(), "\n")

    rows, per_fold = [], []
    preds = df[META + ["fold"]].copy()
    for name, fc in feature_sets(cols).items():
        t0 = time.time()
        preds[name] = cross_validate(df, fc, df.fold.to_numpy())
        pf, total = evaluate(preds, name)
        rows.append(dict(model=name, features=len(fc), **window_scores(preds.label, preds[name]), **total))
        per_fold += [dict(model=name, patients=fold_patients[f["fold"]], **f) for f in pf]
        print(f"{name}: {len(fc)} features, {time.time() - t0:.0f} s", flush=True)

    (CACHE.parent / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_parquet(CACHE.parent / "predictions" / "clean_oof.parquet")
    res, pf = pd.DataFrame(rows).round(4), pd.DataFrame(per_fold).round(4)
    res.to_csv(TABLES / "model_clean_summary.csv", index=False)
    pf.to_csv(TABLES / "model_clean_folds.csv", index=False)
    print(res.to_string(index=False))
    print(pf[["model", "fold", "threshold", "auroc", "sensitivity", "fa_per_h"]].to_string(index=False))


if __name__ == "__main__":
    main()
