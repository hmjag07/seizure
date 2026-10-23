"""Phase A: train on all 90 usable training patients, and false-alarm-aware thresholds.

    python -m seizure.models.phase_a a1    # cross-validation on 90 patients (pilot + extra)
    python -m seizure.models.phase_a a2    # threshold strategies on the a1 out-of-fold predictions

A1: the same pipeline as train.py (per-patient normalisation, context features, XGBoost, sensor-dropout
training), on the 30 pilot and 60 extra training patients (data/fullset.py), 5 folds grouped by
patient. Each fold's threshold comes from the other folds at <= 11 false alarms per hour.

A2: ways of choosing the threshold so that the budget holds on unseen patients, compared on the same
out-of-fold predictions; every choice is made on the other folds only:
  base      highest sensitivity at <= 11 FA/h (as in A1)
  margin    highest sensitivity at <= b FA/h, with b in MARGINS chosen so the other folds' own
            cross-fold check stays within 11 (nested), and reported for each fixed b too
  patient   per-patient threshold = max(base, q-quantile of the patient's own probabilities),
            label-free; reported for each q
  rule      x of the last 10 seconds, x in 5..8, threshold re-chosen on the other folds

Writes results/tables/phaseA_*.csv and cache/predictions/phaseA_oof.parquet.
"""
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.evaluation.metrics import FastEvents, window_scores
from seizure.features.build import ROOT, TABLES
from seizure.models.robustness import dropout_copy, sensor_columns
from seizure.models.sensor_aware import THRESHOLDS
from seizure.models.train import (ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, add_context, context_columns,
                                  feature_sets, folds, normalise)
from seizure.features.report import META

DIRS = [ROOT / "cache" / "features", ROOT / "cache" / "features_extra"]
OOF = ROOT / "cache" / "predictions" / "phaseA_oof.parquet"
MODELS = ("classic", "classic+TDA")
MARGINS = (6, 7, 8, 9, 10, 11)
QUANTILES = (None, 0.95, 0.97, 0.98, 0.99, 0.995)
RULES = (5, 6, 7, 8)


def load_all():
    raw = pd.concat([pd.read_parquet(p) for d in DIRS for p in sorted(d.glob("sub-*.parquet"))], ignore_index=True)
    return (raw.sort_values(["subject", "run", "t"]).drop_duplicates(["subject", "run", "t"])
            .reset_index(drop=True))


def prepare():
    df = load_all()
    df["fold"] = folds(df)
    cols = [c for c in df.columns if c not in META + ["fold"]]
    normalise(df, cols)
    df, ctx = add_context(df, context_columns(cols))
    return df, cols + ctx


def cv_dropout(df, fc):
    prob, fold = np.empty(len(df)), df.fold.to_numpy()
    by_sensor, rng = sensor_columns(fc), np.random.default_rng(SEED)
    for k in range(N_FOLDS):
        tr = fold != k
        X, y = df.loc[tr, fc], df.label.to_numpy()[tr]
        X, y = pd.concat([X, dropout_copy(X, by_sensor, rng)]), np.concatenate([y, y])
        m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        m.fit(X, y)
        prob[fold == k] = m.predict_proba(df.loc[fold == k, fc])[:, 1]
    return prob


def best_at(fe, prob, budget, x=ALARM[0]):
    """Highest sensitivity at <= budget FA/h over THRESHOLDS (full search; ties to the higher threshold)."""
    best, best_sens = THRESHOLDS[-1], -1.0
    for thr in THRESHOLDS:
        d, n, fa, h = fe.score(prob >= thr, x, ALARM[1])
        if fa / h <= budget and d / n >= best_sens:
            best, best_sens = thr, d / n
    return best


def a1():
    t0 = time.time()
    df, cols = prepare()
    print(f"{df.subject.nunique()} patients, {len(df)} seconds, {int(df.label.sum())} seizure seconds, "
          f"{time.time() - t0:.0f} s", flush=True)
    fsets = feature_sets(cols)
    preds = df[["subject", "run", "t", "label", "fold"]].copy()
    for nm in MODELS:
        t1 = time.time()
        preds[nm] = cv_dropout(df, fsets[nm])
        print(f"{nm}: {time.time() - t1:.0f} s", flush=True)
    OOF.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OOF)
    print(f"done in {time.time() - t0:.0f} s; run a2 next")


def score_fold(fe, prob_or_hit, thr=None, x=ALARM[0]):
    hit = prob_or_hit if thr is None else prob_or_hit >= thr
    d, n, fa, h = fe.score(hit, x, ALARM[1])
    return d, n, fa, h


def a2():
    p = pd.read_parquet(OOF)
    fold = p.fold.to_numpy()
    held = {k: FastEvents(p[fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(p[fold != k]) for k in range(N_FOLDS)}
    rows = []
    for nm in MODELS:
        prob = p[nm].to_numpy()
        w = window_scores(p.label, prob)

        def record(strategy, setting, per_fold):
            det = sum(r[0] for r in per_fold); n = sum(r[1] for r in per_fold)
            fa = sum(r[2] for r in per_fold); h = sum(r[3] for r in per_fold)
            fa_fold = [r[2] / r[3] for r in per_fold]
            rows.append(dict(model=nm, strategy=strategy, setting=setting, sensitivity=det / n, fa_per_h=fa / h,
                             fa_fold_max=max(fa_fold), folds_over_budget=sum(f > FA_BUDGET for f in fa_fold),
                             fa_folds=" ".join(f"{f:.1f}" for f in fa_fold), auroc=w["auroc"], aupr=w["aupr"]))

        # base and fixed margins
        for b in MARGINS:
            record("base" if b == FA_BUDGET else "margin", f"budget {b}/h",
                   [score_fold(held[k], prob[fold == k], best_at(other[k], prob[fold != k], b)) for k in range(N_FOLDS)])
        # nested margin: on the other folds, the largest b whose own leave-one-fold-out check stays <= 11
        per_fold = []
        for k in range(N_FOLDS):
            inner = [j for j in range(N_FOLDS) if j != k]
            chosen = MARGINS[0]
            for b in MARGINS:
                ok = True
                for j in inner:
                    rest = np.isin(fold, [i for i in inner if i != j])
                    thr = best_at(FastEvents(p[rest]), prob[rest], b)
                    _, _, fa, h = score_fold(held[j], prob[fold == j], thr)
                    ok &= fa / h <= FA_BUDGET
                if ok:
                    chosen = b
            per_fold.append(score_fold(held[k], prob[fold == k], best_at(other[k], prob[fold != k], chosen)))
        record("margin, nested", "budget chosen on other folds", per_fold)
        # per-patient quantile floor
        quant = {q: p.groupby("subject")[nm].transform(lambda s: s.quantile(q)).to_numpy() for q in QUANTILES if q}
        for q in QUANTILES[1:]:
            per_fold = []
            for k in range(N_FOLDS):
                o = fold != k
                base_thr = best_at(other[k], prob[o], FA_BUDGET)
                # threshold per second: the larger of the base threshold and the patient's own q-quantile
                hit_k = prob[fold == k] >= np.maximum(base_thr, quant[q][fold == k])
                per_fold.append(score_fold(held[k], hit_k))
            record("patient", f"floor at own {q:.3f} quantile", per_fold)
        # stricter alarm rule
        for x in RULES:
            record("rule", f"{x} of 10",
                   [score_fold(held[k], prob[fold == k], best_at(other[k], prob[fold != k], FA_BUDGET, x), x)
                    for k in range(N_FOLDS)])
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / "phaseA_thresholds.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.drop(columns=["auroc", "aupr"]).to_string(index=False))
    print(res.groupby("model")[["auroc", "aupr"]].first().to_string())


if __name__ == "__main__":
    {"a1": a1, "a2": a2}[sys.argv[1]]()
