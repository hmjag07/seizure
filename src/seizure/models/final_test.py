"""Final test on the held-out patients (docs/research-log.md, Step 8).

    python -m seizure.models.final_test [quick]

Everything is fixed on the 30 pilot patients before the held-out patients (data/heldout.py) are scored,
once:
1. Final models, trained on all pilot patients: classic and classic + TDA with sensor-dropout training
   (Step 5.3), and classic + TDA with normal training; and the window-label health check (Step 3.7).
2. Thresholds, from the pilot patients' out-of-fold predictions (the saved fold models): for every set
   of dead sensors, the threshold with the highest sensitivity at <= 11 false alarms per hour; the
   health check's switch-off cut-off is the median of its fold cut-offs.
3. Held-out features are normalised per patient (median and MAD over the patient's own seconds,
   without labels), exactly as on the pilot.

Scored on the held-out patients: clean data, dead sensors (the 102 combinations of robustness.py) and
injected faults (the 192 scenarios of faults/evaluate.py, without / with / with a perfect health check).
Writes results/tables/final_*.csv.
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import FastEvents, event_scores, window_scores
from seizure.faults import evaluate as ev
from seizure.faults import train_health as th
from seizure.faults.inject import FAULTS, SEVERITIES
from seizure.features.build import ROOT, TABLES
from seizure.features.report import META
from seizure.models.robustness import EEG, MODELS_DIR, combinations, dropout_copy, sensor_columns
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, PARAMS, SEED, feature_sets

HELD_FEATURES = ROOT / "cache" / "features_heldout"
HELD_FAULTS = ROOT / "cache" / "faults_heldout"


def load_dir(path, columns):
    raw = pd.concat([pd.read_parquet(p) for p in sorted(path.glob("sub-*.parquet"))], ignore_index=True)
    return raw.reindex(columns=columns)  # a sensor absent for every held-out patient becomes an empty column


def train_final(pilot, fc, dropout):
    X, y = pilot.df[fc], pilot.df.label.to_numpy()
    if dropout:
        X = pd.concat([X, dropout_copy(X, sensor_columns(fc), np.random.default_rng(SEED))])
        y = np.concatenate([y, y])
    m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    m.fit(X, y)
    return m


class FinalThresholds:
    """One threshold per set of dead sensors, chosen on all pilot out-of-fold predictions (fold models).
    .fixed and .for_set() return one-element lists, so faults/evaluate.predict can index them with 0."""

    def __init__(self, pilot, pilot_events, fold_models, fc):
        self.pilot, self.events, self.models, self.fc, self.cache = pilot, pilot_events, fold_models, fc, {}
        self.by_sensor = sensor_columns(fc)
        self.fixed = self.for_set(frozenset())

    def for_set(self, dead):
        if dead not in self.cache:
            p, prob = self.pilot, np.empty(len(self.pilot.df))
            for k, m in enumerate(self.models):
                X = p.df.loc[p.fold == k, self.fc].copy()
                for s in dead:
                    X[self.by_sensor[s]] = np.nan
                prob[p.fold == k] = m.predict_proba(X)[:, 1]
            self.cache[dead] = [best_threshold(self.events, prob)]
        return self.cache[dead]


def train_final_health():
    subs = sorted(p.name.removesuffix("_faulty.parquet") for p in th.OUT.glob("*_faulty.parquet"))
    df = pd.concat([th.patient_rows(s) for s in subs], ignore_index=True)
    m = XGBClassifier(**th.PARAMS)
    m.fit(th.features(df), df.y)
    m.save_model(MODELS_DIR / "final_health.json")
    cut = float(pd.read_csv(TABLES / "health_cutoffs.csv").cutoff.median())
    return m, cut


def main():
    t_start = time.time()
    quick = "quick" in sys.argv
    pilot = ev.Data()
    raw_cols = META + pilot.base
    held = ev.Data(raw=load_dir(HELD_FEATURES, raw_cols), single=True)
    pilot_events = FastEvents(pilot.df)
    fsets = feature_sets(pilot.cols)
    print(f"pilot {len(pilot.df)} rows, held-out {len(held.df)} rows from {held.df.subject.nunique()} patients, "
          f"{held.df.label.sum()} seizure seconds, {time.time() - t_start:.0f} s", flush=True)

    models, thresholds = {}, {}
    for tr, nm in ev.MODELS:
        t0 = time.time()
        models[(tr, nm)] = [train_final(pilot, fsets[nm], tr == "dropout")]
        models[(tr, nm)][0].save_model(MODELS_DIR / f"final_{tr}_{nm}.json")
        thresholds[(tr, nm)] = FinalThresholds(pilot, pilot_events, ev.load_models(tr, nm), fsets[nm])
        print(f"final {tr} {nm}: trained, clean threshold {thresholds[(tr, nm)].fixed[0]}, "
              f"{time.time() - t0:.0f} s", flush=True)

    # 1. Clean data.
    clean = []
    preds = held.df[["subject", "run", "t", "label"]].copy()
    for (tr, nm), (m,) in models.items():
        col = f"{tr}|{nm}"
        preds[col] = m.predict_proba(held.df[fsets[nm]])[:, 1]
        thr = thresholds[(tr, nm)].fixed[0]
        clean.append(dict(training=tr, model=nm, threshold=thr, **window_scores(preds.label, preds[col]),
                          **event_scores(preds, col, thr, *ALARM)))
    clean = pd.DataFrame(clean).round(4)
    clean.to_csv(TABLES / "final_clean.csv", index=False)
    preds.to_parquet(ROOT / "cache" / "predictions" / "final_clean_heldout.parquet")
    print(clean.to_string(index=False), flush=True)

    # 2. Dead sensors.
    dead_rows = []
    combos = combinations()[:3] if quick else combinations()
    for (tr, nm), (m,) in models.items():
        t0, ths, fc = time.time(), thresholds[(tr, nm)], fsets[nm]
        for combo in combos:
            X = held.df[fc].copy()
            for s in combo:
                X[ths.by_sensor[s]] = np.nan
            prob = m.predict_proba(X)[:, 1]
            for rule, thr in (("fixed", ths.fixed[0]), ("sensor-aware", ths.for_set(frozenset(combo))[0])):
                sens, fa = ev.score(held, prob, np.full(len(prob), thr))
                dead_rows.append(dict(training=tr, model=nm, rule=rule, dead=len(combo),
                                      sensors_dead=" ".join(combo) or "none", eeg_alive=not EEG <= set(combo),
                                      threshold=thr, sensitivity=sens, fa_per_h=fa))
        print(f"dead sensors {tr} {nm}: {len(combos)} combinations, {time.time() - t0:.0f} s", flush=True)
    dead = pd.DataFrame(dead_rows).round(4)
    dead.to_csv(TABLES / "final_dead_sensors.csv", index=False)

    # 3. Injected faults.
    health_model, cut = train_final_health()
    ev.HEALTH_CUTS.update({k: cut for k in range(5)})
    print(f"final health check trained, cut-off {cut}, {time.time() - t_start:.0f} s", flush=True)
    clean_q = pd.concat([pd.read_parquet(HELD_FAULTS / f"{s}_quality.parquet") for s in held.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(held.df[["subject", "run", "t"]])).reset_index(drop=True))
    conditions = [("pop", 0.5)] if quick else list(itertools.product(FAULTS, SEVERITIES))
    sets = ev.sensor_sets()[:2] if quick else ev.sensor_sets()
    health_log = []
    rows = ev.run_scenarios(held, HELD_FAULTS, conditions, sets, models, thresholds, fsets, [health_model],
                            clean_q, TABLES / ("final_faults_quick.csv" if quick else "final_faults.csv"),
                            health_log=health_log)
    faults = pd.DataFrame(rows).round(4)
    faults.to_csv(TABLES / ("final_faults_quick.csv" if quick else "final_faults.csv"), index=False)
    hl = pd.concat(health_log, ignore_index=True)
    hs = (hl.groupby(["fault", "wclass"]).off.mean().unstack().rename(columns={0: "healthy", 1: "degraded",
                                                                                2: "failed"}).round(3))
    hs.to_csv(TABLES / "final_health_switch_off.csv")
    print("health check on held-out single-sensor faults, share switched off by true window class:")
    print(hs.to_string())
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
