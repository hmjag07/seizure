"""Phase B1: does TDA-B (the topology of the sensor network) add to the model, and does it degrade gracefully?

    python -m seizure.models.phase_b1 acts     # sensor activities for all 90 training patients (cached)
    python -m seizure.models.phase_b1 cv       # clean cross-validation and dead-sensor robustness

Models (sensor-dropout training, the same 5 patient-grouped folds over 90 patients, thresholds chosen on
the other folds at <= 8 false alarms per hour, judged at 11):
  classic+TDA        the current model (TDA-A)
  classic+TDA+TDAB   plus the 9 TDA-B network features (features/tda_network.py)

TDA-B depends on which sensors are alive together, so it is recomputed without the dead sensors: for each
row of the dropout copy (its own random dead set) and for each dead-sensor combination of the robustness
test (robustness.combinations, 102 sets). Writes results/tables/phaseB1_*.csv.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.data.signals import load_patient
from seizure.evaluation.metrics import FastEvents, window_scores
from seizure.faults import evaluate as ev
from seizure.features.build import ROOT, TABLES
from seizure.features.tda_network import NAMES, activity, network_features
from seizure.models.phase_a import load_all
from seizure.models.robustness import EEG, combinations, sensor_columns
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, SELECT_BUDGET, feature_sets

SIGNAL_DIRS = [ROOT / "cache" / "signals", ROOT / "cache" / "signals_extra"]
ACTS = ROOT / "cache" / "tdab_activities.pkl"
TDAB = [f"TDAB_{n}" for n in NAMES]


def build_activities():
    """{subject: [(run, t0, t1, fetch_t0, {sensor: activity})]} for every chunk of the 90 patients."""
    out = {}
    for d in SIGNAL_DIRS:
        for path in sorted(d.glob("sub-*.npz")):
            meta, data = load_patient(path)
            out[path.stem] = [(c["run"], c["t0"], c["t1"], c["fetch_t0"],
                               # No statistics if the sensor never worked in the background chunks: as in
                               # the feature build, no rescaling (correlations do not depend on scale).
                               {s: activity(data[(c["i"], s)], meta["fs"][s], *meta["stats"].get(s, (0.0, 1.0)))
                                for s in c["signals"]})
                              for c in meta["chunks"]]
    with open(ACTS, "wb") as f:
        pickle.dump(out, f)
    print(f"activities for {len(out)} patients saved to {ACTS}")


class Network:
    """TDA-B for the rows of a Data table, for any per-row pattern of dead sensors."""

    def __init__(self, data):
        with open(ACTS, "rb") as f:
            acts = pickle.load(f)
        df = data.df
        self.groups = []  # (row positions, ends, fetch_t0, activities)
        for sub, chunks in acts.items():
            rows = df.index[df.subject == sub].to_numpy()
            if len(rows) == 0:
                continue
            sub_df = df.loc[rows]
            for run, t0, t1, fetch_t0, a in chunks:
                sel = rows[((sub_df.run == run) & (sub_df.t >= t0) & (sub_df.t <= t1)).to_numpy()]
                if len(sel):
                    self.groups.append((sel, df.t.to_numpy()[sel], fetch_t0, a))
        self.n = len(df)

    def compute(self, dead=None):
        """(n_rows, 9) TDA-B; dead: optional (n_rows, 8) bool, True = sensor dead at that row."""
        out = np.full((self.n, len(TDAB)), np.nan, dtype="float32")
        for sel, ends, fetch_t0, a in self.groups:
            alive = None if dead is None else {s: ~dead[sel, SIGNALS.index(s)] for s in a}
            f = network_features(a, ends, fetch_t0, alive)
            out[sel] = np.column_stack([f[c] for c in TDAB])
        return out


def random_dead(n, rng):
    """1-6 random dead sensors per row (the sensor-dropout rule)."""
    k = rng.integers(1, 7, n)
    return rng.random((n, len(SIGNALS))).argsort(1).argsort(1) < k[:, None]


def blank(X, cols_by_sensor, dead):
    X = X.copy()
    for j, s in enumerate(SIGNALS):
        if cols_by_sensor.get(s) and dead[:, j].any():
            X.loc[dead[:, j], cols_by_sensor[s]] = np.nan
    return X


def cv():
    t_start = time.time()
    data = ev.Data(raw=load_all())
    net = Network(data)
    clean_b = net.compute()
    df = data.df
    df[TDAB] = clean_b
    print(f"{df.subject.nunique()} patients; clean TDA-B in {time.time() - t_start:.0f} s, "
          f"missing {np.isnan(clean_b[:, 0]).mean():.3f}", flush=True)
    base = feature_sets(data.cols)["classic+TDA"]
    sets = {"classic+TDA": base, "classic+TDA+TDAB": base + TDAB}
    fold = data.fold
    events = {k: FastEvents(df[fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(df[fold != k]) for k in range(N_FOLDS)}

    # Training: clean rows + a dropout copy whose TDA-B is recomputed without each row's dead sensors.
    rng = np.random.default_rng(SEED)
    dead_copy = random_dead(len(df), rng)
    copy_b = net.compute(dead_copy)
    print(f"dropout-copy TDA-B in {time.time() - t_start:.0f} s", flush=True)
    models = {}
    for name, fc in sets.items():
        by_sensor = sensor_columns([c for c in fc if c not in TDAB])
        copy = blank(df[fc], by_sensor, dead_copy)
        if "TDAB" in name:
            copy[TDAB] = copy_b
        models[name] = []
        for k in range(N_FOLDS):
            tr = fold != k
            X = pd.concat([df.loc[tr, fc], copy.loc[tr]])
            y = np.concatenate([df.label.to_numpy()[tr]] * 2)
            m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            m.fit(X, y)
            models[name].append(m)
        del copy
        print(f"{name}: trained, {time.time() - t_start:.0f} s", flush=True)

    rows = []
    combos = combinations()
    for ci, combo in enumerate(combos):
        dead = np.zeros((len(df), len(SIGNALS)), bool)
        for s in combo:
            dead[:, SIGNALS.index(s)] = True
        combo_b = clean_b if not combo else net.compute(dead)
        for name, fc in sets.items():
            by_sensor = sensor_columns([c for c in fc if c not in TDAB])
            X = blank(df[fc], by_sensor, dead) if combo else df[fc]
            if "TDAB" in name:
                X = X.copy()
                X[TDAB] = combo_b
            prob = np.empty(len(df))
            for k, m in enumerate(models[name]):
                prob[fold == k] = m.predict_proba(X[fold == k])[:, 1]
            det = n = fa = 0
            hours, fa_fold = 0.0, []
            for k in range(N_FOLDS):
                thr = best_threshold(other[k], prob[fold != k], budget=SELECT_BUDGET)
                d, nk, f, h = events[k].score(prob[fold == k] >= thr, *ALARM)
                det, n, fa, hours = det + d, n + nk, fa + f, hours + h
                fa_fold.append(f / h)
            rows.append(dict(model=name, dead=len(combo), sensors_dead=" ".join(combo) or "none",
                             eeg_alive=not EEG <= set(combo), sensitivity=det / n, fa_per_h=fa / hours,
                             folds_over_budget=sum(x > FA_BUDGET for x in fa_fold),
                             **(window_scores(df.label, prob) if not combo else {})))
        if ci % 10 == 0:
            print(f"combination {ci + 1}/{len(combos)}, {time.time() - t_start:.0f} s", flush=True)
            pd.DataFrame(rows).round(4).to_csv(TABLES / "phaseB1_dead_sensors.csv", index=False)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / "phaseB1_dead_sensors.csv", index=False)
    a = res[res.eeg_alive]
    print(res[res.dead == 0].to_string(index=False))
    print(a.groupby(["model", "dead"])[["sensitivity", "fa_per_h"]].mean().round(3).unstack("model").to_string())
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    {"acts": build_activities, "cv": cv}[sys.argv[1]]()
