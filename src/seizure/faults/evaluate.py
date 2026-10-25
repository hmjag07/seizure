"""Seizure detection under injected faults (docs/pipeline.md, Step 8).

    python -m seizure.faults.evaluate

For every fault type, severity and set of faulty sensors, the faulty sensors' features are swapped in
from the fault onset on (faults/build_faulty.py), normalised with the patient's clean statistics, and
given context features as in train.py. The saved fold models (robustness.py) are then scored three ways:

A  no health check   faulty features go in as they are; the fold's fixed clean threshold
B  health check      sensors the health check calls failed are blanked for that second, and the
                     threshold is the sensor-aware one for the sensors it believes are working
C  perfect check     as B, but using the true window-level labels (faults/window_labels.py)

All scoring: 5-of-10 alarm, any-overlap sensitivity, false alarms per hour, patient-grouped folds.
Writes results/tables/faults_eval.csv.
"""
import itertools
import random
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import FastEvents
from seizure.faults.build_faulty import OUT
from seizure.faults.health import QUALITY, add_context as add_health_context, decide
from seizure.faults.inject import FAULTS, SEVERITIES
from seizure.faults.train_health import features as health_features
from seizure.features.build import CACHE, TABLES
from seizure.features.report import META, load
from seizure.models.robustness import EEG, MODELS_DIR, sensor_columns
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, N_FOLDS, context_columns, feature_sets, folds

MODELS = [("dropout", "classic+TDA"), ("dropout", "classic"), ("clean", "classic+TDA")]
HEALTH_CUTS = {}  # per-fold switch-off cut-offs, read from results/tables/health_cutoffs.csv in main()
FAILED = 0.6
N_FIVE = 6  # sampled 5-of-8 sensor sets with at least one EEG alive


def sensor_sets(seed=42):
    rng = random.Random(seed)
    five = [c for c in itertools.combinations(SIGNALS, 5) if not EEG <= set(c)]
    motion = tuple(s for s in SIGNALS if s.startswith(("ACC", "GYR")))
    return [(s,) for s in SIGNALS] + [("EEG1", "EEG2"), motion] + rng.sample(five, N_FIVE)


def specific(sensor, generic_col):
    if generic_col.startswith("TDA_"):
        return f"TDA_{sensor}_{generic_col[4:]}"
    return f"{sensor}_{generic_col}"


class Data:
    """Clean features (normalised + context), patient statistics, row positions and fold models."""

    def __init__(self, raw=None, single=False, group=None):
        """raw: feature table (default: the pilot patients). single: one group of patients scored by one
        final model (the held-out test) instead of the 5 patient-grouped folds. group: optional function
        subject -> person, so recordings of one person share a fold (CHB-MIT: chb21 is chb01)."""
        raw = load() if raw is None else raw
        # Two adjacent background chunks share their boundary second (once: sub-055); keep one.
        raw = (raw.sort_values(["subject", "run", "t"]).drop_duplicates(["subject", "run", "t"])
               .reset_index(drop=True))
        self.base = [c for c in raw.columns if c not in META]
        self.med = raw.groupby("subject")[self.base].median()
        self.mad = ((raw[self.base] - raw.groupby("subject")[self.base].transform("median")).abs()
                    .groupby(raw.subject).median() * 1.4826).replace(0, np.nan)
        z = ((raw[self.base] - self.med.reindex(raw.subject).to_numpy()) / self.mad.reindex(raw.subject).to_numpy())
        self.df = pd.concat([raw[META], z.astype("float32")], axis=1)
        self.df["fold"] = 0 if single else folds(
            self.df if group is None else self.df.assign(subject=self.df.subject.map(group)))
        self.seg = ((self.df.t.diff() != 1) | (self.df.subject != self.df.subject.shift())
                    | (self.df.run != self.df.run.shift())).cumsum().to_numpy()
        self.pos = pd.Series(np.arange(len(self.df)),
                             index=pd.MultiIndex.from_frame(self.df[["subject", "run", "t"]]))
        self.ctx_cols = context_columns(self.base)
        self.df = pd.concat([self.df, self.context(self.df, self.ctx_cols)], axis=1)
        self.cols = self.base + [f"{c}_{s}" for s in ("m10", "d60") for c in self.ctx_cols]
        self.fold = self.df.fold.to_numpy()
        groups = sorted(set(self.fold))
        self.events = {k: FastEvents(self.df[self.fold == k]) for k in groups}
        self.other = {} if single else {k: FastEvents(self.df[self.fold != k]) for k in groups}

    def context(self, df, cols):
        g = df[cols].groupby(self.seg)
        m10 = g.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
        m60 = g.rolling(60, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
        d60 = m10 - m60.groupby(self.seg).shift(10)
        return pd.concat([m10.add_suffix("_m10"), d60.add_suffix("_d60")], axis=1).astype("float32")


def load_models(training, name):
    out = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"{training}_{name}_fold{k}.json")
        out.append(m)
    return out


def load_health():
    out = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"health_fold{k}.json")
        out.append(m)
    return out


class Thresholds:
    """Per-fold thresholds: the fixed clean one, and sensor-aware ones per set of dead sensors (cached)."""

    def __init__(self, data, models, fc, training, name, budget=None):
        self.data, self.models, self.fc, self.cache = data, models, fc, {}
        self.budget = budget
        self.by_sensor = sensor_columns(fc)
        self.fixed = self.for_set(frozenset())

    def for_set(self, dead):
        if dead not in self.cache:
            d = self.data
            prob = np.empty(len(d.df))
            for k, m in enumerate(self.models):
                X = d.df.loc[d.fold == k, self.fc].copy()
                for s in dead:
                    X[self.by_sensor[s]] = np.nan
                prob[d.fold == k] = m.predict_proba(X)[:, 1]
            kw = {} if self.budget is None else {"budget": self.budget}
            self.cache[dead] = [best_threshold(d.other[k], prob[d.fold != k], **kw) for k in range(N_FOLDS)]
        return self.cache[dead]


def score(data, prob, thr_per_row):
    det = n = fa = 0
    hours = 0.0
    for k in data.events:
        sel = data.fold == k
        d, nk, f, h = data.events[k].score(prob[sel] >= thr_per_row[sel], *ALARM)
        det, n, fa, hours = det + d, n + nk, fa + f, hours + h
    return det / n, fa / hours


def main():
    t_start = time.time()
    data = Data()
    quick = "quick" in sys.argv  # one fault condition, two sensor sets: a timing and sanity run
    sets = sensor_sets()[:2] if quick else sensor_sets()
    feature_cols = feature_sets(data.cols)
    print(f"data ready: {len(data.df)} rows, {len(sets)} sensor sets, {time.time() - t_start:.0f} s", flush=True)

    clean_q = pd.concat([pd.read_parquet(OUT / f"{s}_quality.parquet") for s in data.df.subject.unique()])
    clean_q = clean_q.drop_duplicates(["subject", "run", "t"])
    clean_q = clean_q.set_index(["subject", "run", "t"]).reindex(
        pd.MultiIndex.from_frame(data.df[["subject", "run", "t"]])).reset_index(drop=True)
    health = load_health()
    HEALTH_CUTS.update(pd.read_csv(TABLES / "health_cutoffs.csv").set_index("fold").cutoff.to_dict())
    models = {(tr, nm): load_models(tr, nm) for tr, nm in MODELS}
    thresholds = {(tr, nm): Thresholds(data, models[(tr, nm)], feature_cols[nm], tr, nm) for tr, nm in MODELS}
    print(f"models and clean thresholds ready, {time.time() - t_start:.0f} s", flush=True)

    conditions = [("pop", 0.5)] if quick else list(itertools.product(FAULTS, SEVERITIES))
    out_file = TABLES / ("faults_eval_quick.csv" if quick else "faults_eval.csv")
    part = next((a for a in sys.argv[1:] if "/" in a), None)  # e.g. "1/2": every 2nd condition from the 1st
    if part:
        i, n = map(int, part.split("/"))
        conditions = conditions[i - 1::n]
        out_file = TABLES / f"faults_eval_part{i}of{n}.csv"
    rows = run_scenarios(data, OUT, conditions, sets, models, thresholds, feature_cols, health, clean_q, out_file)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(out_file, index=False)
    print(res.groupby(["training", "model", "mode", "fault", "severity"])[["sensitivity", "fa_per_h"]]
          .mean().round(3).unstack("mode").to_string())
    print(f"done in {time.time() - t_start:.0f} s")


def fault_dir(faults_dir, subject):
    """faults_dir is one folder, or a function subject -> folder (patients spread over several folders)."""
    return faults_dir(subject) if callable(faults_dir) else faults_dir


def load_condition(data, faults_dir, fault, sev):
    """Faulty rows of every patient for one (fault, severity), with window labels and row positions."""
    subs = data.df.subject.unique()
    f = pd.concat([pd.read_parquet(fault_dir(faults_dir, s) / f"{s}_faulty.parquet",
                                   filters=[("fault", "==", fault), ("severity", "==", sev)])
                   for s in subs], ignore_index=True)
    f = f.drop_duplicates(["subject", "run", "t", "sensor"])
    wl = pd.concat([pd.read_parquet(fault_dir(faults_dir, s) / f"{s}_wlabels.parquet",
                                    filters=[("fault", "==", fault), ("severity", "==", sev)]).assign(subject=s)
                    for s in subs], ignore_index=True)
    wl = wl.drop_duplicates(["subject", "run", "t", "sensor"])
    f = f.merge(wl[["subject", "run", "t", "sensor", "wclass"]], on=["subject", "run", "t", "sensor"], how="left")
    f["pos"] = data.pos.reindex(pd.MultiIndex.from_frame(f[["subject", "run", "t"]])).to_numpy()
    f = f.dropna(subset=["pos"])
    f["pos"] = f.pos.astype(int)
    return f


def assemble(data, f, sset, signals=SIGNALS):
    """Feature table with the sensors in sset faulty from their onset (normalised with the patient's clean
    statistics, context recomputed), and per sensor the rows whose window is truly failed."""
    X = data.df[data.cols].copy()
    truly_failed = {s: np.zeros(len(X), bool) for s in signals}
    for s in sset:
        fs_ = f[f.sensor == s]
        if fs_.empty:
            continue
        cols = [c for c in fs_.columns if specific(s, c) in data.base]
        spec = [specific(s, c) for c in cols]
        subj = fs_.subject.to_numpy()
        vals = (fs_[cols].to_numpy(dtype="float64") - data.med.loc[subj, spec].to_numpy()) /             data.mad.loc[subj, spec].to_numpy()
        X.loc[fs_.pos.to_numpy(), spec] = vals.astype("float32")
        truly_failed[s][fs_.pos.to_numpy()] = fs_.wclass.to_numpy() == 2  # window-level truth
    touched = [c for c in data.ctx_cols if any(c.removeprefix("TDA_").startswith(s + "_") for s in sset)]
    if touched:
        X[[f"{c}_{w}" for w in ("m10", "d60") for c in touched]] = data.context(X, touched).to_numpy()
    return X, truly_failed


def run_scenarios(data, faults_dir, conditions, sets, models, thresholds, feature_cols, health, clean_q, out_file,
                  health_log=None, modes=("A", "B", "C")):
    """Score every (fault, severity) condition x sensor set x model x way (A, B, C); see the module docstring.
    health_log, if a list, collects the health check's switch-off decisions on single-sensor faults."""
    rows, clean_flags = [], {}
    for fault, sev in conditions:
        t0 = time.time()
        f = load_condition(data, faults_dir, fault, sev)
        for sset in sets:
            X, truly_failed = assemble(data, f, sset)
            flagged = health_flags(data, clean_q, f, sset, health, clean_flags)
            if health_log is not None and len(sset) == 1 and not f[f.sensor == sset[0]].empty:
                fs_ = f[f.sensor == sset[0]]
                health_log.append(pd.DataFrame({"fault": fault, "severity": sev, "sensor": sset[0],
                                                "wclass": fs_.wclass.to_numpy(),
                                                "off": flagged[sset[0]][fs_.pos.to_numpy()]}))
            for (tr, nm), ms in models.items():
                th = thresholds[(tr, nm)]
                fc = feature_cols[nm]
                done = {}
                for mode in modes:
                    dead_rows = {"A": None, "B": flagged, "C": truly_failed}[mode]
                    if mode != "A" and not any(v.any() for v in dead_rows.values()):
                        sens, fa = done["A"]  # nothing switched off: identical to no health check
                    else:
                        prob, thr = predict(data, ms, X, fc, th, dead_rows)
                        sens, fa = score(data, prob, thr)
                    done[mode] = (sens, fa)
                    rows.append(dict(fault=fault, severity=sev, sensors=" ".join(sset), n_sensors=len(sset),
                                     eeg_alive=not EEG <= set(sset), training=tr, model=nm,
                                     mode=mode, sensitivity=sens, fa_per_h=fa))
        print(f"{fault} {sev}: {len(sets)} sets in {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).round(4).to_csv(out_file, index=False)

    return rows


def health_flags(data, clean_q, f, sset, health, cache, signals=SIGNALS):
    """Per sensor, rows the health check calls failed. Clean quality everywhere, faulty quality after
    the onset for sensors in sset; results for sensors without a fault are cached across scenarios."""
    out = {}
    for s in signals:
        if s in sset:
            out[s] = sensor_flags(data, clean_q, f[f.sensor == s], s, health, signals)
        else:
            if s not in cache:
                cache[s] = sensor_flags(data, clean_q, None, s, health, signals)
            out[s] = cache[s]
    return out


def sensor_flags(data, clean_q, fs_, s, health, signals=SIGNALS):
    """Rows where the health check switches sensor s off: quality features (faulty after the onset if
    fs_ is given), health context, failed-probability per fold model, then smoothing + hysteresis with
    the fold's cut-off (faults/train_health.py)."""
    cols = [f"{s}_{c}" for c in QUALITY]
    if cols[0] not in clean_q:
        return np.zeros(len(data.df), bool)
    part = pd.DataFrame(clean_q[cols].to_numpy(), columns=list(QUALITY))
    if fs_ is not None and not fs_.empty:
        part.loc[fs_.pos.to_numpy(), list(QUALITY)] = fs_[list(QUALITY)].to_numpy()
    part["seg"] = data.seg
    part = add_health_context(part, "seg")
    part["sensor"] = s
    Xh = health_features(part, signals)
    p_failed = np.zeros(len(part))
    valid = part[list(QUALITY)].notna().all(axis=1).to_numpy()
    for k in range(N_FOLDS):
        sel = (data.fold == k) & valid
        if sel.any():
            p_failed[sel] = health[k].predict_proba(Xh[sel])[:, 2]
    off = np.zeros(len(part), bool)
    for k in range(N_FOLDS):
        sel = data.fold == k
        if sel.any():
            off[sel] = decide(p_failed[sel], data.seg[sel], HEALTH_CUTS[k])
    return off & valid


def predict(data, models, X, fc, th, dead_rows, sensor_aware=True, signals=SIGNALS):
    """Probabilities with dead sensors blanked per row, and the threshold for each row (the sensor-aware
    one for the row's dead sensors, or the fixed clean one if sensor_aware is False)."""
    prob = np.empty(len(X))
    thr = np.empty(len(X))
    for k, m in enumerate(models):
        sel = data.fold == k
        Xk = X.loc[sel, fc].copy()
        if dead_rows is None:
            prob[sel] = m.predict_proba(Xk)[:, 1]
            thr[sel] = th.fixed[k]
            continue
        dead = np.stack([dead_rows[s][sel] for s in signals], axis=1)
        for j, s in enumerate(signals):
            if dead[:, j].any():
                Xk.loc[dead[:, j], th.by_sensor[s]] = np.nan
        prob[sel] = m.predict_proba(Xk)[:, 1]
        if not sensor_aware:
            thr[sel] = th.fixed[k]
            continue
        code = dead.astype(np.int64) @ (np.int64(1) << np.arange(len(signals), dtype=np.int64))  # dead set as bits
        thr_k = np.empty(len(code))
        for c in np.unique(code):
            dead_set = frozenset(s for j, s in enumerate(signals) if c >> j & 1)
            thr_k[code == c] = th.for_set(dead_set)[k]
        thr[sel] = thr_k
    return prob, thr


if __name__ == "__main__":
    main()
