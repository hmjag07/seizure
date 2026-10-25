"""Phase D: CHB-MIT, the second dataset (18 scalp EEG channels), through the dataset adapter.

    python -m seizure.models.phase_d acts     # TDA-B channel activities from the signal cache
    python -m seizure.models.phase_d train    # D1: dropout-trained fold models (classic, +TDA, +TDA+TDA-B)
    python -m seizure.models.phase_d dead     # D1 + D2: clean data, and 0-17 dead channels
    python -m seizure.models.phase_d health   # D3: health check trained on CHB-MIT's injected faults
    python -m seizure.models.phase_d faults   # D4: fault training (A3) and injected faults on 11 of 18 channels
    python -m seizure.models.phase_d stats    # patient bootstrap of the TDA effects

The SeizeIT2 system, unchanged in every rule, applied to CHB-MIT:
- features: classic EEG features and TDA-A per channel (features/build_any.py), per-patient robust
  z-score, 10 s / 60 s context; TDA-B: the network of the 18 channels (features/tda_network.py);
- XGBoost, 5 folds grouped by person (chb21 is chb01), thresholds chosen on the other folds at <= 8 false
  alarms per hour (train.SELECT_BUDGET), judged at 11; 5-of-10 alarm; any-overlap scoring;
- sensor-dropout training: each row's copy loses a random 1-13 of the 18 channels (6-72%; SeizeIT2: 1-6 of 8),
  with TDA-B recomputed without them.
What changes is the threshold rule. One threshold per set of dead sensors (SeizeIT2: 256 sets) does not scale
to 18 channels (262,144 sets), so the threshold depends on HOW MANY channels are dead: for k dead channels it
is chosen on the other folds with a random k channels blanked per second (count-aware threshold). It is
also what a deployed monitor could use.
Target, as for SeizeIT2: with 60% of channels dead (11 of 18) sensitivity drops by at most 15 points.
Writes cache/models/d_*.json and results/tables/phaseD_*.csv.
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data import datasets
from seizure.data.signals import load_patient
from seizure.evaluation.metrics import FastEvents, window_scores
from seizure.faults import evaluate as ev
from seizure.features.build import ROOT, TABLES
from seizure.features.report import META
from seizure.features.tda_network import NAMES as TDAB_NAMES, activity, network_features
from seizure.models.robustness import MODELS_DIR
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, SELECT_BUDGET, feature_sets

DS = datasets.get("chbmit")
SIG = DS.signals
N = len(SIG)
ACTS = ROOT / "cache" / "tdab_activities_chbmit.pkl"
TDAB = [f"TDAB_{n}" for n in TDAB_NAMES]
DROP_MIN, DROP_MAX = 1, 13
TARGET_DEAD = 11  # 60% of 18
COMBOS_PER_K = 8
DEAD_LEVELS = tuple(range(18))  # 0-17 dead channels
SETS = ("classic", "classic+TDA", "classic+TDA+TDAB")


def load_raw():
    raw = pd.concat([pd.read_parquet(p) for p in sorted(DS.folder("features").glob("chb*.parquet"))],
                    ignore_index=True)
    return raw.sort_values(["subject", "run", "t"]).drop_duplicates(["subject", "run", "t"]).reset_index(drop=True)


def load_data():
    return ev.Data(raw=load_raw(), group=DS.group)


def by_signal(cols):
    """{channel: its columns}; TDA-B columns belong to no channel."""
    out = {s: [] for s in SIG}
    for c in cols:
        base = c.removeprefix("TDA_")
        for s in SIG:
            if base.startswith(s + "_"):
                out[s].append(c)
                break
    return out


def model_sets(data):
    fs = feature_sets(data.cols)
    return {"classic": fs["classic"], "classic+TDA": fs["classic+TDA"], "classic+TDA+TDAB": fs["classic+TDA"] + TDAB}


def random_dead(n_rows, k_min, k_max, rng):
    """(n_rows, 18) bool: each row loses a random k_min-k_max channels."""
    k = rng.integers(k_min, k_max + 1, n_rows)
    return rng.random((n_rows, N)).argsort(1).argsort(1) < k[:, None]


def fixed_dead(n_rows, combo):
    dead = np.zeros((n_rows, N), bool)
    for s in combo:
        dead[:, SIG.index(s)] = True
    return dead


# ---------------------------------------------------------------- TDA-B

def build_activities():
    out = {}
    for path in sorted(DS.folder("signals").glob("chb*.npz")):
        meta, data = load_patient(path)
        out[path.stem] = [(c["run"], c["t0"], c["t1"], c["fetch_t0"],
                           {s: activity(data[(c["i"], s)], meta["fs"][s], *meta["stats"].get(s, (0.0, 1.0)))
                            for s in c["signals"]})
                          for c in meta["chunks"]]
    with open(ACTS, "wb") as f:
        pickle.dump(out, f)
    print(f"activities for {len(out)} recordings saved to {ACTS}")


class Network:
    """TDA-B for the rows of a Data table, for any per-row pattern of dead channels (as phase_b1.Network)."""

    def __init__(self, data, acts=None):
        if acts is None:
            with open(ACTS, "rb") as f:
                acts = pickle.load(f)
        df = data.df
        self.groups = []  # (rows, ends, fetch_t0, (recording, chunk index), activities)
        for sub, chunks in acts.items():
            rows = df.index[df.subject == sub].to_numpy()
            if len(rows) == 0:
                continue
            sub_df = df.loc[rows]
            for ci, (run, t0, t1, fetch_t0, a) in enumerate(chunks):
                sel = rows[((sub_df.run == run) & (sub_df.t >= t0) & (sub_df.t <= t1)).to_numpy()]
                if len(sel):
                    self.groups.append((sel, df.t.to_numpy()[sel], fetch_t0, (sub, ci), a))
        self.n = len(df)

    def compute(self, dead=None, override=None):
        """dead: optional (rows, 18) bool. override: optional {(recording, chunk): {channel: activity}} that
        replaces some channels' activities (injected faults)."""
        out = np.full((self.n, len(TDAB)), np.nan, dtype="float32")
        for sel, ends, fetch_t0, key, a in self.groups:
            if override and key in override:
                a = {**a, **override[key]}
            alive = None if dead is None else {s: ~dead[sel, SIG.index(s)] for s in a}
            f = network_features(a, ends, fetch_t0, alive)
            out[sel] = np.column_stack([f[c] for c in TDAB])
        return out


def blank(X, cols_by_signal, dead):
    """Copy of X with each row's dead channels' columns set to missing."""
    arr = X.to_numpy(dtype="float32", copy=True)
    pos = {c: i for i, c in enumerate(X.columns)}
    for j, s in enumerate(SIG):
        idx = [pos[c] for c in cols_by_signal[s] if c in pos]
        if idx and dead[:, j].any():
            arr[np.ix_(dead[:, j], idx)] = np.nan
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def table(data, fc, dead, net, tdab_cache=None):
    """Feature table for model columns fc with per-row dead channels (TDA-B recomputed without them)."""
    base = [c for c in fc if c not in TDAB]
    X = blank(data.df[base], by_signal(base), dead) if dead is not None and dead.any() else data.df[base].copy()
    if TDAB[0] in fc:
        X[TDAB] = net.compute(dead) if tdab_cache is None else tdab_cache
    return X[fc]


# ---------------------------------------------------------------- D1: models

def train():
    t0 = time.time()
    data = load_data()
    net = Network(data)
    sets = model_sets(data)
    rng = np.random.default_rng(SEED)
    dead_copy = random_dead(len(data.df), DROP_MIN, DROP_MAX, rng)
    clean_b, copy_b = net.compute(), net.compute(dead_copy)
    print(f"{data.df.subject.nunique()} recordings, {len(data.df)} seconds, {int(data.df.label.sum())} seizure "
          f"seconds; TDA-B ready, {time.time() - t0:.0f} s", flush=True)
    y_all = data.df.label.to_numpy()
    for name, fc in sets.items():
        clean = table(data, fc, None, net, clean_b)
        copy = table(data, fc, dead_copy, net, copy_b)
        for k in range(N_FOLDS):
            tr = data.fold != k
            X = pd.concat([clean.loc[tr], copy.loc[tr]])
            y = np.concatenate([y_all[tr], y_all[tr]])
            m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            m.fit(X, y)
            m.save_model(MODELS_DIR / f"d_{name}_fold{k}.json")
        del clean, copy
        print(f"{name}: {len(fc)} features, trained, {time.time() - t0:.0f} s", flush=True)


def load_models(prefix):
    out = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"{prefix}_fold{k}.json")
        out.append(m)
    return out


def oof(data, models, X):
    prob = np.empty(len(X))
    for k, m in enumerate(models):
        sel = data.fold == k
        prob[sel] = m.predict_proba(X[sel])[:, 1]
    return prob


class CountThresholds:
    """Per fold, the threshold for k dead channels: chosen on the other folds' out-of-fold predictions with a
    random k channels blanked per second (seeded, independent of the test sets). Same interface as
    faults/evaluate.Thresholds (fixed, by_sensor, for_set), so evaluate.predict can use it."""

    def __init__(self, data, models, fc, net, budget=SELECT_BUDGET):
        self.data, self.models, self.fc, self.net, self.budget = data, models, fc, net, budget
        self.by_sensor = by_signal(fc)
        self.by_count = {}
        self.fixed = self.for_count(0)

    def for_count(self, k):
        if k not in self.by_count:
            dead = None if k == 0 else random_dead(len(self.data.df), k, k, np.random.default_rng(1000 + k))
            prob = oof(self.data, self.models, table(self.data, self.fc, dead, self.net))
            d = self.data
            self.by_count[k] = [best_threshold(d.other[f], prob[d.fold != f], budget=self.budget)
                                for f in range(N_FOLDS)]
        return self.by_count[k]

    def for_set(self, dead):
        return self.for_count(len(dead))


def score(data, prob, thr):
    det = n = fa = 0
    hours, fa_fold = 0.0, []
    for k in data.events:
        sel = data.fold == k
        d, nk, f, h = data.events[k].score(prob[sel] >= thr[sel], *ALARM)
        det, n, fa, hours = det + d, n + nk, fa + f, hours + h
        fa_fold.append(f / h)
    return det / n, fa / hours, sum(x > FA_BUDGET for x in fa_fold)


def per_patient(data, prob, thr):
    """(subject, detected, seizures, false alarms, hours) per recording, for the bootstrap."""
    out = []
    for sub in data.df.subject.unique():
        rows = np.flatnonzero((data.df.subject == sub).to_numpy())
        fe = FastEvents(data.df.iloc[rows])
        d, n, fa, h = fe.score(prob[rows] >= thr[rows], *ALARM)
        out.append((sub, d, n, fa, h))
    return out


def test_combos(k, seed=SEED):
    """COMBOS_PER_K random sets of k dead channels (seeded per k; all sets for k = 0)."""
    if k == 0:
        return [()]
    rng = np.random.default_rng(seed + 100 * k)
    combos = set()
    while len(combos) < COMBOS_PER_K:
        combos.add(tuple(sorted(rng.choice(N, k, replace=False))))
    return [tuple(SIG[i] for i in c) for c in sorted(combos)]


# ---------------------------------------------------------------- D1 + D2: clean and dead channels

def dead_channels():
    t_start = time.time()
    data = load_data()
    net = Network(data)
    sets = model_sets(data)
    models = {name: load_models(f"d_{name}") for name in sets}
    thr = {name: CountThresholds(data, models[name], fc, net) for name, fc in sets.items()}
    print(f"ready, {time.time() - t_start:.0f} s", flush=True)
    rows, patients = [], []
    for k in DEAD_LEVELS:
        combos = test_combos(k)
        for combo in combos:
            dead = fixed_dead(len(data.df), combo) if combo else None
            tdab = net.compute(dead)
            for name, fc in sets.items():
                X = table(data, fc, dead, net, tdab)
                prob = oof(data, models[name], X)
                for rule in ("count-aware", "fixed"):
                    t_fold = thr[name].for_count(k) if rule == "count-aware" else thr[name].fixed
                    t_row = np.array(t_fold)[data.fold]
                    sens, fa, over = score(data, prob, t_row)
                    rows.append(dict(model=name, rule=rule, dead=k, channels_dead=" ".join(combo) or "none",
                                     sensitivity=sens, fa_per_h=fa, folds_over_budget=over,
                                     **(window_scores(data.df.label, prob) if not combo else {})))
                    if rule == "count-aware" and (k in (0, TARGET_DEAD)):
                        patients += [dict(model=name, dead=k, channels_dead=" ".join(combo) or "none", subject=s,
                                          detected=d, seizures=n, false_alarms=f, hours=h)
                                     for s, d, n, f, h in per_patient(data, prob, t_row)]
        print(f"{k} dead: {len(combos)} sets, {time.time() - t_start:.0f} s", flush=True)
        pd.DataFrame(rows).round(4).to_csv(TABLES / "phaseD_dead_channels.csv", index=False)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / "phaseD_dead_channels.csv", index=False)
    pd.DataFrame(patients).to_csv(TABLES / "phaseD_dead_per_patient.csv", index=False)
    pd.Series({f"{n}|{k}": " ".join(map(str, v)) for n, t in thr.items() for k, v in t.by_count.items()}
              ).rename("thresholds").to_csv(TABLES / "phaseD_count_thresholds.csv")
    clean = res[res.dead == 0].set_index(["model", "rule"]).sensitivity
    res["drop_pts"] = ((clean.reindex(pd.MultiIndex.from_frame(res[["model", "rule"]])).to_numpy()
                        - res.sensitivity) * 100).round(1)
    res.to_csv(TABLES / "phaseD_dead_channels.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[res.dead == 0].to_string(index=False))
    print(res.groupby(["rule", "model", "dead"]).agg(sens=("sensitivity", "mean"), fa=("fa_per_h", "mean"),
                                                     drop=("drop_pts", "mean"), worst=("drop_pts", "max"))
          .round(3).unstack(["rule", "model"]).to_string())
    print(f"done in {time.time() - t_start:.0f} s")


# ---------------------------------------------------------------- D3: health check

FDIR = DS.folder("faults")
HEALTH_SAMPLE = 0.1  # share of whole series kept per recording (SeizeIT2 90 patients: 0.07)


def health():
    from seizure.faults import train_health as th
    from seizure.faults.health import decide
    t0 = time.time()
    data = load_data()
    fmap = data.df.groupby("subject").fold.first()
    df = pd.concat([th.patient_rows(s, FDIR, HEALTH_SAMPLE, SIG) for s in fmap.index], ignore_index=True)
    df["fold"] = df.subject.map(fmap)
    print(f"health: {len(df)} rows, classes {np.bincount(df.y)}, {time.time() - t0:.0f} s", flush=True)
    X = th.features(df, SIG)
    prob = np.zeros((len(df), 3), dtype="float32")
    for k in range(N_FOLDS):
        tr, te = (df.fold != k).to_numpy(), (df.fold == k).to_numpy()
        m = XGBClassifier(**th.PARAMS)
        m.fit(X[tr], df.y[tr])
        m.save_model(MODELS_DIR / f"d_health_fold{k}.json")
        prob[te] = m.predict_proba(X[te])
    gid = df.groupby("sid", sort=False).ngroup().to_numpy()
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    offs = {c: decide(prob[:, 2], gid, c) for c in th.CUTS}
    cuts, off = {}, np.zeros(len(df), bool)
    for k in range(N_FOLDS):
        o = fold != k
        cuts[k] = float(max(th.CUTS, key=lambda c: th.f1_failed(offs[c][o], y[o])))
        off[fold == k] = offs[cuts[k]][fold == k]
    pd.Series(cuts, name="cutoff").rename_axis("fold").to_csv(TABLES / "phaseD_health_cutoffs.csv")
    cls = df.y.map(dict(enumerate(th.CLASSES)))
    sw = pd.Series(off).groupby(cls.to_numpy()).mean().round(3).rename("switched_off")
    sw.rename_axis("true_class").to_csv(TABLES / "phaseD_health_switch_off.csv")
    clean = df.fault == "none"
    per_ch = pd.Series(off[clean]).groupby(df.sensor[clean].to_numpy()).mean().round(4)
    per_ch.rename("healthy_switched_off").rename_axis("channel").to_csv(TABLES / "phaseD_health_clean_channels.csv")
    flagged = (pd.Series(prob[:, 1:].sum(1) > prob[:, 0])[~clean.to_numpy()]
               .groupby([df.fault[~clean].to_numpy(), df.severity[~clean].to_numpy()]).mean().unstack().round(3))
    flagged.to_csv(TABLES / "phaseD_health_by_fault.csv")
    print(f"cut-offs {cuts}\nswitched off by true class:\n{sw.to_string()}\n"
          f"healthy channels switched off: mean {per_ch.mean():.4f}, max {per_ch.max():.4f}\n"
          f"faulty seconds flagged (degraded or failed):\n{flagged.to_string()}\n{time.time() - t0:.0f} s")


# ---------------------------------------------------------------- D4: injected faults

AUG_SHARE = 0.05
AUG_MAX = 11  # fault-augmentation rows get a random 1-11 faulty channels (SeizeIT2: 1-5 of 8)
N_SETS, N_SINGLE = 6, 3


def faulty_activities(fault, sev):
    """{(recording, chunk): {channel: activity}} with the fault injected into each channel of the cached
    cleaned 64 Hz signal, from the same onset as the feature build (same seeds). The positions of pops and
    flat bursts are drawn for 64 Hz, so they differ in detail from the raw 256 Hz injection behind the
    features: TDA-B under faults is therefore an approximation."""
    from seizure.faults.build_faulty import ONSET_RANGE, seed
    from seizure.faults.inject import inject
    out = {}
    for path in sorted(DS.folder("signals").glob("chb*.npz")):
        meta, data = load_patient(path)
        sub = path.stem
        for ci, c in enumerate(meta["chunks"]):
            onset = c["t0"] + np.random.default_rng(seed(sub, c["run"], c["t0"], fault, sev)).uniform(
                *ONSET_RANGE) * (c["t1"] - c["t0"])
            acts = {}
            for s in c["signals"]:
                x, fs = data[(c["i"], s)].astype(float), meta["fs"][s]
                start = int(round((onset - c["fetch_t0"]) * fs))
                if start >= len(x):
                    continue
                y, _ = inject(fault, x, fs, start, sev, np.random.default_rng(seed(sub, c["run"], c["t0"], fault, sev, s)))
                acts[s] = activity(y, fs, *meta["stats"].get(s, (0.0, 1.0)))
            out[(sub, ci)] = acts
    return out


def only(override, channels):
    return {k: {s: a for s, a in v.items() if s in channels} for k, v in override.items()}


def dead_matrix(flags):
    return np.stack([flags[s] for s in SIG], axis=1)


def fault_sets(seed=SEED):
    import random
    rng = random.Random(seed)
    sets = set()
    while len(sets) < N_SETS:
        sets.add(tuple(sorted(rng.sample(SIG, TARGET_DEAD), key=SIG.index)))
    singles = rng.sample(SIG, N_SINGLE)
    return sorted(sets) + [(s,) for s in singles]


def augmentation(data, net):
    """A3 rows (phase_a3.fault_augmentation) for CHB-MIT: per fault condition a random 1-11 channels faulty,
    5% of the seconds after the onset, truly failed windows blanked; TDA-B from the faulty activities
    without the failed channels."""
    import itertools
    import random
    from seizure.faults.inject import FAULTS, SEVERITIES
    rng, pyrng = np.random.default_rng(SEED), random.Random(SEED)
    cols = data.cols
    parts = []
    for fault, sev in itertools.product(FAULTS, SEVERITIES):
        t0 = time.time()
        f = ev.load_condition(data, FDIR, fault, sev)
        sset = tuple(pyrng.sample(SIG, pyrng.randint(1, AUG_MAX)))
        X, failed = ev.assemble(data, f, sset, SIG)
        after = np.zeros(len(X), bool)
        after[f[f.sensor.isin(sset)].pos.to_numpy()] = True
        rows = np.flatnonzero(after & (rng.random(len(X)) < AUG_SHARE))
        dead = dead_matrix(failed)
        A = blank(X[cols].iloc[rows], by_signal(cols), dead[rows])
        tdab = net.compute(dead, only(faulty_activities(fault, sev), sset))
        A[TDAB] = tdab[rows]
        parts.append((A, data.df.label.to_numpy()[rows], data.fold[rows]))
        print(f"augmentation {fault} {sev}: {len(sset)} channels, {len(rows)} rows, {time.time() - t0:.0f} s",
              flush=True)
        del X, f
    return (pd.concat([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def train_fault_models(data, net):
    t0 = time.time()
    aug_X, aug_y, aug_fold = augmentation(data, net)
    rng = np.random.default_rng(SEED)
    dead_copy = random_dead(len(data.df), DROP_MIN, DROP_MAX, rng)  # the same dropout copy as train()
    clean_b, copy_b = net.compute(), net.compute(dead_copy)
    y_all = data.df.label.to_numpy()
    for name, fc in model_sets(data).items():
        clean = table(data, fc, None, net, clean_b)
        copy = table(data, fc, dead_copy, net, copy_b)
        for k in range(N_FOLDS):
            tr, keep = data.fold != k, aug_fold != k
            X = pd.concat([clean.loc[tr], copy.loc[tr], aug_X.loc[keep, fc]])
            y = np.concatenate([y_all[tr], y_all[tr], aug_y[keep]])
            m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            m.fit(X, y)
            m.save_model(MODELS_DIR / f"d3_{name}_fold{k}.json")
            del X
        del clean, copy
        print(f"{name}: fault-trained models saved, {time.time() - t0:.0f} s", flush=True)


def faults():
    import itertools
    from seizure.faults.inject import FAULTS, SEVERITIES
    t_start = time.time()
    data = load_data()
    net = Network(data)
    sets = model_sets(data)
    if not (MODELS_DIR / "d3_classic+TDA+TDAB_fold4.json").exists():
        train_fault_models(data, net)
    runs = [("A3", name) for name in sets] + [("A1", "classic+TDA")]
    models = {r: load_models(f"{'d3' if r[0] == 'A3' else 'd'}_{r[1]}") for r in runs}
    thr = {r: CountThresholds(data, models[r], sets[r[1]], net) for r in runs}
    health = load_models("d_health")
    ev.HEALTH_CUTS.clear()
    ev.HEALTH_CUTS.update(pd.read_csv(TABLES / "phaseD_health_cutoffs.csv").set_index("fold").cutoff.to_dict())
    clean_q = pd.concat([pd.read_parquet(FDIR / f"{s}_quality.parquet") for s in data.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(data.df[["subject", "run", "t"]])).reset_index(drop=True))
    print(f"ready, {time.time() - t_start:.0f} s", flush=True)
    rows, patients, cache = [], [], {}
    for fault, sev in itertools.product(FAULTS, SEVERITIES):
        t0 = time.time()
        f = ev.load_condition(data, FDIR, fault, sev)
        fa_acts = faulty_activities(fault, sev)
        for sset in fault_sets():
            X, _ = ev.assemble(data, f, sset, SIG)
            flags = ev.health_flags(data, clean_q, f, sset, health, cache, SIG)
            dead = dead_matrix(flags)
            n_dead = dead.sum(1)
            ov = only(fa_acts, sset)
            tdab = {"A": net.compute(None, ov), "B": net.compute(dead, ov)}
            for r in runs:
                fc = sets[r[1]]
                for mode in ("A", "B"):
                    Xm = blank(X[[c for c in fc if c not in TDAB]], by_signal(fc), dead) if mode == "B" \
                        else X[[c for c in fc if c not in TDAB]].copy()
                    if TDAB[0] in fc:
                        Xm[TDAB] = tdab[mode]
                    prob = oof(data, models[r], Xm[fc])
                    if mode == "A":
                        t_row = np.array(thr[r].fixed)[data.fold]
                    else:
                        t_row = np.empty(len(prob))
                        for k in np.unique(n_dead):
                            sel = n_dead == k
                            t_row[sel] = np.array(thr[r].for_count(int(k)))[data.fold[sel]]
                    sens, fa, over = score(data, prob, t_row)
                    rows.append(dict(fault=fault, severity=sev, channels=" ".join(sset), n_channels=len(sset),
                                     training=r[0], model=r[1], mode=mode, sensitivity=sens, fa_per_h=fa,
                                     folds_over_budget=over, switched_off=float(dead.mean()) if mode == "B" else 0.0))
                    if mode == "B" and len(sset) == TARGET_DEAD:
                        patients += [dict(training=r[0], model=r[1], fault=fault, severity=sev,
                                          channels=" ".join(sset), subject=s, detected=d, seizures=n,
                                          false_alarms=fa_, hours=h)
                                     for s, d, n, fa_, h in per_patient(data, prob, t_row)]
        print(f"{fault} {sev}: {len(fault_sets())} sets, {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).round(4).to_csv(TABLES / "phaseD_faults.csv", index=False)
        pd.DataFrame(patients).to_csv(TABLES / "phaseD_faults_per_patient.csv", index=False)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / "phaseD_faults.csv", index=False)
    pd.set_option("display.width", 220)
    print(res[res.n_channels == TARGET_DEAD].groupby(["training", "model", "mode", "fault", "severity"])
          [["sensitivity", "fa_per_h"]].mean().round(3).unstack("mode").to_string())
    print(f"done in {time.time() - t_start:.0f} s")


# ---------------------------------------------------------------- statistics

N_BOOT = 2000


def bootstrap(per_patient_rows, a, b, key):
    """Sensitivity of a and b (summed over every scenario of each person), and the bootstrap over people of
    b - a: point difference, 95% interval, one-sided p (share of resamples where b is not better)."""
    r = per_patient_rows.copy()
    r["person"] = r.subject.map(DS.group)
    g = r.groupby([key, "person"])[["detected", "seizures", "false_alarms", "hours"]].sum()
    people = sorted(r.person.unique())
    A, B = g.loc[a].reindex(people).fillna(0).to_numpy(), g.loc[b].reindex(people).fillna(0).to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(people), (N_BOOT, len(people)))
    sa, sb = A[idx].sum(1), B[idx].sum(1)
    diff = sb[:, 0] / sb[:, 1] - sa[:, 0] / sa[:, 1]
    return dict(a=a, b=b, sens_a=A[:, 0].sum() / A[:, 1].sum(), sens_b=B[:, 0].sum() / B[:, 1].sum(),
                fa_a=A[:, 2].sum() / A[:, 3].sum(), fa_b=B[:, 2].sum() / B[:, 3].sum(),
                diff_pts=(B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum()) * 100,
                lo=np.percentile(diff, 2.5) * 100, hi=np.percentile(diff, 97.5) * 100,
                p_not_better=float((diff <= 0).mean()), people=len(people))


def stats():
    out = []
    d = pd.read_csv(TABLES / "phaseD_dead_per_patient.csv")
    for k in (0, TARGET_DEAD):
        part = d[d.dead == k]
        for a, b in (("classic", "classic+TDA"), ("classic+TDA", "classic+TDA+TDAB"), ("classic", "classic+TDA+TDAB")):
            out.append(dict(test=f"{k} channels dead" if k else "clean", **bootstrap(part, a, b, "model")))
    fp = TABLES / "phaseD_faults_per_patient.csv"
    if fp.exists():
        f = pd.read_csv(fp)
        f3 = f[f.training == "A3"]
        for a, b in (("classic", "classic+TDA"), ("classic+TDA", "classic+TDA+TDAB"), ("classic", "classic+TDA+TDAB")):
            out.append(dict(test=f"faults on {TARGET_DEAD} of 18, health check", **bootstrap(f3, a, b, "model")))
        ft = f[f.model == "classic+TDA"].assign(tr=lambda x: x.training)
        out.append(dict(test=f"faults on {TARGET_DEAD} of 18: fault training (classic+TDA)",
                        **bootstrap(ft, "A1", "A3", "tr")))
    res = pd.DataFrame(out).round(4)
    res.to_csv(TABLES / "phaseD_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


def a3_clean():
    """Clean-data scores of the fault-trained (A3) models, the reference for their drop under faults."""
    data = load_data()
    net = Network(data)
    rows = []
    for name, fc in model_sets(data).items():
        models = load_models(f"d3_{name}")
        th = CountThresholds(data, models, fc, net)
        prob = oof(data, models, table(data, fc, None, net))
        sens, fa, over = score(data, prob, np.array(th.fixed)[data.fold])
        rows.append(dict(training="A3", model=name, sensitivity=sens, fa_per_h=fa, folds_over_budget=over,
                         **window_scores(data.df.label, prob)))
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / "phaseD_a3_clean.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    {"acts": build_activities, "train": train, "dead": dead_channels, "health": health, "faults": faults,
     "stats": stats, "a3clean": a3_clean}[sys.argv[1]]()
