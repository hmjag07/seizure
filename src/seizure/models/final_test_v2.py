"""Phase E: the second and final held-out test (v2) (docs/research-log.md, Phase E).

    python -m seizure.models.final_test_v2 train   # final models and thresholds, training patients only
    python -m seizure.models.final_test_v2 dry     # the test code on 5 training patients (code check only)
    python -m seizure.models.final_test_v2 test    # the held-out patients, once

Protocol (fixed 2026-10-25, before the held-out data was read again; approved by the project owner):
Data: the 26 held-out patients of v1 (sub-097 to sub-125 with an all-modality run), 152 seizures, with the
features, injected faults, window labels and signal copy built for v1. Nothing from them is used to train
or to choose anything. Systems, trained on the 90 training patients:
  1. main     XGBoost classic + TDA, fault training (A3), 90-patient health check, one threshold per set of
              dead sensors at <= 8 false alarms/h (Phase C)                                  -> headline
  2. classic  the same with classic features only                                           -> TDA effect
  3. cnn      spectrogram CNN + GRU with fault training (B5)
  4. hybrid   XGBoost classic + TDA + the fault-trained CNN's score (B5)
  5. xgboost  XGBoost classic + TDA with the per-chunk dropout copy, the reference for 3 and 4
Thresholds come from the training patients' out-of-fold predictions (the saved cross-validation models),
as in v1: systems 1-2 one per set of dead sensors, systems 3-5 one per number of switched-off sensors (B4).
Final models: 1, 2 and 5 retrained on all 90 patients (as in cross-validation); 3 a CNN trained on all 90;
4 an XGBoost on all 90 whose CNN scores come from the 5 cross-validation CNNs (each patient scored by a
network that never saw them), with the mean of the 5 networks' scores on the held-out patients.
The health check is retrained on all 90 patients; its cut-off is the median of the Phase A3 fold cut-offs.
Tests: clean data; the 102 dead-sensor sets; injected faults, systems 1-2 on v1's 16 sensor sets with and
without the health check (192 scenarios), systems 3-5 on the 6 five-of-eight sets with it (72 scenarios).
Statistics: Wilson interval for clean sensitivity; patient bootstrap (2,000 resamples) for TDA vs classic and
for the CNN and the hybrid vs XGBoost. Writes results/tables/final2_*.csv.
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import FastEvents, event_scores, window_scores
from seizure.faults import evaluate as ev
from seizure.faults import train_health as th
from seizure.faults.inject import FAULTS, SEVERITIES
from seizure.features.build import ROOT, TABLES
from seizure.features.report import META
from seizure.models import cnn_faults as cf
from seizure.models import cnn_spec as cs
from seizure.models import phase_a3 as a3
from seizure.models.final_test import HELD_FAULTS, HELD_FEATURES, load_dir
from seizure.models.phase_a import load_all
from seizure.models.robustness import EEG, MODELS_DIR, combinations, sensor_columns
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, SELECT_BUDGET, feature_sets

assert cs.TAG == "f", "run with CNN_TAG=f (the fault-trained CNN of B5)"
HELD_SIGNALS = [ROOT / "cache" / "signals_heldout"]
TRAIN_SIGNALS = [ROOT / "cache" / "signals", ROOT / "cache" / "signals_extra"]
NAMES = ("classic", "classic+TDA")
OUT = "final2"
N_BOOT = 2000


def m_path(name):
    return MODELS_DIR / f"{OUT}_{name}"


def load_xgb(name):
    m = XGBClassifier()
    m.load_model(m_path(name).with_suffix(".json"))
    return m


def load_cnn(path):
    net = cs.SpecNet().to(cs.DEVICE)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


# ---------------------------------------------------------------- stage 1: training patients only

def train():
    t_start = time.time()
    data = ev.Data(raw=load_all())
    fsets = feature_sets(data.cols)
    print(f"90 patients, {len(data.df)} seconds, {time.time() - t_start:.0f} s", flush=True)

    # Systems 1-2: A3 on all 90 patients (clean rows, dropout copy, every fault-augmentation row).
    aug_X, aug_y, _ = a3.fault_augmentation(data)
    y_clean = data.df.label.to_numpy()
    for nm in NAMES:
        fc = fsets[nm]
        pos = {c: i for i, c in enumerate(fc)}
        idx_by_sensor = {s: [pos[c] for c in cs_] for s, cs_ in sensor_columns(fc).items()}
        n = len(data.df)
        table = np.empty((2 * n + len(aug_X), len(fc)), dtype="float32")
        table[:n] = data.df[fc].to_numpy(dtype="float32")
        table[n:2 * n] = table[:n]
        a3.blank_random_sensors(table[n:2 * n], idx_by_sensor, np.random.default_rng(SEED + 99))
        table[2 * n:] = aug_X[fc].to_numpy(dtype="float32")
        y = np.concatenate([y_clean, y_clean, aug_y])
        m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        m.fit(pd.DataFrame(table, columns=fc, copy=False), y)
        m.save_model(m_path(f"a3_{nm}").with_suffix(".json"))
        del table
        print(f"final A3 {nm} trained, {time.time() - t_start:.0f} s", flush=True)
    del aug_X

    # Health check on all 90 patients (7% of whole series, as in A3).
    fmap = data.df.groupby("subject").fold.first()
    df = pd.concat([th.patient_rows(s, a3.faults_dir(s), a3.HEALTH_SAMPLE) for s in fmap.index], ignore_index=True)
    hm = XGBClassifier(**th.PARAMS)
    hm.fit(th.features(df), df.y)
    hm.save_model(m_path("health").with_suffix(".json"))
    del df
    print(f"final health check trained, {time.time() - t_start:.0f} s", flush=True)

    # Systems 3-5.
    del data, fsets, y_clean
    W, fc, by_sensor, pos, X_clean, inner, models, combos, _ = cs.prepare_hybrid()
    net = cs.train_net(W, np.arange(len(W.rows)), SEED + 100, tag="final CNN")
    torch.save(net.state_dict(), m_path("cnn").with_suffix(".pt"))
    fold_cnns = [load_cnn(cs.MODELS / f"cnn_specf_fold{k}.pt") for k in range(N_FOLDS)]
    dead_copy = cs.chunk_dead(W, np.random.default_rng(SEED))
    oof_clean, oof_copy = np.empty(len(W.rows), np.float32), np.empty(len(W.rows), np.float32)
    for k, fnet in enumerate(fold_cnns):
        idx = np.flatnonzero(W.fold == k)
        oof_clean[idx] = cs.score_rows(W, fnet, idx)
        oof_copy[idx] = cs.score_rows(W, fnet, idx, dead_copy)
    X_copy = X_clean.copy()
    for j, s in enumerate(SIGNALS):
        X_copy[np.ix_(dead_copy[:, j], [pos[c] for c in by_sensor[s]])] = np.nan
    y = np.concatenate([W.label, W.label])
    for name in ("xgboost", "hybrid"):
        X = np.concatenate([X_clean, X_copy])
        cols = list(fc)
        if name == "hybrid":
            X = np.column_stack([X, np.concatenate([oof_clean, oof_copy])])
            cols += ["cnn_score"]
        m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        m.fit(pd.DataFrame(X, columns=cols, copy=False), y)
        m.save_model(m_path(name).with_suffix(".json"))
        del X
        print(f"final {name} trained, {time.time() - t_start:.0f} s", flush=True)

    # Count-aware thresholds for systems 3-5 from the cross-validation models (B5), k = 0-8 switched off.
    meta = W.rows[["subject", "run", "t", "label"]]
    everyone = FastEvents(meta)
    rows = []
    for count in range(len(SIGNALS) + 1):
        dead = None
        if count:
            rng = np.random.default_rng(1000 + count)
            dead = (rng.random((W.seg.max() + 1, len(SIGNALS))).argsort(1).argsort(1) < count)[W.seg]
        p = cv_probs(W, fc, by_sensor, pos, X_clean, inner, models, fold_cnns, dead)
        rows.append(dict(count=count, **{m: best_threshold(everyone, pr, budget=SELECT_BUDGET) for m, pr in p.items()}))
        print(f"thresholds for {count} switched off: {rows[-1]}", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / f"{OUT}_count_thresholds.csv", index=False)
    print(f"stage 1 done in {time.time() - t_start:.0f} s")


def blank_rows(X, dead, by_sensor, pos):
    X = X.copy()
    if dead is not None:
        for j, s in enumerate(SIGNALS):
            if dead[:, j].any():
                X[np.ix_(dead[:, j], [pos[c] for c in by_sensor[s]])] = np.nan
    return X


def cv_probs(W, fc, by_sensor, pos, X_clean, inner, models, fold_cnns, dead):
    """Out-of-fold probabilities of the B5 cross-validation models (xgboost, hybrid with inner-network scores,
    cnn) with the given per-row dead sensors."""
    n = len(W.rows)
    X = blank_rows(X_clean, dead, by_sensor, pos)
    inner_s, cnn = np.empty(n, np.float32), np.empty(n, np.float32)
    for k in range(N_FOLDS):
        idx = np.flatnonzero(W.fold == k)
        inner_s[idx] = np.mean([cs.score_rows(W, net, idx, dead) for net in inner[k]], axis=0)
        cnn[idx] = cs.score_rows(W, fold_cnns[k], idx, dead)
    out = {}
    for name, ms in models.items():
        prob = np.empty(n)
        for k, m in enumerate(ms):
            sel = W.fold == k
            Xk = X[sel] if name == "xgboost" else np.column_stack([X[sel], inner_s[sel]])
            cols = list(fc) + ([] if name == "xgboost" else ["cnn_score"])
            prob[sel] = m.predict_proba(pd.DataFrame(Xk, columns=cols, copy=False))[:, 1]
        out[name] = prob
    out["cnn"] = cnn
    return out


class SetThresholds:
    """Systems 1-2: one threshold per set of dead sensors, from the A3 cross-validation models' out-of-fold
    predictions on all 90 training patients, at <= SELECT_BUDGET (final_test.FinalThresholds with the
    Phase A2 budget). Interface of faults/evaluate.Thresholds (one fold)."""

    def __init__(self, data, events, fold_models, fc):
        self.data, self.events, self.models, self.fc, self.cache = data, events, fold_models, fc, {}
        self.by_sensor = sensor_columns(fc)
        self.fixed = self.for_set(frozenset())

    def for_set(self, dead):
        if dead not in self.cache:
            d, prob = self.data, np.empty(len(self.data.df))
            for k, m in enumerate(self.models):
                X = d.df.loc[d.fold == k, self.fc].copy()
                for s in dead:
                    X[self.by_sensor[s]] = np.nan
                prob[d.fold == k] = m.predict_proba(X)[:, 1]
            self.cache[dead] = [best_threshold(self.events, prob, budget=SELECT_BUDGET)]
        return self.cache[dead]


# ---------------------------------------------------------------- stage 2: the test

def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def per_patient(data, prob, thr):
    out = []
    for sub, rows in data.df.groupby("subject").indices.items():
        d, n, fa, h = FastEvents(data.df.iloc[rows]).score(prob[rows] >= thr[rows], *ALARM)
        out.append(dict(subject=sub, detected=d, seizures=n, false_alarms=fa, hours=h))
    return out


def test(dry=False):
    t_start = time.time()
    tag = "_dry" if dry else ""
    train90 = ev.Data(raw=load_all())
    fsets = feature_sets(train90.cols)
    if dry:  # 5 training patients stand in for the held-out set; the numbers mean nothing (models saw them)
        subs = sorted(train90.df.subject.unique())[:5]
        raw = load_all()
        held = ev.Data(raw=raw[raw.subject.isin(subs)].reset_index(drop=True), single=True)
        faults_dir, signal_dirs = a3.faults_dir, TRAIN_SIGNALS
        conditions = [("dropout", 0.25), ("decline", 1.0)]
    else:
        held = ev.Data(raw=load_dir(HELD_FEATURES, META + train90.base), single=True)
        faults_dir, signal_dirs = HELD_FAULTS, HELD_SIGNALS
        conditions = list(itertools.product(FAULTS, SEVERITIES))
    print(f"{'DRY RUN' if dry else 'HELD-OUT TEST'}: {held.df.subject.nunique()} patients, {len(held.df)} seconds, "
          f"{int(held.df.label.sum())} seizure seconds, {time.time() - t_start:.0f} s", flush=True)
    events90 = FastEvents(train90.df)
    patients = []

    # ---- systems 1-2
    models12, thr12 = {}, {}
    for nm in NAMES:
        key = ("A3", nm)
        models12[key] = [load_xgb(f"a3_{nm}")]
        thr12[key] = SetThresholds(train90, events90, a3.load("A3", nm), fsets[nm])
        print(f"{nm}: clean threshold {thr12[key].fixed[0]}", flush=True)
    clean_rows = []
    for (tr, nm), (m,) in models12.items():
        prob = m.predict_proba(held.df[fsets[nm]])[:, 1]
        thr = thr12[(tr, nm)].fixed[0]
        e = event_scores(held.df.assign(p=prob), "p", thr, *ALARM)
        lo, hi = wilson(e["detected"], e["seizures"])
        clean_rows.append(dict(system="main" if nm == "classic+TDA" else "classic", threshold=thr,
                               **window_scores(held.df.label, prob), **e, sens_lo=lo, sens_hi=hi))
        patients += [dict(system=clean_rows[-1]["system"], test="clean", **r)
                     for r in per_patient(held, prob, np.full(len(prob), thr))]
    dead_rows = []
    all_combos = combinations()
    # dry run: a few sets, including one with 5 dead and an EEG alive, so every code path runs
    combos = (all_combos[:4] + [next(c for c in all_combos if len(c) == 5 and not EEG <= set(c))]) if dry         else all_combos
    for (tr, nm), (m,) in models12.items():
        system = "main" if nm == "classic+TDA" else "classic"
        ths = thr12[(tr, nm)]
        for combo in combos:
            X = held.df[fsets[nm]].copy()
            for s in combo:
                X[ths.by_sensor[s]] = np.nan
            prob = m.predict_proba(X)[:, 1]
            for rule, thr in (("fixed", ths.fixed[0]), ("sensor-aware", ths.for_set(frozenset(combo))[0])):
                sens, fa = ev.score(held, prob, np.full(len(prob), thr))
                dead_rows.append(dict(system=system, rule=rule, dead=len(combo), sensors_dead=" ".join(combo) or "none",
                                      eeg_alive=not EEG <= set(combo), sensitivity=sens, fa_per_h=fa))
                if rule == "sensor-aware" and len(combo) == 5 and not EEG <= set(combo):
                    patients += [dict(system=system, test="5 of 8 dead", **r)
                                 for r in per_patient(held, prob, np.full(len(prob), thr))]
        print(f"{system}: dead sensors done, {time.time() - t_start:.0f} s", flush=True)

    health = load_xgb("health")
    ev.HEALTH_CUTS.clear()
    ev.HEALTH_CUTS.update({0: float(pd.read_csv(TABLES / "phaseA3_health_cutoffs.csv").cutoff.median())})
    clean_q = pd.concat([pd.read_parquet(ev.fault_dir(faults_dir, s) / f"{s}_quality.parquet")
                         for s in held.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(held.df[["subject", "run", "t"]])).reset_index(drop=True))
    sets = ev.sensor_sets()[:3] if dry else ev.sensor_sets()
    fault_rows = ev.run_scenarios(held, faults_dir, conditions, sets, models12, thr12, fsets, [health], clean_q,
                                  TABLES / f"{OUT}_faults_main{tag}.csv", modes=("A", "B"))
    pd.DataFrame(fault_rows).round(4).to_csv(TABLES / f"{OUT}_faults_main{tag}.csv", index=False)
    print(f"systems 1-2 done, {time.time() - t_start:.0f} s", flush=True)

    # ---- systems 3-5
    cthr = pd.read_csv(TABLES / f"{OUT}_count_thresholds.csv").set_index("count")
    rows_h = held.df[["subject", "run", "t", "label"]].assign(fold=0)
    W = cs.GpuWindows(rows=rows_h, dirs=signal_dirs)
    W_cpu = {s: W.sig[s].cpu() for s in SIGNALS}
    layout = cf.chunk_layout(W, signal_dirs)
    final_cnn = load_cnn(m_path("cnn").with_suffix(".pt"))
    fold_cnns = [load_cnn(cs.MODELS / f"cnn_specf_fold{k}.pt") for k in range(N_FOLDS)]
    xgb, hyb = load_xgb("xgboost"), load_xgb("hybrid")
    fc = fsets["classic+TDA"]
    by_sensor = sensor_columns(fc)
    pos = {c: i for i, c in enumerate(fc)}
    every = np.arange(len(W.rows))
    X_held = held.df[fc].to_numpy(dtype="float32")

    def probs35(X, dead):
        score = np.mean([cs.score_rows(W, net, every, dead) for net in fold_cnns], axis=0)
        return {"xgboost": xgb.predict_proba(pd.DataFrame(X, columns=fc, copy=False))[:, 1],
                "hybrid": hyb.predict_proba(pd.DataFrame(np.column_stack([X, score]), columns=list(fc) + ["cnn_score"],
                                                         copy=False))[:, 1],
                "cnn": cs.score_rows(W, final_cnn, every, dead)}

    def thr_rows(model, counts):
        return cthr[model].to_numpy()[counts]

    p = probs35(X_held, None)
    for system, prob in p.items():
        thr = thr_rows(system, np.zeros(len(prob), int))
        e = event_scores(held.df.assign(p=prob), "p", thr[0], *ALARM)
        lo, hi = wilson(e["detected"], e["seizures"])
        clean_rows.append(dict(system=system, threshold=thr[0], **window_scores(held.df.label, prob), **e,
                               sens_lo=lo, sens_hi=hi))
        patients += [dict(system=system, test="clean", **r) for r in per_patient(held, prob, thr)]
    clean = pd.DataFrame(clean_rows).round(4)
    clean.to_csv(TABLES / f"{OUT}_clean{tag}.csv", index=False)
    print(clean[["system", "threshold", "auroc", "aupr", "seizures", "detected", "sensitivity", "sens_lo", "sens_hi",
                 "fa_per_h", "median_delay_s"]].to_string(index=False), flush=True)

    for combo in combos:
        dead = cs.combo_dead(len(W.rows), combo) if combo else None
        p = probs35(blank_rows(X_held, dead, by_sensor, pos), dead)
        for system, prob in p.items():
            for rule, k in (("fixed", 0), ("count-aware", len(combo))):
                thr = thr_rows(system, np.full(len(prob), k))
                sens, fa = ev.score(held, prob, thr)
                dead_rows.append(dict(system=system, rule=rule, dead=len(combo), sensors_dead=" ".join(combo) or "none",
                                      eeg_alive=not EEG <= set(combo), sensitivity=sens, fa_per_h=fa))
                if rule == "count-aware" and len(combo) == 5 and not EEG <= set(combo):
                    patients += [dict(system=system, test="5 of 8 dead", **r) for r in per_patient(held, prob, thr)]
    pd.DataFrame(dead_rows).round(4).to_csv(TABLES / f"{OUT}_dead_sensors{tag}.csv", index=False)
    print(f"dead sensors done, {time.time() - t_start:.0f} s", flush=True)

    five = [s for s in ev.sensor_sets() if len(s) == 5][: (2 if dry else None)]
    frows, flag_cache = [], {}
    for fault, sev in conditions:
        f = ev.load_condition(held, faults_dir, fault, sev)
        buffers = cf.faulty_buffers(W_cpu, layout, f, fault, sev)
        for sset in five:
            X_f, _ = ev.assemble(held, f, sset)
            flags = ev.health_flags(held, clean_q, f, sset, [health], flag_cache)
            dead = np.stack([flags[s] for s in SIGNALS], axis=1)
            for s in sset:
                W.sig[s] = buffers[s].to(cs.DEVICE)
            p = probs35(blank_rows(X_f[fc].to_numpy(dtype="float32"), dead, by_sensor, pos), dead)
            for s in sset:
                W.sig[s] = W_cpu[s].to(cs.DEVICE)
            counts = dead.sum(1)
            for system, prob in p.items():
                thr = thr_rows(system, counts)
                sens, fa = ev.score(held, prob, thr)
                frows.append(dict(fault=fault, severity=sev, sensors=" ".join(sset), system=system, sensitivity=sens,
                                  fa_per_h=fa, switched_off=float(dead.mean())))
                patients += [dict(system=system, test="faults on 5 of 8", fault=fault, severity=sev, sensors=" ".join(sset),
                                  **r) for r in per_patient(held, prob, thr)]
        print(f"systems 3-5 {fault} {sev}: done, {time.time() - t_start:.0f} s", flush=True)
        pd.DataFrame(frows).round(4).to_csv(TABLES / f"{OUT}_faults_dl{tag}.csv", index=False)
    pd.DataFrame(frows).round(4).to_csv(TABLES / f"{OUT}_faults_dl{tag}.csv", index=False)

    # Per-patient results of systems 1-2 under faults (health check on, the 6 five-of-eight sets), for the bootstrap.
    pd.DataFrame(patients).to_csv(TABLES / f"{OUT}_per_patient{tag}.csv", index=False)
    stats(tag, held, models12, thr12, fsets, faults_dir, conditions, health, clean_q)
    print(f"done in {time.time() - t_start:.0f} s")


def stats(tag, held, models12, thr12, fsets, faults_dir, conditions, health, clean_q):
    """Per-patient results of systems 1-2 under faults (health check, five-of-eight sets), then the bootstrap."""
    patients = pd.read_csv(TABLES / f"{OUT}_per_patient{tag}.csv")
    five = [s for s in ev.sensor_sets() if len(s) == 5][: (2 if tag else None)]
    extra, flag_cache = [], {}
    for fault, sev in conditions:
        f = ev.load_condition(held, faults_dir, fault, sev)
        for sset in five:
            X, _ = ev.assemble(held, f, sset)
            flags = ev.health_flags(held, clean_q, f, sset, [health], flag_cache)
            for (tr, nm), ms in models12.items():
                prob, thr = ev.predict(held, ms, X, fsets[nm], thr12[(tr, nm)], flags)
                extra += [dict(system="main" if nm == "classic+TDA" else "classic", test="faults on 5 of 8",
                               fault=fault, severity=sev, sensors=" ".join(sset), **r)
                          for r in per_patient(held, prob, thr)]
    patients = pd.concat([patients, pd.DataFrame(extra)], ignore_index=True)
    patients.to_csv(TABLES / f"{OUT}_per_patient{tag}.csv", index=False)
    g = patients.groupby(["test", "system", "subject"])[["detected", "seizures", "false_alarms", "hours"]].sum()
    people = sorted(patients.subject.unique())
    boot = np.random.default_rng(SEED).integers(0, len(people), (N_BOOT, len(people)))
    out = []
    for test_name in ("clean", "5 of 8 dead", "faults on 5 of 8"):
        for a, b in (("classic", "main"), ("xgboost", "cnn"), ("xgboost", "hybrid")):
            if (test_name, a) not in g.index.droplevel(2) or (test_name, b) not in g.index.droplevel(2):
                print(f"no results for {test_name}: {a} vs {b}", flush=True)
                continue
            A = g.loc[(test_name, a)].reindex(people).fillna(0).to_numpy()
            B = g.loc[(test_name, b)].reindex(people).fillna(0).to_numpy()
            ab, bb = A[boot].sum(1), B[boot].sum(1)
            ds = bb[:, 0] / bb[:, 1] - ab[:, 0] / ab[:, 1]
            out.append(dict(test=test_name, a=a, b=b, sens_a=A[:, 0].sum() / A[:, 1].sum(),
                            sens_b=B[:, 0].sum() / B[:, 1].sum(), fa_a=A[:, 2].sum() / A[:, 3].sum(),
                            fa_b=B[:, 2].sum() / B[:, 3].sum(),
                            diff_pts=(B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum()) * 100,
                            lo=np.percentile(ds, 2.5) * 100, hi=np.percentile(ds, 97.5) * 100,
                            p_not_better=float((ds <= 0).mean())))
    res = pd.DataFrame(out).round(4)
    res.to_csv(TABLES / f"{OUT}_stats{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


if __name__ == "__main__":
    {"train": train, "dry": lambda: test(dry=True), "test": test}[sys.argv[1]]()
