"""Phase B4: the spectrogram CNN + GRU and the hybrid under injected faults (docs/lab-notebook.md).

    python -m seizure.models.cnn_faults

The Phase C protocol (faults on 5 of 8 sensors, 12 fault conditions x the 6 sensor sets of
faults/evaluate.sensor_sets, 90 training patients, 5 folds), for three models (cnn_spec.py):
  xgboost   XGBoost classic + TDA with the per-chunk dropout copy (the B3 reference)
  hybrid    the same plus the spectrogram CNN's score
  cnn       the spectrogram CNN + GRU
Each second, the 90-patient health check (phase_a3, its fold cut-offs) decides which sensors are switched
off; the model gets the remaining sensors, and the threshold is the one chosen on the other folds for that
NUMBER of switched-off sensors (count-aware, as for CHB-MIT in Phase D; <= 8 false alarms per hour on clean
data). One threshold per set of switched-off sensors (Phase C) needs a full scoring of every model for each of
the many sets the health check produces, too slow for three networks per model.

XGBoost features under faults come from the fault build (features recomputed from faulty raw signals,
evaluate.assemble). The CNN reads signals, so the same fault (type, severity and the chunk's onset, read from
the fault build) is injected into the cached cleaned, downsampled, normalised signals of the faulty sensors.
Pop and flat-burst positions are drawn at the cached sampling rate, so the CNN's faults match the feature
build's in kind, severity and onset, not sample by sample. Writes results/tables/cnn_faults*.csv.
"""
import itertools
import time

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.data.signals import load_patient
from seizure.evaluation.metrics import FastEvents
from seizure.faults import evaluate as ev
from seizure.faults.build_faulty import seed
from seizure.faults.inject import FAULTS, SEVERITIES, inject, spread
from seizure.features.build import TABLES
from seizure.models import cnn_spec as cs
from seizure.models import phase_a3 as a3
from seizure.models.cnn import FS, SIGNAL_DIRS
from seizure.models.robustness import EEG, MODELS_DIR
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, SELECT_BUDGET

ACC_POP_FLOOR = 0.05  # g: as build_faulty for motion; cached accelerometers are calibrated to g


def chunk_layout(W, dirs=None):
    """[(subject, run, t0, t1, fetch_t0, sensor, offset, length)] of every chunk's signal in the buffers of
    cnn.Windows (the same loop, so the offsets match)."""
    rows = W.rows
    size = {s: 0 for s in SIGNALS}
    out = []
    for d in (dirs or SIGNAL_DIRS):
        for path in sorted(d.glob("sub-*.npz")):
            sub = path.stem
            if not (rows.subject == sub).any():
                continue
            meta, data = load_patient(path)
            for c in meta["chunks"]:
                for s in c["signals"]:
                    n = len(data[(c["i"], s)])
                    out.append((sub, c["run"], c["t0"], c["t1"], c["fetch_t0"], s, size[s], n))
                    size[s] += n
    return out


def faulty_buffers(W_cpu, layout, f, fault, sev):
    """{sensor: CPU float16 buffer} with the fault injected into every chunk from its onset."""
    onsets = f.groupby(["subject", "run"]).apply(lambda g: g.groupby("onset").t.agg(["min", "max"]),
                                                 include_groups=False)
    out = {s: W_cpu[s].clone() for s in SIGNALS}
    for sub, run, t0, t1, fetch_t0, s, off, n in layout:
        try:
            cand = onsets.loc[(sub, run)]
        except KeyError:
            continue
        hit = cand[(cand["min"] >= t0) & (cand["max"] <= t1 + 1)]
        if hit.empty:
            continue
        onset = float(hit.index[0])
        fs = FS[s]
        start = int(round((onset - fetch_t0) * fs))
        x = out[s][off:off + n].float().numpy()
        if start >= n or start < 1:
            continue
        scale = spread(x[:start])
        if s.startswith("ACC"):
            scale = max(scale, ACC_POP_FLOOR)
        y, _ = inject(fault, x.astype(float), fs, start, sev, np.random.default_rng(seed(sub, run, t0, fault, sev, s)),
                      scale)
        out[s][off:off + n] = torch.from_numpy(np.clip(y, -20, 20).astype(np.float16))
    return out


def dead_code(dead):
    return dead.astype(np.int64) @ (np.int64(1) << np.arange(len(SIGNALS), dtype=np.int64))


def main():
    t_start = time.time()
    W, fc, by_sensor, pos, X_clean, inner, xgb_models, combos, _ = cs.prepare_hybrid()
    W_cpu = {s: W.sig[s].cpu() for s in SIGNALS}
    layout = chunk_layout(W)
    for s in SIGNALS:  # the layout must cover each buffer exactly (a sensor that never worked has a 1-sample placeholder)
        total = sum(n_ for *_, ss, _o, n_ in layout if ss == s)
        assert total == len(W_cpu[s]) or (total == 0 and len(W_cpu[s]) == 1), s
    cnn_nets = []
    for k in range(N_FOLDS):
        net = cs.SpecNet().to(cs.DEVICE)
        net.load_state_dict(torch.load(cs.MODELS / f"cnn_spec{cs.TAG}_fold{k}.pt"))
        net.eval()
        cnn_nets.append(net)
    from seizure.models.phase_a import load_all
    data = ev.Data(raw=load_all())
    health = [XGBClassifier() for _ in range(N_FOLDS)]
    for k, m in enumerate(health):
        m.load_model(MODELS_DIR / f"health90_fold{k}.json")
    ev.HEALTH_CUTS.clear()
    ev.HEALTH_CUTS.update(pd.read_csv(TABLES / "phaseA3_health_cutoffs.csv").set_index("fold").cutoff.to_dict())
    clean_q = pd.concat([pd.read_parquet(a3.faults_dir(s) / f"{s}_quality.parquet") for s in data.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(data.df[["subject", "run", "t"]])).reset_index(drop=True))
    meta = W.rows[["subject", "run", "t", "label"]]
    other = {k: FastEvents(meta[W.fold != k]) for k in range(N_FOLDS)}
    events_fold = {k: FastEvents(meta[W.fold == k]) for k in range(N_FOLDS)}
    subject_rows = W.rows.groupby("subject").indices
    subject_events = {sub: FastEvents(meta.iloc[r]) for sub, r in subject_rows.items()}
    n = len(W.rows)
    print(f"ready, {time.time() - t_start:.0f} s", flush=True)

    def probs(X, dead, with_cnn=True):
        """{model: probabilities}: features X (already blanked), signals in W.sig, dead (rows x 8) or None."""
        cnn_inner = np.empty(n, np.float32)
        cnn = np.empty(n, np.float32)
        for k in range(N_FOLDS):
            idx = np.flatnonzero(W.fold == k)
            cnn_inner[idx] = np.mean([cs.score_rows(W, net, idx, dead) for net in inner[k]], axis=0)
            if with_cnn:
                cnn[idx] = cs.score_rows(W, cnn_nets[k], idx, dead)
        out = {}
        for name, ms in xgb_models.items():
            prob = np.empty(n)
            for k, m in enumerate(ms):
                sel = W.fold == k
                Xk = X[sel] if name == "xgboost" else np.column_stack([X[sel], cnn_inner[sel]])
                cols = list(fc) if name == "xgboost" else list(fc) + ["cnn_score"]
                prob[sel] = m.predict_proba(pd.DataFrame(Xk, columns=cols, copy=False))[:, 1]
            out[name] = prob
        if with_cnn:
            out["cnn"] = cnn
        return out

    def blank(X, dead):
        X = X.copy()
        for j, s in enumerate(SIGNALS):
            if dead[:, j].any():
                X[np.ix_(dead[:, j], [pos[c] for c in by_sensor[s]])] = np.nan
        return X

    thr_cache = {}

    def thresholds(count):
        """{model: per-fold thresholds} for `count` switched-off sensors (count-aware, as for CHB-MIT): chosen on
        the other folds' clean data with a random `count` sensors dead in each chunk (one set per chunk, held for
        the whole chunk as the CNN was trained; seed 1000 + count)."""
        if count not in thr_cache:
            dead = None
            if count:
                rng = np.random.default_rng(1000 + count)
                per_chunk = rng.random((W.seg.max() + 1, len(SIGNALS))).argsort(1).argsort(1) < count
                dead = per_chunk[W.seg]
            X = X_clean if dead is None else blank(X_clean, dead)
            p = probs(X, dead)
            thr_cache[count] = {m: [best_threshold(other[k], pr[W.fold != k], budget=SELECT_BUDGET)
                                    for k in range(N_FOLDS)] for m, pr in p.items()}
            print(f"  thresholds for {count} switched off: {thr_cache[count]}", flush=True)
        return thr_cache[count]

    five = [s for s in ev.sensor_sets() if len(s) == 5]
    rows, patients, flag_cache = [], [], {}
    for fault, sev in itertools.product(FAULTS, SEVERITIES):
        t0 = time.time()
        f = ev.load_condition(data, a3.faults_dir, fault, sev)
        buffers = faulty_buffers(W_cpu, layout, f, fault, sev)
        for sset in five:
            X_f, _ = ev.assemble(data, f, sset)
            X_f = X_f[fc].to_numpy(dtype="float32")
            flags = ev.health_flags(data, clean_q, f, sset, health, flag_cache)
            dead = np.stack([flags[s] for s in SIGNALS], axis=1)
            for s in sset:
                W.sig[s] = buffers[s].to(cs.DEVICE)
            p = probs(blank(X_f, dead), dead)
            for s in sset:
                W.sig[s] = W_cpu[s].to(cs.DEVICE)
            counts = dead.sum(1)
            thr_rows = {m: np.empty(n) for m in p}
            for c in np.unique(counts):
                sel = counts == c
                th = thresholds(int(c))
                for m in p:
                    thr_rows[m][sel] = np.array(th[m])[W.fold[sel]]
            for m, prob in p.items():
                det = nn = fa = 0
                hours, fa_fold = 0.0, []
                for k in range(N_FOLDS):
                    sel = W.fold == k
                    d, nk, fk, h = events_fold[k].score(prob[sel] >= thr_rows[m][sel], *ALARM)
                    det, nn, fa, hours = det + d, nn + nk, fa + fk, hours + h
                    fa_fold.append(fk / h)
                rows.append(dict(fault=fault, severity=sev, sensors=" ".join(sset), model=m, sensitivity=det / nn,
                                 fa_per_h=fa / hours, folds_over_budget=sum(x > FA_BUDGET for x in fa_fold),
                                 switched_off=float(dead.mean())))
                for sub, ridx in subject_rows.items():
                    d, nk, fk, h = subject_events[sub].score(prob[ridx] >= thr_rows[m][ridx], *ALARM)
                    patients.append(dict(fault=fault, severity=sev, sensors=" ".join(sset), model=m, subject=sub,
                                         detected=d, seizures=nk, false_alarms=fk, hours=h))
            print(f"  {fault} {sev} {' '.join(sset)}: " + ", ".join(
                f"{r['model']} {r['sensitivity']:.3f}/{r['fa_per_h']:.1f}" for r in rows[-len(p):]), flush=True)
        print(f"{fault} {sev}: {len(five)} sets, {len(thr_cache)} threshold sets, {time.time() - t0:.0f} s, "
              f"total {time.time() - t_start:.0f} s", flush=True)
        pd.DataFrame(rows).round(4).to_csv(TABLES / f"cnn_faults{cs.TAG}.csv", index=False)
        pd.DataFrame(patients).to_csv(TABLES / f"cnn_faults{cs.TAG}_per_patient.csv", index=False)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(TABLES / f"cnn_faults{cs.TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.groupby(["model", "fault", "severity"])[["sensitivity", "fa_per_h"]].mean().round(3)
          .unstack("model").to_string())
    print(res.groupby("model")[["sensitivity", "fa_per_h"]].mean().round(3).to_string())
    print(f"done in {time.time() - t_start:.0f} s")


def stats(n_boot=2000, seed_=42):
    """Patient bootstrap over all 72 scenarios: CNN and hybrid against XGBoost, seizures caught and false alarms."""
    r = pd.read_csv(TABLES / f"cnn_faults{cs.TAG}_per_patient.csv")
    g = r.groupby(["model", "subject"])[["detected", "seizures", "false_alarms", "hours"]].sum()
    people = sorted(r.subject.unique())
    boot = np.random.default_rng(seed_).integers(0, len(people), (n_boot, len(people)))
    base = g.loc["xgboost"].reindex(people).to_numpy()
    out = []
    for m in ("cnn", "hybrid"):
        a = g.loc[m].reindex(people).to_numpy()
        ab, bb = a[boot].sum(1), base[boot].sum(1)
        ds = ab[:, 0] / ab[:, 1] - bb[:, 0] / bb[:, 1]
        df = ab[:, 2] / ab[:, 3] - bb[:, 2] / bb[:, 3]
        out.append(dict(model=m, sens=a[:, 0].sum() / a[:, 1].sum(), fa_per_h=a[:, 2].sum() / a[:, 3].sum(),
                        xgboost_sens=base[:, 0].sum() / base[:, 1].sum(),
                        xgboost_fa=base[:, 2].sum() / base[:, 3].sum(),
                        sens_diff_pts=(a[:, 0].sum() / a[:, 1].sum() - base[:, 0].sum() / base[:, 1].sum()) * 100,
                        sens_lo=np.percentile(ds, 2.5) * 100, sens_hi=np.percentile(ds, 97.5) * 100,
                        p_not_better=float((ds <= 0).mean()),
                        fa_diff=a[:, 2].sum() / a[:, 3].sum() - base[:, 2].sum() / base[:, 3].sum(),
                        fa_lo=np.percentile(df, 2.5), fa_hi=np.percentile(df, 97.5)))
    res = pd.DataFrame(out).round(4)
    res.to_csv(TABLES / f"cnn_faults{cs.TAG}_stats.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    import sys
    stats() if sys.argv[1:] == ["stats"] else main()
