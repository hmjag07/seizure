"""Phase C1 + C2: ablation study with patient-level statistics.

    python -m seizure.models.phase_c train    # the variants that need new models
    python -m seizure.models.phase_c eval     # score every variant, per patient
    python -m seizure.models.phase_c stats    # bootstrap confidence intervals and paired tests

Variants (90 training patients, the 5 patient-grouped folds, thresholds at <= 8 FA/h on the other folds):
  full            A3 classic+TDA, health check, sensor-aware threshold
  -TDA            A3 classic
  -TDA size       A3 classic + TDA shape summaries only
  -TDA shape      A3 classic + TDA size summaries only
  -memory         A3 classic+TDA without the 10 s / 60 s context features
  -fault train    A1 classic+TDA (sensor dropout only)
  -robust train   classic+TDA, no dropout copy and no fault rows
  -health check   full model, faulty sensors stay on, fixed threshold
  -sensor-aware   full model and health check, but one fixed threshold
Scored on clean data and on injected faults on 5 of 8 sensors (12 fault conditions x 6 sensor sets),
per patient (detected seizures, seizures, false alarms, hours), so phase C1 can resample patients.
Writes results/tables/phaseC_*.csv.
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.evaluation.metrics import FastEvents
from seizure.faults import evaluate as ev
from seizure.faults.inject import FAULTS, SEVERITIES
from seizure.features.build import TABLES
from seizure.models import phase_a3 as a3
from seizure.models.phase_a import load_all
from seizure.models.robustness import MODELS_DIR, sensor_columns
from seizure.models.train import ALARM, N_FOLDS, PARAMS, SEED, SELECT_BUDGET, SHAPE, feature_sets

N_BOOT = 2000


def variant_columns(cols):
    fs = feature_sets(cols)
    base = lambda c: c.removesuffix("_m10").removesuffix("_d60")
    full = fs["classic+TDA"]
    return {
        "full": full,
        "-TDA": fs["classic"],
        "-TDA size": fs["classic+TDA_shape"],
        "-TDA shape": [c for c in full if not (c.startswith("TDA_") and base(c).endswith(SHAPE))],
        "-memory": [c for c in full if not c.endswith(("_m10", "_d60"))],
        "-fault train": full,
        "-robust train": full,
    }


# Which saved models each variant uses (prefix of cache/models/<prefix>_fold<k>.json).
MODEL_FILES = {"full": "a3_classic+TDA", "-TDA": "a3_classic", "-TDA size": "c_tda_shape",
               "-TDA shape": "c_tda_size", "-memory": "c_no_memory", "-fault train": "a1_classic+TDA",
               "-robust train": "c_no_robust"}


def train():
    data = ev.Data(raw=load_all())
    cols = variant_columns(data.cols)
    aug_X, aug_y, aug_fold = a3.fault_augmentation(data)
    for name in ("-TDA size", "-TDA shape", "-memory", "-robust train"):
        fc = cols[name]
        pos = {c: i for i, c in enumerate(fc)}
        idx_by_sensor = {s: [pos[c] for c in cs] for s, cs in sensor_columns(fc).items()}
        for k in range(N_FOLDS):
            t0 = time.time()
            tr, keep = data.fold != k, aug_fold != k
            n = int(tr.sum())
            if name == "-robust train":
                table = data.df.loc[tr, fc].to_numpy(dtype="float32")
                y = data.df.label.to_numpy()[tr]
            else:
                n_aug = int(keep.sum())
                table = np.empty((2 * n + n_aug, len(fc)), dtype="float32")
                table[:n] = data.df.loc[tr, fc].to_numpy(dtype="float32")
                table[n:2 * n] = table[:n]
                a3.blank_random_sensors(table[n:2 * n], idx_by_sensor, np.random.default_rng(SEED + k))
                table[2 * n:] = aug_X.loc[keep, fc].to_numpy(dtype="float32")
                y_clean = data.df.label.to_numpy()[tr]
                y = np.concatenate([y_clean, y_clean, aug_y[keep]])
            m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            m.fit(pd.DataFrame(table, columns=fc, copy=False), y)
            m.save_model(MODELS_DIR / f"{MODEL_FILES[name]}_fold{k}.json")
            del table
            print(f"{name} fold {k}: {time.time() - t0:.0f} s", flush=True)


def load_models(prefix):
    out = []
    for k in range(N_FOLDS):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"{prefix}_fold{k}.json")
        out.append(m)
    return out


def per_patient(data, patient_events, prob, thr):
    """Detected, seizures, false alarms and hours for each patient."""
    out = []
    for sub, (rows, fe) in patient_events.items():
        d, n, fa, h = fe.score(prob[rows] >= thr[rows], *ALARM)
        out.append((sub, d, n, fa, h))
    return out


def evaluate():
    t_start = time.time()
    data = ev.Data(raw=load_all())
    cols = variant_columns(data.cols)
    patient_events = {}
    for sub in data.df.subject.unique():
        rows = np.flatnonzero((data.df.subject == sub).to_numpy())
        patient_events[sub] = (rows, FastEvents(data.df.iloc[rows]))
    models = {name: load_models(prefix) for name, prefix in MODEL_FILES.items()}
    thresholds = {name: ev.Thresholds(data, models[name], cols[name], name, name, budget=SELECT_BUDGET)
                  for name in MODEL_FILES}
    health = [XGBClassifier() for _ in range(N_FOLDS)]
    for k, m in enumerate(health):
        m.load_model(MODELS_DIR / f"health90_fold{k}.json")
    ev.HEALTH_CUTS.update(pd.read_csv(TABLES / "phaseA3_health_cutoffs.csv").set_index("fold").cutoff.to_dict())
    clean_q = pd.concat([pd.read_parquet(a3.faults_dir(s) / f"{s}_quality.parquet") for s in data.df.subject.unique()])
    clean_q = (clean_q.drop_duplicates(["subject", "run", "t"]).set_index(["subject", "run", "t"])
               .reindex(pd.MultiIndex.from_frame(data.df[["subject", "run", "t"]])).reset_index(drop=True))
    print(f"ready, {time.time() - t_start:.0f} s", flush=True)

    # (variant, model key, health check on, sensor-aware threshold)
    runs = [(v, v, True, True) for v in MODEL_FILES] + [("-health check", "full", False, False),
                                                        ("-sensor-aware", "full", True, False)]
    rows = []

    def record(variant, scenario, fault, sev, sset, stats):
        for sub, d, n, fa, h in stats:
            rows.append(dict(variant=variant, scenario=scenario, fault=fault, severity=sev, sensors=sset,
                             subject=sub, detected=d, seizures=n, false_alarms=fa, hours=h))

    for variant, key, _, _ in runs:
        prob, thr = ev.predict(data, models[key], data.df[data.cols], cols[key], thresholds[key], None)
        record(variant, "clean", "none", 0.0, "", per_patient(data, patient_events, prob, thr))
    print(f"clean done, {time.time() - t_start:.0f} s", flush=True)

    five = [s for s in ev.sensor_sets() if len(s) == 5]
    clean_flags = {}
    for fault, sev in itertools.product(FAULTS, SEVERITIES):
        t0 = time.time()
        f = ev.load_condition(data, a3.faults_dir, fault, sev)
        for sset in five:
            X, _ = ev.assemble(data, f, sset)
            flagged = ev.health_flags(data, clean_q, f, sset, health, clean_flags)
            for variant, key, use_health, aware in runs:
                prob, thr = ev.predict(data, models[key], X, cols[key], thresholds[key],
                                       flagged if use_health else None, sensor_aware=aware)
                record(variant, f"{fault} {sev}", fault, sev, " ".join(sset),
                       per_patient(data, patient_events, prob, thr))
        print(f"{fault} {sev}: {len(five)} sets x {len(runs)} variants, {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(TABLES / "phaseC_per_patient.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "phaseC_per_patient.csv", index=False)
    print(f"done in {time.time() - t_start:.0f} s")


def stats():
    """Bootstrap over patients: CIs per variant, and paired differences against the full system."""
    r = pd.read_csv(TABLES / "phaseC_per_patient.csv")
    r["kind"] = np.where(r.scenario == "clean", "clean", "faults on 5 of 8")
    # One row per variant x kind x patient: sums over the scenarios of that kind.
    g = r.groupby(["variant", "kind", "subject"])[["detected", "seizures", "false_alarms", "hours"]].sum()
    clean_sens = (r[r.kind == "clean"].groupby("variant").detected.sum()
                  / r[r.kind == "clean"].groupby("variant").seizures.sum())
    subjects = r.subject.unique()
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(subjects), (N_BOOT, len(subjects)))
    variants = list(dict.fromkeys(r.variant))
    out = []
    for kind in ("clean", "faults on 5 of 8"):
        mats = {}
        for v in variants:
            m = g.loc[(v, kind)].reindex(subjects).fillna(0)
            mats[v] = m.to_numpy()  # columns: detected, seizures, false_alarms, hours
        for v in variants:
            m = mats[v]
            sens = m[:, 0].sum() / m[:, 1].sum()
            fa = m[:, 2].sum() / m[:, 3].sum()
            bs = m[boot].sum(1)
            sens_b, fa_b = bs[:, 0] / bs[:, 1], bs[:, 2] / bs[:, 3]
            row = dict(kind=kind, variant=v, sensitivity=sens, sens_lo=np.percentile(sens_b, 2.5),
                       sens_hi=np.percentile(sens_b, 97.5), fa_per_h=fa, fa_lo=np.percentile(fa_b, 2.5),
                       fa_hi=np.percentile(fa_b, 97.5))
            if kind != "clean":
                # Average drop against the variant's own clean sensitivity.
                row["drop_pts"] = (clean_sens[v] - sens) * 100
            if v != "full":
                f = mats["full"][boot].sum(1)
                diff = f[:, 0] / f[:, 1] - sens_b
                row.update(diff_vs_full=(mats["full"][:, 0].sum() / mats["full"][:, 1].sum() - sens) * 100,
                           diff_lo=np.percentile(diff, 2.5) * 100, diff_hi=np.percentile(diff, 97.5) * 100,
                           p_full_not_better=float((diff <= 0).mean()))
            out.append(row)
    res = pd.DataFrame(out).round(4)
    res.to_csv(TABLES / "phaseC_ablation.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


if __name__ == "__main__":
    {"train": train, "eval": evaluate, "stats": stats}[sys.argv[1]]()
