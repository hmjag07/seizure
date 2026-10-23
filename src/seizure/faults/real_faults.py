"""Phase C4: does the health check catch the motion sensors that are genuinely broken in SeizeIT2?

    python -m seizure.faults.real_faults

Step 1.4 found real broken motion sensors (dead, stuck or wrongly scaled axes) in about 11% of the pilot's
chunks; until now they were simply removed. Here, for every chunk of the 90 training patients:
1. Stream the motion file and classify each motion sensor (accelerometer / gyroscope x neck / chest):
   broken-stuck (an axis constant or pinned at its edge for over half the chunk, preprocess.stuck),
   broken-scale (accelerometer whose gravity reads outside 0.1-1.5, preprocess.valid_scale), or healthy.
2. For broken-stuck sensors and an equal number of randomly chosen healthy ones (seed 42): clean the
   sensor without the stuck check (preprocess.clean_one), compute its quality features and context
   (faults/health.py), and apply the 90-patient health check of the patient's fold with its smoothing
   and cut-off (phase_a3). A wrongly scaled accelerometer cannot be cleaned at all (gravity calibration
   fails), so it is rejected before the health check and counted separately.
3. The same sensor-chunks through the stuck rule (faults/health.stuck_motion, the last 60 s of every
   axis, past only): an axis pinned at its full-scale limit for over half the window, or exactly constant;
   a gyroscope frozen at 0 counts only if its accelerometer is frozen too (a gyroscope reads exactly 0 at
   rest). Variants show why this form: a 300 s window, the window's own min / max as the edge (what
   preprocess.stuck uses), and any constant value.
4. The rule's false switch-offs on every healthy motion sensor of the 90 patients.
Reports the share of seconds switched off per sensor by the health check, the rule and both together.
Writes results/tables/phaseC4_real_faults.csv and phaseC4_real_faults_healthy_rule.csv.
"""
import random

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from seizure.data import preprocess as pp
from seizure.data.signals import load_patient
from seizure.data.windows import WINDOW_S, fits, window_ends
from seizure.faults.health import (QUALITY, add_context, decide, quality_features, raw_signal, stuck_motion,
                                   stuck_trailing)
from seizure.faults.train_health import features as health_features
from seizure.data.labels import edf_url
from seizure.data.streaming import read_header, read_physical
from seizure.features.build import ROOT, TABLES
from seizure.models.robustness import MODELS_DIR
from seizure.models.train import SEED

PLANS = [TABLES / "fetch_plan.csv", TABLES / "extra_fetch_plan.csv"]
SIGNAL_DIRS = [ROOT / "cache" / "signals", ROOT / "cache" / "signals_extra"]
MOTION = ("ACC_neck", "ACC_chest", "GYR_neck", "GYR_chest")


def plan(subjects=None):
    p = pd.concat([pd.read_csv(f, dtype={"run": str}) for f in PLANS], ignore_index=True)
    p["run"] = p.run.str.zfill(2)
    return p if subjects is None else p[p.subject.isin(subjects)].reset_index(drop=True)


def stats_for(sub):
    for d in SIGNAL_DIRS:
        if (d / f"{sub}.npz").exists():
            return load_patient(d / f"{sub}.npz")[0]["stats"]
    return {}


def classify(mov):
    """{sensor: 'healthy' | 'stuck' | 'scale' | 'missing'} and the gravity scale per device."""
    out, scales = {}, {}
    for dev, name in pp.DEVICES.items():
        acc = pp.acc_xyz(mov, dev)
        scales[dev] = pp.gravity_scale([acc]) if acc is not None else None
        for kind, axes in (("ACC", "XYZ"), ("GYR", "ABC")):
            keys = [f"{name} {kind} {a}" for a in axes]
            s = f"{kind}_{dev}"
            if not all(k in mov for k in keys):
                out[s] = "missing"
            elif any(pp.stuck(mov[k][0]) for k in keys):
                out[s] = "stuck"
            elif kind == "ACC" and not pp.valid_scale(scales[dev]):
                out[s] = "scale"
            else:
                out[s] = "healthy"
    return out, scales


def main(subjects=None, out_file=TABLES / "phaseC4_real_faults.csv"):
    p = plan(subjects)
    fold_of = (pd.read_parquet(ROOT / "cache" / "predictions" / "phaseA_oof.parquet", columns=["subject", "fold"])
               .drop_duplicates().set_index("subject").fold)
    cuts = pd.read_csv(TABLES / "phaseA3_health_cutoffs.csv").set_index("fold").cutoff
    health = {}
    for k in range(5):
        m = XGBClassifier()
        m.load_model(MODELS_DIR / f"health90_fold{k}.json")
        health[k] = m

    # 1. Scan.
    found, movs = [], {}
    headers = {}
    for row in p.itertuples():
        key = (row.subject, row.run)
        if key not in headers:
            headers[key] = read_header(edf_url(row.subject, row.run, "mov"))
        mov = read_physical(edf_url(row.subject, row.run, "mov"), headers[key], row.fetch_t0, row.t1)
        status, scales = classify(mov)
        for s, st in status.items():
            found.append(dict(subject=row.subject, run=row.run, t0=row.t0, t1=row.t1, fetch_t0=row.fetch_t0,
                              sensor=s, status=st))
        movs[row.Index] = (mov, scales)
    scan = pd.DataFrame(found)
    print("motion sensors per chunk:\n", scan.groupby(["sensor", "status"]).size().unstack(fill_value=0).to_string(),
          flush=True)

    # 2. Health check on the stuck sensors and matched healthy controls.
    rng = random.Random(SEED)
    stuck = scan[scan.status == "stuck"]
    healthy = scan[scan.status == "healthy"]
    controls = []
    for s, n in stuck.sensor.value_counts().items():
        pool = healthy[healthy.sensor == s]
        controls.append(pool.iloc[rng.sample(range(len(pool)), min(n, len(pool)))])
    chosen = pd.concat([stuck.assign(group="broken (stuck)")] + [c.assign(group="healthy control") for c in controls])
    index_of = {(r.subject, r.run, r.t0): r.Index for r in p.itertuples()}

    def axes_of(mov, sensor):
        kind, dev = sensor.split("_")
        return [mov[f"{pp.DEVICES[dev]} {kind} {a}"][0] for a in ("XYZ" if kind == "ACC" else "ABC")]

    # Rule variants compared. "rule" is the one the system uses (health.stuck_motion, 60 s); the others
    # show why: the window's own edge and plain constancy both fire on healthy sensors at rest.
    VARIANTS = ("rule", "rule_300s", "rule_window_edge", "rule_constant_any")

    def rules(r, mov, ends):
        h = headers[(r.subject, r.run)]
        axes = axes_of(mov, r.sensor)
        return {"rule": stuck_motion(mov, h, r.sensor, 25, ends, r.fetch_t0),
                "rule_300s": stuck_motion(mov, h, r.sensor, 25, ends, r.fetch_t0, window_s=300),
                "rule_window_edge": stuck_trailing(axes, 25, ends, r.fetch_t0, limits="window"),
                "rule_constant_any": stuck_trailing(axes, 25, ends, r.fetch_t0)}

    def ends_for(r, n_samples):
        ends = window_ends(r.fetch_t0, r.t0, r.t1)
        return ends[(np.round((ends - r.fetch_t0) * 25) <= n_samples) & (ends - r.fetch_t0 >= WINDOW_S[r.sensor])]

    results = []
    for r in chosen.itertuples():
        mov, scales = movs[index_of[(r.subject, r.run, r.t0)]]
        axes = axes_of(mov, r.sensor)
        ends = ends_for(r, len(axes[0]))
        rl = rules(r, mov, ends)
        raw = {"eeg": {}, "ecg": {}, "emg": {}, "mov": mov}
        x = pp.clean_one(raw, r.sensor, scales)
        if x is None:  # accelerometer whose gravity calibration fails: rejected at cleaning, every second
            health_off, note = np.ones(len(ends), bool), "rejected at cleaning"
        else:
            ends = ends[fits(x, pp.FS[r.sensor], WINDOW_S[r.sensor], r.fetch_t0, ends)]
            rl = {k: v[:len(ends)] for k, v in rl.items()}
            q = quality_features(r.sensor, x, raw_signal(raw, r.sensor), stats_for(r.subject), r.fetch_t0, ends)
            part = pd.DataFrame({c: q[f"{r.sensor}_{c}"] for c in QUALITY})
            part["seg"] = 0
            part = add_context(part, "seg")
            part["sensor"] = r.sensor
            k = int(fold_of[r.subject])
            p_failed = health[k].predict_proba(health_features(part))[:, 2]
            health_off, note = decide(p_failed, np.zeros(len(part), int), float(cuts[k])), ""
        results.append(dict(**{c: getattr(r, c) for c in ("subject", "run", "t0", "sensor", "status", "group")},
                            seconds=len(ends), health_check=health_off.mean(),
                            **{k: v.mean() for k, v in rl.items()},
                            **{"combined_" + k.removeprefix("rule_"): (health_off | v).mean() for k, v in rl.items()},
                            note=note))
    res = pd.DataFrame(results)
    scale_rows = scan[scan.status == "scale"].assign(
        group="broken (wrong scale)", health_check=1.0,
        **{"combined_" + k.removeprefix("rule_"): 1.0 for k in VARIANTS},
        note="rejected at cleaning (gravity calibration)")
    res = pd.concat([res, scale_rows], ignore_index=True)
    res.to_csv(out_file, index=False)
    cols = ["health_check", *VARIANTS, *["combined_" + k.removeprefix("rule_") for k in VARIANTS]]
    print("share of seconds switched off (mean over sensor-chunks):")
    print(res.groupby(["group", "sensor"])[cols].mean().round(3).to_string())
    print(res.groupby("group")[cols].mean().round(3).to_string())

    # 3. False alarms of the rule on every healthy motion sensor (no model needed).
    fp = []
    for r in healthy.itertuples():
        mov, _ = movs[index_of[(r.subject, r.run, r.t0)]]
        axes = axes_of(mov, r.sensor)
        ends = ends_for(r, len(axes[0]))
        fp.append(dict(sensor=r.sensor, seconds=len(ends), **{k: v.mean() for k, v in rules(r, mov, ends).items()}))
    fp = pd.DataFrame(fp)
    fp.to_csv(out_file.with_name(out_file.stem + "_healthy_rule.csv"), index=False)
    print("rule on all healthy motion sensors, share of seconds switched off:")
    print(fp.groupby("sensor")[list(VARIANTS)].mean().round(4).to_string())
    print(fp[list(VARIANTS)].mean().round(4).to_string())


if __name__ == "__main__":
    main()
