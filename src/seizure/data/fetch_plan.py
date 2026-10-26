"""Plan which time ranges to stream for the pilot patients (docs/lab-notebook.md, Step 1.3).

Seizure chunks: each used seizure plus CONTEXT_S before and after, merged when they overlap.
Background chunks: CHUNK_S long, at least GAP_S from any seizure and IMPD_MARGIN_S from any
impedance check, BG_RATIO x the patient's used seizure time, drawn at random (seed 42).
Every chunk is fetched from WARMUP_S earlier so causal filters settle and the longest
window (ECG) is full before the first output second.

    python -m seizure.data.fetch_plan
"""
import csv
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from seizure.data.labels import read_events

CONTEXT_S = 600
GAP_S = 1800
IMPD_MARGIN_S = 60
BG_RATIO = 5
CHUNK_S = 600
WARMUP_S = 30
SEED = 42
TABLES = Path(__file__).resolve().parents[3] / "results" / "tables"


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlaps(a, b, zones):
    return any(a < z1 and z0 < b for z0, z1 in zones)


def make_plan(subjects, runs, seizures, out_file, min_bg_chunks=0):
    """Fetch plan for subjects. runs: rows with all four modalities; seizures: rows with a "used" flag.

    min_bg_chunks: background chunks per patient at least (when available), so patients with few or no
    used seizures still contribute background seconds and normalisation statistics.
    """
    with ThreadPoolExecutor(16) as ex:
        events = dict(zip([(r["subject"], r["run"]) for r in runs],
                          ex.map(lambda r: read_events(r["subject"], r["run"]), runs)))

    rng = random.Random(SEED)
    plan = []
    for sub in subjects:
        sub_runs = {r["run"]: float(r["duration_s"]) for r in runs if r["subject"] == sub}
        used = [s for s in seizures if s["subject"] == sub and s["used"] == "True"]

        by_run = defaultdict(list)
        for s in used:
            on, d = float(s["onset"]), float(s["duration"])
            by_run[s["run"]].append((max(0.0, on - CONTEXT_S), min(sub_runs[s["run"]], on + d + CONTEXT_S)))
        for run, iv in by_run.items():
            for a, b in merge(iv):
                plan.append(dict(subject=sub, run=run, kind="seizure", t0=a, t1=b))

        # Background must avoid every seizure (used or not) and every impedance check.
        candidates = []
        for run, dur in sub_runs.items():
            zones = []
            for e in events[(sub, run)]:
                on, d = float(e["onset"]), float(e["duration"])
                if e["eventType"].startswith("sz"):
                    zones.append((on - GAP_S, on + d + GAP_S))
                elif e["eventType"] == "impd":
                    zones.append((on - IMPD_MARGIN_S, on + d + IMPD_MARGIN_S))
            t = WARMUP_S
            while t + CHUNK_S <= dur:
                if not overlaps(t, t + CHUNK_S, zones):
                    candidates.append((run, t))
                t += CHUNK_S
        target = BG_RATIO * sum(float(s["duration"]) for s in used)
        n = min(len(candidates), max(math.ceil(target / CHUNK_S), min_bg_chunks))
        for run, t in sorted(rng.sample(candidates, n)):
            plan.append(dict(subject=sub, run=run, kind="background", t0=float(t), t1=float(t + CHUNK_S)))

    for p in plan:
        p["fetch_t0"] = max(0.0, p["t0"] - WARMUP_S)
        p["hours"] = round((p["t1"] - p["fetch_t0"]) / 3600, 3)
    plan.sort(key=lambda p: (p["subject"], p["run"], p["t0"]))
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0]))
        w.writeheader()
        w.writerows(plan)

    for kind in ("seizure", "background"):
        ps = [p for p in plan if p["kind"] == kind]
        print(f"{kind}: {len(ps)} chunks, {sum(p['hours'] for p in ps):.1f} h")
    print(f"total: {sum(p['hours'] for p in plan):.1f} h")


def main():
    pilot = [r["subject"] for r in csv.DictReader(open(TABLES / "pilot_patients.csv"))]
    runs = [r for r in csv.DictReader(open(TABLES / "seizeit2_runs.csv"))
            if r["subject"] in pilot and r["all_modalities"] == "True"]
    seizures = [s for s in csv.DictReader(open(TABLES / "seizeit2_seizures.csv")) if s["subject"] in pilot]
    make_plan(pilot, runs, seizures, TABLES / "fetch_plan.csv")


if __name__ == "__main__":
    main()
