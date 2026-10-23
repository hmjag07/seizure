"""The other 66 training-range patients ("extra"), to train on all 96 (docs/pipeline.md, Phase A1).

    python -m seizure.data.fullset

Rules, fixed 2026-10-24 before building:
1. Every patient of sub-001 to sub-096 that is not in the pilot (66 patients), including those with
   fewer than 3 or no usable seizures: they add background variety, which drives false alarms.
2. Every seizure in a run with all four modalities, at most 10 per patient (random, seed 42), as in
   the pilot.
3. Fetch plan as for the pilot (data/fetch_plan.py), with at least one 10-minute background chunk per
   patient.
The 30 pilot patients keep their existing plan and features. Labels come from the scan already in
results/tables/seizeit2_*.csv (training range only).

Writes results/tables/extra_fetch_plan.csv and extra_seizures.csv.
Then: SEIZURE_SET=extra python -m seizure.features.build features (and the fault build and window labels).
"""
import csv
import random
from collections import defaultdict

from seizure.data.fetch_plan import SEED, TABLES, make_plan

MAX_SEIZURES = 10


def main():
    pilot = {r["subject"] for r in csv.DictReader(open(TABLES / "pilot_patients.csv"))}
    subjects = [f"sub-{i:03d}" for i in range(1, 97) if f"sub-{i:03d}" not in pilot]
    runs = [r for r in csv.DictReader(open(TABLES / "seizeit2_runs.csv"))
            if r["subject"] in subjects and r["all_modalities"] == "True"]
    seizures = [s for s in csv.DictReader(open(TABLES / "seizeit2_seizures.csv")) if s["subject"] in subjects]

    rng = random.Random(SEED)
    by_sub = defaultdict(list)
    for s in seizures:
        s["used"] = "False"
        if s["all_modalities"] == "True":
            by_sub[s["subject"]].append(s)
    for sub in sorted(by_sub):
        for s in rng.sample(by_sub[sub], min(MAX_SEIZURES, len(by_sub[sub]))):
            s["used"] = "True"

    with open(TABLES / "extra_seizures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seizures[0]))
        w.writeheader()
        w.writerows(seizures)
    used = [s for s in seizures if s["used"] == "True"]
    print(f"{len(subjects)} patients, {len(seizures)} seizures, {sum(len(v) for v in by_sub.values())} usable, "
          f"{len(used)} used after the cap, from {len({s['subject'] for s in used})} patients")
    make_plan(subjects, runs, seizures, TABLES / "extra_fetch_plan.csv", min_bg_chunks=1)


if __name__ == "__main__":
    main()
