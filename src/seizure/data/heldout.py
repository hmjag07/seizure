"""Held-out test patients: label scan and fetch plan (docs/research-log.md, Step 1.2 and the final test).

    python -m seizure.data.heldout

The 29 patients sub-097 to sub-125 were not read before this point. Rules, fixed before any of their
data was touched (2026-10-23), so the final test cannot be tuned to them:
1. All 29 patients are included.
2. Every seizure in a run with all four modalities is used: no minimum per patient and no cap. The
   pilot's minimum and cap balanced training; a test should count every seizure, as the SeizeIT2
   benchmark does.
3. Fetch plan as for the pilot (data/fetch_plan.py): 10 minutes around each seizure, background at
   least 30 minutes from any seizure and clear of impedance checks, 5x the seizure time, plus at
   least one 10-minute background chunk per patient so every patient contributes background seconds
   and normalisation statistics.
4. TDA settings, health check, models and thresholds stay as chosen on the pilot.

Writes results/tables/heldout_runs.csv, heldout_seizures.csv and heldout_fetch_plan.csv.
Then: SEIZURE_SET=heldout python -m seizure.features.build features (and faults/build_faulty.py,
faults/window_labels.py).
"""
import csv

from seizure.data.fetch_plan import TABLES, make_plan
from seizure.data.labels import scan_seizeit2

HELDOUT_SUBJECTS = [f"sub-{i:03d}" for i in range(97, 126)]
MIN_BG_CHUNKS = 1


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    runs, seizures = scan_seizeit2(HELDOUT_SUBJECTS)
    for s in seizures:
        s["used"] = s["all_modalities"]
    write_csv(TABLES / "heldout_runs.csv", runs)
    write_csv(TABLES / "heldout_seizures.csv", seizures)
    usable = [s for s in seizures if s["used"]]
    print(f"{len(HELDOUT_SUBJECTS)} patients, {len(runs)} runs, {len(seizures)} seizures, "
          f"{len(usable)} in runs with all modalities, from {len({s['subject'] for s in usable})} patients")
    # make_plan reads the CSV convention: strings "True"/"False".
    as_csv = lambda rows: [{k: str(v) if isinstance(v, bool) else v for k, v in r.items()} for r in rows]
    all_mod_runs = [r for r in as_csv(runs) if r["all_modalities"] == "True"]
    make_plan(HELDOUT_SUBJECTS, all_mod_runs, as_csv(seizures), TABLES / "heldout_fetch_plan.csv",
              min_bg_chunks=MIN_BG_CHUNKS)


if __name__ == "__main__":
    main()
