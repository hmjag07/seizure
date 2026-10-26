"""Pick SeizeIT2 pilot patients (docs/lab-notebook.md, Step 1.2).

Rules, fixed before the scan:
1. Training range only: sub-001 to sub-096. Held-out patients are never scanned.
2. A seizure counts only if its run has all four modalities (EEG, ECG, EMG, motion).
3. A patient is eligible with at least MIN_SEIZURES such seizures.
4. N_PILOT eligible patients are drawn at random with a fixed seed.
5. Balance check: the pilot's share of asleep seizures and of each seizure type
   must be within BALANCE_TOL of the eligible pool's share.
6. At most MAX_SEIZURES per patient are used, chosen at random with the same seed.

    python -m seizure.data.pilot
"""
import csv
import random
from collections import Counter
from pathlib import Path

from seizure.data.labels import scan_seizeit2

TRAIN_SUBJECTS = [f"sub-{i:03d}" for i in range(1, 97)]
MIN_SEIZURES = 3
N_PILOT = 30
MAX_SEIZURES = 10
BALANCE_TOL = 0.15
SEED = 42
OUT = Path(__file__).resolve().parents[3] / "results" / "tables"


def shares(seizures):
    n = len(seizures)
    out = {"asleep": sum(s["vigilance"] == "asleep" for s in seizures) / n}
    for g, c in Counter(s["group"] for s in seizures).items():
        out[g] = c / n
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    runs, seizures = scan_seizeit2(TRAIN_SUBJECTS)
    rng = random.Random(SEED)

    usable = [s for s in seizures if s["all_modalities"]]
    per_sub = Counter(s["subject"] for s in usable)
    eligible = sorted(sub for sub, n in per_sub.items() if n >= MIN_SEIZURES)
    pilot = sorted(rng.sample(eligible, N_PILOT))

    pool = [s for s in usable if s["subject"] in eligible]
    chosen = [s for s in usable if s["subject"] in pilot]
    pool_sh, pilot_sh = shares(pool), shares(chosen)
    balance = {k: (pool_sh.get(k, 0), pilot_sh.get(k, 0)) for k in sorted(set(pool_sh) | set(pilot_sh))}
    balanced = all(abs(a - b) <= BALANCE_TOL for a, b in balance.values())

    for s in seizures:
        s["used"] = False
    for sub in pilot:
        own = [s for s in usable if s["subject"] == sub]
        for s in rng.sample(own, min(MAX_SEIZURES, len(own))):
            s["used"] = True

    rows = []
    for sub in pilot:
        own = [s for s in usable if s["subject"] == sub]
        groups = Counter(s["group"] for s in own)
        rows.append(dict(subject=sub, usable_seizures=len(own), used_seizures=sum(s["used"] for s in own),
                         asleep=sum(s["vigilance"] == "asleep" for s in own),
                         types=" ".join(f"{g}:{n}" for g, n in sorted(groups.items())),
                         hours_all_modalities=round(sum(r["duration_s"] for r in runs
                                                        if r["subject"] == sub and r["all_modalities"]) / 3600, 1)))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "seizeit2_runs.csv", runs)
    write_csv(OUT / "seizeit2_seizures.csv", seizures)
    write_csv(OUT / "pilot_patients.csv", rows)

    print(f"subjects scanned: {len(TRAIN_SUBJECTS)}, runs: {len(runs)}, seizures: {len(seizures)}")
    print(f"seizures in runs with all 4 modalities: {len(usable)}")
    print(f"subjects with any seizure: {len({s['subject'] for s in seizures})}, "
          f"with >= 1 usable seizure: {len(per_sub)}, eligible (>= {MIN_SEIZURES}): {len(eligible)}")
    print("not eligible:", " ".join(f"{s}({per_sub.get(s, 0)})" for s in TRAIN_SUBJECTS if s not in eligible))
    print("eligible:", " ".join(eligible))
    print("pilot:", " ".join(pilot))
    print("balance (pool vs pilot):", {k: (round(a, 2), round(b, 2)) for k, (a, b) in balance.items()},
          "OK" if balanced else "FAIL")
    print(f"seizures used after cap: {sum(s['used'] for s in seizures)}")


if __name__ == "__main__":
    main()
