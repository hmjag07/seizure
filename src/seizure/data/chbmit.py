"""CHB-MIT: file list, montage check and seizure times (docs/research-log.md, Phase D).

    python -m seizure.data.chbmit

Reads, for every EDF in RECORDS, its header (channels, duration) by HTTP range request, and every
chbXX-summary.txt for the seizure start and end times. CHB-MIT is recorded in the US, so mains noise
is at 60 Hz (SeizeIT2: 50 Hz).

The project uses the 18 bipolar channels of the standard double-banana montage, present in nearly every
file. chb21 is a later recording of the same child as chb01, so both count as one person ("chb01") when
patients are split into folds.

Writes results/tables/chbmit_runs.csv (one row per file) and chbmit_seizures.csv.
"""
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from seizure.data.labels import get_text
from seizure.data.streaming import read_header
from seizure.features.build import TABLES

BUCKET = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
CHANNELS = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ")
SAME_PERSON = {"chb21": "chb01"}


def edf_url(case, run):
    """run is the file stem, e.g. 'chb01_03'."""
    return f"{BUCKET}/{case}/{urllib.parse.quote(run)}.edf"  # one file is named chb02_16+


def person(case):
    return SAME_PERSON.get(case, case)


def parse_summary(text):
    """{file name: [(start s, end s), ...]} from one chbXX-summary.txt.

    Handles both forms used in the dataset: 'Seizure Start Time: 2996 seconds' and
    'Seizure 1 Start Time: 2996 seconds'."""
    out, current, start = {}, None, None
    for line in text.splitlines():
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            current = m[1].removesuffix(".edf")
            out[current] = []
            continue
        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            start = int(m[1])
            continue
        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m and current is not None and start is not None:
            out[current].append((start, int(m[1])))
            start = None
    return out


def scan_file(key):
    case, run = key.split("/")[0], key.split("/")[1].removesuffix(".edf")
    try:
        h = read_header(edf_url(case, run))
    except Exception as e:  # listed in RECORDS but not readable from the bucket
        return dict(case=case, person=person(case), run=run, duration_s=0.0, n_channels=0, missing="",
                    has_18=False, fs="", error=str(e))
    labels = [l.upper() for l in h["labels"]]
    missing = [c for c in CHANNELS if c not in labels]
    fs = {round(f) for l, f in zip(labels, h["fs"]) if l in CHANNELS}
    return dict(case=case, person=person(case), run=run, duration_s=h["n_records"] * h["record_dur"],
                n_channels=len(labels), missing=" ".join(missing), has_18=not missing,
                fs=" ".join(map(str, sorted(fs))), error="")


MAX_SEIZURES = 10  # per person, as for SeizeIT2 (data/fullset.py)
BG_MIN_CHUNKS = 3


def make_plan(context_s, out_file=None, seed=42):
    """Fetch plan with the SeizeIT2 rules (data/fetch_plan.py), adapted to one-hour files:

    - seizures: in files with all 18 channels; at most MAX_SEIZURES per person, drawn at random (seed 42);
    - seizure chunks: each used seizure plus context_s before and after, clipped to its file and merged
      when they overlap;
    - background: CHUNK_S chunks from files with no seizure whose neighbouring files (the previous and next
      file of the same case) have none either, so every chunk is far from any seizure; BG_RATIO x the
      person's used seizure time, at least BG_MIN_CHUNKS per person, drawn at random (seed 42);
    - every chunk is fetched from WARMUP_S earlier (clipped at the file start).
    """
    import math
    import random

    from seizure.data.fetch_plan import BG_RATIO, CHUNK_S, WARMUP_S, merge

    runs = pd.read_csv(TABLES / "chbmit_runs.csv")
    runs = runs[runs.error.fillna("") == ""].reset_index(drop=True)
    sz = pd.read_csv(TABLES / "chbmit_seizures.csv")
    sz = sz[sz.has_18 == True]  # noqa: E712
    rng = random.Random(seed)
    used = []
    for p, g in sz.groupby("person"):
        idx = sorted(rng.sample(range(len(g)), min(MAX_SEIZURES, len(g))))
        used.append(g.iloc[idx])
    used = pd.concat(used)
    dur = runs.set_index("run").duration_s
    plan = []
    for (case, run), g in used.groupby(["case", "run"]):
        ivs = [(max(0.0, r.onset - context_s), min(dur[run], r.onset + r.duration + context_s)) for r in g.itertuples()]
        for a, b in merge(ivs):
            plan.append(dict(subject=case, person=person(case), run=run, kind="seizure", t0=a, t1=b))
    # Background candidates: seizure-free files with seizure-free neighbours.
    candidates = {}
    for case, g in runs.groupby("case", sort=False):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            near = g.iloc[max(0, i - 1): i + 2]
            if r.has_18 and near.n_seizures.sum() == 0:
                t = WARMUP_S
                while t + CHUNK_S <= r.duration_s:
                    candidates.setdefault(person(case), []).append((case, r.run, float(t)))
                    t += CHUNK_S
    for p in sorted(runs.person.unique()):
        target = BG_RATIO * used[used.person == p].duration.sum()
        pool = candidates.get(p, [])
        n = min(len(pool), max(math.ceil(target / CHUNK_S), BG_MIN_CHUNKS))
        for case, run, t in sorted(rng.sample(pool, n)):
            plan.append(dict(subject=case, person=p, run=run, kind="background", t0=t, t1=t + CHUNK_S))
    plan = pd.DataFrame(plan)
    plan["fetch_t0"] = (plan.t0 - WARMUP_S).clip(lower=0.0)
    plan["hours"] = ((plan.t1 - plan.fetch_t0) / 3600).round(3)
    plan = plan.sort_values(["subject", "run", "t0"]).reset_index(drop=True)
    plan.to_csv(out_file or TABLES / "chbmit_fetch_plan.csv", index=False)
    used.to_csv(TABLES / "chbmit_seizures_used.csv", index=False)
    print(f"used seizures: {len(used)} of {len(sz)} (cap {MAX_SEIZURES} per person), "
          f"{used.duration.sum() / 3600:.2f} h of seizure")
    print(plan.groupby("kind").agg(chunks=("run", "size"), hours=("hours", "sum")).round(1).to_string())
    print(f"total {plan.hours.sum():.1f} h; people {plan.person.nunique()}, recordings {plan.subject.nunique()}")
    return plan


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        make_plan(float(sys.argv[2]) if len(sys.argv) > 2 else 600.0)
        return
    keys =[k for k in get_text(f"{BUCKET}/RECORDS").split() if k.endswith(".edf")]
    with ThreadPoolExecutor(16) as ex:
        runs = pd.DataFrame(list(ex.map(scan_file, keys)))
    if (runs.error != "").any():
        print("unreadable files:", runs[runs.error != ""][["run", "error"]].to_string())
    cases = sorted(runs.case.unique())
    with ThreadPoolExecutor(16) as ex:
        summaries = dict(zip(cases, ex.map(lambda c: parse_summary(get_text(f"{BUCKET}/{c}/{c}-summary.txt")),
                                           cases)))
    seizures = []
    for c in cases:
        for run, ivs in summaries[c].items():
            for a, b in ivs:
                seizures.append(dict(case=c, person=person(c), run=run, onset=float(a), duration=float(b - a)))
    seizures = pd.DataFrame(seizures)
    known = set(runs.run)
    seizures["in_records"] = seizures.run.isin(known)
    seizures = seizures.merge(runs[["run", "has_18"]], on="run", how="left")
    runs["n_seizures"] = runs.run.map(seizures.groupby("run").size()).fillna(0).astype(int)
    runs.to_csv(TABLES / "chbmit_runs.csv", index=False)
    seizures.to_csv(TABLES / "chbmit_seizures.csv", index=False)
    print(f"{len(runs)} files, {runs.duration_s.sum() / 3600:.0f} h, {runs.case.nunique()} cases, "
          f"{runs.person.nunique()} people; sampling rates {sorted(set(runs.fs))}")
    print(f"files with all 18 channels: {runs.has_18.sum()} ({runs[runs.has_18].duration_s.sum() / 3600:.0f} h)")
    print(f"seizures: {len(seizures)}, in listed files {seizures.in_records.sum()}, "
          f"in files with all 18 channels {int(seizures.has_18.fillna(False).sum())}")
    bad = runs[~runs.has_18]
    if len(bad):
        print("files missing channels, by case:")
        print(bad.groupby("case").agg(files=("run", "size"), seizures=("n_seizures", "sum"),
                                      example_missing=("missing", "first")).to_string())
    print(seizures.groupby("case").agg(seizures=("onset", "size"), median_s=("duration", "median"),
                                       min_s=("duration", "min"), max_s=("duration", "max")).to_string())


if __name__ == "__main__":
    main()
