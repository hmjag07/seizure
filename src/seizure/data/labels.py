"""Seizure annotations: SeizeIT2 events.tsv and CHB-MIT summary files."""
import csv
import io
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEIZEIT2_BUCKET = "https://s3.amazonaws.com/openneuro.org"
SEIZEIT2_PREFIX = "ds005873"
MODALITIES = ("eeg", "ecg", "emg", "mov")


def get_text(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()


def list_keys(prefix):
    """All object keys under a prefix in the OpenNeuro S3 bucket."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = get_text(f"{SEIZEIT2_BUCKET}?{urllib.parse.urlencode(q)}")
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        token = (re.findall(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml) or [None])[0]
        if not token:
            return keys


def seizure_group(event_type):
    """Coarse seizure type: FA (focal aware), FIA (focal impaired awareness), FBTC, other."""
    if event_type.startswith("sz_foc_ia"):
        return "FIA"
    if event_type.startswith("sz_foc_a"):
        return "FA"
    if event_type == "sz_foc_f2b":
        return "FBTC"
    return "other"


def events_url(sub, run):
    return f"{SEIZEIT2_BUCKET}/{SEIZEIT2_PREFIX}/{sub}/ses-01/eeg/{sub}_ses-01_task-szMonitoring_run-{run}_events.tsv"


def edf_url(sub, run, modality):
    return f"{SEIZEIT2_BUCKET}/{SEIZEIT2_PREFIX}/{sub}/ses-01/{modality}/{sub}_ses-01_task-szMonitoring_run-{run}_{modality}.edf"


def read_events(sub, run):
    """All rows of one run's events.tsv as dicts."""
    return list(csv.DictReader(io.StringIO(get_text(events_url(sub, run))), delimiter="\t"))


def scan_subject(sub):
    """Runs and seizures for one SeizeIT2 subject, e.g. 'sub-001'."""
    keys = list_keys(f"{SEIZEIT2_PREFIX}/{sub}/")
    runs = {}
    for k in keys:
        m = re.search(r"run-(\d+)_(eeg|ecg|emg|mov)\.edf$", k)
        if m:
            runs.setdefault(m[1], set()).add(m[2])
    run_rows, seizures = [], []
    for k in sorted(k for k in keys if k.endswith("_events.tsv")):
        run = re.search(r"run-(\d+)_", k)[1]
        events = list(csv.DictReader(io.StringIO(get_text(f"{SEIZEIT2_BUCKET}/{k}")), delimiter="\t"))
        mods = runs.get(run, set())
        run_rows.append(dict(subject=sub, run=run, modalities="+".join(m for m in MODALITIES if m in mods),
                             all_modalities=all(m in mods for m in MODALITIES),
                             duration_s=float(events[0]["recordingDuration"]) if events else 0.0))
        for e in events:
            if e["eventType"].startswith("sz"):
                seizures.append(dict(subject=sub, run=run, onset=float(e["onset"]),
                                     duration=float(e["duration"]), event_type=e["eventType"],
                                     group=seizure_group(e["eventType"]), vigilance=e["vigilance"],
                                     lateralization=e["lateralization"], localization=e["localization"],
                                     all_modalities=all(m in mods for m in MODALITIES)))
    return run_rows, seizures


def scan_seizeit2(subjects, workers=16):
    """Scan many subjects in parallel. Returns (runs, seizures) as lists of dicts."""
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(scan_subject, subjects))
    return [r for rs, _ in results for r in rs], [s for _, ss in results for s in ss]
