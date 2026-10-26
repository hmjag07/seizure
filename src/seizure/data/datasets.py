"""Dataset adapter: one interface for SeizeIT2 and CHB-MIT (docs/research-log.md, Phase D).

Everything after streaming (windows, features, faults, health check, models, scoring) only needs to know,
per dataset:
  signals          the names of the cleaned signals ("sensors"), in a fixed order
  fs, window_s     sampling rate and look-back window of each signal
  kind(signal)     which classic features it gets: eeg, ecg, emg or motion
  plan             the chunks to stream (results/tables/<plan>.csv)
  setup(sub)       per patient: headers, seizure intervals, normalisation statistics, ...
  stream(...)      raw signals of one chunk as {modality: {label: (samples, fs, unit)}}
  clean(...)       {signal: cleaned array} for one chunk, broken signals left out
  clean_one(...)   one cleaned signal without the whole-chunk broken-signal check (for injected faults)
  channels(...)    the raw (modality, label) channels behind one signal (where faults are injected)
  raw_signal(...)  the raw signal the health check's raw-sample quality features look at
The SeizeIT2 adapter wraps the existing code (data/preprocess.py, features/build.py) unchanged, so its
results stay exactly as reported; check_seizeit2() verifies that the adapter reproduces the cached features.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, sosfilt

from seizure.data import chbmit
from seizure.data import preprocess as pp
from seizure.data.streaming import read_header, read_physical
from seizure.data.windows import WINDOW_S
from seizure.features.build import ROOT, TABLES


class Dataset:
    name = ""
    signals = ()
    fs = {}
    window_s = {}
    plan_file = None
    warmup_skip_s = 30  # seconds of filter warm-up left out of the normalisation statistics

    @property
    def cache(self):
        return ROOT / "cache"

    def folder(self, what):
        """cache/<what>_<dataset> (SeizeIT2 keeps its original folders)."""
        return self.cache / f"{what}_{self.name}"

    def load_plan(self):
        p = pd.read_csv(self.plan_file, dtype={"run": str})
        return p

    def kind(self, signal):
        raise NotImplementedError

    def group(self, subject):
        """The person a recording belongs to, for patient-grouped folds."""
        return subject

    def scale_floor(self, name, ctx, row):
        """Smallest scale of an injected fault on this signal (0: the signal's own spread decides)."""
        return 0.0

    def stats(self, ctx, bg_clean):
        """Median and robust spread per signal over the clean background chunks (filter warm-up skipped)."""
        collected = {}
        for sig in bg_clean:
            for name, x in sig.items():
                collected.setdefault(name, []).append(x[int(self.warmup_skip_s * self.fs[name]):])
        return {name: pp.robust_stats(xs) for name, xs in collected.items()}


class SeizeIT2(Dataset):
    """The wearable dataset, through the original pilot-set code paths."""
    name = "seizeit2"
    signals = pp.SIGNALS
    fs = pp.FS
    window_s = WINDOW_S
    plan_file = TABLES / "fetch_plan.csv"

    def kind(self, signal):
        if signal.startswith("EEG"):
            return "eeg"
        if signal in ("ECG", "EMG"):
            return signal.lower()
        return "motion"

    def load_plan(self):
        p = super().load_plan()
        p["run"] = p["run"].str.zfill(2)
        return p

    def setup(self, sub):
        from seizure.features.build import patient_setup
        return patient_setup(sub)

    def stream(self, sub, row, ctx):
        from seizure.features.build import stream
        return stream(sub, row.run, row.fetch_t0, row.t1, headers=ctx["headers"])

    def clean(self, raw, ctx, row):
        return pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"], ctx["scales"][row.run])

    def clean_one(self, raw, name, ctx, row):
        return pp.clean_one(raw, name, ctx["scales"][row.run])

    def channels(self, raw, name):
        from seizure.faults.build_faulty import sensor_channels
        return sensor_channels(raw, name)

    def raw_signal(self, raw, name):
        from seizure.faults.health import raw_signal
        return raw_signal(raw, name)

    def scale_floor(self, name, ctx, row):
        from seizure.faults.build_faulty import MOTION_POP_FLOOR
        if self.kind(name) != "motion":
            return 0.0
        return MOTION_POP_FLOOR * (ctx["scales"][row.run].get(name.split("_")[1]) or 1.0)


def clean_eeg_60(x, fs):
    """As preprocess.clean_eeg (causal 0.5-40 Hz band-pass), with the notch at the US mains frequency."""
    y = sosfilt(butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos"), x)
    b, a = iirnotch(60, 30, fs)
    return lfilter(b, a, y)


class CHBMIT(Dataset):
    """Scalp EEG, 18 bipolar channels of the double-banana montage, 256 Hz, one EDF per hour."""
    name = "chbmit"
    signals = chbmit.CHANNELS
    fs = {c: 256 for c in chbmit.CHANNELS}
    window_s = {c: 4 for c in chbmit.CHANNELS}
    plan_file = TABLES / "chbmit_fetch_plan.csv"

    def kind(self, signal):
        return "eeg"

    def group(self, subject):
        return chbmit.person(subject)

    def setup(self, sub):
        plan = self.load_plan().query("subject == @sub")
        runs = sorted(plan.run.unique())
        with ThreadPoolExecutor(8) as ex:
            headers = dict(zip(runs, ex.map(lambda r: read_header(chbmit.edf_url(sub, r)), runs)))
        sz = pd.read_csv(TABLES / "chbmit_seizures.csv")
        sz = {r: [(row.onset, row.onset + row.duration) for row in sz[sz.run == r].itertuples()] for r in runs}
        ctx = dict(plan=plan, headers=headers, sz=sz, impd={r: [] for r in runs})
        bg = {row.Index: self.stream(sub, row, ctx) for row in plan.query("kind == 'background'").itertuples()}
        ctx["stats"] = self.stats(ctx, [self.clean(raw, ctx, None) for raw in bg.values()])
        ctx["bg"] = bg
        return ctx

    def stream(self, sub, row, ctx):
        out = read_physical(chbmit.edf_url(sub, row.run), ctx["headers"][row.run], row.fetch_t0, row.t1)
        # Labels upper-cased (a few files write e.g. 'Fz-Cz'); the first copy of a repeated label is kept.
        eeg = {}
        for k, v in out.items():
            eeg.setdefault(k.upper(), v)
        return {"eeg": eeg}

    def clean(self, raw, ctx, row):
        """Every standard channel present in the file and not flat or pinned (preprocess.stuck)."""
        out = {}
        for c in self.signals:
            if c in raw["eeg"] and not pp.stuck(raw["eeg"][c][0]):
                out[c] = clean_eeg_60(raw["eeg"][c][0], self.fs[c])
        return out

    def clean_one(self, raw, name, ctx, row):
        return clean_eeg_60(raw["eeg"][name][0], self.fs[name]) if name in raw["eeg"] else None

    def channels(self, raw, name):
        return [("eeg", name)] if name in raw["eeg"] else []

    def raw_signal(self, raw, name):
        return raw["eeg"][name][0] if name in raw["eeg"] else None


DATASETS = {"seizeit2": SeizeIT2(), "chbmit": CHBMIT()}


def get(name):
    return DATASETS[name]


def check_seizeit2(sub="sub-004", n_chunks=1):
    """The adapter path (features/build_any.py) reproduces cached SeizeIT2 features for one patient's chunks."""
    from seizure.features.build import CACHE, load_params
    from seizure.features.build_any import chunk_features
    ds = SeizeIT2()
    ctx = ds.setup(sub)
    params = load_params()
    cached = pd.read_parquet(CACHE / f"{sub}.parquet")
    worst = 0.0
    for row in list(ctx["plan"].itertuples())[:n_chunks]:
        raw = ctx["bg"].pop(row.Index, None) or ds.stream(sub, row, ctx)
        new = chunk_features(ds, sub, row, raw, ctx, params)
        old = cached[(cached.run == row.run) & cached.t.isin(new.t)].set_index("t")
        new = new.set_index("t")
        cols = [c for c in old.columns if c in new.columns and old[c].dtype.kind == "f"]
        diff = (new.loc[old.index, cols] - old[cols]).abs().max().max()
        worst = max(worst, float(diff))
        print(f"{sub} {row.run} {row.t0:.0f}-{row.t1:.0f}: {len(old)} seconds, {len(cols)} features, "
              f"largest difference {diff:.2e}")
    return worst
