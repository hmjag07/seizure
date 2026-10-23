"""Local, downsampled copy of the cleaned signals for TDA-B and the CNN (docs/data-access.md, section 8).

    SEIZURE_SET=pilot|extra|heldout python -m seizure.data.signals

For every chunk in the set's fetch plan: stream, clean exactly as for the features (Step 1.4, including the
gravity calibration and broken-sensor checks), and downsample with causal filters:
EEG1, EEG2, ECG to 64 Hz; EMG to 128 Hz (its content is above 20 Hz); motion magnitudes stay at 25 Hz.
Each patient's normalisation statistics (median and robust spread of the clean background, Step 1.4) are
stored too. One file per patient: cache/signals[_set]/<subject>.npz, with arrays named
"<chunk index>|<signal>" plus a "chunks" table (run, kind, t0, t1, fetch_t0, fs per signal).
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import decimate

from seizure.data import preprocess as pp
from seizure.features.build import CACHE, SUFFIX, load_plan, patient_setup, stream

OUT = CACHE.parent / f"signals{SUFFIX}"
TARGET_FS = {"EEG1": 64, "EEG2": 64, "ECG": 64, "EMG": 128,
             "ACC_neck": 25, "ACC_chest": 25, "GYR_neck": 25, "GYR_chest": 25}


def downsample(x, fs, target):
    """Causal anti-alias filter (IIR, forward only) and decimation."""
    q = int(round(fs / target))
    return x if q == 1 else decimate(x, q, ftype="iir", zero_phase=False)


def save_patient(sub):
    t0 = time.time()
    ctx = patient_setup(sub)
    arrays, chunks = {}, []
    for i, row in enumerate(ctx["plan"].itertuples()):
        raw = ctx["bg"].pop(row.Index, None) or stream(sub, row.run, row.fetch_t0, row.t1, headers=ctx["headers"])
        clean = pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"], ctx["scales"][row.run])
        for name, x in clean.items():
            arrays[f"{i}|{name}"] = downsample(x, pp.FS[name], TARGET_FS[name]).astype("float32")
        chunks.append(dict(i=i, run=row.run, kind=row.kind, t0=row.t0, t1=row.t1, fetch_t0=row.fetch_t0,
                           signals=sorted(clean)))
    stats = {name: list(map(float, v)) for name, v in ctx["stats"].items()}
    meta = dict(chunks=chunks, fs=TARGET_FS, stats=stats)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{sub}.npz", meta=np.array(json.dumps(meta)), **arrays)
    return sub, len(chunks), round(sum(a.nbytes for a in arrays.values()) / 1e6, 1), round(time.time() - t0)


def load_patient(path):
    """(meta, {(chunk index, signal): array}) for one saved patient."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    data = {(int(k.split("|")[0]), k.split("|")[1]): z[k] for k in z.files if k != "meta"}
    return meta, data


def main():
    subs = sys.argv[1:] or sorted(load_plan().subject.unique())
    subs = [s for s in subs if not (OUT / f"{s}.npz").exists()]
    print(f"{len(subs)} patients to save into {OUT}", flush=True)
    with ProcessPoolExecutor(6) as ex:
        for sub, n, mb, secs in ex.map(save_patient, subs):
            print(f"{sub}: {n} chunks, {mb} MB, {secs} s", flush=True)


if __name__ == "__main__":
    main()
