"""Window-level fault labels ("fix D"; docs/pipeline.md, Step 3).

    python -m seizure.faults.window_labels

The first health-check labels came from the fault's overall severity, so a window between two pops,
or between two flat bursts, was labelled "failed" even when its data was clean. Here each faulty
window is labelled by what the fault did inside that window:

dropout   share of the window's samples that were flattened:  >= 0.5 failed, > 0 degraded, else healthy
pop       largest added step inside the window, in units of the signal's own spread:
          >= POP_FAILED failed, >= POP_DEGRADED degraded, else healthy (no pop, or it has faded)
decline   effective severity at the window's end, as before:  >= 0.6 failed, > 0 degraded

No features are recomputed. The faults were generated from seeded random numbers whose use does not
depend on the signal's values, so replaying inject() on dummy arrays of the same length, with the same
seed, reproduces exactly where every pop and flat burst fell.

Output: cache/faults/<sub>_wlabels.parquet with run, t, fault, severity, sensor, wclass (0/1/2) and
wdamage (flat share for dropout, largest step for pops, severity for decline).
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from seizure.data import preprocess as pp
from seizure.data.windows import WINDOW_S
from seizure.faults.build_faulty import OUT, seed
from seizure.faults.inject import inject
from seizure.features.build import load_plan

FLAT_FAILED = 0.5
POP_FAILED, POP_DEGRADED = 5.0, 1.0
N_CHANNELS = {s: (3 if s.startswith(("ACC", "GYR")) else 1) for s in pp.SIGNALS}


def replay(fault, s, n, fs, start, rng, channels):
    """Per-sample damage of one sensor: flattened (dropout), |added step| in spreads (pop).

    Calls inject() on arrays of ones with scale 1, once per channel in the same order and with the
    same generator as build_faulty.damage, so the random draws line up exactly."""
    flat, step = np.zeros(n, bool), np.zeros(n)
    for _ in range(channels):
        y, _ = inject(fault, np.ones(n), fs, start, s, rng, scale=1.0)
        if fault == "dropout":
            flat |= y == 0
        elif fault == "pop":
            step = np.maximum(step, np.abs(y - 1.0))
    return flat, step


def label_chunk(sub, row, rows_chunk):
    out = []
    for (fault, s, sensor), g in rows_chunk.groupby(["fault", "severity", "sensor"]):
        fs = pp.FS[sensor]
        n = int(round((row.t1 - row.fetch_t0) * fs))
        onset = float(g.onset.iloc[0])
        start = int(round((onset - row.fetch_t0) * fs))
        t = g.t.to_numpy()
        stop = np.minimum(np.round((t - row.fetch_t0) * fs).astype(int), n)
        L = int(WINDOW_S[sensor] * fs)
        if fault == "decline":
            dmg = g.eff.to_numpy(dtype=float)
            cls = np.where(dmg >= 0.6, 2, np.where(dmg > 0, 1, 0))
        else:
            rng = np.random.default_rng(seed(sub, row.run, row.t0, fault, s, sensor))
            flat, step = replay(fault, s, n, fs, start, rng, N_CHANNELS[sensor])
            if fault == "dropout":
                cs = np.concatenate([[0], np.cumsum(flat)])
                dmg = (cs[stop] - cs[np.maximum(stop - L, 0)]) / L
                cls = np.where(dmg >= FLAT_FAILED, 2, np.where(dmg > 0, 1, 0))
            else:
                wmax = maximum_filter1d(step, size=L, origin=(L - 1) // 2)  # max over the L samples ending here
                dmg = wmax[np.maximum(stop - 1, 0)]
                cls = np.where(dmg >= POP_FAILED, 2, np.where(dmg >= POP_DEGRADED, 1, 0))
        out.append(pd.DataFrame({"run": row.run, "t": t, "fault": fault, "severity": s, "sensor": sensor,
                                 "wclass": cls.astype("int8"), "wdamage": dmg.astype("float32")}))
    return out


def label_patient(sub):
    t0 = time.time()
    plan = load_plan().query("subject == @sub")
    f = pd.read_parquet(OUT / f"{sub}_faulty.parquet",
                        columns=["run", "t", "fault", "severity", "sensor", "onset", "eff"])
    out = []
    for row in plan.itertuples():
        mine = f[(f.run == row.run) & (f.onset >= row.t0) & (f.onset <= row.t1)]
        if not mine.empty:
            out += label_chunk(sub, row, mine)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT / f"{sub}_wlabels.parquet")
    return sub, len(res), round(time.time() - t0)


def main():
    subs = sys.argv[1:] or sorted(load_plan().subject.unique())
    with ProcessPoolExecutor(6) as ex:
        for sub, n, secs in ex.map(label_patient, subs):
            print(f"{sub}: {n} rows, {secs} s", flush=True)


if __name__ == "__main__":
    main()
