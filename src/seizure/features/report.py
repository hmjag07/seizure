"""Summarise the cached feature tables: counts, missing values, single-feature AUROC, TDA speed.

    python -m seizure.features.report

Writes results/tables/features_counts.csv, features_missing.csv, features_auroc.csv
and features_speed.csv.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from seizure.data import preprocess as pp
from seizure.data.windows import WINDOW_S, fits, window_ends, windows
from seizure.features import classic, tda_embedding as tda
from seizure.features.build import CACHE, TABLES, load_params, load_plan, stream

META = ["subject", "run", "kind", "t", "label", "impd"]


def signal_of(feature):
    """'TDA_ACC_neck_H1_max' -> 'ACC_neck', 'EEG1_delta' -> 'EEG1'."""
    f = feature.removeprefix("TDA_")
    return next(s for s in pp.SIGNALS if f.startswith(s + "_"))


def load():
    return pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("sub-*.parquet"))], ignore_index=True)


def per_patient_z(df, cols):
    """Robust z-score of each feature within each patient, using that patient's background windows."""
    out = df[cols].copy()
    for sub, idx in df.groupby("subject").groups.items():
        bg = df.loc[idx][df.loc[idx, "label"] == 0][cols]
        med = bg.median()
        mad = (bg - med).abs().median() * 1.4826
        out.loc[idx] = (df.loc[idx, cols] - med) / mad.replace(0, np.nan)
    return out


def auroc_table(df, cols):
    z = per_patient_z(df, cols)
    rows = []
    for c in cols:
        ok = z[c].notna() & np.isfinite(z[c])
        if df.loc[ok, "label"].nunique() == 2:
            a = roc_auc_score(df.loc[ok, "label"], z.loc[ok, c])
            rows.append(dict(feature=c, family="TDA-A" if c.startswith("TDA_") else "classic",
                             signal=signal_of(c),
                             auroc=round(a, 3), separation=round(abs(a - 0.5), 3), n=int(ok.sum())))
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def speed(n_windows=300):
    """Time per window for TDA-A and classic features, on one real background chunk."""
    params = load_params()
    row = load_plan().query("kind == 'background' and subject == 'sub-004'").iloc[0]
    raw = stream(row.subject, row.run, row.fetch_t0, row.t1)
    sig = pp.clean_chunk(raw["eeg"], raw["ecg"], raw["emg"], raw["mov"],
                         {d: pp.gravity_scale([pp.acc_xyz(raw["mov"], d)]) for d in pp.DEVICES})
    ends = window_ends(row.fetch_t0, row.t0, row.t1)
    ok = np.all([fits(x, pp.FS[k], WINDOW_S[k], row.fetch_t0, ends) for k, x in sig.items()], axis=0)
    ends = ends[ok][:n_windows]
    rows = []
    for name, x in sig.items():
        w = windows(x / x.std(), pp.FS[name], WINDOW_S[name], row.fetch_t0, ends)
        t0 = time.perf_counter()
        tda.tda_features(w, *params[name], name=name)
        rows.append(dict(signal=name, points=min(tda.N_POINTS, w.shape[1] - (params[name][0] - 1) * params[name][1]),
                         tda_ms_per_window=round(1000 * (time.perf_counter() - t0) / len(w), 1)))
    t0 = time.perf_counter()
    for name, x in sig.items():
        w = windows(x, pp.FS[name], WINDOW_S[name], row.fetch_t0, ends)
        if name.startswith("EEG"):
            classic.eeg_features(w, pp.FS[name], name)
        elif name == "EMG":
            classic.emg_features(w)
        elif name != "ECG":
            classic.motion_features(w, name)
    rows.append(dict(signal="all classic", points=None,
                     tda_ms_per_window=round(1000 * (time.perf_counter() - t0) / len(ends), 2)))
    return pd.DataFrame(rows)


def main():
    df = load()
    feats = [c for c in df.columns if c not in META]
    counts = (df.groupby("kind").agg(windows=("label", "size"), seizure_windows=("label", "sum"),
                                     impd_windows=("impd", "sum")).reset_index())
    counts.loc[len(counts)] = ["total", len(df), int(df.label.sum()), int(df.impd.sum())]
    missing = df[feats].isna().mean().rename("missing").reset_index().rename(columns={"index": "feature"})
    missing["signal"] = [signal_of(f) for f in missing.feature]
    missing = missing.groupby("signal").missing.max().round(4).reset_index()
    au = auroc_table(df, feats)
    sp = speed()
    for name, t in (("counts", counts), ("missing", missing), ("auroc", au), ("speed", sp)):
        t.to_csv(TABLES / f"features_{name}.csv", index=False)
    print(f"patients {df.subject.nunique()}, features {len(feats)}")
    print(counts.to_string(index=False), "\n")
    print(missing.to_string(index=False), "\n")
    print(au.head(15).to_string(index=False), "\n")
    print(au.groupby(["family", "signal"]).separation.max().unstack(0).round(3).to_string(), "\n")
    print(sp.to_string(index=False))


if __name__ == "__main__":
    main()
