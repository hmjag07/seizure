"""Small 1D CNN baseline on the raw (cleaned, downsampled) signals (docs/research-log.md, Step 5.6 and Phase B2).

    python -m seizure.models.cnn cv      # 5-fold CV on the 90 training patients, clean + dead sensors

Inputs per second t (data/signals.py, normalised with the patient's clean statistics, clipped to +-20):
  EEG1, EEG2   last 4 s at 64 Hz   (2 x 256)
  ECG          last 15 s at 64 Hz  (1 x 960)
  EMG          last 4 s at 128 Hz  (1 x 512)
  motion       last 4 s at 25 Hz   (4 x 100: ACC/GYR neck/chest)
plus 8 "sensor working" flags. One convolutional branch per group, pooled, joined by a small dense head.
A dead sensor's input is zero and its flag 0. Sensor-dropout training: each training window has, with
probability 0.5, 1-6 random sensors made dead (the dropout copy of Step 5.3, done on the fly).

Same rows, labels and 5 patient-grouped folds as the XGBoost models (phase_a.load_all + train.folds);
thresholds chosen on the other folds at <= 8 false alarms per hour (sensor-aware for dead-sensor sets);
judged at 11. Trains on the GPU when available. Writes results/tables/cnn_*.csv.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from seizure.data.preprocess import SIGNALS
from seizure.data.signals import load_patient
from seizure.evaluation.metrics import FastEvents, window_scores
from seizure.features.build import ROOT, TABLES
from seizure.models.phase_a import load_all
from seizure.models.robustness import EEG, combinations
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, SEED, SELECT_BUDGET, folds

SIGNAL_DIRS = [ROOT / "cache" / "signals", ROOT / "cache" / "signals_extra"]
FS = {"EEG1": 64, "EEG2": 64, "ECG": 64, "EMG": 128, "ACC_neck": 25, "ACC_chest": 25, "GYR_neck": 25,
      "GYR_chest": 25}
SECONDS = {"EEG1": 4, "EEG2": 4, "ECG": 15, "EMG": 4, "ACC_neck": 4, "ACC_chest": 4, "GYR_neck": 4,
           "GYR_chest": 4}
GROUPS = {"eeg": ["EEG1", "EEG2"], "ecg": ["ECG"], "emg": ["EMG"],
          "mot": ["ACC_neck", "ACC_chest", "GYR_neck", "GYR_chest"]}
EPOCHS, BATCH, LR = 8, 512, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Windows:
    """All signals of the 90 patients in one float16 array per sensor, and per row the start of its window."""

    def __init__(self, rows=None, dirs=None):
        """rows: optional table (subject, run, t, label, fold) in place of the 90 training patients;
        dirs: optional signal-cache folders (default: the training patients' folders)."""
        if rows is None:
            rows = load_all()[["subject", "run", "t", "label"]]
            rows["fold"] = folds(rows)
        self.rows = rows.reset_index(drop=True)
        rows = self.rows
        bufs = {s: [] for s in SIGNALS}
        size = {s: 0 for s in SIGNALS}
        self.start = np.full((len(rows), len(SIGNALS)), -1, dtype=np.int64)
        for d in (dirs or SIGNAL_DIRS):
            for path in sorted(d.glob("sub-*.npz")):
                sub = path.stem
                meta, data = load_patient(path)
                sel_sub = np.flatnonzero((self.rows.subject == sub).to_numpy())
                if not len(sel_sub):
                    continue
                r = self.rows.iloc[sel_sub]
                for c in meta["chunks"]:
                    in_chunk = sel_sub[((r.run == c["run"]) & (r.t >= c["t0"]) & (r.t <= c["t1"])).to_numpy()]
                    t = self.rows.t.to_numpy()[in_chunk]
                    for s in c["signals"]:
                        med, sd = meta["stats"].get(s, (0.0, 1.0))
                        x = np.clip((data[(c["i"], s)] - med) / sd, -20, 20).astype(np.float16)
                        L = SECONDS[s] * FS[s]
                        stop = np.round((t - c["fetch_t0"]) * FS[s]).astype(np.int64)
                        ok = (stop >= L) & (stop <= len(x))
                        j = SIGNALS.index(s)
                        self.start[in_chunk[ok], j] = size[s] + stop[ok] - L
                        bufs[s].append(x)
                        size[s] += len(x)
        self.sig = {s: torch.from_numpy(np.concatenate(bufs[s]) if bufs[s] else np.zeros(1, np.float16))
                    for s in SIGNALS}
        self.label = self.rows.label.to_numpy().astype(np.float32)
        self.fold = self.rows.fold.to_numpy()

    def batch(self, idx, dead=None):
        """Tensors for rows idx; dead: optional (len(idx), 8) bool."""
        avail = self.start[idx] >= 0
        if dead is not None:
            avail &= ~dead
        out = {}
        for g, sensors in GROUPS.items():
            chans = []
            for s in sensors:
                j, L = SIGNALS.index(s), SECONDS[s] * FS[s]
                st = np.where(avail[:, j], self.start[idx, j], 0)
                w = self.sig[s][torch.from_numpy(st[:, None] + np.arange(L))].float()
                chans.append(w * torch.from_numpy(avail[:, j]).float()[:, None])
            out[g] = torch.stack(chans, 1).to(DEVICE, non_blocking=True)
        out["flags"] = torch.from_numpy(avail.astype(np.float32)).to(DEVICE)
        return out


def branch(c_in, width, k, n_layers):
    layers, c = [], c_in
    for _ in range(n_layers):
        layers += [nn.Conv1d(c, width, k, stride=2, padding=k // 2), nn.BatchNorm1d(width), nn.ReLU()]
        c = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg, self.ecg = branch(2, 64, 7, 4), branch(1, 64, 9, 5)
        self.emg, self.mot = branch(1, 32, 7, 4), branch(4, 32, 5, 3)
        self.head = nn.Sequential(nn.Linear(64 + 64 + 32 + 32 + 8, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, b):
        z = torch.cat([self.eeg(b["eeg"]), self.ecg(b["ecg"]), self.emg(b["emg"]), self.mot(b["mot"]),
                       b["flags"]], 1)
        return self.head(z).squeeze(1)


def random_dead(n, rng, share=0.5):
    k = rng.integers(1, 7, n)
    dead = rng.random((n, len(SIGNALS))).argsort(1).argsort(1) < k[:, None]
    dead[rng.random(n) >= share] = False
    return dead


def train_fold(W, k, rng):
    torch.manual_seed(SEED + k)
    tr = np.flatnonzero(W.fold != k)
    y = W.label[tr]
    net = Net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / (y == 1).sum(), device=DEVICE))
    net.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(tr)
        total = 0.0
        for b in range(len(tr) // BATCH):
            idx = order[b * BATCH:(b + 1) * BATCH]
            batch = W.batch(idx, random_dead(len(idx), rng))
            target = torch.from_numpy(W.label[idx]).to(DEVICE)
            loss = loss_fn(net(batch), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  fold {k} epoch {epoch + 1}/{EPOCHS}: loss {total / (len(tr) // BATCH):.4f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def predict(W, net, idx, dead=None):
    out = np.empty(len(idx), dtype=np.float32)
    for b in range(0, len(idx), 4096):
        sel = idx[b:b + 4096]
        out[b:b + 4096] = torch.sigmoid(net(W.batch(sel, None if dead is None else dead[b:b + 4096]))).cpu().numpy()
    return out


def cv():
    t_start = time.time()
    W = Windows()
    print(f"{W.rows.subject.nunique()} patients, {len(W.rows)} rows, device {DEVICE}, "
          f"windows missing per sensor {dict(zip(SIGNALS, (W.start < 0).mean(0).round(3)))}, "
          f"{time.time() - t_start:.0f} s", flush=True)
    rng = np.random.default_rng(SEED)
    nets = []
    for k in range(N_FOLDS):
        nets.append(train_fold(W, k, rng))
        torch.save(nets[-1].state_dict(), ROOT / "cache" / "models" / f"cnn_fold{k}.pt")
        print(f"fold {k} trained, {time.time() - t_start:.0f} s", flush=True)

    rows_df = W.rows[["subject", "run", "t", "label"]]
    events = {k: FastEvents(rows_df[W.fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(rows_df[W.fold != k]) for k in range(N_FOLDS)}
    res = []
    for ci, combo in enumerate(combinations()):
        prob = np.empty(len(W.rows), dtype=np.float32)
        for k, net in enumerate(nets):
            idx = np.flatnonzero(W.fold == k)
            dead = np.zeros((len(idx), len(SIGNALS)), bool)
            for s in combo:
                dead[:, SIGNALS.index(s)] = True
            prob[idx] = predict(W, net, idx, dead if combo else None)
        det = n = fa = 0
        hours, fa_fold = 0.0, []
        for k in range(N_FOLDS):
            thr = best_threshold(other[k], prob[W.fold != k], budget=SELECT_BUDGET)
            d, nk, f, h = events[k].score(prob[W.fold == k] >= thr, *ALARM)
            det, n, fa, hours = det + d, n + nk, fa + f, hours + h
            fa_fold.append(f / h)
        res.append(dict(model="cnn", dead=len(combo), sensors_dead=" ".join(combo) or "none",
                        eeg_alive=not EEG <= set(combo), sensitivity=det / n, fa_per_h=fa / hours,
                        folds_over_budget=sum(x > FA_BUDGET for x in fa_fold),
                        **(window_scores(W.label, prob) if not combo else {})))
        if not combo:
            pd.DataFrame({"subject": rows_df.subject, "run": rows_df.run, "t": rows_df.t, "label": W.label,
                          "fold": W.fold, "cnn": prob}).to_parquet(ROOT / "cache" / "predictions" / "cnn_oof.parquet")
            print(res[-1], flush=True)
    out = pd.DataFrame(res).round(4)
    out.to_csv(TABLES / "cnn_dead_sensors.csv", index=False)
    a = out[out.eeg_alive]
    print(a.groupby("dead")[["sensitivity", "fa_per_h"]].mean().round(3).to_string())
    print(f"done in {time.time() - t_start:.0f} s")




# ---------------------------------------------------------------------------------------------------------
# CNN v2: the v1 encoder per second, plus a GRU over the last 60 seconds of encodings (temporal context,
# matching the 10 s / 60 s context features of the XGBoost models).
# ---------------------------------------------------------------------------------------------------------
HISTORY = 60
V2_EPOCHS, V2_BATCH = 4, 1024


def encode(net, b):
    """The v1 network without its head: pooled branch outputs and sensor flags (200 numbers)."""
    return torch.cat([net.eeg(b["eeg"]), net.ecg(b["ecg"]), net.emg(b["emg"]), net.mot(b["mot"]), b["flags"]], 1)


@torch.no_grad()
def embed(W, net, idx, dead=None):
    """Encodings for rows idx (float16, on the GPU)."""
    out = torch.empty((len(idx), 200), dtype=torch.float16, device=DEVICE)
    for b in range(0, len(idx), 4096):
        out[b:b + 4096] = encode(net, W.batch(idx[b:b + 4096], None if dead is None else dead[b:b + 4096])).half()
    return out


class Temporal(nn.Module):
    def __init__(self, d=200, h=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gru = nn.GRU(d, h, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, seq):
        out, _ = self.gru(self.norm(seq))
        return self.head(out[:, -1]).squeeze(1)


def history_index(W):
    """(rows, HISTORY) row positions of the last HISTORY seconds of each row's chunk, padded with the
    chunk's first row; rows are sorted by subject, run, t."""
    r = W.rows
    new_seg = ((r.t.diff() != 1) | (r.subject != r.subject.shift()) | (r.run != r.run.shift())).to_numpy()
    seg_start = np.maximum.accumulate(np.where(new_seg, np.arange(len(r)), 0))
    idx = np.arange(len(r))[:, None] - np.arange(HISTORY - 1, -1, -1)[None, :]
    return torch.from_numpy(np.maximum(idx, seg_start[:, None]))


def segment_dead(W, rng):
    """One random dead set (1-6 sensors) per chunk, held for the whole chunk; half the chunks have none."""
    r = W.rows
    seg = (((r.t.diff() != 1) | (r.subject != r.subject.shift()) | (r.run != r.run.shift())).cumsum()).to_numpy()
    per_seg = random_dead(seg.max() + 1, rng)
    return per_seg[seg]


def v2():
    t_start = time.time()
    W = Windows()
    hist = history_index(W).to(DEVICE)
    rng = np.random.default_rng(SEED)
    dead_drop = segment_dead(W, rng)
    print(f"{len(W.rows)} rows, history {HISTORY} s, {time.time() - t_start:.0f} s", flush=True)
    rows_df = W.rows[["subject", "run", "t", "label"]]
    events = {k: FastEvents(rows_df[W.fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(rows_df[W.fold != k]) for k in range(N_FOLDS)}
    everyone = np.arange(len(W.rows))
    encoders, temporals = [], []
    for k in range(N_FOLDS):
        enc = Net().to(DEVICE)
        enc.load_state_dict(torch.load(ROOT / "cache" / "models" / f"cnn_fold{k}.pt"))
        enc.eval()
        E_clean, E_drop = embed(W, enc, everyone), embed(W, enc, everyone, dead_drop)
        tr = np.flatnonzero(W.fold != k)
        y = torch.from_numpy(W.label).to(DEVICE)
        torch.manual_seed(SEED + k)
        tmp = Temporal().to(DEVICE)
        opt = torch.optim.AdamW(tmp.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = torch.tensor((W.label[tr] == 0).sum() / (W.label[tr] == 1).sum(), device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        tmp.train()
        for epoch in range(V2_EPOCHS):
            order = rng.permutation(np.concatenate([tr, tr + len(W.rows)]))  # clean and dropout versions
            total, nb = 0.0, len(order) // V2_BATCH
            for b in range(nb):
                sel = order[b * V2_BATCH:(b + 1) * V2_BATCH]
                is_drop = torch.from_numpy(sel >= len(W.rows)).to(DEVICE)
                rows = torch.from_numpy(sel % len(W.rows)).to(DEVICE)
                h = hist[rows]
                seq = torch.where(is_drop[:, None, None], E_drop[h], E_clean[h]).float()
                loss = loss_fn(tmp(seq), y[rows])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  fold {k} epoch {epoch + 1}/{V2_EPOCHS}: loss {total / nb:.4f}", flush=True)
        tmp.eval()
        encoders.append(enc)
        temporals.append(tmp)
        torch.save(tmp.state_dict(), ROOT / "cache" / "models" / f"cnn2_fold{k}.pt")
        del E_clean, E_drop
        torch.cuda.empty_cache()
        print(f"fold {k} trained, {time.time() - t_start:.0f} s", flush=True)

    res = []
    for combo in combinations():
        prob = np.empty(len(W.rows), dtype=np.float32)
        for k in range(N_FOLDS):
            idx = np.flatnonzero(W.fold == k)
            dead = None
            if combo:
                dead = np.zeros((len(idx), len(SIGNALS)), bool)
                for s in combo:
                    dead[:, SIGNALS.index(s)] = True
            E = torch.zeros((len(W.rows), 200), dtype=torch.float16, device=DEVICE)
            E[torch.from_numpy(idx).to(DEVICE)] = embed(W, encoders[k], idx, dead)
            with torch.no_grad():
                for b in range(0, len(idx), 8192):
                    rows = torch.from_numpy(idx[b:b + 8192]).to(DEVICE)
                    prob[idx[b:b + 8192]] = torch.sigmoid(temporals[k](E[hist[rows]].float())).cpu().numpy()
            del E
        det = n = fa = 0
        hours, fa_fold = 0.0, []
        for k in range(N_FOLDS):
            thr = best_threshold(other[k], prob[W.fold != k], budget=SELECT_BUDGET)
            d, nk, f, h = events[k].score(prob[W.fold == k] >= thr, *ALARM)
            det, n, fa, hours = det + d, n + nk, fa + f, hours + h
            fa_fold.append(f / h)
        res.append(dict(model="cnn+gru", dead=len(combo), sensors_dead=" ".join(combo) or "none",
                        eeg_alive=not EEG <= set(combo), sensitivity=det / n, fa_per_h=fa / hours,
                        folds_over_budget=sum(x > FA_BUDGET for x in fa_fold),
                        **(window_scores(W.label, prob) if not combo else {})))
        if not combo:
            pd.DataFrame({"subject": rows_df.subject, "run": rows_df.run, "t": rows_df.t, "label": W.label,
                          "fold": W.fold, "cnn2": prob}).to_parquet(ROOT / "cache" / "predictions" / "cnn2_oof.parquet")
            print(res[-1], flush=True)
    out = pd.DataFrame(res).round(4)
    out.to_csv(TABLES / "cnn2_dead_sensors.csv", index=False)
    a = out[out.eeg_alive]
    print(a.groupby("dead")[["sensitivity", "fa_per_h"]].mean().round(3).to_string())
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    {"cv": cv, "v2": v2}[sys.argv[1]]()
