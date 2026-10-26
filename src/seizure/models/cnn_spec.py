"""Spectrogram CNN + GRU, trained end to end, and a hybrid XGBoost (docs/research-log.md, Phase B3).

    python -m seizure.models.cnn_spec cv       # 5-fold CV on the 90 training patients, clean + dead sensors
    python -m seizure.models.cnn_spec hybrid   # XGBoost classic + TDA + the CNN's score, vs without

Network (one output per second t):
  1. Each sensor's last 4 s (ECG 15 s), from the signal cache (data/signals.py), becomes a log-power
     spectrogram, |STFT|^2 with a Hann window:
        EEG1, EEG2  64 Hz, 1 s window, 0.25 s step   -> 33 frequencies (0-32 Hz) x 13 time steps
        ECG         64 Hz, 2 s window, 1 s step      -> 65 x 14 (over 15 s)
        EMG        128 Hz, 0.5 s window, 0.25 s step -> 33 x 15 (0-64 Hz)
        motion      25 Hz, 1 s window, 0.24 s step   -> 13 x 13 (4 channels: ACC/GYR neck/chest)
     A dead sensor's spectrogram is zero and its "working" flag 0.
  2. A small 2-D CNN per sensor group (EEG, ECG, EMG, motion) turns each spectrogram into a few numbers;
     with the 8 flags this is a 128-number summary of second t.
  3. A GRU reads the summaries of the last HISTORY seconds; its last state gives the seizure probability.
Trained end to end (the earlier CNN v2 froze its encoder): sequences of HISTORY seconds, loss on the last
LOSS_STEPS seconds of each sequence, balanced sampling (half the sequences end on a seizure second), and
sensor dropout (half the sequences lose a random 1-6 sensors for the whole sequence).

Hybrid: the CNN's score (logit) per second is added as one feature to XGBoost classic + TDA. Stacking without
leaks: for outer fold k, the four training folds are split into two halves; a CNN trained on one half scores
the other, so no training row's score comes from a network that saw it, and the test fold gets the mean of
the two networks' scores. The dropout copy for XGBoost uses one random dead set per chunk (1-6 sensors, half
the chunks), so the CNN score can be computed with the same sensors dead; the XGBoost-only comparison uses
the same copy.

Same rows, labels and patient-grouped folds as the XGBoost models; thresholds chosen on the other folds at
<= 8 false alarms per hour, per set of dead sensors (sensor-aware); judged at 11. Writes
results/tables/cnn_spec_*.csv and hybrid_*.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier

from seizure.data.preprocess import SIGNALS
from seizure.evaluation.metrics import FastEvents, window_scores
from seizure.features.build import ROOT, TABLES
from seizure.models.cnn import FS, GROUPS, SECONDS, Windows, random_dead
from seizure.models.robustness import EEG, combinations
from seizure.models.sensor_aware import best_threshold
from seizure.models.train import ALARM, FA_BUDGET, N_FOLDS, PARAMS, SEED, SELECT_BUDGET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STFT = {"EEG1": (64, 16), "EEG2": (64, 16), "ECG": (128, 64), "EMG": (64, 16 * 2), "ACC_neck": (25, 6),
        "ACC_chest": (25, 6), "GYR_neck": (25, 6), "GYR_chest": (25, 6)}  # (window, step) in samples
HISTORY, LOSS_STEPS = 90, 30
SEQ_BATCH, STEPS, LR = 24, 4000, 1e-3
MODELS = ROOT / "cache" / "models"
# CNN_TAG=f: fault training (Phase B5). Models and tables get the tag; the XGBoost reference has no CNN, so it is
# shared between the two runs.
TAG = os.environ.get("CNN_TAG", "")
FAULT_TRAIN = TAG == "f"
FAULT_SHARE = 0.4  # share of training sequences with an injected fault (on top of sensor dropout)
FAULT_MAX_SENSORS = 5
RAMP_S, POP_RATE, POP_SIZE, POP_DECAY, BURST = 300, 6, 20, (0.3, 2.0), (0.5, 5.0)  # as faults/inject.py
MAX_EVENTS = 48


def sample_faults(B, rng, avail):
    """Per training sequence and sensor: fault type (0 none, 1 pop, 2 decline, 3 dropout), severity in
    [0.25, 1], onset as a fraction of the sequence span, and random draws for pops and flat bursts.
    The fault is applied in absolute buffer positions, so every overlapping window of a sequence sees the
    same fault (faults/inject.py rules; scale 1, since the cached signals are normalised per patient)."""
    kind = np.zeros((B, len(SIGNALS)), np.int64)
    for b in np.flatnonzero(rng.random(B) < FAULT_SHARE):
        ok = np.flatnonzero(avail[b])
        if len(ok) == 0:
            continue
        chosen = rng.choice(ok, min(len(ok), rng.integers(1, FAULT_MAX_SENSORS + 1)), replace=False)
        kind[b, chosen] = rng.integers(1, 4)
    return dict(kind=kind, sev=rng.uniform(0.25, 1.0, (B, len(SIGNALS))),
                onset=rng.uniform(0.0, 1.0, (B, len(SIGNALS))),  # 0 = start of the sequence's span, 1 = its end
                ev_pos=rng.random((B, len(SIGNALS), MAX_EVENTS)), ev_a=rng.random((B, len(SIGNALS), MAX_EVENTS)),
                ev_b=rng.random((B, len(SIGNALS), MAX_EVENTS)), sign=rng.choice([-1.0, 1.0], (B, len(SIGNALS), MAX_EVENTS)))


def apply_fault(w, pos, j, fs, f, seq, span0, span1):
    """Damage windows w (n, L) of sensor j in place-free fashion. pos: (n, L) absolute buffer positions;
    seq: (n,) sequence of each window; span0/span1: (B,) first and last buffer position of each sequence."""
    sel = torch.nonzero(f["kind"][seq, j] > 0).squeeze(1)
    if len(sel) == 0:
        return w
    full, w, pos, seq = w, w[sel], pos[sel], seq[sel]  # only the windows with a fault on this sensor
    kind = f["kind"][seq, j]
    sev = f["sev"][seq, j][:, None]
    # onset anywhere from RAMP_S before the sequence to its end, so a decline can be at any stage of its ramp
    s0, s1 = span0[seq].float(), span1[seq].float()
    onset = (s1 - f["onset"][seq, j] * (s1 - s0 + RAMP_S * fs))[:, None]
    after = pos.float() >= onset
    out = w.clone()
    # decline: shrink to (1 - level) and add noise of size level; level ramps to sev over RAMP_S
    lvl = sev * torch.clamp((pos.float() - onset) / (RAMP_S * fs), 0, 1) * after
    dec = w * (1 - lvl) + torch.randn_like(w) * lvl
    out = torch.where((kind == 2)[:, None], dec, out)
    first = torch.maximum(onset[:, 0], s0)  # pops and bursts are only placed inside the sequence's span
    after_len = s1 - first
    p = pos.float()
    # pops: steps of up to POP_SIZE x sev, decaying over 0.3-2 s, at about sev x 6 per minute after the onset
    n_pop = torch.clamp(torch.ceil(sev[:, 0] * POP_RATE * after_len / fs / 60), 0, MAX_EVENTS)
    # dropout: flat bursts of 0.5-5 s (mean 2.75 s) covering about sev of the time after the onset
    n_burst = torch.clamp(torch.ceil(sev[:, 0] * after_len / (2.75 * fs)), 0, MAX_EVENTS)
    add = torch.zeros_like(w)
    flat = (sev >= 1.0) & after  # sev = 1: flat from the onset (a dead sensor)
    n_events = int(torch.max(torch.where(kind == 1, n_pop, torch.where(kind == 3, n_burst, torch.zeros_like(n_pop)))))
    for e in range(n_events):  # one event at a time keeps memory at (windows x samples)
        at = first + f["ev_pos"][seq, j, e] * after_len
        d = p - at[:, None]
        is_pop = ((kind == 1) & (e < n_pop))[:, None]
        if is_pop.any():
            size = sev[:, 0] * POP_SIZE * f["sign"][seq, j, e] * (0.5 + 0.5 * f["ev_a"][seq, j, e])
            tau = (POP_DECAY[0] + (POP_DECAY[1] - POP_DECAY[0]) * f["ev_b"][seq, j, e]) * fs
            add = add + torch.where(is_pop & (d >= 0), size[:, None] * torch.exp(-torch.clamp(d, min=0) / tau[:, None]),
                                    torch.zeros_like(w))
        is_burst = ((kind == 3) & (e < n_burst))[:, None]
        if is_burst.any():
            b_len = (BURST[0] + (BURST[1] - BURST[0]) * f["ev_a"][seq, j, e]) * fs
            flat = flat | (is_burst & (d >= 0) & (d < b_len[:, None]))
    out = torch.where((kind == 1)[:, None], w + add, out)
    out = torch.where((kind == 3)[:, None] & flat & after, torch.zeros_like(w), out)
    full = full.clone()
    full[sel] = torch.clamp(out, -20, 20)
    return full


class GpuWindows:
    """The CNN v1 window table (cnn.Windows), with the signals and window starts on the GPU."""

    def __init__(self, rows=None, dirs=None):
        W = Windows(rows, dirs)
        self.rows, self.label, self.fold = W.rows, W.label, W.fold
        self.sig = {s: W.sig[s].to(DEVICE) for s in SIGNALS}
        self.start = torch.from_numpy(W.start).to(DEVICE)
        r = self.rows
        new_seg = ((r.t.diff() != 1) | (r.subject != r.subject.shift()) | (r.run != r.run.shift())).to_numpy()
        self.seg = np.cumsum(new_seg) - 1
        seg_start = np.maximum.accumulate(np.where(new_seg, np.arange(len(r)), 0))
        idx = np.arange(len(r))[:, None] - np.arange(HISTORY - 1, -1, -1)[None, :]
        self.hist = torch.from_numpy(np.maximum(idx, seg_start[:, None])).to(DEVICE)  # (rows, HISTORY)

    def spectra(self, rows, dead, faults=None, seq=None, B=None):
        """rows: (n,) GPU long tensor; dead: (n, 8) GPU bool. {group: (n, channels, freq, time)}, flags.
        faults: optional sample_faults() output (as GPU tensors) with seq, the sequence of each row (training)."""
        avail = (self.start[rows] >= 0) & ~dead
        out = {}
        for g, sensors in GROUPS.items():
            chans = []
            for s in sensors:
                j, L = SIGNALS.index(s), SECONDS[s] * FS[s]
                st = torch.where(avail[:, j], self.start[rows, j], torch.zeros_like(rows))
                pos = st[:, None] + torch.arange(L, device=DEVICE)
                w = self.sig[s][pos].float()
                if faults is not None:
                    # span of each sequence in this sensor's buffer: first window start to last window end
                    # (only windows where the sensor works; others read position 0 and are masked out anyway)
                    big = torch.full_like(st, 2 ** 62)
                    s0 = torch.full((B,), 2 ** 62, dtype=torch.long, device=DEVICE).scatter_reduce(
                        0, seq, torch.where(avail[:, j], st, big), "amin")
                    s1 = torch.zeros(B, dtype=torch.long, device=DEVICE).scatter_reduce(
                        0, seq, torch.where(avail[:, j], st + L, torch.zeros_like(st)), "amax")
                    s0 = torch.minimum(s0, s1)  # sequences where the sensor never works: empty span
                    w = apply_fault(w, pos, j, FS[s], faults, seq, s0, s1)
                n_fft, hop = STFT[s]
                spec = torch.stft(w, n_fft, hop, window=torch.hann_window(n_fft, device=DEVICE), center=False,
                                  return_complex=True).abs().pow(2)
                chans.append(torch.log(spec + 1e-4) * avail[:, j, None, None].float())
            out[g] = torch.stack(chans, 1)
        return out, avail.float()


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2, ceil_mode=True))


def encoder(c_in, widths):
    layers, c = [nn.BatchNorm2d(c_in)], c_in
    for w in widths:
        layers.append(block(c, w))
        c = w
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class SpecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleDict({"eeg": encoder(2, (16, 32, 64)), "ecg": encoder(1, (16, 32, 32)),
                                  "emg": encoder(1, (16, 32, 32)), "mot": encoder(4, (16, 32, 32))})
        self.proj = nn.Sequential(nn.Linear(64 + 32 + 32 + 32 + 8, 128), nn.ReLU())
        self.norm = nn.LayerNorm(128)
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def summarise(self, spectra, flags):
        return self.proj(torch.cat([self.enc[g](spectra[g]) for g in GROUPS] + [flags], 1))

    def sequence(self, z):
        out, _ = self.gru(self.norm(z))
        return self.head(out).squeeze(-1)  # (batch, time) logits


def train_net(W, train_rows, seed, steps=STEPS, tag=""):
    """End-to-end training on the given rows (numpy positions)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SpecNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    pos = train_rows[W.label[train_rows] == 1]
    label = torch.from_numpy(W.label).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    net.train()
    total = 0.0
    for step in range(steps):
        half = SEQ_BATCH // 2
        ends = np.concatenate([rng.choice(pos, half), rng.choice(train_rows, SEQ_BATCH - half)])
        rows = W.hist[torch.from_numpy(ends).to(DEVICE)]  # (B, HISTORY)
        dead_np = random_dead(SEQ_BATCH, rng)
        dead = torch.from_numpy(dead_np).to(DEVICE)  # one dead set per sequence
        flat = rows.reshape(-1)
        kw = {}
        if FAULT_TRAIN:
            f = sample_faults(SEQ_BATCH, rng, ~dead_np)
            kw = dict(faults={k: torch.from_numpy(v).to(DEVICE) if k == "kind" else
                              torch.from_numpy(v).float().to(DEVICE) for k, v in f.items()},
                      seq=torch.arange(SEQ_BATCH, device=DEVICE).repeat_interleave(HISTORY), B=SEQ_BATCH)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            spectra, flags = W.spectra(flat, dead.repeat_interleave(HISTORY, 0), **kw)
            z = net.summarise(spectra, flags).reshape(SEQ_BATCH, HISTORY, -1)
            logits = net.sequence(z.float())[:, -LOSS_STEPS:]
        loss = loss_fn(logits.float(), label[rows[:, -LOSS_STEPS:]])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
        if (step + 1) % 500 == 0:
            print(f"  {tag} step {step + 1}/{steps}: loss {total / 500:.4f}", flush=True)
            total = 0.0
    net.eval()
    return net


@torch.no_grad()
def score_rows(W, net, rows, dead_rows=None):
    """Logits for rows (numpy positions); every row's history must lie in its own chunk, so rows are
    scored from summaries of their chunk-mates. dead_rows: optional (len(W.rows), 8) bool array."""
    rows_t = torch.from_numpy(rows).to(DEVICE)
    need = torch.unique(W.hist[rows_t].reshape(-1))
    z = torch.zeros((len(W.rows), 128), dtype=torch.float16, device=DEVICE)
    dead_all = None if dead_rows is None else torch.from_numpy(dead_rows).to(DEVICE)
    for b in range(0, len(need), 8192):
        sel = need[b:b + 8192]
        dead = torch.zeros((len(sel), len(SIGNALS)), dtype=torch.bool, device=DEVICE) if dead_all is None \
            else dead_all[sel]
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            z[sel] = net.summarise(*W.spectra(sel, dead)).half()
    out = np.empty(len(rows), dtype=np.float32)
    for b in range(0, len(rows), 4096):
        h = W.hist[rows_t[b:b + 4096]]
        out[b:b + 4096] = net.sequence(z[h].float())[:, -1].cpu().numpy()
    return out


def combo_dead(n, combo):
    dead = np.zeros((n, len(SIGNALS)), bool)
    for s in combo:
        dead[:, SIGNALS.index(s)] = True
    return dead


def sensor_aware(W, events, other, prob):
    det = n = fa = 0
    hours, fa_fold = 0.0, []
    for k in range(N_FOLDS):
        thr = best_threshold(other[k], prob[W.fold != k], budget=SELECT_BUDGET)
        d, nk, f, h = events[k].score(prob[W.fold == k] >= thr, *ALARM)
        det, n, fa, hours = det + d, n + nk, fa + f, hours + h
        fa_fold.append(f / h)
    return det / n, fa / hours, sum(x > FA_BUDGET for x in fa_fold)


def cv():
    t_start = time.time()
    W = GpuWindows()
    print(f"{len(W.rows)} rows on {DEVICE}, {time.time() - t_start:.0f} s", flush=True)
    nets = []
    for k in range(N_FOLDS):
        path = MODELS / f"cnn_spec{TAG}_fold{k}.pt"
        if path.exists():
            net = SpecNet().to(DEVICE)
            net.load_state_dict(torch.load(path))
            net.eval()
        else:
            net = train_net(W, np.flatnonzero(W.fold != k), SEED + k, tag=f"fold {k}")
            torch.save(net.state_dict(), path)
        nets.append(net)
        print(f"fold {k} ready, {time.time() - t_start:.0f} s", flush=True)
    meta = W.rows[["subject", "run", "t", "label"]]
    events = {k: FastEvents(meta[W.fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(meta[W.fold != k]) for k in range(N_FOLDS)}
    res = []
    for combo in combinations():
        dead = combo_dead(len(W.rows), combo) if combo else None
        prob = np.empty(len(W.rows), dtype=np.float32)
        for k, net in enumerate(nets):
            idx = np.flatnonzero(W.fold == k)
            prob[idx] = score_rows(W, net, idx, dead)
        sens, fa, over = sensor_aware(W, events, other, prob)
        res.append(dict(model="spectrogram cnn+gru", dead=len(combo), sensors_dead=" ".join(combo) or "none",
                        eeg_alive=not EEG <= set(combo), sensitivity=sens, fa_per_h=fa, folds_over_budget=over,
                        **(window_scores(W.label, prob) if not combo else {})))
        if not combo:
            meta.assign(fold=W.fold, cnn_spec=prob).to_parquet(ROOT / "cache" / "predictions" / f"cnn_spec{TAG}_oof.parquet")
            print(res[-1], flush=True)
    out = pd.DataFrame(res).round(4)
    out.to_csv(TABLES / f"cnn_spec{TAG}_dead_sensors.csv", index=False)
    print(out[out.eeg_alive].groupby("dead")[["sensitivity", "fa_per_h"]].mean().round(3).to_string())
    print(f"done in {time.time() - t_start:.0f} s")


# ---------------------------------------------------------------- hybrid XGBoost

def chunk_dead(W, rng):
    """One random dead set (1-6 sensors) per chunk, for half the chunks (cnn.segment_dead)."""
    per_seg = random_dead(W.seg.max() + 1, rng)
    return per_seg[W.seg]


def prepare_hybrid():
    """Window table, feature table, inner CNNs and the fold models of XGBoost and the hybrid (saved after
    the first run and reused)."""
    from seizure.faults import evaluate as ev
    from seizure.models.phase_a import load_all
    from seizure.models.robustness import sensor_columns
    from seizure.models.train import feature_sets
    t_start = time.time()
    W = GpuWindows()
    data = ev.Data(raw=load_all())
    assert (data.df[["subject", "run", "t"]].to_numpy() == W.rows[["subject", "run", "t"]].to_numpy()).all()
    assert (data.fold == W.fold).all()
    fc = feature_sets(data.cols)["classic+TDA"]
    by_sensor = sensor_columns(fc)
    dead_copy = chunk_dead(W, np.random.default_rng(SEED))
    combos = combinations()
    n = len(W.rows)
    # CNN scores: for outer fold k, two inner networks on halves of the training folds.
    # train_score[k]: (clean, copy) scores of the training rows; test_score[k][combo]: the test fold's scores.
    train_score, inner = {}, {}
    for k in range(N_FOLDS):
        others = [j for j in range(N_FOLDS) if j != k]
        halves = (others[:2], others[2:])
        nets = []
        for h, half in enumerate(halves):
            path = MODELS / f"cnn_spec{TAG}_inner{k}{h}.pt"
            if path.exists():
                net = SpecNet().to(DEVICE)
                net.load_state_dict(torch.load(path))
                net.eval()
            else:
                net = train_net(W, np.flatnonzero(np.isin(W.fold, half)), SEED + 10 * k + h,
                                steps=STEPS // 2, tag=f"outer {k} inner {h}")
                torch.save(net.state_dict(), path)
            nets.append(net)
        clean, copy = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
        for h, half in enumerate(halves):
            rows = np.flatnonzero(np.isin(W.fold, halves[1 - h]))  # scored by the network that did not see them
            clean[rows] = score_rows(W, nets[h], rows)
            copy[rows] = score_rows(W, nets[h], rows, dead_copy)
        train_score[k], inner[k] = (clean, copy), nets
        print(f"outer fold {k}: inner networks ready, {time.time() - t_start:.0f} s", flush=True)

    y_all = data.df.label.to_numpy()
    X_clean = data.df[fc].to_numpy(dtype="float32")
    pos = {c: i for i, c in enumerate(fc)}
    X_copy = X_clean.copy()
    for j, s in enumerate(SIGNALS):
        idx = [pos[c] for c in by_sensor[s]]
        X_copy[np.ix_(dead_copy[:, j], idx)] = np.nan
    models = {"xgboost": [], "hybrid": []}
    for k in range(N_FOLDS):
        paths = {name: MODELS / f"{name}_spec{TAG if name == 'hybrid' else ''}_fold{k}.json" for name in models}
        tr = W.fold != k
        y = np.concatenate([y_all[tr], y_all[tr]])
        for name in models:
            if paths[name].exists():
                m = XGBClassifier()
                m.load_model(paths[name])
                models[name].append(m)
                continue
            X = np.concatenate([X_clean[tr], X_copy[tr]])
            cols = list(fc)
            if name == "hybrid":
                X = np.column_stack([X, np.concatenate([train_score[k][0][tr], train_score[k][1][tr]])])
                cols = cols + ["cnn_score"]
            m = XGBClassifier(**PARAMS, scale_pos_weight=(y == 0).sum() / (y == 1).sum())
            m.fit(pd.DataFrame(X, columns=cols, copy=False), y)
            m.save_model(paths[name])
            models[name].append(m)
            del X
        print(f"outer fold {k}: XGBoost and hybrid trained, {time.time() - t_start:.0f} s", flush=True)

    return W, fc, by_sensor, pos, X_clean, inner, models, combos, t_start


def hybrid_probs(W, fc, by_sensor, pos, X_clean, inner, models, combo):
    """{model: probabilities} for XGBoost and the hybrid, with the sensors in combo dead."""
    n = len(W.rows)
    dead = combo_dead(n, combo) if combo else None
    X = X_clean.copy()
    for s in combo:
        X[:, [pos[c] for c in by_sensor[s]]] = np.nan
    cnn = np.empty(n, np.float32)
    for k in range(N_FOLDS):
        idx = np.flatnonzero(W.fold == k)
        cnn[idx] = np.mean([score_rows(W, net, idx, dead) for net in inner[k]], axis=0)
    out = {}
    for name, ms in models.items():
        prob = np.empty(n)
        for k, m in enumerate(ms):
            sel = W.fold == k
            Xk = X[sel] if name == "xgboost" else np.column_stack([X[sel], cnn[sel]])
            cols = list(fc) if name == "xgboost" else list(fc) + ["cnn_score"]
            prob[sel] = m.predict_proba(pd.DataFrame(Xk, columns=cols, copy=False))[:, 1]
        out[name] = prob
    return out


def hybrid():
    W, fc, by_sensor, pos, X_clean, inner, models, combos, t_start = prepare_hybrid()
    meta = W.rows[["subject", "run", "t", "label"]]
    events = {k: FastEvents(meta[W.fold == k]) for k in range(N_FOLDS)}
    other = {k: FastEvents(meta[W.fold != k]) for k in range(N_FOLDS)}
    res = []
    for ci, combo in enumerate(combos):
        for name, prob in hybrid_probs(W, fc, by_sensor, pos, X_clean, inner, models, combo).items():
            sens, fa, over = sensor_aware(W, events, other, prob)
            res.append(dict(model=name, dead=len(combo), sensors_dead=" ".join(combo) or "none",
                            eeg_alive=not EEG <= set(combo), sensitivity=sens, fa_per_h=fa, folds_over_budget=over,
                            **(window_scores(W.label, prob) if not combo else {})))
            if not combo:
                print(res[-1], flush=True)
        if not combo:
            gain = models["hybrid"][0].get_booster().get_score(importance_type="gain")
            top = sorted(gain.items(), key=lambda kv: -kv[1])
            rank = [c for c, _ in top].index("cnn_score") + 1 if "cnn_score" in gain else None
            print(f"cnn_score gain rank in fold 0: {rank} of {len(top)}", flush=True)
        if ci % 10 == 0:
            pd.DataFrame(res).round(4).to_csv(TABLES / f"hybrid{TAG}_dead_sensors.csv", index=False)
            print(f"combination {ci + 1}/{len(combos)}, {time.time() - t_start:.0f} s", flush=True)
    out = pd.DataFrame(res).round(4)
    out.to_csv(TABLES / f"hybrid{TAG}_dead_sensors.csv", index=False)
    print(out[out.eeg_alive].groupby(["model", "dead"])[["sensitivity", "fa_per_h"]].mean().round(3)
          .unstack("model").to_string())
    print(f"done in {time.time() - t_start:.0f} s")


N_BOOT = 2000


def per_patient(W, prob):
    """Per patient: detected, seizures, false alarms, hours, with each fold's threshold for these sensors."""
    meta = W.rows[["subject", "run", "t", "label"]]
    thr = np.empty(len(prob))
    for k in range(N_FOLDS):
        thr[W.fold == k] = best_threshold(FastEvents(meta[W.fold != k]), prob[W.fold != k], budget=SELECT_BUDGET)
    out = []
    for sub, rows in W.rows.groupby("subject").indices.items():
        d, n, fa, h = FastEvents(meta.iloc[rows]).score(prob[rows] >= thr[rows], *ALARM)
        out.append((sub, d, n, fa, h))
    return out


def stats():
    """Patient bootstrap: spectrogram CNN + GRU and the hybrid against XGBoost, on clean data and with 5 of 8
    sensors dead (the sets with an EEG alive, summed per patient)."""
    W, fc, by_sensor, pos, X_clean, inner, models, combos, t_start = prepare_hybrid()
    cnn_nets = []
    for k in range(N_FOLDS):
        net = SpecNet().to(DEVICE)
        net.load_state_dict(torch.load(MODELS / f"cnn_spec{TAG}_fold{k}.pt"))
        net.eval()
        cnn_nets.append(net)
    rows = []
    for combo in [c for c in combos if len(c) in (0, 5) and not EEG <= set(c)]:
        probs = hybrid_probs(W, fc, by_sensor, pos, X_clean, inner, models, combo)
        dead = combo_dead(len(W.rows), combo) if combo else None
        cnn = np.empty(len(W.rows), np.float32)
        for k, net in enumerate(cnn_nets):
            idx = np.flatnonzero(W.fold == k)
            cnn[idx] = score_rows(W, net, idx, dead)
        probs["spectrogram cnn+gru"] = cnn
        for name, prob in probs.items():
            rows += [dict(model=name, dead=len(combo), sensors=" ".join(combo) or "none", subject=s, detected=d,
                          seizures=n, false_alarms=f, hours=h) for s, d, n, f, h in per_patient(W, prob)]
        print(f"{' '.join(combo) or 'clean'}: {time.time() - t_start:.0f} s", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(TABLES / f"cnn_spec{TAG}_per_patient.csv", index=False)
    g = r.groupby(["model", "dead", "subject"])[["detected", "seizures", "false_alarms", "hours"]].sum()
    people = sorted(r.subject.unique())
    boot = np.random.default_rng(SEED).integers(0, len(people), (N_BOOT, len(people)))
    out = []
    for dead in (0, 5):
        base = g.loc[("xgboost", dead)].reindex(people).fillna(0).to_numpy()
        for name in ("spectrogram cnn+gru", "hybrid"):
            m = g.loc[(name, dead)].reindex(people).fillna(0).to_numpy()
            mb, bb = m[boot].sum(1), base[boot].sum(1)
            diff = mb[:, 0] / mb[:, 1] - bb[:, 0] / bb[:, 1]
            out.append(dict(test="clean" if dead == 0 else "5 of 8 sensors dead", model=name,
                            sens=m[:, 0].sum() / m[:, 1].sum(), fa_per_h=m[:, 2].sum() / m[:, 3].sum(),
                            xgboost_sens=base[:, 0].sum() / base[:, 1].sum(),
                            diff_pts=(m[:, 0].sum() / m[:, 1].sum() - base[:, 0].sum() / base[:, 1].sum()) * 100,
                            lo=np.percentile(diff, 2.5) * 100, hi=np.percentile(diff, 97.5) * 100,
                            p_not_better=float((diff <= 0).mean())))
    res = pd.DataFrame(out).round(4)
    res.to_csv(TABLES / f"cnn_spec{TAG}_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    {"cv": cv, "hybrid": hybrid, "stats": stats}[sys.argv[1]]()
