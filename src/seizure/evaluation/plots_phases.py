"""Figures 17-25: the health check, Phase A, Phase B and Phase C (docs/pipeline.md).

    python -m seizure.evaluation.plots_phases

Reads only results/tables/*.csv (no models, no predictions) and writes PNGs to results/figures/,
in the same style as plots.py.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seizure.evaluation.plots import FAULT_NAMES, INK, INK2, MODELS, MUTED, SURFACE, save
from seizure.features.build import TABLES
from seizure.models.train import FA_BUDGET

TDA_C, CLASSIC_C, SHAPE_C = MODELS["classic+TDA"][1], MODELS["classic"][1], MODELS["classic+TDA_shape"][1]
PCT = matplotlib.ticker.PercentFormatter(1)
SENSORS = ["EEG1", "EEG2", "ECG", "EMG", "ACC_neck", "ACC_chest", "GYR_neck", "GYR_chest"]


def label_bars(ax, bars, fmt="{:.0%}", dy=0.015):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v), ha="center", fontsize=8, color=INK)


def fig_health():
    """17: what the health check switches off."""
    cls = pd.read_csv(TABLES / "phaseA3_health_switch_off.csv").set_index("y")["0"]
    clean = pd.read_csv(TABLES / "phaseA3_health_clean_sensors.csv", index_col=0).healthy_sensor_switched_off
    fault = pd.read_csv(TABLES / "health_by_fault.csv").set_index("fault")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[0.8, 1.6, 1.1]))
    ax = axes[0]
    order = ["healthy", "degraded", "failed"]
    bars = ax.bar(["Healthy", "Degraded", "Failed"], cls[order], color=[MUTED, "#f2a37f", TDA_C], width=0.6)
    label_bars(ax, bars)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Share of windows switched off")
    ax.set_title("By true state of the window", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    v = clean.reindex(SENSORS)
    bars = ax.bar(SENSORS, v, color=[TDA_C if s.startswith("GYR") else CLASSIC_C for s in SENSORS], width=0.6)
    label_bars(ax, bars, "{:.1%}", 0.005)
    ax.set_ylim(0, max(v) * 1.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title("Sensors with no fault at all, switched off by mistake", fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    sev = [0.25, 0.5, 0.75, 1.0]
    for f, c in zip(("pop", "decline", "dropout"), (SHAPE_C, CLASSIC_C, TDA_C)):
        ax.plot(sev, fault.loc[f, [str(s) for s in sev]], marker="o", color=c, label=FAULT_NAMES[f])
    ax.set_xticks(sev, ["25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fault severity")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Failed windows caught, by fault (pilot)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("The health check: what it switches off", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.03, "Left and middle: 90-patient health check (Phase A3). Left counts windows of faulty recordings; "
             "its 'healthy' bar includes the gyroscopes, which the middle panel shows are over-flagged even with no fault. "
             "Right: pilot health check.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "17_health_check")


def fig_a1():
    """18: 30 vs 90 training patients (numbers from docs/pipeline.md, Phase A1)."""
    rows = {"Seizures caught": ((0.739, 0.779), (0.708, 0.719)), "AUROC": ((0.813, 0.824), (0.841, 0.851)),
            "AUPR": ((0.253, 0.331), (0.385, 0.440))}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (name, (pilot, full)) in zip(axes, rows.items()):
        x = np.arange(2)
        b1 = ax.bar(x - 0.18, [pilot[0], full[0]], 0.34, color=CLASSIC_C, label="Classic")
        b2 = ax.bar(x + 0.18, [pilot[1], full[1]], 0.34, color=TDA_C, label="Classic + TDA")
        fmt = "{:.0%}" if name == "Seizures caught" else "{:.3f}"
        label_bars(ax, list(b1) + list(b2), fmt, 0.01)
        ax.set_xticks(x, ["Pilot, 30 patients", "A1, 90 patients"])
        ax.set_ylim(0, 1.15 if name != "AUPR" else 0.6)
        if name == "Seizures caught":
            ax.yaxis.set_major_formatter(PCT)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.suptitle("A1: training on 90 patients instead of 30 (cross-validation, threshold at ≤ 11 false alarms/h)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "18_a1_30_vs_90")


def fig_a2():
    """19: false alarms of each unseen fold, per threshold strategy (classic + TDA)."""
    t = pd.read_csv(TABLES / "phaseA_thresholds.csv").query("model == 'classic+TDA'").set_index(["strategy", "setting"])
    picks = [(("base", "budget 11/h"), "Best at ≤ 11/h (before)"),
             (("margin", "budget 10/h"), "Margin: best at ≤ 10/h"),
             (("margin", "budget 9/h"), "Margin: best at ≤ 9/h"),
             (("margin", "budget 8/h"), "Margin: best at ≤ 8/h (adopted)"),
             (("margin, nested", "budget chosen on other folds"), "Margin chosen by nested CV"),
             (("patient", "floor at own 0.950 quantile"), "Per-patient floor (95th pct.)"),
             (("rule", "6 of 10"), "Stricter alarm: 6 of 10"),
             (("rule", "8 of 10"), "Stricter alarm: 8 of 10")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, name) in enumerate(picks):
        r = t.loc[key]
        folds = [float(v) for v in r.fa_folds.split()]
        adopted = "adopted" in name
        c = TDA_C if adopted else CLASSIC_C
        ax.plot(folds, [i] * 5, "o", ms=8, color=c, alpha=0.8, mec=SURFACE)
        ax.text(15.2, i, f"{r.sensitivity:.0%} caught", va="center", fontsize=9, color=INK if adopted else INK2,
                fontweight="bold" if adopted else "normal")
    ax.axvline(FA_BUDGET, color=INK2, ls="--", lw=1)
    ax.text(FA_BUDGET + 0.1, -0.55, "budget 11/h", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(picks)), [p[1] for p in picks])
    ax.set_ylim(len(picks) - 0.5, -0.8)
    ax.set_xlim(0, 17.5)
    ax.set_xlabel("False alarms per hour on each unseen fold (one dot per fold)")
    ax.set_title("A2: which threshold rule keeps the false-alarm budget on unseen patients (classic + TDA)")
    ax.grid(axis="y", visible=False)
    save(fig, "19_a2_threshold_rules")


def fig_a3():
    """20: drop under faults on 5 of 8 sensors, A1 vs A3 (health check on)."""
    f = pd.read_csv(TABLES / "phaseA3_faults.csv")
    c = pd.read_csv(TABLES / "phaseA3_clean.csv").set_index(["training", "model"]).sensitivity
    f = f[(f.n_sensors == 5) & f.eeg_alive & (f["mode"] == "B")]
    lines = [(("A1", "classic+TDA"), TDA_C, ":", "Classic + TDA, dropout training (A1)"),
             (("A3", "classic+TDA"), TDA_C, "-", "Classic + TDA, + fault training (A3)"),
             (("A3", "classic"), CLASSIC_C, "-", "Classic, + fault training (A3)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, fault in zip(axes, ("pop", "decline", "dropout")):
        for (tr, m), color, style, label in lines:
            g = f[(f.fault == fault) & (f.training == tr) & (f.model == m)].groupby("severity").sensitivity.mean()
            ax.plot([0.0] + list(g.index), [0] + list((c[(tr, m)] - g.values) * 100), ls=style, color=color,
                    marker="o", ms=5, label=label)
        ax.axhline(15, color=MUTED, lw=1, ls="--")
        ax.set_title(FAULT_NAMES[fault], fontsize=11)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Fault severity")
        ax.set_ylim(-2, 16)
    axes[0].set_ylabel("Drop from own clean result (points)")
    axes[0].text(0.02, 15.3, "target: at most 15 points", fontsize=8, color=INK2)
    axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 0.93))
    fig.suptitle("A3: training with injected faults (faults on 5 of 8 sensors, health check on, 90 patients)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "20_a3_fault_training")


def fig_b1():
    """21: TDA-B, paired difference over dead-sensor sets with an EEG alive."""
    d = pd.read_csv(TABLES / "phaseB1_dead_sensors.csv")
    d = d[d.eeg_alive].pivot(index="sensors_dead", columns="model", values="sensitivity")
    diff = (d["classic+TDA+TDAB"] - d["classic+TDA"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    bins = np.arange(np.floor(diff.min()) - 0.5, np.ceil(diff.max()) + 1.5, 1)
    ax.hist(diff, bins=bins, color=SHAPE_C, edgecolor=SURFACE)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.axvline(diff.mean(), color=TDA_C, lw=2)
    ax.text(-5.3, ax.get_ylim()[1] * 0.92, f"mean {diff.mean():+.2f} points\n(orange line)", color=TDA_C, fontsize=9,
            va="top")
    ax.set_xlabel("Seizures caught with TDA-B minus without (points)")
    ax.set_ylabel("Dead-sensor sets")
    ax.set_title(f"{len(diff)} sensor sets: {int((diff > 0).sum())} better, {int((diff < 0).sum())} worse, "
                 f"{int((diff == 0).sum())} tied (Wilcoxon p = 0.54)", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    raw = pd.read_csv(TABLES / "phaseB1_dead_sensors.csv")
    raw = raw[raw.eeg_alive]
    for m, color, name in (("classic+TDA", TDA_C, "Classic + TDA"), ("classic+TDA+TDAB", SHAPE_C, "+ TDA-B")):
        g = raw[raw.model == m].groupby("dead").sensitivity.mean()
        ax.plot(g.index, g.values, marker="o", color=color, label=name, ls="-" if m == "classic+TDA" else "--")
    ax.set_xticks(range(0, 8), [str(k) for k in range(8)])
    ax.set_xlabel("Sensors dead (out of 8)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Seizures caught")
    ax.set_title("Average over sets: the two lines overlap", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("B1: the shape of the sensor network (TDA-B) adds nothing measurable on SeizeIT2",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "21_b1_tda_network")


def fig_b2():
    """22: CNN baselines vs XGBoost as sensors die."""
    x = pd.read_csv(TABLES / "phaseB1_dead_sensors.csv").query("model == 'classic+TDA'")
    c1 = pd.read_csv(TABLES / "cnn_dead_sensors.csv")
    c2 = pd.read_csv(TABLES / "cnn2_dead_sensors.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for d, color, style, name in ((x, TDA_C, "-", "XGBoost, classic + TDA features"),
                                  (c1, CLASSIC_C, "-", "CNN on raw signals, last 4 s"),
                                  (c2, CLASSIC_C, ":", "CNN + GRU, last 60 s")):
        g = d[d.eeg_alive].groupby("dead").sensitivity.mean()
        ax.plot(g.index, g.values, ls=style, color=color, marker="o", ms=5, label=name)
    ax.set_xticks(range(0, 8), [f"{k}\n({k / 8:.0%})" for k in range(8)])
    ax.set_xlabel("Sensors dead (out of 8)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Seizures caught (at ≤ 8 false alarms/h on training folds)")
    ax.set_title("B2: hand-made features vs a deep model learned from raw signals (90 patients)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0, -0.2, "Average over dead-sensor sets with at least one EEG alive. The CNN's flat line reflects its low "
            "starting point, not robustness.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "22_b2_cnn_baseline")


ABLATION_NAMES = {"-TDA": "TDA features", "-TDA size": "TDA size summaries only", "-TDA shape": "TDA shape summaries only",
                  "-memory": "10 s / 60 s memory", "-fault train": "Fault training",
                  "-robust train": "All robustness training", "-health check": "Health check",
                  "-sensor-aware": "Sensor-aware threshold"}


def fig_ablation():
    """23: points lost when each part is removed, with patient-bootstrap 95% intervals."""
    a = pd.read_csv(TABLES / "phaseC_ablation.csv")
    order = ["-memory", "-health check", "-robust train", "-TDA", "-fault train", "-TDA shape", "-TDA size",
             "-sensor-aware"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kind, title in zip(axes, ("clean", "faults on 5 of 8"), ("Clean data", "Injected faults on 5 of 8 sensors")):
        r = a[a.kind == kind].set_index("variant")
        y = np.arange(len(order))
        for i, v in enumerate(order):
            d, lo, hi = r.loc[v, ["diff_vs_full", "diff_lo", "diff_hi"]]
            sig = lo > 0
            color = TDA_C if v == "-TDA" else (CLASSIC_C if sig else MUTED)
            ax.plot([lo, hi], [i, i], color=color, lw=2.5, solid_capstyle="butt")
            ax.plot(d, i, "o", ms=8, color=color, mec=SURFACE)
            note = f"{d:+.1f}"
            if v == "-sensor-aware" and kind != "clean":
                note += f"  (but {r.loc[v, 'fa_per_h']:.1f} false alarms/h)"
            ax.text(max(hi, d) + 0.6, i, note, va="center", fontsize=8.5, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [ABLATION_NAMES[v] for v in order])
        ax.invert_yaxis()
        ax.set_xlim(-9, 26)
        ax.set_xlabel("Seizures caught: full system minus system without this part (points)")
        full = r.loc["full"]
        ax.set_title(f"{title}: full system {full.sensitivity:.1%} at {full.fa_per_h:.1f}/h", fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.suptitle("C1 + C2: what each part contributes (bars: 95% interval from 2,000 patient resamples)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, "Right of 0: removing the part loses seizures. Coloured: the interval excludes 0. "
             "Orange: TDA (p = 0.019 under faults, no effect on clean data).", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "23_c_ablation")


def fig_timing():
    """24: time per second of data in the live run."""
    t = pd.read_csv(TABLES / "phaseC3_timing.csv", index_col=0)
    steps = [s for s in t.index if s not in ("total", "alarm")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw=dict(height_ratios=[1, 2.2]))
    ax = axes[0]
    left = 0
    colors = [MUTED, TDA_C, "#f2a37f", SHAPE_C, CLASSIC_C, INK, "#8fb8ea"]
    for s, c in zip(steps, colors):
        ax.barh(0, t.loc[s, "mean_ms"], left=left, color=c, height=0.5)
        left += t.loc[s, "mean_ms"]
    ax.axvline(1000, color=INK2, ls="--", lw=1)
    ax.text(990, 0, "real time:\n1,000 ms per second", ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(left + 10, 0, f"{t.loc['total', 'mean_ms']:.0f} ms on average", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1050)
    ax.set_yticks([])
    ax.set_title("Each second of data takes under a tenth of a second", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    y = np.arange(len(steps))
    ax.barh(y, t.loc[steps, "mean_ms"], color=colors, height=0.6)
    for i, s in enumerate(steps):
        ax.plot([t.loc[s, "mean_ms"], t.loc[s, "p99_ms"]], [i, i], color=INK2, lw=1)
        ax.text(t.loc[s, "p99_ms"] + 0.8, i, f"{t.loc[s, 'mean_ms']:.1f} ms (99th pct. {t.loc[s, 'p99_ms']:.1f})",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [s[0].upper() + s[1:] for s in steps])
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("Milliseconds per second of data (bar: mean, line to 99th percentile)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("C3: live run, 1,800 seconds from 3 patients on a laptop CPU", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "24_c3_live_timing")


def fig_real_faults():
    """25: genuinely broken motion sensors in SeizeIT2."""
    r = pd.read_csv(TABLES / "phaseC4_real_faults.csv")
    h = pd.read_csv(TABLES / "phaseC4_real_faults_healthy_rule.csv")
    stuck = r[r.group == "broken (stuck)"]
    groups = [("Pinned at full scale\nor dead device", stuck[stuck.rule >= 0.5]),
              ("Labelled stuck, but\nlooks like rest", stuck[stuck.rule < 0.5]),
              ("Matched healthy\nsensors", r[r.group == "healthy control"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    x = np.arange(len(groups))
    for j, (col, color, name) in enumerate((("health_check", CLASSIC_C, "Health check"),
                                            ("rule", SHAPE_C, "Stuck rule"),
                                            ("combined_rule", TDA_C, "Both (the system)"))):
        bars = ax.bar(x + (j - 1) * 0.26, [g[col].mean() for _, g in groups], 0.26, color=color, label=name)
        label_bars(ax, bars, "{:.0%}", 0.015)
    ax.set_xticks(x, [f"{n}\n({len(g)} sensor-chunks)" for n, g in groups], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Share of seconds switched off")
    ax.set_title("Wrongly scaled accelerometers (48): all rejected at cleaning", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    variants = [("rule_window_edge", "Pinned at the window's\nown min / max"), ("rule_constant_any", "Any exact constant"),
                ("rule", "Full scale or constant,\ngyroscope at 0 allowed\n(adopted)"), ("rule_300s", "Same, 300 s window")]
    v = [h[c].mean() for c, _ in variants]
    y = np.arange(len(variants))
    ax.barh(y, v, color=[MUTED, MUTED, TDA_C, CLASSIC_C], height=0.6)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.002, yi, f"{vi:.1%}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [n for _, n in variants], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_title(f"Rule versions on all {len(h):,} healthy motion sensor-chunks:\nseconds switched off by mistake",
                 fontsize=10.5)
    ax.grid(axis="y", visible=False)
    fig.suptitle("C4: the motion sensors that are really broken in SeizeIT2 (90 patients)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "25_c4_real_faults")


D_MODELS = (("classic", CLASSIC_C, "Classic"), ("classic+TDA", TDA_C, "Classic + TDA"),
            ("classic+TDA+TDAB", SHAPE_C, "Classic + TDA + TDA-B (network)"))


def fig_d_dead():
    """26: CHB-MIT, seizures caught and false alarms as channels die."""
    d = pd.read_csv(TABLES / "phaseD_dead_channels.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    for m, color, name in D_MODELS:
        g = d[(d.model == m) & (d.rule == "count-aware")].groupby("dead").sensitivity.mean()
        ax.plot(g.index, g.values, marker="o", ms=5, color=color, label=name)
    clean = d[(d.model == "classic+TDA") & (d.dead == 0)].sensitivity.iloc[0]
    ax.axhline(clean - 0.15, color=INK2, ls="--", lw=1)
    ax.text(0.2, clean - 0.15 + 0.012, "target: at most 15 points below clean (classic + TDA)", fontsize=8.5,
            color=INK2)
    ax.axvline(11, color=MUTED, lw=1, ls=":")
    ax.text(11.15, 0.05, "11 of 18 dead (60%)", fontsize=8.5, color=INK2)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xlabel("Channels dead (out of 18)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Seizures caught")
    ax.set_title("Seizures caught (count-aware threshold)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axes[1]
    for rule, style in (("fixed", ":"), ("count-aware", "-")):
        g = d[(d.model == "classic+TDA") & (d.rule == rule)].groupby("dead").fa_per_h.mean()
        ax.plot(g.index, g.values, ls=style, marker="o", ms=4, color=TDA_C,
                label=f"{'one fixed threshold' if rule == 'fixed' else 'threshold by number of dead channels'}")
    ax.axhline(FA_BUDGET, color=INK2, ls="--", lw=1)
    ax.text(0.2, FA_BUDGET + 1, "budget 11/h", fontsize=8.5, color=INK2)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xlabel("Channels dead (out of 18)")
    ax.set_ylabel("False alarms per hour")
    ax.set_title("False alarms, classic + TDA", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("CHB-MIT: dead channels (23 people, sensor-dropout training, 8 random sets per level)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "26_d_dead_channels")


def fig_d_faults():
    """27: CHB-MIT, injected faults on 11 of 18 channels."""
    f = pd.read_csv(TABLES / "phaseD_faults.csv")
    clean = pd.read_csv(TABLES / "phaseD_a3_clean.csv").set_index("model").sensitivity
    f = f[(f.n_channels == 11) & (f.training == "A3")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, fault in zip(axes, ("pop", "decline", "dropout")):
        for m, color, name in D_MODELS:
            for mode, style in (("B", "-"), ("A", ":")):
                if mode == "A" and m != "classic+TDA":
                    continue
                g = f[(f.fault == fault) & (f.model == m) & (f["mode"] == mode)].groupby("severity").sensitivity.mean()
                ax.plot([0.0] + list(g.index), [clean[m]] + list(g.values), ls=style, color=color, marker="o", ms=5,
                        label=f"{name}, {'health check' if mode == 'B' else 'no health check'}")
        ax.axhline(clean["classic+TDA"] - 0.15, color=MUTED, lw=1, ls="--")
        ax.set_title(FAULT_NAMES[fault], fontsize=11)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Fault severity")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Seizures caught")
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[0].text(0.02, clean["classic+TDA"] - 0.15 + 0.015, "target line", fontsize=8, color=INK2)
    fig.suptitle("CHB-MIT: injected faults on 11 of 18 channels (fault training, 6 channel sets per condition)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "27_d_faults")


def fig_d_stats():
    """28: CHB-MIT, bootstrap differences between feature sets."""
    s = pd.read_csv(TABLES / "phaseD_stats.csv")
    s = s[s.a != "A1"]
    tests = ["clean", "11 channels dead", "faults on 11 of 18, health check"]
    pairs = [("classic", "classic+TDA", "TDA added to classic", TDA_C),
             ("classic+TDA", "classic+TDA+TDAB", "TDA-B added to classic + TDA", SHAPE_C)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y, labels = 0, []
    for test in tests:
        for a, b, name, color in pairs:
            r = s[(s.test == test) & (s.a == a) & (s.b == b)].iloc[0]
            sig = r.lo > 0
            ax.plot([r.lo, r.hi], [y, y], color=color if sig else MUTED, lw=2.5, solid_capstyle="butt")
            ax.plot(r.diff_pts, y, "o", ms=8, color=color if sig else MUTED, mec=SURFACE)
            ax.text(max(r.hi, r.diff_pts) + 0.3, y, f"{r.diff_pts:+.1f}  (p = {r.p_not_better:.3f})", va="center",
                    fontsize=8.5, color=INK2)
            labels.append(f"{test}: {name}")
            y += 1
        y += 0.5
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([i + (i // 2) * 0.5 for i in range(len(labels))], labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-4, 12)
    ax.set_xlabel("Change in seizures caught (points), 95% interval from 2,000 resamples of the 23 people")
    ax.set_title("CHB-MIT: what TDA and TDA-B add (coloured: interval above 0)")
    ax.grid(axis="y", visible=False)
    save(fig, "28_d_bootstrap")


def fig_d_health():
    """29: CHB-MIT health check."""
    cls = pd.read_csv(TABLES / "phaseD_health_switch_off.csv").set_index("true_class").switched_off
    fault = pd.read_csv(TABLES / "phaseD_health_by_fault.csv", index_col=0)
    clean = pd.read_csv(TABLES / "phaseD_health_clean_channels.csv").healthy_switched_off
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[0.9, 1.2]))
    ax = axes[0]
    bars = ax.bar(["Channels with\nno fault", "Faulty: window\nunharmed", "Degraded", "Failed"],
                  [clean.mean(), cls["healthy"], cls["degraded"], cls["failed"]],
                  color=[CLASSIC_C, MUTED, "#f2a37f", TDA_C], width=0.6)
    label_bars(ax, bars, "{:.2%}", 0.015)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Share of seconds switched off")
    ax.set_title("What it switches off", fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    sev = [0.25, 0.5, 0.75, 1.0]
    for fl, c in zip(("pop", "decline", "dropout"), (SHAPE_C, CLASSIC_C, TDA_C)):
        ax.plot(sev, fault.loc[fl, [str(x) for x in sev]], marker="o", color=c, label=FAULT_NAMES[fl])
    ax.set_xticks(sev, ["25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fault severity")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Faulty seconds flagged (degraded or failed)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("CHB-MIT: the health check, retrained on CHB-MIT's injected faults", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "29_d_health")


def fig_spec():
    """30: spectrogram CNN + GRU and the hybrid against XGBoost and the first CNNs, as sensors die."""
    c = pd.read_csv(TABLES / "cnn_spec_dead_sensors.csv")
    h = pd.read_csv(TABLES / "hybrid_dead_sensors.csv")
    c1 = pd.read_csv(TABLES / "cnn_dead_sensors.csv")
    lines = [(h, "xgboost", TDA_C, ":", "XGBoost, classic + TDA"),
             (h, "hybrid", TDA_C, "-", "Hybrid: XGBoost + the spectrogram CNN's score"),
             (c, "spectrogram cnn+gru", SHAPE_C, "-", "Spectrogram CNN + GRU (end to end)"),
             (c1, "cnn", CLASSIC_C, ":", "First CNN (raw signals, 4 s)")]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for d, m, color, style, name in lines:
        g = d[(d.model == m) & d.eeg_alive].groupby("dead").sensitivity.mean()
        ax.plot(g.index, g.values, ls=style, color=color, marker="o", ms=5, label=name)
    ax.set_xticks(range(0, 8), [f"{k}\n({k / 8:.0%})" for k in range(8)])
    ax.set_xlabel("Sensors dead (out of 8)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Seizures caught (at <= 8 false alarms/h on training folds)")
    ax.set_title("Deep learning revisited: spectrogram CNN + GRU and the hybrid (SeizeIT2, 90 patients)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(0, -0.2, "Average over dead-sensor sets with at least one EEG alive; thresholds per set of dead sensors, "
            "chosen on the other folds.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "30_spectrogram_cnn_hybrid")


def main():
    for f in (fig_health, fig_a1, fig_a2, fig_a3, fig_b1, fig_b2, fig_ablation, fig_timing, fig_real_faults,
              fig_d_dead, fig_d_faults, fig_d_stats, fig_d_health, fig_spec):
        f()
        print(f.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
