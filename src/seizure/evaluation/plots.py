"""Figures for the clean-data models (docs/pipeline.md, Step 5.8).

    python -m seizure.evaluation.plots

Reads cache/predictions/clean_oof.parquet and results/tables/model_clean_*.csv,
writes PNGs to results/figures/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from seizure.evaluation.metrics import event_scores, runs
from seizure.features.build import CACHE, ROOT, TABLES
from seizure.models.alarm import alarm
from seizure.models.train import ALARM, FA_BUDGET

FIGS = ROOT / "results" / "figures"
MODELS = {"classic": ("Classic", "#2a78d6"),
          "classic+TDA": ("Classic + TDA", "#eb6834"),
          "classic+TDA_shape": ("Classic + TDA shape", "#1baf7a")}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEIZURE_FILL = "#e6e5e1"
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def segments(p):
    p = p.sort_values(["subject", "run", "t"])
    seg = ((p.t.diff() != 1) | (p.subject != p.subject.shift()) | (p.run != p.run.shift())).cumsum()
    return [g for _, g in p.groupby(seg, sort=False)]


def seizure_outcomes(segs, col, thr):
    """One row per seizure: segment index, start/stop positions, length, detected."""
    rows = []
    for i, g in enumerate(segs):
        al = runs(alarm(g[col].to_numpy(), thr, *ALARM))
        for s0, s1 in runs(g.label.to_numpy() == 1):
            rows.append(dict(seg=i, s0=s0, s1=s1, dur=s1 - s0,
                             detected=any(a0 < s1 and s0 < a1 for a0, a1 in al)))
    return pd.DataFrame(rows)


def fig_timelines(segs, out, thr):
    """Probability over time around three seizures: long detected, short detected, missed."""
    col = "classic+TDA"
    picks = [("Long seizure, detected", out[out.detected & (out.dur >= 60)]),
             ("Short seizure, detected", out[out.detected & (out.dur <= 20)]),
             ("Seizure missed", out[~out.detected & (out.dur >= 15)])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=False)
    for ax, (title, cand) in zip(axes, picks):
        r = cand.sort_values("dur").iloc[len(cand) // 2]
        g = segs[r.seg]
        lo, hi = max(0, r.s0 - 300), min(len(g), r.s1 + 300)
        x = (np.arange(lo, hi) - r.s0) / 60
        prob = g[col].to_numpy()[lo:hi]
        al = alarm(g[col].to_numpy(), thr, *ALARM)[lo:hi]
        ax.axvspan(0, (r.s1 - r.s0) / 60, color=SEIZURE_FILL, lw=0, label="Seizure (doctors' label)")
        ax.plot(x, prob, color=MODELS[col][1], lw=1.2, label="Model's seizure probability")
        ax.axhline(thr, color=INK2, lw=1, ls="--", label=f"Threshold ({thr})")
        for a0, a1 in runs(al):
            ax.plot([x[a0], x[min(a1, len(x) - 1)]], [-0.08, -0.08], color=INK, lw=5, solid_capstyle="butt")
        ax.set_ylim(-0.14, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Probability")
        sub = g.subject.iloc[0]
        ax.set_title(f"{title}: {sub}, {int(r.dur)} s long", fontsize=11)
        ax.text(x[0], -0.08, "alarm on ", ha="right", va="center", fontsize=8, color=INK2)
    axes[-1].set_xlabel("Minutes relative to seizure onset")
    fig.suptitle("What the model does: second-by-second seizure probability (classic + TDA)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.01, 0.955))
    save(fig, "1_timelines")


def fig_tradeoff(p, summary):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.add_patch(plt.Rectangle((0, 0.6), FA_BUDGET, 0.4, color="#cde2fb", alpha=0.45, lw=0))
    ax.text(0.3, 0.97, "Target T1: ≥ 60% caught at ≤ 11 false alarms/h", fontsize=9, color=INK2, va="top")
    for col, (name, color) in MODELS.items():
        pts = [event_scores(p, col, t, *ALARM) for t in SWEEP]
        s = summary.set_index("model").loc[col]
        ax.plot([e["fa_per_h"] for e in pts], [e["sensitivity"] for e in pts], color=color, marker="o",
                ms=4, label=f"{name}: {s.sensitivity:.0%} caught at {s.fa_per_h:.1f}/h")
        ax.plot(s.fa_per_h, s.sensitivity, marker="o", ms=10, color=color, mec=SURFACE, mew=2, ls="none")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("False alarms per hour (lower is better)")
    ax.set_ylabel("Seizures caught (higher is better)")
    ax.set_title("Trade-off: catching seizures vs false alarms")
    ax.legend(loc="lower right")
    ax.text(0, -0.17, "Each line: all thresholds 0.05-0.95, 5-of-10 alarm. Large dots: honest operating point "
            "(threshold chosen on other patients).", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "2_tradeoff")


def fig_progression():
    steps = [("First model (raw features,\n8-of-10 alarm, threshold 0.5)", 0.442),
             ("+ per-patient normalisation", 0.528),
             ("+ 5-of-10 alarm", 0.573),
             ("+ short-term context features", 0.769)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(steps))[::-1]
    ax.barh(y, [s[1] for s in steps], color=MODELS["classic+TDA"][1], height=0.55)
    for yi, (_, v) in zip(y, steps):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=10, color=INK, zorder=3,
                bbox=dict(boxstyle="square,pad=0.2", fc=SURFACE, ec="none"))
    ax.axvline(0.6, color=INK2, ls="--", lw=1)
    ax.text(0.6, len(steps) - 0.45, " target 60%", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [s[0] for s in steps])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Seizures caught at ≤ 11 false alarms/h (classic + TDA)")
    ax.set_title("How each improvement added up")
    save(fig, "3_progression")


def fig_by_length(out):
    bins = [0, 10, 20, 40, 80, 10_000]
    labels = ["≤ 10 s", "11–20 s", "21–40 s", "41–80 s", "> 80 s"]
    out = out.assign(b=pd.cut(out.dur, bins, labels=labels))
    agg = out.groupby("b", observed=False).detected.agg(["mean", "sum", "size"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, agg["mean"], color=MODELS["classic+TDA"][1], width=0.6)
    for i, (m, s, n) in enumerate(agg.itertuples(index=False)):
        ax.text(i, m + 0.02, f"{m:.0%}\n({s}/{n})", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Seizure length")
    ax.set_ylabel("Seizures caught")
    ax.set_title("Short seizures are the hardest (classic + TDA)")
    save(fig, "4_by_seizure_length")


def fig_folds(folds):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    off = {"classic": -0.2, "classic+TDA": 0, "classic+TDA_shape": 0.2}
    for col, (name, color) in MODELS.items():
        f = folds[folds.model == col]
        axes[0].plot(f.fold + off[col], f.sensitivity, "o", ms=8, color=color, label=name)
        axes[1].plot(f.fold + off[col], f.fa_per_h, "o", ms=8, color=color, label=name)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Seizures caught")
    axes[0].set_title("Seizures caught, per fold")
    axes[1].axhline(FA_BUDGET, color=INK2, ls="--", lw=1)
    axes[1].text(4.45, FA_BUDGET + 0.3, "budget", fontsize=8.5, color=INK2, ha="right")
    axes[1].set_ylim(0, 16)
    axes[1].set_ylabel("False alarms per hour")
    axes[1].set_title("False alarms, per fold")
    for ax in axes:
        ax.set_xticks(range(5), [f"Fold {k}" for k in range(5)])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "5_per_fold")


def fig_pr(p):
    base = p.label.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, (name, color) in MODELS.items():
        prec, rec, _ = precision_recall_curve(p.label, p[col])
        from sklearn.metrics import average_precision_score
        ax.plot(rec, prec, color=color, lw=1.8, label=f"{name} (AUPR {average_precision_score(p.label, p[col]):.3f})")
    ax.axhline(base, color=INK2, ls="--", lw=1)
    ax.text(0.01, base + 0.015, f"random guessing ({base:.3f})", ha="left", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: share of seizure seconds flagged")
    ax.set_ylabel("Precision: share of flagged seconds that are seizure")
    ax.set_title("Precision-recall curves (area under each = AUPR)")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "6_precision_recall")


def fig_robustness(r):
    """Sensitivity vs number of dead sensors (combinations with at least one EEG alive)."""
    alive = r[r.eeg_alive]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for col, (name, color) in MODELS.items():
        g = alive[alive.model == col].groupby("dead").sensitivity.agg(["mean", "min", "max"])
        if col == "classic+TDA":
            ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.15, lw=0,
                            label="Classic + TDA: best to worst combination")
        ax.plot(g.index, g["mean"], color=color, marker="o", ms=6, label=name)
    clean = alive[(alive.model == "classic+TDA") & (alive.dead == 0)].sensitivity.iloc[0]
    ax.axhline(clean - 0.15, color=INK2, ls="--", lw=1)
    ax.text(0.05, clean - 0.15 + 0.012, "target: at most 15 points below clean (classic + TDA)", fontsize=8.5, color=INK2)
    ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.text(5.08, 0.03, "5 of 8 dead (60%)", fontsize=8.5, color=INK2)
    ax.set_xticks(range(0, 8), [f"{k}\n({k / 8:.0%})" for k in range(8)])
    ax.set_xlabel("Sensors dead (out of 8)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("Seizures caught")
    ax.set_title("Robustness: seizures caught as sensors die (at least one EEG alive)")
    ax.legend(loc="lower left")
    ax.text(0, -0.2, "Lines: average over sampled sensor combinations; band: best to worst combination (classic + TDA). "
            "Thresholds fixed at their clean-data values.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "7_robustness_dead_sensors")


def fig_single_sensor(r):
    one = r[r.dead == 1].pivot(index="sensors_dead", columns="model", values="drop_pts")
    order = one["classic+TDA"].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.26
    x = np.arange(len(order))
    for i, (col, (name, color)) in enumerate(MODELS.items()):
        ax.bar(x + (i - 1) * w, one.loc[order, col], width=w, color=color, label=name)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Drop in seizures caught (percentage points)")
    ax.set_title("Which single sensor hurts most when it dies")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "8_single_sensor_loss")


def fig_dropout_compare(before, after):
    """Robustness curves before and after sensor-dropout training (at least one EEG alive)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for col in ("classic", "classic+TDA"):
        name, color = MODELS[col]
        for r, style, label in ((before, ":", "normal training"), (after, "-", "sensor-dropout training")):
            g = r[r.eeg_alive & (r.model == col)].groupby("dead").sensitivity.mean()
            ax.plot(g.index, g.values, ls=style, color=color, marker="o", ms=5, label=f"{name}, {label}")
    clean = after[(after.model == "classic+TDA") & (after.dead == 0)].sensitivity.iloc[0]
    ax.axhline(clean - 0.15, color=INK2, ls="--", lw=1)
    ax.text(0.05, clean - 0.15 + 0.012, "target: at most 15 points below clean (classic + TDA, dropout)",
            fontsize=8.5, color=INK2)
    ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.text(5.08, 0.03, "5 of 8 dead (60%)", fontsize=8.5, color=INK2)
    ax.set_xticks(range(0, 8), [f"{k}\n({k / 8:.0%})" for k in range(8)])
    ax.set_xlabel("Sensors dead (out of 8)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("Seizures caught")
    ax.set_title("Before and after sensor-dropout training")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "9_dropout_before_after")


def main():
    p = pd.read_parquet(CACHE.parent / "predictions" / "clean_oof.parquet")
    summary = pd.read_csv(TABLES / "model_clean_summary.csv")
    folds = pd.read_csv(TABLES / "model_clean_folds.csv")
    thr = float(folds.query("model == 'classic+TDA'").threshold.median())
    segs = segments(p)
    out = seizure_outcomes(segs, "classic+TDA", thr)
    fig_timelines(segs, out, thr)
    fig_tradeoff(p, summary)
    fig_progression()
    fig_by_length(out)
    fig_folds(folds)
    fig_pr(p)
    rob = TABLES / "robustness_dead_sensors.csv"
    if rob.exists():
        r = pd.read_csv(rob)
        fig_robustness(r)
        fig_single_sensor(r)
        drop = TABLES / "robustness_dead_sensors_dropout.csv"
        if drop.exists():
            fig_dropout_compare(r, pd.read_csv(drop))
    print("written to", FIGS)


if __name__ == "__main__":
    main()
