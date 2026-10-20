"""Sensitivity (any-overlap), false alarms/h, detection delay, AUROC/AUPR, fault-detection metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from seizure.models.alarm import alarm


def runs(mask):
    """(start, stop) index pairs of the True runs in a boolean array; stop is exclusive."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


class FastEvents:
    """Same any-overlap scoring as event_scores, vectorised for many thresholds on fixed rows.

    Rows must be sorted by subject, run, t. Built once per set of rows; score() then takes
    a boolean array (prob >= threshold) and returns detected, seizures, false alarms, hours.
    """

    def __init__(self, df):
        new_seg = ((df.t.diff() != 1) | (df.subject != df.subject.shift()) | (df.run != df.run.shift())).to_numpy()
        self.seg_start = np.maximum.accumulate(np.where(new_seg, np.arange(len(df)), 0))
        self.pos = np.arange(len(df)) - self.seg_start
        self.label = df.label.to_numpy() == 1
        self.sz_id = self._run_ids(self.label, new_seg)
        self.n_sz = int(self.sz_id.max())
        self.new_seg = new_seg
        self.hours = (~self.label).sum() / 3600

    @staticmethod
    def _run_ids(mask, new_seg):
        """1, 2, 3 ... for each run of True (runs break at segment starts), 0 elsewhere."""
        prev = np.concatenate([[False], mask[:-1]]) & ~new_seg
        return np.cumsum(mask & ~prev) * mask

    def score(self, hit, x=5, y=10):
        cs = np.concatenate([[0], np.cumsum(hit)])
        i = np.arange(len(hit))
        lo = np.maximum(i - y + 1, self.seg_start)
        on = (cs[i + 1] - cs[lo] >= x) & (self.pos >= y - 1)
        al_id = self._run_ids(on, self.new_seg)
        n_al = int(al_id.max())
        both = on & self.label
        detected = int((np.bincount(self.sz_id[both], minlength=self.n_sz + 1)[1:] > 0).sum())
        overlapping = int((np.bincount(al_id[both], minlength=n_al + 1)[1:] > 0).sum())
        return detected, self.n_sz, n_al - overlapping, self.hours


def window_scores(y, prob):
    return dict(auroc=roc_auc_score(y, prob), aupr=average_precision_score(y, prob))


def event_scores(df, prob_col="prob", threshold=0.5, x=5, y=10):
    """Event-level scores over contiguous segments of seconds.

    df needs subject, run, t, label and prob_col. A segment is a stretch of consecutive seconds
    of one run (one fetched chunk). Within each segment, seizures are the runs of label == 1 and
    alarm events are the runs of the alarm rule's output.
    - A seizure is detected if any alarm event overlaps it (any-overlap scoring).
    - An alarm event that overlaps no seizure is a false alarm.
    - Delay is the first alarm second inside the seizure minus its onset.
    """
    n_sz = n_det = n_fa = 0
    delays = []
    df = df.sort_values(["subject", "run", "t"])
    seg = (df.t.diff() != 1) | (df.subject != df.subject.shift()) | (df.run != df.run.shift())
    for _, g in df.groupby(seg.cumsum(), sort=False):
        lab = g.label.to_numpy()
        alarms = runs(alarm(g[prob_col].to_numpy(), threshold, x, y))
        seizures = runs(lab == 1)
        n_sz += len(seizures)
        for s0, s1 in seizures:
            hit = [max(a0, s0) for a0, a1 in alarms if a0 < s1 and s0 < a1]
            if hit:
                n_det += 1
                delays.append(min(hit) - s0)
        n_fa += sum(not any(a0 < s1 and s0 < a1 for s0, s1 in seizures) for a0, a1 in alarms)
    hours = (df.label == 0).sum() / 3600
    return dict(seizures=n_sz, detected=n_det, sensitivity=n_det / n_sz if n_sz else np.nan,
                false_alarms=n_fa, fa_per_h=n_fa / hours, fa_per_24h=24 * n_fa / hours,
                median_delay_s=float(np.median(delays)) if delays else np.nan)
