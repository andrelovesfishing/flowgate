"""Paired comparison of two gates scored on the same points.

Both gates see the same eval set, so their AUROCs are correlated; a paired bootstrap (resample
points once, score both gates on the same resample) gives a much tighter CI for the difference
than two separate standard errors. Resampling is stratified: in-scope and OOS points are resampled
separately, so every replicate keeps the original class counts.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_oos(score: np.ndarray, is_oos: np.ndarray) -> float:
    """AUROC of 'higher score = in-scope': the chance a random in-scope point outscores a random OOS one."""
    if is_oos.all() or (~is_oos).all():
        return float("nan")
    return float(roc_auc_score((~is_oos).astype(int), score))


def _auroc_rows(S: np.ndarray, oos: np.ndarray) -> np.ndarray:
    """AUROC ('higher = in-scope') of every row at once, via Mann-Whitney ranks (ties count half,
    as in roc_auc_score)."""
    from scipy.stats import rankdata
    pos = ~oos
    r = rankdata(S, axis=1)
    n_pos, n_neg = pos.sum(1), oos.sum(1)
    return ((r * pos).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def paired_auroc_diff(a: np.ndarray, b: np.ndarray, is_oos: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, chunk: int = 250) -> dict:
    """AUROC(a) - AUROC(b) with a 95% percentile CI. Scores are 'higher = in-scope'."""
    a, b, is_oos = np.asarray(a), np.asarray(b), np.asarray(is_oos, bool)
    ins, oos = np.flatnonzero(~is_oos), np.flatnonzero(is_oos)
    rng = np.random.default_rng(seed)
    idx = np.stack([np.r_[rng.choice(ins, len(ins)), rng.choice(oos, len(oos))] for _ in range(n_boot)])
    lab = is_oos[idx[0]]  # stratified: every replicate has the same label layout
    diffs = np.concatenate([_auroc_rows(a[i], np.broadcast_to(lab, i.shape)) -
                            _auroc_rows(b[i], np.broadcast_to(lab, i.shape)) for i in np.array_split(idx, max(1, n_boot // chunk))])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": auroc_oos(a, is_oos) - auroc_oos(b, is_oos), "lo": float(lo), "hi": float(hi),
            "p_a_better": float((diffs > 0).mean()), "n_boot": n_boot}
