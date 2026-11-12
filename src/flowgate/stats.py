"""Paired comparison of two gates scored on the same points.

Both gates see the same eval set, so their AUROCs are correlated; a paired bootstrap (resample
points once, score both gates on the same resample) gives a much tighter CI for the difference
than two separate standard errors. Resampling is stratified: in-scope and OOS points are resampled
separately, so every replicate keeps the original class counts.
"""
from __future__ import annotations

import numpy as np
from kwa.metrics import auroc_oos


def paired_auroc_diff(a: np.ndarray, b: np.ndarray, is_oos: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) with a 95% percentile CI. Scores are 'higher = in-scope'."""
    a, b, is_oos = np.asarray(a), np.asarray(b), np.asarray(is_oos, bool)
    ins, oos = np.flatnonzero(~is_oos), np.flatnonzero(is_oos)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.r_[rng.choice(ins, len(ins)), rng.choice(oos, len(oos))]
        diffs[i] = auroc_oos(a[idx], is_oos[idx]) - auroc_oos(b[idx], is_oos[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": auroc_oos(a, is_oos) - auroc_oos(b, is_oos), "lo": float(lo), "hi": float(hi),
            "p_a_better": float((diffs > 0).mean()), "n_boot": n_boot}
