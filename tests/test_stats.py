import numpy as np

from flowgate.stats import paired_auroc_diff


def _data(seed=0, n_in=600, n_oos=150):
    rng = np.random.default_rng(seed)
    is_oos = np.r_[np.zeros(n_in, bool), np.ones(n_oos, bool)]
    base = rng.normal(size=n_in + n_oos) - 1.0 * is_oos  # higher = in-scope
    return base, is_oos, rng


def test_identical_scores_give_zero_difference_and_ci_containing_zero():
    s, is_oos, _ = _data()
    r = paired_auroc_diff(s, s, is_oos, n_boot=300)
    assert r["diff"] == 0 and r["lo"] <= 0 <= r["hi"]


def test_clearly_better_score_has_ci_above_zero():
    s, is_oos, rng = _data()
    better = s - 2.0 * is_oos
    r = paired_auroc_diff(better, s, is_oos, n_boot=300)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p_a_better"] > 0.99


def test_bootstrap_is_seeded():
    s, is_oos, rng = _data()
    t = s + rng.normal(size=len(s))
    assert paired_auroc_diff(s, t, is_oos, n_boot=200) == paired_auroc_diff(s, t, is_oos, n_boot=200)
