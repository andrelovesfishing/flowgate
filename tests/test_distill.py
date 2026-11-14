import math

import numpy as np
import torch

from flowgate.distill import (KINDS, DistillConfig, Student, Trajectories, make_starts, record, sample_jumps,
                              student_score, train_student, val_error)
from flowgate.likelihood import std_normal_logpdf
from flowgate.net import FlowMapNet

SIG = 2.0


def _gauss_vel(x, t):  # exact velocity carrying N(0, I) at t = 0 onto N(0, SIG^2 I) at t = 1
    c = (t * SIG**2 - (1 - t)) / ((1 - t) ** 2 + t**2 * SIG**2)
    return c[:, None] * x


def _true_score(x):  # log N(x; 0, SIG^2 I) - log N(x; 0, I)
    d = x.shape[1]
    return -0.5 * (x**2).sum(1) / SIG**2 - d * math.log(SIG) + 0.5 * (x**2).sum(1)


def test_starts_keep_train_points_first_then_label_free_off_manifold_kinds():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(100, 3)).astype(np.float32)
    X, kind, hold = make_starts(z, holdout_idx=np.array([3, 7]), n_off=20, seed=0)
    assert X.shape == (160, 3) and X.dtype == np.float32
    assert np.array_equal(X[:100], z) and (kind[:100] == 0).all()
    assert [int((kind == k).sum()) for k in range(len(KINDS))] == [100, 20, 20, 20]
    assert set(np.flatnonzero(hold[:100])) == {3, 7}
    assert all(int(hold[kind == k].sum()) == 2 for k in (1, 2, 3))  # 10% of each off-manifold kind
    X2, _, _ = make_starts(z, holdout_idx=np.array([3, 7]), n_off=20, seed=0)
    assert np.array_equal(X, X2)


def test_jumps_are_dyadic_and_aligned_to_the_scoring_grid():
    n, i, j = sample_jumps(n_pool=50, steps=32, batch=4000, g=torch.Generator().manual_seed(0))
    length = j - i
    assert ((length & (length - 1)) == 0).all() and (length >= 1).all()  # powers of two
    assert (i % length == 0).all() and (i >= 0).all() and (j <= 32).all()
    assert set(length.unique().tolist()) == {1, 2, 4, 8, 16, 32} and n.max() < 50


def test_untrained_student_is_the_identity_and_scores_zero():
    st = Student(FlowMapNet(3, width=16, depth=2))
    z = torch.randn(5, 3)
    for k in (1, 4):
        assert torch.equal(student_score(st, z, k, "flowmap"), torch.zeros(5))


def _toy(seed=0, n=3000):
    g = torch.Generator().manual_seed(seed)
    X = torch.cat([SIG * torch.randn(n, 2, generator=g), torch.randn(n // 2, 2, generator=g)])
    kind = np.r_[np.zeros(n, np.int8), np.ones(n // 2, np.int8)]
    hold = np.zeros(len(X), bool); hold[::10] = True
    xs, L = record(_gauss_vel, X, steps=32)
    tr = Trajectories(xs, L, kind, hold)
    return tr.subset(~hold), tr.subset(hold)


def _check_learns(mode, tol):
    tr, va = _toy()
    torch.manual_seed(0)
    st = Student(FlowMapNet(2, width=64, depth=3))
    cfg = DistillConfig(steps=1500, batch=256, lr=2e-3, eval_every=250, patience=100, mode=mode)
    st, hist = train_student(st, tr, va, cfg)
    x = SIG * torch.randn(300, 2, generator=torch.Generator().manual_seed(9))
    for k in ((1, 4) if mode == "flowmap" else (1,)):
        err = (student_score(st, x, k, mode) - _true_score(x)).abs().mean()
        assert err < tol, f"{mode} k={k}: mean |score error| = {err:.3f}"
    return hist


def test_flowmap_student_learns_few_step_density_of_gaussian_teacher():
    hist = _check_learns("flowmap", tol=0.1)
    assert hist[-1]["val"] <= hist[0]["val"]


def test_direct_ablation_learns_the_score_in_one_shot():
    _check_learns("direct", tol=0.1)


def test_val_error_reports_each_k_and_teacher_score_is_consistent():
    tr, va = _toy(n=300)
    s = va.teacher_score()
    x0 = va.xs[0]
    torch.testing.assert_close(s, std_normal_logpdf(va.xs[-1]) + va.L[-1] - std_normal_logpdf(x0))
    st = Student(FlowMapNet(2, width=16, depth=2))
    e = val_error(st, va, "flowmap", ks=(1, 2))
    assert set(e) == {1, 2} and all(v > 0 for v in e.values())


def test_same_seed_same_student():
    tr, va = _toy(n=300)

    def run():
        torch.manual_seed(0)
        st = Student(FlowMapNet(2, width=16, depth=2))
        st, _ = train_student(st, tr, va, DistillConfig(steps=40, batch=64, eval_every=20, seed=1))
        return st.state_dict()
    a, b = run(), run()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_loss_scale_per_jump_length_never_collapses_when_targets_are_identical():
    """A linear teacher gives every path the same log-density change, so its spread is 0. The loss
    divides by the spread, so it needs a floor (1 nat): otherwise that jump length gets weight ~1e12."""
    from flowgate.distill import _jump_stats
    tr, _ = _toy(n=300)
    _, _, sd = _jump_stats(tr)
    assert (sd >= 1.0).all()
