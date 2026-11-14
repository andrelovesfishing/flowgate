import numpy as np
import pytest

from flowgate.data import SplitData
from flowgate.gate import FlowRatioGate
from flowgate.stats import auroc_oos
from flowgate.train import TrainConfig


def _clusters(n, rng, D=10):
    """In-scope = 4 tight clusters on the corners of a square, embedded in D dims with small noise.
    The overall mean is the most typical point for a single Gaussian but empty under the mixture."""
    centres = 3.0 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    X = np.zeros((n, D))
    X[:, :2] = centres[rng.integers(4, size=n)] + 0.4 * rng.normal(size=(n, 2))
    X[:, 2:] = 0.1 * rng.normal(size=(n, D - 2))
    return X


def _gate():
    return FlowRatioGate(d=2, kind="pca", width=64, depth=3, ode_steps=16,
                         train_cfg=TrainConfig(steps=3000, batch=256, lr=2e-3, eval_every=250, patience=100))


def test_flow_ratio_detects_oos_that_a_single_gaussian_calls_typical():
    rng = np.random.default_rng(0)
    train = SplitData(labels=np.zeros(3000, int), emb={"frozen": _clusters(3000, rng)})
    X_in = _clusters(200, rng)
    X_oos = np.zeros((200, 10)); X_oos[:, :2] = 0.4 * rng.normal(size=(200, 2))
    X_oos[:, 2:] = 0.1 * rng.normal(size=(200, 8))
    ev = SplitData(labels=np.r_[np.zeros(200, int), -np.ones(200, int)], emb={"frozen": np.vstack([X_in, X_oos])})
    is_oos = ev.labels < 0

    s = _gate().fit(train).score(ev)
    assert s.shape == (400,) and s.dtype == np.float64
    gauss = -((ev.emb["frozen"] - train.emb["frozen"].mean(0)) ** 2).sum(1)
    assert auroc_oos(gauss, is_oos) < 0.3  # sanity: the Gaussian gets it backwards
    assert auroc_oos(s, is_oos) > 0.9


def test_refuses_to_fit_on_oos_examples():
    rng = np.random.default_rng(0)
    train = SplitData(labels=np.array([0, 1, -1]), emb={"frozen": rng.normal(size=(3, 10))})
    with pytest.raises(ValueError):
        _gate().fit(train)


def test_saved_and_reloaded_gate_scores_identically(tmp_path):
    rng = np.random.default_rng(0)
    train = SplitData(labels=np.zeros(500, int), emb={"frozen": _clusters(500, rng)})
    ev = SplitData(labels=np.zeros(20, int), emb={"frozen": _clusters(20, rng)})
    g = FlowRatioGate(d=2, width=32, depth=2, ode_steps=4,
                      train_cfg=TrainConfig(steps=50, batch=64, eval_every=25)).fit(train)
    g.save(tmp_path / "ckpt.pt")
    h = FlowRatioGate.load(tmp_path / "ckpt.pt")
    np.testing.assert_array_equal(g.score(ev), h.score(ev))
    assert h.history == g.history and h.fit_seconds == g.fit_seconds


def test_early_stop_split_is_seeded_disjoint_and_matches_fraction():
    import torch

    from flowgate.gate import early_stop_split
    tr, ho = early_stop_split(1000, seed=3, frac=0.1)
    tr2, ho2 = early_stop_split(1000, seed=3, frac=0.1)
    assert len(ho) == 100 and len(tr) == 900 and torch.equal(ho, ho2)
    assert set(tr.tolist()).isdisjoint(ho.tolist()) and len(set(tr.tolist()) | set(ho.tolist())) == 1000
