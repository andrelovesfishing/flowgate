import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from flowgate.space import Space


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    return rng.normal(size=(2000, 6)) @ A + 3.0


def test_pca_space_is_standardised_on_train(X):
    z = Space.fit(X, d=4, kind="pca").transform(X)
    assert z.shape == (2000, 4)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(np.cov(z.T), np.eye(4), atol=1e-6)


def test_pca_keeps_largest_variance_directions(X):
    sp = Space.fit(X, d=2, kind="pca")
    full = Space.fit(X, d=6, kind="pca")
    # the first d coordinates of a larger space are the d-dim space
    np.testing.assert_allclose(np.abs(sp.transform(X)), np.abs(full.transform(X)[:, :2]), atol=1e-8)


@pytest.mark.parametrize("kind", ["pca", "wpca"])
def test_full_rank_space_turns_background_gaussian_into_standard_normal(X, kind):
    """With d = D, ||z||^2 is the background Gaussian's Mahalanobis distance, so the
    background density is N(0, I) in z (which is what lets the dropped dims cancel)."""
    z = Space.fit(X, d=6, kind=kind).transform(X[:50])
    mu = X.mean(0)
    cov = np.cov(X.T) if kind == "pca" else LedoitWolf().fit(X).covariance_
    m = np.einsum("ni,ij,nj->n", X[:50] - mu, np.linalg.inv(cov), X[:50] - mu)
    np.testing.assert_allclose((z**2).sum(1), m, rtol=1e-6)


def test_transform_returns_float32(X):
    assert Space.fit(X, d=3).transform(X).dtype == np.float32


def test_rejects_unknown_kind(X):
    with pytest.raises(ValueError):
        Space.fit(X, d=3, kind="nope")
