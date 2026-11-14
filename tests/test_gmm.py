import numpy as np
import torch

from flowgate.gmm import TiedGMMScorer


def _fit():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(m, 1.0, (100, 6)) for m in (-2.0, 0.0, 3.0)])
    return GaussianMixture(3, covariance_type="tied", random_state=0).fit(z), rng.normal(size=(20, 6))


def test_torch_scorer_matches_sklearn_gmm_ratio_score():
    gm, q = _fit()
    want = gm.score_samples(q) + 0.5 * (q**2).sum(1)  # exactly the baseline's gmm150_ratio score
    got = TiedGMMScorer.from_sklearn(gm)(torch.from_numpy(q))
    np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=1e-8)


def test_params_round_trip_and_float32_copy_stays_close():
    gm, q = _fit()
    s = TiedGMMScorer.from_sklearn(gm)
    again = TiedGMMScorer.from_params(s.params())
    q64 = torch.from_numpy(q)
    assert torch.equal(s(q64), again(q64))
    f32 = TiedGMMScorer.from_params(s.params(), dtype=torch.float32)(q64.float())
    assert f32.dtype == torch.float32 and torch.allclose(f32.double(), s(q64), atol=1e-3)
