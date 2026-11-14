"""The GMM-150 baseline's score, log p_GMM(z) - log N(z; 0, I) up to a constant, as a torch scorer.

sklearn's score_samples validates and copies its input on every call, which at batch 1 would time
sklearn's overhead rather than the model. With a tied covariance the score needs one d x d matmul
and one K x d matmul:
    y = z P,  m_k = mu_k P              (precision = P P^T)
    log p(z) = logsumexp_k(c_k + y . m_k) - |y|^2 / 2,
    c_k = log w_k - |m_k|^2 / 2 + log det P - (d / 2) log 2 pi
and the ratio score adds |z|^2 / 2, exactly as `cli.baselines` does.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class TiedGMMScorer:
    def __init__(self, means: np.ndarray, prec_chol: np.ndarray, weights: np.ndarray, dtype=torch.float64):
        self.means, self.prec_chol, self.weights = means, prec_chol, weights
        P = torch.as_tensor(prec_chol, dtype=torch.float64)
        m = torch.as_tensor(means, dtype=torch.float64) @ P
        d = P.shape[0]
        c = (torch.log(torch.as_tensor(weights, dtype=torch.float64)) - 0.5 * (m**2).sum(1)
             + torch.log(torch.diagonal(P)).sum() - 0.5 * d * math.log(2 * math.pi))
        self.P, self.mT, self.c = P.to(dtype), m.T.contiguous().to(dtype), c.to(dtype)

    @classmethod
    def from_sklearn(cls, gm, dtype=torch.float64) -> "TiedGMMScorer":
        assert gm.covariance_type == "tied"
        return cls(gm.means_, gm.precisions_cholesky_, gm.weights_, dtype)

    def params(self) -> dict[str, np.ndarray]:
        return {"means": self.means, "prec_chol": self.prec_chol, "weights": self.weights}

    @classmethod
    def from_params(cls, p: dict, dtype=torch.float64) -> "TiedGMMScorer":
        return cls(p["means"], p["prec_chol"], p["weights"], dtype)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        y = z @ self.P
        return torch.logsumexp(self.c + y @ self.mT, dim=1) - 0.5 * (y**2).sum(1) + 0.5 * (z**2).sum(1)
