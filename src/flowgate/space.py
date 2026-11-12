"""The d-dim space the flow models.

`transform` is linear and set up so that the *background* Gaussian (one Gaussian fitted to all
in-scope train embeddings) is exactly N(0, I) in z. The full model is "flow on z, background
Gaussian on the dropped dims", so in the score log p_flow - log p_background the dropped dims
cancel exactly and only z matters.

kinds:
  pca  - top-d principal components of the raw embeddings, each divided by its std.
         Background = Gaussian with the sample covariance.
  wpca - whiten with the Ledoit-Wolf covariance, then rotate onto the principal axes of the
         whitened train data and keep d. Background = Gaussian with the Ledoit-Wolf covariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class Space:
    mu: np.ndarray  # [D]
    A: np.ndarray  # [D, d]; z = (x - mu) @ A
    kind: str

    @classmethod
    def fit(cls, X: np.ndarray, d: int, kind: str = "pca") -> "Space":
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(0)
        if kind == "pca":
            _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            sd = S / np.sqrt(len(X) - 1)
            A = Vt[:d].T / sd[:d]
        elif kind == "wpca":
            L = np.linalg.cholesky(LedoitWolf().fit(X).precision_)
            W = (X - mu) @ L
            _, _, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
            A = L @ Vt[:d].T
        else:
            raise ValueError(f"unknown space kind {kind!r}")
        return cls(mu=mu, A=A, kind=kind)

    @property
    def d(self) -> int:
        return self.A.shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mu) @ self.A).astype(np.float32)
