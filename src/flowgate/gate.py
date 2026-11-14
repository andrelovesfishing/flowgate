"""Label-free scope gate: log p_flow(z) - log N(z; 0, I).

The flow is fitted to unlabeled in-scope train embeddings. The score asks how much more likely a
query is under the flexible density than under one Gaussian fitted to the same data, so it
cancels the "generic sentence" directions that a raw likelihood is dominated by. Higher = act.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .data import SplitData
from .likelihood import ode_logp, std_normal_logpdf
from .net import FlowMapNet
from .space import Space
from .train import TrainConfig, train_velocity


def early_stop_split(n: int, seed: int, frac: float = 0.1):
    """(train indices, held-out indices) for early stopping. Taken from train, so val stays clean;
    shared with R2 so the student is validated on exactly the points the teacher never trained on."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_es = max(1, int(frac * n))
    return perm[n_es:], perm[:n_es]


class FlowRatioGate:
    def __init__(self, space_key: str = "frozen", d: int = 128, kind: str = "pca", width: int = 512,
                 depth: int = 4, ode_steps: int = 32, train_cfg: TrainConfig | None = None,
                 early_stop_frac: float = 0.1):
        self.space_key, self.d, self.kind = space_key, d, kind
        self.width, self.depth, self.ode_steps = width, depth, ode_steps
        self.train_cfg = train_cfg or TrainConfig()
        self.early_stop_frac = early_stop_frac
        self.name = f"flowratio-{kind}{d}"

    def fit(self, train: SplitData, ref: SplitData | None = None, log=None) -> "FlowRatioGate":
        if (np.asarray(train.labels) < 0).any():
            raise ValueError("fit on in-scope examples only")
        X = np.asarray(train.emb[self.space_key])
        self.space = Space.fit(X, self.d, self.kind)
        z = torch.from_numpy(self.space.transform(X))
        # early-stopping hold-out comes from train, so the val split stays clean for selection
        tr_idx, es_idx = early_stop_split(len(z), self.train_cfg.seed, self.early_stop_frac)
        torch.manual_seed(self.train_cfg.seed)
        self.net = FlowMapNet(self.d, self.width, self.depth)
        t0 = time.perf_counter()
        self.net, self.history = train_velocity(self.net, z[tr_idx], z[es_idx], self.train_cfg, log=log)
        self.fit_seconds = time.perf_counter() - t0
        self.net.eval()
        return self

    def _kwargs(self) -> dict:
        return {"space_key": self.space_key, "d": self.d, "kind": self.kind, "width": self.width,
                "depth": self.depth, "ode_steps": self.ode_steps, "early_stop_frac": self.early_stop_frac}

    def save(self, path) -> None:
        torch.save({"kwargs": self._kwargs(), "train_cfg": self.train_cfg.to_dict(),
                    "space": {"mu": torch.from_numpy(self.space.mu), "A": torch.from_numpy(self.space.A),
                              "kind": self.space.kind},
                    "net": self.net.state_dict(), "history": self.history, "fit_seconds": self.fit_seconds}, path)

    @classmethod
    def load(cls, path) -> "FlowRatioGate":
        ck = torch.load(path, weights_only=True)
        g = cls(**ck["kwargs"], train_cfg=TrainConfig(**ck["train_cfg"]))
        sp = ck["space"]
        g.space = Space(mu=sp["mu"].numpy(), A=sp["A"].numpy(), kind=sp["kind"])
        g.net = FlowMapNet(g.d, g.width, g.depth)
        g.net.load_state_dict(ck["net"])
        g.net.eval()
        g.history, g.fit_seconds = ck["history"], ck["fit_seconds"]
        return g

    def score(self, d: SplitData) -> np.ndarray:
        z = torch.from_numpy(self.space.transform(np.asarray(d.emb[self.space_key])))
        logp = ode_logp(self.net.velocity, z, steps=self.ode_steps)
        return (logp - std_normal_logpdf(z)).double().numpy()
