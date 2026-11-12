"""R1 training: conditional flow matching for the velocity u(x, t, t).

Pair each data point x1 with fresh noise x0 ~ N(0, I) and a time t ~ U(0, 1), form
x_t = (1 - t) x0 + t x1, and regress u(x_t, t, t) onto x1 - x0. The minimiser is the marginal
velocity whose ODE carries N(0, I) onto the data distribution.

Early stopping uses the same loss on held-out in-scope embeddings with *fixed* noise and times,
so values are comparable across evaluations. No OOS data is involved.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import torch

from .net import FlowMapNet


@dataclass
class TrainConfig:
    steps: int = 10_000
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 500
    patience: int = 6  # evaluations without improvement before stopping
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def cfm_loss(net: FlowMapNet, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    return ((net.velocity(xt, t) - (x1 - x0)) ** 2).mean()


def val_cfm_loss(net: FlowMapNet, zval: torch.Tensor, seed: int) -> float:
    g = torch.Generator().manual_seed(10_000 + seed)
    x0 = torch.randn(zval.shape, generator=g)
    t = torch.rand(len(zval), generator=g)
    with torch.no_grad():
        return float(cfm_loss(net, zval, x0, t))


def train_velocity(net: FlowMapNet, ztr: torch.Tensor, zval: torch.Tensor, cfg: TrainConfig, log=None):
    torch.use_deterministic_algorithms(True)
    g = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, since_best, hist = float("inf"), copy.deepcopy(net.state_dict()), 0, []
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(len(ztr), (cfg.batch,), generator=g)
        x1 = ztr[idx]
        loss = cfm_loss(net, x1, torch.randn(x1.shape, generator=g), torch.rand(cfg.batch, generator=g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            v = val_cfm_loss(net, zval, cfg.seed)
            hist.append({"step": step, "train": loss.item(), "val": v, "sec": time.perf_counter() - t0})
            if log:
                h = hist[-1]
                log(f"  step {step:>6}/{cfg.steps}  train {h['train']:.4f}  val {v:.4f}  {h['sec']:.0f}s")
            if v < best:
                best, best_state, since_best = v, copy.deepcopy(net.state_dict()), 0
            else:
                since_best += 1
                if since_best >= cfg.patience:
                    break
    net.load_state_dict(best_state)
    return net, hist
