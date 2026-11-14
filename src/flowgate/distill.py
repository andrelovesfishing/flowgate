"""R2: distil the slow exact-ODE teacher into a few-step flow map with a log-density head.

The teacher (R1) only gives log-densities through a 32-step ODE with an exact Jacobian trace per
step, which costs about d backward passes each. Its whole path is recorded once (`record`). The
student is the same FlowMapNet, started from the teacher's weights, trained by plain regression to
reproduce jumps along those paths:
    Phi(x_i, t_i, t_j)          -> x_j                                   where the point lands
    (t_j - t_i) D(x_i, t_i, t_j) -> log p_{t_j}(x_j) - log p_{t_i}(x_i)   how its log-density changes
Jumps are dyadic (1/32, 1/16, ..., 1 of the way) and aligned, exactly the jumps used when scoring
with 32, 16, ..., 1 steps. No Jacobians are needed for training or scoring.

Coverage: a gate is queried on out-of-scope points, which lie off the training cloud. So paths also
start from label-free off-manifold points: N(0, I) samples, noise-jittered train points, and blends
of two train points. No OOS data is used anywhere.

mode="direct" is the ablation: the same net and data, trained to output the final score in one shot
(no flow structure). It answers "why a flow map rather than a plain regressor?".
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .likelihood import ode_trajectory, std_normal_logpdf

KINDS = ("train", "gauss", "jitter", "blend")


def make_starts(z: np.ndarray, holdout_idx: np.ndarray, n_off: int, seed: int = 0):
    """Distillation start points: every train point (kind 0, held out where the teacher held it out),
    then n_off each of N(0, I) samples, jittered train points (noise sd U(0.25, 1)) and blends of two
    train points (weight U(0.2, 0.8)). The last 10% of each off-manifold kind is held out."""
    rng = np.random.default_rng(seed)
    n, d = z.shape
    gauss = rng.standard_normal((n_off, d))
    jitter = z[rng.integers(n, size=n_off)] + rng.uniform(0.25, 1.0, (n_off, 1)) * rng.standard_normal((n_off, d))
    a, b, lam = rng.integers(n, size=n_off), rng.integers(n, size=n_off), rng.uniform(0.2, 0.8, (n_off, 1))
    blend = lam * z[a] + (1 - lam) * z[b]
    X = np.concatenate([z, gauss, jitter, blend]).astype(np.float32)
    kind = np.repeat(np.arange(len(KINDS), dtype=np.int8), [n, n_off, n_off, n_off])
    hold = np.zeros(len(X), bool)
    hold[np.asarray(holdout_idx)] = True
    n_h = int(round(0.1 * n_off))
    for k in range(1, len(KINDS)):
        idx = np.flatnonzero(kind == k)
        hold[idx[len(idx) - n_h:]] = True
    return X, kind, hold


def record(vel, X, steps: int = 32, block: int = 1000, log=None):
    """The teacher's exact paths for every start, in blocks (with progress). Returns xs, L as in
    ode_trajectory: xs [steps + 1, N, d], L [steps + 1, N]."""
    X = torch.as_tensor(X, dtype=torch.float32)
    xs, Ls, t0 = [], [], time.perf_counter()
    for b in range(0, len(X), block):
        x, L = ode_trajectory(vel, X[b : b + block], steps=steps)
        xs.append(x); Ls.append(L)
        if log:
            done, sec = min(b + block, len(X)), time.perf_counter() - t0
            log(f"  recorded {done}/{len(X)} paths  {sec:.0f}s  (~{sec / done * (len(X) - done) / 60:.1f} min left)")
    return torch.cat(xs, 1), torch.cat(Ls, 1)


@dataclass
class Trajectories:
    xs: torch.Tensor      # [T + 1, N, d], xs[k] at t = 1 - k / T
    L: torch.Tensor       # [T + 1, N], integral from 1 to t_k of div v
    kind: np.ndarray      # [N] index into KINDS
    holdout: np.ndarray   # [N] bool

    @property
    def steps(self) -> int:
        return self.xs.shape[0] - 1

    def subset(self, mask: np.ndarray) -> "Trajectories":
        idx = torch.from_numpy(np.flatnonzero(mask))
        return Trajectories(self.xs[:, idx], self.L[:, idx], self.kind[idx.numpy()], self.holdout[idx.numpy()])

    def teacher_score(self) -> torch.Tensor:
        """The teacher's gate score for each start: log p_1(x) - log N(x; 0, I)."""
        return std_normal_logpdf(self.xs[-1]) + self.L[-1] - std_normal_logpdf(self.xs[0])

    def save(self, path) -> None:
        np.savez(path, xs=self.xs.numpy(), L=self.L.numpy(), kind=self.kind, holdout=self.holdout)

    @classmethod
    def load(cls, path) -> "Trajectories":
        with np.load(path) as f:
            return cls(torch.from_numpy(f["xs"]), torch.from_numpy(f["L"]), f["kind"], f["holdout"])


def sample_jumps(n_pool: int, steps: int, batch: int, g: torch.Generator):
    """Random (path n, grid start i, grid end j): length j - i = 2^m, start aligned to a multiple of it."""
    assert steps & (steps - 1) == 0, "grid must have a power-of-two number of steps"
    levels = steps.bit_length()  # lengths 1, 2, ..., steps
    length = 2 ** torch.randint(levels, (batch,), generator=g)
    i = (torch.rand(batch, generator=g) * (steps // length)).floor().long() * length
    n = torch.randint(n_pool, (batch,), generator=g)
    return n, i, i + length


class Student(nn.Module):
    """A FlowMapNet with a fixed affine output scale on D, so its raw head output is O(1)
    (log-density rates in d = 128 are in the hundreds; a zero-initialised head can't get there fast)."""

    def __init__(self, net, d_shift: float = 0.0, d_scale: float = 1.0):
        super().__init__()
        self.net = net
        self.register_buffer("d_shift", torch.tensor(float(d_shift)))
        self.register_buffer("d_scale", torch.tensor(float(d_scale)))

    def _heads(self, x, t, s):
        u, D = self.net(x, t, s)
        return u, self.d_shift + self.d_scale * D

    def jump(self, x, t, s):
        """(where x lands at time s, log p_s(landing) - log p_t(x))."""
        u, D = self._heads(x, t, s)
        step = s - t
        return x + step[:, None] * u, step * D

    def direct(self, x):
        t = torch.ones(len(x), dtype=x.dtype)
        return self._heads(x, t, t)[1]


@torch.no_grad()
def student_score(st: Student, z: torch.Tensor, k: int, mode: str = "flowmap") -> torch.Tensor:
    """Gate score log p_1(z) - log N(z; 0, I) from k equal flow-map jumps 1 -> 0 (or one shot, direct)."""
    if mode == "direct":
        return st.direct(z)
    x, total = z, torch.zeros(len(z), dtype=z.dtype)
    for a in range(k):
        t, s = torch.full((len(z),), 1 - a / k), torch.full((len(z),), 1 - (a + 1) / k)
        x, dl = st.jump(x, t, s)
        total = total + dl
    # total = log p_0(x_0) - log p_1(z)
    return std_normal_logpdf(x) - total - std_normal_logpdf(z)


def val_error(st: Student, va: Trajectories, mode: str, ks=(1, 2, 4, 8)) -> dict[int, float]:
    """Mean |student score - teacher score| on held-out starts, per number of jumps k."""
    truth, x = va.teacher_score(), va.xs[0]
    return {k: float((student_score(st, x, k, mode) - truth).abs().mean()) for k in (ks if mode == "flowmap" else (1,))}


@dataclass
class DistillConfig:
    steps: int = 10_000
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_every: int = 500
    patience: int = 6
    seed: int = 0
    mode: str = "flowmap"   # or "direct" (ablation)
    starts: str = "all"     # or "train" (ablation: in-scope paths only)
    val_ks: tuple = (1, 2, 4, 8)

    def to_dict(self) -> dict:
        return asdict(self)


def _jump_stats(tr: Trajectories):
    """Mean/sd of the log-density rate D over all aligned jumps (sets the output scale), and the
    spread of the log-density change per jump length (weights the loss so every length counts
    equally). The spread is floored at 1 nat, sqrt(var + 1): when paths barely differ (e.g. a linear
    teacher, where they don't differ at all), errors are then measured in plain nats instead of
    being blown up by ~1/0."""
    T, rates, sd = tr.steps, [], []
    for m in range(T.bit_length()):
        length = 2**m
        Lal = tr.L[::length]
        dl = -(Lal[1:] - Lal[:-1])
        rates.append((dl / (-length / T)).flatten())
        sd.append((dl.var() + 1.0).sqrt())
    r = torch.cat(rates)
    return float(r.mean()), float(r.std().clamp_min(1e-6)), torch.stack(sd)


def train_student(st: Student, tr: Trajectories, va: Trajectories, cfg: DistillConfig, log=None):
    torch.use_deterministic_algorithms(True)
    g = torch.Generator().manual_seed(cfg.seed)
    T, N = tr.steps, tr.xs.shape[1]
    if cfg.mode == "flowmap":
        shift, scale, sd_len = _jump_stats(tr)
    else:
        target = tr.teacher_score()
        shift, scale = float(target.mean()), float(target.std().clamp_min(1e-6))
    st.d_shift.fill_(shift); st.d_scale.fill_(scale)
    opt = torch.optim.AdamW(st.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, since_best, hist, t0 = float("inf"), copy.deepcopy(st.state_dict()), 0, [], time.perf_counter()
    for step in range(1, cfg.steps + 1):
        if cfg.mode == "flowmap":
            n, i, j = sample_jumps(N, T, cfg.batch, g)
            phi, dl = st.jump(tr.xs[i, n], 1 - i / T, 1 - j / T)
            dl_t = -(tr.L[j, n] - tr.L[i, n])
            m = torch.log2((j - i).float()).round().long()
            loss = ((phi - tr.xs[j, n]) ** 2).mean() + (((dl - dl_t) / sd_len[m]) ** 2).mean()
        else:
            n = torch.randint(N, (cfg.batch,), generator=g)
            loss = (((st.direct(tr.xs[0, n]) - target[n]) / scale) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            e = val_error(st, va, cfg.mode, cfg.val_ks)
            v = float(np.mean(list(e.values())))
            hist.append({"step": step, "train": loss.item(), "val": v, "val_k": e, "sec": time.perf_counter() - t0})
            if log:
                ks = "  ".join(f"k{k} {x:.3f}" for k, x in e.items())
                log(f"  step {step:>6}/{cfg.steps}  train {loss.item():.4f}  val |score err| {ks}  {hist[-1]['sec']:.0f}s")
            if v < best:
                best, best_state, since_best = v, copy.deepcopy(st.state_dict()), 0
            else:
                since_best += 1
                if since_best >= cfg.patience:
                    break
    st.load_state_dict(best_state)
    return st, hist
