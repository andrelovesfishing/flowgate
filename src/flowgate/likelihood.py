"""Exact ODE likelihood: the oracle every fast approximation is checked against.

Convention: t = 0 is noise N(0, I), t = 1 is data, dx/dt = v(x, t). Along a trajectory
    d log p_t(x_t) / dt = -div v(x_t, t)
so for a data point x1, integrating back from t = 1 to t = 0 gives
    log p_1(x1) = log N(x0; 0, I) - integral_0^1 div v(x_t, t) dt.
The divergence is the exact trace of the Jacobian (d backward passes, vectorised). That is why
this is slow, and why the few-step flow map exists.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch.func import jacrev, vmap

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x [B, d], t [B]) -> [B, d]


def std_normal_logpdf(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x**2).sum(1) - 0.5 * x.shape[1] * math.log(2 * math.pi)


def exact_div(vel: Velocity, x: torch.Tensor, t: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Trace of dv/dx for each row of x. Works inside torch.no_grad (torch.func ignores it)."""

    def f(xi, ti):
        return vel(xi[None], ti[None])[0]

    out = []
    for i in range(0, len(x), chunk):
        J = vmap(jacrev(f))(x[i : i + chunk], t[i : i + chunk])
        out.append(J.diagonal(dim1=-2, dim2=-1).sum(-1))
    return torch.cat(out)


def ode_logp(vel: Velocity, x1: torch.Tensor, steps: int = 32, chunk: int = 256) -> torch.Tensor:
    """log p_1(x1) by the midpoint rule on the joint (x, log-density) ODE, from t = 1 to t = 0."""
    with torch.no_grad():
        x = x1.clone()
        acc = torch.zeros(len(x), dtype=x.dtype)
        h = -1.0 / steps
        for k in range(steps):
            t = torch.full((len(x),), 1.0 + k * h, dtype=x.dtype)
            tm = t + h / 2
            xm = x + (h / 2) * vel(x, t)
            acc = acc + h * exact_div(vel, xm, tm, chunk)
            x = x + h * vel(xm, tm)
        # acc = integral from 1 to 0 of div v dt = -integral_0^1 div v dt
        return std_normal_logpdf(x) + acc
