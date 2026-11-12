"""One network for the whole project: a flow map with a log-density head.

    Phi(x, t, s)  = x + (s - t) * u(x, t, s)      where the sample goes from time t to time s
    dlogp(x, t, s) = (s - t) * D(x, t, s)          how log-density changes along that jump

At s = t the jump has zero length, so u(x, t, t) is the instantaneous velocity (what plain flow
matching learns, R1) and D(x, t, t) is the instantaneous divergence term (R2/R3). Both heads start
at zero, so an untrained net is the identity map.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class FlowMapNet(nn.Module):
    def __init__(self, d: int, width: int = 512, depth: int = 4, n_freq: int = 8):
        super().__init__()
        self.d = d
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        layers: list[nn.Module] = [nn.Linear(d + 4 * n_freq, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.u_head = nn.Linear(width, d)
        self.D_head = nn.Linear(width, 1)
        for head in (self.u_head, self.D_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _time(self, t: torch.Tensor) -> torch.Tensor:
        f = t[:, None] * self.freqs
        return torch.cat([f.sin(), f.cos()], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        h = self.trunk(torch.cat([x, self._time(t), self._time(s)], dim=1))
        return self.u_head(h), self.D_head(h)[:, 0]

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self(x, t, t)[0]

    def flow_map(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        u, D = self(x, t, s)
        step = s - t
        return x + step[:, None] * u, step * D
