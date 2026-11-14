import math

import torch

from flowgate.likelihood import exact_div, ode_logp, std_normal_logpdf


def test_divergence_of_linear_field_is_trace():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(4, 4, generator=g)
    x = torch.randn(10, 4, generator=g)
    div = exact_div(lambda x, t: x @ A.T, x, torch.zeros(10))
    torch.testing.assert_close(div, torch.full((10,), A.trace().item()))


def test_divergence_of_module_works_under_no_grad():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 3)
    x = torch.randn(7, 3)
    with torch.no_grad():
        div = exact_div(lambda x, t: lin(x), x, torch.zeros(7), chunk=3)
    torch.testing.assert_close(div, torch.full((7,), lin.weight.trace().item()))


def test_zero_field_gives_standard_normal_logpdf():
    x = torch.randn(6, 3)
    logp = ode_logp(lambda x, t: torch.zeros_like(x), x, steps=4)
    torch.testing.assert_close(logp, std_normal_logpdf(x))


def test_recovers_gaussian_data_logpdf_from_analytic_velocity():
    """Data N(0, sig^2 I), linear path from N(0, I): marginal variance (1-t)^2 + t^2 sig^2 and
    v(x, t) = (t sig^2 - (1 - t)) / ((1 - t)^2 + t^2 sig^2) * x. The ODE likelihood must match."""
    sig, d = 2.0, 3

    def vel(x, t):
        c = (t * sig**2 - (1 - t)) / ((1 - t) ** 2 + t**2 * sig**2)
        return c[:, None] * x

    x1 = sig * torch.randn(20, d, generator=torch.Generator().manual_seed(1))
    truth = -0.5 * (x1**2).sum(1) / sig**2 - d * math.log(sig) - 0.5 * d * math.log(2 * math.pi)
    torch.testing.assert_close(ode_logp(vel, x1, steps=64), truth, atol=1e-3, rtol=0)


def _gauss_vel(sig):
    def vel(x, t):
        c = (t * sig**2 - (1 - t)) / ((1 - t) ** 2 + t**2 * sig**2)
        return c[:, None] * x
    return vel


def test_trajectory_ends_agree_with_ode_logp_exactly():
    from flowgate.likelihood import ode_trajectory
    vel = _gauss_vel(2.0)
    x1 = 2.0 * torch.randn(15, 3, generator=torch.Generator().manual_seed(4))
    xs, L = ode_trajectory(vel, x1, steps=8)
    assert xs.shape == (9, 15, 3) and L.shape == (9, 15)
    assert torch.equal(xs[0], x1) and torch.equal(L[0], torch.zeros(15))
    torch.testing.assert_close(std_normal_logpdf(xs[-1]) + L[-1], ode_logp(vel, x1, steps=8), atol=0, rtol=0)


def test_trajectory_log_density_increments_match_analytic_marginals():
    """log p_{t_j}(x_j) - log p_{t_i}(x_i) = -(L[j] - L[i]); marginal at t is N(0, (1-t)^2 + t^2 sig^2)."""
    from flowgate.likelihood import ode_trajectory
    sig, d, steps = 2.0, 3, 64
    x1 = sig * torch.randn(10, d, generator=torch.Generator().manual_seed(5))
    xs, L = ode_trajectory(_gauss_vel(sig), x1, steps=steps)

    def logp(x, t):
        v = (1 - t) ** 2 + t**2 * sig**2
        return -0.5 * (x**2).sum(1) / v - 0.5 * d * math.log(2 * math.pi * v)

    i, j = 16, 48  # t = 0.75 -> t = 0.25
    torch.testing.assert_close(logp(xs[j], 1 - j / steps) - logp(xs[i], 1 - i / steps), -(L[j] - L[i]),
                               atol=1e-3, rtol=0)
