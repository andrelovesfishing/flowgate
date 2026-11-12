import math

import torch

from flowgate.likelihood import ode_logp
from flowgate.net import FlowMapNet
from flowgate.train import TrainConfig, train_velocity, val_cfm_loss


def _gauss(n, seed, sig=2.0, d=2):
    return sig * torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def test_learned_velocity_gives_correct_gaussian_likelihood():
    sig, d = 2.0, 2
    torch.manual_seed(0)
    net = FlowMapNet(d=d, width=64, depth=3)
    # 2000 steps gave 0.12 (learning error, not integration: 32 vs 128 ODE steps agree); 6000 gives ~0.07
    cfg = TrainConfig(steps=6000, batch=256, lr=2e-3, eval_every=500, patience=100, seed=0)
    net, _ = train_velocity(net, _gauss(4000, 0), _gauss(500, 1), cfg)
    x = _gauss(200, 2)
    truth = -0.5 * (x**2).sum(1) / sig**2 - d * math.log(sig) - 0.5 * d * math.log(2 * math.pi)
    err = (ode_logp(net.velocity, x, steps=32) - truth).abs().mean()
    assert err < 0.1, f"mean |log p error| = {err:.3f}"


def test_returns_the_best_validation_checkpoint():
    torch.manual_seed(0)
    net = FlowMapNet(d=2, width=32, depth=2)
    cfg = TrainConfig(steps=600, batch=128, eval_every=100, patience=100, seed=0)
    zv = _gauss(300, 1)
    net, hist = train_velocity(net, _gauss(1000, 0), zv, cfg)
    assert abs(val_cfm_loss(net, zv, cfg.seed) - min(h["val"] for h in hist)) < 1e-6


def test_same_seed_same_weights():
    def run():
        torch.manual_seed(0)
        net = FlowMapNet(d=2, width=32, depth=2)
        cfg = TrainConfig(steps=50, batch=64, eval_every=25, seed=3)
        return train_velocity(net, _gauss(500, 0), _gauss(100, 1), cfg)[0]

    a, b = run(), run()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)
