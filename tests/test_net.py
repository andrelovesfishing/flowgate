import torch

from flowgate.net import FlowMapNet


def _batch(B=8, d=5):
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, d, generator=g), torch.rand(B, generator=g), torch.rand(B, generator=g)


def test_heads_start_at_zero():
    x, t, s = _batch()
    u, D = FlowMapNet(d=5, width=32, depth=2)(x, t, s)
    assert u.shape == (8, 5) and D.shape == (8,)
    assert torch.all(u == 0) and torch.all(D == 0)


def test_flow_map_with_zero_length_jump_is_identity_even_after_training_step():
    torch.manual_seed(0)
    net = FlowMapNet(d=5, width=32, depth=2)
    x, t, s = _batch()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    y, dlogp = net.flow_map(x, t, t)
    torch.testing.assert_close(y, x)
    assert torch.all(dlogp == 0)


def test_velocity_is_u_on_the_diagonal():
    torch.manual_seed(0)
    net = FlowMapNet(d=5, width=32, depth=2)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x, t, _ = _batch()
    torch.testing.assert_close(net.velocity(x, t), net(x, t, t)[0])
