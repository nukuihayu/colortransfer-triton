import numpy as np
import pytest
import torch

import colortransfer_triton as ct
from colortransfer_triton.colorfm import _couple, _loss


def velocity(x, t, w1, w2):
    z = np.concatenate((x, np.full((len(x), 1), t)), 1) @ w1.T
    return (z / (1 + np.exp(-z))) @ w2.T


def midpoint(x, w1, w2, steps, horizon):
    x = x.astype(np.float64).copy()
    dt = horizon / steps
    for step in range(steps):
        v = velocity(x, step * dt, w1, w2)
        x += dt * velocity(x + dt * v / 2, (step + 0.5) * dt, w1, w2)
    return x


def test_colorfm_loss_and_gradient():
    rng = np.random.default_rng(8)
    pairs = torch.tensor(rng.random((7, 2, 3)), dtype=torch.float64)
    times = torch.linspace(0, 1, 7, dtype=torch.float64)
    a = torch.tensor(rng.normal(0, 0.2, (9, 4)), requires_grad=True)
    b = torch.tensor(rng.normal(0, 0.2, (3, 9)), requires_grad=True)
    loss = _loss(pairs, times, a, b)
    x, y = pairs.numpy()[:, 0], pairs.numpy()[:, 1]
    t = times.numpy()
    z = (
        np.concatenate(((1 - t[:, None]) * x + t[:, None] * y, t[:, None]), 1)
        @ a.detach().numpy().T
    )
    pred = (z / (1 + np.exp(-z))) @ b.detach().numpy().T
    expected = (((y - x - pred) ** 2).sum(1) / (1e-4 + np.linalg.norm(y - x, axis=1))).mean()
    np.testing.assert_allclose(loss.detach(), expected, atol=1e-12)
    assert torch.autograd.gradcheck(lambda w1, w2: _loss(pairs, times, w1, w2), (a, b))


def test_colorfm_octant_coupling():
    x = torch.cartesian_prod(*[torch.tensor([0.2, 0.7])] * 3)
    y = x * 0.8 + 0.1
    pairs = _couple(x, y, 3, 7)
    assert pairs.shape == (8, 2, 3)
    torch.testing.assert_close(pairs[:, 1], pairs[:, 0] * 0.8 + 0.1)
    # Constant and unequal clouds reach the same leaf and keep min(M,N) pairs.
    constant = _couple(torch.full((9, 3), 0.2), torch.full((13, 3), 0.6), 3, 7)
    assert constant.shape == (9, 2, 3)
    torch.testing.assert_close(constant[:, 1] - constant[:, 0], torch.full((9, 3), 0.4))


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")


@gpu
@pytest.mark.parametrize(
    "hidden,steps,horizon", [(8, 1, 1.0), (19, 5, 0.35), (64, 11, 1.0), (512, 5, 1.0), (32, 5, 0.0)]
)
def test_colorfm_integrator_reference(hidden, steps, horizon):
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.2, (hidden, 4)).astype(np.float32)
    b = rng.normal(0, 0.1, (3, hidden)).astype(np.float32)
    x = rng.random((47, 3), dtype=np.float32)
    flow = ct.ColorFlow(torch.from_numpy(a[None]).cuda(), torch.from_numpy(b[None]).cuda(), ())
    actual = flow.transform_points(torch.from_numpy(x).cuda(), ode_steps=steps, time=horizon)
    np.testing.assert_allclose(
        actual.cpu(),
        midpoint(x, a.astype(np.float64), b.astype(np.float64), steps, horizon),
        atol=5e-6,
        rtol=5e-6,
    )


@gpu
def test_colorfm_fit_seed_batch_inference_mode():
    rng = torch.Generator(device="cuda").manual_seed(5)
    x = torch.rand(2, 3, 11, 13, device="cuda", generator=rng)
    r = x[:1] * 0.7 + 0.15
    state_cpu = torch.random.get_rng_state().clone()
    state_cuda = torch.cuda.get_rng_state().clone()
    opts = dict(samples=128, steps=80, hidden=32, batch_size=128, learning_rate=0.01, seed=3)
    with torch.inference_mode():
        a = ct.fit_colorfm_model(x, r, **opts)
    b = ct.fit_colorfm_model(x, r, **opts)
    torch.testing.assert_close(a.input_weight, b.input_weight, atol=0, rtol=0)
    torch.testing.assert_close(a.output_weight, b.output_weight, atol=0, rtol=0)
    assert torch.equal(state_cpu, torch.random.get_rng_state())
    assert torch.equal(state_cuda, torch.cuda.get_rng_state())
    for d in a.diagnostics:
        assert d["final_loss"] < d["initial_loss"]
    table = a.to_lut(lut_size=5)
    actual = table(x)
    assert actual.shape == x.shape and torch.isfinite(actual).all()
    assert actual.min() >= 0 and actual.max() <= 1
    torch.testing.assert_close(table(x, strength=0), x, atol=0, rtol=0)


@gpu
def test_colorfm_dispatch_and_validation():
    x = torch.full((1, 3, 7, 9), 0.3, device="cuda", dtype=torch.float16)
    r = torch.full((1, 3, 9, 7), 0.5, device="cuda")
    for method in (ct.colorfm, lambda x, r, **kw: ct.transfer(x, r, method="colorfm", **kw)):
        result = method(x, r, steps=2, samples=16, hidden=8, batch_size=16, lut_size=3)
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
    table = ct.fit_lut(x, r, method="colorfm", iterations=2, samples=16, lut_size=3)
    assert table.method == "colorfm"
    for kw in (
        {"steps": 0},
        {"samples": 0},
        {"hidden": 2048},
        {"learning_rate": 0},
        {"ode_steps": 0},
    ):
        with pytest.raises(ValueError):
            ct.fit_colorfm(x, r, **kw)


@gpu
def test_colorfm_default_device_and_autocast():
    x = torch.rand(1, 3, 5, 7, device="cuda")
    with torch.device("cuda"), torch.autocast("cuda", dtype=torch.float16):
        flow = ct.fit_colorfm_model(x, x, samples=16, steps=2, hidden=8, batch_size=16)
    assert flow.input_weight.dtype == torch.float32
    assert torch.isfinite(flow.output_weight).all()
