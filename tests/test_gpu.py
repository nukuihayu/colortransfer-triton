import numpy as np
import pytest
import reference
import torch

import colortransfer_triton as ct
from colortransfer_triton.ot import _barycentric, _directions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")


def test_explicit_balanced_mass_and_default_dtype():
    x = torch.rand(1, 3, 8, 9, device="cuda")
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for method in ("sliced_ot", "sinkhorn", "partial_sinkhorn"):
            lut = ct.fit_lut(x, x, method=method, samples=16, lut_size=3, iterations=3, mass=1)
            assert lut.values.dtype == torch.float32
            assert torch.isfinite(lut(x)).all()
            if lut.diagnostics:
                torch.testing.assert_close(
                    lut.diagnostics[0]["transported_mass"],
                    torch.ones((), device="cuda", dtype=torch.float32),
                )
    finally:
        torch.set_default_dtype(old)


def data(shape, seed=0):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize(
    "shape,rshape",
    [
        ((2, 3, 17, 29), (1, 3, 13, 7)),
        ((1, 3, 1, 1), (1, 3, 1, 1)),
        ((2, 3, 67, 73), (2, 3, 19, 23)),
    ],
)
def test_adain_reference(shape, rshape, correction):
    x = data(shape)
    r = data(rshape, 7)
    y = ct.adain(
        torch.from_numpy(x).cuda(), torch.from_numpy(r).cuda(), correction=correction, clamp=False
    )
    np.testing.assert_allclose(
        y.cpu(), reference.adain(x, r, correction=correction), atol=3e-6, rtol=3e-6
    )


@pytest.mark.parametrize("levels", [0, 1, 3, 5])
@pytest.mark.parametrize("shape", [(2, 3, 17, 23), (1, 3, 1, 7), (1, 3, 1, 1)])
def test_wavelet_reference(shape, levels):
    x = data(shape)
    r = data((1, 3, *shape[-2:]), 6)
    y = ct.wavelet(
        torch.from_numpy(x).cuda(), torch.from_numpy(r).cuda(), levels=levels, clamp=False
    )
    np.testing.assert_allclose(y.cpu(), reference.wavelet(x, r, levels), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "method", ["adain", "wavelet", "sliced_ot", "sinkhorn", "partial_sinkhorn"]
)
def test_dtypes_layout_strength(method, dtype):
    x = torch.rand(1, 3, 11, 13, device="cuda", dtype=dtype).transpose(2, 3)
    r = torch.rand_like(x)
    kwargs = (
        {} if method in ("adain", "wavelet") else {"samples": 24, "lut_size": 5, "iterations": 12}
    )
    out = ct.transfer(x, r, method=method, **kwargs)
    assert out.is_contiguous() and out.dtype == torch.float32 and out.shape == x.shape
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    identity = ct.transfer(x, r, method=method, strength=0, clamp=False, **kwargs)
    torch.testing.assert_close(identity, x.float(), rtol=0, atol=0)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("query", [False, True])
def test_sliced_reference(ties, query):
    x = data((29, 3))
    y = data((37, 3), 3)
    q = data((43, 3), 5) if query else None
    if ties:
        x[3:14] = x[0]
    d = _directions(12, 17, "cpu").numpy()
    out = ct.sliced_transport(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(y).cuda(),
        queries=torch.from_numpy(q).cuda() if query else None,
        iterations=12,
        seed=17,
    )
    np.testing.assert_allclose(out.cpu(), reference.sliced(x, y, d, q), atol=3e-5, rtol=3e-5)


def test_sliced_constant_and_identity():
    x = torch.full((17, 3), 0.4, device="cuda")
    y = torch.full((23, 3), 0.7, device="cuda")
    out = ct.sliced_transport(x, y, iterations=64)
    torch.testing.assert_close(out, torch.full_like(x, 0.7), atol=2e-5, rtol=0)
    x = torch.rand(43, 3, device="cuda")
    torch.testing.assert_close(ct.sliced_transport(x, x, iterations=12), x, atol=2e-6, rtol=0)


@pytest.mark.parametrize("mass", [1.0, 0.8, 0.3])
@pytest.mark.parametrize("epsilon", [0.08, 0.01])
def test_sinkhorn_plan_reference(mass, epsilon):
    x = data((13, 3))
    y = data((17, 3), 2)
    plan = ct.sinkhorn_plan(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(y).cuda(),
        mass=mass,
        epsilon=epsilon,
        iterations=400,
    )
    expected, v = reference.sinkhorn(x, y, epsilon, 400, mass)
    np.testing.assert_allclose(
        plan.coupling.cpu(), expected[: len(x), : len(y)], atol=2e-6, rtol=2e-4
    )
    assert plan.marginal_error.item() < 2e-4
    assert abs(plan.transported_mass.item() - mass) < 2e-4
    q = data((19, 3), 4)
    mapped = _barycentric(torch.from_numpy(q).cuda(), torch.from_numpy(y).cuda(), plan)
    logits = (
        v[: len(y)][None, :] - ((q[:, None, :].astype(np.float64) - y[None, :, :]) ** 2).sum(-1)
    ) / epsilon
    if mass < 1:
        logits = np.concatenate((logits, np.full((len(q), 1), v[-1] / epsilon)), 1)
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    target = weights[:, : len(y)] @ y
    if mass < 1:
        target += weights[:, -1, None] * q
    np.testing.assert_allclose(mapped.cpu(), target, atol=3e-6, rtol=3e-5)


def test_sinkhorn_small_epsilon():
    x = torch.eye(3, device="cuda")
    result = ct.sinkhorn_plan(x, x, epsilon=1e-5, iterations=20)
    assert torch.isfinite(result.coupling).all()
    torch.testing.assert_close(result.coupling, torch.eye(3, device="cuda") / 3, atol=1e-6, rtol=0)


def test_lut_affine_and_batch():
    axis = torch.linspace(0, 1, 7, device="cuda")
    identity = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    lut = torch.stack((identity * 0.6 + 0.1, identity * 0.3 + 0.4))
    x = torch.rand(2, 3, 31, 29, device="cuda")
    y = ct.apply_lut(x, lut, clamp=False)
    expected = torch.stack((x[0] * 0.6 + 0.1, x[1] * 0.3 + 0.4))
    torch.testing.assert_close(y, expected, atol=3e-7, rtol=2e-6)
    edge = torch.tensor([0.0, 1.0, 0.5], device="cuda").reshape(1, 3, 1, 1)
    torch.testing.assert_close(
        ct.apply_lut(edge, identity[None], clamp=False), edge, atol=2e-7, rtol=0
    )


def test_reusable_lut_graph_and_seed():
    x = torch.rand(2, 3, 19, 23, device="cuda")
    r = torch.rand(1, 3, 17, 21, device="cuda")
    a = ct.fit_lut(x, r, samples=32, lut_size=5, iterations=4, seed=18)
    b = ct.fit_lut(x, r, samples=32, lut_size=5, iterations=4, seed=18)
    torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expected = a(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = a(x)
    graph.replay()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_lab_and_ms_swd():
    x = data((1, 3, 29, 31))
    y = data((1, 3, 29, 31), 4)
    a = torch.from_numpy(x).cuda()
    b = torch.from_numpy(y).cuda()
    np.testing.assert_allclose(ct.srgb_to_lab(a).cpu(), reference.lab(x), atol=1e-4, rtol=1e-5)
    actual = ct.ms_swd(a, b, scales=2, projections=7, patch_size=3, stride=2, seed=6, max_size=None)
    np.testing.assert_allclose(
        actual.cpu(), reference.ms_swd(x, y, 2, 7, 3, 2, 6), atol=2e-4, rtol=2e-5
    )
    torch.testing.assert_close(
        ct.ms_swd(a, a, scales=2, projections=7, patch_size=3),
        torch.zeros(1, device="cuda"),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual,
        ct.ms_swd(b, a, scales=2, projections=7, patch_size=3, stride=2, seed=6, max_size=None),
        atol=0,
        rtol=0,
    )


def test_gpu_validation():
    x = torch.rand(1, 3, 8, 9, device="cuda")
    for fn, kwargs in [
        (ct.adain, {"eps": 0}),
        (ct.wavelet, {"levels": -1}),
        (ct.sliced_ot, {"samples": 0}),
        (ct.sinkhorn, {"mass": 0}),
    ]:
        with pytest.raises(ValueError):
            fn(x, x, **kwargs)
    with pytest.raises(ValueError):
        ct.wavelet(x, x[:, :, :, :-1])
    with pytest.raises(ValueError):
        ct.adain(x.requires_grad_(), x)
    with pytest.raises(ValueError):
        ct.ms_swd(x.detach(), x.detach(), scales=5)


@pytest.mark.parametrize("m,n", [(511, 1024), (4096, 37), (4097, 19), (19, 4097)])
def test_sliced_sort_dispatch(m, n):
    x, y = data((m, 3), 12), data((n, 3), 17)
    q = data((23, 3), 18)
    directions = _directions(3, 19, "cpu").numpy()
    actual = ct.sliced_transport(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(y).cuda(),
        queries=torch.from_numpy(q).cuda(),
        iterations=3,
        seed=19,
    )
    np.testing.assert_allclose(
        actual.cpu(), reference.sliced(x, y, directions, q), atol=4e-5, rtol=4e-5
    )


@pytest.mark.parametrize("levels", [2, 4, 6])
def test_wavelet_even_levels(levels):
    x, r = data((2, 3, 9, 11)), data((2, 3, 9, 11), 8)
    actual = ct.wavelet(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(r).cuda(),
        levels=levels,
        strength=0.37,
        clamp=False,
    )
    expected = x + 0.37 * (reference.wavelet(x, r, levels) - x)
    np.testing.assert_allclose(actual.cpu(), expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize(
    "size,dtype,broadcast",
    [
        (2, torch.float32, False),
        (17, torch.float16, True),
        (33, torch.bfloat16, False),
        (65, torch.float32, True),
    ],
)
def test_packed_lut_against_grid_sample(size, dtype, broadcast):
    # Nonlinear random tables exercise all corners; includes tails and clipped coordinates.
    import torch.nn.functional as F

    x = torch.rand(2, 3, 769, 683, device="cuda", dtype=dtype) * 1.2 - 0.1
    lut = torch.rand(1 if broadcast else 2, size, size, size, 3, device="cuda")

    def expected():
        table = lut.permute(0, 4, 1, 2, 3).expand(2, -1, -1, -1, -1)
        grid = x.float()[:, [2, 1, 0]].permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1
        mapped = F.grid_sample(
            table, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(2)
        return x.float() + 0.6 * (mapped - x.float())

    torch.testing.assert_close(
        ct.apply_lut(x, lut, strength=0.6, clamp=False), expected(), atol=8e-6, rtol=8e-6
    )
    # Public values are mutable: a packed representation must not become stale.
    lut.mul_(0.5)
    torch.testing.assert_close(
        ct.apply_lut(x, lut, strength=0.6), expected().clamp(0, 1), atol=8e-6, rtol=8e-6
    )


def test_large_lut_graph():
    x = torch.rand(1, 3, 513, 513, device="cuda")
    table = torch.rand(1, 17, 17, 17, 3, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expected = ct.apply_lut(x, table)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ct.apply_lut(x, table)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("pixels,samples", [(5, 12), (100003, 512), (100003, 1024)])
def test_fused_sampling(pixels, samples):
    from colortransfer_triton.ot import _sample

    x = torch.rand(2, 3, 1, pixels, device="cuda")
    m = min(samples, pixels)
    if m == pixels:
        index = torch.arange(pixels, device="cuda")
    else:
        gen = torch.Generator(device="cuda").manual_seed(7)
        lo = torch.arange(m, device="cuda", dtype=torch.int64) * pixels // m
        hi = (torch.arange(m, device="cuda", dtype=torch.int64) + 1) * pixels // m
        index = lo + (torch.rand(m, device="cuda", generator=gen) * (hi - lo)).long()
    torch.testing.assert_close(_sample(x, samples, 1, 7), x[1, :, 0, index].T, atol=0, rtol=0)
