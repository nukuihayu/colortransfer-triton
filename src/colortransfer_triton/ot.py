"""Sampled optimal transport with explicit, reusable 3D LUT acceleration."""

from dataclasses import dataclass

import torch

from . import _validation as check


def _directions(iterations, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(iterations, 3, generator=generator, dtype=torch.float32)
    return (d / torch.linalg.vector_norm(d, dim=1, keepdim=True)).to(device)


def _sliced(source, reference, queries, directions):
    import triton

    from . import _kernels as k

    m, n = source.shape[0], reference.shape[0]
    state = torch.cat((source, queries), dim=0) if queries is not None else source.clone()
    scratch = torch.empty_like(state)
    sx = torch.empty(m, device=source.device, dtype=torch.float32)
    precomputed = n <= 4096
    ry = torch.empty(
        n * (directions.shape[0] if precomputed else 1), device=source.device, dtype=torch.float32
    )
    if precomputed:
        k.project_sort[(directions.shape[0],)](
            reference,
            directions,
            ry,
            n,
            -1,
            triton.next_power_of_2(n),
            ALL=True,
            enable_fp_fusion=False,
        )
    avg = reference.mean(0)
    for step in range(directions.shape[0]):
        if m <= 4096:
            k.project_sort[(1,)](
                state, directions, sx, m, step, triton.next_power_of_2(m), enable_fp_fusion=False
            )
            sorted_x = sx
        else:
            k.project[(triton.cdiv(m, 256),)](
                state, directions, sx, m, step, 256, enable_fp_fusion=False
            )
            sorted_x = torch.sort(sx).values
        if precomputed:
            sorted_y = ry
        else:
            k.project[(triton.cdiv(n, 256),)](
                reference, directions, ry, n, step, 256, enable_fp_fusion=False
            )
            sorted_y = torch.sort(ry).values
        k.sliced_update[(triton.cdiv(state.shape[0], 256),)](
            state,
            scratch,
            sorted_x,
            sorted_y,
            directions,
            avg,
            state.shape[0],
            m,
            n,
            step,
            256,
            PRECOMPUTED=precomputed,
            enable_fp_fusion=False,
        )
        state, scratch = scratch, state
    return state[m:] if queries is not None else state


def sliced_transport(source, reference, *, queries=None, iterations=32, seed=0):
    """Sequential 1D sliced transport on finite RGB point clouds (M,3).

    No image sampling/LUT approximation here. Optional queries follow the maps
    estimated from source points; their projections are linearly interpolated.
    Equal projection values use their midrank; a constant cloud uses translation.
    """
    check.points(source, "source")
    check.points(reference, "reference")
    if source.device != reference.device:
        raise ValueError("source and reference devices must match")
    check.integer(iterations, "iterations", 1, 1024)
    check.integer(seed, "seed", 0, 2**63 - 1)
    if queries is not None:
        check.points(queries, "queries")
        if queries.device != source.device:
            raise ValueError("queries must share the source device")
        queries = queries.float().contiguous()
    with torch.cuda.device(source.device):
        return _sliced(
            source.float().contiguous(),
            reference.float().contiguous(),
            queries,
            _directions(iterations, seed, source.device),
        )


@dataclass(frozen=True)
class TransportPlan:
    """Finite-iteration entropic plan; diagnostics remain GPU tensors."""

    coupling: torch.Tensor
    column_potential: torch.Tensor
    marginal_error: torch.Tensor
    transported_mass: torch.Tensor
    epsilon: float
    mass: float
    iterations: int


def sinkhorn_plan(source, reference, *, epsilon=0.03, iterations=100, mass=1.0):
    """Uniform-mass log-domain Sinkhorn for point clouds, up to 4096 points each.

    mass<1 uses a balanced dummy-node reduction with a forbidden dummy/dummy edge.
    Entropy applies to the augmented coupling, including the dummy edges.
    Fixed iterations do not imply convergence: inspect marginal_error. The
    returned coupling excludes dummy nodes; transported_mass should approach mass.
    """
    import triton

    from . import _kernels as k

    check.points(source, "source", 4096)
    check.points(reference, "reference", 4096)
    if source.device != reference.device:
        raise ValueError("source and reference devices must match")
    epsilon = check.scalar(epsilon, "epsilon", strict=True)
    check.integer(iterations, "iterations", 1, 10000)
    mass = check.scalar(mass, "mass", maximum=1, strict=True)
    with torch.cuda.device(source.device):
        x = source.float().contiguous()
        y = reference.float().contiguous()
        m, n = x.shape[0], y.shape[0]
        partial = mass < 1
        rows, cols = m + partial, n + partial
        # Aligned row strides avoid scalar memory transactions for dummy-node rows.
        cs, cts = triton.cdiv(cols, 32) * 32, triton.cdiv(rows, 32) * 32
        cost = torch.empty((rows, cs), device=x.device, dtype=torch.float32)
        cost_t = torch.empty((cols, cts), device=x.device, dtype=torch.float32)
        k.cost_matrix[(triton.cdiv(rows * cols, 256),)](
            x, y, cost, cost_t, m, n, partial, cs, cts, 256
        )
        u = torch.zeros(rows, device=x.device, dtype=torch.float32)
        v = torch.zeros(cols, device=x.device, dtype=torch.float32)
        for _ in range(iterations):
            k.sinkhorn_step[(triton.cdiv(rows, 4),)](
                cost, v, u, rows, cols, cs, m, mass, epsilon, triton.next_power_of_2(cols), 4
            )
            k.sinkhorn_step[(triton.cdiv(cols, 4),)](
                cost_t, u, v, cols, rows, cts, n, mass, epsilon, triton.next_power_of_2(rows), 4
            )
        plan = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
        k.coupling[(triton.cdiv(rows * cols, 256),)](cost, u, v, plan, rows, cols, cs, epsilon, 256)
        a = torch.full((m,), 1 / m, device=x.device, dtype=torch.float32)
        b = torch.full((n,), 1 / n, device=x.device, dtype=torch.float32)
        if partial:
            # GPU fill avoids a host-scalar assignment that synchronizes CUDA.
            dummy = torch.full((1,), 1 - mass, device=x.device, dtype=torch.float32)
            a = torch.cat((a, dummy))
            b = torch.cat((b, dummy))
        error = torch.maximum((plan.sum(1) - a).abs().sum(), (plan.sum(0) - b).abs().sum())
        real = plan[:m, :n].contiguous()
        return TransportPlan(real, v, error, real.sum(), epsilon, mass, iterations)


def _barycentric(queries, reference, plan):
    import triton

    from . import _kernels as k

    out = torch.empty_like(queries)
    k.barycentric[(triton.cdiv(queries.shape[0], 16),)](
        queries,
        reference,
        plan.column_potential,
        out,
        queries.shape[0],
        reference.shape[0],
        plan.epsilon,
        plan.mass < 1,
        16,
        min(512, triton.next_power_of_2(reference.shape[0])),
        num_warps=4,
    )
    return out


def _sample(image, count, batch, seed):
    import triton

    from . import _kernels as k

    pixels = image.shape[-2] * image.shape[-1]
    m = min(count, pixels)
    if m == pixels:
        random = None
    else:
        # One jittered sample in each equal-length raster interval; no replacement.
        gen = torch.Generator(device=image.device).manual_seed(seed)
        random = torch.rand(m, device=image.device, generator=gen, dtype=torch.float32)
    out = torch.empty((m, 3), device=image.device, dtype=torch.float32)
    k.sample_image[(triton.cdiv(m, 256),)](image, random, out, pixels, m, batch, 256)
    return out


@dataclass(frozen=True)
class ColorLUT:
    """RGB LUT in (N,R,G,B,3) order. Immutable configuration, inference-only."""

    values: torch.Tensor
    method: str
    diagnostics: tuple = ()

    def __call__(self, image, *, strength=1.0, clamp=True):
        return apply_lut(image, self, strength=strength, clamp=clamp)


def apply_lut(image, lut, *, strength=1.0, clamp=True):
    """Trilinear RGB lookup, fused with strength blending and optional clipping.

    Lookup coordinates are clipped to [0,1]; the original input is used for
    strength blending. LUT batch can be 1 or match the input batch.
    """
    import triton

    from . import _kernels as k

    check.image(image)
    values = lut.values if isinstance(lut, ColorLUT) else lut
    check.tensor(values, "lut")
    if (
        values.ndim != 5
        or values.shape[-1] != 3
        or len(set(values.shape[1:4])) != 1
        or values.shape[1] < 2
        or values.shape[0] not in (1, image.shape[0])
        or values.device != image.device
    ):
        raise ValueError("lut must have shape (N,L,L,L,3), L>=2, matching device and batch 1 or N")
    strength = check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    with torch.cuda.device(image.device):
        x = image.contiguous()
        table = values.float().contiguous()
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        p = x.shape[-2] * x.shape[-1]
        total = x.shape[0] * p
        size = table.shape[1]
        packed_elements = table.shape[0] * (size - 1) ** 3 * 24
        if p >= max(262144, 2 * (size - 1) ** 3) and size <= 65 and packed_elements < 2**31:
            # Repack each call: mutations of public LUT tensors must remain visible.
            packed = torch.empty(packed_elements, device=x.device, dtype=torch.float32)
            k.lut_pack[(triton.cdiv(packed_elements, 256),)](
                table, packed, size, packed_elements, 256
            )
            k.lut_apply_packed[(triton.cdiv(p, 64), x.shape[0])](
                x,
                packed,
                out,
                p,
                size,
                table.shape[0],
                strength,
                clamp,
                64,
            )
            return out
        k.lut_apply[(triton.cdiv(total, 256),)](
            x, table, out, p, total, table.shape[1], table.shape[0], strength, clamp, 256
        )
        return out


def fit_lut(
    source,
    reference,
    *,
    method="sliced_ot",
    samples=1024,
    lut_size=33,
    iterations=None,
    seed=0,
    epsilon=0.03,
    mass=None,
):
    """Fit a sampled color map and evaluate it on an RGB grid for fast application.

    This is an explicit sampling/LUT approximation, not full-resolution OT.
    A fitted table is specific to the source/reference distributions.
    """
    if method == "colorfm":
        from .colorfm import fit_colorfm

        if mass is not None and check.scalar(mass, "mass", maximum=1, strict=True) != 1:
            raise ValueError("mass is only supported by Sinkhorn")
        return fit_colorfm(
            source,
            reference,
            samples=samples,
            lut_size=lut_size,
            steps=700 if iterations is None else iterations,
            seed=seed,
        )
    check.pair(source, reference)
    if method not in ("sliced_ot", "sinkhorn", "partial_sinkhorn"):
        raise ValueError("method must be sliced_ot, sinkhorn, partial_sinkhorn or colorfm")
    check.integer(samples, "samples", 1, 4096 if method != "sliced_ot" else 65536)
    check.integer(lut_size, "lut_size", 2, 65)
    check.integer(seed, "seed", 0, 2**63 - 3)
    epsilon = check.scalar(epsilon, "epsilon", strict=True)
    mass = (0.8 if method == "partial_sinkhorn" else 1.0) if mass is None else mass
    mass = check.scalar(mass, "mass", maximum=1, strict=True)
    if method == "sliced_ot" and mass != 1:
        raise ValueError("mass is only supported by Sinkhorn")
    iterations = (32 if method == "sliced_ot" else 100) if iterations is None else iterations
    check.integer(iterations, "iterations", 1, 1024 if method == "sliced_ot" else 10000)
    with torch.cuda.device(source.device):
        x = source.contiguous()
        ref = reference.contiguous()
        axis = torch.linspace(0, 1, lut_size, device=x.device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        directions = _directions(iterations, seed, x.device) if method == "sliced_ot" else None
        tables = []
        diagnostics = []
        for batch in range(x.shape[0]):
            sx = _sample(x, samples, batch, seed + 1)
            ry = _sample(ref, samples, 0 if ref.shape[0] == 1 else batch, seed + 2)
            if method == "sliced_ot":
                mapped = _sliced(sx, ry, grid, directions)
            else:
                plan = sinkhorn_plan(sx, ry, epsilon=epsilon, iterations=iterations, mass=mass)
                mapped = _barycentric(grid, ry, plan)
                diagnostics.append(
                    {
                        "marginal_error": plan.marginal_error,
                        "transported_mass": plan.transported_mass,
                    }
                )
            tables.append(mapped.reshape(lut_size, lut_size, lut_size, 3))
        return ColorLUT(torch.stack(tables), method, tuple(diagnostics))


def sliced_ot(
    source, reference, *, samples=1024, lut_size=33, iterations=32, seed=0, strength=1.0, clamp=True
):
    """Sampled sliced transport + LUT application. Use sliced_transport for points."""
    check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    lut = fit_lut(
        source,
        reference,
        method="sliced_ot",
        samples=samples,
        lut_size=lut_size,
        iterations=iterations,
        seed=seed,
    )
    return lut(source, strength=strength, clamp=clamp)


def sinkhorn(
    source,
    reference,
    *,
    samples=512,
    lut_size=33,
    epsilon=0.03,
    iterations=100,
    mass=1.0,
    seed=0,
    strength=1.0,
    clamp=True,
):
    """Sampled entropic OT with barycentric extension and a trilinear LUT."""
    check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    lut = fit_lut(
        source,
        reference,
        method="sinkhorn",
        samples=samples,
        lut_size=lut_size,
        epsilon=epsilon,
        iterations=iterations,
        mass=mass,
        seed=seed,
    )
    return lut(source, strength=strength, clamp=clamp)


def partial_sinkhorn(source, reference, *, mass=0.8, **kwargs):
    """Partial transport; unmoved probability keeps the original query color."""
    return sinkhorn(source, reference, mass=mass, **kwargs)
