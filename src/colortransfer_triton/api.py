"""Image-domain transfer without pretrained checkpoints."""

import torch

from . import _validation as check


def _moments(x, correction):
    import triton

    from . import _kernels as k

    p = x.shape[-2] * x.shape[-1]
    chunks = triton.cdiv(p, 4096)
    partial = torch.empty((x.shape[0] * 3, chunks, 2), device=x.device, dtype=torch.float32)
    stats = torch.empty((x.shape[0] * 3, 2), device=x.device, dtype=torch.float32)
    k.moments_part[(x.shape[0] * 3, chunks)](x, partial, p, chunks, 4096)
    k.moments_finish[(x.shape[0] * 3,)](
        partial, stats, p, chunks, correction, 4096, triton.next_power_of_2(chunks)
    )
    return stats


def adain(source, reference, *, eps=1e-5, correction=1, strength=1.0, clamp=True):
    """Match RGB channel mean/variance. Reference may have a different spatial size.

    correction=1 uses sample variance; correction=0 uses population variance.
    Singleton spatial dimensions use zero variance. Output is float32 NCHW.
    """
    import triton

    from . import _kernels as k

    check.pair(source, reference)
    eps = check.scalar(eps, "eps", strict=True)
    check.integer(correction, "correction", 0, 1)
    strength = check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    with torch.cuda.device(source.device):
        x = source.contiguous()
        r = reference.contiguous()
        s = _moments(x, correction)
        rs = _moments(r, correction)
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        k.adain_apply[(triton.cdiv(x.shape[-2] * x.shape[-1], 512), x.shape[0] * 3)](
            x,
            out,
            s,
            rs,
            x.shape[-2] * x.shape[-1],
            x.numel(),
            r.shape[0],
            eps,
            strength,
            clamp,
            512,
        )
        return out


def wavelet(source, reference, *, levels=5, strength=1.0, clamp=True):
    """Replace low frequencies using a stationary dilated binomial pyramid.

    Reference must have the same spatial dimensions and be spatially aligned.
    This is an undecimated filter pyramid, not an orthogonal Haar/DWT transform.
    """
    import triton

    from . import _kernels as k

    check.pair(source, reference, aligned=True)
    check.integer(levels, "levels", 0, 16)
    strength = check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    with torch.cuda.device(source.device):
        x = source.contiguous()
        r = reference.contiguous()
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        block = 256
        grid = (triton.cdiv(x.numel(), block),)
        if levels == 0:
            k.blend_reference[grid](
                x, r, out, x.shape[-2] * x.shape[-1], x.numel(), r.shape[0], strength, clamp, block
            )
            return out
        scratch = torch.empty_like(out) if levels > 1 else out
        low = out
        for level in range(levels):
            target = out if level % 2 == 0 else scratch
            k.wavelet_pass[grid](
                x,
                r,
                low,
                target,
                *x.shape[-2:],
                x.numel(),
                2**level,
                level == 0,
                level == levels - 1,
                r.shape[0],
                strength,
                clamp,
                block,
            )
            low = target
        return low


def transfer(source, reference, *, method="adain", **kwargs):
    """Dispatch to adain, wavelet, sliced_ot, sinkhorn, partial_sinkhorn or colorfm."""
    from .colorfm import colorfm
    from .ot import partial_sinkhorn, sinkhorn, sliced_ot

    methods = {
        "colorfm": colorfm,
        "adain": adain,
        "wavelet": wavelet,
        "sliced_ot": sliced_ot,
        "sinkhorn": sinkhorn,
        "partial_sinkhorn": partial_sinkhorn,
    }
    if method not in methods:
        raise ValueError(f"method must be one of {tuple(methods)}")
    return methods[method](source, reference, **kwargs)
