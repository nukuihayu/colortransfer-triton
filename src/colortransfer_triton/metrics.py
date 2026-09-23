"""Non-learned multiscale patch sliced-Wasserstein color distance."""

import torch
import torch.nn.functional as F

from . import _validation as check


def srgb_to_lab(image):
    """sRGB [0,1] -> CIE Lab under D65 (L in [0,100], unscaled a/b)."""
    import triton

    from . import _kernels as k

    check.image(image)
    with torch.cuda.device(image.device):
        x = image.contiguous()
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        pixels = x.shape[-2] * x.shape[-1]
        total = x.shape[0] * pixels
        k.rgb_lab[(triton.cdiv(total, 256),)](x, out, pixels, total, 256)
        return out


def ms_swd(
    source, reference, *, scales=5, projections=128, patch_size=11, stride=1, seed=0, max_size=256
):
    """Non-learned Gaussian-pyramid / Lab / random-patch sliced L1 distance.

    This is an evaluation metric, not a transfer method. Uses Triton for Lab
    conversion and PyTorch/cuDNN for patch convolution and global sorting.
    max_size explicitly bounds evaluation resolution; None disables resizing.
    """
    check.pair(source, reference)
    if source.shape != reference.shape:
        raise ValueError("ms_swd requires matching input shapes, including batch")
    check.integer(scales, "scales", 1, 10)
    check.integer(projections, "projections", 1, 1024)
    check.integer(patch_size, "patch_size", 1, 31)
    check.integer(stride, "stride", 1, 32)
    check.integer(seed, "seed", 0, 2**63 - 1)
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    if max_size is not None:
        check.integer(max_size, "max_size", 1)
    with torch.cuda.device(source.device):
        x = source.float()
        y = reference.float()
        if max_size is not None and max(x.shape[-2:]) > max_size:
            ratio = max_size / max(x.shape[-2:])
            size = tuple(max(1, round(d * ratio)) for d in x.shape[-2:])
            x = F.interpolate(x, size=size, mode="area")
            y = F.interpolate(y, size=size, mode="area")
        # Reflect padding must remain valid at every requested scale.
        h, w = x.shape[-2:]
        for level in range(scales):
            if min(h, w) <= patch_size // 2 or (level < scales - 1 and min(h, w) <= 2):
                raise ValueError("image too small for requested scales and reflection padding")
            h, w = (h + 1) // 2, (w + 1) // 2
        generator = torch.Generator(device="cpu").manual_seed(seed)
        axis = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=torch.float32)
        blur = (axis[:, None] * axis[None, :] / 256)[None, None].repeat(3, 1, 1, 1)
        distance = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        for level in range(scales):
            directions = torch.randn(
                projections, 3 * patch_size**2, generator=generator, dtype=torch.float32
            )
            directions = (
                (directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True))
                .reshape(projections, 3, patch_size, patch_size)
                .to(x.device)
            )
            lx = srgb_to_lab(x)
            ly = srgb_to_lab(y)
            p = patch_size // 2
            px = F.conv2d(
                F.pad(lx, (p, p, p, p), mode="reflect"), directions, stride=stride
            ).flatten(2)
            py = F.conv2d(
                F.pad(ly, (p, p, p, p), mode="reflect"), directions, stride=stride
            ).flatten(2)
            distance += (px.sort(dim=-1).values - py.sort(dim=-1).values).abs().mean((1, 2))
            if level < scales - 1:
                x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), blur, groups=3)[:, :, ::2, ::2]
                y = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), blur, groups=3)[:, :, ::2, ::2]
        return distance / scales
