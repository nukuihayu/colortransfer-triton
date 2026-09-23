"""CPU float64 references, deliberately separate from GPU kernels."""

import numpy as np
import torch
import torch.nn.functional as F


def adain(x, r, eps=1e-5, correction=1):
    x = x.astype(np.float64)
    r = r.astype(np.float64)
    vx = (
        np.var(x, axis=(-2, -1), keepdims=True, ddof=correction)
        if x.shape[-1] * x.shape[-2] > 1
        else np.zeros_like(x)
    )
    vr = (
        np.var(r, axis=(-2, -1), keepdims=True, ddof=correction)
        if r.shape[-1] * r.shape[-2] > 1
        else np.zeros_like(r)
    )
    return (x - x.mean((-2, -1), keepdims=True)) * np.sqrt((vr + eps) / (vx + eps)) + r.mean(
        (-2, -1), keepdims=True
    )


def wavelet(x, r, levels):
    def low(image):
        out = torch.from_numpy(image.astype(np.float64))
        v = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        kernel = (v[:, None] * v[None, :] / 16)[None, None].repeat(3, 1, 1, 1)
        for level in range(levels):
            radius = 2**level
            out = F.conv2d(
                F.pad(out, (radius,) * 4, mode="replicate"), kernel, dilation=radius, groups=3
            )
        return out.numpy()

    return x.astype(np.float64) - low(x) + low(r)


def sliced(x, y, directions, queries=None):
    m, n = len(x), len(y)
    state = (
        x.astype(np.float64).copy()
        if queries is None
        else np.concatenate((x, queries)).astype(np.float64)
    )
    reference = y.astype(np.float64)
    for direction in directions.astype(np.float64):
        projected = (
            state[:, 0] * direction[0] + state[:, 1] * direction[1] + state[:, 2] * direction[2]
        )
        sp = np.sort(projected[:m])
        rp = np.sort(
            reference[:, 0] * direction[0]
            + reference[:, 1] * direction[1]
            + reference[:, 2] * direction[2]
        )
        if sp[0] == sp[-1]:
            delta = reference.mean(0) @ direction - sp[0]
        else:
            lo = np.searchsorted(sp, projected, "left")
            hi = np.searchsorted(sp, projected, "right")
            left = np.clip(lo - 1, 0, m - 1)
            right = np.minimum(lo, m - 1)
            frac = np.clip((projected - sp[left]) / np.maximum(sp[right] - sp[left], 1e-20), 0, 1)
            ranks = np.where(hi > lo, (lo + hi - 1) / 2, left + frac * (right - left))
            pos = np.clip((ranks + 0.5) * n / m - 0.5, 0, n - 1)
            delta = np.interp(pos, np.arange(n), rp) - projected
        state += np.asarray(delta)[..., None] * direction
    return state if queries is None else state[m:]


def sinkhorn(x, y, epsilon, iterations, mass):
    cost = ((x[:, None, :].astype(np.float64) - y[None, :, :]) ** 2).sum(-1)
    m, n = cost.shape
    a = np.full(m, 1 / m)
    b = np.full(n, 1 / n)
    if mass < 1:
        augmented = np.zeros((m + 1, n + 1))
        augmented[:m, :n] = cost
        augmented[-1, -1] = np.inf
        cost = augmented
        a = np.append(a, 1 - mass)
        b = np.append(b, 1 - mass)

    def lse(z, axis):
        peak = z.max(axis=axis, keepdims=True)
        return (peak + np.log(np.exp(z - peak).sum(axis=axis, keepdims=True))).squeeze(axis)

    u = np.zeros(len(a))
    v = np.zeros(len(b))
    for _ in range(iterations):
        u = epsilon * (np.log(a) - lse((v[None, :] - cost) / epsilon, 1))
        v = epsilon * (np.log(b) - lse((u[:, None] - cost) / epsilon, 0))
    return np.exp((u[:, None] + v[None, :] - cost) / epsilon), v


def lab(x):
    rgb = np.moveaxis(x.astype(np.float64), 1, -1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((np.maximum(rgb, 0.04045) + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = (lin @ matrix.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    out = np.stack(
        (116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), -1
    )
    return np.moveaxis(out, -1, 1)


def ms_swd(x, y, scales, projections, patch_size, stride, seed):
    gen = torch.Generator().manual_seed(seed)
    axis = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64)
    kernel = (axis[:, None] * axis[None, :] / 256)[None, None].repeat(3, 1, 1, 1)
    x = torch.from_numpy(x).double()
    y = torch.from_numpy(y).double()
    result = torch.zeros(x.shape[0], dtype=torch.float64)
    for level in range(scales):
        directions = torch.randn(projections, 3 * patch_size**2, generator=gen, dtype=torch.float32)
        directions = (
            (directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True))
            .double()
            .reshape(projections, 3, patch_size, patch_size)
        )
        p = patch_size // 2
        a = (
            F.conv2d(
                F.pad(torch.from_numpy(lab(x.numpy())), (p,) * 4, mode="reflect"),
                directions,
                stride=stride,
            )
            .flatten(2)
            .sort()
            .values
        )
        b = (
            F.conv2d(
                F.pad(torch.from_numpy(lab(y.numpy())), (p,) * 4, mode="reflect"),
                directions,
                stride=stride,
            )
            .flatten(2)
            .sort()
            .values
        )
        result += (a - b).abs().mean((1, 2))
        if level < scales - 1:
            x = F.conv2d(F.pad(x, (2,) * 4, mode="reflect"), kernel, groups=3)[:, :, ::2, ::2]
            y = F.conv2d(F.pad(y, (2,) * 4, mode="reflect"), kernel, groups=3)[:, :, ::2, ::2]
    return (result / scales).numpy()
