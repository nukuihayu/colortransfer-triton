"""ColorFM-O without semantic segmentation or pretrained weights.

Each image pair fits a fresh 4 -> hidden -> 3 bias-free SiLU velocity field.
PyTorch handles differentiation/Adam; Triton integrates the field on a LUT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import _validation as check


def _couple(source, reference, depth, seed):
    """Recursive mean-centered octant coupling; small sampled clouds live on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x, y = source.detach().cpu(), reference.detach().cpu()

    def random_pairs(a, b):
        count = min(len(a), len(b))
        ia = torch.randperm(len(a), generator=generator, device="cpu")[:count]
        ib = torch.randperm(len(b), generator=generator, device="cpu")[:count]
        return [torch.stack((a[ia], b[ib]), 1)]

    def partition(a, b, level):
        if level == depth:
            return random_pairs(a, b)
        weights = torch.tensor([4, 2, 1], dtype=torch.int64, device="cpu")
        sa = ((a >= a.mean(0)).long() * weights).sum(1)
        sb = ((b >= b.mean(0)).long() * weights).sum(1)
        leaves = []
        for octant in range(8):
            aa, bb = a[sa == octant], b[sb == octant]
            if len(aa) and len(bb):
                leaves.extend(partition(aa, bb, level + 1))
        return leaves or random_pairs(a, b)

    return torch.cat(partition(x, y, 0)).to(source.device)


def _velocity(points, times, w1, w2):
    inputs = torch.cat((points, times[:, None]), 1)
    return F.linear(F.silu(F.linear(inputs, w1)), w2)


def _loss(pairs, times, w1, w2):
    source, reference = pairs[:, 0], pairs[:, 1]
    velocity = reference - source
    position = (1 - times[:, None]) * source + times[:, None] * reference
    error = (_velocity(position, times, w1, w2) - velocity).square().sum(1)
    return (error / (1e-4 + torch.linalg.vector_norm(velocity, dim=1))).mean()


def _fit(pairs, hidden, steps, batch_size, learning_rate, seed):
    init = torch.Generator(device="cpu").manual_seed(seed)
    w1 = torch.empty(hidden, 4, dtype=torch.float32, device="cpu").uniform_(
        -0.5, 0.5, generator=init
    )
    bound = 1 / math.sqrt(hidden)
    w2 = torch.empty(3, hidden, dtype=torch.float32, device="cpu").uniform_(
        -bound, bound, generator=init
    )
    w1 = w1.to(pairs.device).requires_grad_()
    w2 = w2.to(pairs.device).requires_grad_()
    optimizer = torch.optim.Adam([w1, w2], lr=learning_rate, betas=(0.9, 0.999), fused=True)
    generator = torch.Generator(device=pairs.device).manual_seed(seed + 1)
    # The fixed probe makes initial/final losses comparable, unlike the last minibatch.
    probe = pairs[: min(4096, len(pairs))]
    times = torch.linspace(0, 1, len(probe), device=pairs.device, dtype=torch.float32)
    with torch.no_grad():
        initial = _loss(probe, times, w1, w2)
    for _ in range(steps):
        index = torch.randint(len(pairs), (batch_size,), device=pairs.device, generator=generator)
        t = torch.rand(batch_size, device=pairs.device, generator=generator, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(pairs[index], t, w1, w2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = _loss(probe, times, w1, w2)
    return w1.detach(), w2.detach(), {"initial_loss": initial, "final_loss": final}


@dataclass(frozen=True)
class ColorFlow:
    """Per-pair fitted velocity fields; no checkpoint is loaded or downloaded."""

    input_weight: torch.Tensor  # (N, hidden, 4)
    output_weight: torch.Tensor  # (N, 3, hidden)
    diagnostics: tuple

    def transform_points(self, points, *, batch=0, ode_steps=5, time=1.0):
        """Unclipped explicit-midpoint integration on (M,3) points.

        time controls the integration horizon, as in upstream transfer_strength.
        The image APIs' strength instead uses the library's usual output blending.
        """
        import triton

        from . import _kernels as k

        check.points(points)
        check.tensor(self.input_weight, "input_weight")
        check.tensor(self.output_weight, "output_weight")
        w1, w2 = self.input_weight, self.output_weight
        if (
            w1.ndim != 3
            or w1.shape[-1] != 4
            or w2.shape != (w1.shape[0], 3, w1.shape[1])
            or w1.shape[1] > 1024
            or w1.device != points.device
            or w2.device != points.device
        ):
            raise ValueError(
                "velocity weights must be (N,hidden,4)/(N,3,hidden) on the input device"
            )
        check.integer(batch, "batch", 0, w1.shape[0] - 1)
        check.integer(ode_steps, "ode_steps", 1, 128)
        time = check.scalar(time, "time", maximum=1)
        with torch.cuda.device(points.device):
            x = points.float().contiguous()
            a, b = w1[batch].float().contiguous(), w2[batch].float().contiguous()
            out = torch.empty_like(x)
            k.colorfm_integrate[(triton.cdiv(len(x), 16),)](
                x,
                a,
                b,
                out,
                len(x),
                w1.shape[1],
                ode_steps,
                time,
                triton.next_power_of_2(w1.shape[1]),
                16,
            )
            return out

    def to_lut(self, *, lut_size=33, ode_steps=5, time=1.0):
        """Evaluate the fitted field on a regular RGB grid; no further optimization."""
        from .ot import ColorLUT

        check.integer(lut_size, "lut_size", 2, 65)
        axis = torch.linspace(0, 1, lut_size, device=self.input_weight.device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        tables = [
            self.transform_points(grid, batch=n, ode_steps=ode_steps, time=time).reshape(
                lut_size, lut_size, lut_size, 3
            )
            for n in range(self.input_weight.shape[0])
        ]
        return ColorLUT(torch.stack(tables), "colorfm", self.diagnostics)


def fit_colorfm_model(
    source,
    reference,
    *,
    samples=16384,
    steps=700,
    hidden=512,
    batch_size=4096,
    learning_rate=5e-4,
    coupling_depth=3,
    seed=0,
):
    """Fit the nonsemantic ColorFM-O variant from scratch for each image pair.

    Sampled RGB clouds replace upstream 512x512 image resizing. Minibatches are
    sampled with replacement; results need not match upstream optimizer iterates.
    This method performs local training, even inside no_grad/inference_mode.
    """
    from .ot import _sample

    check.pair(source, reference)
    check.integer(samples, "samples", 1, 262144)
    check.integer(steps, "steps", 1, 10000)
    check.integer(hidden, "hidden", 8, 1024)
    check.integer(batch_size, "batch_size", 1, 16384)
    check.integer(coupling_depth, "coupling_depth", 0, 4)
    check.integer(seed, "seed", 0, 2**63 - 8)
    learning_rate = check.scalar(learning_rate, "learning_rate", strict=True)
    weights1, weights2, diagnostics = [], [], []
    with (
        torch.cuda.device(source.device),
        torch.inference_mode(False),
        torch.enable_grad(),
        torch.autocast("cuda", enabled=False),
    ):
        x, ref = source.contiguous(), reference.contiguous()
        for n in range(x.shape[0]):
            a = _sample(x, samples, n, seed + 1)
            b = _sample(ref, samples, 0 if ref.shape[0] == 1 else n, seed + 2)
            pairs = _couple(a, b, coupling_depth, seed + 3)
            w1, w2, diagnostic = _fit(pairs, hidden, steps, batch_size, learning_rate, seed + 4)
            weights1.append(w1)
            weights2.append(w2)
            diagnostics.append(diagnostic)
        return ColorFlow(torch.stack(weights1), torch.stack(weights2), tuple(diagnostics))


def fit_colorfm(source, reference, *, lut_size=33, ode_steps=5, time=1.0, **kwargs):
    """Fit ColorFM-O and return a reusable ColorLUT; no pretrained/segmentation model."""
    check.integer(lut_size, "lut_size", 2, 65)
    check.integer(ode_steps, "ode_steps", 1, 128)
    check.scalar(time, "time", maximum=1)
    model = fit_colorfm_model(source, reference, **kwargs)
    return model.to_lut(lut_size=lut_size, ode_steps=ode_steps, time=time)


def colorfm(source, reference, *, strength=1.0, clamp=True, **kwargs):
    """ColorFM-O without pretrained weights, followed by the common LUT image path."""
    strength = check.scalar(strength, "strength", maximum=1)
    check.boolean(clamp, "clamp")
    return fit_colorfm(source, reference, **kwargs)(source, strength=strength, clamp=clamp)
