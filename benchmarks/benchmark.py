"""Reproducible CUDA-event benchmarks; no hardware scores are bundled."""

import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F
import triton

import colortransfer_triton as ct


def torch_adain(x, y):
    x, y = x.float(), y.float()
    vx, mx = torch.var_mean(x, dim=(-2, -1), correction=1, keepdim=True)
    vy, my = torch.var_mean(y, dim=(-2, -1), correction=1, keepdim=True)
    return ((x - mx) * torch.sqrt((vy + 1e-5) / (vx + 1e-5)) + my).clamp(0, 1)


def torch_wavelet(x, y):
    x, y = x.float(), y.float()
    axis = x.new_tensor([0.25, 0.5, 0.25])
    kernel = (axis[:, None] * axis[None, :])[None, None].repeat(3, 1, 1, 1)
    low_x, low_y = x, y
    for level in range(5):
        d = 2**level
        low_x = F.conv2d(F.pad(low_x, (d,) * 4, mode="replicate"), kernel, dilation=d, groups=3)
        low_y = F.conv2d(F.pad(low_y, (d,) * 4, mode="replicate"), kernel, dilation=d, groups=3)
    return (x - low_x + low_y).clamp(0, 1)


def measure(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values, wall_values = [], []
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        wall_values.append((time.perf_counter() - wall_start) * 1000)
        values.append(start.elapsed_time(end))
        del out
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "median_wall_ms": statistics.median(wall_values),
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - baseline,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--lut-size", type=int, default=33)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--torch-reference", action="store_true")
    p.add_argument(
        "--colorfm", action="store_true", help="Include per-pair ColorFM-O fitting and application"
    )
    p.add_argument("--colorfm-steps", type=int, default=700)
    p.add_argument("--colorfm-samples", type=int, default=16384)
    args = p.parse_args()
    if min(args.height, args.width, args.batch, args.repeats) < 1:
        p.error("dimensions, batch and repeats must be positive")
    assert torch.cuda.is_available(), "NVIDIA GPU required"
    torch.manual_seed(42)
    x = torch.rand(
        args.batch, 3, args.height, args.width, device="cuda", dtype=getattr(torch, args.dtype)
    )
    y = (0.75 * x + x.new_tensor([0.20, 0.05, 0.12])[None, :, None, None]).clamp(0, 1)
    jobs = {"adain": lambda: ct.adain(x, y), "wavelet": lambda: ct.wavelet(x, y)}
    for method in ("sliced_ot", "sinkhorn", "partial_sinkhorn"):
        opts = dict(method=method, samples=args.samples, lut_size=args.lut_size)
        lut = ct.fit_lut(x, y, **opts)
        jobs[f"{method}/apply"] = lambda lut=lut: lut(x)
        jobs[f"{method}/fit"] = lambda opts=opts: ct.fit_lut(x, y, **opts)
        jobs[f"{method}/total"] = lambda opts=opts: ct.fit_lut(x, y, **opts)(x)
    if args.colorfm:
        opts = dict(samples=args.colorfm_samples, steps=args.colorfm_steps, lut_size=args.lut_size)
        flow_lut = ct.fit_colorfm(x, y, **opts)
        jobs["colorfm/apply"] = lambda: flow_lut(x)
        jobs["colorfm/fit"] = lambda: ct.fit_colorfm(x, y, **opts)
        jobs["colorfm/total"] = lambda: ct.fit_colorfm(x, y, **opts)(x)
    if args.torch_reference:
        jobs["torch/adain"] = lambda: torch_adain(x, y)
        jobs["torch/wavelet"] = lambda: torch_wavelet(x, y)
        torch.testing.assert_close(ct.adain(x, y), torch_adain(x, y), atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(ct.wavelet(x, y), torch_wavelet(x, y), atol=3e-6, rtol=3e-6)
    for name, fn in jobs.items():
        out = fn()
        values = out.values if isinstance(out, ct.ColorLUT) else out
        assert torch.isfinite(values).all(), name
        if not isinstance(out, ct.ColorLUT):
            assert out.shape == x.shape and out.dtype == torch.float32, name
            assert out.min() >= 0 and out.max() <= 1, name
    if args.check_only:
        print("All image and LUT paths validated.")
        return
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "settings": vars(args),
        "measurements": {name: measure(fn, args.repeats) for name, fn in jobs.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
