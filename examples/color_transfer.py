"""Run all transfer methods on RGB photographs and draw matched comparison panels."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

import colortransfer_triton as ct

METHODS = (
    ("AdaIN", "adain"),
    ("Wavelet", "wavelet"),
    ("Sliced OT", "sliced_ot"),
    ("Sinkhorn", "sinkhorn"),
    ("Partial Sinkhorn", "partial_sinkhorn"),
    ("ColorFM-O", "colorfm"),
)
CASES = ("astronaut", "chelsea", "astronaut-to-coffee", "cross-image")
ALIGNED_CASES = ("astronaut", "chelsea")
REGIONS = {
    "astronaut": (0.31, 0.09, 0.69, 0.47),
    "chelsea": (0.32, 0.08, 0.76, 0.72),
    "astronaut-to-coffee": (0.31, 0.09, 0.69, 0.47),
    "cross-image": (0.23, 0.16, 0.74, 0.75),
}


def load_rgb(path, device):
    with Image.open(path) as image:
        array = np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def demo_pair(device="cuda", case="astronaut"):
    data = Path(__file__).parent / "data"
    if case == "cross-image":
        return load_rgb(data / "coffee.png", device), load_rgb(data / "chelsea.png", device)
    if case == "astronaut-to-coffee":
        return load_rgb(data / "astronaut.png", device), load_rgb(data / "coffee.png", device)
    image = load_rgb(data / f"{case}.png", device)
    if case == "astronaut":
        gain = image.new_tensor([0.70, 0.88, 1.0])[None, :, None, None]
        bias = image.new_tensor([0.025, 0.015, 0.0])[None, :, None, None]
        return image * gain + bias, image
    if case == "chelsea":
        gamma = image.new_tensor([0.9, 1.03, 1.10])[None, :, None, None]
        gain = image.new_tensor([1.08, 0.95, 0.85])[None, :, None, None]
        bias = image.new_tensor([0.02, 0.015, 0.005])[None, :, None, None]
        return image, (image.pow(gamma) * gain + bias).clamp(0, 1)
    raise ValueError(f"Unknown example: {case}")


def run_methods(
    source,
    reference,
    *,
    aligned=True,
    strength=1.0,
    samples=1024,
    lut_size=33,
    epsilon=0.01,
    iterations=500,
    seed=0,
    colorfm_steps=700,
    colorfm_samples=16384,
):
    outputs = {"Source": source, "Reference": reference}
    for label, method in METHODS:
        if method == "wavelet" and not aligned:
            continue
        if method in ("adain", "wavelet"):
            out = ct.transfer(source, reference, method=method, strength=strength)
        elif method == "colorfm":
            lut = ct.fit_colorfm(
                source,
                reference,
                samples=colorfm_samples,
                steps=colorfm_steps,
                lut_size=lut_size,
                seed=seed,
            )
            for item in lut.diagnostics:
                assert torch.isfinite(item["final_loss"]), "ColorFM optimization diverged"
            out = lut(source, strength=strength)
        else:
            lut = ct.fit_lut(
                source,
                reference,
                method=method,
                samples=samples,
                lut_size=lut_size,
                epsilon=epsilon,
                iterations=32 if method == "sliced_ot" else iterations,
                seed=seed,
            )
            for item in lut.diagnostics:
                assert item["marginal_error"].item() < 5e-3, (
                    f"{method}: solver has not converged; increase --iterations or --epsilon"
                )
            out = lut(source, strength=strength)
        assert out.shape == source.shape and out.dtype == torch.float32, method
        assert torch.isfinite(out).all(), method
        assert out.min() >= 0 and out.max() <= 1, method
        outputs[label] = out
    return outputs


def draw(outputs, output, *, aligned, save_images=False, region=REGIONS["astronaut"]):
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: value[0].permute(1, 2, 0).cpu().numpy() for name, value in outputs.items()}
    h, w = arrays["Source"].shape[:2]

    def rectangle(height, width):
        left, top = (
            min(round(width * region[0]), width - 1),
            min(round(height * region[1]), height - 1),
        )
        right, bottom = (
            max(left + 1, round(width * region[2])),
            max(top + 1, round(height * region[3])),
        )
        return left, top, right, bottom

    roi = rectangle(h, w)
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "pdf.fonttype": 42})
    # One horizontal strip per case; each panel retains more than 512 export pixels.
    columns = len(arrays)
    for detail in (False, True):
        aspect = (roi[2] - roi[0]) / (roi[3] - roi[1]) if detail else w / h
        fig, axes = plt.subplots(
            1, columns, figsize=(4.2 * columns, 4.2 / aspect + 0.5), squeeze=False
        )
        fig.subplots_adjust(left=0.003, right=0.997, bottom=0.015, top=0.90, wspace=0.035)
        for index, (name, array) in enumerate(arrays.items()):
            ax = axes.flat[index]
            left, top, right, bottom = roi
            if detail:
                rh, rw = array.shape[:2]
                left, top, right, bottom = rectangle(rh, rw)
                array = array[top:bottom, left:right]
            ax.imshow(array, interpolation="nearest" if detail else "none", vmin=0, vmax=1)
            ax.set_title(f"({chr(97 + index)}) {name}", pad=7)
            ax.set_axis_off()
            ax.set_anchor("N")
            if not detail and (name != "Reference" or aligned):
                ax.add_patch(
                    Rectangle(
                        (left, top),
                        right - left,
                        bottom - top,
                        fill=False,
                        edgecolor="#a33232",
                        linewidth=0.8,
                    )
                )
        for ax in list(axes.flat)[len(arrays) :]:
            ax.set_axis_off()
        name = "detail" if detail else "color-transfer"
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=160, facecolor="white")
        plt.close(fig)
    if save_images:
        folder = output / "images"
        folder.mkdir(exist_ok=True)
        for name, array in arrays.items():
            Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(
                folder / f"{name.lower().replace(' ', '-')}.png"
            )


def draw_suite(cases, output):
    """Export separate aligned and different-content two-row comparisons."""
    for name, selected in (("aligned", CASES[:2]), ("unaligned", CASES[2:])):
        draw_group(cases, output, name, selected)


def draw_group(cases, output, group, selected):
    columns = ("Source", "Reference", *(label for label, _ in METHODS))
    columns = tuple(name for name in columns if name in cases[selected[0]])
    labels = (
        "Aligned 1 · Astronaut / cool-cast recovery",
        "Aligned 2 · Chelsea / warm reference",
        "Different-content 1 · Astronaut → Coffee",
        "Different-content 2 · Coffee → Chelsea",
    )
    plt.rcParams.update({"font.family": "serif", "font.size": 16, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(columns), figsize=(4.2 * len(columns), 8))
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.015, top=0.86, wspace=0.04, hspace=0.23)
    headings = []
    for row, case in enumerate(selected):
        label = labels[CASES.index(case)]
        for col, name in enumerate(columns):
            ax = axes[row, col]
            ax.set_axis_off()
            position = ax.get_position()
            if row == 0:
                headings.append(
                    fig.text(
                        (position.x0 + position.x1) / 2,
                        0.975,
                        name,
                        ha="center",
                        va="top",
                    )
                )
            if col == 0:
                headings.append(
                    fig.text(
                        position.x0,
                        position.y1 + 0.014,
                        label,
                        fontsize=14,
                        ha="left",
                        va="bottom",
                    )
                )
            if name in cases[case]:
                ax.imshow(cases[case][name], interpolation="none", vmin=0, vmax=1)
                ax.set_aspect("equal", adjustable="datalim")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A\nUnaligned pair",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.5",
                )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [heading.get_window_extent(renderer) for heading in headings]
    for index, box in enumerate(bounds):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), (
            "Comparison heading extends beyond the canvas"
        )
        assert not any(box.overlaps(other) for other in bounds[index + 1 :]), (
            "Comparison headings overlap"
        )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{group}.{ext}", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--case", choices=CASES, default="astronaut")
    p.add_argument("--suite", action="store_true", help="Run all four bundled comparisons")
    p.add_argument(
        "--aligned", action="store_true", help="Include wavelet for spatially aligned custom images"
    )
    p.add_argument("--output", type=Path, default=Path("example-output"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--lut-size", type=int, default=33)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--colorfm-steps", type=int, default=700)
    p.add_argument("--colorfm-samples", type=int, default=16384)
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()
    if (args.source is None) != (args.reference is None):
        p.error("Supply both --source and --reference, or neither for the bundled demo")
    demo = args.source is None
    if args.suite and not demo:
        p.error("--suite cannot be combined with custom images")
    suite_arrays = {}
    for case in CASES if args.suite else (args.case,):
        x, r = (
            demo_pair(args.device, case)
            if demo
            else (load_rgb(args.source, args.device), load_rgb(args.reference, args.device))
        )
        aligned = case in ALIGNED_CASES if demo else args.aligned
        if aligned and x.shape != r.shape:
            p.error("Aligned images must have matching dimensions")
        outputs = run_methods(
            x,
            r,
            aligned=aligned,
            strength=args.strength,
            samples=args.samples,
            lut_size=args.lut_size,
            epsilon=args.epsilon,
            iterations=args.iterations,
            seed=args.seed,
            colorfm_steps=args.colorfm_steps,
            colorfm_samples=args.colorfm_samples,
        )
        if demo and case == "astronaut" and args.strength == 1:
            before = (x - r).square().mean()
            after = (outputs["AdaIN"] - r).square().mean()
            assert after < before * 0.01, "Demo color-cast correction regressed"
        output = args.output / case if args.suite and case != "astronaut" else args.output
        draw(outputs, output, aligned=aligned, save_images=args.save_images, region=REGIONS[case])
        if args.suite:
            suite_arrays[case] = {
                name: value[0].permute(1, 2, 0).cpu().numpy() for name, value in outputs.items()
            }
        print(f"{case}: validated {len(outputs) - 2} methods; saved to {output}", flush=True)
    if args.suite:
        draw_suite(suite_arrays, args.output)


if __name__ == "__main__":
    main()
