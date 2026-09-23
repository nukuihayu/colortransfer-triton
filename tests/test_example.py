"""The shipped photo example must work from the installed package too."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
def test_photo_example(tmp_path):
    pytest.importorskip("matplotlib")
    script = Path(__file__).resolve().parents[1] / "examples/color_transfer.py"
    subprocess.run(
        [sys.executable, str(script), "--suite", "--output", str(tmp_path), "--save-images"],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    for group, width in (("aligned", 5376), ("unaligned", 4704)):
        with Image.open(tmp_path / f"{group}.png") as image:
            assert image.size == (width, 1280)
        assert (tmp_path / f"{group}.pdf").stat().st_size > 1000
    for name in ("color-transfer", "detail"):
        for extension in ("png", "pdf"):
            assert (tmp_path / f"{name}.{extension}").stat().st_size > 1000
        with Image.open(tmp_path / f"{name}.png") as image:
            assert image.width >= 8 * 512 and image.width > 5 * image.height
    for name in (
        "source",
        "reference",
        "adain",
        "wavelet",
        "sliced-ot",
        "sinkhorn",
        "partial-sinkhorn",
        "colorfm-o",
    ):
        with Image.open(tmp_path / "images" / f"{name}.png") as image:
            assert image.size == (512, 512) and image.mode == "RGB"
    for case, shape in (
        ("chelsea", (451, 300)),
        ("astronaut-to-coffee", (512, 512)),
        ("cross-image", (600, 400)),
    ):
        folder = tmp_path / case
        for name in ("color-transfer", "detail"):
            with Image.open(folder / f"{name}.png") as image:
                assert image.width >= 7 * 512 and image.width > 5 * image.height
            assert (folder / f"{name}.pdf").stat().st_size > 1000
        for name in ("source", "adain", "sliced-ot", "sinkhorn", "partial-sinkhorn", "colorfm-o"):
            with Image.open(folder / "images" / f"{name}.png") as image:
                assert image.size == shape and image.mode == "RGB"
        assert (folder / "images/wavelet.png").exists() == (case == "chelsea")
        with Image.open(folder / "images/reference.png") as image:
            expected = (600, 400) if case == "astronaut-to-coffee" else (451, 300)
            assert image.size == expected
