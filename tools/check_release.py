"""Validate source version, release tag and distributable contents (Python 3.11+)."""

import ast
import email
import os
import tarfile
import zipfile
from pathlib import Path

import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    tag = os.environ.get("RELEASE_TAG", "")
    if tag and tag != f"v{version}":
        raise SystemExit(f"Tag {tag} does not match v{version}")
    tree = ast.parse((root / "src/colortransfer_triton/__init__.py").read_text())
    source_version = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    assert source_version == version, "Source version differs from package version"
    wheels = list((root / "dist").glob("*.whl"))
    sources = list((root / "dist").glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Exactly one wheel and sdist required"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        meta = email.message_from_bytes(
            archive.read(next(n for n in names if n.endswith("/METADATA")))
        )
        assert meta["Version"] == version
        assert "colortransfer_triton/_kernels.py" in names
        assert "colortransfer_triton/colorfm.py" in names
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        for path in (
            "README.md",
            "README.zh-CN.md",
            "LICENSE",
            "assets/color-transfer.png",
            "assets/color-transfer.pdf",
            "examples/color_transfer.py",
            "examples/data/astronaut.png",
            "examples/data/chelsea.png",
            "examples/data/coffee.png",
            "assets/chelsea/color-transfer.png",
            "assets/astronaut-to-coffee/color-transfer.png",
            "assets/cross-image/color-transfer.png",
            "assets/detail.png",
            "tools/check_release.py",
            "benchmarks/benchmark.py",
            "tests/reference.py",
            "tests/test_gpu.py",
            "tests/test_colorfm.py",
            ".github/workflows/ci.yml",
        ):
            assert any(n.endswith("/" + path) for n in names), f"Missing {path}"
        metadata = next(n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO"))
        assert email.message_from_bytes(archive.extractfile(metadata).read())["Version"] == version
    print(f"Validated colortransfer-triton {version}")


if __name__ == "__main__":
    main()
