# Publishing colortransfer-triton

## One-time PyPI setup

Configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
on PyPI (use a pending publisher if the project does not exist yet):

| Field | Value |
| --- | --- |
| PyPI project | `colortransfer-triton` |
| GitHub owner | `nukuihayu` |
| GitHub repository | `colortransfer-triton` |
| Workflow filename | `ci.yml` |
| Environment | `pypi` |

The GitHub repository must have an environment named `pypi`. No PyPI API token
or GitHub Actions secret is needed; the publish job uses GitHub OIDC.

## Release a version

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/colortransfer_triton/__init__.py` to the same new version.
2. Commit and push the changes to `main`.
3. Create and push a matching version tag:

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

Use the new version instead of `0.1.0` for subsequent releases. Never move an
existing release tag or reuse a version already uploaded to PyPI.

The tag runs lint, builds a wheel and source distribution, checks package
metadata and versions, and tests on Python 3.10, 3.12 and 3.13. Only after these
checks pass does it attach the tested artifacts to a GitHub Release and publish
the same artifacts to PyPI. Branch pushes and pull requests do not publish.
GPU numerical validation remains available separately via the manual GPU
workflow on a self-hosted runner labeled `self-hosted`, `linux`, `x64`, `gpu`.

These packages ship Triton source in a `py3-none-any` wheel. GPU kernels are
compiled by Triton at runtime on the user's GPU; publishing does not require
building a CUDA extension or a wheel per CUDA/Python version.

If PyPI authentication fails, correct the Trusted Publisher settings and rerun
the failed job. PyPI versions are immutable; a successful upload must be followed
by a version bump for the next release. Do not pre-create the GitHub Release;
the workflow creates it, or refreshes its assets if it already exists.
