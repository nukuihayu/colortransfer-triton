import pytest
import torch

import colortransfer_triton as ct
from colortransfer_triton import _validation as check


def test_exports():
    for name in ct.__all__:
        assert hasattr(ct, name)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, None])
def test_invalid_scalar(value):
    with pytest.raises(ValueError):
        check.scalar(value, "test")


def test_cpu_contracts():
    with pytest.raises(TypeError):
        check.image(None)
    with pytest.raises(ValueError):
        check.image(torch.ones(1, 3, 2, 2))
    with pytest.raises(ValueError):
        check.integer(2.5, "count")
    with pytest.raises(TypeError):
        check.boolean(1, "clamp")
    with pytest.raises(ValueError):
        ct.transfer(None, None, method="unknown")
