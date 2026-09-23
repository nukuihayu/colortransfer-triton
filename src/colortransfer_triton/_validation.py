import math

import torch


def integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def scalar(value, name, minimum=0.0, maximum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite Python scalar")
    if (
        value < minimum
        or (strict and value == minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is out of range")
    return float(value)


def boolean(value, name):
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


def tensor(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError(f"{name} must be on an NVIDIA CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16 or float32")
    if x.requires_grad:
        raise ValueError(f"{name} requires gradients; detach explicitly for inference")
    if x.numel() == 0 or x.numel() > 2147483647:
        raise ValueError(f"{name} must be nonempty and fit int32 indexing")


def image(x, name="source"):
    tensor(x, name)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3,H,W)")


def pair(source, reference, aligned=False):
    image(source)
    image(reference, "reference")
    if reference.device != source.device or reference.shape[0] not in (1, source.shape[0]):
        raise ValueError("reference must share device and have batch 1 or the source batch size")
    if aligned and reference.shape[-2:] != source.shape[-2:]:
        raise ValueError("wavelet requires matching spatial sizes; align/resize reference first")


def points(x, name="points", maximum=None):
    tensor(x, name)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must have shape (M,3)")
    if maximum is not None and x.shape[0] > maximum:
        raise ValueError(f"{name} exceeds {maximum} points; sample the image first")
