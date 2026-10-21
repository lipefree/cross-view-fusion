"""Shared configuration and tensor validation."""

import torch


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        value = (value, value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an integer or a (height, width) pair")
    return positive_int(value[0], name), positive_int(value[1], name)


def aligned_features(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 4 or b.ndim != 4:
        raise ValueError("inputs must have shape [batch, channels, height, width]")
    if any(size < 1 for size in (*a.shape, *b.shape)):
        raise ValueError("input dimensions must be nonempty")
    if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ValueError("inputs must have matching batch and spatial dimensions")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("inputs must share a device and floating-point dtype")
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError("inputs must have a floating-point dtype")
