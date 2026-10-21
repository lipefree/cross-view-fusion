"""Cell-centered reference coordinates and deterministic 2D sine encoding."""

import math

import torch


def reference_grid(height: int, width: int) -> torch.Tensor:
    """Return [H*W, 2] normalized (x, y) coordinates in row-major order."""
    y, x = torch.meshgrid(
        (torch.arange(height, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(height * width, 2)


def sine_position_encoding(grid: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Encode y then x with paired sine/cosine frequencies; D must divide by 4."""
    if embed_dim < 4 or embed_dim % 4:
        raise ValueError("embed_dim must be a positive multiple of 4")
    frequency = 10000 ** (
        torch.arange(embed_dim // 4, device=grid.device, dtype=grid.dtype) / (embed_dim // 4)
    )
    phase = grid.flip(-1).unsqueeze(-1) * (2 * math.pi) / frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
