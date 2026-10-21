"""Content-dependent convex fusion of aligned feature patches."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._validation import aligned_features, pair, positive_int


class PatchFusion(nn.Module):
    """Fuse equal-channel inputs using one pair of weights per spatial patch.

    Args:
        channels: Number of channels in each input and in the output.
        patch_size: Patch extent in feature cells, as int or (height, width).
        hidden_dim: MLP hidden width; defaults to max(channels // 2, 1).

    The descriptor is the spatial mean of the elementwise modality average.
    Returned weights have shape [B, 2, H/ph, W/pw], in input argument order.
    Spatial dimensions must be divisible by the patch dimensions.
    """

    def __init__(
        self, channels: int, patch_size: int | tuple[int, int] = 1, hidden_dim: int | None = None
    ):
        super().__init__()
        self.channels = positive_int(channels, "channels")
        self.patch_size = pair(patch_size, "patch_size")
        hidden_dim = max(channels // 2, 1) if hidden_dim is None else hidden_dim
        positive_int(hidden_dim, "hidden_dim")
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, *, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        aligned_features(a, b)
        if a.shape[1] != self.channels or b.shape[1] != self.channels:
            raise ValueError(f"both inputs must have {self.channels} channels")
        height, width = a.shape[-2:]
        ph, pw = self.patch_size
        if height % ph or width % pw:
            raise ValueError("spatial dimensions must be divisible by patch_size")
        descriptor = F.avg_pool2d(a * 0.5 + b * 0.5, self.patch_size, self.patch_size)
        logits = self.mlp(descriptor.permute(0, 2, 3, 1)) / math.sqrt(self.channels)
        weights = logits.softmax(dim=-1).permute(0, 3, 1, 2)
        dense = weights.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
        fused = dense[:, :1] * a + dense[:, 1:] * b
        return (fused, weights) if return_weights else fused
