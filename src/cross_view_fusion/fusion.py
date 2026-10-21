"""Shared-query feature processing followed by patch-level fusion."""

import torch
from torch import nn

from ._validation import aligned_features, pair, positive_int
from .attention import DeformableAttention2d
from .patch_fusion import PatchFusion
from .positional_encoding import reference_grid, sine_position_encoding


class CrossViewFusion(nn.Module):
    """Fuse two aligned [B, C, H, W] feature maps at a configured resolution.

    Args:
        in_channels: Shared channel count or (channels_a, channels_b).
        feature_size: Spatial extent in feature cells, as int or (H, W).
        embed_dim: Output channels; divisible by 4 and num_heads.
        patch_size: Patch extent in feature cells; must divide feature_size.
        num_heads: Deformable attention heads per modality.
        num_points: Sampling points per query and head.
        hidden_dim: Patch-weight MLP hidden width.

    Learnable queries are shared by the two modality branches. Each branch has
    independent projections and attention parameters. The grid is fixed for a
    module instance; create another instance for another scale. The output is
    [B, embed_dim, H, W]. Optional weights are [B, 2, H/ph, W/pw].
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        feature_size: int | tuple[int, int],
        *,
        embed_dim: int = 256,
        patch_size: int | tuple[int, int] = 1,
        num_heads: int = 8,
        num_points: int = 4,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.in_channels = pair(in_channels, "in_channels")
        self.feature_size = pair(feature_size, "feature_size")
        self.embed_dim = positive_int(embed_dim, "embed_dim")
        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by 4 for sine positional encoding")
        patch = pair(patch_size, "patch_size")
        if any(size % extent for size, extent in zip(self.feature_size, patch)):
            raise ValueError("feature_size must be divisible by patch_size")
        height, width = self.feature_size
        self.query = nn.Parameter(torch.empty(height * width, embed_dim))
        nn.init.normal_(self.query, std=0.02)
        grid = reference_grid(height, width)
        self.register_buffer("reference_points", grid, persistent=False)
        self.register_buffer(
            "position_encoding", sine_position_encoding(grid, embed_dim), persistent=False
        )
        self.projection_a = nn.Conv2d(self.in_channels[0], embed_dim, 1)
        self.projection_b = nn.Conv2d(self.in_channels[1], embed_dim, 1)
        self.attention_a = DeformableAttention2d(embed_dim, num_heads, num_points)
        self.attention_b = DeformableAttention2d(embed_dim, num_heads, num_points)
        self.patch_fusion = PatchFusion(embed_dim, patch, hidden_dim)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, *, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        aligned_features(a, b)
        if (a.shape[1], b.shape[1]) != self.in_channels:
            raise ValueError(f"input channels must be {self.in_channels}")
        if tuple(a.shape[2:]) != self.feature_size:
            raise ValueError(f"input spatial dimensions must be {self.feature_size}")
        batch = a.shape[0]
        query = (self.query + self.position_encoding).unsqueeze(0).expand(batch, -1, -1)
        processed_a = self.attention_a(query, self.projection_a(a), self.reference_points)
        processed_b = self.attention_b(query, self.projection_b(b), self.reference_points)
        shape = (batch, self.embed_dim, *self.feature_size)
        processed_a = processed_a.transpose(1, 2).reshape(shape)
        processed_b = processed_b.transpose(1, 2).reshape(shape)
        return self.patch_fusion(processed_a, processed_b, return_weights=return_weights)
