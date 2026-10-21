"""Single-scale deformable attention implemented with PyTorch grid_sample.

See docs/implementation.md for the relationship to the research backup and MMCV.
This implementation does not require compiled third-party attention operators.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._validation import positive_int


class DeformableAttention2d(nn.Module):
    """Attend from [B, H*W, D] queries to a [B, D, H, W] value map.

    Offsets are measured in feature cells, with zero padding outside the map.
    Sampling weights are normalized over points independently for each head.
    The output is projected but has no residual addition or normalization.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.embed_dim = positive_int(embed_dim, "embed_dim")
        self.num_heads = positive_int(num_heads, "num_heads")
        self.num_points = positive_int(num_points, "num_points")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize uniform attention and a radial sampling pattern."""
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        with torch.no_grad():
            angles = torch.arange(
                self.num_heads,
                device=self.sampling_offsets.bias.device,
                dtype=self.sampling_offsets.bias.dtype,
            ) * (2 * math.pi / self.num_heads)
            directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
            directions = directions / directions.abs().amax(-1, keepdim=True)
            radius = torch.arange(1, self.num_points + 1, device=angles.device, dtype=angles.dtype)
            offsets = directions[:, None, :] * radius[None, :, None]
            self.sampling_offsets.bias.copy_(offsets.flatten())
        for projection in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self, query: torch.Tensor, values: torch.Tensor, reference_points: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = values.shape
        tokens = height * width
        if query.shape != (batch, tokens, self.embed_dim) or channels != self.embed_dim:
            raise ValueError("query and values must match the configured grid and embed_dim")
        if reference_points.shape != (tokens, 2):
            raise ValueError("reference_points must have shape [H*W, 2]")
        heads, points = self.num_heads, self.num_points
        per_head = channels // heads
        projected = self.value_proj(values.flatten(2).transpose(1, 2))
        projected = projected.transpose(1, 2).reshape(batch * heads, per_head, height, width)
        offsets = self.sampling_offsets(query).reshape(batch, tokens, heads, points, 2)
        weights = self.attention_weights(query).reshape(batch, tokens, heads, points)
        weights = weights.softmax(dim=-1)

        # grid_sample lacks some low-precision device kernels. Sample in float32
        # for fp16/bfloat16, retaining float64 for numerical/gradient checks.
        sample_dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
        normalizer = torch.tensor([width, height], device=query.device, dtype=sample_dtype)
        locations = reference_points.to(sample_dtype)[None, :, None, None, :]
        locations = locations + offsets.to(sample_dtype) / normalizer
        sampling_grid = (2 * locations - 1).permute(0, 2, 1, 3, 4)
        sampling_grid = sampling_grid.reshape(batch * heads, tokens, points, 2)
        sampled = F.grid_sample(
            projected.to(sample_dtype),
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        weights = weights.permute(0, 2, 1, 3).reshape(batch * heads, 1, tokens, points)
        combined = (sampled * weights.to(sample_dtype)).sum(-1)
        combined = combined.reshape(batch, channels, tokens).transpose(1, 2)
        return self.output_proj(combined.to(projected.dtype))
