"""Independent fusion modules for aligned feature pyramids."""

from collections.abc import Sequence

import torch
from torch import nn

from .fusion import CrossViewFusion


class MultiScaleFusion(nn.Module):
    """Apply explicitly configured CrossViewFusion instances in scale order.

    Pass distinct modules for each scale; parameters are not shared. Inputs are
    sequences in the same order. Output is a list, or (features, weights) lists
    when return_weights=True. This is not multi-level deformable attention.
    """

    def __init__(self, modules: Sequence[CrossViewFusion]):
        super().__init__()
        modules = list(modules)
        if not modules or not all(isinstance(m, CrossViewFusion) for m in modules):
            raise ValueError("modules must be a nonempty sequence of CrossViewFusion instances")
        if len({id(m) for m in modules}) != len(modules):
            raise ValueError("each scale must have its own independent module instance")
        self.scales = nn.ModuleList(modules)

    def forward(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        return_weights: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            raise ValueError("inputs must be sequences of feature maps")
        if len(a) != len(self.scales) or len(b) != len(self.scales):
            raise ValueError("input sequences must contain one feature map per configured scale")
        outputs = [
            module(x, y, return_weights=return_weights) for module, x, y in zip(self.scales, a, b)
        ]
        if return_weights:
            return [item[0] for item in outputs], [item[1] for item in outputs]
        return outputs
