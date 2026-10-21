"""Fuse two feature pyramids with independently configured scales."""

import torch

from cross_view_fusion import CrossViewFusion, MultiScaleFusion


def main() -> None:
    torch.manual_seed(7)
    fusion = MultiScaleFusion(
        [
            CrossViewFusion(16, (8, 12), embed_dim=16, patch_size=2, num_heads=4),
            CrossViewFusion((24, 32), (4, 6), embed_dim=32, patch_size=2, num_heads=4),
        ]
    )
    a = [torch.randn(2, 16, 8, 12), torch.randn(2, 24, 4, 6)]
    b = [torch.randn(2, 16, 8, 12), torch.randn(2, 32, 4, 6)]
    fused, weights = fusion(a, b, return_weights=True)
    sum(feature.square().mean() for feature in fused).backward()
    for index, (feature, weight) in enumerate(zip(fused, weights)):
        print(f"scale {index}: features={tuple(feature.shape)}, weights={tuple(weight.shape)}")


if __name__ == "__main__":
    main()
