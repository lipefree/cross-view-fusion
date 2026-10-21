"""Train a toy dense prediction model with two encoders and feature fusion."""

import torch
from torch import nn

from cross_view_fusion import CrossViewFusion


def main() -> None:
    torch.manual_seed(7)
    encoder_a = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    encoder_b = nn.Conv2d(5, 24, kernel_size=3, stride=2, padding=1)
    fusion = CrossViewFusion(
        (16, 24), feature_size=(8, 12), embed_dim=32, patch_size=(2, 3), num_heads=4
    )
    head = nn.Conv2d(32, 3, kernel_size=1)
    model = nn.ModuleDict(dict(encoder_a=encoder_a, encoder_b=encoder_b, fusion=fusion, head=head))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    a, b = torch.randn(2, 3, 16, 24), torch.randn(2, 5, 16, 24)
    target = torch.randint(3, (2, 8, 12))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    features, weights = fusion(encoder_a(a), encoder_b(b), return_weights=True)
    loss = nn.functional.cross_entropy(head(features), target)
    loss.backward()
    for encoder in (encoder_a, encoder_b):
        assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    optimizer.step()
    print(
        f"features={tuple(features.shape)}, weights={tuple(weights.shape)}, loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
