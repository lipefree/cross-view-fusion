# Integrating fusion into your task

## Input contract

Both modalities use tensors in `[batch, channels, height, width]` order. Batch,
height, width, device, and floating-point dtype must match. Channel counts may
differ for `CrossViewFusion`, which projects each branch to `embed_dim` using a
separate 1×1 convolution. `PatchFusion` requires equal channel counts.

Inputs must already be spatially aligned. Resize, crop, rotate, and register the
source modalities consistently before extracting features. A patch in one input
must refer to the same region as that patch in the other. This package does not
solve geometric registration, missing-modality inference, or correspondence
between a ground image and an overhead image.

## Single-scale configuration

```python
from cross_view_fusion import CrossViewFusion

fusion = CrossViewFusion(
    in_channels=(64, 96),  # or one integer when both encoders output the same C
    feature_size=(16, 24), # H, W; one integer means a square
    embed_dim=64,         # fused output channels; divisible by 4 and num_heads
    patch_size=(4, 4),    # feature cells, not original image pixels
    num_heads=4,
    num_points=4,
    hidden_dim=32,        # defaults to max(embed_dim // 2, 1)
)
```

`feature_size` is required. The module learns a distinct query at every feature
location and does not resize those parameters during forward passes. Each patch
extent must divide its feature dimension exactly; invalid sizes raise a
`ValueError`. There is no automatic padding or truncation. Rectangular grids and
rectangular patches are supported as an extension of the square-grid description
in the paper.

The default `patch_size=1` gives weights per feature cell. Larger values make
weights constant over a region. These defaults are API choices, not recovered
paper training configurations. Choose a patch size suitable for your resolution.

## Train with your existing loss

Create fusion in your model's `__init__`, so its parameters are registered and
included in the optimizer. Do not recreate it in `forward`.

```python
import torch
from torch import nn
from cross_view_fusion import CrossViewFusion

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_a = nn.Conv2d(3, 16, 3, padding=1)
        self.encoder_b = nn.Conv2d(5, 24, 3, padding=1)
        self.fusion = CrossViewFusion(
            (16, 24), (16, 24), embed_dim=32, patch_size=4, num_heads=4
        )
        self.head = nn.Conv2d(32, 3, 1)

    def forward(self, a, b):
        features = self.fusion(self.encoder_a(a), self.encoder_b(b))
        return self.head(features)

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
a, b = torch.randn(2, 3, 16, 24), torch.randn(2, 5, 16, 24)
targets = torch.randint(3, (2, 16, 24))
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.cross_entropy(model(a, b), targets)
loss.backward()
optimizer.step()
```

There is no required fusion-specific loss. If your existing task head expects a
different channel count, set `embed_dim` appropriately or add your own output
projection. The package does not freeze or initialize your encoders.

## Patch fusion alone

```python
from cross_view_fusion import PatchFusion

fusion = PatchFusion(channels=64, patch_size=4)
# fused, weights = fusion(features_a, features_b, return_weights=True)
```

Weights are `[B, 2, H/ph, W/pw]`: index 0 weights the first input and index 1 the
second. They are differentiable. For plotting, detach them and expand each patch
to its feature cells:

```python
# dense = weights.detach().repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
```

They describe the final mixture of processed features, not calibrated confidence
or a complete explanation of the model.

## Multiple scales

```python
from cross_view_fusion import CrossViewFusion, MultiScaleFusion

fusion = MultiScaleFusion([
    CrossViewFusion(32, (16, 24), embed_dim=32, patch_size=4, num_heads=4),
    CrossViewFusion((64, 96), (8, 12), embed_dim=64, patch_size=2, num_heads=4),
])
# outputs, weights = fusion(pyramid_a, pyramid_b, return_weights=True)
```

Lists must contain exactly one map per configured scale, in the same order.
Parameters are independent across scales. There is no attention across scales,
automatic scale matching, pyramid generation, or shared module instance.

## Devices, precision, and checkpoints

Move the complete model and inputs with `.to(device)`. Registered reference-grid
and positional-encoding buffers follow device and dtype changes. Sampling uses
float32 for float16/bfloat16 features and float64 for double precision. CUDA
`grid_sample` backward may be nondeterministic. GPU speed, memory use at large
resolutions, and Apple MPS behavior must be measured in your environment.

Save the constructor configuration alongside the state dictionary:

```python
config = dict(in_channels=(16, 24), feature_size=(8, 12), embed_dim=32,
              patch_size=(2, 3), num_heads=4)
module = CrossViewFusion(**config)
torch.save({"config": config, "state_dict": module.state_dict()}, "fusion.pt")
checkpoint = torch.load("fusion.pt", map_location="cpu", weights_only=True)
restored = CrossViewFusion(**checkpoint["config"])
restored.load_state_dict(checkpoint["state_dict"])
```

Fixed positional buffers are reconstructed from configuration. Historical CCVPE
checkpoints are not compatible with this reconstructed architecture.
