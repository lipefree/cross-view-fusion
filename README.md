# Cross-View Fusion

Reusable PyTorch modules accompanying **[Fusing Satellite Imagery and Planimetric
Maps for Cross-View Localization](https://arxiv.org/abs/2606.10166)**, by Quang Long
Ho Ngo, Zimin Xia, and Alexandre Alahi.

Fuse two aligned feature maps between your own encoders and task head. The package
provides deformable attention with shared learnable queries, content-dependent
patch weighting, and independent fusion at multiple feature scales.

```text
modality A → your encoder A ─┐
                           ├→ CrossViewFusion → your task head → your task loss
modality B → your encoder B ─┘
```

**Release scope:** this modular implementation was reconstructed from an earlier
research backup and the paper. It is not the exact training code used for the
reported benchmarks, and benchmark equivalence has not been established. See
[implementation notes](docs/implementation.md) for the source mapping and explicit
reconstruction choices. This release contains no pretrained weights, datasets,
OSM tiles, or full localization training/evaluation pipelines.

## Install

Requires Python 3.10+ and PyTorch 2.4+. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then create the
project environment (the commands below use Python 3.12):

```bash
git clone https://github.com/lipefree/cross-view-fusion.git
cd cross-view-fusion
uv sync --python 3.12
```

`uv sync` creates `.venv` and installs the package and its dependencies. Use
`uv run` to run commands in that environment without activating it. For a
specific CUDA or CPU PyTorch build, follow the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

The only runtime dependency is PyTorch. No MMCV, custom CUDA extension, backbone,
or dataset download is required. CPU is the default for the examples. GPU support
uses PyTorch's `grid_sample`; see the notes for validation coverage and limitations.

## Quick start

```python
import torch
from cross_view_fusion import CrossViewFusion

fusion = CrossViewFusion(
    in_channels=(64, 96),
    feature_size=(16, 24),
    embed_dim=64,
    patch_size=(4, 4),
    num_heads=4,
    num_points=4,
)
a = torch.randn(2, 64, 16, 24)
b = torch.randn(2, 96, 16, 24)
fused, weights = fusion(a, b, return_weights=True)
# fused:   [2, 64, 16, 24]
# weights: [2, 2, 4, 6], with modality order (a, b)
fused.square().mean().backward()  # replace with your task head and loss
```

Inputs must describe the same spatial area with matching orientation, resolution,
and pixel alignment. Fusion does not register or warp unrelated views into a
common coordinate system. Patch sizes are measured in **feature cells**, and must
divide both spatial dimensions. Each module has a fixed feature grid because its
queries are learned per location. Parameters must be trained on your task.

## Modules

| Module | Input | Output |
|---|---|---|
| `CrossViewFusion` | Two aligned maps, possibly different channel counts | Processed and fused map |
| `PatchFusion` | Two aligned maps with the same channel count | Patch-weighted map without attention |
| `MultiScaleFusion` | Two lists of maps, with one fusion module per scale | List of fused maps |

All modules optionally return patch weights. The two modality weights sum to one
at each patch. Full constructor options, integration patterns, device handling,
and checkpoint instructions are in the [integration guide](docs/integration.md).

## Runnable examples

```bash
uv run python examples/minimal.py
uv run python examples/multiscale.py
```

The first example runs a training step through two small encoders, fusion, and a
dense prediction head. The second handles a feature pyramid. Both use synthetic
data and run without external files or network access after installation.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv build
```

Tests cover patch equations and spatial placement, bilinear sampling, attention
head ordering, numerical gradients, both encoder branches, serialization, and
input validation. CUDA tests run when CUDA is available; CPU CI does not establish
GPU performance or reproduce the paper's benchmark results.

## Citation

```bibtex
@misc{ngo2026fusing,
  title={Fusing Satellite Imagery and Planimetric Maps for Cross-View Localization},
  author={Quang Long Ho Ngo and Zimin Xia and Alexandre Alahi},
  year={2026},
  eprint={2606.10166},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2606.10166}
}
```

## License and acknowledgments

Distributed under [GPL-3.0](LICENSE), retaining the license supplied with the
research backup. See [NOTICE](NOTICE) for provenance and acknowledgments of CCVPE,
Deformable DETR, MMCV, and DETR. The paper and external projects have their own
licenses. No third-party backbone implementation is bundled.
