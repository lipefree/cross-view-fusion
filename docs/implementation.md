# Implementation and reconstruction notes

## Scope and evidence

This release was prepared from an earlier CCVPE-based research backup and
[arXiv:2606.10166v1, Section III](https://arxiv.org/html/2606.10166v1#S3).
The final implementation used to obtain the paper's results was unavailable.
This is a reconstruction for reuse on downstream tasks, not a recovered benchmark
artifact. No equivalence to the reported localization results is claimed.

## Source mapping

Paths below refer to the archived research backup, which is not shipped here.

| Component | Backup evidence | Release behavior |
|---|---|---|
| Shared learned queries | `CCVPE/model_utils/fused_image_deformable_fusion_v2.py`, `deformable_fusion` | One learned query per feature cell, shared across modality branches |
| Modality projections and attention | Same file, `deformable_cross_attention` | Independent projections and single-scale attention for each modality |
| Positional information | `CCVPE/model_utils/position_encoding.py`, `PositionEmbeddingLearned` | Fixed 2D sine/cosine encoding, as described in the paper |
| Fusion rule | Scalar sigmoid weights or a token-averaged adaptive scalar in the backup | Two logits and softmax weights independently per patch |
| Multiple scales | Four-level MMCV attention and a learned output pyramid | Separate complete fusion instance for each scale |
| Task integration | CCVPE encoders, decoders, losses, and dataset paths | Caller supplies encoders, task head, data, and loss |

Unused branches, image reconstruction, MultiMAE experiments, scalar-weight
ablations, and extra fusion losses from the backup are excluded.

## Explicit implementation choices

The paper does not specify every detail needed to reconstruct an executable model.
The following decisions are part of this release, not verified historical facts:

1. **Patch descriptor:** average the two processed modality maps elementwise,
   then spatially average each patch into a C-dimensional vector. Use
   `Linear(C, hidden_dim) → ReLU → Linear(hidden_dim, 2)`, divide logits by
   `sqrt(C)`, and softmax across the two modalities. The paper does not fully
   specify how the patch tensor is reduced for the MLP; spatial mean pooling is
   the explicit reduction used here. Weights are broadcast over all channels
   and cells in that patch.
2. **Reference coordinates:** cell centers `((x + 0.5)/W, (y + 0.5)/H)`, with x
   first and row-major token ordering. This differs from the backup's endpoint
   grid. Sampling is bilinear with zero padding and `align_corners=False`.
3. **Attention:** multi-head, single-level deformable attention; offsets and
   sampling logits are linear functions of the shared query plus position.
   Both branches have their own attention parameters. Project values and outputs
   with linear layers, following the structure of the backup's MMCV operator.
   There is no dot-product key/query attention.
4. **Residuals and normalization:** no query residual, post-fusion residual,
   dropout, GroupNorm, or extra feed-forward block is added. The backup inherited
   an MMCV query residual and added another residual after fusion; those are not
   specified by the paper's fusion equations. This release treats attention as
   the projected weighted sampling operation only.
5. **Initialization:** queries use a normal distribution with standard deviation
   0.02; sampling offsets start in a radial pattern, with radii 1 through K;
   sampling logits initially yield uniform weights. Value/output projections use
   Xavier initialization. Other layers use PyTorch defaults. These are not
   recovered paper hyperparameters.
6. **Shared context:** the query parameters are shared and trained jointly, but
   the single attention layer does not concatenate input modalities or make one
   modality's sampling offsets depend directly on the other's current image.
   Content-dependent interaction occurs in the joint patch descriptor. This
   follows the explicit shared-query formulation rather than adding an
   undocumented cross-input attention stage.
7. **Portability:** the backend uses PyTorch `grid_sample`, replacing the MMCV
   extension. It is a reference implementation for ease of installation, not an
   optimized kernel. No numerical parity with MMCV or the full backup model is
   claimed. Rectangular grids and unequal encoder channels are supported.

The patch weights form a convex combination of the *processed* features. This
does not imply that the complete module is a convex combination of raw inputs.
The two modality branches need not behave identically under input swapping.

## Validation and limits

Tests independently calculate patch descriptors and mixtures, check exact spatial
regions and head ordering, and use known bilinear interpolation cases to verify
sampling. Numerical gradient checks cover patch fusion and deformable attention.
Integration tests check task gradients, per-scale independence, noncontiguous
inputs, CPU autocast, and state-dictionary round trips.

The initial local validation used Python 3.12.5, PyTorch 2.4.0, and pytest 8.3.5
on macOS ARM64 CPU: 34 tests passed and the CUDA test was skipped. Both examples
ran successfully, and wheel/source archive builds succeeded. CI additionally
checks PyTorch 2.4.1 and the current PyTorch release on Linux CPU.

CUDA parity/backward coverage is conditional on hardware availability. CPU CI
cannot validate CUDA performance or determinism. GPU benchmarks, real dataset
training, published localization metrics, historical checkpoint conversion, and
full-model comparison against the backup are outside this release's validation.

## Data and benchmark availability

The paper describes map-data availability through the project repository. This
module-only release does not include or host those tiles, map-generation scripts,
or dataset download links that could not be verified from the backup. It also
does not include pretrained fusion weights or CCVPE/HC-Net/Loc² reproduction
pipelines. Users supply aligned data and train on their task.

## References and provenance

- [Paper](https://arxiv.org/abs/2606.10166), Sections III-A–C.
- [CCVPE](https://arxiv.org/abs/2303.05915), the baseline represented in the backup.
- [Deformable DETR](https://arxiv.org/abs/2010.04159), the deformable sampling method.
- [MMCV attention source, v2.2.0](https://github.com/open-mmlab/mmcv/blob/v2.2.0/mmcv/ops/multi_scale_deform_attn.py), consulted to understand operator behavior.
- [DETR positional encodings](https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py), credited by the backup's positional-encoding file.

See [NOTICE](../NOTICE) and [LICENSE](../LICENSE). Public API defaults are intended
for configuration, not as a claim of recovered paper experiment settings.
