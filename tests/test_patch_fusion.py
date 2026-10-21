import math

import pytest
import torch
from torch import nn

from cross_view_fusion import PatchFusion


def test_weighted_sum_matches_independent_patch_loop():
    torch.manual_seed(1)
    fusion = PatchFusion(4, patch_size=(2, 3), hidden_dim=5).double()
    a, b = torch.randn(2, 4, 4, 9, dtype=torch.double), torch.randn(2, 4, 4, 9, dtype=torch.double)
    fused, weights = fusion(a, b, return_weights=True)
    expected = torch.empty_like(a)
    for row in range(2):
        for col in range(3):
            region = (..., slice(row * 2, (row + 1) * 2), slice(col * 3, (col + 1) * 3))
            pa, pb = a[region], b[region]
            descriptor = ((pa + pb) / 2).mean(dim=(-2, -1))
            logits = fusion.mlp(descriptor) / math.sqrt(4)
            alpha = logits.softmax(-1)
            expected[region] = (
                alpha[:, 0, None, None, None] * pa + alpha[:, 1, None, None, None] * pb
            )
            torch.testing.assert_close(weights[:, :, row, col], alpha)
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    assert (weights >= 0).all() and (weights <= 1).all()


def test_equal_logits_average_and_preserve_identical_inputs():
    fusion = PatchFusion(4, (2, 3))
    for parameter in fusion.parameters():
        nn.init.zeros_(parameter)
    a, b = torch.randn(2, 4, 4, 6), torch.randn(2, 4, 4, 6)
    torch.testing.assert_close(fusion(a, b), (a + b) / 2)
    torch.testing.assert_close(fusion(a, a), a)


def test_weights_can_vary_by_patch_and_select_either_modality():
    fusion = PatchFusion(1, (2, 2), hidden_dim=1)
    with torch.no_grad():
        fusion.mlp[0].weight.fill_(1)
        fusion.mlp[0].bias.zero_()
        fusion.mlp[2].weight.copy_(torch.tensor([[-20.0], [20.0]]))
        fusion.mlp[2].bias.copy_(torch.tensor([10.0, -10.0]))
    a = torch.tensor([[[[-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]]]])
    b = a + 1
    fused, weights = fusion(a, b, return_weights=True)
    assert weights[0, 0, 0, 0] > 0.999
    assert weights[0, 1, 0, 1] > 0.999
    torch.testing.assert_close(fused[..., :2], a[..., :2])
    torch.testing.assert_close(fused[..., 2:], b[..., 2:])


def test_patch_fusion_gradcheck():
    torch.manual_seed(4)
    fusion = PatchFusion(2, 2).double()
    a, b = (torch.randn(1, 2, 2, 2, dtype=torch.double, requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(fusion, (a, b))


@pytest.mark.parametrize("shape", [(1, 4, 5, 6), (1, 3, 4, 6), (4, 4, 6)])
def test_invalid_shapes_fail(shape):
    with pytest.raises(ValueError):
        PatchFusion(4, 2)(torch.randn(shape), torch.randn(shape))


@pytest.mark.parametrize("patch", [0, -1, True, (2, 0), (2, 3, 4)])
def test_invalid_patch_configuration(patch):
    with pytest.raises(ValueError):
        PatchFusion(4, patch)
