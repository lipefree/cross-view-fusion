import io

import pytest
import torch

from cross_view_fusion import CrossViewFusion, MultiScaleFusion


def small_fusion(**kwargs):
    return CrossViewFusion((3, 5), (4, 6), embed_dim=8, patch_size=2, num_heads=2, **kwargs)


def test_forward_and_task_gradients_reach_both_branches_and_queries():
    torch.manual_seed(5)
    fusion = small_fusion()
    a, b = torch.randn(2, 3, 4, 6, requires_grad=True), torch.randn(2, 5, 4, 6, requires_grad=True)
    fused, weights = fusion(a, b, return_weights=True)
    assert fused.shape == (2, 8, 4, 6)
    assert weights.shape == (2, 2, 2, 3)
    fused.square().mean().backward()
    for tensor in (
        a,
        b,
        fusion.projection_a.weight,
        fusion.projection_b.weight,
        fusion.attention_a.sampling_offsets.weight,
        fusion.attention_b.sampling_offsets.weight,
        fusion.patch_fusion.mlp[-1].weight,
    ):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    # Initial offset/weight matrices are zero, so the query's first-step gradient
    # can be zero. After an optimizer step, queries must receive task gradients.
    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    fusion(a, b).square().mean().backward()
    assert fusion.query.grad is not None and fusion.query.grad.abs().sum() > 0


def test_state_dict_round_trip_and_double_precision():
    original, restored = small_fusion().double().eval(), small_fusion().double().eval()
    a, b = torch.randn(1, 3, 4, 6, dtype=torch.double), torch.randn(1, 5, 4, 6, dtype=torch.double)
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(original(a, b), restored(a, b))
    assert original.reference_points.dtype == torch.double
    assert original(a, b).dtype == torch.double


def test_noncontiguous_inputs():
    fusion = small_fusion()
    a = torch.randn(1, 3, 6, 4).transpose(-1, -2)
    b = torch.randn(1, 5, 6, 4).transpose(-1, -2)
    torch.testing.assert_close(fusion(a, b), fusion(a.contiguous(), b.contiguous()))


def test_cpu_autocast_backward_is_finite():
    fusion = small_fusion()
    a, b = torch.randn(1, 3, 4, 6), torch.randn(1, 5, 4, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fused = fusion(a, b)
        loss = fused.float().square().mean()
    loss.backward()
    assert torch.isfinite(fused).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())


def test_multiscale_outputs_match_independent_modules():
    scales = [small_fusion(), CrossViewFusion(8, (2, 3), embed_dim=8, num_heads=2)]
    fusion = MultiScaleFusion(scales)
    a = [torch.randn(1, 3, 4, 6), torch.randn(1, 8, 2, 3)]
    b = [torch.randn(1, 5, 4, 6), torch.randn(1, 8, 2, 3)]
    outputs, weights = fusion(a, b, return_weights=True)
    for module, x, y, output, weight in zip(scales, a, b, outputs, weights):
        expected, expected_weight = module(x, y, return_weights=True)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(weight, expected_weight)
    sum(x.square().mean() for x in outputs).backward()
    assert all(module.projection_a.weight.grad.abs().sum() > 0 for module in scales)
    assert set(map(id, scales[0].parameters())).isdisjoint(map(id, scales[1].parameters()))
    with pytest.raises(ValueError, match="one feature map"):
        fusion(a[:1], b)
    with pytest.raises(ValueError, match="independent"):
        MultiScaleFusion([scales[0], scales[0]])
    with pytest.raises(ValueError, match="nonempty"):
        MultiScaleFusion([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"feature_size": (3, 6)},
        {"embed_dim": 6},
        {"num_heads": 3},
        {"num_points": 0},
        {"hidden_dim": 0},
    ],
)
def test_invalid_configuration(kwargs):
    config = dict(in_channels=(3, 5), feature_size=(4, 6), embed_dim=8, patch_size=2, num_heads=2)
    config.update(kwargs)
    with pytest.raises(ValueError):
        CrossViewFusion(**config)


@pytest.mark.parametrize(
    "a,b",
    [
        (torch.randn(1, 3, 4, 6), torch.randn(2, 5, 4, 6)),
        (torch.randn(1, 3, 4, 6), torch.randn(1, 5, 4, 5)),
        (torch.randn(1, 3, 4, 6), torch.randn(1, 5, 4, 6).double()),
        (torch.ones(1, 3, 4, 6, dtype=torch.long), torch.ones(1, 5, 4, 6, dtype=torch.long)),
        (torch.randn(1, 3, 6, 4), torch.randn(1, 5, 6, 4)),
        (torch.randn(1, 4, 4, 6), torch.randn(1, 5, 4, 6)),
    ],
)
def test_invalid_inputs(a, b):
    with pytest.raises(ValueError):
        small_fusion()(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_supports_backward():
    cpu = small_fusion().eval()
    gpu = small_fusion().eval().cuda()
    gpu.load_state_dict(cpu.state_dict())
    a, b = torch.randn(2, 3, 4, 6), torch.randn(2, 5, 4, 6)
    output = gpu(a.cuda(), b.cuda())
    torch.testing.assert_close(output.cpu(), cpu(a, b), atol=1e-5, rtol=1e-4)
    output.square().mean().backward()
    assert gpu.projection_a.weight.grad.abs().sum() > 0
