import pytest
import torch
from torch import nn

from cross_view_fusion.attention import DeformableAttention2d
from cross_view_fusion.positional_encoding import reference_grid, sine_position_encoding


def identity_attention(heads=2, points=1):
    attention = DeformableAttention2d(4, heads, points).double()
    with torch.no_grad():
        for projection in (attention.value_proj, attention.output_proj):
            nn.init.eye_(projection.weight)
            projection.bias.zero_()
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.zero_()
    return attention


def test_zero_offsets_recover_rectangular_feature_grid():
    attention = identity_attention()
    values = torch.arange(2 * 4 * 3 * 5, dtype=torch.double).reshape(2, 4, 3, 5)
    queries = torch.zeros(2, 15, 4, dtype=torch.double)
    result = attention(queries, values, reference_grid(3, 5).double())
    torch.testing.assert_close(result, values.flatten(2).transpose(1, 2), atol=2e-5, rtol=1e-6)


def test_half_cell_shift_uses_bilinear_sampling_and_zero_padding():
    attention = identity_attention(heads=1)
    with torch.no_grad():
        attention.sampling_offsets.bias.copy_(torch.tensor([0.5, 0.0]))
    values = torch.arange(24, dtype=torch.double).reshape(1, 4, 2, 3)
    result = attention(
        torch.zeros(1, 6, 4, dtype=torch.double), values, reference_grid(2, 3).double()
    )
    expected = values * 0.5
    expected[..., :-1] += values[..., 1:] * 0.5
    torch.testing.assert_close(result, expected.flatten(2).transpose(1, 2), atol=2e-6, rtol=1e-6)


def test_head_and_point_weights_are_applied_to_the_correct_channels():
    attention = identity_attention(heads=2, points=2)
    with torch.no_grad():
        attention.sampling_offsets.bias.copy_(
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        attention.attention_weights.bias.copy_(torch.tensor([0.25, 0.75, 0.8, 0.2]).log())
    values = torch.arange(24, dtype=torch.double).reshape(1, 4, 2, 3)
    output = attention(
        torch.zeros(1, 6, 4, dtype=torch.double), values, reference_grid(2, 3).double()
    )
    expected = (
        values * torch.tensor([0.25, 0.25, 0.8, 0.8], dtype=torch.double)[None, :, None, None]
    )
    expected[..., :-1] += (
        values[..., 1:]
        * torch.tensor([0.75, 0.75, 0.2, 0.2], dtype=torch.double)[None, :, None, None]
    )
    torch.testing.assert_close(output, expected.flatten(2).transpose(1, 2), atol=3e-6, rtol=1e-6)


def test_attention_gradcheck_values_queries_and_offsets():
    torch.manual_seed(2)
    attention = identity_attention(heads=2, points=2)
    # Avoid interpolation knots and make offsets depend on queries.
    with torch.no_grad():
        attention.sampling_offsets.weight.normal_(std=0.03)
        attention.sampling_offsets.bias.fill_(0.23)
        attention.attention_weights.weight.normal_(std=0.05)
    query = torch.randn(1, 6, 4, dtype=torch.double, requires_grad=True)
    values = torch.randn(1, 4, 2, 3, dtype=torch.double, requires_grad=True)
    grid = reference_grid(2, 3).double()
    assert torch.autograd.gradcheck(lambda q, v: attention(q, v, grid), (query, values))


def test_reference_coordinates_and_position_encoding():
    grid = reference_grid(2, 3)
    torch.testing.assert_close(grid[0], torch.tensor([1 / 6, 1 / 4]))
    torch.testing.assert_close(grid[-1], torch.tensor([5 / 6, 3 / 4]))
    encoding = sine_position_encoding(grid, 16)
    assert encoding.shape == (6, 16)
    assert torch.isfinite(encoding).all()
    assert not torch.equal(encoding[0], encoding[1])
    with pytest.raises(ValueError):
        sine_position_encoding(grid, 6)
