
import pytest
import torch
import torch.nn as nn

from autoregressive_core.decoder import PreNormResidual


def test_prenorm_residual_equation():
    eps = 1e-6

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    block = PreNormResidual(
        dim=4,
        sublayer=nn.Identity(),
        eps=eps,
    )

    actual = block(x)

    # Independently calculate the expected normalized input.
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x / torch.sqrt(mean_square + eps)

    expected = x + normalized

    torch.testing.assert_close(actual, expected)
def test_residual_preserves_shape():
    x = torch.randn(2, 5, 64)

    block = PreNormResidual(
        dim=64,
        sublayer=nn.Identity(),
    )

    output = block(x)

    assert output.shape == x.shape
def test_rejects_mismatched_shape():
    x = torch.randn(2, 5, 4)

    block = PreNormResidual(
        dim=4,
        sublayer=nn.Linear(4, 8),
    )

    with pytest.raises(ValueError, match="shape must match"):
        block(x)
class ZeroSublayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_identity_gradient_path():
    x = torch.randn(2, 5, 4, requires_grad=True)

    block = PreNormResidual(
        dim=4,
        sublayer=ZeroSublayer(),
    )

    output = block(x)

    torch.testing.assert_close(output, x)

    output.sum().backward()

    # The zero-valued branch contributes no gradient.
    # The direct residual path contributes gradient 1.
    torch.testing.assert_close(
        x.grad,
        torch.ones_like(x),
    )