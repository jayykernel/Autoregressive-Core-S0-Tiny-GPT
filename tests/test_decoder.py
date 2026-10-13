
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