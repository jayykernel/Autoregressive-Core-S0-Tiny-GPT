
import torch
import torch.nn as nn

from autoregressive_core.normalization import RMSNorm


def test_output_shape():
    x = torch.randn(2, 5, 64)
    norm = RMSNorm(dim=64)

    output = norm(x)

    assert output.shape == x.shape


def test_matches_pytorch_reference():
    torch.manual_seed(42)

    dim = 64
    eps = 1e-6

    x = torch.randn(2, 5, dim)

    # Our implementation
    custom = RMSNorm(dim=dim, eps=eps)

    # PyTorch reference implementation
    reference = nn.RMSNorm(
        normalized_shape=dim,
        eps=eps,
    )

    # Both implementations must use identical weights.
    with torch.no_grad():
        reference.weight.copy_(custom.weight)

    actual = custom(x)
    expected = reference(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_zero_input():
    x = torch.zeros(2, 5, 64)
    norm = RMSNorm(dim=64)

    output = norm(x)

    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_gradients():
    x = torch.randn(
        2, 5, 64,
        requires_grad=True,
    )

    norm = RMSNorm(dim=64)

    output = norm(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_positive_rescaling():
    torch.manual_seed(42)

    x = torch.randn(2, 5, 64)
    norm = RMSNorm(dim=64)

    original = norm(x)
    scaled = norm(3.0 * x)

    # Epsilon makes this approximate, not exact,
    # scale invariance.
    torch.testing.assert_close(
        original,
        scaled,
        rtol=1e-5,
        atol=1e-6,
    )