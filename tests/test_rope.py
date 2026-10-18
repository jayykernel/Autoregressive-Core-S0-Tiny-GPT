import pytest
import torch

from autoregressive_core.rotary import apply_rope


def test_rope_preserves_shape():
    x = torch.randn(2, 4, 8, 16)

    y = apply_rope(x)

    assert y.shape == x.shape


def test_rope_position_zero_is_unchanged():
    x = torch.randn(2, 4, 8, 16)

    y = apply_rope(x)

    # Every rotation angle at position zero is zero.
    torch.testing.assert_close(y[:, :, 0, :], x[:, :, 0, :])


def test_rope_rejects_odd_head_dimension():
    x = torch.randn(2, 4, 8, 15)

    with pytest.raises(ValueError):
        apply_rope(x)


def test_rope_supports_gradients():
    x = torch.randn(2, 4, 8, 16, requires_grad=True)

    y = apply_rope(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()