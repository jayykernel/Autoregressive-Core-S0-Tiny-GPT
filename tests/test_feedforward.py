
import torch

from autoregressive_core.feedforward import SwiGLU


def test_swiglu_output_shape():
    model = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64)

    y = model(x)

    assert y.shape == (2, 8, 64)


def test_swiglu_gradients_are_finite():
    model = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64, requires_grad=True)

    y = model(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
