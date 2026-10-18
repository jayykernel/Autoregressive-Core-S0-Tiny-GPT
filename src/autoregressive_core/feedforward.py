import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)

        hidden = gate * up
        return self.down(hidden)
    def test_swiglu_shapes_and_gradients():
    torch.manual_seed(0)

    model = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64, requires_grad=True)

    y = model(x)

    assert y.shape == (2, 8, 64)

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()