
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward network used by the decoder."""

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
