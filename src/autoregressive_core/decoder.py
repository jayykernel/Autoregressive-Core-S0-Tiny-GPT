
import torch
from torch import nn

from .attention import CausalSelfAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class DecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block with RoPE and GQA."""

    def __init__(
        self,
        dim: int = 64,
        num_query_heads: int = 4,
        num_kv_heads: int = 2,
        hidden_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim

        self.attention_norm = RMSNorm(dim, eps=eps)
        self.attention = CausalSelfAttention(
            dim=dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        )

        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feedforward = SwiGLU(
            dim=dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape [B, T, D].")

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Expected final dimension {self.dim}, "
                f"got {x.size(-1)}."
            )

        # Sublayer 1: pre-norm attention + residual connection
        x = x + self.attention(self.attention_norm(x))

        # Sublayer 2: pre-norm SwiGLU + residual connection
        x = x + self.feedforward(self.ffn_norm(x))

        return x
