
import torch
from torch import nn
from torch.nn import functional as F

from .decoder import DecoderBlock
from .normalization import RMSNorm


class TinyGPT(nn.Module):
    """A small decoder-only GPT using RoPE, RMSNorm, SwiGLU and GQA."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 64,
        num_layers: int = 2,
        num_query_heads: int = 4,
        num_kv_heads: int = 2,
        hidden_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()

        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive.")
        if num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("Head counts must be positive.")
        if dim % num_query_heads != 0:
            raise ValueError("dim must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads."
            )
        if (dim // num_query_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.vocab_size = vocab_size
        self.dim = dim

        # Convert integer token IDs into vectors.
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Stack independent decoder blocks.
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(dim, eps=eps)

        # Convert each token representation into vocabulary logits.
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: [B, T] integer token IDs
        targets:   optional [B, T] next-token target IDs

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar tensor if targets are supplied, else None
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T].")

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long.")

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids."
                )
            if targets.dtype != torch.long:
                raise TypeError("targets must have dtype torch.long.")

        # [B, T] -> [B, T, D]
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        # [B, T, D]
        x = self.final_norm(x)

        # [B, T, D] -> [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss
