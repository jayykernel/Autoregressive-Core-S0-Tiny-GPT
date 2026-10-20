import math
import torch
from torch import nn
from .rotary import apply_rope
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_query_heads: int = 4,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        if dim % num_query_heads != 0:
            raise ValueError("dim must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("Query heads must be divisible by KV heads.")
        head_dim = dim // num_query_heads
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        self.dim = dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Q has four heads; K and V have two.
        self.q_proj = nn.Linear(dim, num_query_heads * head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        hq = self.num_query_heads
        hkv = self.num_kv_heads
        dh = self.head_dim
        # 1. Project the input.
        q = self.q_proj(x)  # [B, T, 64]
        k = self.k_proj(x)  # [B, T, 32]
        v = self.v_proj(x)  # [B, T, 32]
        # 2. Split into heads and move the head axis.
        q = q.reshape(batch, seq_len, hq, dh).transpose(1, 2)
        k = k.reshape(batch, seq_len, hkv, dh).transpose(1, 2)
        v = v.reshape(batch, seq_len, hkv, dh).transpose(1, 2)
        # Q: [B, 4, T, 16]
        # K: [B, 2, T, 16]
        # V: [B, 2, T, 16]
        # 3. Apply RoPE to queries and keys, not values.
        q = apply_rope(q)
        k = apply_rope(k)
        # 4. Expand KV heads for a simple GQA reference path.
        # Each KV head serves two query heads.
        groups = hq // hkv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        # K, V: [B, 4, T, 16]
        # 5. Calculate scaled attention scores.
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(dh)
        # Scores: [B, 4, T, T]
        # 6. Block future positions.
        future_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))
        # 7. Turn scores into attention probabilities.
        weights = torch.softmax(scores, dim=-1)
        # 8. Combine the value vectors.
        context = weights @ v  # [B, 4, T, 16]
        # 9. Merge heads and restore the model dimension.
        context = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(batch, seq_len, self.dim)
        )
        # 10. Mix information across the merged heads.
        return self.out_proj(context)