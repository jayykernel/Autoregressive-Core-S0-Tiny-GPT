import torch


def apply_rope(
    x: torch.Tensor,
    base: float = 10_000.0,
) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings.

    Args:
        x: Tensor shaped [batch, heads, sequence_length, head_dim].
        base: Frequency base used by the rotation formula.

    Returns:
        Tensor with the same shape as x.
    """
    if x.ndim != 4:
        raise ValueError("Expected x with shape [B, H, T, D].")

    batch, heads, seq_len, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    # Positions: [T]
    positions = torch.arange(
        seq_len, device=x.device, dtype=torch.float32
    )

    # One frequency for each pair of features: [D // 2]
    pair_indices = torch.arange(
        0, head_dim, 2, device=x.device, dtype=torch.float32
    )
    inv_freq = base ** (-pair_indices / head_dim)

    # Each angle depends on a position and a feature-pair frequency.
    # Shape: [T, D // 2]
    angles = positions[:, None] * inv_freq[None, :]

    # Add broadcast dimensions for batch and heads.
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

    # Group neighboring features into pairs.
    # [B, H, T, D] -> [B, H, T, D // 2, 2]
    pairs = x.reshape(batch, heads, seq_len, head_dim // 2, 2)
    x1 = pairs[..., 0]
    x2 = pairs[..., 1]

    rotated_pairs = torch.stack(
        (
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ),
        dim=-1,
    )

    # Restore the original shape.
    return rotated_pairs.flatten(-2)