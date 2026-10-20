import torch
import torch.nn.functional as F

from autoregressive_core.decoder import DecoderBlock


def reference_rope(x, base=10_000.0):
    """Independent adjacent-pair RoPE reference."""
    _, _, seq_len, head_dim = x.shape

    positions = torch.arange(
        seq_len, device=x.device, dtype=torch.float32
    )
    pair_indices = torch.arange(
        0, head_dim, 2, device=x.device, dtype=torch.float32
    )

    inv_freq = base ** (-pair_indices / head_dim)
    angles = positions[:, None] * inv_freq[None, :]

    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]

    result = torch.empty_like(x)
    result[..., 0::2] = even * cos - odd * sin
    result[..., 1::2] = even * sin + odd * cos

    return result


def reference_decoder(block, x):
    """Compose a decoder reference without calling block.forward()."""

    # First pre-norm.
    norm1 = block.attention_norm
    a = F.rms_norm(
        x, (block.dim,), weight=norm1.weight, eps=norm1.eps
    )

    att = block.attention
    batch, seq_len, _ = a.shape
    hq = att.num_query_heads
    hkv = att.num_kv_heads
    dh = att.head_dim

    # Explicit linear projections.
    q = F.linear(a, att.q_proj.weight, att.q_proj.bias)
    k = F.linear(a, att.k_proj.weight, att.k_proj.bias)
    v = F.linear(a, att.v_proj.weight, att.v_proj.bias)

    q = q.reshape(batch, seq_len, hq, dh).transpose(1, 2)
    k = k.reshape(batch, seq_len, hkv, dh).transpose(1, 2)
    v = v.reshape(batch, seq_len, hkv, dh).transpose(1, 2)

    # Independently rotate Q and K.
    q = reference_rope(q)
    k = reference_rope(k)

    # Reference GQA: expand KV heads to match query heads.
    groups = hq // hkv
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)

    future_mask = torch.triu(
        torch.ones(
            seq_len, seq_len,
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )
    scores = scores.masked_fill(future_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    context = weights @ v
    context = (
        context.transpose(1, 2)
        .contiguous()
        .reshape(batch, seq_len, block.dim)
    )

    attention_output = F.linear(
        context,
        att.out_proj.weight,
        att.out_proj.bias,
    )

    # First residual.
    h = x + attention_output

    # Second pre-norm and explicit SwiGLU.
    norm2 = block.ffn_norm
    z = F.rms_norm(
        h, (block.dim,), weight=norm2.weight, eps=norm2.eps
    )

    ff = block.feedforward
    gate = F.silu(F.linear(z, ff.gate.weight, ff.gate.bias))
    up = F.linear(z, ff.up.weight, ff.up.bias)
    hidden = gate * up

    feedforward_output = F.linear(
        hidden, ff.down.weight, ff.down.bias
    )

    # Second residual.
    return h + feedforward_output


def test_decoder_matches_independent_reference():
    torch.manual_seed(123)

    block = DecoderBlock()
    x = torch.randn(2, 8, 64)

    actual = block(x)
    expected = reference_decoder(block, x)

    torch.testing.assert_close(
        actual, expected, rtol=1e-5, atol=1e-6
    )


def test_decoder_gradients_match_reference():
    torch.manual_seed(456)

    block = DecoderBlock()
    original_x = torch.randn(2, 8, 64)
    upstream = torch.randn(2, 8, 64)

    # Gradients from the actual implementation.
    block.zero_grad(set_to_none=True)
    x1 = original_x.clone().requires_grad_(True)

    actual = block(x1)
    (actual * upstream).sum().backward()

    input_grad_actual = x1.grad.detach().clone()
    parameter_grads_actual = {
        name: param.grad.detach().clone()
        for name, param in block.named_parameters()
    }

    # Gradients from the independently composed reference.
    block.zero_grad(set_to_none=True)
    x2 = original_x.clone().requires_grad_(True)

    expected = reference_decoder(block, x2)
    (expected * upstream).sum().backward()

    torch.testing.assert_close(
        actual.detach(), expected.detach(),
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        input_grad_actual, x2.grad,
        rtol=2e-5, atol=2e-6,
    )

    for name, param in block.named_parameters():
        torch.testing.assert_close(
            parameter_grads_actual[name],
            param.grad,
            rtol=2e-5,
            atol=2e-6,
            msg=lambda message: f"{name}: {message}",
        )