import torch
import torch.nn.functional as F

from autoregressive_core.attention import CausalSelfAttention
from autoregressive_core.rotary import apply_rope


def test_attention_output_shape():
    model = CausalSelfAttention(
        dim=64,
        num_query_heads=4,
        num_kv_heads=2,
    )
    x = torch.randn(2, 8, 64)

    output = model(x)

    assert output.shape == (2, 8, 64)


def test_attention_matches_pytorch_reference():
    torch.manual_seed(0)

    model = CausalSelfAttention(64, 4, 2)
    x = torch.randn(2, 8, 64)

    # Independently compose the same projections and RoPE,
    # then use PyTorch SDPA for attention.
    B, T, _ = x.shape
    Hq, Hkv = model.num_query_heads, model.num_kv_heads
    dh = model.head_dim

    q = model.q_proj(x).reshape(B, T, Hq, dh).transpose(1, 2)
    k = model.k_proj(x).reshape(B, T, Hkv, dh).transpose(1, 2)
    v = model.v_proj(x).reshape(B, T, Hkv, dh).transpose(1, 2)

    q, k = apply_rope(q), apply_rope(k)

    groups = Hq // Hkv
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    reference = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        dropout_p=0.0,
    )

    reference = (
        reference.transpose(1, 2)
        .contiguous()
        .reshape(B, T, model.dim)
    )
    reference = model.out_proj(reference)

    torch.testing.assert_close(
        model(x),
        reference,
        rtol=1e-5,
        atol=1e-6,
    )


def test_future_tokens_do_not_change_past_outputs():
    torch.manual_seed(1)

    model = CausalSelfAttention(64, 4, 2).eval()
    x1 = torch.randn(2, 8, 64)
    x2 = x1.clone()

    # Change only tokens at positions 5, 6 and 7.
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    # Outputs at positions 0 through 4 must remain unchanged.
    torch.testing.assert_close(
        y1[:, :5, :],
        y2[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_gradients_are_finite():
    model = CausalSelfAttention(64, 4, 2)
    x = torch.randn(2, 8, 64, requires_grad=True)

    output = model(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()