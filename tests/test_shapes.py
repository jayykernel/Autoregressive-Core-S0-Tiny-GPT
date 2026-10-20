
import torch

from autoregressive_core.attention import CausalSelfAttention
from autoregressive_core.decoder import DecoderBlock
from autoregressive_core.feedforward import SwiGLU
from autoregressive_core.model import TinyGPT
from autoregressive_core.normalization import RMSNorm
from autoregressive_core.rotary import apply_rope


def test_rmsnorm_shape():
    x = torch.randn(2, 8, 64)
    assert RMSNorm(64)(x).shape == (2, 8, 64)


def test_rope_shape():
    x = torch.randn(2, 4, 8, 16)
    assert apply_rope(x).shape == (2, 4, 8, 16)


def test_gqa_projection_shapes():
    attention = CausalSelfAttention(dim=64, num_query_heads=4, num_kv_heads=2)
    x = torch.randn(2, 8, 64)

    q = attention.q_proj(x)
    k = attention.k_proj(x)
    v = attention.v_proj(x)

    assert q.shape == (2, 8, 64)
    assert k.shape == (2, 8, 32)
    assert v.shape == (2, 8, 32)

    q = q.reshape(2, 8, 4, 16).transpose(1, 2)
    k = k.reshape(2, 8, 2, 16).transpose(1, 2)
    v = v.reshape(2, 8, 2, 16).transpose(1, 2)

    assert q.shape == (2, 4, 8, 16)
    assert k.shape == (2, 2, 8, 16)
    assert v.shape == (2, 2, 8, 16)

    # Expand each KV head to serve two query heads.
    k = k.repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=1)

    scores = q @ k.transpose(-2, -1)
    context = torch.softmax(scores / (16 ** 0.5), dim=-1) @ v

    assert scores.shape == (2, 4, 8, 8)
    assert context.shape == (2, 4, 8, 16)


def test_swiglu_intermediate_shapes():
    model = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64)

    gate = model.gate(x)
    up = model.up(x)

    assert gate.shape == (2, 8, 128)
    assert up.shape == (2, 8, 128)
    assert (gate * up).shape == (2, 8, 128)
    assert model(x).shape == (2, 8, 64)


def test_decoder_and_model_shapes():
    x = torch.randn(2, 8, 64)

    assert DecoderBlock()(x).shape == (2, 8, 64)

    model = TinyGPT(vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 8))
    logits, loss = model(input_ids)

    assert logits.shape == (2, 8, 256)
    assert loss is None
