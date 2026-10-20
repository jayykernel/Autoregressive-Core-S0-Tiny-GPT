
import pytest
import torch

from autoregressive_core.decoder import DecoderBlock


def test_decoder_output_shape():
    block = DecoderBlock(
        dim=64,
        num_query_heads=4,
        num_kv_heads=2,
        hidden_dim=128,
    )

    x = torch.randn(2, 8, 64)
    y = block(x)

    assert y.shape == (2, 8, 64)


def test_decoder_gradients_are_finite():
    torch.manual_seed(0)

    block = DecoderBlock()
    x = torch.randn(2, 8, 64, requires_grad=True)

    y = block(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient: {name}"
        )


def test_decoder_preserves_causality():
    torch.manual_seed(1)

    block = DecoderBlock().eval()
    x1 = torch.randn(2, 8, 64)
    x2 = x1.clone()

    # Change only future tokens.
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])

    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)

    # Earlier outputs must remain unchanged.
    torch.testing.assert_close(
        y1[:, :5, :],
        y2[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_decoder_rejects_wrong_feature_dimension():
    block = DecoderBlock(dim=64)
    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        block(x)
