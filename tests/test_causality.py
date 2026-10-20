
import torch

from autoregressive_core.attention import CausalSelfAttention
from autoregressive_core.model import TinyGPT


def assert_past_outputs_unchanged(model, x, changed_inputs, split):
    model.eval()

    with torch.no_grad():
        y1 = model(x)
        y2 = model(changed_inputs)

    # Attention module returns a tensor; TinyGPT returns (logits, loss).
    if isinstance(y1, tuple):
        y1 = y1[0]
        y2 = y2[0]

    torch.testing.assert_close(
        y1[:, :split, :],
        y2[:, :split, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_prevents_future_information_leakage():
    torch.manual_seed(0)

    model = CausalSelfAttention(64, 4, 2)
    x1 = torch.randn(2, 8, 64)
    x2 = x1.clone()

    # Alter only positions 5, 6 and 7.
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])

    assert_past_outputs_unchanged(model, x1, x2, split=5)


def test_tinygpt_prevents_future_token_leakage():
    torch.manual_seed(1)

    model = TinyGPT(vocab_size=256)
    ids1 = torch.randint(0, 256, (2, 8))
    ids2 = ids1.clone()

    ids2[:, 5:] = torch.randint(0, 256, (2, 3))

    assert_past_outputs_unchanged(model, ids1, ids2, split=5)
