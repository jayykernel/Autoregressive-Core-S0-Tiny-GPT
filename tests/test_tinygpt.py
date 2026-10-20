
import pytest
import torch

from autoregressive_core.model import TinyGPT


def test_tinygpt_output_shape():
    model = TinyGPT(vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 8))

    logits, loss = model(input_ids)

    assert logits.shape == (2, 8, 256)
    assert loss is None


def test_tinygpt_computes_next_token_loss():
    torch.manual_seed(0)

    model = TinyGPT(vocab_size=256)

    # Nine tokens produce eight input/target pairs.
    tokens = torch.randint(0, 256, (2, 9))
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 8, 256)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_tinygpt_gradients_are_finite():
    torch.manual_seed(1)

    model = TinyGPT(vocab_size=256)

    tokens = torch.randint(0, 256, (2, 9))
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    _, loss = model(input_ids, targets)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"Non-finite gradient: {name}"
        )


def test_tinygpt_preserves_causality():
    torch.manual_seed(2)

    model = TinyGPT(vocab_size=256).eval()

    ids1 = torch.randint(0, 256, (2, 8))
    ids2 = ids1.clone()

    # Change tokens at positions 5, 6 and 7 only.
    ids2[:, 5:] = torch.randint(0, 256, (2, 3))

    with torch.no_grad():
        logits1, _ = model(ids1)
        logits2, _ = model(ids2)

    # Earlier token predictions must not change.
    torch.testing.assert_close(
        logits1[:, :5, :],
        logits2[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_tinygpt_rejects_mismatched_targets():
    model = TinyGPT(vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 8))
    targets = torch.randint(0, 256, (2, 7))

    with pytest.raises(ValueError):
        model(input_ids, targets)
