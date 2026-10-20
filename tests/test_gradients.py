
import torch

from autoregressive_core.decoder import DecoderBlock
from autoregressive_core.model import TinyGPT
from autoregressive_core.normalization import RMSNorm


def test_rmsnorm_gradient_check():
    torch.manual_seed(0)

    model = RMSNorm(4, eps=1e-6).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        model,
        (x,),
        eps=1e-6,
        atol=1e-4,
    )


def test_decoder_input_and_parameter_gradients():
    torch.manual_seed(1)

    model = DecoderBlock()
    x = torch.randn(2, 8, 64, requires_grad=True)

    model(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient: {name}"
        )


def test_full_tinygpt_gradients():
    torch.manual_seed(2)

    model = TinyGPT(vocab_size=256)
    tokens = torch.randint(0, 256, (2, 9))

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 8, 256)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient: {name}"
        )
