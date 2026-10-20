import torch
import pytest

from autoregressive_core.generation import generate
from autoregressive_core.model import TinyGPT
from autoregressive_core.tokenizer import CharTokenizer
from autoregressive_core.training import TrainConfig, get_batch, load_checkpoint, save_checkpoint


def test_tokenizer_round_trip():
    tokenizer = CharTokenizer.from_text("cabca")
    assert tokenizer.decode(tokenizer.encode("abc")) == "abc"


def test_training_batch_shapes():
    x, y = get_batch(torch.arange(100), batch_size=3, sequence_length=8)
    assert x.shape == y.shape == (3, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_generation_and_validation():
    tokenizer = CharTokenizer.from_text("abcd")
    model = TinyGPT(vocab_size=4, dim=16, num_layers=1, num_query_heads=2, num_kv_heads=1, hidden_dim=24)
    out = generate(model, tokenizer, "ab", top_k=2, max_new_tokens=3)
    assert out.startswith("ab") and len(out) == 5
    with pytest.raises(ValueError):
        generate(model, tokenizer, "ab", temperature=0)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "ab", top_k=5)


def test_checkpoint_round_trip(tmp_path):
    tokenizer = CharTokenizer.from_text("abcd")
    config = TrainConfig(dim=16, num_layers=1, num_query_heads=2, num_kv_heads=1, hidden_dim=24)
    model = TinyGPT(vocab_size=4, dim=16, num_layers=1, num_query_heads=2, num_kv_heads=1, hidden_dim=24)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tokenizer, config, [{"step": 1, "train_loss": 1.0, "val_loss": 1.1}])
    loaded, loaded_tokenizer, checkpoint = load_checkpoint(path)
    assert loaded_tokenizer.vocabulary == tokenizer.vocabulary
    assert checkpoint["loss_history"][0]["step"] == 1
    for left, right in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(left, right)
