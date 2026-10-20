from autoregressive_core import api


def test_api_reports_model_readiness(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    assert api.health()["model_ready"] is False
    assert api.model_info()["ready"] is False


def test_api_reports_loaded_model(monkeypatch):
    from autoregressive_core.model import TinyGPT
    from autoregressive_core.tokenizer import CharTokenizer

    tokenizer = CharTokenizer.from_text("abcd")
    model = TinyGPT(vocab_size=4, dim=16, num_layers=1, num_query_heads=2,
                    num_kv_heads=1, hidden_dim=24)
    monkeypatch.setattr(api, "model", model)
    monkeypatch.setattr(api, "tokenizer", tokenizer)
    monkeypatch.setattr(api, "checkpoint", {"config": {"dim": 16, "num_layers": 1,
        "num_query_heads": 2, "num_kv_heads": 1, "hidden_dim": 24}, "loss_history": []})
    assert api.health()["model_ready"] is True
    info = api.model_info()
    assert info["ready"] is True
    assert info["parameter_count"] == sum(p.numel() for p in model.parameters())
