from __future__ import annotations

import torch
from .model import TinyGPT
from .tokenizer import CharTokenizer


@torch.inference_mode()
def generate(model: TinyGPT, tokenizer: CharTokenizer, prompt: str, *, temperature: float = 0.8,
             top_k: int = 20, max_new_tokens: int = 160, device: str | torch.device | None = None) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k < 1 or top_k > len(tokenizer.vocabulary):
        raise ValueError("top_k must be between 1 and vocabulary size")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if not prompt:
        prompt = tokenizer.vocabulary[0]
    ids = tokenizer.encode(prompt)
    target_device = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    tokens = torch.tensor([ids], dtype=torch.long, device=target_device)
    for _ in range(max_new_tokens):
        context = tokens[:, -model.max_seq_len:] if hasattr(model, "max_seq_len") else tokens
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -torch.inf
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        tokens = torch.cat((tokens, next_id), dim=1)
    return tokenizer.decode(tokens[0].tolist())
