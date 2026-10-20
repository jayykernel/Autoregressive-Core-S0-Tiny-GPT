from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import TinyGPT
from .tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 8
    sequence_length: int = 64
    learning_rate: float = 3e-4
    seed: int = 42
    eval_interval: int = 50
    dim: int = 64
    num_layers: int = 2
    num_query_heads: int = 4
    num_kv_heads: int = 2
    hidden_dim: int = 128


def load_text(data_path: str | Path = "data/input.txt", download: bool = True) -> str:
    path = Path(data_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if not download:
        raise FileNotFoundError(f"Dataset not found: {path}. Provide a local UTF-8 text file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    except Exception as exc:
        raise RuntimeError(f"Could not download Tiny Shakespeare; place UTF-8 text at {path} instead.") from exc
    return path.read_text(encoding="utf-8")


def get_batch(data: torch.Tensor, batch_size: int, sequence_length: int, device: str = "cpu"):
    if data.numel() <= sequence_length:
        raise ValueError("Training split must be longer than sequence_length")
    starts = torch.randint(data.numel() - sequence_length, (batch_size,))
    x = torch.stack([data[i:i + sequence_length] for i in starts])
    y = torch.stack([data[i + 1:i + sequence_length + 1] for i in starts])
    return x.to(device), y.to(device)


def save_checkpoint(path: str | Path, model: TinyGPT, tokenizer: CharTokenizer, config: TrainConfig, history: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(config), "vocabulary": tokenizer.vocabulary,
                "loss_history": history}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    model = TinyGPT(vocab_size=len(tokenizer.vocabulary), **{k: config[k] for k in
        ("dim", "num_layers", "num_query_heads", "num_kv_heads", "hidden_dim")})
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    model.max_seq_len = config["sequence_length"]
    return model, tokenizer, checkpoint


def train(config: TrainConfig = TrainConfig(), data_path: str | Path = "data/input.txt",
          checkpoint_path: str | Path = "checkpoints/tinygpt.pt", download: bool = True):
    if (config.steps < 1 or config.batch_size < 1 or config.sequence_length < 2
            or config.learning_rate <= 0 or config.eval_interval < 1):
        raise ValueError("steps, batch_size, sequence_length, learning_rate and eval_interval must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    text = load_text(data_path, download=download)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = max(config.sequence_length + 1, int(len(ids) * 0.9))
    train_data, val_data = ids[:split], ids[split:]
    if val_data.numel() <= config.sequence_length:
        val_data = train_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(vocab_size=len(tokenizer.vocabulary), **{k: getattr(config, k) for k in
        ("dim", "num_layers", "num_query_heads", "num_kv_heads", "hidden_dim")}).to(device)
    model.max_seq_len = config.sequence_length
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    for step in range(1, config.steps + 1):
        model.train()
        x, y = get_batch(train_data, config.batch_size, config.sequence_length, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, config.batch_size, config.sequence_length, device)
                _, val_loss = model(vx, vy)
            row = {"step": step, "train_loss": float(loss.item()), "val_loss": float(val_loss.item())}
            history.append(row)
            print(json.dumps(row))
    save_checkpoint(checkpoint_path, model, tokenizer, config, history)
    return model, tokenizer, history
