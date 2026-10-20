"""Small local FastAPI wrapper around a trained TinyGPT checkpoint."""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generation import generate
from .training import load_checkpoint

CHECKPOINT = Path(os.environ.get("TINYGPT_CHECKPOINT", "checkpoints/tinygpt.pt"))
app = FastAPI(title="TinyGPT Studio API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
model = tokenizer = checkpoint = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model, tokenizer, checkpoint
    if CHECKPOINT.exists():
        model, tokenizer, checkpoint = load_checkpoint(CHECKPOINT, device=device)


load_model()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    temperature: float = Field(default=0.8, gt=0, le=3)
    top_k: int = Field(default=20, ge=1)
    max_new_tokens: int = Field(default=160, ge=1, le=512)


@app.get("/api/health")
def health():
    return {"application": "ok", "model_ready": model is not None, "device": device}


@app.get("/api/model-info")
def model_info():
    if model is None:
        return {"ready": False, "checkpoint": str(CHECKPOINT)}
    cfg = checkpoint["config"]
    params = sum(p.numel() for p in model.parameters())
    return {"ready": True, "config": {**cfg, "vocab_size": len(tokenizer.vocabulary)},
            "parameter_count": params, "dtype": str(next(model.parameters()).dtype), "device": device,
            "vocab_size": len(tokenizer.vocabulary), "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "tensor_shapes": {"embedding": [len(tokenizer.vocabulary), cfg["dim"]],
                "residual": ["B", "T", cfg["dim"]], "attention_scores": ["B", cfg["num_query_heads"], "T", "T"],
                "logits": ["B", "T", len(tokenizer.vocabulary)]}}


@app.post("/api/generate")
def generate_text(body: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="No trained checkpoint is available. Run the training command first.")
    if body.top_k > len(tokenizer.vocabulary):
        raise HTTPException(status_code=422, detail="top_k exceeds the model vocabulary size")
    started = time.perf_counter()
    try:
        output = generate(model, tokenizer, body.prompt, temperature=body.temperature, top_k=body.top_k,
                          max_new_tokens=body.max_new_tokens, device=device)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"text": output, "latency_ms": elapsed_ms, "new_tokens": body.max_new_tokens}


@app.get("/api/loss-history")
def loss_history():
    return {"history": checkpoint.get("loss_history", []) if checkpoint else []}
