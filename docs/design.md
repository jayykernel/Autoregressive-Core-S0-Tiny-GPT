
# Autoregressive Core — Design Report

## 1. Objective

Build a small decoder-only Transformer in PyTorch that exposes the
mathematics and tensor operations behind GPT-style language models.

The implementation prioritizes explicit tensor shapes, modular components,
reproducible tests, reference verification, and an explainable memory model.

## 2. Architecture

The model consists of:

1. Token embeddings
2. Two pre-norm decoder blocks
3. Final RMSNorm
4. A vocabulary projection head

Each decoder block consists of:

1. RMSNorm before attention
2. RoPE-enhanced grouped-query causal self-attention
3. First residual connection
4. RMSNorm before the feed-forward sublayer
5. SwiGLU feed-forward network
6. Second residual connection

## 3. Architectural decisions

### Rotary positional embeddings

RoPE rotates pairs of query and key features according to token position
and pair-specific frequencies. It preserves tensor shapes and does not
introduce learned position-embedding parameters.

The current implementation uses adjacent-pair rotation.

### RMSNorm and pre-norm

RMSNorm scales activations using the root mean square of their features.
Each major sublayer is preceded by RMSNorm, and its output is added to
the residual stream.

### SwiGLU

The feed-forward network uses a SiLU gate, an up-projection branch,
element-wise multiplication, and a down projection.

### Grouped-query attention

The model uses four query heads and two KV heads by default.
Each key/value head serves two query heads.

The reference implementation expands KV heads explicitly. This is
intended to make the computation understandable and testable, not to
represent an optimized inference kernel.

### Causal masking

A query at position t may attend to keys at positions <= t. Future
positions receive negative-infinity scores before softmax.

### Language-model head

The final projection maps the model dimension to the vocabulary size.
The input embedding and output projection are not tied in this version.

## 4. Validation strategy

The test suite verifies:

- Intermediate and end-to-end tensor shapes
- RMSNorm, RoPE and SwiGLU behavior
- Attention output agreement with PyTorch scaled dot-product attention
- Decoder output and gradient agreement with an independent reference
- Gradient existence and numerical finiteness
- Causal isolation from future tokens
- End-to-end forward pass, next-token loss and backward propagation

Passing tests demonstrate correctness for the covered cases; they do not
prove correctness for every possible configuration or hardware backend.

## 5. Current scope and limitations

The implementation is a small educational GPT core. It is not a
production language model.

The current baseline does not implement:

- A persistent autoregressive KV cache
- Optimized fused GQA kernels
- Distributed or multi-GPU training
- A production tokenizer
- A large-scale pretraining pipeline
- Guaranteed support for every dtype and device

The training-data and text-generation workflow should be treated as a
separate follow-on milestone.

## 6. Reproducibility

Run the complete test suite from the repository root:

    python -m pytest -q

Run the demonstration:

    python examples/forward_pass.py

Record the Python, PyTorch, device, dtype, and test outcomes when publishing
the results.
