# Autoregressive Core — S0 TinyGPT

A small, educational decoder-only Transformer built with Python and PyTorch. The project implements a complete TinyGPT forward pass and makes the core mechanics explicit: rotary positional embeddings (RoPE), RMSNorm with pre-norm residual connections, SwiGLU, causal self-attention, and configurable multi-head attention (MHA) or grouped-query attention (GQA).

The focus is not model scale. It is learning how a GPT-style decoder works, verifying tensor shapes and gradients, comparing implementations against independent references, and documenting memory requirements.

## Features

- **RoPE** applied to query and key tensors, without changing their shapes.
- **RMSNorm and pre-norm residual connections** in each decoder block.
- **SwiGLU** feed-forward network.
- **Causal self-attention** that blocks attention to future positions.
- **MHA and GQA configurations** through configurable query-head and KV-head counts.
- **TinyGPT forward pass** from token IDs to vocabulary logits.
- **Next-token cross-entropy loss** when target token IDs are supplied.
- **Verification suite** covering shapes, gradients, causality, and decoder output/gradient agreement with an independently composed PyTorch reference.
- **Memory report script** measuring parameter, gradient, optimizer-state, input, and logits storage.

## Architecture

```text
Token IDs [B, T]
      │
      ▼
Token Embedding [B, T, D]
      │
      ▼
┌────────────────────────────────────────────┐
│ DecoderBlock × N                           │
│                                            │
│  RMSNorm → Q/K/V projections               │
│            → RoPE on Q and K               │
│            → causal GQA/MHA attention      │
│            → output projection              │
│            → residual addition              │
│                                            │
│  RMSNorm → SwiGLU                           │
│          → residual addition                │
└────────────────────────────────────────────┘
      │
      ▼
Final RMSNorm [B, T, D]
      │
      ▼
Language-model head [B, T, vocab_size]
      │
      ▼
Logits (and optional next-token loss)
```

RoPE supplies positional information inside attention, so this implementation does not use separate learned absolute-position embeddings. The token embedding and output language-model head are **not tied** in this version.

## Default configuration

| Parameter | Default |
|---|---:|
| Vocabulary size | 256 |
| Model dimension (`D`) | 64 |
| Decoder blocks | 2 |
| Query heads (`Hq`) | 4 |
| Key/value heads (`Hkv`) | 2 |
| Head dimension (`D / Hq`) | 16 |
| SwiGLU hidden dimension | 128 |
| Reference dtype | `torch.float32` |

With this configuration, the default attention mode is GQA: four query heads share two key/value heads. Set `num_kv_heads` equal to `num_query_heads` to configure MHA, provided the head-count constraints are met.

## Repository layout

```text
autoregressive-core/
├── src/
│   └── autoregressive_core/
│       ├── __init__.py
│       ├── normalization.py
│       ├── rotary.py
│       ├── feedforward.py
│       ├── attention.py
│       ├── decoder.py
│       └── model.py
├── tests/
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_decoder.py
│   ├── test_decoder_reference.py
│   ├── test_tinygpt.py
│   ├── test_shapes.py
│   ├── test_gradients.py
│   └── test_causality.py
├── docs/
│   ├── tensor_shapes.md
│   ├── design.md
│   └── tensor_memory.md
├── examples/
│   ├── forward_pass.py
│   └── memory_report.py
├── pyproject.toml
└── README.md
```

The test directory may also contain additional component tests, such as RoPE or RMSNorm tests, depending on the current working tree.

## Requirements

- Python 3.11 or newer
- PyTorch
- pytest for running the test suite

The reported development run used Python 3.14 on CPU. The PyTorch version was not recorded in the measured output.

## Setup

Run these commands from the repository root.

### 1. Create and activate a virtual environment (recommended)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the project and test dependency

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

The editable install makes the `autoregressive_core` package importable from `src/` while you work on it.

## Run the tests

Run the complete test suite:

```powershell
python -m pytest -q
```

Or run the main suites individually:

```powershell
python -m pytest -q tests/test_attention.py
python -m pytest -q tests/test_feedforward.py
python -m pytest -q tests/test_decoder.py
python -m pytest -q tests/test_decoder_reference.py
python -m pytest -q tests/test_tinygpt.py
python -m pytest -q tests/test_shapes.py
python -m pytest -q tests/test_gradients.py
python -m pytest -q tests/test_causality.py
```

The author reports that the complete test suite and the decoder reference tests passed in the development environment. Tests cover the implemented cases; passing them does not prove correctness for every possible configuration, dtype, or hardware backend.

## Run the forward-pass example

```powershell
python examples/forward_pass.py
```

The example creates simulated token IDs, shifts them into input/target sequences, runs the model, calculates cross-entropy loss, calls `backward()`, and performs one AdamW optimizer step.

This is a smoke test of model mechanics. It is **not** a full training pipeline and does not demonstrate convergence or language-generation quality.

## Run the memory report

```powershell
python examples/memory_report.py
```

The script reports parameter count and storage, input/logits storage, loss, gradient storage, AdamW state storage, and CUDA peak allocation when run on CUDA. On CPU, it does not measure total process memory or runtime peak memory.

### Reported CPU results

The following values were recorded from a development run using the default configuration and `torch.float32`:

| Measurement | Recorded result |
|---|---:|
| Device | CPU |
| Model dtype | `torch.float32` |
| Parameter count | 107,840 |
| Parameter storage | 431,360 bytes (421.25 KiB) |
| Input IDs shape | `(2, 8)` |
| Input IDs storage | 128 bytes |
| Logits shape | `(2, 8, 256)` |
| Logits storage | 16,384 bytes (16 KiB) |
| Cross-entropy loss | 5.50712251663208 |
| Gradient storage | 431,360 bytes (421.25 KiB) |
| AdamW state storage | 862,860 bytes (842.64 KiB) |
| CUDA peak memory | Not available; run executed on CPU |

The recorded loss was produced using randomly initialized parameters and random token IDs. It demonstrates a finite loss computation only; it is not a benchmark of model quality.

The sum of parameter, gradient, and optimizer-state storage in this run is 1,725,580 bytes, approximately 1.65 MiB. This excludes activations, temporary tensors, Python objects, allocator overhead, and other runtime allocations, so it is **not total peak memory**.

## Important tensor shapes

For the example configuration `B=2`, `T=8`, `D=64`, `Hq=4`, `Hkv=2`, `Dh=16`, and `F=128`:

| Operation | Shape |
|---|---|
| Token IDs | `[2, 8]` |
| Embeddings / residual stream | `[2, 8, 64]` |
| Query projection | `[2, 8, 64]` |
| Key projection | `[2, 8, 32]` |
| Value projection | `[2, 8, 32]` |
| Q after head reshape | `[2, 4, 8, 16]` |
| K/V after head reshape | `[2, 2, 8, 16]` |
| Attention scores | `[2, 4, 8, 8]` |
| Merged attention context | `[2, 8, 64]` |
| SwiGLU hidden branches | `[2, 8, 128]` |
| Vocabulary logits | `[2, 8, 256]` |

See [`docs/tensor_shapes.md`](docs/tensor_shapes.md) for the full tensor-shape contract and [`docs/tensor_memory.md`](docs/tensor_memory.md) for storage calculations and measurement limitations.

## Current limitations

This is an educational reference implementation, not a production inference engine. In particular:

- GQA currently uses explicit KV-head expansion in the reference attention path; optimized native GQA execution is not implemented here.
- There is no persistent autoregressive KV cache.
- There is no complete dataset/tokenizer pipeline or multi-epoch training loop yet.
- There is no quality evaluation on a held-out language dataset.
- CPU runtime peak memory is not measured by the current memory script.
- Tests cover the documented reference configuration and selected variations, not every dtype/device/backend combination.

## Learning and project deliverables

This repository is designed to make the following concepts and engineering checks explicit:

- RoPE, RMSNorm/pre-norm, SwiGLU, causal masking, MHA vs. GQA
- Exact tensor-shape derivations
- Complete TinyGPT forward pass and next-token loss
- Independent output and gradient reference checks
- Shape, gradient and causality verification
- Tensor memory calculations and measured storage

## License

No license has been specified yet. Add a `LICENSE` file before distributing the repository as an open-source project.
