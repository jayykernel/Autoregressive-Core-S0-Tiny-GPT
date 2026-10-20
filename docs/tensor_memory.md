
# TinyGPT Tensor Memory Report

## 1. Configuration and units

Reference configuration:

- Batch size B = 2
- Sequence length T = 8
- Vocabulary V = 256
- Model dimension D = 64
- Query heads Hq = 4
- KV heads Hkv = 2
- Head dimension Dh = 16
- Feed-forward hidden dimension F = 128
- Decoder blocks L = 2
- Reference dtype = float32

For binary storage units:

    1 KiB = 1024 bytes
    float32 = 4 bytes per element
    int64 = 8 bytes per element

Tensor storage is calculated as:

    tensor_bytes = tensor.numel() * tensor.element_size()

## 2. Important activation tensors

The following figures are logical storage requirements for individual
tensors. They are not the total runtime peak memory of the model.

| Tensor | Elements | Float32 storage |
|---|---:|---:|
| Token embeddings [2, 8, 64] | 1,024 | 4 KiB |
| Q [2, 4, 8, 16] | 1,024 | 4 KiB |
| K before expansion [2, 2, 8, 16] | 512 | 2 KiB |
| V before expansion [2, 2, 8, 16] | 512 | 2 KiB |
| Attention scores [2, 4, 8, 8] | 512 | 2 KiB |
| Attention weights [2, 4, 8, 8] | 512 | 2 KiB |
| Context [2, 4, 8, 16] | 1,024 | 4 KiB |
| SwiGLU gate [2, 8, 128] | 2,048 | 8 KiB |
| SwiGLU up branch [2, 8, 128] | 2,048 | 8 KiB |
| Logits [2, 8, 256] | 4,096 | 16 KiB |

The input token IDs [2, 8] contain 16 int64 elements and occupy 128 bytes.

During execution, multiple tensors and intermediate results can coexist.
Autograd retains additional data for backward propagation. Temporary
allocations and allocator overhead also affect runtime memory.

Consequently, summing the table does not give the model's peak memory.

## 3. Parameter count

For the specified default configuration:

| Component | Parameters |
|---|---:|
| Token embedding | 16,384 |
| Decoder block 1 | 37,504 |
| Decoder block 2 | 37,504 |
| Final RMSNorm | 64 |
| Output language-model head | 16,384 |
| Total | 107,840 |

These counts assume the current implementation's linear-layer biases,
scale-only RMSNorm modules, and untied embedding/output weights.

With float32 parameters:

    107,840 * 4 = 431,360 bytes
                  = 421.25 KiB

This is parameter storage only. It excludes gradients, optimizer state,
activations, and temporary memory.

## 4. KV-cache comparison

The following calculation describes the theoretical storage required
for a KV cache if one is added to the model.

For a single layer:

    KV bytes = 2 * B * T * Hkv * Dh * bytes_per_element

The factor of two accounts for keys and values.

For GQA with two KV heads:

    2 * 2 * 8 * 2 * 16 * 4 = 4,096 bytes = 4 KiB per layer

For MHA with four KV heads:

    2 * 2 * 8 * 4 * 16 * 4 = 8,192 bytes = 8 KiB per layer

Across two layers:

| Configuration | KV-cache storage |
|---|---:|
| GQA, 2 KV heads | 8 KiB |
| MHA, 4 KV heads | 16 KiB |

Under these assumptions, GQA halves the KV-cache storage.

These are theoretical cache sizes, not measurements of an implemented
cache. The current forward pass does not maintain a persistent KV cache.

GQA does not reduce the number of query heads or the logical attention
score shape [B, Hq, T, T].

## 5. Attention complexity

The score matrix has Hq * T * T entries per batch item.

Logical score storage therefore scales as:

    O(B * Hq * T^2)

The main attention matrix multiplications scale approximately as:

    O(B * Hq * T^2 * Dh)

At a fixed batch size and head count, doubling T makes the score matrix
four times larger.

Optimized scaled dot-product attention kernels can avoid materializing
the entire score matrix in memory, depending on the selected backend.

## 6. Tensor layout

- reshape reorganizes dimensions when the underlying layout permits it;
  it may return a view or require a copy.
- transpose swaps dimensions and usually changes strides.
- contiguous creates a contiguous layout when needed, potentially copying
  the tensor.
- view requires a compatible underlying memory layout.
- element_size reports bytes per element for the actual tensor dtype.

Logical shapes alone do not fully describe memory layout or runtime memory.

## 7. Measurement policy

Use examples/memory_report.py to record the actual parameter count,
parameter bytes, gradient storage and available CUDA peak-memory data.

Record the execution device and dtype alongside measurements.

Do not report theoretical activation or cache calculations as measured
runtime memory. CPU process memory and CUDA allocator memory are different
metrics and should be identified separately.

## 8. Limitations

The figures in this report describe a small reference configuration.
Different batch sizes, sequence lengths, dtypes, optimizer states,
autograd behavior, and attention backends will produce different memory
requirements.


## 9. Measured Results

### Environment

- Execution device: CPU
- Model dtype: float32
- Input batch shape: [2, 8]
- Logits shape: [2, 8, 256]
- Model configuration: TinyGPT with 2 decoder blocks
- Measurement script: examples/memory_report.py

### Actual measurements

| Measurement | Measured result |
|---|---:|
| Parameter count | 107,840 |
| Parameter storage | 431,360 bytes (421.25 KiB) |
| Input token IDs storage | 128 bytes |
| Logits storage | 16,384 bytes (16 KiB) |
| Cross-entropy loss | 5.50712251663208 |
| Gradient storage | 431,360 bytes (421.25 KiB) |
| AdamW optimizer state storage | 862,860 bytes (842.64 KiB) |
| CUDA peak allocation | Not available; executed on CPU |

### Interpretation

The parameter-storage result matches the theoretical calculation:

    107,840 parameters * 4 bytes = 431,360 bytes

The gradient-storage result is also 431,360 bytes because the model's
parameters and gradients are stored in float32 in this run.

AdamW maintains optimizer state, including first and second moment
estimates. The reported optimizer-state total also includes the small
step-state tensors counted by the measurement script.

The sum of parameter storage, gradient storage, and measured AdamW state
storage is:

    431,360 + 431,360 + 862,860 = 1,725,580 bytes
                                  = approximately 1.65 MiB

This total excludes activations, temporary tensors, Python objects,
allocator overhead, and other runtime allocations.

The logits occupy 16 KiB because the output tensor contains:

    2 * 8 * 256 = 4,096 float32 elements

The recorded loss is a smoke-test result obtained with randomly initialized
model parameters and randomly generated token IDs. It verifies that the
loss calculation returns a finite scalar; it does not demonstrate language
model quality or successful training convergence.

Because this run used the CPU, CUDA peak allocation was not measured.
CPU process memory was not measured by the current script either.

### Measurement limitations

These results describe the specified model and test configuration. Memory
usage changes with batch size, sequence length, dtype, architecture,
optimizer, and execution environment.

The current attention implementation explicitly expands KV heads for its
GQA reference calculation. The theoretical KV-cache comparison in this
report describes cache storage requirements, not a measured persistent
cache implemented by this model.
