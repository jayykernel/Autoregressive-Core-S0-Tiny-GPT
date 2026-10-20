
# TinyGPT Tensor Shape Contract

## 1. Configuration

| Symbol | Meaning | Value |
|---|---|---:|
| B | Batch size | 2 |
| T | Sequence length | 8 |
| V | Vocabulary size | 256 |
| D | Model dimension | 64 |
| L | Number of decoder blocks | 2 |
| Hq | Query heads | 4 |
| Hkv | Key/value heads | 2 |
| Dh | Head dimension = D / Hq | 16 |
| F | SwiGLU hidden dimension | 128 |

The reference dtype is float32.

## 2. Input and embeddings

Input token IDs have shape:

    [B, T] = [2, 8]

The token embedding matrix has shape:

    [V, D] = [256, 64]

After embedding:

    [B, T] -> [B, T, D]
    [2, 8] -> [2, 8, 64]

RoPE supplies positional information inside attention. This implementation
does not use a separate learned absolute-position embedding.

## 3. Attention projections

The input to attention is [B, T, D].

| Operation | Expected shape |
|---|---|
| Input | [2, 8, 64] |
| Q projection | [2, 8, 64] |
| K projection | [2, 8, 32] |
| V projection | [2, 8, 32] |
| Q after reshape and transpose | [2, 4, 8, 16] |
| K after reshape and transpose | [2, 2, 8, 16] |
| V after reshape and transpose | [2, 2, 8, 16] |

The query projection has Hq * Dh = 4 * 16 = 64 features.
The key and value projections each have Hkv * Dh = 2 * 16 = 32 features.

The projection operation is followed by a reshape from [B, T, H, Dh]
to [B, H, T, Dh].

## 4. RoPE and grouped-query attention

RoPE rotates adjacent pairs of features in Q and K.

It preserves the tensor shape:

    [2, 4, 8, 16] -> [2, 4, 8, 16]
    [2, 2, 8, 16] -> [2, 2, 8, 16]

V is not rotated.

For the reference GQA implementation, each KV head serves two query heads.
The KV tensors are expanded along the head dimension:

    K: [2, 2, 8, 16] -> [2, 4, 8, 16]
    V: [2, 2, 8, 16] -> [2, 4, 8, 16]

This explicit expansion is used for clarity. A native GQA kernel can avoid
physically replicating K and V.

## 5. Attention score calculation

Scaled dot-product attention is:

    scores = Q @ K.transpose(-2, -1) / sqrt(Dh)

Shape arithmetic:

    [2, 4, 8, 16] @ [2, 4, 16, 8]
    -> [2, 4, 8, 8]

The score tensor is [B, Hq, T, T].

The causal mask has shape [T, T] = [8, 8] and broadcasts across the
batch and query-head dimensions. Future positions are assigned negative
infinity before softmax.

Attention weights:

    [2, 4, 8, 8]

Weighted sum of values:

    [2, 4, 8, 8] @ [2, 4, 8, 16]
    -> [2, 4, 8, 16]

Merge the heads:

    [2, 4, 8, 16] -> [2, 8, 64]

The output projection returns [2, 8, 64].

## 6. SwiGLU feed-forward network

SwiGLU uses two expansion projections and one contraction projection.

| Operation | Shape |
|---|---|
| Input | [2, 8, 64] |
| Gate projection | [2, 8, 128] |
| SiLU gate | [2, 8, 128] |
| Up projection | [2, 8, 128] |
| Element-wise product | [2, 8, 128] |
| Down projection | [2, 8, 64] |

Both branches must have matching shapes for element-wise multiplication.

## 7. Decoder block

Pre-norm attention:

    h = x + Attention(RMSNorm1(x))

Pre-norm feed-forward network:

    y = h + SwiGLU(RMSNorm2(h))

The input and output shape of each decoder block is [B, T, D].

## 8. Complete TinyGPT

| Operation | Expected shape |
|---|---|
| Token IDs | [2, 8] |
| Embeddings | [2, 8, 64] |
| After decoder block 1 | [2, 8, 64] |
| After decoder block 2 | [2, 8, 64] |
| Final RMSNorm | [2, 8, 64] |
| Language-model head | [2, 8, 256] |

The output contains one logit for every vocabulary item at every input
position. Cross-entropy converts these logits into a training loss when
next-token targets are supplied.

## 9. Verification requirements

- Assert every documented shape using PyTorch tests.
- Verify that RoPE preserves shape and supports gradients.
- Verify that the causal mask blocks future-token leakage.
- Verify that the complete model produces finite loss and gradients.
- Keep the documented configuration synchronized with the implementation.
S