# RMSNorm

## Purpose
Normalize each token's feature vector using its root mean square.

## Mathematical operation
RMSNorm divides the input by the square root of the
mean squared features plus epsilon, then applies
a learnable feature-wise scale.

## Implementation
- PyTorch nn.Module
- Learnable scaling parameters
- Configurable epsilon
- Normalization along the final tensor dimension
- Float32 computation for float16 and bfloat16 inputs
- Float64 support for numerical gradient checking

## Validation
- Output shape test
- PyTorch reference comparison
- Zero-input test
- Gradient validity test
- Positive-rescaling test
- Numerical gradient check

## Limitations
The implementation has basic correctness tests.
Additional precision, performance, and extreme-value
testing may be added in future iterations.