
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")

        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps

        # One learnable weight for each feature.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            raise ValueError("Input must have at least one dimension")

        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                f"Expected last dimension {self.weight.numel()}, "
                f"got {x.shape[-1]}"
            )

        if not torch.is_floating_point(x):
            raise TypeError("Input must have a floating-point dtype")

        input_dtype = x.dtype

        # Use float32 for lower-precision inputs.
        # Keep float64 inputs in float64 for numerical accuracy.
        if x.dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        else:
            compute_dtype = x.dtype

        x_compute = x.to(dtype=compute_dtype)

        # Step 1: Square each feature and calculate mean square.
        mean_square = x_compute.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        # Step 2: Calculate RMS, including epsilon.
        rms = torch.sqrt(mean_square + self.eps)

        # Step 3: Normalize the input.
        normalized = x_compute / rms

        # Step 4: Apply the learnable feature weights.
        output = normalized * self.weight.to(
            dtype=compute_dtype
        )

        # Return the output in the input dtype.
        return output.to(dtype=input_dtype)