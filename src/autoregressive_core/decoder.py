
import torch
import torch.nn as nn

from autoregressive_core.normalization import RMSNorm


class PreNormResidual(nn.Module):
    """
    Apply RMSNorm before a sublayer, then add the original input.

    Computes:
        output = x + sublayer(RMSNorm(x))
    """

    def __init__(
        self,
        dim: int,
        sublayer: nn.Module,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm = RMSNorm(dim=dim, eps=eps)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, *args, **kwargs):
        residual = x

        # Normalize before passing the input to the sublayer.
        normalized = self.norm(x)

        # Apply the supplied transformation.
        transformed = self.sublayer(
            normalized, *args, **kwargs
        )

        if not isinstance(transformed, torch.Tensor):
            raise TypeError("Sublayer must return a Tensor")

        # Residual addition requires identical shapes.
        if transformed.shape != residual.shape:
            raise ValueError(
                "Sublayer output shape must match input shape: "
                f"input={tuple(residual.shape)}, "
                f"output={tuple(transformed.shape)}"
            )

        return residual + transformed