import torch
from autoregressive_core.normalization import RMSNorm
x = torch.randn(2, 5, 64)
norm = RMSNorm(dim=64)
y = norm(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Weight shape:", norm.weight.shape)
print("Output is finite:", torch.isfinite(y).all().item())