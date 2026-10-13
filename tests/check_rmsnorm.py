import torch
from autoregressive_core.normalization import RMSNorm
x = torch.randn(2, 5, 64,requires_grad=True)
norm = RMSNorm(dim=64)
y = norm(x)
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Weight shape:", norm.weight.shape)
print("Output is finite:", torch.isfinite(y).all().item())
loss = y.square().mean()
loss.backward()
print("Input gradients exist:", x.grad is not None)
print("Weight gradients exist:", norm.weight.grad is not None)