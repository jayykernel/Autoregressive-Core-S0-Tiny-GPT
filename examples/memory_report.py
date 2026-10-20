
import torch

from autoregressive_core.model import TinyGPT


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def format_bytes(nbytes: int) -> str:
    return f"{nbytes:,} bytes ({nbytes / 1024:.2f} KiB)"


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(vocab_size=256).to(device)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    parameter_bytes = sum(
        tensor_bytes(parameter) for parameter in model.parameters()
    )

    print("Device:", device)
    print("Model dtype:", next(model.parameters()).dtype)
    print("Parameter count:", f"{parameter_count:,}")
    print("Parameter storage:", format_bytes(parameter_bytes))

    # Two sequences of nine tokens give eight input/target positions.
    tokens = torch.randint(0, 256, (2, 9), device=device)
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(input_ids, targets)
    loss.backward()

    gradient_bytes = sum(
        tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    print("Input IDs:", tuple(input_ids.shape),
          format_bytes(tensor_bytes(input_ids)))
    print("Logits:", tuple(logits.shape),
          format_bytes(tensor_bytes(logits)))
    print("Loss:", float(loss.detach().cpu()))
    print("Gradient storage:", format_bytes(gradient_bytes))

    optimizer.step()

    optimizer_state_bytes = sum(
        tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    print("AdamW state storage:", format_bytes(optimizer_state_bytes))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        print("CUDA peak allocated:",
              format_bytes(peak_bytes))
    else:
        print(
            "CUDA peak memory unavailable on this run. "
            "CPU process memory is not measured by this script."
        )


if __name__ == "__main__":
    main()
