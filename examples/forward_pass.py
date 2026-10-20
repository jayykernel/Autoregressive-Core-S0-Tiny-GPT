
import torch

from autoregressive_core.model import TinyGPT


def main():
    torch.manual_seed(42)

    model = TinyGPT(vocab_size=256)

    # Simulated tokenized text.
    tokens = torch.randint(0, 256, (2, 9))

    # Predict the next token at every input position.
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(input_ids, targets)

    print("Input shape:", input_ids.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    loss.backward()

    print("Backward pass: successful")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    optimizer.step()

    print("Optimizer step: successful")


if __name__ == "__main__":
    main()
