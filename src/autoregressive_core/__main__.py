import argparse
from .training import TrainConfig, train

parser = argparse.ArgumentParser(description="Train the TinyGPT character model")
parser.add_argument("--data", default="data/input.txt")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--sequence-length", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", default="checkpoints/tinygpt.pt")
args = parser.parse_args()
train(TrainConfig(steps=args.steps, batch_size=args.batch_size, sequence_length=args.sequence_length,
                  learning_rate=args.learning_rate, seed=args.seed), args.data, args.checkpoint)
